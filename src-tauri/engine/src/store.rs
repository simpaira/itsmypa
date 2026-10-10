//! Saved meetings, in one SQLite file (`meetings.db` in the data dir).
//!
//! Replaces the Python engine's notes.json, which was rewritten in full on
//! every keystroke-save. On first open, meetings from notes.json are copied
//! in once; notes.json itself is never modified, so it stays as a backup.
//! A full-text index (FTS5) over titles, summaries, notes and transcripts
//! backs search and "ask across all meetings".
//!
//! Notes keep the exact JSON shape the UI and backup files already use.

use anyhow::{bail, Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Note {
    pub id: String,
    pub title: String,
    /// Local time, ISO 8601 (as the Python engine wrote it).
    pub date: String,
    pub transcript: String,
    pub summary: String,
    pub duration: f64,
    /// Speaker segments from the pipeline, or null.
    pub segments: Value,
    /// Action-item checkbox state, keyed by item index.
    pub action_status: Value,
    pub scratchpad: String,
    pub enhanced: String,
    /// "recording" | "processing" | "ready"
    pub status: String,
}

/// Fields a PATCH may change (same allow-list as the Python engine).
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct NotePatch {
    pub title: Option<String>,
    pub action_status: Option<Value>,
    pub summary: Option<String>,
    pub transcript: Option<String>,
    pub segments: Option<Value>,
    pub scratchpad: Option<String>,
    pub enhanced: Option<String>,
    pub status: Option<String>,
}

pub struct ImportResult {
    pub imported: usize,
    pub skipped: usize,
}

pub struct Store {
    db: Mutex<Connection>,
    /// Bumped on every change, so caches (the "ask" index) know to rebuild.
    version: AtomicU64,
}

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS notes (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL DEFAULT '',
    date          TEXT NOT NULL DEFAULT '',
    transcript    TEXT NOT NULL DEFAULT '',
    summary       TEXT NOT NULL DEFAULT '',
    duration      REAL NOT NULL DEFAULT 0,
    segments      TEXT NOT NULL DEFAULT 'null',
    action_status TEXT NOT NULL DEFAULT '{}',
    scratchpad    TEXT NOT NULL DEFAULT '',
    enhanced      TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'ready'
);
CREATE INDEX IF NOT EXISTS notes_by_date ON notes(date DESC);

-- Full-text search, kept in sync with `notes` by the triggers below.
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    title, summary, notes, transcript,
    content='', contentless_delete=1, tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
    INSERT INTO notes_fts(rowid, title, summary, notes, transcript)
    VALUES (new.rowid, new.title, new.summary, new.scratchpad || ' ' || new.enhanced, new.transcript);
END;
CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
    DELETE FROM notes_fts WHERE rowid = old.rowid;
END;
CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
    DELETE FROM notes_fts WHERE rowid = old.rowid;
    INSERT INTO notes_fts(rowid, title, summary, notes, transcript)
    VALUES (new.rowid, new.title, new.summary, new.scratchpad || ' ' || new.enhanced, new.transcript);
END;

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"#;

const COLUMNS: &str = "id, title, date, transcript, summary, duration, segments, action_status, scratchpad, enhanced, status";

fn row_to_note(r: &rusqlite::Row) -> rusqlite::Result<Note> {
    let json = |i: usize| -> rusqlite::Result<Value> {
        let s: String = r.get(i)?;
        Ok(serde_json::from_str(&s).unwrap_or(Value::Null))
    };
    Ok(Note {
        id: r.get(0)?,
        title: r.get(1)?,
        date: r.get(2)?,
        transcript: r.get(3)?,
        summary: r.get(4)?,
        duration: r.get(5)?,
        segments: json(6)?,
        action_status: json(7)?,
        scratchpad: r.get(8)?,
        enhanced: r.get(9)?,
        status: r.get(10)?,
    })
}

pub fn now_iso() -> String {
    chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.6f").to_string()
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let conn = Connection::open(path).with_context(|| format!("could not open {}", path.display()))?;
        // WAL: a crash mid-write can't corrupt the database, and reads never
        // wait for a save.
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self { db: Mutex::new(conn), version: AtomicU64::new(0) })
    }

    /// Copy meetings from the Python engine's notes.json, once. The file is
    /// left untouched. Returns how many meetings were copied (0 if done before
    /// or there's no file).
    pub fn import_legacy_json(&self, json_path: &Path) -> Result<usize> {
        if self.meta("legacy_json_imported")?.is_some() || !json_path.exists() {
            return Ok(0);
        }
        let text = std::fs::read_to_string(json_path)?;
        let result = self.import_json(&text).context("notes.json could not be read")?;
        self.set_meta("legacy_json_imported", &now_iso())?;
        Ok(result.imported)
    }

    pub fn meta(&self, key: &str) -> Result<Option<String>> {
        let db = self.db.lock().unwrap();
        Ok(db.query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| r.get(0)).optional()?)
    }

    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        let db = self.db.lock().unwrap();
        db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?1, ?2)", [key, value])?;
        Ok(())
    }

    /// Changes whenever any meeting is added, edited or deleted.
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::SeqCst)
    }

    fn changed(&self) {
        self.version.fetch_add(1, Ordering::SeqCst);
    }

    /// All meetings, newest first.
    pub fn list(&self) -> Result<Vec<Note>> {
        let db = self.db.lock().unwrap();
        let mut stmt = db.prepare(&format!("SELECT {COLUMNS} FROM notes ORDER BY date DESC"))?;
        let notes = stmt.query_map([], row_to_note)?.collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(notes)
    }

    pub fn get(&self, id: &str) -> Result<Option<Note>> {
        let db = self.db.lock().unwrap();
        Ok(db
            .query_row(&format!("SELECT {COLUMNS} FROM notes WHERE id = ?1"), [id], row_to_note)
            .optional()?)
    }

    /// Save a new meeting. A fresh id and the current time are assigned.
    pub fn create(&self, mut note: Note) -> Result<Note> {
        note.id = uuid::Uuid::new_v4().to_string();
        note.date = now_iso();
        // An empty title is kept empty: the screen fills it in with an AI
        // title once the transcript is ready (and shows "Untitled" until then).
        if note.status.is_empty() {
            note.status = "ready".into();
        }
        if note.action_status.is_null() {
            note.action_status = Value::Object(Default::default());
        }
        self.insert(&self.db.lock().unwrap(), &note)?;
        self.changed();
        Ok(note)
    }

    fn insert(&self, db: &Connection, n: &Note) -> Result<()> {
        db.execute(
            &format!("INSERT INTO notes ({COLUMNS}) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)"),
            params![
                n.id,
                n.title,
                n.date,
                n.transcript,
                n.summary,
                n.duration,
                n.segments.to_string(),
                n.action_status.to_string(),
                n.scratchpad,
                n.enhanced,
                n.status
            ],
        )?;
        Ok(())
    }

    /// Change some fields of a meeting. Returns the updated meeting, or None
    /// if there's no meeting with that id.
    pub fn update(&self, id: &str, patch: NotePatch) -> Result<Option<Note>> {
        let Some(mut n) = self.get(id)? else { return Ok(None) };
        if let Some(v) = patch.title { n.title = v; }
        if let Some(v) = patch.action_status { n.action_status = v; }
        if let Some(v) = patch.summary { n.summary = v; }
        if let Some(v) = patch.transcript { n.transcript = v; }
        if let Some(v) = patch.segments { n.segments = v; }
        if let Some(v) = patch.scratchpad { n.scratchpad = v; }
        if let Some(v) = patch.enhanced { n.enhanced = v; }
        if let Some(v) = patch.status { n.status = v; }
        let db = self.db.lock().unwrap();
        db.execute(
            "UPDATE notes SET title=?2, transcript=?3, summary=?4, segments=?5, action_status=?6,
             scratchpad=?7, enhanced=?8, status=?9 WHERE id=?1",
            params![
                id,
                n.title,
                n.transcript,
                n.summary,
                n.segments.to_string(),
                n.action_status.to_string(),
                n.scratchpad,
                n.enhanced,
                n.status
            ],
        )?;
        drop(db);
        self.changed();
        Ok(Some(n))
    }

    /// Set a meeting's date (ISO 8601). For imports and tests; meetings
    /// created normally are stamped with the current time.
    pub fn set_date(&self, id: &str, date: &str) -> Result<()> {
        let db = self.db.lock().unwrap();
        db.execute("UPDATE notes SET date = ?2 WHERE id = ?1", [id, date])?;
        drop(db);
        self.changed();
        Ok(())
    }

    pub fn delete(&self, id: &str) -> Result<bool> {
        let db = self.db.lock().unwrap();
        let deleted = db.execute("DELETE FROM notes WHERE id = ?1", [id])? > 0;
        drop(db);
        self.changed();
        Ok(deleted)
    }

    /// Ids of meetings matching a full-text query, best match first.
    /// Words are matched with stemming ("decided" finds "decide").
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<String>> {
        let terms: Vec<String> = query
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(|w| format!("\"{w}\"")) // quoted: user text is never FTS syntax
            .collect();
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        let db = self.db.lock().unwrap();
        let mut stmt = db.prepare(
            "SELECT notes.id FROM notes_fts JOIN notes ON notes.rowid = notes_fts.rowid
             WHERE notes_fts MATCH ?1 ORDER BY bm25(notes_fts) LIMIT ?2",
        )?;
        let ids = stmt
            .query_map(params![terms.join(" OR "), limit as i64], |r| r.get(0))?
            .collect::<rusqlite::Result<Vec<String>>>()?;
        Ok(ids)
    }

    /// Backup file in the Python engine's format, so old and new backups
    /// can be imported by either.
    pub fn export_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(&serde_json::json!({
            "app": "itsmypa",
            "version": 1,
            "exported": now_iso(),
            "notes": self.list()?,
        }))?)
    }

    /// Restore a backup (or a raw list of notes). Meetings whose id already
    /// exists are skipped, as are entries that aren't valid meetings.
    pub fn import_json(&self, text: &str) -> Result<ImportResult> {
        let data: Value = serde_json::from_str(text).context("not a valid JSON backup file")?;
        let items = match &data {
            Value::Array(a) => a,
            Value::Object(o) => match o.get("notes") {
                Some(Value::Array(a)) => a,
                _ => bail!("no notes found in the file"),
            },
            _ => bail!("no notes found in the file"),
        };
        let mut db = self.db.lock().unwrap();
        let tx = db.transaction()?;
        let (mut imported, mut skipped) = (0, 0);
        for item in items {
            let valid = item.get("id").and_then(Value::as_str).is_some_and(valid_id)
                && item.get("title").is_some_and(Value::is_string);
            let note: Option<Note> = if valid { serde_json::from_value(item.clone()).ok() } else { None };
            let Some(mut note) = note else {
                skipped += 1;
                continue;
            };
            let exists: bool = tx.query_row("SELECT 1 FROM notes WHERE id = ?1", [&note.id], |_| Ok(true)).optional()?.is_some();
            if exists {
                skipped += 1;
                continue;
            }
            if note.status.is_empty() {
                note.status = "ready".into();
            }
            self.insert(&tx, &note)?;
            imported += 1;
        }
        tx.commit()?;
        drop(db);
        self.changed();
        Ok(ImportResult { imported, skipped })
    }
}

/// Ids end up in the UI's HTML and handlers: allow only safe characters.
fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 64 && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store(name: &str) -> (Store, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("itsmypa-store-{name}-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        (Store::open(&dir.join("meetings.db")).unwrap(), dir)
    }

    fn note(title: &str, transcript: &str) -> Note {
        Note { title: title.into(), transcript: transcript.into(), ..Default::default() }
    }

    // Port of test_notes_crud_roundtrip
    #[test]
    fn crud_roundtrip() {
        let (s, dir) = temp_store("crud");
        let n = s.create(note("Standup", "we ship friday")).unwrap();
        assert_eq!(n.status, "ready");
        assert_eq!(s.list().unwrap().len(), 1);

        let patch = NotePatch { title: Some("Renamed".into()), scratchpad: Some("- ship".into()), ..Default::default() };
        let u = s.update(&n.id, patch).unwrap().unwrap();
        assert_eq!((u.title.as_str(), u.scratchpad.as_str(), u.transcript.as_str()), ("Renamed", "- ship", "we ship friday"));
        assert_eq!(s.get(&n.id).unwrap().unwrap().title, "Renamed");

        assert!(s.delete(&n.id).unwrap());
        assert!(s.list().unwrap().is_empty());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn empty_title_stays_empty_for_the_ai_title() {
        let (s, dir) = temp_store("title");
        assert_eq!(s.create(note("", "")).unwrap().title, "");
        std::fs::remove_dir_all(dir).ok();
    }

    // Port of test_patch_unknown_note_404
    #[test]
    fn update_unknown_is_none() {
        let (s, dir) = temp_store("unknown");
        assert!(s.update("nope", NotePatch::default()).unwrap().is_none());
        std::fs::remove_dir_all(dir).ok();
    }

    // Port of test_export_import_roundtrip
    #[test]
    fn export_import_roundtrip() {
        let (a, dir_a) = temp_store("export");
        let n = a.create(note("Q3 review", "revenue up")).unwrap();
        let backup = a.export_json().unwrap();

        let (b, dir_b) = temp_store("import");
        let r = b.import_json(&backup).unwrap();
        assert_eq!((r.imported, r.skipped), (1, 0));
        assert_eq!(b.get(&n.id).unwrap().unwrap(), n);
        // Importing again skips what's already there.
        assert_eq!(b.import_json(&backup).unwrap().skipped, 1);
        std::fs::remove_dir_all(dir_a).ok();
        std::fs::remove_dir_all(dir_b).ok();
    }

    // Port of test_import_rejects_garbage
    #[test]
    fn import_rejects_garbage_and_unsafe_ids() {
        let (s, dir) = temp_store("garbage");
        assert!(s.import_json("not json").is_err());
        assert!(s.import_json("{\"hello\": 1}").is_err());
        let r = s
            .import_json(r#"[{"id":"x');alert(1);//","title":"evil"},{"title":"no id"},{"id":"ok-1","title":"fine"}]"#)
            .unwrap();
        assert_eq!((r.imported, r.skipped), (1, 2));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn legacy_json_is_copied_once_and_left_untouched() {
        let (s, dir) = temp_store("legacy");
        let json = dir.join("notes.json");
        let original = r#"[{"id":"a1","title":"Old meeting","date":"2026-07-15T10:00:00","transcript":"hello","summary":"",
            "duration":42,"segments":[{"speaker":"You","text":"hello","start":0,"end":1}],"action_status":{"0":true},
            "scratchpad":"","enhanced":"","status":"ready"}]"#;
        std::fs::write(&json, original).unwrap();

        assert_eq!(s.import_legacy_json(&json).unwrap(), 1);
        assert_eq!(s.import_legacy_json(&json).unwrap(), 0, "only once");
        assert_eq!(std::fs::read_to_string(&json).unwrap(), original, "file unchanged");
        let n = s.get("a1").unwrap().unwrap();
        assert_eq!(n.duration, 42.0);
        assert_eq!(n.segments[0]["speaker"], "You");
        assert_eq!(n.action_status["0"], true);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn search_finds_by_stem_and_follows_edits() {
        let (s, dir) = temp_store("search");
        let a = s.create(note("Launch sync", "We decided the launch is March 14")).unwrap();
        s.create(note("Hiring", "two backend engineers")).unwrap();
        assert_eq!(s.search("decide launch", 10).unwrap(), vec![a.id.clone()]);
        assert!(s.search("\"); DROP TABLE notes; --", 10).unwrap().is_empty());

        s.update(&a.id, NotePatch { transcript: Some("nothing here".into()), ..Default::default() }).unwrap();
        // Title still matches "launch", but "decide" is gone from the index.
        assert_eq!(s.search("decided", 10).unwrap(), Vec::<String>::new());
        s.delete(&a.id).unwrap();
        assert!(s.search("launch", 10).unwrap().is_empty());
        std::fs::remove_dir_all(dir).ok();
    }
}
