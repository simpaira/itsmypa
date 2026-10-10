//! Recordings the app never finished processing — it quit or crashed while
//! recording or transcribing. Their WAVs are still in the recordings folder
//! (capture keeps them readable on disk); the home screen offers to recover
//! or discard them.

use anyhow::{bail, Result};
use itsmypa_capture::{repair_wav, track_path, Kind};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::jobs::Input;
use crate::recording::Finished;

/// Under a second of audio isn't worth offering.
const MIN_BYTES: u64 = 32_000;

#[derive(Debug, Serialize)]
pub struct Orphan {
    pub id: String,
    pub bytes: u64,
    pub seconds: f32,
    /// Local time the audio was last written.
    pub modified: String,
}

/// Recording ids allowed as file names (uuids, or "upload-<uuid>.<ext>").
fn valid_id(id: &str) -> bool {
    !id.is_empty() && !id.contains("..") && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.')
}

/// Files making up one recording id.
fn files_of(id: &str) -> Vec<PathBuf> {
    let dir = itsmypa_engine::paths::recordings_dir();
    if id.starts_with("upload-") {
        vec![dir.join(id)]
    } else {
        [Kind::Mic, Kind::System].into_iter().map(|k| track_path(&dir, id, k)).filter(|p| p.exists()).collect()
    }
}

/// Unfinished recordings, newest first. `busy` says whether a file belongs
/// to the live recording or a running job (those aren't orphans).
pub fn list(busy: &dyn Fn(&std::path::Path) -> bool) -> Vec<Orphan> {
    let dir = itsmypa_engine::paths::recordings_dir();
    let Ok(entries) = std::fs::read_dir(&dir) else { return Vec::new() };
    let mut by_id: BTreeMap<String, (u64, std::time::SystemTime)> = BTreeMap::new();
    for e in entries.flatten() {
        let path = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        let id = if name.starts_with("upload-") {
            name.clone()
        } else if let Some(id) = name.strip_suffix(".mic.wav").or_else(|| name.strip_suffix(".system.wav")) {
            id.to_string()
        } else {
            continue;
        };
        if busy(&path) || !valid_id(&id) {
            continue;
        }
        let Ok(meta) = e.metadata() else { continue };
        let modified = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
        let entry = by_id.entry(id).or_insert((0, modified));
        entry.0 += meta.len();
        entry.1 = entry.1.max(modified);
    }
    let mut out: Vec<Orphan> = by_id
        .into_iter()
        .filter(|(_, (bytes, _))| *bytes >= MIN_BYTES)
        .map(|(id, (bytes, modified))| Orphan {
            // 16 kHz 16-bit mono; with two tracks, roughly half is each.
            seconds: if id.starts_with("upload-") { 0.0 } else { bytes as f32 / 32_000.0 / files_of(&id).len().max(1) as f32 },
            modified: chrono::DateTime::<chrono::Local>::from(modified).format("%Y-%m-%dT%H:%M:%S").to_string(),
            bytes,
            id,
        })
        .collect();
    out.sort_by(|a, b| b.modified.cmp(&a.modified));
    out
}

/// Fix the WAV headers a crash left stale and turn the recording into a job input.
pub fn recover(id: &str) -> Result<Input> {
    if !valid_id(id) {
        bail!("Bad recording id");
    }
    let files = files_of(id);
    if files.is_empty() || !files.iter().all(|p| p.exists()) {
        bail!("Recording not found");
    }
    if id.starts_with("upload-") {
        return Ok(Input::Upload(files[0].clone()));
    }
    let dir = itsmypa_engine::paths::recordings_dir();
    let (mic, system) = (track_path(&dir, id, Kind::Mic), track_path(&dir, id, Kind::System));
    let mut duration_secs: f32 = 0.0;
    for p in [&mic, &system] {
        if p.exists() {
            duration_secs = duration_secs.max(repair_wav(p)? as f32 / 32_000.0);
        }
    }
    Ok(Input::Recording(Finished {
        id: id.to_string(),
        mic: mic.exists().then_some(mic),
        system: system.exists().then_some(system),
        duration_secs,
    }))
}

pub fn discard(id: &str) -> Result<()> {
    if !valid_id(id) {
        bail!("Bad recording id");
    }
    for p in files_of(id) {
        std::fs::remove_file(p)?;
    }
    Ok(())
}
