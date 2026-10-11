//! User settings, stored in the meetings database. Replaces the Python
//! engine's .env file, which a crafted calendar URL containing a newline
//! could inject extra settings into (e.g. HOST=0.0.0.0). Settings from that
//! .env are copied in once; the file itself is left untouched.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::store::Store;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// "" = auto-detect, else a 2-3 letter code ("en", "hi", …).
    pub whisper_language: String,
    pub diarize: bool,
    /// -1 = detect automatically, else 1-16.
    pub num_speakers: i32,
    /// Clustering distance; lower = more speakers.
    pub diarize_threshold: f32,
    /// Secret iCal address for "record this meeting?" prompts ("" = off).
    pub calendar_ics_url: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            whisper_language: String::new(),
            diarize: true,
            num_speakers: -1,
            diarize_threshold: 0.8,
            calendar_ics_url: String::new(),
        }
    }
}

/// A change from the Settings screen: only the fields that are present.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct SettingsUpdate {
    pub whisper_language: Option<String>,
    pub diarize: Option<bool>,
    pub num_speakers: Option<i32>,
    pub diarize_threshold: Option<f32>,
    pub calendar_ics_url: Option<String>,
}

pub struct Saved {
    pub settings: Settings,
    /// Speech or notes models must be reloaded for the change to apply.
    pub reload_models: bool,
}

const KEY: &str = "settings";

pub fn load(store: &Store) -> Result<Settings> {
    Ok(match store.meta(KEY)? {
        Some(json) => serde_json::from_str(&json).unwrap_or_default(),
        None => Settings::default(),
    })
}

/// Validate and save a change. Nothing is saved if any field is invalid.
pub fn update(store: &Store, change: SettingsUpdate) -> Result<Saved> {
    let old = load(store)?;
    let mut s = old.clone();
    if let Some(lang) = change.whisper_language {
        let lang = lang.trim().to_lowercase();
        if !lang.is_empty() && (!(2..=3).contains(&lang.len()) || !lang.chars().all(|c| c.is_ascii_lowercase())) {
            bail!("Language must be a 2-3 letter code (or empty for auto)");
        }
        s.whisper_language = lang;
    }
    if let Some(d) = change.diarize {
        s.diarize = d;
    }
    if let Some(n) = change.num_speakers {
        if n != -1 && !(1..=16).contains(&n) {
            bail!("num_speakers must be -1 (auto) or 1-16");
        }
        s.num_speakers = n;
    }
    if let Some(t) = change.diarize_threshold {
        if !(0.1..=1.5).contains(&t) {
            bail!("diarize_threshold must be between 0.1 and 1.5");
        }
        s.diarize_threshold = t;
    }
    if let Some(url) = change.calendar_ics_url {
        s.calendar_ics_url = clean_calendar_url(&url)?;
    }

    store.set_meta(KEY, &serde_json::to_string(&s)?)?;
    let reload_models = s.whisper_language != old.whisper_language
        || s.diarize != old.diarize
        || s.num_speakers != old.num_speakers
        || s.diarize_threshold != old.diarize_threshold;
    Ok(Saved { settings: s, reload_models })
}

fn clean_calendar_url(url: &str) -> Result<String> {
    let url = url.trim();
    if url.chars().any(|c| c.is_control() || c.is_whitespace()) {
        bail!("Calendar URL can't contain spaces or line breaks");
    }
    let url = match url.strip_prefix("webcal://") {
        Some(rest) => format!("https://{rest}"),
        None => url.to_string(),
    };
    if !url.is_empty() && !(url.starts_with("https://") || url.starts_with("http://")) {
        bail!("Calendar URL must start with https:// (or webcal://)");
    }
    Ok(url)
}

/// Copy settings from the Python engine's .env, once. Values that wouldn't
/// pass today's validation are skipped. The file is left untouched.
pub fn import_legacy_env(store: &Store, env_path: &Path) -> Result<bool> {
    if store.meta("legacy_env_imported")?.is_some() || !env_path.exists() {
        return Ok(false);
    }
    let text = std::fs::read_to_string(env_path)?;
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else { continue };
        let value = value.trim().trim_matches('"').trim_matches('\'').to_string();
        let mut change = SettingsUpdate::default();
        match key.trim() {
            "WHISPER_LANGUAGE" => change.whisper_language = Some(value),
            "DIARIZE" => change.diarize = Some(!matches!(value.to_lowercase().as_str(), "0" | "false" | "no")),
            "NUM_SPEAKERS" => change.num_speakers = value.parse().ok(),
            "DIARIZE_THRESHOLD" => change.diarize_threshold = value.parse().ok(),
            "CALENDAR_ICS_URL" => change.calendar_ics_url = Some(value),
            _ => continue,
        }
        // One setting at a time, so a single bad value can't block the rest.
        update(store, change).ok();
    }
    store.set_meta("legacy_env_imported", &crate::store::now_iso())?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store(name: &str) -> (Store, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("itsmypa-settings-{name}-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        (Store::open(&dir.join("meetings.db")).unwrap(), dir)
    }

    // Port of test_settings_roundtrip_writes_env
    #[test]
    fn defaults_then_roundtrip() {
        let (s, dir) = temp_store("roundtrip");
        assert_eq!(load(&s).unwrap(), Settings::default());
        let saved = update(&s, SettingsUpdate { whisper_language: Some("de".into()), num_speakers: Some(3), ..Default::default() }).unwrap();
        assert!(saved.reload_models);
        let back = load(&s).unwrap();
        assert_eq!((back.whisper_language.as_str(), back.num_speakers), ("de", 3));
        // Calendar changes apply live: no model reload.
        let saved = update(&s, SettingsUpdate { calendar_ics_url: Some("webcal://cal.example/x.ics".into()), ..Default::default() }).unwrap();
        assert!(!saved.reload_models);
        assert_eq!(saved.settings.calendar_ics_url, "https://cal.example/x.ics");
        std::fs::remove_dir_all(dir).ok();
    }

    // Port of test_settings_rejects_bad_values, plus the .env injection fix.
    #[test]
    fn bad_values_are_rejected_and_nothing_is_saved() {
        let (s, dir) = temp_store("bad");
        let bad = [
            SettingsUpdate { whisper_language: Some("english".into()), ..Default::default() },
            SettingsUpdate { num_speakers: Some(0), ..Default::default() },
            SettingsUpdate { diarize_threshold: Some(3.0), ..Default::default() },
            SettingsUpdate { calendar_ics_url: Some("ftp://x".into()), ..Default::default() },
            SettingsUpdate { calendar_ics_url: Some("https://x.example/a\nHOST=0.0.0.0".into()), ..Default::default() },
        ];
        for change in bad {
            assert!(update(&s, change).is_err());
        }
        assert_eq!(load(&s).unwrap(), Settings::default());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn legacy_env_is_copied_once_skipping_bad_values() {
        let (s, dir) = temp_store("env");
        let env = dir.join(".env");
        let original = "WHISPER_MODEL=medium\nWHISPER_LANGUAGE=hi\nDIARIZE=0\nNUM_SPEAKERS=99\nCALENDAR_ICS_URL=webcal://cal.example/a.ics\nHOST=0.0.0.0\n";
        std::fs::write(&env, original).unwrap();
        assert!(import_legacy_env(&s, &env).unwrap());
        assert!(!import_legacy_env(&s, &env).unwrap(), "only once");
        assert_eq!(std::fs::read_to_string(&env).unwrap(), original, "file unchanged");
        let got = load(&s).unwrap();
        assert_eq!(got.whisper_language, "hi");
        assert!(!got.diarize);
        assert_eq!(got.num_speakers, -1, "99 is invalid, default kept");
        assert_eq!(got.calendar_ics_url, "https://cal.example/a.ics");
        std::fs::remove_dir_all(dir).ok();
    }
}
