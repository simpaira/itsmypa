//! Recordings the app never finished processing — it quit or crashed while
//! recording or transcribing. Their WAVs are still in the recordings folder
//! (capture keeps them readable on disk); the home screen offers to recover
//! or discard them.
//!
//! Also handles what the Python-era app (0.1) left there: single-track
//! `mic_<uuid>.wav` / `sys_<uuid>.wav` files from its capture helper. Those
//! with real audio are offered like any other; empty ones (failed mic starts)
//! are removed at startup.

use anyhow::{bail, Result};
use itsmypa_capture::{repair_wav, track_path, Kind};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::jobs::Input;
use crate::recording::Finished;

/// Under a second of audio isn't worth offering.
const MIN_SECONDS: f32 = 1.0;

#[derive(Debug, Serialize)]
pub struct Orphan {
    pub id: String,
    pub bytes: u64,
    pub seconds: f32,
    /// Local time the audio was last written.
    pub modified: String,
}

/// Recording ids allowed as file names: uuids, "upload-<uuid>.<ext>", or the
/// old app's "mic_<uuid>" / "sys_<uuid>".
fn valid_id(id: &str) -> bool {
    !id.is_empty() && !id.contains("..") && id.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_'))
}

fn is_legacy(id: &str) -> bool {
    id.starts_with("mic_") || id.starts_with("sys_")
}

/// Files making up one recording id.
fn files_of(dir: &Path, id: &str) -> Vec<PathBuf> {
    if id.starts_with("upload-") {
        vec![dir.join(id)]
    } else if is_legacy(id) {
        vec![dir.join(format!("{id}.wav"))]
    } else {
        [Kind::Mic, Kind::System].into_iter().map(|k| track_path(dir, id, k)).filter(|p| p.exists()).collect()
    }
}

/// Seconds of audio in a WAV, from its format and the file's real size (the
/// header can be stale after a crash).
fn wav_seconds(path: &Path) -> f32 {
    let Ok(reader) = hound::WavReader::open(path) else { return 0.0 };
    let spec = reader.spec();
    let bytes_per_sec = spec.sample_rate as f32 * spec.channels as f32 * spec.bits_per_sample as f32 / 8.0;
    let len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    // Data starts after a ~44-byte header; close enough for "how long".
    (len.saturating_sub(44)) as f32 / bytes_per_sec.max(1.0)
}

/// The recording id a file in the recordings folder belongs to, if any.
fn id_of(name: &str) -> Option<String> {
    if name.starts_with("upload-") {
        Some(name.to_string())
    } else if let Some(id) = name.strip_suffix(".mic.wav").or_else(|| name.strip_suffix(".system.wav")) {
        Some(id.to_string())
    } else if is_legacy(name) {
        name.strip_suffix(".wav").map(str::to_string)
    } else {
        None
    }
}

/// Unfinished recordings, newest first. `busy` says whether a file belongs
/// to the live recording or a running job (those aren't orphans).
pub fn list(busy: &dyn Fn(&Path) -> bool) -> Vec<Orphan> {
    list_in(&itsmypa_engine::paths::recordings_dir(), busy)
}

fn list_in(dir: &Path, busy: &dyn Fn(&Path) -> bool) -> Vec<Orphan> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    // id → (bytes, seconds of the longest track, last modified)
    let mut by_id: BTreeMap<String, (u64, f32, std::time::SystemTime)> = BTreeMap::new();
    for e in entries.flatten() {
        let path = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(id) = id_of(&name) else { continue };
        if busy(&path) || !valid_id(&id) {
            continue;
        }
        let Ok(meta) = e.metadata() else { continue };
        let modified = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
        // Uploads are compressed audio: any size counts; WAVs are measured.
        let secs = if name.starts_with("upload-") { MIN_SECONDS } else { wav_seconds(&path) };
        let entry = by_id.entry(id).or_insert((0, 0.0, modified));
        entry.0 += meta.len();
        entry.1 = entry.1.max(secs);
        entry.2 = entry.2.max(modified);
    }
    let mut out: Vec<Orphan> = by_id
        .into_iter()
        .filter(|(_, (_, secs, _))| *secs >= MIN_SECONDS)
        .map(|(id, (bytes, secs, modified))| Orphan {
            seconds: if id.starts_with("upload-") { 0.0 } else { secs },
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
    recover_in(&itsmypa_engine::paths::recordings_dir(), id)
}

fn recover_in(dir: &Path, id: &str) -> Result<Input> {
    if !valid_id(id) {
        bail!("Bad recording id");
    }
    let files = files_of(dir, id);
    if files.is_empty() || !files.iter().all(|p| p.exists()) {
        bail!("Recording not found");
    }
    if id.starts_with("upload-") {
        return Ok(Input::Upload(files[0].clone()));
    }
    for p in &files {
        repair_wav(p)?;
    }
    let duration_secs = files.iter().map(|p| wav_seconds(p)).fold(0.0, f32::max);
    if is_legacy(id) {
        // One track from the old app: the mic is "You", computer audio is
        // diarized; either way it's transcribed as a single channel.
        let (mic, system) = if id.starts_with("mic_") { (Some(files[0].clone()), None) } else { (None, Some(files[0].clone())) };
        return Ok(Input::Recording(Finished { id: id.to_string(), mic, system, duration_secs }));
    }
    let (mic, system) = (track_path(dir, id, Kind::Mic), track_path(dir, id, Kind::System));
    Ok(Input::Recording(Finished {
        id: id.to_string(),
        mic: mic.exists().then_some(mic),
        system: system.exists().then_some(system),
        duration_secs,
    }))
}

pub fn discard(id: &str) -> Result<()> {
    let dir = itsmypa_engine::paths::recordings_dir();
    if !valid_id(id) {
        bail!("Bad recording id");
    }
    for p in files_of(&dir, id) {
        std::fs::remove_file(p)?;
    }
    Ok(())
}

/// Delete the old app's leftover recordings that hold no real audio (failed
/// mic starts). Returns how many were removed.
pub fn remove_empty_legacy() -> usize {
    remove_empty_legacy_in(&itsmypa_engine::paths::recordings_dir())
}

fn remove_empty_legacy_in(dir: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else { return 0 };
    entries
        .flatten()
        .filter(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            is_legacy(&name) && name.ends_with(".wav") && wav_seconds(&e.path()) < MIN_SECONDS
        })
        .filter(|e| std::fs::remove_file(e.path()).is_ok())
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(path: &Path, rate: u32, seconds: f32) {
        let spec = hound::WavSpec { channels: 1, sample_rate: rate, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        for _ in 0..(rate as f32 * seconds) as usize {
            w.write_sample(0i16).unwrap();
        }
        w.finalize().unwrap();
    }

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("itsmypa-recovery-{name}-{}", std::process::id()));
        std::fs::remove_dir_all(&d).ok();
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn old_app_leftovers_with_audio_are_offered_and_empty_ones_removed() {
        let d = temp("legacy");
        wav(&d.join("mic_68957c06-dd63-4cb5-89e6-d8a3bf276868.wav"), 48_000, 0.0); // failed mic start
        wav(&d.join("mic_e797e756-8cb0-4d0d-884c-70170b901df0.wav"), 48_000, 0.1);
        wav(&d.join("sys_11111111-2222-3333-4444-555555555555.wav"), 48_000, 3.0); // real audio
        wav(&d.join("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.mic.wav"), 16_000, 2.0); // new-format crash

        let ids: Vec<String> = list_in(&d, &|_| false).into_iter().map(|o| o.id).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"sys_11111111-2222-3333-4444-555555555555".to_string()));
        assert!(ids.contains(&"aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee".to_string()));

        assert_eq!(remove_empty_legacy_in(&d), 2);
        assert!(d.join("sys_11111111-2222-3333-4444-555555555555.wav").exists(), "real audio is kept");

        let Input::Recording(f) = recover_in(&d, "sys_11111111-2222-3333-4444-555555555555").unwrap() else { panic!("not a recording") };
        assert!(f.mic.is_none() && f.system.is_some());
        assert!((f.duration_secs - 3.0).abs() < 0.1, "{}", f.duration_secs);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn ids_cannot_escape_the_recordings_folder() {
        assert!(!valid_id("../notes"));
        assert!(!valid_id("a/b"));
        assert!(valid_id("mic_68957c06-dd63-4cb5-89e6-d8a3bf276868"));
    }
}
