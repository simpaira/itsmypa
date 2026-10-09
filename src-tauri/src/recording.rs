//! The meeting being recorded right now (at most one).
//!
//! Microphone and computer audio are captured together by itsmypa-capture on
//! one shared clock: pausing pauses both, and the two WAVs line up exactly
//! for the dual-channel transcription pipeline.

use anyhow::{bail, Result};
use itsmypa_capture::{Kind, Recording, Sources, TrackState};
use serde::Serialize;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

/// Opening a macOS audio tap can take a few seconds (longer the first time,
/// while the permission prompt is up).
const START_TIMEOUT: Duration = Duration::from_secs(8);

struct Active {
    id: String,
    rec: Recording,
}

/// What a finished recording left on disk, for the transcription job.
#[derive(Clone, Debug, Serialize)]
pub struct Finished {
    pub id: String,
    pub mic: Option<PathBuf>,
    pub system: Option<PathBuf>,
    pub duration_secs: f32,
}

#[derive(Default)]
pub struct Recorder {
    active: Mutex<Option<Active>>,
}

impl Recorder {
    pub fn start(&self, mic: bool, system: bool) -> Result<Value> {
        let mut active = self.active.lock().unwrap();
        if active.is_some() {
            bail!("A recording is already running");
        }
        let id = uuid::Uuid::new_v4().to_string();
        let rec = Recording::start(&itsmypa_engine::paths::recordings_dir(), &id, Sources { mic, system })?;
        let live = rec.wait_live(START_TIMEOUT);
        let tracks: Vec<Value> = rec
            .status()
            .iter()
            .map(|t| {
                json!({
                    "kind": if t.kind == Kind::Mic { "mic" } else { "system" },
                    "live": t.state == TrackState::Live,
                    "device": t.device,
                    "message": t.message,
                })
            })
            .collect();
        *active = Some(Active { id: id.clone(), rec });
        Ok(json!({"id": id, "live": live, "tracks": tracks}))
    }

    pub fn set_paused(&self, paused: bool) -> Result<()> {
        let active = self.active.lock().unwrap();
        let Some(a) = active.as_ref() else { bail!("Nothing is recording") };
        if paused {
            a.rec.pause();
        } else {
            a.rec.resume();
        }
        Ok(())
    }

    /// Live levels for the waveform, plus each track's state (so the UI can
    /// say "Reconnecting microphone…" when a headset is switched).
    pub fn level(&self) -> Value {
        let active = self.active.lock().unwrap();
        let Some(a) = active.as_ref() else { return json!({"recording": false, "mic": 0.0, "system": 0.0}) };
        let lv = a.rec.levels();
        let reconnecting: Vec<&str> = a
            .rec
            .status()
            .iter()
            .filter(|t| t.state == TrackState::Reconnecting)
            .map(|t| if t.kind == Kind::Mic { "mic" } else { "system" })
            .collect();
        json!({
            "recording": true,
            "paused": a.rec.is_paused(),
            "elapsed": a.rec.elapsed().as_secs_f32(),
            "mic": lv.mic,
            "system": lv.system,
            "reconnecting": reconnecting,
        })
    }

    pub fn stop(&self) -> Result<Finished> {
        let Some(a) = self.active.lock().unwrap().take() else { bail!("Nothing is recording") };
        let done = a.rec.stop()?;
        Ok(Finished { id: a.id, mic: done.mic, system: done.system, duration_secs: done.duration_secs })
    }

    /// Finalize any capture still running when the app quits, so its WAVs
    /// are complete and the audio devices are released cleanly.
    pub fn stop_on_exit(&self) {
        if let Some(a) = self.active.lock().unwrap().take() {
            let _ = a.rec.stop();
        }
    }
}
