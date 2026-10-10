//! Native meeting capture for macOS and Windows, with one code path for both:
//!
//! - **Microphone** ("You"): the default input device. No voice processing,
//!   so the rest of the computer's audio is never muted or ducked.
//! - **Computer audio** (everyone else): loopback of the default output
//!   device. cpal implements it with Core Audio process taps on macOS 14.6+
//!   ("System Audio Recording" permission; no Screen Recording needed) and
//!   WASAPI loopback on Windows (no permission at all).
//!
//! Each source becomes one 16 kHz mono WAV on a **shared clock**: pausing
//! pauses every track, and any stretch with no audio (device switch, WASAPI
//! delivering nothing during silence) is padded with silence, so the files
//! stay sample-aligned for the dual-channel transcription pipeline. WAV
//! headers are rewritten every 2 s, so a crash leaves a readable file.

pub mod resample;
mod repair;
mod track;

use anyhow::{bail, Result};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub use repair::repair_wav;
pub use track::{Kind, TrackState, TrackStatus};

/// Every track is written at this rate — what Whisper and the VAD consume.
pub const OUT_RATE: u32 = 16_000;

/// Wall clock that stops while paused. Tracks place audio by this clock.
pub(crate) struct Clock {
    start: Instant,
    state: Mutex<ClockState>,
}

struct ClockState {
    paused_since: Option<Instant>,
    paused_total: Duration,
}

impl Clock {
    fn new() -> Self {
        Self {
            start: Instant::now(),
            state: Mutex::new(ClockState { paused_since: None, paused_total: Duration::ZERO }),
        }
    }

    /// Recording time elapsed, excluding pauses.
    pub(crate) fn active(&self) -> Duration {
        let st = self.state.lock().unwrap();
        let now = st.paused_since.unwrap_or_else(Instant::now);
        now.duration_since(self.start).saturating_sub(st.paused_total)
    }

    pub(crate) fn is_paused(&self) -> bool {
        self.state.lock().unwrap().paused_since.is_some()
    }

    fn set_paused(&self, paused: bool) {
        let mut st = self.state.lock().unwrap();
        match (paused, st.paused_since) {
            (true, None) => st.paused_since = Some(Instant::now()),
            (false, Some(since)) => {
                st.paused_total += since.elapsed();
                st.paused_since = None;
            }
            _ => {}
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Sources {
    pub mic: bool,
    pub system: bool,
}

pub struct Levels {
    pub mic: f32,
    pub system: f32,
}

pub struct Finished {
    pub mic: Option<PathBuf>,
    pub system: Option<PathBuf>,
    pub duration_secs: f32,
}

pub struct Recording {
    clock: Arc<Clock>,
    mic: Option<track::Track>,
    system: Option<track::Track>,
}

/// `{dir}/{id}.mic.wav` / `{dir}/{id}.system.wav`.
pub fn track_path(dir: &Path, id: &str, kind: Kind) -> PathBuf {
    dir.join(format!("{id}.{}.wav", kind.file_tag()))
}

impl Recording {
    pub fn start(dir: &Path, id: &str, sources: Sources) -> Result<Self> {
        if !sources.mic && !sources.system {
            bail!("nothing to record: enable the microphone and/or computer audio");
        }
        std::fs::create_dir_all(dir)?;
        let clock = Arc::new(Clock::new());
        let open = |kind| track::Track::start(kind, &track_path(dir, id, kind), clock.clone());
        let mic = if sources.mic { Some(open(Kind::Mic)?) } else { None };
        let system = if sources.system { Some(open(Kind::System)?) } else { None };
        Ok(Self { clock, mic, system })
    }

    /// Block until every track is capturing, or `timeout` passes. Opening a
    /// macOS process tap can take a couple of seconds (and waits on the
    /// permission prompt the first time) — the UI should say "Starting…"
    /// until this returns true. Returns false on timeout.
    pub fn wait_live(&self, timeout: Duration) -> bool {
        let until = Instant::now() + timeout;
        loop {
            if self.status().iter().all(|t| t.state == TrackState::Live) {
                return true;
            }
            if Instant::now() >= until {
                return false;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// Pauses every track at once, so the channels stay aligned.
    pub fn pause(&self) {
        self.clock.set_paused(true);
    }

    pub fn resume(&self) {
        self.clock.set_paused(false);
    }

    pub fn is_paused(&self) -> bool {
        self.clock.is_paused()
    }

    /// Recorded time so far, excluding pauses.
    pub fn elapsed(&self) -> Duration {
        self.clock.active()
    }

    pub fn levels(&self) -> Levels {
        Levels {
            mic: self.mic.as_ref().map_or(0.0, |t| t.level()),
            system: self.system.as_ref().map_or(0.0, |t| t.level()),
        }
    }

    pub fn status(&self) -> Vec<TrackStatus> {
        [&self.mic, &self.system].into_iter().flatten().map(|t| t.status()).collect()
    }

    /// Stop capture and finalize the WAVs. Both end at the same clock time.
    pub fn stop(self) -> Result<Finished> {
        // Freeze the clock first: every writer pads to the same end time.
        self.clock.set_paused(true);
        let duration_secs = self.clock.active().as_secs_f32();
        let mic = self.mic.map(|t| t.stop()).transpose()?;
        let system = self.system.map(|t| t.stop()).transpose()?;
        Ok(Finished { mic, system, duration_secs })
    }
}
