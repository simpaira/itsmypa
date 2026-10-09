//! One capture source → one aligned 16 kHz WAV.
//!
//! Two threads per track:
//! - **supervisor** owns the cpal stream (not `Send` on every platform). It
//!   opens the current default device and rebuilds the stream whenever that
//!   device fails or the system default changes (AirPods connect/disconnect).
//! - **writer** receives audio, resamples it to 16 kHz, and places it on the
//!   recording clock, padding silence wherever audio is missing.

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::resample::Resampler;
use crate::{Clock, OUT_RATE};

/// How often the supervisor checks for device changes and stop requests.
const POLL: Duration = Duration::from_millis(500);
/// No audio for this long = a gap (device switch, silent WASAPI loopback):
/// the writer pads silence up to the clock.
const GAP: Duration = Duration::from_millis(600);
/// Audio still in flight from the device when padding — never pad over it.
const IN_FLIGHT: Duration = Duration::from_millis(300);
/// Rewrite the WAV header this often so a crash leaves a readable file.
const HEADER_EVERY: Duration = Duration::from_secs(2);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Mic,
    System,
}

impl Kind {
    pub(crate) fn file_tag(self) -> &'static str {
        match self {
            Kind::Mic => "mic",
            Kind::System => "system",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Kind::Mic => "microphone",
            Kind::System => "computer audio",
        }
    }

    fn default_device(self, host: &cpal::Host) -> Option<cpal::Device> {
        match self {
            Kind::Mic => host.default_input_device(),
            // Loopback: cpal records an *output* device when asked for input.
            Kind::System => host.default_output_device(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackState {
    Starting,
    Live,
    /// Device lost or switching; recording continues (silence is padded).
    Reconnecting,
}

#[derive(Clone, Debug)]
pub struct TrackStatus {
    pub kind: Kind,
    pub state: TrackState,
    pub device: String,
    /// Last problem, for the UI ("No microphone found", OS error, …).
    pub message: Option<String>,
}

enum Msg {
    /// Mono samples at the device rate; `end` is the clock time at delivery.
    Audio { rate: u32, samples: Vec<f32>, end: Duration },
    /// The stream was rebuilt: flush the resampler, re-align on next audio.
    Restart,
}

struct Shared {
    stop: AtomicBool,
    level: AtomicU32, // f32 bits
    status: Mutex<TrackStatus>,
}

impl Shared {
    fn set_status(&self, state: TrackState, device: Option<String>, message: Option<String>) {
        let mut st = self.status.lock().unwrap();
        st.state = state;
        if let Some(d) = device {
            st.device = d;
        }
        st.message = message;
    }
}

pub(crate) struct Track {
    path: PathBuf,
    shared: Arc<Shared>,
    supervisor: JoinHandle<()>,
    writer: JoinHandle<Result<()>>,
}

impl Track {
    pub(crate) fn start(kind: Kind, path: &Path, clock: Arc<Clock>) -> Result<Self> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: OUT_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let wav = hound::WavWriter::create(path, spec)
            .with_context(|| format!("could not create {}", path.display()))?;

        let shared = Arc::new(Shared {
            stop: AtomicBool::new(false),
            level: AtomicU32::new(0),
            status: Mutex::new(TrackStatus {
                kind,
                state: TrackState::Starting,
                device: String::new(),
                message: None,
            }),
        });
        let (tx, rx) = mpsc::channel::<Msg>();

        let writer = {
            let (shared, clock) = (shared.clone(), clock.clone());
            thread::Builder::new()
                .name(format!("capture-{}-writer", kind.file_tag()))
                .spawn(move || write_loop(wav, rx, &clock, &shared))?
        };
        let supervisor = {
            let shared = shared.clone();
            thread::Builder::new()
                .name(format!("capture-{}", kind.file_tag()))
                .spawn(move || supervise(kind, tx, clock, shared))?
        };
        Ok(Self { path: path.to_path_buf(), shared, supervisor, writer })
    }

    pub(crate) fn level(&self) -> f32 {
        f32::from_bits(self.shared.level.load(Ordering::Relaxed))
    }

    pub(crate) fn status(&self) -> TrackStatus {
        self.shared.status.lock().unwrap().clone()
    }

    pub(crate) fn stop(self) -> Result<PathBuf> {
        self.shared.stop.store(true, Ordering::SeqCst);
        // Supervisor drops the stream (and with it every Sender) on exit,
        // which ends the writer's receive loop.
        self.supervisor.join().map_err(|_| anyhow!("capture thread panicked"))?;
        self.writer.join().map_err(|_| anyhow!("writer thread panicked"))??;
        Ok(self.path)
    }
}

// ── supervisor ──────────────────────────────────────────────────────────────

fn supervise(kind: Kind, tx: Sender<Msg>, clock: Arc<Clock>, shared: Arc<Shared>) {
    let host = cpal::default_host();
    let mut first = true;
    while !shared.stop.load(Ordering::SeqCst) {
        if !first {
            let _ = tx.send(Msg::Restart);
        }
        first = false;

        let Some(device) = kind.default_device(&host) else {
            let msg = match kind {
                Kind::Mic => "No microphone found",
                Kind::System => "No audio output device found",
            };
            shared.set_status(TrackState::Reconnecting, None, Some(msg.into()));
            sleep_unless_stopped(&shared, Duration::from_secs(1));
            continue;
        };
        let id = device.id().ok();
        let name = device.description().map(|d| d.name().to_string()).unwrap_or_default();

        let broken = Arc::new(AtomicBool::new(false));
        let stream = match open_stream(kind, &device, tx.clone(), clock.clone(), shared.clone(), broken.clone()) {
            Ok(s) => s,
            Err(e) => {
                shared.set_status(
                    TrackState::Reconnecting,
                    Some(name),
                    Some(format!("Could not open {}: {e:#}", kind.label())),
                );
                sleep_unless_stopped(&shared, Duration::from_secs(1));
                continue;
            }
        };
        shared.set_status(TrackState::Live, Some(name), None);

        // Hold the stream until it breaks, the default device changes, or stop.
        loop {
            thread::sleep(POLL);
            if shared.stop.load(Ordering::SeqCst) || broken.load(Ordering::SeqCst) {
                break;
            }
            let current = kind.default_device(&host).and_then(|d| d.id().ok());
            if current != id {
                break;
            }
        }
        drop(stream);
        if !shared.stop.load(Ordering::SeqCst) {
            shared.set_status(TrackState::Reconnecting, None, None);
        }
    }
}

fn sleep_unless_stopped(shared: &Shared, total: Duration) {
    let until = Instant::now() + total;
    while Instant::now() < until && !shared.stop.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(100));
    }
}

fn open_stream(
    kind: Kind,
    device: &cpal::Device,
    tx: Sender<Msg>,
    clock: Arc<Clock>,
    shared: Arc<Shared>,
    broken: Arc<AtomicBool>,
) -> Result<cpal::Stream> {
    let supported = match kind {
        Kind::Mic => device.default_input_config()?,
        Kind::System => device.default_output_config()?,
    };
    let format = supported.sample_format();
    let config = supported.config();
    let stream = match format {
        SampleFormat::F32 => build::<f32>(device, &config, tx, clock, shared, broken)?,
        SampleFormat::I16 => build::<i16>(device, &config, tx, clock, shared, broken)?,
        SampleFormat::I32 => build::<i32>(device, &config, tx, clock, shared, broken)?,
        SampleFormat::U16 => build::<u16>(device, &config, tx, clock, shared, broken)?,
        other => return Err(anyhow!("unsupported sample format {other:?}")),
    };
    stream.play()?;
    Ok(stream)
}

fn build<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    tx: Sender<Msg>,
    clock: Arc<Clock>,
    shared: Arc<Shared>,
    broken: Arc<AtomicBool>,
) -> Result<cpal::Stream>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = config.channels.max(1) as usize;
    let rate = config.sample_rate;
    let data_shared = shared.clone();
    let on_data = move |data: &[T], _: &cpal::InputCallbackInfo| {
        if clock.is_paused() {
            return; // paused: drop audio on every track alike
        }
        let mono: Vec<f32> = data
            .chunks_exact(channels)
            .map(|f| f.iter().map(|&s| <f32 as FromSample<T>>::from_sample_(s)).sum::<f32>() / channels as f32)
            .collect();
        if mono.is_empty() {
            return;
        }
        let rms = (mono.iter().map(|s| s * s).sum::<f32>() / mono.len() as f32).sqrt();
        data_shared.level.store((rms * 4.0).min(1.0).to_bits(), Ordering::Relaxed);
        let _ = tx.send(Msg::Audio { rate, samples: mono, end: clock.active() });
    };
    let on_error = move |err: cpal::Error| {
        // The OS already rerouted to the new default: capture continues.
        if err.kind() == cpal::ErrorKind::DeviceChanged {
            return;
        }
        shared.status.lock().unwrap().message = Some(err.to_string());
        broken.store(true, Ordering::SeqCst);
    };
    Ok(device.build_input_stream::<T, _, _>(config.clone(), on_data, on_error, None)?)
}

// ── writer ──────────────────────────────────────────────────────────────────

struct Placed<W: std::io::Write + std::io::Seek> {
    wav: hound::WavWriter<W>,
    written: u64, // samples at OUT_RATE
}

impl<W: std::io::Write + std::io::Seek> Placed<W> {
    fn write(&mut self, samples: &[f32]) -> Result<()> {
        for &s in samples {
            self.wav.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
        }
        self.written += samples.len() as u64;
        Ok(())
    }

    /// Pad with silence until the file reaches clock time `at`.
    fn pad_to(&mut self, at: Duration) -> Result<()> {
        let target = (at.as_secs_f64() * OUT_RATE as f64) as u64;
        for _ in self.written..target {
            self.wav.write_sample(0i16)?;
        }
        self.written = self.written.max(target);
        Ok(())
    }
}

fn write_loop(
    wav: hound::WavWriter<std::io::BufWriter<std::fs::File>>,
    rx: mpsc::Receiver<Msg>,
    clock: &Clock,
    shared: &Shared,
) -> Result<()> {
    let mut out = Placed { wav, written: 0 };
    let mut rs: Option<(u32, Resampler)> = None;
    let mut last_audio = Instant::now();
    let mut last_header = Instant::now();

    loop {
        match rx.recv_timeout(Duration::from_millis(250)) {
            Ok(Msg::Audio { rate, samples, end }) => {
                if rs.as_ref().map(|(r, _)| *r) != Some(rate) {
                    if let Some((_, mut old)) = rs.take() {
                        out.write(&old.flush())?;
                    }
                    // (Re)aligning: this chunk started at `end - its length`.
                    let begin = end.saturating_sub(Duration::from_secs_f64(samples.len() as f64 / rate as f64));
                    out.pad_to(begin)?;
                    rs = Some((rate, Resampler::new(rate, OUT_RATE)));
                }
                let resampled = rs.as_mut().unwrap().1.process(&samples);
                out.write(&resampled)?;
                last_audio = Instant::now();
            }
            Ok(Msg::Restart) => {
                if let Some((_, mut old)) = rs.take() {
                    out.write(&old.flush())?;
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                // No audio arriving (silent WASAPI loopback, device switch,
                // pause): keep the file on the clock with silence.
                if last_audio.elapsed() > GAP && !clock.is_paused() {
                    shared.level.store(0f32.to_bits(), Ordering::Relaxed);
                    if let Some((_, mut old)) = rs.take() {
                        out.write(&old.flush())?;
                    }
                    out.pad_to(clock.active().saturating_sub(IN_FLIGHT))?;
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
        if last_header.elapsed() > HEADER_EVERY {
            out.wav.flush()?;
            last_header = Instant::now();
        }
    }

    if let Some((_, mut old)) = rs.take() {
        out.write(&old.flush())?;
    }
    // The clock is frozen at stop, so every track is padded to the same end
    // time (a track can run a few ms longer with audio still in flight).
    out.pad_to(clock.active())?;
    out.wav.finalize()?;
    Ok(())
}
