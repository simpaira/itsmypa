//! Transcription jobs. A finished recording (or an uploaded file) is queued
//! and transcribed on one background worker, so a long meeting never blocks
//! the window; the screen polls `/jobs/{id}` for progress.

use anyhow::{bail, Result};
use itsmypa_engine::audio;
use itsmypa_engine::pipeline::{self, Phase, Transcript};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};

use crate::recording::Finished;
use crate::state::Models;

/// What to transcribe.
pub enum Input {
    /// A native recording: mic and/or computer-audio WAVs on one clock.
    Recording(Finished),
    /// A file the user picked (any format `audio::load_audio_16k` reads).
    Upload(PathBuf),
}

#[derive(Clone)]
struct Job {
    status: &'static str, // queued | processing | done | error
    phase: Option<&'static str>,
    progress: f32,
    transcript: String,
    segments: Option<Value>,
    error: Option<String>,
}

pub struct Jobs {
    jobs: Arc<Mutex<HashMap<String, Job>>>,
    queue: Sender<(String, Input)>,
    /// Audio files of queued or running jobs (not crash leftovers).
    busy: Arc<Mutex<HashSet<PathBuf>>>,
}

impl Jobs {
    pub fn start(models: Arc<Models>) -> Self {
        let jobs: Arc<Mutex<HashMap<String, Job>>> = Arc::default();
        let busy: Arc<Mutex<HashSet<PathBuf>>> = Arc::default();
        let (queue, rx) = mpsc::channel::<(String, Input)>();
        let (worker_jobs, worker_busy) = (jobs.clone(), busy.clone());
        std::thread::Builder::new()
            .name("transcribe-worker".into())
            .spawn(move || {
                for (id, input) in rx {
                    let update = |f: &dyn Fn(&mut Job)| {
                        if let Some(j) = worker_jobs.lock().unwrap().get_mut(&id) {
                            f(j);
                        }
                    };
                    update(&|j| j.status = "processing");
                    let files = input_files(&input);
                    let result = run(&models, &input, &mut |phase, p| {
                        update(&|j| {
                            j.phase = Some(match phase {
                                Phase::Diarizing => "diarizing",
                                Phase::Transcribing => "transcribing",
                            });
                            j.progress = p;
                        })
                    });
                    match result {
                        Ok(t) => {
                            update(&|j| {
                                j.status = "done";
                                j.progress = 1.0;
                                j.transcript = t.text.clone();
                                j.segments = t.segments.as_ref().map(|s| serde_json::to_value(s).unwrap_or(Value::Null));
                            });
                            // The transcript now holds everything; audio isn't kept.
                            for f in files {
                                let _ = std::fs::remove_file(f);
                            }
                        }
                        // On failure the audio stays on disk so it can be retried.
                        Err(e) => update(&|j| {
                            j.status = "error";
                            j.error = Some(format!("{e:#}"));
                        }),
                    }
                    update(&|j| j.phase = None);
                    let mut b = worker_busy.lock().unwrap();
                    for f in input_files(&input) {
                        b.remove(&f);
                    }
                }
            })
            .expect("could not start the transcription worker");
        Self { jobs, queue, busy }
    }

    pub fn is_busy(&self, path: &std::path::Path) -> bool {
        self.busy.lock().unwrap().contains(path)
    }

    pub fn enqueue(&self, input: Input) -> String {
        self.busy.lock().unwrap().extend(input_files(&input));
        let id = uuid::Uuid::new_v4().to_string();
        let job = Job { status: "queued", phase: Some("queued"), progress: 0.0, transcript: String::new(), segments: None, error: None };
        self.jobs.lock().unwrap().insert(id.clone(), job);
        let _ = self.queue.send((id.clone(), input));
        id
    }

    /// Same shape the Python engine returned, which the screens poll for.
    pub fn status(&self, id: &str) -> Option<Value> {
        let jobs = self.jobs.lock().unwrap();
        let j = jobs.get(id)?;
        let done = j.status == "done";
        Some(json!({
            "id": id,
            "status": j.status,
            "phase": j.phase,
            "progress": (j.progress * 1000.0).round() / 1000.0,
            "transcript": j.transcript,
            "segments": if done { j.segments.clone() } else { None },
            "diarized": done && j.segments.is_some(),
            "error": j.error,
        }))
    }
}

fn input_files(input: &Input) -> Vec<PathBuf> {
    match input {
        Input::Recording(f) => f.mic.iter().chain(f.system.iter()).cloned().collect(),
        Input::Upload(p) => vec![p.clone()],
    }
}

fn run(models: &Models, input: &Input, progress: &mut dyn FnMut(Phase, f32)) -> Result<Transcript> {
    let asr = models.asr.lock().unwrap();
    let Some(asr) = asr.as_ref() else { bail!("Transcription models are not loaded yet — try again in a moment.") };
    let diarizer = models.diarizer.lock().unwrap();
    match input {
        Input::Recording(f) => match (&f.mic, &f.system) {
            (Some(mic), Some(system)) => pipeline::transcribe_dual(
                asr,
                diarizer.as_ref(),
                &audio::load_audio_16k(mic)?,
                &audio::load_audio_16k(system)?,
                progress,
            ),
            (Some(only), None) | (None, Some(only)) => {
                pipeline::transcribe_single(asr, diarizer.as_ref(), &audio::load_audio_16k(only)?, progress)
            }
            (None, None) => bail!("No audio was captured."),
        },
        Input::Upload(path) => pipeline::transcribe_single(asr, diarizer.as_ref(), &audio::load_audio_16k(path)?, progress),
    }
}
