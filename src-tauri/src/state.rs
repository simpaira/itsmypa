//! Everything the running app holds: saved meetings, settings, and the AI
//! models, which load (and on first run, download) in the background while
//! the screens are already usable.

use anyhow::Result;
use itsmypa_engine::asr::{Asr, ModelPaths};
use itsmypa_engine::ask::Asker;
use itsmypa_engine::diarize::{DiarizeConfig, Diarizer};
use itsmypa_engine::llm::{Llm, DEFAULT_N_CTX};
use itsmypa_engine::models::{self, ModelFile, Phase};
use itsmypa_engine::settings::{self, Settings};
use itsmypa_engine::store::Store;
use itsmypa_engine::paths;
use serde::Serialize;

use crate::calendar::Calendar;
use crate::jobs::Jobs;
use crate::recording::{Finished, Recorder};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// One row of the first-run setup list the home screen shows.
#[derive(Clone, Debug, Serialize)]
pub struct Component {
    pub id: &'static str,
    pub label: String,
    /// pending | downloading | verifying | extracting | loading | ready | skipped | error
    pub status: &'static str,
    pub bytes: u64,
    pub total: u64,
    /// Bytes this model still needed to download when setup started (0 if
    /// it was already installed) — what the progress bar counts.
    pub size: u64,
    /// Bytes already on disk from an interrupted download (resumed, so not
    /// counted as progress).
    pub resumed: u64,
    pub error: Option<String>,
}

/// The loaded models. Each sits behind its own lock: the speech models and
/// the notes model can't run two jobs at once, and a lock is the simplest
/// way to make work queue up instead of colliding.
#[derive(Default)]
pub struct Models {
    pub asr: Mutex<Option<Asr>>,
    pub diarizer: Mutex<Option<Diarizer>>,
    pub llm: Mutex<Option<Llm>>,
    pub llm_name: Mutex<String>,
}

pub struct AppState {
    pub store: Store,
    pub asker: Asker,
    pub recorder: Recorder,
    pub jobs: Jobs,
    pub calendar: Calendar,
    /// Recordings stopped but not yet handed to /transcribe, by id.
    pub finished: Mutex<HashMap<String, Finished>>,
    pub models: Arc<Models>,
    pub setup: Arc<Mutex<Vec<Component>>>,
    pub loading: Arc<AtomicBool>,
    /// Bumped on every (re)load, so a stale loader stops updating status.
    generation: Arc<AtomicU64>,
}

impl AppState {
    pub fn open() -> Result<Self> {
        let data = paths::data_dir();
        let store = Store::open(&data.join("meetings.db"))?;
        // First launch after upgrading from the Python app: copy its meetings
        // and settings in (both files are left untouched as a backup).
        let copied = store.import_legacy_json(&data.join("notes.json"))?;
        if copied > 0 {
            eprintln!("Copied {copied} meetings from notes.json");
        }
        settings::import_legacy_env(&store, &data.join(".env"))?;
        let removed = crate::recovery::remove_empty_legacy();
        if removed > 0 {
            eprintln!("Removed {removed} empty leftover recordings from the old app");
        }
        let models = Arc::new(Models::default());
        Ok(Self {
            store,
            asker: Asker::default(),
            recorder: Recorder::default(),
            jobs: Jobs::start(models.clone()),
            calendar: Calendar::default(),
            finished: Mutex::default(),
            models,
            setup: Arc::new(Mutex::new(Vec::new())),
            loading: Arc::new(AtomicBool::new(true)),
            generation: Arc::new(AtomicU64::new(0)),
        })
    }

    pub fn settings(&self) -> Settings {
        settings::load(&self.store).unwrap_or_default()
    }

    /// (Re)load every model for the current settings in the background,
    /// downloading whatever is missing. The screens poll `setup` for progress.
    pub fn load_models(&self) {
        let gen = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        let s = self.settings();

        // What's needed, in load order.
        let mut plan: Vec<(&'static str, &'static ModelFile)> = vec![("vad", &models::VAD), ("asr", &models::WHISPER)];
        if s.diarize {
            plan.push(("seg", &models::SPEAKER_SEGMENTATION));
            plan.push(("emb", &models::SPEAKER_EMBEDDING));
        }
        let llm_override = std::env::var_os("LLM_GGUF").map(std::path::PathBuf::from);
        if llm_override.is_none() {
            plan.push(("llm", &models::NOTES_LLM));
        }

        let dir = paths::models_dir();
        let component = |id: &'static str, label: String, status: &'static str| Component { id, label, status, bytes: 0, total: 0, size: 0, resumed: 0, error: None };
        let mut rows: Vec<Component> = plan
            .iter()
            .map(|(id, m)| {
                let (size, resumed) = if m.is_installed(&dir) { (0, 0) } else { let r = m.partial_bytes(&dir); (m.bytes - r, r) };
                Component { size, resumed, ..component(id, m.label.to_string(), "pending") }
            })
            .collect();
        if !s.diarize {
            rows.push(component("seg", models::SPEAKER_SEGMENTATION.label.into(), "skipped"));
            rows.push(component("emb", models::SPEAKER_EMBEDDING.label.into(), "skipped"));
        }
        if let Some(p) = &llm_override {
            rows.push(component("llm", format!("Meeting notes AI ({})", p.display()), "pending"));
        }
        *self.setup.lock().unwrap() = rows;
        self.loading.store(true, Ordering::SeqCst);

        let (setup, models_, loading, generation) = (self.setup.clone(), self.models.clone(), self.loading.clone(), self.generation.clone());
        std::thread::Builder::new()
            .name("model-loader".into())
            .spawn(move || {
                let current = || generation.load(Ordering::SeqCst) == gen;
                let set = |id: &str, f: &dyn Fn(&mut Component)| {
                    if current() {
                        if let Some(c) = setup.lock().unwrap().iter_mut().find(|c| c.id == id) {
                            f(c);
                        }
                    }
                };

                let fail = |id: &str, e: anyhow::Error| set(id, &|c| {
                    c.status = "error";
                    c.error = Some(format!("{e:#}"));
                });
                let ready = |id: &str| set(id, &|c| c.status = "ready");
                // Download one planned model if it's missing. False on failure
                // (or when it isn't part of this plan).
                let fetch = |id: &str| -> bool {
                    let Some((_, m)) = plan.iter().find(|(pid, _)| *pid == id) else { return false };
                    let res = models::download(m, &dir, &mut |phase, done, total| {
                        let status = match phase {
                            Phase::Downloading => "downloading",
                            Phase::Verifying => "verifying",
                            Phase::Unpacking => "extracting",
                        };
                        set(id, &|c| {
                            c.status = status;
                            c.bytes = done;
                            c.total = total;
                        });
                    });
                    if res.is_ok() {
                        // Also tidies installs from before unused files were removed.
                        let freed = m.remove_unused(&dir);
                        if freed > 0 {
                            eprintln!("Removed {:.0} MB of unused {} files", freed as f64 / 1e6, m.label);
                        }
                    }
                    res.map_err(|e| fail(id, e)).is_ok()
                };

                // In order of need, each downloaded *and loaded* before the
                // next starts: transcription first, so recording works while
                // the much bigger notes AI is still downloading.

                // 1. Transcription.
                *models_.asr.lock().unwrap() = None;
                if fetch("vad") && fetch("asr") && current() {
                    set("asr", &|c| c.status = "loading");
                    match Asr::load(&ModelPaths::new(&dir), &s.whisper_language) {
                        Ok(asr) => {
                            *models_.asr.lock().unwrap() = Some(asr);
                            ready("asr");
                            ready("vad");
                        }
                        Err(e) => fail("asr", e),
                    }
                }

                // 2. Speaker detection (optional: transcripts just lose labels).
                *models_.diarizer.lock().unwrap() = None;
                if s.diarize && fetch("seg") && fetch("emb") && current() {
                    let cfg = DiarizeConfig { num_speakers: s.num_speakers, threshold: s.diarize_threshold, ..Default::default() };
                    match Diarizer::load(&dir, cfg) {
                        Ok(d) => {
                            *models_.diarizer.lock().unwrap() = Some(d);
                            ready("seg");
                            ready("emb");
                        }
                        Err(e) => fail("seg", e),
                    }
                }

                // 3. Notes AI.
                let llm_path = llm_override.clone().or_else(|| (fetch("llm") && current()).then(|| models::NOTES_LLM.path(&dir)));
                *models_.llm.lock().unwrap() = None;
                if let Some(path) = llm_path {
                    set("llm", &|c| c.status = "loading");
                    match Llm::load(&path, DEFAULT_N_CTX) {
                        Ok(llm) => {
                            *models_.llm_name.lock().unwrap() = llm.name.clone();
                            *models_.llm.lock().unwrap() = Some(llm);
                            ready("llm");
                        }
                        Err(e) => fail("llm", e),
                    }
                }
                if current() {
                    loading.store(false, Ordering::SeqCst);
                }
            })
            .expect("could not start the model loader");
    }

    pub fn models_loading(&self) -> bool {
        self.loading.load(Ordering::SeqCst)
    }
}
