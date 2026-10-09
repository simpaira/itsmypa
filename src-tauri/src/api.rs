//! The screens' requests, answered by the Rust engine.
//!
//! The UI was written against the Python engine's HTTP API (`fetch('/notes')`
//! etc.). frontend/api.js routes those same calls to the `api` command, so the
//! screens keep working unchanged: same paths, same JSON in and out, same
//! error shape (`{"detail": …}` with a status code).

use itsmypa_engine::settings::{self, SettingsUpdate};
use itsmypa_engine::store::{Note, NotePatch};
use serde::Serialize;
use serde_json::{json, Value};

use crate::jobs::Input;
use crate::state::AppState;

#[derive(Debug, Serialize)]
pub struct ApiError {
    pub status: u16,
    pub detail: String,
}

impl ApiError {
    pub fn new(status: u16, detail: impl Into<String>) -> Self {
        Self { status, detail: detail.into() }
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        Self::new(500, format!("{e:#}"))
    }
}

type ApiResult = Result<Value, ApiError>;

fn parse<T: serde::de::DeserializeOwned>(body: Option<Value>) -> Result<T, ApiError> {
    serde_json::from_value(body.unwrap_or(Value::Null)).map_err(|e| ApiError::new(400, format!("Invalid request: {e}")))
}

fn to_json<T: Serialize>(v: T) -> ApiResult {
    serde_json::to_value(v).map_err(|e| ApiError::new(500, e.to_string()))
}

/// A file the UI attached to a form, as api.js sends it.
#[derive(serde::Deserialize)]
struct Upload {
    text: Option<String>,
}

pub fn handle(state: &AppState, method: &str, path: &str, body: Option<Value>) -> ApiResult {
    let segs: Vec<&str> = path.trim_matches('/').split('/').collect();
    match (method, segs.as_slice()) {
        ("GET", ["health"]) => health(state),
        ("GET", ["setup", "status"]) => setup_status(state),

        ("GET", ["notes"]) => to_json(state.store.list()?),
        ("POST", ["notes"]) => to_json(state.store.create(parse::<Note>(body)?)?),
        ("PATCH", ["notes", id]) => match state.store.update(id, parse::<NotePatch>(body)?)? {
            Some(n) => to_json(n),
            None => Err(ApiError::new(404, "Note not found")),
        },
        ("DELETE", ["notes", id]) => {
            state.store.delete(id)?;
            Ok(json!({"ok": true}))
        }

        ("GET", ["settings"]) => {
            let mut v = to_json(state.settings())?;
            v["llm_name"] = json!(state.models.llm_name.lock().unwrap().clone());
            Ok(v)
        }
        ("POST", ["settings"]) => {
            let saved = settings::update(&state.store, parse::<SettingsUpdate>(body)?).map_err(|e| ApiError::new(400, e.to_string()))?;
            if saved.reload_models {
                // Applies right away: the home screen shows the reload progress.
                state.load_models();
            }
            Ok(json!({"ok": true, "restart_required": false}))
        }

        ("GET", ["export"]) => export(state),
        ("POST", ["import"]) => {
            #[derive(serde::Deserialize)]
            struct Form {
                backup: Option<Upload>,
            }
            let text = parse::<Form>(body)?.backup.and_then(|u| u.text).ok_or_else(|| ApiError::new(400, "No backup file"))?;
            let r = state.store.import_json(&text).map_err(|e| ApiError::new(400, e.to_string()))?;
            Ok(json!({"ok": true, "imported": r.imported, "skipped": r.skipped}))
        }

        ("POST", ["recording", "start"]) => {
            #[derive(serde::Deserialize)]
            struct Start {
                mic: bool,
                system: bool,
            }
            let s = parse::<Start>(body)?;
            state.recorder.start(s.mic, s.system).map_err(|e| ApiError::new(500, format!("Could not start recording: {e:#}")))
        }
        ("GET", ["recording", "level"]) => Ok(state.recorder.level()),
        ("POST", ["recording", "pause"]) => {
            state.recorder.set_paused(true)?;
            Ok(json!({"ok": true}))
        }
        ("POST", ["recording", "resume"]) => {
            state.recorder.set_paused(false)?;
            Ok(json!({"ok": true}))
        }
        ("POST", ["recording", "stop"]) => {
            let f = state.recorder.stop()?;
            let out = json!({"recording_id": f.id, "duration": f.duration_secs});
            state.finished.lock().unwrap().insert(f.id.clone(), f);
            Ok(out)
        }

        ("POST", ["transcribe"]) => transcribe(state, body),
        ("GET", ["jobs", id]) => state.jobs.status(id).ok_or_else(|| ApiError::new(404, "Job not found")),

        // Wired in later groups (recovery, calendar, permissions); safe
        // placeholders so the home screen loads.
        ("GET", ["recordings"]) => Ok(json!([])),
        ("GET", ["calendar", "upcoming"]) => Ok(json!({"configured": false, "events": [], "error": null})),
        ("GET", ["mic", "status"]) | ("GET", ["system_audio", "status"]) => Ok(json!({"available": true, "state": "granted"})),

        _ => Err(ApiError::new(404, format!("{method} {path} is not available yet"))),
    }
}

/// Queue a finished recording (`recording_id`) or an uploaded file
/// (`audio.upload_id`, see the `upload` command) for transcription.
fn transcribe(state: &AppState, body: Option<Value>) -> ApiResult {
    if state.models.asr.lock().unwrap().is_none() {
        let msg = if state.models_loading() {
            "Still warming up the AI models — try again in a few seconds."
        } else {
            "Transcription models are not loaded — check Settings."
        };
        return Err(ApiError::new(503, msg));
    }
    #[derive(serde::Deserialize)]
    struct UploadRef {
        upload_id: String,
    }
    #[derive(serde::Deserialize)]
    struct Req {
        recording_id: Option<String>,
        audio: Option<UploadRef>,
    }
    let req = parse::<Req>(body)?;
    let input = if let Some(id) = req.recording_id {
        let f = state.finished.lock().unwrap().remove(&id).ok_or_else(|| ApiError::new(404, "Recording not found"))?;
        Input::Recording(f)
    } else if let Some(up) = req.audio {
        Input::Upload(crate::upload_path(&up.upload_id).ok_or_else(|| ApiError::new(400, "Bad upload id"))?)
    } else {
        return Err(ApiError::new(400, "No audio was captured."));
    };
    Ok(json!({"job_id": state.jobs.enqueue(input)}))
}

fn health(state: &AppState) -> ApiResult {
    let s = state.settings();
    let m = &state.models;
    Ok(json!({
        "whisper": s.whisper_model,
        "engine": "itsmypa-rust",
        "loading": state.models_loading(),
        "asr_ready": m.asr.lock().unwrap().is_some(),
        "ffmpeg": true,
        "model": m.llm_name.lock().unwrap().clone(),
        "model_ready": m.llm.lock().unwrap().is_some(),
        "diarize": m.diarizer.lock().unwrap().is_some(),
        "system_audio": true,
    }))
}

fn setup_status(state: &AppState) -> ApiResult {
    let components = state.setup.lock().unwrap().clone();
    let terminal = |s: &str| matches!(s, "ready" | "error" | "skipped");
    let pct = |c: &crate::state::Component| match c.status {
        "ready" | "skipped" | "error" => 100.0,
        "downloading" | "extracting" if c.total > 0 => (c.bytes as f64 / c.total as f64 * 100.0).min(99.0),
        _ => 0.0,
    };
    // Recording only truly needs transcription and the notes AI to reach a
    // terminal state; diarization is optional and degrades gracefully.
    let ready = components.iter().filter(|c| c.id == "asr" || c.id == "llm").all(|c| terminal(c.status));
    let overall = if components.is_empty() { 100.0 } else { components.iter().map(pct).sum::<f64>() / components.len() as f64 };
    Ok(json!({
        "ready": ready,
        "has_error": components.iter().any(|c| c.status == "error"),
        "overall_percent": (overall * 10.0).round() / 10.0,
        "components": components,
    }))
}

/// Saves the backup straight into the Downloads folder (the webview can't
/// download files) and tells the UI where it went.
fn export(state: &AppState) -> ApiResult {
    let dir = dirs::download_dir().unwrap_or_else(itsmypa_engine::paths::data_dir);
    let path = dir.join(format!("itsmypa-backup-{}.json", chrono::Local::now().format("%Y%m%d-%H%M")));
    std::fs::write(&path, state.store.export_json()?).map_err(|e| ApiError::new(500, format!("Could not save the backup: {e}")))?;
    Ok(json!({"ok": true, "path": path.display().to_string()}))
}
