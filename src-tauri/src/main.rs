// ItsMyPA desktop app: the screens (frontend/) plus the on-device Rust engine,
// in one process. No Python, no local web server.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod ai;
mod api;
mod jobs;
mod recording;
mod state;

use serde_json::Value;
use std::sync::Arc;
use tauri::Manager;

use state::AppState;

/// Every screen request (see api.rs and frontend/api.js). The work runs on a
/// background thread so the window never freezes.
#[tauri::command]
async fn api(state: tauri::State<'_, Arc<AppState>>, method: String, path: String, body: Option<Value>) -> Result<Value, api::ApiError> {
    let state = Arc::clone(&state);
    tauri::async_runtime::spawn_blocking(move || api::handle(&state, &method, &path, body))
        .await
        .map_err(|e| api::ApiError::new(500, e.to_string()))?
}

/// The AI requests that stream their answer (see ai.rs). Each piece of text
/// goes to the screen through `on_piece` as soon as it's written.
#[tauri::command]
async fn api_stream(
    state: tauri::State<'_, Arc<AppState>>,
    path: String,
    body: Option<Value>,
    on_piece: tauri::ipc::Channel<String>,
) -> Result<(), api::ApiError> {
    let state = Arc::clone(&state);
    tauri::async_runtime::spawn_blocking(move || {
        // A failed send means the screen went away: stop generating.
        ai::stream(&state, &path, body, &mut |piece| on_piece.send(piece.to_string()).is_ok())
    })
    .await
    .map_err(|e| api::ApiError::new(500, e.to_string()))?
}

/// An audio file the user picked to transcribe, sent as raw bytes (too big
/// to pass as JSON). Saved into the recordings folder; returns its id for
/// `/transcribe`.
#[tauri::command]
fn upload(request: tauri::ipc::Request<'_>) -> Result<String, api::ApiError> {
    let tauri::ipc::InvokeBody::Raw(bytes) = request.body() else {
        return Err(api::ApiError::new(400, "Expected the file's bytes"));
    };
    let name = request.headers().get("x-file-name").and_then(|v| v.to_str().ok()).unwrap_or("audio");
    // Keep only a safe extension; it tells the decoder the format.
    let ext: String = std::path::Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .filter(|e| e.len() <= 5 && e.chars().all(|c| c.is_ascii_alphanumeric()))
        .unwrap_or("bin")
        .to_lowercase();
    let id = format!("upload-{}.{ext}", uuid::Uuid::new_v4());
    let dir = itsmypa_engine::paths::recordings_dir();
    std::fs::create_dir_all(&dir).map_err(|e| api::ApiError::new(500, e.to_string()))?;
    std::fs::write(dir.join(&id), bytes).map_err(|e| api::ApiError::new(500, format!("Could not save the file: {e}")))?;
    Ok(id)
}

/// Path of an uploaded file from its id — only ids `upload` could have made.
pub fn upload_path(id: &str) -> Option<std::path::PathBuf> {
    let ok = id.starts_with("upload-") && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.') && !id.contains("..");
    ok.then(|| itsmypa_engine::paths::recordings_dir().join(id))
}

/// Development aid: with ITSMYPA_DEBUG_REPORT=<file>, the page writes its
/// visible text and any script errors there after launch (6 s, or
/// ITSMYPA_DEBUG_REPORT_AFTER seconds). ITSMYPA_DEBUG_SCRIPT=<file.js> runs a
/// test script first; it reports by calling `__report(text)` itself.
#[tauri::command]
fn debug_report(text: String) {
    if let Some(path) = std::env::var_os("ITSMYPA_DEBUG_REPORT") {
        let _ = std::fs::write(path, text);
    }
}

fn main() {
    let state = AppState::open().expect("could not open the ItsMyPA data folder");
    state.load_models();

    tauri::Builder::default()
        .manage(Arc::new(state))
        .invoke_handler(tauri::generate_handler![api, api_stream, upload, debug_report])
        .setup(|app| {
            if std::env::var_os("ITSMYPA_DEBUG_REPORT").is_some() {
                let win = app.get_webview_window("main").expect("main window");
                let after = std::env::var("ITSMYPA_DEBUG_REPORT_AFTER").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
                let script = std::env::var_os("ITSMYPA_DEBUG_SCRIPT")
                    .and_then(|p| std::fs::read_to_string(p).ok())
                    .unwrap_or_else(|| "__report(document.body.innerText);".into());
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_secs(after));
                    let _ = win.eval(&format!(
                        "(async () => {{ const __report = t => window.__TAURI__.core.invoke('debug_report', \
                         {{text: 'ERRORS: ' + JSON.stringify(window.__itsmypaErrors || []) + '\\n\\n' + t}}); \
                         try {{ {script} }} catch (e) {{ __report('SCRIPT FAILED: ' + e); }} }})()"
                    ));
                });
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building ItsMyPA")
        .run(|app, event| {
            // Finish any running capture so its audio is complete on disk and
            // the audio devices are released cleanly.
            if let tauri::RunEvent::Exit = event {
                app.state::<Arc<AppState>>().recorder.stop_on_exit();
            }
        });
}
