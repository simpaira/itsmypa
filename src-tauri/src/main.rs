// ItsMyPA desktop app: the screens (frontend/) plus the on-device Rust engine,
// in one process. No Python, no local web server.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod api;
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
        .invoke_handler(tauri::generate_handler![api, debug_report])
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
        .run(tauri::generate_context!())
        .expect("error while running ItsMyPA");
}
