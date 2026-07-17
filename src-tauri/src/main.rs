// ItsMyPA desktop shell: a native window that runs the local Python engine as a
// managed subprocess, shows a splash until it's listening, then loads the app.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tauri::{Manager, RunEvent};

struct ServerState(Mutex<Option<Child>>);

const PORT: u16 = 8765;

/// Bundled path (resource_dir/server/itsmypa-server) or dev fallback.
fn locate_server(app: &tauri::AppHandle) -> std::path::PathBuf {
    if let Ok(res) = app.path().resource_dir() {
        let p = res.join("server").join("itsmypa-server");
        if p.exists() {
            return p;
        }
    }
    std::path::PathBuf::from("src-tauri/pybundle/itsmypa-server/itsmypa-server")
}

fn server_up() -> bool {
    let addr = format!("127.0.0.1:{PORT}").parse().unwrap();
    TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok()
}

fn main() {
    tauri::Builder::default()
        .manage(ServerState(Mutex::new(None)))
        .setup(|app| {
            let handle = app.handle().clone();

            // Start the engine unless one is already listening (dev, or a stray).
            if !server_up() {
                let bin = locate_server(&handle);
                let log = std::env::temp_dir().join("itsmypa-server.log");
                let mut cmd = Command::new(&bin);
                cmd.env("ITSMYPA_EMBEDDED", "1"); // this window is the UI — don't open a browser
                if let Ok(out) = std::fs::File::create(&log) {
                    if let Ok(err) = out.try_clone() {
                        cmd.stdout(Stdio::from(out)).stderr(Stdio::from(err));
                    }
                }
                match cmd.spawn() {
                    Ok(child) => {
                        *app.state::<ServerState>().0.lock().unwrap() = Some(child);
                    }
                    Err(e) => eprintln!("ItsMyPA: failed to start engine at {bin:?}: {e}"),
                }
            }

            // Poll until the engine binds its port, then swap splash → app.
            thread::spawn(move || {
                for _ in 0..600 {
                    if server_up() {
                        if let Some(win) = handle.get_webview_window("main") {
                            // Cache-bust: the server also sends no-store headers, but
                            // WKWebView can still reuse an entry it cached before that
                            // fix shipped. A unique URL per launch sidesteps any
                            // caching layer entirely, so a rebuilt ui.html always shows.
                            let ts = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .map(|d| d.as_millis())
                                .unwrap_or(0);
                            // embedded=1 tells the UI it runs inside the desktop
                            // shell, so it records the mic natively (the webview's
                            // getUserMedia mutes system audio via voice processing).
                            let url = format!("http://localhost:{PORT}/?embedded=1&v={ts}").parse().unwrap();
                            let _ = win.navigate(url);
                        }
                        return;
                    }
                    thread::sleep(Duration::from_millis(200));
                }
            });

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building ItsMyPA")
        .run(|app, event| {
            // Don't leave the engine running after the window closes. SIGTERM
            // first so uvicorn's shutdown hook can stop any live audio-capture
            // helpers — SIGKILL orphans them mid-capture, which leaves dangling
            // audio-HAL clients and can wedge system audio (silent Mac).
            if let RunEvent::Exit = event {
                if let Some(mut child) = app.state::<ServerState>().0.lock().unwrap().take() {
                    #[cfg(unix)]
                    {
                        unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGTERM) };
                        for _ in 0..40 {
                            if matches!(child.try_wait(), Ok(Some(_))) {
                                return;
                            }
                            thread::sleep(Duration::from_millis(100));
                        }
                    }
                    let _ = child.kill();
                }
            }
        });
}
