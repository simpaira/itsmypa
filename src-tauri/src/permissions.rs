//! Microphone and computer-audio permission, for the welcome screen.
//!
//! macOS: the microphone has a real status (AVCaptureDevice). Computer audio
//! ("System Audio Recording", used by Core Audio process taps) has no public
//! way to check — only to ask: opening a tap shows the prompt. So "enable"
//! runs a short test capture, and the answer is remembered.
//! Windows needs no permission for loopback, and the mic setting is
//! system-wide, so both report granted.

use anyhow::{bail, Result};
use itsmypa_capture::{Recording, Sources};
use std::time::Duration;

use itsmypa_engine::store::Store;

const SYSTEM_ENABLED_KEY: &str = "system_audio_enabled";

/// "granted" | "denied" | "undetermined"
pub fn mic_status() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        use objc2_av_foundation::{AVAuthorizationStatus, AVCaptureDevice, AVMediaTypeAudio};
        let Some(audio) = (unsafe { AVMediaTypeAudio }) else { return "undetermined" };
        match unsafe { AVCaptureDevice::authorizationStatusForMediaType(audio) } {
            AVAuthorizationStatus::Authorized => "granted",
            AVAuthorizationStatus::Denied | AVAuthorizationStatus::Restricted => "denied",
            _ => "undetermined",
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        "granted"
    }
}

/// Show the macOS microphone prompt (only appears the first time) and wait
/// for the answer. Returns whether the mic may be used.
pub fn request_mic() -> bool {
    #[cfg(target_os = "macos")]
    {
        use objc2::runtime::Bool;
        use objc2_av_foundation::{AVCaptureDevice, AVMediaTypeAudio};
        if mic_status() != "undetermined" {
            return mic_status() == "granted";
        }
        let Some(audio) = (unsafe { AVMediaTypeAudio }) else { return false };
        let (tx, rx) = std::sync::mpsc::channel();
        let done = block2::RcBlock::new(move |granted: Bool| {
            let _ = tx.send(granted.as_bool());
        });
        unsafe { AVCaptureDevice::requestAccessForMediaType_completionHandler(audio, &done) };
        // The user may take a while to read the prompt.
        rx.recv_timeout(Duration::from_secs(120)).unwrap_or(false)
    }
    #[cfg(not(target_os = "macos"))]
    {
        true
    }
}

/// "granted" once the user has enabled computer audio here, else "undetermined".
pub fn system_status(store: &Store) -> &'static str {
    if cfg!(not(target_os = "macos")) || store.meta(SYSTEM_ENABLED_KEY).ok().flatten().is_some() {
        "granted"
    } else {
        "undetermined"
    }
}

/// Trigger the "System Audio Recording" prompt with a short test capture
/// (nothing is kept). Returns whether capture started — i.e. the prompt was
/// answered and the tap opened.
pub fn request_system(store: &Store) -> Result<bool> {
    let dir = std::env::temp_dir().join(format!("itsmypa-permission-{}", std::process::id()));
    let rec = Recording::start(&dir, "probe", Sources { mic: false, system: true })?;
    let live = rec.wait_live(Duration::from_secs(120));
    let stopped = rec.stop();
    let _ = std::fs::remove_dir_all(&dir);
    stopped?;
    if !live {
        bail!("Computer audio didn't start. Allow ItsMyPA in System Settings → Privacy & Security → System Audio Recording Only, then try again.");
    }
    store.set_meta(SYSTEM_ENABLED_KEY, &itsmypa_engine::store::now_iso())?;
    Ok(true)
}
