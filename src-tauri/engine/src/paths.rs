//! Where mutable state lives. Same directory the Python engine used
//! (platformdirs `user_data_dir("ItsMyPA", appauthor=False)`), so upgrading
//! users keep their notes and their ~2.5 GB of already-downloaded models.

use std::path::PathBuf;

pub const APP_NAME: &str = "ItsMyPA";

pub fn data_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("ITSMYPA_DATA_DIR") {
        return PathBuf::from(p);
    }
    // macOS: ~/Library/Application Support/ItsMyPA · Windows: %LOCALAPPDATA%\ItsMyPA
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(APP_NAME)
}

pub fn models_dir() -> PathBuf {
    data_dir().join("models")
}

pub fn recordings_dir() -> PathBuf {
    data_dir().join("recordings")
}
