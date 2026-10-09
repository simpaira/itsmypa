//! Fix the header of a WAV left behind by a crash.
//!
//!   cargo run -p itsmypa-capture --example repair -- <file.wav>

fn main() -> anyhow::Result<()> {
    let path = std::path::PathBuf::from(std::env::args().nth(1).expect("usage: repair <file.wav>"));
    let bytes = itsmypa_capture::repair_wav(&path)?;
    println!("{}: {:.2}s of audio", path.display(), bytes as f32 / 2.0 / itsmypa_capture::OUT_RATE as f32);
    Ok(())
}
