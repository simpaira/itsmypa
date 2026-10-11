//! Transcribe an audio file with the installed models (no speaker labels).
//!
//!   cargo run --release -p itsmypa-engine --example transcribe -- file.wav

use itsmypa_engine::{asr, audio, paths, pipeline};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let wav = PathBuf::from(args.next().expect("usage: transcribe <file.wav>"));

    let t0 = Instant::now();
    let asr = asr::Asr::load(&asr::ModelPaths::new(&paths::models_dir()), "")?;
    eprintln!("loaded whisper-small in {:.1}s", t0.elapsed().as_secs_f32());

    let samples = audio::load_audio_16k(&wav)?;
    let secs = samples.len() as f32 / itsmypa_engine::SAMPLE_RATE as f32;

    let t1 = Instant::now();
    let text = pipeline::transcribe_single(&asr, None, &samples, &mut |_, p| eprint!("\r{:3.0}%", p * 100.0))?.text;
    let took = t1.elapsed().as_secs_f32();
    eprintln!("\r{secs:.1}s of audio in {took:.1}s ({:.1}x realtime)", secs / took);
    println!("{text}");
    Ok(())
}
