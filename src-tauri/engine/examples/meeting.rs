//! Speaker-labelled transcript of one recording, or of a mic + computer-audio
//! pair from itsmypa-capture.
//!
//!   cargo run --release -p itsmypa-engine --example meeting -- <audio.wav> [system.wav]

use itsmypa_engine::diarize::{DiarizeConfig, Diarizer};
use itsmypa_engine::{asr, audio, paths, pipeline};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let args: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    let first = args.first().expect("usage: meeting <audio.wav> [system.wav]");

    let models = paths::models_dir();
    let asr = asr::Asr::load(&asr::ModelPaths::new(&models, "small"), "")?;
    let diarizer = Diarizer::load(&models, DiarizeConfig::default())?;

    let t0 = Instant::now();
    let mut progress = |phase, p: f32| eprint!("\r{phase:?} {:3.0}%   ", p * 100.0);
    let transcript = match args.get(1) {
        Some(system) => pipeline::transcribe_dual(
            &asr,
            Some(&diarizer),
            &audio::load_audio_16k(first)?,
            &audio::load_audio_16k(system)?,
            &mut progress,
        )?,
        None => pipeline::transcribe_single(&asr, Some(&diarizer), &audio::load_audio_16k(first)?, &mut progress)?,
    };
    eprintln!("\rdone in {:.1}s            ", t0.elapsed().as_secs_f32());
    println!("{}", transcript.text);
    Ok(())
}
