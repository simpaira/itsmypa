//! Audio file → mono 16 kHz f32 samples, the format every model consumes.
//! WAV only for now (that's what native capture writes); compressed uploads
//! (mp3/m4a/…) arrive in a later phase via symphonia.

use anyhow::{bail, Context, Result};
use sherpa_onnx::LinearResampler;
use std::path::Path;

use crate::SAMPLE_RATE;

pub fn load_wav_16k(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("could not open {}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;

    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };
    if interleaved.is_empty() {
        bail!("{} contains no audio", path.display());
    }

    let mono: Vec<f32> = if channels == 1 {
        interleaved
    } else {
        interleaved
            .chunks_exact(channels)
            .map(|f| f.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    let rate = spec.sample_rate as i32;
    if rate == SAMPLE_RATE {
        return Ok(mono);
    }
    let rs = LinearResampler::create(rate, SAMPLE_RATE)
        .context("could not create resampler")?;
    Ok(rs.resample(&mono, true))
}
