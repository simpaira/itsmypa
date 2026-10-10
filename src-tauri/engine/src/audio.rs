//! Audio file → mono 16 kHz f32 samples, the format every model consumes.
//! WAV (what native capture writes) is read directly; uploaded recordings in
//! other formats (mp3, m4a/aac, flac, ogg/vorbis) are decoded
//! with symphonia, in pure Rust — no ffmpeg to ship.

use anyhow::{bail, Context, Result};
use sherpa_onnx::LinearResampler;
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::SAMPLE_RATE;

/// Any supported audio file → mono 16 kHz samples.
pub fn load_audio_16k(path: &Path) -> Result<Vec<f32>> {
    let is_wav = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("wav"));
    if is_wav {
        if let Ok(samples) = load_wav_16k(path) {
            return Ok(samples);
        }
        // Not plain PCM (e.g. a compressed WAV): let symphonia try.
    }
    let (mono, rate) = decode(path)?;
    to_16k(mono, rate)
}

/// Decode with symphonia → (mono samples, sample rate).
fn decode(path: &Path) -> Result<(Vec<f32>, u32)> {
    let unsupported = || {
        format!(
            "Couldn't read {} — supported formats are WAV, MP3, M4A/AAC, FLAC and OGG.",
            path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
        )
    };
    let file = std::fs::File::open(path).with_context(|| format!("could not open {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .with_context(unsupported)?;
    let mut format = probed.format;
    let track = format.default_track().with_context(unsupported)?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .with_context(unsupported)?;

    let mut mono = Vec::new();
    let mut rate = track.codec_params.sample_rate.unwrap_or(0);
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(e).with_context(unsupported),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SymphoniaError::DecodeError(_)) => continue, // skip a corrupt frame
            Err(e) => return Err(e).with_context(unsupported),
        };
        let spec = *decoded.spec();
        rate = spec.rate;
        let channels = spec.channels.count().max(1);
        let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buf.copy_interleaved_ref(decoded);
        mono.extend(buf.samples().chunks_exact(channels).map(|f| f.iter().sum::<f32>() / channels as f32));
    }
    if mono.is_empty() || rate == 0 {
        bail!("{} contains no audio", path.display());
    }
    Ok((mono, rate))
}

fn to_16k(mono: Vec<f32>, rate: u32) -> Result<Vec<f32>> {
    if rate as i32 == SAMPLE_RATE {
        return Ok(mono);
    }
    let rs = LinearResampler::create(rate as i32, SAMPLE_RATE).context("could not create resampler")?;
    Ok(rs.resample(&mono, true))
}

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

    to_16k(mono, spec.sample_rate)
}
