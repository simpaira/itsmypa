//! Transcription pipeline, ported from app.py. P0 covers the speaker-agnostic
//! path (VAD → ≤28 s windows → Whisper); diarization and the dual-channel
//! merge land in P2.

use crate::asr::Asr;
use crate::SAMPLE_RATE;
use anyhow::{bail, Result};

/// Whisper's context window is 30 s; decode windows this size give it full
/// sentence context (decoding isolated 1-2 s fragments wrecks accuracy).
pub const MAX_CHUNK_SEC: f32 = 28.0;

pub struct Chunk<'a> {
    pub speaker: Option<i32>,
    pub start: f32,
    pub end: f32,
    pub samples: &'a [f32],
}

/// Merge consecutive same-speaker VAD spans into ≤28 s decode windows.
/// Port of `_group_spans`.
pub fn group_spans<'a>(
    spans: &[(f32, f32)],
    speakers: &[Option<i32>],
    samples: &'a [f32],
) -> Vec<Chunk<'a>> {
    let mut groups: Vec<(Option<i32>, f32, f32)> = Vec::new();
    for (&(s, e), &sp) in spans.iter().zip(speakers) {
        match groups.last_mut() {
            Some(g) if g.0 == sp && s - g.2 <= 3.0 && e - g.1 <= MAX_CHUNK_SEC => g.2 = e,
            _ => groups.push((sp, s, e)),
        }
    }
    let sr = SAMPLE_RATE as f32;
    groups
        .into_iter()
        .map(|(speaker, start, end)| {
            let a = ((start * sr) as usize).min(samples.len());
            let b = ((end * sr) as usize).min(samples.len());
            Chunk { speaker, start, end, samples: &samples[a..b] }
        })
        .collect()
}

/// Single-channel transcription without diarization.
pub fn transcribe_plain(asr: &Asr, samples: &[f32], progress: &mut dyn FnMut(f32)) -> Result<String> {
    let spans = asr.vad_split(samples)?;
    if spans.is_empty() {
        bail!("No speech detected. Try speaking closer to the mic or record longer.");
    }
    let speakers = vec![None; spans.len()];
    let chunks = group_spans(&spans, &speakers, samples);
    let mut parts = Vec::new();
    for (i, c) in chunks.iter().enumerate() {
        let text = asr.decode(c.samples);
        if !text.is_empty() {
            parts.push(text);
        }
        progress((i + 1) as f32 / chunks.len() as f32);
    }
    if parts.is_empty() {
        bail!("No speech detected. Try speaking closer to the mic or record longer.");
    }
    Ok(parts.join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Port of tests/test_app.py::test_group_spans_merges_same_speaker_and_respects_cap
    #[test]
    fn group_spans_merges_same_speaker_and_respects_cap() {
        let sr = SAMPLE_RATE as usize;
        let samples = vec![0f32; sr * 100];
        let spans = [(0.0, 5.0), (5.5, 10.0), (11.0, 15.0), (16.0, 40.0), (40.5, 50.0)];
        let speakers = [Some(0), Some(0), Some(1), Some(1), Some(1)];
        let chunks = group_spans(&spans, &speakers, &samples);

        assert_eq!(chunks[0].speaker, Some(0));
        assert_eq!(chunks[1].speaker, Some(1));
        for c in &chunks {
            assert!(c.end - c.start <= MAX_CHUNK_SEC + 1e-6);
        }
        let c = &chunks[0];
        assert_eq!(
            c.samples.len(),
            (c.end * sr as f32) as usize - (c.start * sr as f32) as usize
        );
    }
}
