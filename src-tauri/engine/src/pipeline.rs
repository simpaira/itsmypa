//! Transcription pipeline, ported from app.py `_run_transcription` and
//! `_run_dual_channel`. Diarization runs first so each Whisper window holds a
//! single speaker's speech with maximum context.

use crate::asr::Asr;
use crate::diarize::{split_spans_at_turns, Diarizer};
use crate::SAMPLE_RATE;
use anyhow::{bail, Result};

/// Whisper's context window is 30 s; decode windows this size give it full
/// sentence context (decoding isolated 1-2 s fragments wrecks accuracy).
pub const MAX_CHUNK_SEC: f32 = 28.0;

const NO_SPEECH: &str = "No speech detected. Try speaking closer to the mic or record longer.";

pub struct Chunk<'a, S> {
    pub speaker: S,
    pub start: f32,
    pub end: f32,
    pub samples: &'a [f32],
}

/// Merge consecutive same-speaker VAD spans into ≤28 s decode windows.
/// Port of `_group_spans`.
pub fn group_spans<'a, S: Copy + PartialEq>(
    spans: &[(f32, f32)],
    speakers: &[S],
    samples: &'a [f32],
) -> Vec<Chunk<'a, S>> {
    let mut groups: Vec<(S, f32, f32)> = Vec::new();
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

#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub speaker: String,
    pub text: String,
    pub start: f32,
    pub end: f32,
}

#[derive(Clone, Debug)]
pub struct Transcript {
    /// "[Speaker 1 00:04] text" blocks, or plain text without diarization.
    pub text: String,
    /// None when speakers couldn't be told apart (diarization off).
    pub segments: Option<Vec<Segment>>,
}

/// Where the pipeline is, for the progress bar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Diarizing,
    Transcribing,
}

pub type OnProgress<'a> = &'a mut dyn FnMut(Phase, f32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Who {
    You,
    Them,
    Cluster(i32),
}

pub fn fmt_time(s: f32) -> String {
    format!("{:02}:{:02}", (s / 60.0) as u32, (s % 60.0) as u32)
}

/// Decode every chunk, keeping the non-empty ones as (who, start, end, text).
fn decode_all<S: Copy>(asr: &Asr, chunks: &[Chunk<S>], base: f32, progress: OnProgress) -> Vec<(S, f32, f32, String)> {
    let mut parts = Vec::new();
    for (i, c) in chunks.iter().enumerate() {
        let text = asr.decode(c.samples);
        if !text.is_empty() {
            parts.push((c.speaker, c.start, c.end, text));
        }
        progress(Phase::Transcribing, base + (1.0 - base) * (i + 1) as f32 / chunks.len() as f32);
    }
    parts
}

/// Label, then merge consecutive same-speaker parts less than 2 s apart.
fn finalize(parts: Vec<(Who, f32, f32, String)>, first_cluster_number: usize) -> Transcript {
    let mut mapping: Vec<i32> = Vec::new();
    let mut merged: Vec<Segment> = Vec::new();
    for (who, start, end, text) in parts {
        let speaker = match who {
            Who::You => "You".to_string(),
            Who::Them => "Them".to_string(),
            Who::Cluster(id) => {
                let n = mapping.iter().position(|&m| m == id).unwrap_or_else(|| {
                    mapping.push(id);
                    mapping.len() - 1
                });
                format!("Speaker {}", n + first_cluster_number)
            }
        };
        match merged.last_mut() {
            Some(last) if last.speaker == speaker && start - last.end < 2.0 => {
                last.text.push(' ');
                last.text.push_str(&text);
                last.end = end;
            }
            _ => merged.push(Segment { speaker, text, start, end }),
        }
    }
    let text = merged
        .iter()
        .map(|s| format!("[{} {}] {}", s.speaker, fmt_time(s.start), s.text))
        .collect::<Vec<_>>()
        .join("\n\n");
    Transcript { text, segments: Some(merged) }
}

/// One audio channel (an uploaded file, or a mic-only recording):
/// VAD → diarize → Whisper over speaker-grouped windows → merge.
pub fn transcribe_single(asr: &Asr, diarizer: Option<&Diarizer>, samples: &[f32], progress: OnProgress) -> Result<Transcript> {
    let spans = asr.vad_split(samples)?;
    if spans.is_empty() {
        bail!(NO_SPEECH);
    }

    let Some(diarizer) = diarizer else {
        let chunks = group_spans(&spans, &vec![(); spans.len()], samples);
        let parts = decode_all(asr, &chunks, 0.0, progress);
        if parts.is_empty() {
            bail!(NO_SPEECH);
        }
        let text = parts.into_iter().map(|p| p.3).collect::<Vec<_>>().join(" ");
        return Ok(Transcript { text, segments: None });
    };

    progress(Phase::Diarizing, 0.0);
    let turns = diarizer.turns(samples);
    let (spans, speakers) = if turns.is_empty() {
        (spans.clone(), vec![0; spans.len()])
    } else {
        // Cut spans where the diarizer says the speaker changes — rapid
        // dialogue with no silence gap must not collapse into one label.
        split_spans_at_turns(&spans, &turns, 0.7)
    };
    let who: Vec<Who> = speakers.into_iter().map(Who::Cluster).collect();
    let chunks = group_spans(&spans, &who, samples);
    let parts = decode_all(asr, &chunks, 0.35, progress);
    if parts.is_empty() {
        bail!(NO_SPEECH);
    }
    Ok(finalize(parts, 1))
}

/// Two channels from native capture: the mic IS the user, so it's labeled
/// "You" with no ML at all; diarization only has to separate the remote
/// participants on the computer-audio channel. Both tracks share one clock
/// (itsmypa-capture), so their timestamps line up with no offset.
pub fn transcribe_dual(
    asr: &Asr,
    diarizer: Option<&Diarizer>,
    mic: &[f32],
    system: &[f32],
    progress: OnProgress,
) -> Result<Transcript> {
    let mic_spans = asr.vad_split(mic)?;
    let sys_spans = asr.vad_split(system)?;
    if mic_spans.is_empty() && sys_spans.is_empty() {
        bail!(NO_SPEECH);
    }

    // Adaptive: if little or nothing came through the computer audio (an
    // in-person meeting, or a call where the remote side stayed quiet), the
    // "mic = You" assumption is wrong — the mic has everyone. Fall back to
    // single-channel diarization so in-person meetings get real labels.
    let sys_speech: f32 = sys_spans.iter().map(|(s, e)| e - s).sum();
    if sys_speech < 2.0 {
        return transcribe_single(asr, diarizer, mic, progress);
    }

    let (mut sys_spans, mut sys_who) = (sys_spans.clone(), vec![Who::Them; sys_spans.len()]);
    let mut base = 0.0;
    if let Some(d) = diarizer {
        progress(Phase::Diarizing, 0.0);
        let turns = d.turns(system);
        if !turns.is_empty() {
            let (spans, speakers) = split_spans_at_turns(&sys_spans, &turns, 0.7);
            sys_spans = spans;
            sys_who = speakers.into_iter().map(Who::Cluster).collect();
        }
        base = 0.3;
    }

    let mut chunks = group_spans(&mic_spans, &vec![Who::You; mic_spans.len()], mic);
    chunks.extend(group_spans(&sys_spans, &sys_who, system));
    chunks.sort_by(|a, b| a.start.total_cmp(&b.start));

    let mut parts = decode_all(asr, &chunks, base, progress);
    if parts.is_empty() {
        bail!(NO_SPEECH);
    }
    parts.sort_by(|a, b| a.1.total_cmp(&b.1));
    // Remote speakers become Speaker 2, 3, … ("You" is speaker 1).
    Ok(finalize(parts, 2))
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
        let speakers = [0, 0, 1, 1, 1];
        let chunks = group_spans(&spans, &speakers, &samples);

        assert_eq!(chunks[0].speaker, 0);
        assert_eq!(chunks[1].speaker, 1);
        for c in &chunks {
            assert!(c.end - c.start <= MAX_CHUNK_SEC + 1e-6);
        }
        let c = &chunks[0];
        assert_eq!(c.samples.len(), (c.end * sr as f32) as usize - (c.start * sr as f32) as usize);
    }

    #[test]
    fn finalize_labels_and_merges_close_same_speaker_parts() {
        let parts = vec![
            (Who::You, 0.0, 2.0, "hello".to_string()),
            (Who::Cluster(7), 2.5, 4.0, "hi".to_string()),
            (Who::Cluster(7), 4.5, 6.0, "there".to_string()),
            (Who::Cluster(3), 70.0, 72.0, "bye".to_string()),
        ];
        let t = finalize(parts, 2);
        let segs = t.segments.unwrap();
        assert_eq!(
            segs.iter().map(|s| (s.speaker.as_str(), s.text.as_str())).collect::<Vec<_>>(),
            vec![("You", "hello"), ("Speaker 2", "hi there"), ("Speaker 3", "bye")]
        );
        assert!(t.text.ends_with("[Speaker 3 01:10] bye"));
    }
}
