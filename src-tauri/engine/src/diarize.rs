//! Speaker diarization: pyannote segmentation-3.0 + NeMo TitaNet embeddings
//! (sherpa-onnx), followed by the two repair passes tuned in the Python
//! engine. Port of app.py `_merge_similar_clusters`, `_consolidate_turns` and
//! `_split_spans_at_turns`; constants and calibration notes are unchanged.

use anyhow::{bail, Context, Result};
use sherpa_onnx::{
    FastClusteringConfig, OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
    OfflineSpeakerSegmentationModelConfig, OfflineSpeakerSegmentationPyannoteModelConfig,
    SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig,
};
use std::collections::HashMap;
use std::path::Path;

use crate::asr::num_threads;
use crate::SAMPLE_RATE;

#[derive(Clone, Copy, Debug)]
pub struct DiarizeConfig {
    /// -1 = detect automatically; >0 pins the count (repairs are skipped).
    pub num_speakers: i32,
    /// Clustering distance; lower = more speakers. Do not raise past ~0.85:
    /// verified empirically that 0.9 merges even two very distinct voices.
    /// Cluster over-splitting is repaired by `merge_similar_clusters` instead.
    pub threshold: f32,
    /// Calibration on known audio: same voice ≥ 0.93 cosine similarity,
    /// different voices ≤ 0.26.
    pub merge_sim: f32,
    /// A "speaker" with less total speech than this is clustering noise.
    pub min_speaker_secs: f32,
}

impl Default for DiarizeConfig {
    fn default() -> Self {
        Self { num_speakers: -1, threshold: 0.8, merge_sim: 0.60, min_speaker_secs: 6.0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Turn {
    pub speaker: i32,
    pub start: f32,
    pub end: f32,
}

impl Turn {
    fn dur(&self) -> f32 {
        self.end - self.start
    }
}

pub struct Diarizer {
    diar: OfflineSpeakerDiarization,
    extractor: SpeakerEmbeddingExtractor,
    cfg: DiarizeConfig,
}

fn existing(p: &Path) -> Result<String> {
    if !p.exists() {
        bail!("model file missing: {}", p.display());
    }
    Ok(p.to_string_lossy().into_owned())
}

impl Diarizer {
    pub fn load(models_dir: &Path, cfg: DiarizeConfig) -> Result<Self> {
        let seg = existing(&models_dir.join("sherpa-onnx-pyannote-segmentation-3-0").join("model.onnx"))?;
        let emb = existing(&models_dir.join("nemo_en_titanet_small.onnx"))?;
        let emb_cfg = SpeakerEmbeddingExtractorConfig {
            model: Some(emb),
            num_threads: num_threads(),
            ..Default::default()
        };
        let config = OfflineSpeakerDiarizationConfig {
            segmentation: OfflineSpeakerSegmentationModelConfig {
                pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                    model: Some(seg),
                    ..Default::default()
                },
                num_threads: num_threads(),
                ..Default::default()
            },
            embedding: emb_cfg.clone(),
            clustering: FastClusteringConfig {
                num_clusters: cfg.num_speakers,
                threshold: cfg.threshold,
                ..Default::default()
            },
            // 0.6 so sub-second blips ("Um,", "Yeah.") never become turns —
            // their embeddings are noise that spawns phantom speakers; the
            // text still inherits the neighbor's label.
            min_duration_on: 0.6,
            min_duration_off: 0.5,
        };
        let diar = OfflineSpeakerDiarization::create(&config).context("could not load speaker detection")?;
        let extractor = SpeakerEmbeddingExtractor::create(&emb_cfg).context("could not load speaker embeddings")?;
        Ok(Self { diar, extractor, cfg })
    }

    /// Who spoke when, after both repair passes.
    pub fn turns(&self, samples: &[f32]) -> Vec<Turn> {
        let Some(result) = self.diar.process(samples) else {
            return Vec::new();
        };
        let raw: Vec<Turn> = result
            .sort_by_start_time()
            .into_iter()
            .map(|s| Turn { speaker: s.speaker, start: s.start, end: s.end })
            .collect();
        let merged = self.merge_similar_clusters(raw, samples);
        consolidate_turns(merged, &self.cfg)
    }

    /// Embedding of one cluster from up to 20 s of its longest turns.
    fn cluster_embedding(&self, turns: &[Turn], samples: &[f32]) -> Option<Vec<f32>> {
        let mut sorted: Vec<&Turn> = turns.iter().collect();
        sorted.sort_by(|a, b| b.dur().total_cmp(&a.dur())); // longest first
        let sr = SAMPLE_RATE as f32;
        let (mut audio, mut got) = (Vec::new(), 0.0);
        for t in sorted {
            let a = ((t.start * sr) as usize).min(samples.len());
            let b = ((t.end * sr) as usize).min(samples.len());
            audio.extend_from_slice(&samples[a..b]);
            got += t.dur();
            if got >= 20.0 {
                break;
            }
        }
        if audio.len() < SAMPLE_RATE as usize {
            return None; // under 1 s can't give a stable embedding
        }
        let stream = self.extractor.create_stream()?;
        stream.accept_waveform(SAMPLE_RATE, &audio);
        stream.input_finished();
        let emb = self.extractor.compute(&stream)?;
        let norm = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        (norm > 0.0).then(|| emb.iter().map(|x| x / norm).collect())
    }

    /// The clusterer works on short-turn embeddings, which are noisy — one
    /// person can fragment into several clusters (an interview once yielded 11
    /// "speakers" for 2 people). Re-embed every cluster on up to 20 s of its
    /// own speech and merge clusters whose voices actually match.
    fn merge_similar_clusters(&self, turns: Vec<Turn>, samples: &[f32]) -> Vec<Turn> {
        if self.cfg.num_speakers > 0 || turns.is_empty() {
            return turns;
        }
        let mut by: Vec<(i32, Vec<Turn>)> = Vec::new();
        for t in &turns {
            match by.iter_mut().find(|(sp, _)| *sp == t.speaker) {
                Some((_, v)) => v.push(*t),
                None => by.push((t.speaker, vec![*t])),
            }
        }
        if by.len() < 2 {
            return turns;
        }
        let embs: Vec<Option<Vec<f32>>> = by.iter().map(|(_, ts)| self.cluster_embedding(ts, samples)).collect();

        // Union-find over clusters whose voices match.
        let mut parent: Vec<usize> = (0..by.len()).collect();
        fn find(p: &mut [usize], mut x: usize) -> usize {
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        for i in 0..by.len() {
            for j in i + 1..by.len() {
                if let (Some(a), Some(b)) = (&embs[i], &embs[j]) {
                    let sim: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                    if sim >= self.cfg.merge_sim {
                        let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                        parent[ri] = rj;
                    }
                }
            }
        }
        // The longest-speaking member keeps the label.
        let dur: Vec<f32> = by.iter().map(|(_, ts)| ts.iter().map(Turn::dur).sum()).collect();
        let mut canon: HashMap<usize, usize> = HashMap::new();
        for i in 0..by.len() {
            let root = find(&mut parent, i);
            let best = canon.entry(root).or_insert(i);
            if dur[i] > dur[*best] {
                *best = i;
            }
        }
        let remap: HashMap<i32, i32> = (0..by.len())
            .map(|i| (by[i].0, by[canon[&find(&mut parent, i)]].0))
            .collect();
        turns.into_iter().map(|t| Turn { speaker: remap[&t.speaker], ..t }).collect()
    }
}

/// Fold phantom speakers (tiny total speech) into the temporally nearest real
/// speaker. An interview that clustered into 11 "speakers" had 6 of them
/// speaking under 4 seconds each — noise, not people. Skipped when the user
/// pinned an exact speaker count (clustering already honored it).
pub fn consolidate_turns(turns: Vec<Turn>, cfg: &DiarizeConfig) -> Vec<Turn> {
    if cfg.num_speakers > 0 || turns.is_empty() {
        return turns;
    }
    let mut totals: HashMap<i32, f32> = HashMap::new();
    for t in &turns {
        *totals.entry(t.speaker).or_default() += t.dur();
    }
    let major = |sp: i32| totals[&sp] >= cfg.min_speaker_secs;
    let majors: Vec<Turn> = turns.iter().copied().filter(|t| major(t.speaker)).collect();
    if majors.is_empty() || majors.len() == turns.len() {
        return turns;
    }
    turns
        .iter()
        .map(|t| {
            if major(t.speaker) {
                return *t;
            }
            // nearest real speaker by temporal gap (0 when overlapping)
            let gap = |m: &Turn| (m.start - t.end).max(t.start - m.end).max(0.0);
            let nearest = majors.iter().min_by(|a, b| gap(a).total_cmp(&gap(b))).unwrap();
            Turn { speaker: nearest.speaker, ..*t }
        })
        .collect()
}

/// Cut VAD speech spans at diarization turn boundaries, then label each piece.
///
/// Labeling whole spans by majority overlap merged rapid back-and-forth
/// dialogue (no silence gap → one VAD span) under a single speaker. Cutting at
/// the diarizer's own turn boundaries puts the label changes where the
/// speakers actually change. Pieces shorter than `min_len` are absorbed into
/// their neighbor — isolated sub-second fragments decode terribly.
pub fn split_spans_at_turns(spans: &[(f32, f32)], turns: &[Turn], min_len: f32) -> (Vec<(f32, f32)>, Vec<i32>) {
    let mut out_spans = Vec::new();
    let mut out_speakers: Vec<Option<i32>> = Vec::new();
    for &(s, e) in spans {
        let mut cuts = vec![s, e];
        for t in turns {
            for b in [t.start, t.end] {
                if s + min_len < b && b < e - min_len {
                    cuts.push(b);
                }
            }
        }
        cuts.sort_by(f32::total_cmp);
        cuts.dedup();
        let mut pieces: Vec<(f32, f32, Option<i32>)> = Vec::new();
        for w in cuts.windows(2) {
            let (a, b) = (w[0], w[1]);
            let (mut best, mut best_ov) = (None, 0.0f32);
            for t in turns {
                let ov = b.min(t.end) - a.max(t.start);
                if ov > best_ov {
                    best = Some(t.speaker);
                    best_ov = ov;
                }
            }
            match pieces.last_mut() {
                Some(last) if last.2 == best || (b - a) < min_len => last.1 = b,
                _ => pieces.push((a, b, best)),
            }
        }
        for (a, b, sp) in pieces {
            out_spans.push((a, b));
            out_speakers.push(sp);
        }
    }
    // Pieces no turn covered inherit the previous piece's speaker.
    for i in 0..out_speakers.len() {
        if out_speakers[i].is_none() {
            out_speakers[i] = match i.checked_sub(1).and_then(|p| out_speakers[p]) {
                Some(prev) => Some(prev),
                None => Some(out_speakers[i..].iter().flatten().next().copied().unwrap_or(0)),
            };
        }
    }
    (out_spans, out_speakers.into_iter().map(|s| s.unwrap_or(0)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(speaker: i32, start: f32, end: f32) -> Turn {
        Turn { speaker, start, end }
    }

    #[test]
    fn phantom_speakers_fold_into_nearest_real_one() {
        let cfg = DiarizeConfig::default();
        let turns = vec![t(0, 0.0, 10.0), t(2, 10.5, 11.0), t(1, 20.0, 30.0), t(3, 29.0, 29.5)];
        let out = consolidate_turns(turns, &cfg);
        assert_eq!(out.iter().map(|t| t.speaker).collect::<Vec<_>>(), vec![0, 0, 1, 1]);
    }

    #[test]
    fn pinned_speaker_count_skips_consolidation() {
        let cfg = DiarizeConfig { num_speakers: 4, ..Default::default() };
        let turns = vec![t(0, 0.0, 10.0), t(2, 10.5, 11.0)];
        assert_eq!(consolidate_turns(turns.clone(), &cfg), turns);
    }

    #[test]
    fn rapid_dialogue_in_one_span_is_split_at_turns() {
        // One VAD span 0–10 s, but two speakers alternate inside it.
        let turns = vec![t(0, 0.0, 4.0), t(1, 4.0, 7.0), t(0, 7.0, 10.0)];
        let (spans, speakers) = split_spans_at_turns(&[(0.0, 10.0)], &turns, 0.7);
        assert_eq!(spans, vec![(0.0, 4.0), (4.0, 7.0), (7.0, 10.0)]);
        assert_eq!(speakers, vec![0, 1, 0]);
    }

    #[test]
    fn tiny_pieces_are_absorbed_and_uncovered_inherit() {
        // A 0.3 s blip of speaker 1 is too short to stand alone; the span
        // after the last turn has no coverage and inherits speaker 0.
        let turns = vec![t(0, 0.0, 5.0), t(1, 5.0, 5.3), t(0, 5.3, 8.0)];
        let (_, speakers) = split_spans_at_turns(&[(0.0, 8.0), (9.0, 10.0)], &turns, 0.7);
        assert_eq!(speakers, vec![0, 0]);
    }
}
