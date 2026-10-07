//! Whisper (int8, sherpa-onnx) + Silero VAD — the same model files the Python
//! engine downloaded, so existing installs need no re-download.

use anyhow::{bail, Context, Result};
use sherpa_onnx::{
    OfflineModelConfig, OfflineRecognizer, OfflineRecognizerConfig, OfflineWhisperModelConfig,
    SileroVadModelConfig, VadModelConfig, VoiceActivityDetector,
};
use std::path::{Path, PathBuf};

use crate::SAMPLE_RATE;

pub struct ModelPaths {
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub tokens: PathBuf,
    pub vad: PathBuf,
}

impl ModelPaths {
    /// Layout of the sherpa-onnx release archives under `models/`.
    pub fn new(models_dir: &Path, whisper_model: &str) -> Self {
        let asr = models_dir.join(format!("sherpa-onnx-whisper-{whisper_model}"));
        Self {
            encoder: asr.join(format!("{whisper_model}-encoder.int8.onnx")),
            decoder: asr.join(format!("{whisper_model}-decoder.int8.onnx")),
            tokens: asr.join(format!("{whisper_model}-tokens.txt")),
            vad: models_dir.join("silero_vad.onnx"),
        }
    }
}

fn path_str(p: &Path) -> Result<String> {
    if !p.exists() {
        bail!("model file missing: {}", p.display());
    }
    Ok(p.to_string_lossy().into_owned())
}

pub fn num_threads() -> i32 {
    let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    (cpus / 2).max(2) as i32
}

pub struct Asr {
    recognizer: OfflineRecognizer,
    vad_model: String,
}

impl Asr {
    /// `language` empty = auto-detect.
    pub fn load(paths: &ModelPaths, language: &str) -> Result<Self> {
        let config = OfflineRecognizerConfig {
            model_config: OfflineModelConfig {
                whisper: OfflineWhisperModelConfig {
                    encoder: Some(path_str(&paths.encoder)?),
                    decoder: Some(path_str(&paths.decoder)?),
                    language: Some(language.to_string()),
                    task: Some("transcribe".into()),
                    tail_paddings: -1,
                    ..Default::default()
                },
                tokens: Some(path_str(&paths.tokens)?),
                num_threads: num_threads(),
                ..Default::default()
            },
            ..Default::default()
        };
        let recognizer =
            OfflineRecognizer::create(&config).context("could not load the Whisper model")?;
        Ok(Self { recognizer, vad_model: path_str(&paths.vad)? })
    }

    /// Decode one ≤30 s window.
    pub fn decode(&self, chunk: &[f32]) -> String {
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(SAMPLE_RATE, chunk);
        self.recognizer.decode(&stream);
        stream.get_result().map(|r| r.text.trim().to_string()).unwrap_or_default()
    }

    /// Detect speech regions: [(start_sec, end_sec), …]. Port of `_vad_split`.
    pub fn vad_split(&self, samples: &[f32]) -> Result<Vec<(f32, f32)>> {
        let config = VadModelConfig {
            silero_vad: SileroVadModelConfig {
                model: Some(self.vad_model.clone()),
                threshold: 0.5,
                min_silence_duration: 0.5,
                min_speech_duration: 0.25,
                max_speech_duration: 20.0,
                window_size: 512,
            },
            sample_rate: SAMPLE_RATE,
            num_threads: 1,
            ..Default::default()
        };
        let vad = VoiceActivityDetector::create(&config, 60.0)
            .context("could not load the voice detector")?;

        let sr = SAMPLE_RATE as f32;
        let mut out = Vec::new();
        let mut drain = |vad: &VoiceActivityDetector| {
            while !vad.is_empty() {
                if let Some(seg) = vad.front() {
                    let start = seg.start() as f32 / sr;
                    out.push((start, start + seg.n() as f32 / sr));
                }
                vad.pop();
            }
        };
        for window in samples.chunks(512) {
            vad.accept_waveform(window);
            drain(&vad);
        }
        vad.flush();
        drain(&vad);
        Ok(out)
    }
}
