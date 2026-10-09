//! ItsMyPA engine — everything the app does on-device, with no server and no
//! Python: audio decoding, Whisper transcription (sherpa-onnx), and the local
//! notes LLM (llama.cpp). The Tauri shell calls into this crate directly.

pub mod asr;
pub mod audio;
pub mod diarize;
pub mod llm;
pub mod paths;
pub mod pipeline;

pub const SAMPLE_RATE: i32 = 16_000;
