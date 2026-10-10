//! Local notes LLM (llama.cpp, Metal on macOS). One `Llm` is owned by a single
//! worker thread (P3); calls never overlap, so no locks are needed here.

use anyhow::{anyhow, bail, Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;
use std::path::Path;

/// n_ctx sizes the pre-allocated KV cache — 32k cost multiple GB and pushed
/// 16 GB machines into swap (system-wide hang + coreaudiod starvation = dead
/// audio). 8k fits every prompt we build once prompts are token-budgeted.
pub const DEFAULT_N_CTX: u32 = 8192;
const N_BATCH: usize = 512;

pub struct Message {
    pub role: &'static str,
    pub content: String,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user", content: content.into() }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system", content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant", content: content.into() }
    }
}

pub struct Llm {
    backend: LlamaBackend,
    model: LlamaModel,
    n_ctx: u32,
    pub name: String,
}

impl Llm {
    pub fn load(path: &Path, n_ctx: u32) -> Result<Self> {
        // Silence llama.cpp *and* ggml/Metal logs (void_logs only covers llama).
        llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default().with_logs_enabled(false));
        let backend = LlamaBackend::init()?;
        // Offload every layer: Metal on Apple Silicon; a no-op on CPU builds.
        let params = LlamaModelParams::default().with_n_gpu_layers(u32::MAX);
        let model = LlamaModel::load_from_file(&backend, path, &params)
            .with_context(|| format!("could not load {}", path.display()))?;
        let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        Ok(Self { backend, model, n_ctx, name })
    }

    pub fn n_ctx(&self) -> u32 {
        self.n_ctx
    }

    /// Token count of plain text — used to budget transcript excerpts so a
    /// prompt plus its answer always fits the context window.
    pub fn count_tokens(&self, text: &str) -> usize {
        self.model.vocab().tokenize(text.as_bytes(), false, false).len()
    }

    /// Turn messages into the model's prompt text using its own chat template.
    fn render(&self, messages: &[Message]) -> Result<String> {
        let template = self.model.chat_template(None).context("model has no chat template")?;
        // Gemma 4's `<|turn>role … <turn|>` format is newer than llama.cpp's
        // built-in template list (which only knows Gemma 1–3), so write it here.
        if template.to_str().is_ok_and(|t| t.contains("<|turn>")) {
            let mut p = String::from("<bos>");
            for m in messages {
                let role = if m.role == "assistant" { "model" } else { m.role };
                p.push_str(&format!("<|turn>{role}\n{}<turn|>\n", m.content.trim()));
            }
            p.push_str("<|turn>model\n");
            return Ok(p);
        }
        let chat = messages
            .iter()
            .map(|m| LlamaChatMessage::new(m.role.to_string(), m.content.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self.model.apply_chat_template(&template, &chat, true)?)
    }

    /// Run a chat completion, calling `on_piece` with each decoded text piece
    /// as it streams. Returns the full text. `on_piece` returning false stops
    /// generation early (the UI closed the stream).
    pub fn chat(
        &self,
        messages: &[Message],
        max_tokens: usize,
        temperature: f32,
        on_piece: &mut dyn FnMut(&str) -> bool,
    ) -> Result<String> {
        let prompt = self.render(messages)?;
        let vocab = self.model.vocab();
        let tokens = vocab.tokenize(prompt.as_bytes(), false, true);

        if tokens.len() + max_tokens > self.n_ctx as usize {
            bail!(
                "prompt is {} tokens; with {} for the answer it exceeds the {}-token context",
                tokens.len(),
                max_tokens,
                self.n_ctx
            );
        }

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.n_ctx))
            .with_n_batch(N_BATCH as u32);
        let mut ctx = self.model.new_context(&self.backend, ctx_params)?;

        // Prefill the prompt in n_batch slices; only the last token needs logits.
        let mut batch = LlamaBatch::new(N_BATCH, 1);
        let last = tokens.len() - 1;
        for (ci, slice) in tokens.chunks(N_BATCH).enumerate() {
            batch.clear();
            for (j, &tok) in slice.iter().enumerate() {
                let pos = ci * N_BATCH + j;
                batch.add(tok, pos as i32, &[0], pos == last)?;
            }
            ctx.decode(&mut batch).map_err(|e| anyhow!("prefill failed: {e}"))?;
        }

        // Same sampling defaults llama-cpp-python used.
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::top_k(40),
            LlamaSampler::top_p(0.95, 1),
            LlamaSampler::min_p(0.05, 1),
            LlamaSampler::temp(temperature),
            LlamaSampler::dist(1234),
        ]);

        let mut out = String::new();
        let mut pending: Vec<u8> = Vec::new(); // bytes of a split multi-byte char
        let mut pos = tokens.len() as i32;
        for _ in 0..max_tokens {
            let tok = sampler.sample(&ctx, batch.n_tokens() - 1);
            if vocab.is_eog(tok) {
                break;
            }
            pending.extend(vocab.token_to_piece(tok, false, None));
            let valid = match std::str::from_utf8(&pending) {
                Ok(s) => s.len(),
                Err(e) => e.valid_up_to(),
            };
            if valid > 0 {
                let piece = String::from_utf8_lossy(&pending[..valid]).into_owned();
                pending.drain(..valid);
                out.push_str(&piece);
                if !on_piece(&piece) {
                    break;
                }
            }
            batch.clear();
            batch.add(tok, pos, &[0], true)?;
            pos += 1;
            ctx.decode(&mut batch).map_err(|e| anyhow!("decode failed: {e}"))?;
        }
        Ok(out)
    }
}
