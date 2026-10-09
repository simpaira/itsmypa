//! Stream structured notes for a transcript with the app's real summary
//! prompt — used to compare candidate models.
//!
//!   cargo run --release -p itsmypa-engine --example summarize -- transcript.txt [model.gguf]

use itsmypa_engine::llm::{Llm, Message, DEFAULT_N_CTX};
use itsmypa_engine::{models, paths, prompts};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let transcript = std::fs::read_to_string(args.next().expect("usage: summarize <transcript.txt> [model.gguf]"))?;
    let gguf = match args.next() {
        Some(p) => PathBuf::from(p),
        None => match models::locate_llm(&paths::models_dir(), models::DEFAULT_LLM) {
            models::LlmLocation::Ready(p) => p,
            models::LlmLocation::NeedsDownload(m) => {
                anyhow::bail!("{} isn't downloaded yet ({:.1} GB)", m.label, m.bytes as f64 / 1e9)
            }
        },
    };

    let t0 = Instant::now();
    let llm = Llm::load(&gguf, DEFAULT_N_CTX)?;
    eprintln!("loaded {} in {:.1}s", llm.name, t0.elapsed().as_secs_f32());

    let prompt = prompts::fill(&prompts::summary_prompt(), &[("TRANSCRIPT", &transcript)]);
    eprintln!("prompt: {} tokens", llm.count_tokens(&prompt));

    let t1 = Instant::now();
    let mut first: Option<f32> = None;
    let mut n = 0usize;
    let out = llm.chat(&[Message::user(prompt)], 1200, 0.15, &mut |piece| {
        first.get_or_insert(t1.elapsed().as_secs_f32());
        n += 1;
        print!("{piece}");
        std::io::stdout().flush().ok();
        true
    })?;
    let took = t1.elapsed().as_secs_f32();
    let first = first.unwrap_or(took);
    eprintln!(
        "\n\nfirst words after {first:.1}s · {} chars · ~{n} tokens in {took:.1}s ({:.1} tok/s)",
        out.len(),
        n as f32 / (took - first).max(0.01)
    );
    Ok(())
}
