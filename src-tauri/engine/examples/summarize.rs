//! P0 smoke test: stream structured notes for a transcript from the local LLM.
//!
//!   cargo run --release -p itsmypa-engine --example summarize -- transcript.txt [model.gguf]

use itsmypa_engine::llm::{Llm, Message, DEFAULT_N_CTX};
use itsmypa_engine::paths;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let transcript = std::fs::read_to_string(args.next().expect("usage: summarize <transcript.txt> [model.gguf]"))?;
    let gguf = match args.next() {
        Some(p) => PathBuf::from(p),
        None => std::fs::read_dir(paths::models_dir())?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "gguf"))
            .min()
            .expect("no .gguf in the models dir"),
    };

    let t0 = Instant::now();
    let llm = Llm::load(&gguf, DEFAULT_N_CTX)?;
    eprintln!("loaded {} in {:.1}s", llm.name, t0.elapsed().as_secs_f32());

    let prompt = format!(
        "You are an expert meeting analyst. Read the transcript below and produce structured notes \
         with the sections: ## Executive Summary, ## Key Decisions, ## Action Items, ## Topics Discussed.\
         \n\n---\nTRANSCRIPT:\n{transcript}\n---"
    );
    eprintln!("prompt: {} tokens", llm.count_tokens(&prompt));

    let t1 = Instant::now();
    let mut n = 0usize;
    let out = llm.chat(&[Message::user(prompt)], 600, 0.15, &mut |piece| {
        n += 1;
        print!("{piece}");
        std::io::stdout().flush().ok();
        true
    })?;
    let took = t1.elapsed().as_secs_f32();
    eprintln!("\n\n{} chars, ~{n} tokens in {took:.1}s ({:.1} tok/s)", out.len(), n as f32 / took);
    Ok(())
}
