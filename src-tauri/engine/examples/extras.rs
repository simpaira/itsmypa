//! Title, follow-up email and "enhance my notes" for one meeting.
//!
//!   cargo run --release -p itsmypa-engine --example extras -- transcript.txt "rough note 1" "rough note 2" …

use itsmypa_engine::llm::{Llm, DEFAULT_N_CTX};
use itsmypa_engine::{models, notes, paths};
use std::io::Write;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let transcript = std::fs::read_to_string(args.next().expect("usage: extras <transcript.txt> [note]…"))?;
    let scratchpad = args.map(|n| format!("- {n}")).collect::<Vec<_>>().join("\n");
    let models::LlmLocation::Ready(gguf) = models::locate_llm(&paths::models_dir(), models::DEFAULT_LLM) else {
        anyhow::bail!("the default notes model isn't downloaded yet");
    };
    let llm = Llm::load(&gguf, DEFAULT_N_CTX)?;
    let mut print = |p: &str| {
        print!("{p}");
        std::io::stdout().flush().ok();
        true
    };

    println!("── TITLE ──\n{}\n", notes::title(&llm, &transcript)?);

    println!("── FOLLOW-UP EMAIL ──");
    notes::email(&llm, &transcript, &mut print)?;

    if !scratchpad.is_empty() {
        println!("\n\n── YOUR NOTES ──\n{scratchpad}\n\n── ENHANCED ──");
        notes::enhance(&llm, &transcript, &scratchpad, &mut print)?;
    }
    println!();
    Ok(())
}
