//! Ask questions about one meeting, as a conversation.
//!
//!   cargo run --release -p itsmypa-engine --example chat -- transcript.txt "question 1" "question 2" …

use itsmypa_engine::chat::{ask_meeting, Turn};
use itsmypa_engine::llm::{Llm, DEFAULT_N_CTX};
use itsmypa_engine::{models, paths};
use std::io::Write;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let transcript = std::fs::read_to_string(args.next().expect("usage: chat <transcript.txt> <question>…"))?;
    let models::LlmLocation::Ready(gguf) = models::locate_llm(&paths::models_dir()) else {
        anyhow::bail!("the default notes model isn't downloaded yet");
    };
    let llm = Llm::load(&gguf, DEFAULT_N_CTX)?;

    let mut history: Vec<Turn> = Vec::new();
    for question in args {
        println!("\nQ: {question}");
        print!("A: ");
        let t0 = Instant::now();
        let answer = ask_meeting(&llm, &transcript, &question, &history, &mut |p| {
            print!("{p}");
            std::io::stdout().flush().ok();
            true
        })?;
        println!("   ({:.1}s)", t0.elapsed().as_secs_f32());
        history.push(Turn { from_user: true, text: question });
        history.push(Turn { from_user: false, text: answer });
    }
    Ok(())
}
