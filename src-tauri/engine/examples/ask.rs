//! Ask questions across every meeting in a database.
//!
//!   cargo run --release -p itsmypa-engine --example ask -- <meetings.db> "question" …
//!   cargo run --release -p itsmypa-engine --example ask -- --seed <meetings.db> <title|YYYY-MM-DD|transcript.txt> …

use itsmypa_engine::ask::Asker;
use itsmypa_engine::llm::{Llm, DEFAULT_N_CTX};
use itsmypa_engine::store::{Note, Store};
use itsmypa_engine::{models, paths};
use std::io::Write;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--seed") {
        // Build a test database: each spec is "title|date|transcript file".
        let store = Store::open(&PathBuf::from(&args[1]))?;
        for spec in &args[2..] {
            let parts: Vec<&str> = spec.splitn(3, '|').collect();
            let n = store.create(Note { title: parts[0].into(), transcript: std::fs::read_to_string(parts[2])?, ..Default::default() })?;
            // Backdate it (create() stamps "now").
            store.set_date(&n.id, &format!("{}T10:00:00", parts[1]))?;
            println!("seeded {} ({})", parts[0], parts[1]);
        }
        return Ok(());
    }

    let store = Store::open(&PathBuf::from(args.remove(0)))?;
    let models::LlmLocation::Ready(gguf) = models::locate_llm(&paths::models_dir()) else {
        anyhow::bail!("the default notes model isn't downloaded yet");
    };
    let llm = Llm::load(&gguf, DEFAULT_N_CTX)?;
    let asker = Asker::default();
    let today = chrono::Local::now().format("%Y-%m-%d (%A)").to_string();
    for q in args {
        println!("\nQ: {q}");
        print!("A: ");
        let (_, sources) = asker.ask(&llm, &store, &q, &today, &mut |p| {
            print!("{p}");
            std::io::stdout().flush().ok();
            true
        })?;
        let names: Vec<String> = sources.iter().map(|s| format!("{} ({})", s.title, s.date)).collect();
        println!("\n   sources: {}", if names.is_empty() { "—".into() } else { names.join(", ") });
    }
    Ok(())
}
