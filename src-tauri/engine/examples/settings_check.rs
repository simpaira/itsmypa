//! Import an old .env into a fresh database and show the result
//! (the calendar address is masked — it's a secret link).
//!
//!   cargo run -p itsmypa-engine --example settings_check -- <.env> <work_dir>

use itsmypa_engine::{settings, store::Store};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let args: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    let store = Store::open(&args[1].join("meetings.db"))?;
    settings::import_legacy_env(&store, &args[0])?;
    let mut s = settings::load(&store)?;
    if !s.calendar_ics_url.is_empty() {
        let host = s.calendar_ics_url.split('/').take(3).collect::<Vec<_>>().join("/");
        s.calendar_ics_url = format!("{host}/…(hidden)");
    }
    println!("{s:#?}");
    Ok(())
}
