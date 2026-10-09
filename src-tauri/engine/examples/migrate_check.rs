//! Import a notes.json into a fresh database and export it back out, so the
//! two can be compared field by field.
//!
//!   cargo run -p itsmypa-engine --example migrate_check -- <notes.json> <work_dir>

use itsmypa_engine::store::Store;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let args: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    let (json, dir) = (&args[0], &args[1]);
    let store = Store::open(&dir.join("meetings.db"))?;
    println!("imported {} meetings", store.import_legacy_json(json)?);
    println!("imported again: {} (should be 0)", store.import_legacy_json(json)?);
    std::fs::write(dir.join("export.json"), store.export_json()?)?;
    Ok(())
}
