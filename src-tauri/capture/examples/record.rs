//! Manual capture test.
//!
//!   cargo run -p itsmypa-capture --example record -- [--no-mic] [--no-system]
//!       [--secs 15] [--pause 5-8] <out_dir>
//!
//! Prints track status and levels once a second, then the output files.

use itsmypa_capture::{Recording, Sources};
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn main() -> anyhow::Result<()> {
    let mut sources = Sources { mic: true, system: true };
    let mut secs = 15u64;
    let mut pause: Option<(u64, u64)> = None;
    let mut dir = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--no-mic" => sources.mic = false,
            "--no-system" => sources.system = false,
            "--secs" => secs = args.next().unwrap().parse()?,
            "--pause" => {
                let v = args.next().unwrap();
                let (a, b) = v.split_once('-').expect("--pause A-B");
                pause = Some((a.parse()?, b.parse()?));
            }
            flag if flag.starts_with('-') => anyhow::bail!("unknown option {flag}"),
            _ => dir = Some(PathBuf::from(a)),
        }
    }
    let dir = dir.expect("usage: record [options] <out_dir>");

    let rec = Recording::start(&dir, "test", sources)?;
    let t0 = Instant::now();
    let live = rec.wait_live(Duration::from_secs(10));
    println!("live={live} after {:.2}s", t0.elapsed().as_secs_f32());
    for s in 1..=secs {
        std::thread::sleep(Duration::from_secs(1));
        if let Some((a, b)) = pause {
            if s == a { rec.pause(); }
            if s == b { rec.resume(); }
        }
        let lv = rec.levels();
        let st: Vec<String> = rec
            .status()
            .iter()
            .map(|t| format!("{:?}={:?}[{}]{}", t.kind, t.state, t.device,
                t.message.as_deref().map(|m| format!(" ({m})")).unwrap_or_default()))
            .collect();
        println!(
            "{:>4.1}s wall  {:>4.1}s rec{}  mic {:.2}  sys {:.2}  {}",
            t0.elapsed().as_secs_f32(),
            rec.elapsed().as_secs_f32(),
            if rec.is_paused() { " (paused)" } else { "" },
            lv.mic, lv.system, st.join("  ")
        );
    }
    let done = rec.stop()?;
    println!("recorded {:.1}s", done.duration_secs);
    for p in [done.mic, done.system].into_iter().flatten() {
        let r = hound::WavReader::open(&p)?;
        println!("{}  {:.2}s @ {} Hz", p.display(), r.duration() as f32 / r.spec().sample_rate as f32, r.spec().sample_rate);
    }
    Ok(())
}
