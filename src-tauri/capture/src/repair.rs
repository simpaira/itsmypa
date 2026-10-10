//! Crash recovery for capture WAVs.
//!
//! Writers rewrite the WAV header every 2 s, so after a crash (or force-quit)
//! the header can undercount up to ~2 s of audio that *is* on disk. This
//! rewrites the RIFF and `data` sizes from the real file length so recovery
//! keeps every sample that reached the disk.

use anyhow::{bail, Context, Result};
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Fix the size fields of a PCM WAV whose writer never finalized it.
/// Returns the number of audio bytes the repaired header now declares.
pub fn repair_wav(path: &Path) -> Result<u64> {
    let mut f = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .with_context(|| format!("could not open {}", path.display()))?;
    let len = f.metadata()?.len();

    let mut riff = [0u8; 12];
    f.read_exact(&mut riff).context("file too short for a WAV header")?;
    if &riff[0..4] != b"RIFF" || &riff[8..12] != b"WAVE" {
        bail!("{} is not a WAV file", path.display());
    }

    // Walk chunks until `data`; everything after its header is audio.
    let mut pos = 12u64;
    let data_start = loop {
        let mut hdr = [0u8; 8];
        f.seek(SeekFrom::Start(pos))?;
        if f.read_exact(&mut hdr).is_err() {
            bail!("no data chunk in {}", path.display());
        }
        if &hdr[0..4] == b"data" {
            break pos + 8;
        }
        let size = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as u64;
        pos += 8 + size + (size & 1); // chunks are word-aligned
    };

    // Whole 16-bit mono frames only; a torn final sample is dropped.
    let data_len = (len.saturating_sub(data_start)) & !1;
    f.seek(SeekFrom::Start(4))?;
    f.write_all(&((data_start - 8 + data_len) as u32).to_le_bytes())?;
    f.seek(SeekFrom::Start(data_start - 4))?;
    f.write_all(&(data_len as u32).to_le_bytes())?;
    f.set_len(data_start + data_len)?;
    Ok(data_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repairs_header_left_stale_by_a_crash() {
        let path = std::env::temp_dir().join(format!("itsmypa-repair-{}.wav", std::process::id()));
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        for i in 0..1_000 {
            w.write_sample(i as i16).unwrap();
        }
        w.flush().unwrap(); // header now says 1000 samples
        w.finalize().unwrap();

        // Simulate audio written after the last header update, plus a torn byte.
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        for i in 1_000..1_600i16 {
            f.write_all(&i.to_le_bytes()).unwrap();
        }
        f.write_all(&[7]).unwrap();
        drop(f);
        assert_eq!(hound::WavReader::open(&path).unwrap().duration(), 1_000);

        assert_eq!(repair_wav(&path).unwrap(), 1_600 * 2);
        let mut r = hound::WavReader::open(&path).unwrap();
        assert_eq!(r.duration(), 1_600);
        let last: i16 = r.samples::<i16>().last().unwrap().unwrap();
        assert_eq!(last, 1_599);
        std::fs::remove_file(&path).ok();
    }
}
