//! Which model files the app uses, and downloading them safely.
//!
//! Every download is pinned to an exact upstream revision and verified by
//! size and (where we have it) SHA-256, so a mirror serving an error page, a
//! truncated file, or a tampered model can never be loaded. Interrupted
//! downloads resume. `.tar.bz2` archives are unpacked after verifying.

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub struct ModelFile {
    pub id: &'static str,
    pub label: &'static str,
    /// Name of the downloaded file in the models dir.
    pub file: &'static str,
    pub urls: &'static [&'static str],
    /// None for the large optional Whisper archives, which upstream publishes
    /// without a digest; those are checked by exact size and by unpacking.
    pub sha256: Option<&'static str>,
    pub bytes: u64,
    /// For a .tar.bz2 archive: the file (relative to the models dir) that
    /// exists once it is unpacked. None for plain files.
    pub unpacked: Option<&'static str>,
    /// Files from the archive the app never uses, deleted after unpacking
    /// (relative to the models dir).
    pub unused: &'static [&'static str],
}

impl ModelFile {
    /// Where the model lives once installed.
    pub fn path(&self, dir: &Path) -> PathBuf {
        dir.join(self.unpacked.unwrap_or(self.file))
    }

    pub fn is_installed(&self, dir: &Path) -> bool {
        self.path(dir).exists()
    }

    /// Bytes of an earlier, interrupted download that will be resumed.
    pub fn partial_bytes(&self, dir: &Path) -> u64 {
        fs::metadata(self.part_path(dir)).map(|m| m.len().min(self.bytes)).unwrap_or(0)
    }

    /// Free disk space needed to install this model from where it is now:
    /// what's left to download, plus room to unpack an archive (briefly about
    /// twice its size, until the archive and unused files are removed), plus
    /// a little headroom.
    pub fn space_needed(&self, dir: &Path) -> u64 {
        let remaining = self.bytes - self.partial_bytes(dir);
        let unpack = if self.unpacked.is_some() { self.bytes * 2 } else { 0 };
        remaining + unpack + SPACE_HEADROOM
    }

    fn part_path(&self, dir: &Path) -> PathBuf {
        dir.join(format!("{}.part", self.file))
    }

    /// Delete archive files the app never uses. Only runs once the model is
    /// installed, so it can't remove anything needed. Returns bytes freed.
    pub fn remove_unused(&self, dir: &Path) -> u64 {
        if !self.is_installed(dir) {
            return 0;
        }
        self.unused
            .iter()
            .map(|f| dir.join(f))
            .filter_map(|p| {
                let len = fs::metadata(&p).ok()?.len();
                fs::remove_file(&p).ok().map(|_| len)
            })
            .sum()
    }
}

pub const VAD: ModelFile = ModelFile {
    id: "vad",
    label: "Voice detector",
    file: "silero_vad.onnx",
    urls: &[concat!("https://github.com/k2-fsa/sherpa-onnx/releases/download", "/asr-models/silero_vad.onnx")],
    sha256: Some("9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6"),
    bytes: 643_854,
    unpacked: None,
    unused: &[],
};

pub const SPEAKER_SEGMENTATION: ModelFile = ModelFile {
    id: "seg",
    label: "Speaker detection",
    file: "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    urls: &[concat!("https://github.com/k2-fsa/sherpa-onnx/releases/download", "/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2")],
    sha256: Some("24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488"),
    bytes: 6_958_444,
    unpacked: Some("sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
    unused: &[],
};

pub const SPEAKER_EMBEDDING: ModelFile = ModelFile {
    id: "emb",
    label: "Speaker identification",
    file: "nemo_en_titanet_small.onnx",
    // NB: "recongition" is a real typo in the upstream release tag — don't fix it.
    urls: &[concat!("https://github.com/k2-fsa/sherpa-onnx/releases/download", "/speaker-recongition-models/nemo_en_titanet_small.onnx")],
    sha256: Some("ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e"),
    bytes: 40_257_283,
    unpacked: None,
    unused: &[],
};

/// Speech recognition: Whisper "small" (int8). The best balance for an
/// average computer — about 6x faster than realtime on Apple Silicon, fine on
/// older laptops, ~100 languages. The app uses this one model only.
pub const WHISPER: ModelFile = ModelFile {
    id: "asr",
    label: "Speech recognition",
    file: "sherpa-onnx-whisper-small.tar.bz2",
    urls: &["https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-small.tar.bz2"],
    sha256: Some("486a46afbb7ba798507190ffe02fea2dd726049af212e774537efac6afb210a6"),
    bytes: 639_387_718,
    unpacked: Some("sherpa-onnx-whisper-small/small-encoder.int8.onnx"),
    // The archive also ships full-precision copies (~1 GB); the app only
    // loads the int8 ones.
    unused: &["sherpa-onnx-whisper-small/small-encoder.onnx", "sherpa-onnx-whisper-small/small-decoder.onnx"],
};

/// Meeting notes AI: Qwen2.5 3B (Q4_K_M). Small first download (1.9 GB),
/// fast, and runs well even on 8 GB machines. The app uses this one model
/// only (developers can point `$LLM_GGUF` at another GGUF to experiment).
pub const NOTES_LLM: ModelFile = ModelFile {
    id: "llm",
    label: "Meeting notes AI",
    file: "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    // HF's CDN rate-limits anonymous large downloads per IP (403), so fall
    // back to ModelScope, which mirrors the identical bartowski file.
    urls: &[
        "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/f302c64a2269a69fb27b2f9473b362f5bb8e78d8/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "https://modelscope.cn/api/v1/models/bartowski/Qwen2.5-3B-Instruct-GGUF/repo?Revision=master&FilePath=Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    ],
    sha256: Some("9c9f56a391a3abbd5b89d0245bf6106081bcc3173119d4229235dd9d23253f94"),
    bytes: 1_929_903_264,
    unpacked: None,
    unused: &[],
};

pub enum LlmLocation {
    Ready(PathBuf),
    NeedsDownload(&'static ModelFile),
}

/// Where the notes model comes from: `$LLM_GGUF` (developers) wins; then
/// the notes model if it's on disk; otherwise it must be downloaded.
pub fn locate_llm(models_dir: &Path) -> LlmLocation {
    if let Some(p) = std::env::var_os("LLM_GGUF") {
        return LlmLocation::Ready(PathBuf::from(p));
    }
    let path = NOTES_LLM.path(models_dir);
    if path.exists() {
        LlmLocation::Ready(path)
    } else {
        LlmLocation::NeedsDownload(&NOTES_LLM)
    }
}

/// Kept free beyond what a model needs, so setup never fills the disk.
const SPACE_HEADROOM: u64 = 300_000_000;

/// Fails with a plain message when `free` bytes can't fit `needed`.
fn ensure_space(needed: u64, free: u64) -> Result<()> {
    if free < needed {
        let gb = |b: u64| (b as f64 / 1e9 * 10.0).ceil() / 10.0;
        bail!(
            "Not enough disk space: ItsMyPA needs about {:.1} GB free to finish setting up, and this Mac has {:.1} GB. Free up some space, then try again.",
            gb(needed),
            (free as f64 / 1e8).floor() / 10.0
        );
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Downloading,
    /// Checking the SHA-256 (a few seconds for the big files).
    Verifying,
    Unpacking,
}

/// Download `model` into `dir` (resuming a previous partial download),
/// verify it, and unpack it if it's an archive. Returns the installed path.
/// `progress` receives (phase, bytes_done, bytes_total).
pub fn download(model: &ModelFile, dir: &Path, progress: &mut dyn FnMut(Phase, u64, u64)) -> Result<PathBuf> {
    fs::create_dir_all(dir)?;
    if model.is_installed(dir) {
        return Ok(model.path(dir));
    }
    // Fail before downloading anything, not halfway through with a disk error.
    if let Ok(free) = fs4::available_space(dir) {
        ensure_space(model.space_needed(dir), free)?;
    }
    let part = model.part_path(dir);
    let mut last_err = None;
    for url in model.urls {
        // A dropped connection resumes automatically a few times before the
        // user is asked to "Try again".
        let mut got = Err(anyhow::anyhow!("not started"));
        for attempt in 1..=DOWNLOAD_ATTEMPTS {
            got = fetch(url, &part, model.bytes, &mut |d, t| progress(Phase::Downloading, d, t));
            if got.is_ok() || attempt == DOWNLOAD_ATTEMPTS {
                break;
            }
            std::thread::sleep(std::time::Duration::from_secs(2 * attempt as u64));
        }
        let got = got.and_then(|_| {
            progress(Phase::Verifying, model.bytes, model.bytes);
            model.sha256.map_or(Ok(()), |sha| verify(&part, sha))
        });
        match got {
            Ok(()) => {
                if model.unpacked.is_some() {
                    progress(Phase::Unpacking, model.bytes, model.bytes);
                    let unpacked = unpack(&part, dir);
                    fs::remove_file(&part).ok();
                    unpacked?;
                    model.remove_unused(dir);
                    if !model.is_installed(dir) {
                        bail!("{} unpacked without {}", model.file, model.unpacked.unwrap_or_default());
                    }
                } else {
                    fs::rename(&part, model.path(dir))?;
                }
                return Ok(model.path(dir));
            }
            Err(e) => {
                // A bad file must not poison the next attempt.
                if e.to_string().contains("checksum") {
                    fs::remove_file(&part).ok();
                }
                last_err = Some(e.context(format!("download from {} failed", host(url))));
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no download URL for {}", model.file)))
}

/// Unpack a .tar.bz2 into `dir`, refusing entries that would land outside it.
fn unpack(archive: &Path, dir: &Path) -> Result<()> {
    let mut tar = tar::Archive::new(bzip2::read::BzDecoder::new(std::io::BufReader::new(File::open(archive)?)));
    for entry in tar.entries()? {
        let mut entry = entry?;
        // unpack_in skips absolute paths and "..", so a crafted archive
        // can't write outside the models dir.
        entry.unpack_in(dir).context("could not unpack the model")?;
    }
    Ok(())
}

fn host(url: &str) -> &str {
    url.split('/').nth(2).unwrap_or(url)
}

/// Tries per mirror before giving up (each one resumes where the last stopped).
const DOWNLOAD_ATTEMPTS: u32 = 4;

fn fetch(url: &str, part: &Path, total: u64, progress: &mut dyn FnMut(u64, u64)) -> Result<()> {
    let have = fs::metadata(part).map(|m| m.len()).unwrap_or(0);
    if have == total {
        return Ok(()); // fully downloaded earlier; just needs verifying
    }
    // Give up on a *stalled* connection (nothing for 60 s), never on a long
    // download: an overall timeout would cut every big model off after a
    // minute on a normal home connection.
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(30))
        .timeout_read(std::time::Duration::from_secs(60))
        .build();
    let mut req = agent.get(url);
    if have > 0 {
        req = req.set("Range", &format!("bytes={have}-"));
    }
    let resp = req.call().with_context(|| format!("could not reach {}", host(url)))?;
    // 206 = server honored the resume; 200 = it's sending the whole file.
    let (mut file, mut done) = if resp.status() == 206 {
        (OpenOptions::new().append(true).open(part)?, have)
    } else {
        (File::create(part)?, 0)
    };
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        done += n as u64;
        progress(done, total);
    }
    file.flush()?;
    if done != total {
        bail!("download stopped at {done} of {total} bytes");
    }
    Ok(())
}

fn verify(path: &Path, expected: &str) -> Result<()> {
    let mut f = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let got = format!("{:x}", hasher.finalize());
    if got != expected {
        bail!("checksum mismatch for {} (got {got})", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_accepts_right_hash_and_rejects_wrong_one() {
        let p = std::env::temp_dir().join(format!("itsmypa-verify-{}", std::process::id()));
        fs::write(&p, b"hello").unwrap();
        // sha256("hello")
        verify(&p, "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824").unwrap();
        assert!(verify(&p, &"0".repeat(64)).unwrap_err().to_string().contains("checksum"));
        fs::remove_file(&p).ok();
    }

    #[test]
    fn unused_whisper_files_are_removed_only_when_installed() {
        let dir = std::env::temp_dir().join(format!("itsmypa-unused-{}", std::process::id()));
        let w = dir.join("sherpa-onnx-whisper-small");
        fs::create_dir_all(&w).unwrap();
        let small = &WHISPER;
        fs::write(w.join("small-encoder.onnx"), vec![0u8; 1000]).unwrap();
        fs::write(w.join("small-decoder.onnx"), vec![0u8; 500]).unwrap();
        // Without the int8 model present nothing is touched.
        assert_eq!(small.remove_unused(&dir), 0);
        assert!(w.join("small-encoder.onnx").exists());

        fs::write(w.join("small-encoder.int8.onnx"), b"int8").unwrap();
        fs::write(w.join("small-decoder.int8.onnx"), b"int8").unwrap();
        assert_eq!(small.remove_unused(&dir), 1500);
        assert!(!w.join("small-encoder.onnx").exists() && !w.join("small-decoder.onnx").exists());
        assert!(w.join("small-encoder.int8.onnx").exists() && w.join("small-decoder.int8.onnx").exists());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn disk_space_check_counts_unpacking_and_explains_itself() {
        let dir = std::env::temp_dir().join(format!("itsmypa-space-{}", std::process::id()));
        // Archive: download + ~2x for unpacking + headroom.
        assert_eq!(WHISPER.space_needed(&dir), 639_387_718 * 3 + SPACE_HEADROOM);
        assert_eq!(NOTES_LLM.space_needed(&dir), 1_929_903_264 + SPACE_HEADROOM);
        assert!(ensure_space(1_000, 2_000).is_ok());
        let msg = ensure_space(2_230_000_000, 1_460_000_000).unwrap_err().to_string();
        assert_eq!(
            msg,
            "Not enough disk space: ItsMyPA needs about 2.3 GB free to finish setting up, and this Mac has 1.4 GB. Free up some space, then try again."
        );
    }

    #[test]
    fn default_choice_needs_download_in_an_empty_dir() {
        let dir = std::env::temp_dir().join(format!("itsmypa-models-{}", std::process::id()));
        if std::env::var_os("LLM_GGUF").is_none() {
            assert!(matches!(locate_llm(&dir), LlmLocation::NeedsDownload(m) if m.id == "llm"));
        }
    }
}

/// Network test: `cargo test -p itsmypa-engine -- --ignored download_resumes`
#[cfg(test)]
mod network_tests {
    use super::*;

    #[test]
    #[ignore]
    fn download_resumes_and_verifies() {
        let dir = std::env::temp_dir().join(format!("itsmypa-dl-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        // Simulate an interrupted download: the first 100 KB are already on disk.
        let full = ureq::get(VAD.urls[0]).call().unwrap();
        let mut bytes = Vec::new();
        full.into_reader().read_to_end(&mut bytes).unwrap();
        fs::write(dir.join("silero_vad.onnx.part"), &bytes[..100_000]).unwrap();

        let mut first_report = None;
        let path = download(&VAD, &dir, &mut |_, done, _| {
            first_report.get_or_insert(done);
        })
        .unwrap();
        assert!(first_report.unwrap() > 100_000, "resumed instead of restarting");
        assert_eq!(fs::metadata(&path).unwrap().len(), VAD.bytes);
        assert!(!dir.join("silero_vad.onnx.part").exists());

        // A corrupt partial file is caught by the checksum and thrown away.
        fs::remove_file(&path).unwrap();
        let mut bad = bytes.clone();
        bad[500_000] ^= 0xff;
        fs::write(dir.join("silero_vad.onnx.part"), &bad).unwrap();
        let err = download(&VAD, &dir, &mut |_, _, _| {}).unwrap_err();
        assert!(format!("{err:#}").contains("checksum"), "{err:#}");
        assert!(!dir.join("silero_vad.onnx.part").exists());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[ignore]
    fn archive_downloads_unpack() {
        let dir = std::env::temp_dir().join(format!("itsmypa-unpack-{}", std::process::id()));
        let path = download(&SPEAKER_SEGMENTATION, &dir, &mut |_, _, _| {}).unwrap();
        assert!(path.ends_with("sherpa-onnx-pyannote-segmentation-3-0/model.onnx") && path.exists());
        assert!(!dir.join("sherpa-onnx-pyannote-segmentation-3-0.tar.bz2.part").exists());
        fs::remove_dir_all(&dir).ok();
    }
}
