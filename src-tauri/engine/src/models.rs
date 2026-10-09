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
}

impl ModelFile {
    /// Where the model lives once installed.
    pub fn path(&self, dir: &Path) -> PathBuf {
        dir.join(self.unpacked.unwrap_or(self.file))
    }

    pub fn is_installed(&self, dir: &Path) -> bool {
        self.path(dir).exists()
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
};

pub const SPEAKER_SEGMENTATION: ModelFile = ModelFile {
    id: "seg",
    label: "Speaker detection",
    file: "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    urls: &[concat!("https://github.com/k2-fsa/sherpa-onnx/releases/download", "/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2")],
    sha256: Some("24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488"),
    bytes: 6_958_444,
    unpacked: Some("sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
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
};

macro_rules! whisper {
    ($size:literal, $bytes:expr, $sha:expr) => {
        ModelFile {
            id: concat!("whisper-", $size),
            label: concat!("Speech recognition (", $size, ")"),
            file: concat!("sherpa-onnx-whisper-", $size, ".tar.bz2"),
            urls: &[concat!("https://github.com/k2-fsa/sherpa-onnx/releases/download", "/asr-models/sherpa-onnx-whisper-", $size, ".tar.bz2")],
            sha256: $sha,
            bytes: $bytes,
            unpacked: Some(concat!("sherpa-onnx-whisper-", $size, "/", $size, "-encoder.int8.onnx")),
        }
    };
}

pub const WHISPERS: &[ModelFile] = &[
    whisper!("tiny", 116_204_861, Some(WHISPER_TINY_SHA)),
    whisper!("base", 207_557_382, Some(WHISPER_BASE_SHA)),
    whisper!("small", 639_387_718, Some(WHISPER_SMALL_SHA)),
    whisper!("medium", 1_931_372_882, None),
    whisper!("large-v3", 1_068_482_488, None),
];

const WHISPER_TINY_SHA: &str = "c46116994e539aa165266d96b325252728429c12535eb9d8b6a2b10f129e66b1";
const WHISPER_BASE_SHA: &str = "911b2083efd7c0dca2ac3b358b75222660dc09fb716d64fbfc417ba6c99ff3de";
const WHISPER_SMALL_SHA: &str = "486a46afbb7ba798507190ffe02fea2dd726049af212e774537efac6afb210a6";

pub fn whisper(size: &str) -> Option<&'static ModelFile> {
    WHISPERS.iter().find(|m| m.id.strip_prefix("whisper-") == Some(size))
}

/// Optional "better notes" model for bigger machines (16 GB+). In a
/// side-by-side test it caught every decision and used participants' names
/// where Qwen2.5-3B missed one — at 2.5x the download and ~1.6x the memory.
pub const GEMMA4_E4B: ModelFile = ModelFile {
    id: "gemma4-e4b",
    label: "Gemma 4 E4B (better notes, larger)",
    file: "gemma-4-E4B-it-Q4_K_M.gguf",
    urls: &["https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/e1d90e5fb9f61d8dc71ef016580784a054e5c787/gemma-4-E4B-it-Q4_K_M.gguf"],
    sha256: Some("519b9793ed6ce0ff530f1b7c96e848e08e49e7af4d57bb97f76215963a54146d"),
    bytes: 4_977_169_568,
    unpacked: None,
};

/// Default notes model: small first download (1.9 GB), fast, and runs well
/// even on 8 GB laptops — keeping installs light matters more than the small
/// quality gain from a bigger model.
pub const QWEN25_3B: ModelFile = ModelFile {
    id: "qwen2.5-3b",
    label: "Qwen2.5 3B (default)",
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
};

pub const DEFAULT_LLM: &ModelFile = &QWEN25_3B;

pub const LLMS: &[ModelFile] = &[QWEN25_3B, GEMMA4_E4B];

pub fn llm_by_id(id: &str) -> Option<&'static ModelFile> {
    LLMS.iter().find(|m| m.id == id)
}

pub enum LlmLocation {
    Ready(PathBuf),
    NeedsDownload(&'static ModelFile),
}

/// Where the notes model comes from. `$LLM_GGUF` (power users) wins; then the
/// chosen catalogue model if it's on disk; otherwise it must be downloaded.
pub fn locate_llm(models_dir: &Path, choice: &'static ModelFile) -> LlmLocation {
    if let Some(p) = std::env::var_os("LLM_GGUF") {
        return LlmLocation::Ready(PathBuf::from(p));
    }
    let path = models_dir.join(choice.file);
    if path.exists() {
        LlmLocation::Ready(path)
    } else {
        LlmLocation::NeedsDownload(choice)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Downloading,
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
    let part = dir.join(format!("{}.part", model.file));
    let mut last_err = None;
    for url in model.urls {
        let got = fetch(url, &part, model.bytes, &mut |d, t| progress(Phase::Downloading, d, t))
            .and_then(|_| model.sha256.map_or(Ok(()), |sha| verify(&part, sha)));
        match got {
            Ok(()) => {
                if model.unpacked.is_some() {
                    progress(Phase::Unpacking, model.bytes, model.bytes);
                    let unpacked = unpack(&part, dir);
                    fs::remove_file(&part).ok();
                    unpacked?;
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

fn fetch(url: &str, part: &Path, total: u64, progress: &mut dyn FnMut(u64, u64)) -> Result<()> {
    let have = fs::metadata(part).map(|m| m.len()).unwrap_or(0);
    if have == total {
        return Ok(()); // fully downloaded earlier; just needs verifying
    }
    let mut req = ureq::get(url).timeout(std::time::Duration::from_secs(60));
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
    fn default_choice_needs_download_in_an_empty_dir() {
        let dir = std::env::temp_dir().join(format!("itsmypa-models-{}", std::process::id()));
        if std::env::var_os("LLM_GGUF").is_none() {
            assert!(matches!(locate_llm(&dir, DEFAULT_LLM), LlmLocation::NeedsDownload(m) if m.id == "qwen2.5-3b"));
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
