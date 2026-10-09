//! Which model files the app uses, and downloading them safely.
//!
//! Every download is pinned to an exact upstream revision and verified
//! against its SHA-256, so a mirror serving an error page, a truncated file,
//! or a tampered model can never be loaded. Interrupted downloads resume.

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub struct ModelFile {
    pub id: &'static str,
    pub label: &'static str,
    pub file: &'static str,
    pub urls: &'static [&'static str],
    pub sha256: &'static str,
    pub bytes: u64,
}

/// Optional "better notes" model for bigger machines (16 GB+). In a
/// side-by-side test it caught every decision and used participants' names
/// where Qwen2.5-3B missed one — at 2.5x the download and ~1.6x the memory.
pub const GEMMA4_E4B: ModelFile = ModelFile {
    id: "gemma4-e4b",
    label: "Gemma 4 E4B (better notes, larger)",
    file: "gemma-4-E4B-it-Q4_K_M.gguf",
    urls: &["https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/e1d90e5fb9f61d8dc71ef016580784a054e5c787/gemma-4-E4B-it-Q4_K_M.gguf"],
    sha256: "519b9793ed6ce0ff530f1b7c96e848e08e49e7af4d57bb97f76215963a54146d",
    bytes: 4_977_169_568,
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
    sha256: "9c9f56a391a3abbd5b89d0245bf6106081bcc3173119d4229235dd9d23253f94",
    bytes: 1_929_903_264,
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

/// Download `model` into `dir`, resuming a previous partial download and
/// verifying the SHA-256 before the file gets its final name. `progress`
/// receives (bytes_done, bytes_total).
pub fn download(model: &ModelFile, dir: &Path, progress: &mut dyn FnMut(u64, u64)) -> Result<PathBuf> {
    fs::create_dir_all(dir)?;
    let dest = dir.join(model.file);
    if dest.exists() {
        return Ok(dest);
    }
    let part = dir.join(format!("{}.part", model.file));
    let mut last_err = None;
    for url in model.urls {
        match fetch(url, &part, model.bytes, progress).and_then(|_| verify(&part, model.sha256)) {
            Ok(()) => {
                fs::rename(&part, &dest)?;
                return Ok(dest);
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

    const VAD: ModelFile = ModelFile {
        id: "silero-vad",
        label: "Voice detector",
        file: "silero_vad.onnx",
        urls: &["https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"],
        sha256: "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
        bytes: 643_854,
    };

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
        let path = download(&VAD, &dir, &mut |done, _| {
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
        let err = download(&VAD, &dir, &mut |_, _| {}).unwrap_err();
        assert!(format!("{err:#}").contains("checksum"), "{err:#}");
        assert!(!dir.join("silero_vad.onnx.part").exists());
        fs::remove_dir_all(&dir).ok();
    }
}
