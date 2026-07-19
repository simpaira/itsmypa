#!/usr/bin/env python3
"""
ItsMyPA — Your private AI meeting PA. Everything on-device.
============================================================
Run:   ./run.sh          (creates venv, installs deps, starts server)
Open:  http://localhost:8765

Speech stack: sherpa-onnx (ONNX Runtime) — Whisper ASR + Silero VAD +
pyannote segmentation-3.0 + TitaNet speaker embeddings. All models download
automatically on first run; no Hugging Face token needed. The same engine and
model files run on macOS, Linux, Windows, Android, and iOS.

Long recordings are processed as background jobs: POST /transcribe returns a
job id immediately and the UI polls GET /jobs/{id} for progress and results.
"""

import os, json, math, re, sys, uuid, queue, shutil, subprocess, tarfile, tempfile, threading, time, webbrowser
import signal as signal_module
from collections import Counter, namedtuple
from dotenv import load_dotenv
from llama_cpp import Llama
from datetime import datetime, timedelta, timezone
from pathlib import Path
from platformdirs import user_data_dir
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import uvicorn

APP_NAME = "ItsMyPA"

# ── Locations ──────────────────────────────────────────────────────────────────
# When packaged, the install directory is read-only — all mutable state (notes,
# models, recording backups, settings) lives in the platform app-data dir.
# _BUNDLE_DIR is where read-only assets (ui.html) ship; inside a PyInstaller
# build that's the unpack dir, in development it's the repo.
_BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
_REPO_DIR   = Path(__file__).resolve().parent
DATA_DIR    = Path(os.getenv("ITSMYPA_DATA_DIR") or user_data_dir(APP_NAME, appauthor=False))
DATA_DIR.mkdir(parents=True, exist_ok=True)

NOTES_FILE     = DATA_DIR / "notes.json"
MODELS_DIR     = DATA_DIR / "models"
RECORDINGS_DIR = DATA_DIR / "recordings"
ENV_FILE       = DATA_DIR / ".env"

# ── Config ─────────────────────────────────────────────────────────────────────
load_dotenv(ENV_FILE)
# Meetings are private — bind to localhost only by default. Set HOST=0.0.0.0
# explicitly if you want other devices on your network to reach the app.
HOST          = os.getenv("HOST", "127.0.0.1")
PORT          = int(os.getenv("PORT", "8765"))
SKIP_MODELS   = os.getenv("ITSMYPA_SKIP_MODELS", "") == "1"  # fast imports for tests
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")   # tiny | base | small | medium | large-v3
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "")  # "" = auto-detect, or e.g. "en"
NUM_THREADS   = max(2, (os.cpu_count() or 4) // 2)
SAMPLE_RATE   = 16000

# Map-reduce kicks in once a transcript crosses this many words, so very long
# meetings don't blow the summarizer's context window or produce shallow notes.
LONG_TRANSCRIPT_WORD_THRESHOLD = 3500
CHUNK_WORD_SIZE = 2800

# ── Diarization config ─────────────────────────────────────────────────────────
DIARIZE           = os.getenv("DIARIZE", "1").lower() not in ("0", "false", "no")
# Clustering distance; lower = more speakers. Do not raise past ~0.85: verified
# empirically that 0.9 merges even two very distinct voices into one. Cluster
# over-splitting is repaired downstream (_merge_similar_clusters) instead.
DIARIZE_THRESHOLD = float(os.getenv("DIARIZE_THRESHOLD", "0.8"))
NUM_SPEAKERS      = int(os.getenv("NUM_SPEAKERS", "-1"))          # -1 = detect automatically

# ── Speech models (sherpa-onnx / ONNX Runtime) ────────────────────────────────
# One engine for ASR + VAD + diarization; the same library and model files run
# on macOS, Linux, Windows, Android, and iOS — no torch, no HF token.
import sherpa_onnx

_GITHUB = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
_MODEL_URLS = {
    "asr": f"{_GITHUB}/asr-models/sherpa-onnx-whisper-{WHISPER_MODEL}.tar.bz2",
    "vad": f"{_GITHUB}/asr-models/silero_vad.onnx",
    "seg": f"{_GITHUB}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    # NB: "recongition" is a real typo in the upstream release tag — don't fix it.
    "emb": f"{_GITHUB}/speaker-recongition-models/nemo_en_titanet_small.onnx",
}

# ── First-run setup progress ────────────────────────────────────────────────────
# Everything the app needs (ASR, VAD, diarization, summarizer) downloads on
# first use rather than shipping in the installer. This tracks per-component
# state so the UI can show real progress and refuse to start a meeting until
# every required piece is actually ready — instead of a vague spinner.
_setup_progress = {
    "vad": {"label": "Voice detector",                          "status": "pending", "bytes": 0, "total": 0, "error": None},
    "asr": {"label": f"Speech recognition ({WHISPER_MODEL})",   "status": "pending", "bytes": 0, "total": 0, "error": None},
    "seg": {"label": "Speaker detection",                       "status": "pending", "bytes": 0, "total": 0, "error": None},
    "emb": {"label": "Speaker identification",                  "status": "pending", "bytes": 0, "total": 0, "error": None},
    "llm": {"label": "Meeting notes AI",                        "status": "pending", "bytes": 0, "total": 0, "error": None},
}
_setup_lock = threading.Lock()

def _setup_set(key: str, **fields):
    with _setup_lock:
        if key in _setup_progress:
            _setup_progress[key].update(fields)

def _download(url: str, dest: Path, key: str = None):
    import requests
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"\n⬇️  Downloading {url.rsplit('/', 1)[-1]}…", flush=True)
    if key:
        _setup_set(key, status="downloading", bytes=0, total=0)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        if key:
            _setup_set(key, total=total)
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r   {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
                if key:
                    _setup_set(key, bytes=done)
    tmp.rename(dest)
    print(" ✅")

def _fetch_archive(url: str, marker: Path, key: str = None):
    """Download + extract a .tar.bz2 into MODELS_DIR unless `marker` already exists."""
    if marker.exists():
        if key:
            _setup_set(key, status="ready")
        return
    if key:
        _setup_set(key, status="downloading")
    arc = MODELS_DIR / url.rsplit("/", 1)[-1]
    _download(url, arc, key)
    if key:
        _setup_set(key, status="extracting")
    with tarfile.open(arc) as t:
        t.extractall(MODELS_DIR)
    arc.unlink()

def _fetch_file(url: str, dest: Path, key: str = None):
    if not dest.exists():
        _download(url, dest, key)
    elif key:
        _setup_set(key, status="ready")

MODELS_DIR.mkdir(exist_ok=True)
_asr_dir  = MODELS_DIR / f"sherpa-onnx-whisper-{WHISPER_MODEL}"
_enc      = _asr_dir / f"{WHISPER_MODEL}-encoder.int8.onnx"
_dec      = _asr_dir / f"{WHISPER_MODEL}-decoder.int8.onnx"
_tokens   = _asr_dir / f"{WHISPER_MODEL}-tokens.txt"
_vad_path = MODELS_DIR / "silero_vad.onnx"
_seg_path = MODELS_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
_emb_path = MODELS_DIR / "nemo_en_titanet_small.onnx"

# We decode audio by calling ffmpeg directly (no ffprobe needed). Prefer the
# static ffmpeg bundled via imageio-ffmpeg so the app works with nothing
# installed; fall back to a system ffmpeg (adding the usual Homebrew locations
# a GUI-launched app's minimal PATH would otherwise miss).
_ffmpeg = None
try:
    import imageio_ffmpeg
    _cand = imageio_ffmpeg.get_ffmpeg_exe()
    if _cand and os.path.exists(_cand):
        _ffmpeg = _cand
except Exception:
    pass
if not _ffmpeg:
    for _p in ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/opt/local/bin"):
        if os.path.isdir(_p) and _p not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + _p
    _ffmpeg = shutil.which("ffmpeg")
FFMPEG_OK = _ffmpeg is not None
if FFMPEG_OK:
    try:
        os.chmod(_ffmpeg, 0o755)   # PyInstaller can drop the exec bit on collected binaries
    except Exception:
        pass
else:
    print("⚠️  ffmpeg not found — recordings and most uploads will fail to decode.")

_recognizer = None
_diarizer = None
MODELS_LOADING = not SKIP_MODELS   # flips to False once the background load finishes

def _load_speech_models():
    global _recognizer, _diarizer, DIARIZE, MODELS_LOADING
    try:
        _fetch_archive(_MODEL_URLS["asr"], _enc, "asr")
        _fetch_file(_MODEL_URLS["vad"], _vad_path, "vad")
        _setup_set("asr", status="loading")
        _setup_set("vad", status="loading")
        print(f"🔄 Loading Whisper '{WHISPER_MODEL}' (int8, sherpa-onnx)…", end=" ", flush=True)
        _recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(_enc), decoder=str(_dec), tokens=str(_tokens),
            language=WHISPER_LANGUAGE, task="transcribe", num_threads=NUM_THREADS,
        )
        print("✅")
        _setup_set("asr", status="ready")
        _setup_set("vad", status="ready")
    except Exception as e:
        print(f"\n❌ Could not set up transcription models: {e}")
        _setup_set("asr", status="error", error=str(e))
        _setup_set("vad", status="error", error=str(e))

    if not DIARIZE:
        _setup_set("seg", status="skipped")
        _setup_set("emb", status="skipped")
    elif _recognizer is not None:
        try:
            _fetch_archive(_MODEL_URLS["seg"], _seg_path, "seg")
            _fetch_file(_MODEL_URLS["emb"], _emb_path, "emb")
            _setup_set("seg", status="loading")
            _setup_set("emb", status="loading")
            print("🔄 Loading diarization (pyannote seg-3.0 + TitaNet, ONNX)…", end=" ", flush=True)
            _diarizer = sherpa_onnx.OfflineSpeakerDiarization(
                sherpa_onnx.OfflineSpeakerDiarizationConfig(
                    segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(_seg_path)),
                        num_threads=NUM_THREADS,
                    ),
                    embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                        model=str(_emb_path), num_threads=NUM_THREADS,
                    ),
                    clustering=sherpa_onnx.FastClusteringConfig(
                        num_clusters=NUM_SPEAKERS, threshold=DIARIZE_THRESHOLD,
                    ),
                    # 0.6 so sub-second blips ("Um,", "Yeah.") never become
                    # turns — their embeddings are noise that spawns phantom
                    # speakers; the text still inherits the neighbor's label.
                    min_duration_on=0.6,
                    min_duration_off=0.5,
                )
            )
            print("✅")
            _setup_set("seg", status="ready")
            _setup_set("emb", status="ready")
        except Exception as e:
            print(f"\n⚠️  Diarization unavailable ({e}) — transcripts will not have speaker labels.")
            _diarizer = None
            _setup_set("seg", status="error", error=str(e))
            _setup_set("emb", status="error", error=str(e))
    else:
        _setup_set("seg", status="skipped")
        _setup_set("emb", status="skipped")
    DIARIZE = _diarizer is not None

# ── Notes persistence ────────────────────────────────────────────────────────
def _load_notes() -> list:
    if not NOTES_FILE.exists():
        return []
    try:
        return json.loads(NOTES_FILE.read_text())
    except json.JSONDecodeError as e:
        print(f"⚠️  Notes file corrupted ({e}), resetting. Backup → notes.bak.json")
        NOTES_FILE.rename(NOTES_FILE.with_suffix(".bak.json"))
        return []

_notes: list = _load_notes()
_lock = threading.Lock()

# Serializes access to the Whisper model. The model objects are not safe for
# concurrent inference; the lock is held per decode chunk so parallel jobs
# interleave instead of piling up.
_infer_lock = threading.Lock()

_notes_version = 0   # bumped on every save so the ask-index knows to rebuild

def save_notes():
    # Atomic write: a crash mid-save can't corrupt the whole archive.
    global _notes_version
    tmp = NOTES_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(_notes, indent=2))
    tmp.replace(NOTES_FILE)
    _notes_version += 1

# ── Load Summarizer (local GGUF via llama-cpp-python, in-process) ─────────────
# Resolution order: $LLM_GGUF → local Gemma file (if present) → auto-download a
# small default model, so first run works on any machine without setup.
DEFAULT_LLM_NAME = "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
# HF's CDN rate-limits anonymous large downloads per IP (403 AccessDenied), so
# fall back to ModelScope, which mirrors the identical bartowski file.
DEFAULT_LLM_URLS = [
    f"https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/{DEFAULT_LLM_NAME}",
    "https://modelscope.cn/api/v1/models/bartowski/Qwen2.5-3B-Instruct-GGUF/repo"
    f"?Revision=master&FilePath={DEFAULT_LLM_NAME}",
]
def _resolve_llm_path() -> Path:
    if os.getenv("LLM_GGUF"):
        _setup_set("llm", status="ready")   # user-supplied — nothing to download
        return Path(os.getenv("LLM_GGUF")).expanduser()
    # Any GGUF already in the models dir works (user-dropped or previously
    # downloaded) — also the escape hatch when the default host denies
    # anonymous downloads (HF's CDN rate-limits big files per IP).
    existing = sorted(MODELS_DIR.glob("*.gguf"))
    if existing:
        _setup_set("llm", status="ready")
        return existing[0]
    dest = MODELS_DIR / DEFAULT_LLM_NAME
    last_err = None
    for url in DEFAULT_LLM_URLS:
        try:
            _fetch_file(url, dest, "llm")
            # A mirror returning an HTML/XML error page with HTTP 200 must not
            # poison the models dir — real model files start with GGUF magic.
            with open(dest, "rb") as f:
                if f.read(4) != b"GGUF":
                    dest.unlink()
                    raise ValueError("downloaded file is not a valid GGUF")
            return dest
        except Exception as e:
            last_err = e
            print(f"⚠️ LLM download failed from {url.split('/')[2]}: {e}")
    raise last_err

_llm = None
LLM_NAME = "offline"

def _load_llm():
    global _llm, LLM_NAME
    try:
        _llm_path = _resolve_llm_path()
        LLM_NAME = _llm_path.stem
        _setup_set("llm", status="loading")
        print(f"🔄 Loading summarizer '{_llm_path.name}'…", end=" ", flush=True)
        # n_ctx sizes the pre-allocated KV cache — 32k cost multiple GB and
        # pushed 16 GB machines into swap (system-wide hang + coreaudiod
        # starvation = dead audio). 8k comfortably fits every prompt we build
        # (summary map chunks ≈4k tokens, chat/ask context ≈7k max).
        _llm = Llama(model_path=str(_llm_path), n_ctx=int(os.getenv("LLM_CTX", "8192")),
                     n_gpu_layers=-1, verbose=False)
        print("✅")
        _setup_set("llm", status="ready")
    except Exception as e:
        print(f"❌ Summarizer failed to load: {e}")
        _llm = None
        _setup_set("llm", status="error", error=str(e))

def _load_all_models():
    """Load everything in the background so the server (and UI) start instantly."""
    global MODELS_LOADING
    try:
        _load_speech_models()
        _load_llm()
    finally:
        MODELS_LOADING = False
        print(f"✅ All models ready — http://localhost:{PORT}")

if not SKIP_MODELS:
    threading.Thread(target=_load_all_models, daemon=True, name="model-loader").start()

# ── Audio helpers ──────────────────────────────────────────────────────────────
def load_audio_16k(src: str) -> np.ndarray:
    """Decode any audio/video file to mono 16 kHz float32 samples by piping it
    through ffmpeg directly — no ffprobe, no pydub, works with the bundled
    static ffmpeg alone."""
    if not FFMPEG_OK:
        raise ValueError("Could not decode the audio — ffmpeg is unavailable.")
    cmd = [_ffmpeg, "-nostdin", "-i", src, "-vn", "-f", "s16le",
           "-ac", "1", "-ar", str(SAMPLE_RATE), "-loglevel", "error", "pipe:1"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        raise ValueError("Could not decode the audio file: "
                         + proc.stderr.decode(errors="replace")[:200])
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0

def fmt_time(s: float) -> str:
    return f"{int(s // 60):02d}:{int(s % 60):02d}"

# ── Transcription pipeline ─────────────────────────────────────────────────────
# Whisper's context window is 30 s; decode windows this size give it full
# sentence context (decoding isolated 1-2 s fragments wrecks accuracy).
MAX_CHUNK_SEC = 28.0

def _vad_split(samples: np.ndarray):
    """Detect speech regions: [(start_sec, end_sec), …]."""
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(_vad_path)
    cfg.silero_vad.threshold = 0.5
    cfg.silero_vad.min_silence_duration = 0.5
    cfg.silero_vad.min_speech_duration = 0.25
    cfg.silero_vad.max_speech_duration = 20
    cfg.sample_rate = SAMPLE_RATE
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)
    window = cfg.silero_vad.window_size

    out = []
    def drain():
        while not vad.empty():
            seg = vad.front
            start = seg.start / SAMPLE_RATE
            out.append((start, start + len(seg.samples) / SAMPLE_RATE))
            vad.pop()

    for i in range(0, len(samples), window):
        vad.accept_waveform(samples[i:i + window])
        drain()
    vad.flush()
    drain()
    return out


def _group_spans(spans, speakers, samples: np.ndarray):
    """Merge consecutive same-speaker VAD spans into ≤28 s decode windows.
    Returns [(speaker_or_None, start_sec, end_sec, chunk_samples), …]."""
    chunks = []  # [speaker, start, end]
    for (s, e), sp in zip(spans, speakers):
        c = chunks[-1] if chunks else None
        if c and c[0] == sp and s - c[2] <= 3.0 and e - c[1] <= MAX_CHUNK_SEC:
            c[2] = e
        else:
            chunks.append([sp, s, e])
    return [(sp, s, e, samples[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)])
            for sp, s, e in chunks]


def _decode(chunk: np.ndarray) -> str:
    st = _recognizer.create_stream()
    st.accept_waveform(SAMPLE_RATE, chunk)
    _recognizer.decode_stream(st)
    return st.result.text.strip()


_Turn = namedtuple("_Turn", ["speaker", "start", "end"])

# ── Second-stage cluster repair ──────────────────────────────────────────────
# The clusterer works on short-turn embeddings, which are noisy — one person
# can fragment into several clusters (an interview once yielded 11 "speakers"
# for 2 people). Each fragment has plenty of total audio though, so we
# re-embed every cluster on up to 20 s of its own speech (stable statistics)
# and merge clusters whose voices actually match. Calibration on known audio:
# same voice ≥ 0.93 cosine similarity, different voices ≤ 0.26.
DIARIZE_MERGE_SIM = float(os.getenv("DIARIZE_MERGE_SIM", "0.60"))
_emb_extractor = None
_emb_lock = threading.Lock()

def _cluster_embedding(ex, turns_for_sp, samples, cap_sec=20.0):
    pieces, got = [], 0.0
    for t in sorted(turns_for_sp, key=lambda t: -(t.end - t.start)):   # longest first
        pieces.append(samples[int(t.start * SAMPLE_RATE):int(t.end * SAMPLE_RATE)])
        got += t.end - t.start
        if got >= cap_sec:
            break
    if not pieces:
        return None
    audio = np.concatenate(pieces)
    if len(audio) < SAMPLE_RATE:   # under 1 s can't give a stable embedding
        return None
    st = ex.create_stream()
    st.accept_waveform(SAMPLE_RATE, audio)
    st.input_finished()
    emb = np.array(ex.compute(st))
    n = np.linalg.norm(emb)
    return emb / n if n else None

def _merge_similar_clusters(turns, samples):
    """Merge diarization clusters that are the same voice. Never raises — on
    any failure the original turns pass through unchanged."""
    if NUM_SPEAKERS > 0 or not turns:
        return turns
    by = {}
    for t in turns:
        by.setdefault(t.speaker, []).append(t)
    if len(by) < 2:
        return turns
    global _emb_extractor
    try:
        with _emb_lock:
            if _emb_extractor is None:
                _emb_extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
                    sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                        model=str(_emb_path), num_threads=NUM_THREADS))
            embs = {sp: _cluster_embedding(_emb_extractor, ts, samples) for sp, ts in by.items()}
    except Exception as e:
        print(f"⚠️  Cluster re-merge skipped: {e}")
        return turns

    parent = {sp: sp for sp in by}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    sps = list(by)
    for i in range(len(sps)):
        for j in range(i + 1, len(sps)):
            a, b = embs.get(sps[i]), embs.get(sps[j])
            if a is not None and b is not None and float(np.dot(a, b)) >= DIARIZE_MERGE_SIM:
                parent[find(sps[i])] = find(sps[j])

    dur = {sp: sum(t.end - t.start for t in ts) for sp, ts in by.items()}
    groups = {}
    for sp in sps:
        groups.setdefault(find(sp), []).append(sp)
    remap = {}
    for members in groups.values():
        canon = max(members, key=lambda s: dur[s])   # longest speaker keeps the label
        for m in members:
            remap[m] = canon
    merged = len(by) - len(groups)
    if merged:
        print(f"🗣  Diarization: merged {merged} same-voice cluster{'s' if merged != 1 else ''} "
              f"({len(by)} → {len(groups)} speakers)")
    return [_Turn(remap[t.speaker], t.start, t.end) for t in turns]

# A "speaker" with less than this much total speech across the whole meeting is
# almost always clustering noise: sub-second interjections ("Next.", "Um,")
# can't produce a stable voice embedding, so they spawn phantom clusters.
_MIN_SPEAKER_SPEECH_SEC = float(os.getenv("MIN_SPEAKER_SPEECH", "6.0"))

def _consolidate_turns(turns):
    """Fold phantom speakers (tiny total speech) into the temporally nearest
    real speaker. An interview that clustered into 11 'speakers' had 6 of them
    speaking under 4 seconds each — noise, not people. Skipped when the user
    pinned an exact speaker count in Settings (clustering already honored it).
    """
    if NUM_SPEAKERS > 0 or not turns:
        return turns
    totals = {}
    for t in turns:
        totals[t.speaker] = totals.get(t.speaker, 0.0) + (t.end - t.start)
    major = {sp for sp, dur in totals.items() if dur >= _MIN_SPEAKER_SPEECH_SEC}
    if not major or len(major) == len(totals):
        return turns
    majors = [t for t in turns if t.speaker in major]
    out = []
    for t in turns:
        if t.speaker in major:
            out.append(_Turn(t.speaker, t.start, t.end))
            continue
        # nearest real speaker by temporal gap (0 when overlapping)
        nearest = min(majors, key=lambda m: max(m.start - t.end, t.start - m.end, 0.0))
        out.append(_Turn(nearest.speaker, t.start, t.end))
    dropped = len(totals) - len(major)
    print(f"🗣  Diarization: {len(totals)} clusters → {len(major)} speakers "
          f"({dropped} phantom cluster{'s' if dropped != 1 else ''} folded in)")
    return out


def _split_spans_at_turns(spans, turns, min_len: float = 0.7):
    """Cut VAD speech spans at diarization turn boundaries, then label each piece.

    Labeling whole spans by majority overlap merged rapid back-and-forth
    dialogue (no silence gap → one VAD span) under a single speaker, so one
    block contained both voices' words. Cutting at the diarizer's own turn
    boundaries puts the label changes where the speakers actually change.
    Pieces shorter than `min_len` are absorbed into their neighbor — isolated
    sub-second fragments decode terribly and are usually segmentation jitter.
    Returns (spans, speakers) of equal length.
    """
    out_spans, out_speakers = [], []
    for s, e in spans:
        cuts = {s, e}
        for t in turns:
            for b in (t.start, t.end):
                if s + min_len < b < e - min_len:
                    cuts.add(b)
        pieces = []   # [start, end, speaker]
        for a, b in zip(sorted(cuts), sorted(cuts)[1:]):
            best, best_ov = None, 0.0
            for t in turns:
                ov = min(b, t.end) - max(a, t.start)
                if ov > best_ov:
                    best, best_ov = t.speaker, ov
            if pieces and (pieces[-1][2] == best or (b - a) < min_len):
                pieces[-1][1] = b
            else:
                pieces.append([a, b, best])
        for a, b, sp in pieces:
            out_spans.append((a, b))
            out_speakers.append(sp)
    # pieces no turn covered inherit the previous piece's speaker
    for i in range(len(out_speakers)):
        if out_speakers[i] is None:
            out_speakers[i] = (out_speakers[i - 1] if i and out_speakers[i - 1] is not None
                               else next((sp for sp in out_speakers[i:] if sp is not None), 0))
    return out_spans, out_speakers


def _run_dual_channel(job: dict):
    """Two-channel pipeline (Granola-style, but better): the mic channel IS the
    user, so it's labeled "You" with no ML at all; diarization only has to
    separate the remote participants on the system-audio channel."""
    job["phase"] = "converting"
    mic = load_audio_16k(job["path"])
    sys_a = load_audio_16k(job["system_path"])
    offset = float(job.get("system_offset") or 0.0)

    mic_spans = _vad_split(mic)
    sys_spans = _vad_split(sys_a)
    if not mic_spans and not sys_spans:
        raise ValueError("No speech detected. Try speaking closer to the mic or record longer.")

    # Adaptive: computer audio is captured on every recording, but if little or
    # nothing came through it (an in-person meeting, or a call where the remote
    # side stayed quiet), the "mic = You" assumption is wrong — the mic has
    # everyone. Fall back to normal single-channel diarization so in-person
    # meetings still get real per-speaker labels instead of everyone as "You".
    sys_speech = sum(e - s for s, e in sys_spans)
    if sys_speech < 2.0:
        job["system_path"] = None
        return _run_transcription(job, job["path"])

    sys_speakers = ["them"] * len(sys_spans)
    base = 0.0
    if _diarizer is not None and sys_spans:
        job["phase"] = "diarizing"
        def on_progress(done: int, total: int) -> int:
            job["progress"] = 0.3 * (done / max(total, 1))
            return 0
        turns = _diarizer.process(sys_a, callback=on_progress).sort_by_start_time()
        turns = _consolidate_turns(_merge_similar_clusters(turns, sys_a))
        if len(turns) > 0:
            sys_spans, sys_speakers = _split_spans_at_turns(sys_spans, turns)
        base = 0.3

    tagged = ([(sp, s, e, c) for sp, s, e, c in _group_spans(mic_spans, ["you"] * len(mic_spans), mic)]
              + [(sp, s + offset, e + offset, c) for sp, s, e, c
                 in _group_spans(sys_spans, sys_speakers, sys_a)])
    tagged.sort(key=lambda x: x[1])

    job["phase"] = "transcribing"
    parts = []
    for i, (sp, start, end, chunk) in enumerate(tagged):
        with _infer_lock:
            text = _decode(chunk)
        if text:
            parts.append((sp, start, end, text))
        job["progress"] = base + (1 - base) * (i + 1) / len(tagged)
        job["transcript"] = " ".join(p[3] for p in sorted(parts, key=lambda p: p[1]))

    if not parts:
        raise ValueError("No speech detected. Try speaking closer to the mic or record longer.")

    job["phase"] = "finalizing"
    parts.sort(key=lambda p: p[1])
    mapping = {}   # remote speakers become Speaker 2, 3, … ("You" is speaker 1)
    merged = []
    for sp, start, end, text in parts:
        if sp == "you":
            name = "You"
        elif sp == "them":
            name = "Them"
        else:
            if sp not in mapping:
                mapping[sp] = f"Speaker {len(mapping) + 2}"
            name = mapping[sp]
        seg = {"speaker": name, "text": text, "start": round(start, 2), "end": round(end, 2)}
        if (merged and merged[-1]["speaker"] == seg["speaker"]
                and seg["start"] - merged[-1]["end"] < 2):
            merged[-1]["text"] += " " + seg["text"]
            merged[-1]["end"]   = seg["end"]
        else:
            merged.append(seg)

    job["segments"]   = merged
    job["transcript"] = "\n\n".join(f"[{s['speaker']} {fmt_time(s['start'])}] {s['text']}" for s in merged)


def _run_transcription(job: dict, audio_path: str):
    """Full pipeline: decode file → VAD → diarize → Whisper over speaker-grouped
    windows → merge. Diarization runs first so each decode window contains a
    single speaker's speech with maximum context."""
    if job.get("system_path"):
        return _run_dual_channel(job)
    job["phase"] = "converting"
    samples = load_audio_16k(audio_path)

    spans = _vad_split(samples)
    if not spans:
        raise ValueError("No speech detected. Try speaking closer to the mic or record longer.")

    speakers = [None] * len(spans)
    if _diarizer is not None:
        job["phase"] = "diarizing"
        def on_progress(done: int, total: int) -> int:
            job["progress"] = 0.35 * (done / max(total, 1))
            return 0
        turns = _diarizer.process(samples, callback=on_progress).sort_by_start_time()
        turns = _consolidate_turns(_merge_similar_clusters(turns, samples))
        if len(turns) > 0:
            # Cut spans where the diarizer says the speaker changes — rapid
            # dialogue with no silence gap must not collapse into one label.
            spans, speakers = _split_spans_at_turns(spans, turns)

    chunks = _group_spans(spans, speakers, samples)

    # Transcribe window by window, persisting partial text as we go — a failure
    # at minute 50 keeps the first 49. The lock is held per window (not for the
    # whole meeting) so concurrent jobs interleave instead of stalling.
    job["phase"] = "transcribing"
    base = 0.35 if _diarizer is not None else 0.0
    parts = []  # (speaker, start_sec, end_sec, text)
    for i, (sp, start, end, chunk) in enumerate(chunks):
        with _infer_lock:
            text = _decode(chunk)
        if text:
            parts.append((sp, start, end, text))
        job["progress"] = base + (1 - base) * (i + 1) / len(chunks)
        job["transcript"] = " ".join(p[3] for p in parts)

    if not parts:
        raise ValueError("No speech detected. Try speaking closer to the mic or record longer.")

    if _diarizer is None:
        return

    job["phase"] = "finalizing"
    mapping = {}
    merged = []
    for sp, start, end, text in parts:
        if sp not in mapping:
            mapping[sp] = f"Speaker {len(mapping) + 1}"
        seg = {"speaker": mapping[sp], "text": text, "start": start, "end": end}
        if (merged and merged[-1]["speaker"] == seg["speaker"]
                and seg["start"] - merged[-1]["end"] < 2):
            merged[-1]["text"] += " " + seg["text"]
            merged[-1]["end"]   = seg["end"]
        else:
            merged.append(seg)

    job["segments"]   = merged
    job["transcript"] = "\n\n".join(f"[{s['speaker']} {fmt_time(s['start'])}] {s['text']}" for s in merged)


# ── Background job queue ───────────────────────────────────────────────────────
# Long transcriptions run in a worker thread instead of inside the HTTP request:
# the upload returns a job id immediately and the UI polls /jobs/{id}, so a
# 90-minute meeting can't be killed by a browser timeout or a laptop nap.
_jobs: dict = {}
_job_queue: "queue.Queue[str]" = queue.Queue()

def _prune_jobs():
    # Finished jobs are kept for hours (they're a few KB) so a browser refresh —
    # or reopening the app later — can still recover an unsaved transcript.
    cutoff = time.time() - 12 * 3600
    for jid in [j for j, v in _jobs.items() if v["status"] in ("done", "error") and v["created"] < cutoff]:
        _jobs.pop(jid, None)

def _job_worker():
    while True:
        jid = _job_queue.get()
        job = _jobs.get(jid)
        if job is None:
            continue
        job["status"] = "processing"
        try:
            _run_transcription(job, job["path"])
            job["progress"] = 1.0
            job["status"]   = "done"
        except Exception as e:
            job["error"]  = str(e) or e.__class__.__name__
            job["status"] = "error"
            print(f"⚠️  Transcription job {jid} failed: {e}")
        finally:
            job["phase"] = None
            for p in (job["path"], job.get("system_path")):
                if p:
                    try: os.unlink(p)
                    except Exception: pass

threading.Thread(target=_job_worker, daemon=True, name="transcribe-worker").start()


# ── Summarization prompts ────────────────────────────────────────────────────
# Every summary — direct or map-reduced — is forced into this exact template so
# the frontend can reliably parse it into Executive Summary / Decisions /
# Action Items / Topics cards instead of a single wall of markdown.
SUMMARY_TEMPLATE_RULES = """Follow this EXACT markdown template, with these exact section headers, in this exact order, every time:

## Executive Summary
2-4 sentences: what the meeting was about and its outcome.

## Key Decisions
- One bullet per decision actually made. If none were made, write a single bullet: "No decisions were recorded."

## Action Items
- [ ] Task description — @Owner
(One bullet per task. Use @Unassigned if no owner is stated in the transcript. If there are no action items, write "- [ ] No action items were recorded. — @Unassigned")

## Topics Discussed
- Short topic label (3-8 labels, not sentences)

Rules:
- Be concise, factual, and specific. Never invent names, numbers, or facts not present in the source.
- Output ONLY the template above — no preamble, no closing remarks, no extra headers.
"""

SUMMARY_PROMPT = (
    "You are an expert meeting analyst. Read the transcript below and produce structured notes.\n\n"
    + SUMMARY_TEMPLATE_RULES +
    "\n---\nTRANSCRIPT:\n{{TRANSCRIPT}}\n---"
)

MAP_PROMPT = (
    "Extract the key facts, decisions, and action items from this excerpt "
    "(part {{IDX}} of {{TOTAL}}) of a longer meeting transcript. "
    "Be terse: short bullet points only, no commentary, no headers.\n\n"
    "EXCERPT:\n{{CHUNK}}"
)

REDUCE_PROMPT = (
    "You are an expert meeting analyst. Below are bullet-point notes extracted independently "
    "from sequential parts of one long meeting transcript. Synthesize them into a single set "
    "of meeting notes, removing duplicates and resolving the full picture.\n\n"
    + SUMMARY_TEMPLATE_RULES +
    "\n---\nNOTES FROM TRANSCRIPT PARTS:\n{{NOTES}}\n---"
)


# Note-style templates (Granola-style): a flavor line steers what the
# summarizer weights. Keys must match the <select> options in ui.html.
SUMMARY_TEMPLATES = {
    "general":    "",
    "standup":    "Context: this is a daily standup. Weight per-person updates, blockers, and who is waiting on whom.",
    "one_on_one": "Context: this is a 1:1 conversation. Weight feedback, career/personal topics, agreements, and follow-ups; keep the tone neutral and discreet.",
    "sales":      "Context: this is a sales or client call. Weight the client's needs and pain points, objections raised, budget/timeline signals, and concrete next steps.",
    "interview":  "Context: this is a job interview. Weight the candidate's background, demonstrated skills, notable answers, concerns, and suggested follow-up questions.",
}

# Granola-style "enhance my notes": the user's rough bullets guide the AI, and
# the output repeats each user note verbatim (rendered black in the UI) followed
# by AI expansions from the transcript (rendered gray).
ENHANCE_PROMPT = (
    "You are enhancing a user's rough meeting notes using the meeting transcript.\n"
    "For EACH line or bullet of the user's notes, output exactly:\n"
    "### <the user's note, repeated verbatim>\n"
    "- 2 to 5 factual bullets expanding that note with specifics from the transcript: "
    "decisions, numbers, owners, dates, and short quotes.\n"
    "Keep the user's original order. If a note has no match in the transcript, output a "
    "single bullet: 'Not discussed in the transcript.'\n"
    "After covering every user note, add one final section:\n"
    "### Also worth noting\n"
    "- up to 4 bullets for important points the user's notes missed. If none, skip this section.\n"
    "Rules: never invent facts, names, or numbers not present in the transcript. "
    "Output ONLY the format above — no preamble, no closing remarks.\n\n"
    "---\nUSER NOTES:\n{{NOTES}}\n---\nTRANSCRIPT:\n{{TRANSCRIPT}}\n---"
)

EMAIL_PROMPT = (
    "Write a short, professional follow-up email for the meeting below.\n"
    "Rules:\n"
    "- Start with 'Subject: ' on the first line.\n"
    "- Briefly thank attendees, recap the key decisions, then list action items with owners.\n"
    "- Use only facts from the meeting notes/transcript — never invent names, dates, or commitments.\n"
    "- Keep it under 180 words. Plain text only, no markdown.\n\n"
    "---\nMEETING NOTES:\n{{NOTES}}\n---"
)

TITLE_PROMPT = (
    "Give this meeting a short, specific title (3-7 words). "
    "Output ONLY the title — no quotes, no punctuation at the end, no commentary.\n\n"
    "---\nTRANSCRIPT (start of meeting):\n{{TRANSCRIPT}}\n---"
)


def chunk_words(text: str, max_words: int) -> list:
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


# A multi-hour transcript can overflow the model's context window; for chat we
# keyword-retrieve the chunks most relevant to the question instead.
CHAT_CONTEXT_WORDS = int(os.getenv("CHAT_CONTEXT_WORDS", "5000"))   # ≈7k tokens, fits LLM_CTX=8192

def select_chat_context(transcript: str, question: str, max_words: int = CHAT_CONTEXT_WORDS) -> str:
    words = transcript.split()
    if len(words) <= max_words:
        return transcript
    chunk_size = 200
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    q_terms = {w for w in re.findall(r"[a-z0-9']+", question.lower()) if len(w) > 3}
    scored = sorted(((sum(c.lower().count(t) for t in q_terms), i) for i, c in enumerate(chunks)),
                    reverse=True)
    keep = sorted(i for _, i in scored[:max(1, max_words // chunk_size)])
    parts, prev = [], None
    for i in keep:
        if prev is not None and i != prev + 1:
            parts.append("[…]")
        parts.append(chunks[i])
        prev = i
    return ("(Excerpts of a long transcript, selected for relevance to the question)\n\n"
            + "\n".join(parts))


# llama-cpp is not safe for concurrent inference; RLock because the map-reduce
# path calls llm_complete while the /summarize generator already holds it.
_llm_lock = threading.RLock()

def llm_complete(prompt: str, max_tokens: int = 400, temperature: float = 0.1) -> str:
    if _llm is None:
        return ""
    with _llm_lock:
        out = _llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
    return out["choices"][0]["message"]["content"]


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title=APP_NAME)
# The UI is served same-origin; only allow the app's own origins rather than "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}"],
    allow_methods=["*"], allow_headers=["*"],
)


def _enqueue_job(audio_path: str, system_path: str = None, system_offset: float = 0.0) -> str:
    """Register a queued transcription job for an audio file and return its id."""
    job_id = str(uuid.uuid4())
    with _lock:
        _prune_jobs()
        _jobs[job_id] = {
            "id": job_id, "status": "queued", "phase": "queued", "progress": 0.0,
            "transcript": "", "segments": None, "error": None,
            "path": audio_path, "system_path": system_path, "system_offset": system_offset,
            "created": time.time(),
        }
    _job_queue.put(job_id)
    return job_id


def _save_upload(upload: UploadFile) -> str:
    suffix = ".webm" if "webm" in (upload.content_type or "") else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        shutil.copyfileobj(upload.file, f)
        return f.name


# Sync (`def`) endpoints run in Starlette's threadpool, so audio decoding never
# blocks the event loop — /health, /notes and job polls stay responsive.
@app.post("/transcribe")
def transcribe(audio: UploadFile = File(None),
               system_audio: UploadFile = File(None),
               system_offset: float = Form(0.0),
               system_capture_id: str = Form(""),
               mic_capture_id: str = Form("")):
    """The microphone (the user) arrives either as an uploaded `audio` blob
    (browser getUserMedia) or, in the desktop app, as a `mic_capture_id` from
    the native helper. The computer/remote channel arrives either as an
    uploaded `system_audio` blob (browser getDisplayMedia) or as a
    `system_capture_id` from the native ScreenCaptureKit helper. Either way it
    becomes the diarized "them" channel; the mic is "You"."""
    if _recognizer is None:
        raise HTTPException(503, "Still warming up the AI models — try again in a few seconds."
                                 if MODELS_LOADING else
                                 "Transcription models are not loaded — check the server logs.")
    if mic_capture_id:
        cap = _mic_caps.pop(mic_capture_id, None)
        if not (cap and cap["wav"].exists() and cap["wav"].stat().st_size > 1000):
            raise HTTPException(400, "The microphone capture produced no audio.")
        tmp = str(cap["wav"])
    elif audio is not None:
        tmp = _save_upload(audio)
    else:
        # Computer-audio-only recording: no mic at all. The system capture
        # becomes the primary (single-channel) audio and is diarized normally.
        cap = _sys_caps.pop(system_capture_id, None) if system_capture_id else None
        if not (cap and cap["wav"].exists() and cap["wav"].stat().st_size > 1000):
            raise HTTPException(400, "No audio was captured.")
        return {"job_id": _enqueue_job(str(cap["wav"]), None, 0.0)}
    sys_tmp = None
    if system_audio is not None:
        sys_tmp = _save_upload(system_audio)
    elif system_capture_id:
        cap = _sys_caps.pop(system_capture_id, None)
        if cap and cap["wav"].exists() and cap["wav"].stat().st_size > 1000:
            sys_tmp = str(cap["wav"])
    return {"job_id": _enqueue_job(tmp, sys_tmp, system_offset)}


# ── Crash-safe recording backups ───────────────────────────────────────────────
# While recording, the browser streams chunks here every few seconds. If the tab
# crashes or is refreshed mid-recording, the audio survives on disk and shows up
# in the UI as a recoverable recording. The backup is deleted once the finished
# recording has been handed to /transcribe.
RECORDINGS_DIR.mkdir(exist_ok=True)

def _rec_path(rec_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-fA-F-]{36}", rec_id):
        raise HTTPException(400, "Bad recording id")
    return RECORDINGS_DIR / f"{rec_id}.webm.part"


@app.post("/recordings/{rec_id}/chunk")
def recording_chunk(rec_id: str, audio: UploadFile = File(...)):
    with open(_rec_path(rec_id), "ab") as f:
        shutil.copyfileobj(audio.file, f)
    return {"ok": True}


@app.post("/recordings/{rec_id}/finish")
def recording_finish(rec_id: str):
    _rec_path(rec_id).unlink(missing_ok=True)
    return {"ok": True}


@app.get("/recordings")
def list_recordings():
    out = []
    for p in sorted(RECORDINGS_DIR.glob("*.webm.part")):
        st = p.stat()
        out.append({"id": p.name.split(".")[0], "bytes": st.st_size,
                    "modified": datetime.fromtimestamp(st.st_mtime).isoformat()})
    return out


@app.delete("/recordings/{rec_id}")
def discard_recording(rec_id: str):
    _rec_path(rec_id).unlink(missing_ok=True)
    return {"ok": True}


# ── Native system-audio capture (macOS ScreenCaptureKit helper) ─────────────────
# The bundled `itsmypa-audio` helper taps the computer's audio output natively
# (like Granola) instead of the browser's unreliable getDisplayMedia. Its WAV
# becomes the "system"/remote-speaker channel of the two-channel pipeline.
def _audiocap_bin():
    for p in (_BUNDLE_DIR / "itsmypa-audio",
              _REPO_DIR / "native" / "audiocap" / "itsmypa-audio"):
        if p.exists():
            return p
    return None

SYSTEM_AUDIO_AVAILABLE = _audiocap_bin() is not None
_sys_caps: dict = {}   # capture_id -> {"proc": Popen, "wav": Path}


@app.get("/system_audio/status")
def system_audio_status():
    """Screen Recording TCC state without prompting (CGPreflightScreenCaptureAccess)."""
    binp = _audiocap_bin()
    if binp is None:
        return {"available": False, "state": "unavailable"}
    try:
        out = subprocess.run([str(binp), "--sys-status"], capture_output=True, timeout=10)
        return {"available": True, "state": out.stdout.decode().strip() or "denied"}
    except Exception:
        return {"available": True, "state": "unknown"}


@app.post("/system_audio/start")
def system_audio_start():
    binp = _audiocap_bin()
    if binp is None:
        raise HTTPException(501, "Native computer-audio capture isn't available in this build.")
    cap_id = str(uuid.uuid4())
    wav = RECORDINGS_DIR / f"sys_{cap_id}.wav"
    proc = subprocess.Popen([str(binp), str(wav)], stderr=subprocess.PIPE)
    time.sleep(0.5)   # let it either start capturing or fail on permission
    if proc.poll() is not None:
        err = (proc.stderr.read() or b"").decode(errors="replace")
        if "permission" in err.lower():
            raise HTTPException(403, "macOS needs Screen Recording permission to capture computer audio. "
                                     "Open System Settings → Privacy & Security → Screen Recording, enable "
                                     "ItsMyPA, then try again.")
        raise HTTPException(500, f"Could not start computer-audio capture: {err[:200]}")
    _sys_caps[cap_id] = {"proc": proc, "wav": wav}
    return {"capture_id": cap_id}


@app.post("/system_audio/stop")
def system_audio_stop(body: dict):
    cap = _sys_caps.get((body or {}).get("capture_id"))
    if not cap:
        return {"ok": False}
    cap["proc"].terminate()
    try:
        cap["proc"].wait(timeout=5)
    except Exception:
        cap["proc"].kill()
    ok = cap["wav"].exists() and cap["wav"].stat().st_size > 1000
    return {"ok": True, "has_audio": ok}


# ── Native microphone capture (desktop app) ─────────────────────────────────
# The webview's getUserMedia engages macOS voice processing, which MUTES all
# other system audio while the mic is open — unusable for a meeting recorder.
# The desktop app records the mic through the same native helper instead
# (plain AVAudioEngine input tap: no voice processing, no muting).
_mic_caps: dict = {}   # capture_id -> {"proc": Popen, "wav": Path, "level": float, "err": str}


@app.get("/mic/status")
def mic_status():
    """Mic TCC state without prompting — lets the app check silently at launch."""
    binp = _audiocap_bin()
    if binp is None:
        return {"available": False, "state": "unavailable"}
    try:
        out = subprocess.run([str(binp), "--mic-status"], capture_output=True, timeout=10)
        return {"available": True, "state": out.stdout.decode().strip() or "undetermined"}
    except Exception:
        return {"available": True, "state": "undetermined"}


@app.post("/mic/start")
def mic_start():
    binp = _audiocap_bin()
    if binp is None:
        raise HTTPException(501, "Native microphone capture isn't available in this build.")
    cap_id = str(uuid.uuid4())
    wav = RECORDINGS_DIR / f"mic_{cap_id}.wav"
    proc = subprocess.Popen([str(binp), "--mic", str(wav)], stderr=subprocess.PIPE)
    state = {"proc": proc, "wav": wav, "level": 0.0, "err": ""}

    def _read_stderr():
        for line in iter(proc.stderr.readline, b""):
            if line.startswith(b"L "):
                try: state["level"] = float(line[2:])
                except ValueError: pass
            elif not line.startswith(b"capturing"):
                state["err"] = line.decode(errors="replace").strip()
    threading.Thread(target=_read_stderr, daemon=True).start()

    # The helper triggers the macOS mic prompt on first run, so give it time
    # to either start capturing or exit with an error.
    for _ in range(240):   # up to ~60s: the user may be reading the prompt
        if proc.poll() is not None:
            err = state["err"]
            if "permission" in err.lower():
                raise HTTPException(403, "macOS needs Microphone permission. Open System Settings → "
                                         "Privacy & Security → Microphone, enable ItsMyPA, then try again.")
            raise HTTPException(500, f"Could not start microphone capture: {err[:200]}")
        if wav.exists():   # first buffer written — capture is live
            break
        time.sleep(0.25)
    _mic_caps[cap_id] = state
    return {"capture_id": cap_id}


@app.post("/mic/stop")
def mic_stop(body: dict):
    cap = _mic_caps.get((body or {}).get("capture_id"))
    if not cap:
        return {"ok": False}
    cap["proc"].terminate()
    try:
        cap["proc"].wait(timeout=5)
    except Exception:
        cap["proc"].kill()
    ok = cap["wav"].exists() and cap["wav"].stat().st_size > 1000
    return {"ok": True, "has_audio": ok}


@app.post("/mic/pause")
def mic_pause(body: dict):
    cap = _mic_caps.get((body or {}).get("capture_id"))
    if not cap or cap["proc"].poll() is not None:
        return {"ok": False}
    cap["proc"].send_signal(signal_module.SIGUSR1)
    return {"ok": True}


@app.post("/mic/resume")
def mic_resume(body: dict):
    cap = _mic_caps.get((body or {}).get("capture_id"))
    if not cap or cap["proc"].poll() is not None:
        return {"ok": False}
    cap["proc"].send_signal(signal_module.SIGUSR2)
    return {"ok": True}


@app.get("/mic/level/{capture_id}")
def mic_level(capture_id: str):
    cap = _mic_caps.get(capture_id)
    return {"level": cap["level"] if cap else 0.0}


# Abruptly killed captures leave dangling audio-HAL clients in coreaudiod and
# can wedge system audio, so always SIGTERM the helpers and wait on shutdown.
@app.on_event("shutdown")
def _stop_all_system_captures():
    for cap in list(_sys_caps.values()) + list(_mic_caps.values()):
        try:
            cap["proc"].terminate()
            cap["proc"].wait(timeout=4)
        except Exception:
            try: cap["proc"].kill()
            except Exception: pass
    _sys_caps.clear()
    _mic_caps.clear()


@app.post("/recordings/{rec_id}/recover")
def recover_recording(rec_id: str):
    """Turn an orphaned recording backup into a normal transcription job."""
    src = _rec_path(rec_id)
    if not src.exists():
        raise HTTPException(404, "Recording not found")
    if _recognizer is None:
        raise HTTPException(503, "Transcription models are not loaded — check the server logs.")
    fd, tmp = tempfile.mkstemp(suffix=".webm")
    os.close(fd)
    shutil.move(str(src), tmp)
    return {"job_id": _enqueue_job(tmp)}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return {
        "id":         job["id"],
        "status":     job["status"],
        "phase":      job["phase"],
        "progress":   round(job["progress"], 3),
        # partial transcript streams in while the job runs (and survives errors)
        "transcript": job["transcript"],
        "segments":   job["segments"] if job["status"] == "done" else None,
        "diarized":   job["status"] == "done" and job["segments"] is not None,
        "error":      job["error"],
    }


@app.post("/summarize")
async def summarize(body: dict):
    transcript = (body.get("transcript") or "").strip()
    if not transcript:
        raise HTTPException(400, "No transcript provided")

    flavor = SUMMARY_TEMPLATES.get((body.get("template") or "general").strip(), "")
    word_count = len(transcript.split())

    def stream():
        if _llm is None:
            yield "## Executive Summary\n⚠️ The notes AI model isn't loaded — check Settings.\n\n## Key Decisions\n- No decisions were recorded.\n\n## Action Items\n- [ ] No action items were recorded. — @Unassigned\n\n## Topics Discussed\n- N/A"
            return
        try:
            with _llm_lock:
                if word_count > LONG_TRANSCRIPT_WORD_THRESHOLD:
                    yield "_Long meeting detected — condensing in stages, this takes a little longer…_\n\n"
                    chunks = chunk_words(transcript, CHUNK_WORD_SIZE)
                    partials = []
                    for i, c in enumerate(chunks):
                        p = MAP_PROMPT.replace("{{IDX}}", str(i + 1)).replace("{{TOTAL}}", str(len(chunks))).replace("{{CHUNK}}", c)
                        partials.append(llm_complete(p, max_tokens=400))
                    combined = "\n\n".join(partials)
                    final_prompt = REDUCE_PROMPT.replace("{{NOTES}}", combined)
                else:
                    final_prompt = SUMMARY_PROMPT.replace("{{TRANSCRIPT}}", transcript)
                if flavor:
                    final_prompt = flavor + "\n\n" + final_prompt

                output = _llm.create_chat_completion(
                    messages=[{"role": "user", "content": final_prompt}],
                    stream=True,
                    max_tokens=1200,
                    temperature=0.15,
                )
                for chunk in output:
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        yield delta["content"]
        except Exception as e:
            yield f"\n\n_Error generating summary: {e}_"

    return StreamingResponse(stream(), media_type="text/plain")


@app.post("/enhance")
async def enhance_notes(body: dict):
    """Expand the user's rough scratchpad notes with facts from the transcript."""
    transcript = (body.get("transcript") or "").strip()
    user_notes = (body.get("notes") or "").strip()
    if not transcript:
        raise HTTPException(400, "No transcript provided")
    if not user_notes:
        raise HTTPException(400, "No notes provided")

    # For very long meetings, keep the transcript chunks most relevant to what
    # the user actually wrote about.
    context = select_chat_context(transcript, user_notes)
    prompt = ENHANCE_PROMPT.replace("{{NOTES}}", user_notes[:4000]).replace("{{TRANSCRIPT}}", context)

    def stream():
        if _llm is None:
            yield "⚠️ Summarizer model not loaded."
            return
        try:
            with _llm_lock:
                output = _llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    stream=True, max_tokens=1400, temperature=0.15,
                )
                for chunk in output:
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        yield delta["content"]
        except Exception as e:
            yield f"\n\n_Error enhancing notes: {e}_"

    return StreamingResponse(stream(), media_type="text/plain")


@app.post("/generate/email")
async def generate_email(body: dict):
    """Draft a follow-up email from the meeting summary (or transcript)."""
    notes = (body.get("summary") or "").strip() or (body.get("transcript") or "").strip()
    if not notes:
        raise HTTPException(400, "No summary or transcript provided")

    prompt = EMAIL_PROMPT.replace("{{NOTES}}", notes[:12000])

    def stream():
        if _llm is None:
            yield "⚠️ Summarizer model not loaded."
            return
        try:
            with _llm_lock:
                output = _llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    stream=True, max_tokens=400, temperature=0.2,
                )
                for chunk in output:
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        yield delta["content"]
        except Exception as e:
            yield f"Error: {e}"

    return StreamingResponse(stream(), media_type="text/plain")


@app.post("/generate/title")
def generate_title(body: dict):
    """Suggest a short meeting title from the start of the transcript."""
    transcript = (body.get("transcript") or "").strip()
    if not transcript:
        raise HTTPException(400, "No transcript provided")
    words = " ".join(transcript.split()[:400])
    title = llm_complete(TITLE_PROMPT.replace("{{TRANSCRIPT}}", words),
                         max_tokens=24, temperature=0.2)
    title = title.strip().strip('"').strip().splitlines()[0][:80] if title.strip() else ""
    return {"title": title}


# Greetings and pleasantries must never reach the retrieval/LLM pipeline: a
# grounded prompt turns "hi" into an unsolicited meeting summary (or worse,
# a citation of a transcript line where somebody happened to say "hi").
_SMALLTALK_RE = re.compile(
    r"^(hi+|hii+|hello+|hey+|yo|sup|howdy|good\s*(morning|afternoon|evening)|"
    r"thanks?|thank\s*you|thx|ty|ok(ay)?|cool|nice|great|got\s*it|bye|goodbye|"
    r"test(ing)?|how\s*are\s*you\??)[\s!.?,]*$", re.I)

@app.post("/chat")
async def chat(body: dict):
    transcript = (body.get("transcript") or "").strip()
    question   = (body.get("question") or "").strip()
    history    = body.get("history", []) or []

    if not transcript:
        raise HTTPException(400, "No transcript provided")
    if not question:
        raise HTTPException(400, "No question provided")

    if _SMALLTALK_RE.match(question):
        return StreamingResponse(iter((
            "Hi! Ask me anything about this meeting — decisions, action items, "
            "or who said what.",)), media_type="text/plain")

    context = select_chat_context(transcript, question)
    system = (
        "You are a helpful assistant answering questions about a meeting. "
        "Answer ONLY using the meeting transcript provided below — never invent facts, names, or numbers. "
        "If the answer isn't in the transcript, say plainly that it wasn't discussed. "
        "If the user's message is small talk or not a question about the meeting, reply in one short "
        "friendly sentence and offer to answer questions about the meeting — never summarize unprompted. "
        "Reference speaker labels (e.g. 'Speaker 2') when relevant. Be concise and direct.\n\n"
        f"---\nTRANSCRIPT:\n{context}\n---"
    )

    messages = [{"role": "system", "content": system}]
    for turn in history[-6:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})

    def stream():
        if _llm is None:
            yield "⚠️ The notes AI model isn't loaded — check Settings."
            return
        try:
            with _llm_lock:
                output = _llm.create_chat_completion(messages=messages, stream=True, max_tokens=768, temperature=0.2)
                for chunk in output:
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        yield delta["content"]
        except Exception as e:
            yield f"Error: {e}"

    return StreamingResponse(stream(), media_type="text/plain")


# ── Ask across ALL meetings (hybrid BM25 retrieval + LLM answer) ─────────────
# The corpus (a person's meetings) is small enough that lexical BM25 over
# in-memory chunks beats a vector store: no embedding model to download, no
# index files to keep in sync. Semantic recall comes from the LLM expanding
# the question into synonyms/filters before retrieval.
_ASK_STOP = set(("the a an and or of to in on for is are was were be been am i we you he she it they "
                 "at with about that this these those my our your his her its from as by what which who "
                 "whom when where how did does do can could should would will").split())

def _ask_tokens(text: str) -> list:
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(w) > 1 and w not in _ASK_STOP]

_ask_index = {"version": None, "chunks": [], "df": Counter(), "avg": 1.0, "N": 0}
_ask_lock = threading.Lock()

def _ask_build_index() -> dict:
    chunks = []
    for n in _notes:
        if n.get("status") not in (None, "", "ready"):
            continue
        base = {"id": n["id"], "title": n.get("title") or "Untitled meeting",
                "date": (n.get("date") or "")[:10]}
        def add(kind, text, step=250):
            words = (text or "").split()
            if len(words) < 8:
                return
            for i in range(0, len(words), step):
                seg = " ".join(words[i:i + step])
                toks = _ask_tokens(base["title"] + " " + seg)
                if toks:
                    chunks.append({**base, "kind": kind, "text": seg,
                                   "tf": Counter(toks), "len": len(toks)})
        add("summary", n.get("summary"))
        add("notes", n.get("enhanced") or n.get("scratchpad"))
        add("transcript", n.get("transcript"))
    df = Counter()
    for c in chunks:
        df.update(c["tf"].keys())
    avg = (sum(c["len"] for c in chunks) / len(chunks)) if chunks else 1.0
    return {"chunks": chunks, "df": df, "avg": avg, "N": len(chunks)}

def _ask_retrieve(terms, date_from="", date_to="", people=(), k=10) -> list:
    with _ask_lock:
        if _ask_index["version"] != _notes_version:
            _ask_index.update(_ask_build_index(), version=_notes_version)
        chunks, df, avg, N = (_ask_index["chunks"], _ask_index["df"],
                              _ask_index["avg"], max(_ask_index["N"], 1))
    p_toks = set()
    for p in people:
        p_toks.update(_ask_tokens(str(p)))
    scored = []
    for c in chunks:
        if date_from and c["date"] and c["date"] < date_from: continue
        if date_to and c["date"] and c["date"] > date_to: continue
        s = 0.0
        for t in terms:
            f = c["tf"].get(t, 0)
            if not f: continue
            idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * (f * 2.5) / (f + 1.5 * (0.25 + 0.75 * c["len"] / avg))
        if not s: continue
        if p_toks and (p_toks & set(c["tf"])): s *= 1.6
        if c["kind"] == "summary": s *= 1.3   # summaries carry decisions & action items
        scored.append((s, c))
    scored.sort(key=lambda x: -x[0])
    out, per_note = [], Counter()
    for s, c in scored:
        if per_note[c["id"]] >= 3: continue   # spread context across meetings
        per_note[c["id"]] += 1
        out.append(c)
        if len(out) >= k: break
    return out

ASK_ANALYZE_PROMPT = (
    "Extract search filters from a question about the user's meeting-notes archive.\n"
    "Today is {{TODAY}}.\n"
    "Reply with ONLY a JSON object, no other text:\n"
    '{"keywords": [4-8 search words: the key nouns/verbs of the question PLUS close synonyms '
    '(e.g. for "front end" also "frontend", "UI", "web")], '
    '"people": [person names mentioned in the question, else []], '
    '"date_from": "YYYY-MM-DD or empty string", "date_to": "YYYY-MM-DD or empty string"}\n'
    'Set dates ONLY when the question names a specific day or period ("on July 8", "last week"); '
    "for a single day set date_from and date_to to that same day. Otherwise use empty strings.\n\n"
    "Question: {{QUESTION}}"
)

ASK_ANSWER_PROMPT = (
    "You answer questions from the user's saved meeting notes. Today is {{TODAY}}.\n"
    "Use ONLY the meeting excerpts below; each is headed [Meeting title — date].\n"
    "- Answer directly and concisely, in Markdown.\n"
    "- Mention which meeting each fact comes from naturally in prose (e.g. \"In the Q3 review "
    "on 2026-07-01, …\") — never reproduce the bracketed [ … ] header format itself.\n"
    "- If the excerpts don't answer the question, say so plainly — never invent details.\n"
    "- If the message is small talk rather than a question about meetings, reply in one short "
    "friendly sentence inviting a question — do not mention or cite any meeting.\n\n"
    "EXCERPTS:\n{{CONTEXT}}\n\nQuestion: {{QUESTION}}"
)

@app.post("/ask")
async def ask_meetings(body: dict):
    question = ((body or {}).get("question") or "").strip()
    if not question:
        raise HTTPException(400, "No question provided")

    if _SMALLTALK_RE.match(question):
        return StreamingResponse(iter((
            "Hi! Ask me anything about your saved meetings — decisions, action "
            "items, who said what, and when.",)), media_type="text/plain")

    def stream():
        if _llm is None:
            yield "⚠️ The notes AI model isn't loaded yet — check Settings."
            return
        today = datetime.now().strftime("%Y-%m-%d (%A)")
        filters = {}
        try:
            raw = llm_complete(ASK_ANALYZE_PROMPT.replace("{{TODAY}}", today)
                               .replace("{{QUESTION}}", question[:500]), max_tokens=220)
            m = re.search(r"\{.*\}", raw, re.S)
            if m:
                filters = json.loads(m.group(0))
        except Exception:
            pass
        terms = _ask_tokens(question)
        for kw in (filters.get("keywords") or []):
            terms += _ask_tokens(str(kw))
        terms = list(dict.fromkeys(terms))
        people = filters.get("people") or []
        d_from = str(filters.get("date_from") or "")[:10]
        d_to = str(filters.get("date_to") or "")[:10]

        chunks = _ask_retrieve(terms, d_from, d_to, people)
        if not chunks and (d_from or d_to):
            chunks = _ask_retrieve(terms, "", "", people)   # date guess may be off — retry unfiltered
        if not chunks:
            yield "I couldn't find anything about that in your saved meetings."
            return

        context = "\n\n".join(f"[{c['title']} — {c['date']}]\n{c['text']}" for c in chunks)
        prompt = (ASK_ANSWER_PROMPT.replace("{{TODAY}}", today)
                  .replace("{{CONTEXT}}", context).replace("{{QUESTION}}", question))
        try:
            with _llm_lock:
                output = _llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    stream=True, max_tokens=900, temperature=0.15)
                for chunk in output:
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        yield delta["content"]
        except Exception as e:
            yield f"\n\n_Error answering: {e}_"
            return
        # Trailer the UI parses into clickable source chips.
        seen, sources = set(), []
        for c in chunks:
            if c["id"] not in seen:
                seen.add(c["id"])
                sources.append({"id": c["id"], "title": c["title"], "date": c["date"]})
        yield "\n@@SOURCES@@" + json.dumps(sources[:6])

    return StreamingResponse(stream(), media_type="text/plain")


@app.post("/notes")
async def save_note(body: dict):
    note = {
        "id":            str(uuid.uuid4()),
        "title":         body.get("title", "Untitled Meeting"),
        "date":          datetime.now().isoformat(),
        "transcript":    body.get("transcript", ""),
        "summary":       body.get("summary", ""),
        "duration":      body.get("duration", 0),
        "segments":      body.get("segments"),
        "action_status": body.get("action_status", {}),
        "scratchpad":    body.get("scratchpad", ""),
        "enhanced":      body.get("enhanced", ""),
        "status":        body.get("status", "ready"),   # recording | processing | ready
    }
    with _lock:
        _notes.insert(0, note)
        save_notes()
    return note


@app.get("/notes")
async def get_notes():
    return _notes


@app.patch("/notes/{note_id}")
async def update_note(note_id: str, body: dict):
    allowed = {"title", "action_status", "summary", "transcript", "segments", "scratchpad", "enhanced", "status"}
    with _lock:
        for n in _notes:
            if n["id"] == note_id:
                n.update({k: v for k, v in body.items() if k in allowed})
                save_notes()
                return n
    raise HTTPException(404, "Note not found")


@app.delete("/notes/{note_id}")
async def delete_note(note_id: str):
    global _notes
    with _lock:
        _notes = [n for n in _notes if n["id"] != note_id]
        save_notes()
    return {"ok": True}


# ── Calendar (iCal/ICS feed) ────────────────────────────────────────────────────
# The user pastes their calendar's secret iCal address (Google/Outlook/Apple all
# provide one) — no OAuth, no account, read-only. The UI polls /calendar/upcoming
# and offers to record when a meeting starts.
CALENDAR_ICS_URL = os.getenv("CALENDAR_ICS_URL", "")
_cal_cache = {"fetched": 0.0, "url": "", "events": [], "error": None}

def _parse_ics(data: bytes, window_start: datetime, window_end: datetime) -> list:
    """Expand an ICS file (including recurring events) into occurrences in the window."""
    import icalendar
    import recurring_ical_events
    cal = icalendar.Calendar.from_ical(data)
    out = []
    for ev in recurring_ical_events.of(cal).between(window_start, window_end):
        start = ev.get("DTSTART").dt
        if not isinstance(start, datetime):
            continue  # skip all-day events — nothing to record
        end = ev.get("DTEND").dt if ev.get("DTEND") else start + timedelta(hours=1)
        out.append({"title": str(ev.get("SUMMARY", "Untitled meeting")),
                    "start": start.isoformat(), "end": end.isoformat()})
    out.sort(key=lambda x: x["start"])
    return out


@app.get("/calendar/upcoming")
def calendar_upcoming():
    if not CALENDAR_ICS_URL:
        return {"configured": False, "events": [], "error": None}
    now = time.time()
    if _cal_cache["url"] != CALENDAR_ICS_URL or now - _cal_cache["fetched"] > 300:
        try:
            import requests
            r = requests.get(CALENDAR_ICS_URL, timeout=15)
            r.raise_for_status()
            now_dt = datetime.now(timezone.utc)
            _cal_cache["events"] = _parse_ics(r.content, now_dt - timedelta(hours=1),
                                              now_dt + timedelta(hours=24))
            _cal_cache["error"] = None
        except Exception as e:
            _cal_cache["error"] = str(e)
            _cal_cache["events"] = []
        _cal_cache["fetched"] = now
        _cal_cache["url"] = CALENDAR_ICS_URL
    return {"configured": True, "events": _cal_cache["events"], "error": _cal_cache["error"]}


# ── Settings ───────────────────────────────────────────────────────────────────
# Settings persist to DATA_DIR/.env; model-affecting changes apply on next launch.
def _write_env(updates: dict):
    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
    out, seen = [], set()
    for line in lines:
        m = re.match(r"\s*([A-Z_]+)\s*=", line)
        if m and m.group(1) in updates:
            out.append(f"{m.group(1)}={updates[m.group(1)]}")
            seen.add(m.group(1))
        else:
            out.append(line)
    out += [f"{k}={v}" for k, v in updates.items() if k not in seen]
    ENV_FILE.write_text("\n".join(out) + "\n")


@app.get("/settings")
def get_settings():
    return {
        "whisper_model":     WHISPER_MODEL,
        "whisper_language":  WHISPER_LANGUAGE,
        "diarize":           DIARIZE,
        "num_speakers":      NUM_SPEAKERS,
        "diarize_threshold": DIARIZE_THRESHOLD,
        "calendar_ics_url":  CALENDAR_ICS_URL,
        "llm":               LLM_NAME,
    }


@app.post("/settings")
def update_settings(body: dict):
    updates = {}
    if "whisper_model" in body:
        m = str(body["whisper_model"])
        if m not in {"tiny", "base", "small", "medium", "large-v3"}:
            raise HTTPException(400, f"Unknown model '{m}'")
        updates["WHISPER_MODEL"] = m
    if "whisper_language" in body:
        lang = str(body["whisper_language"]).strip().lower()
        if lang and not re.fullmatch(r"[a-z]{2,3}", lang):
            raise HTTPException(400, "Language must be a 2-3 letter code (or empty for auto)")
        updates["WHISPER_LANGUAGE"] = lang
    if "diarize" in body:
        updates["DIARIZE"] = "1" if body["diarize"] else "0"
    if "num_speakers" in body:
        try:
            n = int(body["num_speakers"])
        except (TypeError, ValueError):
            raise HTTPException(400, "num_speakers must be an integer")
        if n != -1 and not 1 <= n <= 16:
            raise HTTPException(400, "num_speakers must be -1 (auto) or 1-16")
        updates["NUM_SPEAKERS"] = str(n)
    if "diarize_threshold" in body:
        try:
            t = float(body["diarize_threshold"])
        except (TypeError, ValueError):
            raise HTTPException(400, "diarize_threshold must be a number")
        if not 0.1 <= t <= 1.5:
            raise HTTPException(400, "diarize_threshold must be between 0.1 and 1.5")
        updates["DIARIZE_THRESHOLD"] = str(t)
    if "calendar_ics_url" in body:
        global CALENDAR_ICS_URL
        url = str(body["calendar_ics_url"]).strip()
        if url.startswith("webcal://"):
            url = "https://" + url[len("webcal://"):]
        if url and not url.startswith(("http://", "https://")):
            raise HTTPException(400, "Calendar URL must start with https:// (or webcal://)")
        updates["CALENDAR_ICS_URL"] = url
        CALENDAR_ICS_URL = url          # applies immediately, no restart
        _cal_cache["fetched"] = 0.0     # force refetch on next poll
    if not updates:
        raise HTTPException(400, "No recognized settings provided")
    _write_env(updates)
    # Only speech-model settings need a relaunch; the calendar applies live.
    MODEL_KEYS = {"WHISPER_MODEL", "WHISPER_LANGUAGE", "DIARIZE", "NUM_SPEAKERS", "DIARIZE_THRESHOLD"}
    return {"ok": True, "restart_required": bool(MODEL_KEYS & updates.keys())}


# ── Backup / restore ───────────────────────────────────────────────────────────
@app.get("/export")
def export_notes():
    payload = json.dumps({
        "app": "itsmypa", "version": 1,
        "exported": datetime.now().isoformat(), "notes": _notes,
    }, indent=2)
    fname = f"itsmypa-backup-{datetime.now():%Y%m%d}.json"
    return Response(content=payload, media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@app.post("/import")
def import_notes(backup: UploadFile = File(...)):
    try:
        data = json.loads(backup.file.read())
    except Exception:
        raise HTTPException(400, "Not a valid JSON backup file")
    incoming = data.get("notes") if isinstance(data, dict) else data
    if not isinstance(incoming, list):
        raise HTTPException(400, "No notes found in the file")

    NOTE_KEYS = ("id", "title", "date", "transcript", "summary", "duration", "segments",
                 "action_status", "scratchpad", "enhanced", "status")
    imported = skipped = 0
    with _lock:
        existing = {n["id"] for n in _notes}
        for n in incoming:
            if not isinstance(n, dict) or "id" not in n or "title" not in n:
                skipped += 1
                continue
            if n["id"] in existing:
                skipped += 1
                continue
            _notes.append({k: n.get(k) for k in NOTE_KEYS})
            existing.add(n["id"])
            imported += 1
        _notes.sort(key=lambda n: n.get("date") or "", reverse=True)
        save_notes()
    return {"ok": True, "imported": imported, "skipped": skipped}


@app.get("/health")
async def health():
    return {
        "whisper":     WHISPER_MODEL,
        "engine":      "sherpa-onnx",
        "loading":     MODELS_LOADING,
        "asr_ready":   _recognizer is not None,
        "ffmpeg":      FFMPEG_OK,
        "model":       LLM_NAME,
        "model_ready": _llm is not None,
        "diarize":     DIARIZE,
        "system_audio": SYSTEM_AUDIO_AVAILABLE,
    }


_TERMINAL_STATUSES = {"ready", "error", "skipped"}

@app.get("/setup/status")
async def setup_status():
    """Per-component download/load progress for first-run (and model-switch)
    setup, so the UI can show real progress and gate recording on actual
    readiness instead of a vague spinner."""
    with _setup_lock:
        components = [{"id": k, **v} for k, v in _setup_progress.items()]

    def pct(c):
        if c["status"] in ("ready", "skipped", "error"):
            return 100.0
        if c["status"] in ("downloading", "extracting") and c["total"]:
            return min(99.0, c["bytes"] / c["total"] * 100.0)
        return 0.0

    # Recording only truly needs transcription (asr) and the summarizer (llm) to
    # reach a terminal state — diarization is an optional enhancement and is
    # already handled gracefully elsewhere when unavailable.
    required = [c for c in components if c["id"] in ("asr", "llm")]
    ready = all(c["status"] in _TERMINAL_STATUSES for c in required)
    has_error = any(c["status"] == "error" for c in components)
    overall_pct = sum(pct(c) for c in components) / len(components) if components else 100.0

    return {
        "ready": ready,
        "has_error": has_error,
        "overall_percent": round(overall_pct, 1),
        "components": components,
    }


@app.get("/favicon.ico")
async def favicon():
    import base64
    ico = base64.b64decode(
        "AAABAAEAAQEAAAEAGAAoAAAAFgAAACgAAAABAAAAAgAAAAEAGAAAAAAA"
        "BAAAAAAAAAAAAAAAAAAAAAAA////AAAAAAAA"
    )
    return Response(content=ico, media_type="image/x-icon")


# ── Frontend ───────────────────────────────────────────────────────────────────
UI_FILE = _BUNDLE_DIR / "ui.html"


@app.get("/", response_class=HTMLResponse)
async def index():
    # The desktop app's WKWebView persists its HTTP cache across launches, so
    # without this an updated ui.html can keep showing the previous version
    # after a rebuild. There's no benefit to caching a localhost-only,
    # single-user document, so just always revalidate.
    return HTMLResponse(content=UI_FILE.read_text(), headers={
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma": "no-cache",
    })


# ── Entry point ────────────────────────────────────────────────────────────────
def _already_running() -> bool:
    """True if another ItsMyPA instance is serving on our port."""
    try:
        import requests
        r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=1.5)
        return r.ok and r.json().get("engine") == "sherpa-onnx"
    except Exception:
        return False


if __name__ == "__main__":
    # Single instance: if ItsMyPA is already running, just show it and exit —
    # a second copy would crash on the busy port.
    if _already_running():
        print(f"ItsMyPA is already running — opening http://localhost:{PORT}")
        webbrowser.open(f"http://localhost:{PORT}")
        sys.exit(0)

    print(f"""
╔══════════════════════════════════════════╗
║            ItsMyPA is ready!             ║
╠══════════════════════════════════════════╣
║  Open    → http://localhost:{PORT}         ║
║                                          ║
║  Whisper : {WHISPER_MODEL:<28}║
║  Engine  : sherpa-onnx (loading in background)║
║  Diarize : {'requested — loading' if DIARIZE else 'OFF (set DIARIZE=1)':<28}║
║  Data    : {str(DATA_DIR)[-29:]:<29} ║
╚══════════════════════════════════════════╝
{'' if HOST != '0.0.0.0' else '⚠️  HOST=0.0.0.0 — the app (and your meetings) are reachable from your whole network.'}""")
    # Packaged app: the server binds within a second — open the UI right away;
    # it shows a "warming up" state until the models finish loading.
    # Skip when embedded in the Tauri shell — that window IS the UI.
    if getattr(sys, "frozen", False) and os.getenv("ITSMYPA_EMBEDDED") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    try:
        uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
    except Exception as e:
        # Exit cleanly (no macOS crash dialog) with a readable reason in the log.
        print(f"❌ ItsMyPA could not start: {e}")
        sys.exit(1)
