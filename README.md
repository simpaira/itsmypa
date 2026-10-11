<div align="center">

# 🎙️ ItsMyPA

### Your meetings are being uploaded to someone else's computer. Stop that.

**ItsMyPA is an AI meeting notetaker that runs 100% on your Mac.**
Records, transcribes, tells speakers apart, and writes structured notes —
**no cloud, no account, no subscription, and your audio never leaves your machine.**

[**⬇️ Download for macOS (Apple Silicon)**](https://github.com/simpaira/itsmypa/releases/latest) · [Install guide](INSTALL.md) · [How it works](#how-it-works)

![ItsMyPA demo — record a meeting, get structured notes with speaker separation](docs/demo.gif)

*Real capture: recorded, transcribed, diarized, and summarized 100% on-device.*

</div>

---

## The uncomfortable truth about AI notetakers

Every popular meeting notetaker works the same way: it records your meeting — your
salary negotiation, your health call, your company's roadmap, your customer's data —
and **ships the audio to a server you don't control**, where it's transcribed,
stored, and governed by a privacy policy you didn't read.

That bot in your meeting? It's a microphone that reports to someone else.

ItsMyPA does everything those tools do, **on your own hardware**:

|  | ☁️ Cloud notetakers | 🔒 ItsMyPA |
|---|---|---|
| Where your audio goes | Their servers | Never leaves your Mac |
| Account required | Yes, always | No |
| Monthly subscription | $8–30/mo forever | Free |
| Works offline | No | Yes (after first-run model download) |
| Who can read your meetings | Them, their vendors, subpoenas | You |
| A bot joins your call | Usually | Never — captures audio locally |

## What it does

- **One-button capture** — records your mic *and* the computer's meeting audio
  (Zoom, Meet, Teams, anything) natively. No bot joining your call, no
  tab-sharing dialog.
- **Transcription with speaker separation** — Whisper + speaker diarization tell
  "You" apart from Speaker 2 and 3, fully locally.
- **Structured AI notes** — Executive Summary, Key Decisions, Action Items with
  owners and checkboxes. Not a wall of text.
- **Granola-style scratchpad** — jot rough bullets during the meeting; AI expands
  each line with the decisions, numbers, and quotes from the transcript.
- **Ask your meeting anything** — chat with the transcript. "What did we agree
  about pricing?" Answered locally.
- **Note styles** — Standup / 1:1 / Sales call / Interview templates.
- **Follow-up email in one click**, auto-titles, Markdown export, calendar
  record-prompts, full-text search.

All of it powered by on-device models (Whisper, pyannote, TitaNet, a local LLM).
The first launch downloads them (~2.6 GB, one time). After that: airplane mode works.

## Get it

**Requirements:** Apple Silicon Mac (M1 or newer), macOS 14.6+, 16 GB RAM recommended.

1. [Download the latest DMG](https://github.com/simpaira/itsmypa/releases/latest)
2. Drag ItsMyPA to Applications
3. **First launch:** macOS will complain because we haven't paid Apple's $99/yr
   signing fee (we're indie — the code is public instead). Go to
   **System Settings → Privacy & Security → "Open Anyway"**. Full walkthrough
   with screenshots: [INSTALL.md](INSTALL.md)

Or with Homebrew (no Gatekeeper dance):

```bash
brew install --cask simpaira/tap/itsmypa
```

## "Unsigned? Why should I trust it?"

Fair question — the answer is better than a certificate: **the source code is
right here.** Read it, build it yourself, or watch its network traffic (spoiler:
after the one-time model download, there isn't any). A $99 Apple certificate
proves someone paid Apple $99. Open source proves what the app actually does.

Like the project? A ⭐ on this repo is free and helps more people find it.

---

## How it works

ItsMyPA is one small native app (Tauri + Rust, ~30 MB) — no Python, no local
web server, nothing listening on the network.

- **Capture** (`src-tauri/capture`): your microphone and the computer's audio,
  recorded together on one clock via cpal — Core Audio process taps on macOS
  14.6+ and WASAPI loopback on Windows. Pausing pauses both; switching
  headphones mid-meeting reconnects; WAVs are crash-safe on disk.
- **Speech** (`src-tauri/engine`): **sherpa-onnx** runs Whisper (int8),
  Silero VAD and pyannote segmentation-3.0 + NeMo TitaNet speaker embeddings.
  Your mic is labelled "You"; the computer audio is diarized into Speaker 2, 3…
- **Notes**: a local GGUF model via **llama.cpp** (Metal on Apple Silicon)
  writes the summary, chat answers, follow-up email and "enhance my notes".
  The models are fixed — Whisper "small" and Qwen2.5 3B, the best balance for an
  average computer — so there's nothing to choose or tune.
- **Storage**: meetings live in one SQLite file with full-text search.

Long recordings are transcribed as background jobs; if ItsMyPA quits mid-
meeting, the next launch offers to recover the recording.

## Run from source

Needs Rust, the Tauri CLI (`cargo install tauri-cli`) and CMake.

```bash
cd src-tauri
cargo tauri dev                                   # run the app
cargo test -p itsmypa-engine -p itsmypa-capture   # tests
```

Models download on first launch into the app-data folder (macOS:
`~/Library/Application Support/ItsMyPA`, Windows: `%LOCALAPPDATA%\ItsMyPA`) and
are verified by checksum. Settings (spoken language, speaker detection,
calendar) are in the app's Settings screen.

## Build the desktop app

```bash
./package/build_shell.sh          # → dist/ItsMyPA.dmg
```

## Support the project

ItsMyPA is free. If it saved you a subscription, you can
[**buy me a coffee on Ko-fi** ☕](https://ko-fi.com/simpaira) — the first goal
is Apple's $99 signing fee, so the security warning at install goes away for
everyone. Every coffee is a brick in that wall.
