#!/usr/bin/env bash
# ItsMyPA launcher: creates a venv, installs deps, and starts the server.
set -e
cd "$(dirname "$0")"

# Prefer pyenv-selected interpreter when available so we get a Python with
# prebuilt wheels for sherpa-onnx/llama-cpp (e.g., 3.11/3.12) instead of system 3.14.
if command -v pyenv >/dev/null 2>&1; then
  export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
  export PATH="$PYENV_ROOT/bin:$PATH"
  eval "$(pyenv init -)"
  PYTHON_BIN="$(pyenv which python)"
else
  PYTHON_BIN="$(command -v python3)"
fi

if [ -z "$PYTHON_BIN" ]; then
  echo "❌ Could not find a Python interpreter."
  exit 1
fi

if [ -d ".venv" ]; then
  VENV_VER="$(.venv/bin/python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null || echo unknown)"
  TARGET_VER="$($PYTHON_BIN -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
  if [ "$VENV_VER" != "$TARGET_VER" ]; then
    echo "Recreating .venv (was Python $VENV_VER, target is $TARGET_VER)…"
    rm -rf .venv
  fi
fi

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment with $($PYTHON_BIN --version 2>/dev/null)…"
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing dependencies (first run may take a few minutes)…"
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Speech models (Whisper int8, Silero VAD, pyannote segmentation, TitaNet)
# auto-download to ./models on first launch — no Hugging Face token needed.
echo "Starting ItsMyPA → http://localhost:8765"
python app.py
