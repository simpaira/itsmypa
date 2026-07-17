#!/usr/bin/env bash
# Build the ItsMyPA Python engine as a console server bundle that the Tauri
# shell spawns as a subprocess. Output: src-tauri/pybundle/itsmypa-server/
set -e
cd "$(dirname "$0")/.."

PY=.venv/bin/python
$PY -m PyInstaller --version >/dev/null 2>&1 || { echo "pip install -r requirements-dev.txt first"; exit 1; }

# Compile the native system-audio capture helper (macOS/ScreenCaptureKit).
if [ "$(uname)" = "Darwin" ]; then
  echo "Compiling native audio-capture helper…"
  ( cd native/audiocap && swiftc -O -framework ScreenCaptureKit -framework AVFoundation -o itsmypa-audio main.swift )
  AUDIOCAP_ADD=(--add-binary "$(pwd)/native/audiocap/itsmypa-audio:.")
else
  AUDIOCAP_ADD=()
fi

rm -rf build src-tauri/pybundle

# --exclude-module guards: onnxruntime ships benchmarking submodules
# (onnxruntime.transformers etc.) that import the full PyTorch/HF stack when it
# happens to be installed in the build env, ballooning the bundle by ~600 MB.
EXCLUDES=(
  torch torchvision torchaudio transformers tokenizers
  numba llvmlite scipy sklearn pandas matplotlib nltk
  av grpc tiktoken PIL
  onnxruntime.transformers onnxruntime.training
  onnxruntime.quantization onnxruntime.datasets
)
EXCLUDE_ARGS=()
for m in "${EXCLUDES[@]}"; do EXCLUDE_ARGS+=(--exclude-module "$m"); done

$PY -m PyInstaller \
  --noconfirm \
  --console \
  --name itsmypa-server \
  --distpath src-tauri/pybundle \
  --workpath build \
  --add-data "$(pwd)/ui.html:." \
  "${AUDIOCAP_ADD[@]}" \
  --collect-all sherpa_onnx \
  --collect-all llama_cpp \
  --collect-all uvicorn \
  --collect-all imageio_ffmpeg \
  --hidden-import multiprocessing \
  "${EXCLUDE_ARGS[@]}" \
  app.py

echo "Server bundle: src-tauri/pybundle/itsmypa-server/"
du -sh src-tauri/pybundle/itsmypa-server 2>/dev/null
