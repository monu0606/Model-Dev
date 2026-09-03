#!/usr/bin/env bash
# Transcribe audio with Gemma 4's audio-capable variants (E2B / E4B) via
# llama.cpp + GGUF. Only E2B and E4B are wired up here - the 12B variant
# also supports audio but is deliberately excluded, it's too large for
# this machine.
#
# Why GGUF + llama.cpp instead of transformers: on an 8GB-RAM machine,
# even E2B's raw safetensors weights (10.25GB) don't fit in memory.
# Quantized GGUF weights (Q4_K_M, ~3-5GB) do.
#
# Why CPU-only (-ngl 0): with -ngl 99 (full GPU/Metal offload), the audio
# encoder's decode step overflows Metal's GPU buffer allocator on this
# machine (kIOGPUCommandBufferCallbackErrorOutOfMemory), even with RAM
# free system-wide. Plain CPU inference (regular mmap'd memory, no fixed
# GPU working-set ceiling) works reliably instead, just slower.
#
# GGUF weights are never committed to git (see .gitignore) - they're
# re-downloaded here (or via bootstrap.sh) on whatever machine needs them.
#
# Usage:
#   ./gemma4_transcribe.sh <audio_file> [e2b|e4b]
#   ./gemma4_transcribe.sh --download-only [e2b|e4b]   # fetch weights, skip inference
#
# Example:
#   ./gemma4_transcribe.sh test_audio.wav e2b
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DOWNLOAD_ONLY=0
if [[ "${1:-}" == "--download-only" ]]; then
    DOWNLOAD_ONLY=1
    shift
    SIZE="${1:-e2b}"
else
    AUDIO_FILE="${1:?Usage: $0 <audio_file> [e2b|e4b]  (or: $0 --download-only [e2b|e4b])}"
    SIZE="${2:-e2b}"
fi

case "$SIZE" in
    e2b) REPO="unsloth/gemma-4-E2B-it-GGUF"; MODEL_FILE="gemma-4-E2B-it-Q4_K_M.gguf" ;;
    e4b) REPO="unsloth/gemma-4-E4B-it-GGUF"; MODEL_FILE="gemma-4-E4B-it-Q4_K_M.gguf" ;;
    *) echo "Unknown size '$SIZE'. Choose one of: e2b, e4b" >&2; exit 1 ;;
esac

MODEL_DIR="$SCRIPT_DIR/gguf_models/$SIZE"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"
MMPROJ_PATH="$MODEL_DIR/mmproj-BF16.gguf"

if [[ ! -f "$MODEL_PATH" || ! -f "$MMPROJ_PATH" ]]; then
    echo "Downloading $SIZE GGUF weights + mmproj to $MODEL_DIR ..."
    mkdir -p "$MODEL_DIR"
    if [[ ! -d "$SCRIPT_DIR/.venv" ]]; then
        echo "No .venv found. Run setup.sh first." >&2
        exit 1
    fi
    source "$SCRIPT_DIR/.venv/bin/activate"
    export HF_HUB_DISABLE_XET=1
    python3 - "$REPO" "$MODEL_DIR" "$MODEL_FILE" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
repo, local_dir, model_file = sys.argv[1], sys.argv[2], sys.argv[3]
for fname in (model_file, "mmproj-BF16.gguf"):
    print(f"downloading {fname} ...")
    hf_hub_download(repo_id=repo, filename=fname, local_dir=local_dir)
EOF
else
    echo "$SIZE weights already present, skipping download."
fi

if [[ "$DOWNLOAD_ONLY" == "1" ]]; then
    exit 0
fi

if ! command -v llama-mtmd-cli >/dev/null 2>&1; then
    echo "llama-mtmd-cli not found. Install it with: brew install llama.cpp" >&2
    exit 1
fi

SYSTEM_PROMPT="You are a speech transcription engine. Transcribe the given \
audio exactly as spoken, in its original language. Output only the \
transcription - no commentary, no chain-of-thought, no preamble, no \
newlines. Write numbers as digits (e.g. 1.7, not one point seven)."

llama-mtmd-cli \
    -m "$MODEL_PATH" \
    --mmproj "$MMPROJ_PATH" \
    --audio "$AUDIO_FILE" \
    -sys "$SYSTEM_PROMPT" \
    -p "Transcribe this audio." \
    -ngl 0 --jinja \
    --temp 1.0 --top-k 64 --top-p 0.95
