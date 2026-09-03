#!/usr/bin/env bash
# One-shot setup for a fresh clone on a new machine: Python env +
# IndicConformer dependencies, llama.cpp (for Gemma 4 audio), and Gemma 4
# GGUF weights.
#
# Model weights are never committed to git (see .gitignore) - IndicConformer
# (~2.56GB, fetched by start.sh on first run) and the Gemma 4 GGUF weights
# fetched here (~4.1GB for E2B, ~6GB for E4B) are re-downloaded fresh on
# every machine that needs them, so cloning the repo stays fast and small.
#
# Usage:
#   ./bootstrap.sh              # env + IndicConformer deps + llama.cpp + Gemma 4 E2B
#   ./bootstrap.sh e2b e4b      # also fetch E4B
#   ./bootstrap.sh --no-gemma4  # skip llama.cpp + GGUF downloads entirely
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SIZES=("$@")
SKIP_GEMMA4=0
if [[ "${1:-}" == "--no-gemma4" ]]; then
    SKIP_GEMMA4=1
    SIZES=()
elif [[ ${#SIZES[@]} -eq 0 ]]; then
    SIZES=(e2b)
fi

echo "== 1/3: Python environment + IndicConformer dependencies =="
./setup.sh

if [[ "$SKIP_GEMMA4" == "1" ]]; then
    echo
    echo "Skipping Gemma 4 setup (--no-gemma4). Bootstrap complete."
    echo "  ./start.sh   # start the IndicConformer web UI"
    exit 0
fi

echo
echo "== 2/3: llama.cpp (for Gemma 4 audio) =="
if command -v llama-mtmd-cli >/dev/null 2>&1; then
    echo "llama-mtmd-cli already installed: $(command -v llama-mtmd-cli)"
elif command -v brew >/dev/null 2>&1; then
    brew install llama.cpp
else
    echo "Homebrew not found - install it from https://brew.sh, then re-run" >&2
    echo "this script (or run 'brew install llama.cpp' yourself)." >&2
    exit 1
fi

echo
echo "== 3/3: Gemma 4 GGUF weights (${SIZES[*]}) =="
for size in "${SIZES[@]}"; do
    ./gemma4_transcribe.sh --download-only "$size"
done

echo
echo "Bootstrap complete. Next steps:"
echo "  ./start.sh                                  # web UI: IndicConformer + Gemma 4"
echo "  ./gemma4_transcribe.sh test_audio.wav e2b    # or test Gemma 4 from the CLI"
