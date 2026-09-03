#!/usr/bin/env bash
# One-time setup for the IndicConformer ASR project.
# Creates a virtualenv, installs pinned dependencies, and points the
# Hugging Face cache at a project-local folder (avoids permission issues
# with a shared/misconfigured ~/.cache).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "== IndicConformer setup =="

# Pick a Python interpreter (3.10+). Deliberately avoids picking up some
# *other* project's already-activated venv (common if your shell profile
# auto-activates one) by preferring known Homebrew locations first, then
# falling back to PATH lookup while rejecting anything that lives inside
# a venv directory.
PYTHON_BIN=""
for candidate in \
    /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.13 \
    /usr/local/bin/python3.11 /usr/local/bin/python3.12 /usr/local/bin/python3.13
do
    if [[ -x "$candidate" ]]; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    for name in python3.11 python3.12 python3.13 python3; do
        resolved="$(command -v "$name" 2>/dev/null || true)"
        if [[ -n "$resolved" && "$resolved" != *"/.venv/"* && "$resolved" != *"/venv/"* ]]; then
            PYTHON_BIN="$resolved"
            break
        fi
    done
fi

if [[ -z "$PYTHON_BIN" ]]; then
    echo "No suitable python3 found (only an active virtualenv's Python was on PATH)." >&2
    echo "Install Python 3.11+ (e.g. 'brew install python@3.11') and re-run, or deactivate" >&2
    echo "your current virtualenv first." >&2
    exit 1
fi

echo "Using interpreter: $PYTHON_BIN ($("$PYTHON_BIN" --version))"

if [[ ! -d .venv ]]; then
    "$PYTHON_BIN" -m venv .venv
    echo "Created .venv"
else
    echo ".venv already exists, reusing it"
fi

source .venv/bin/activate
python -m pip install --upgrade pip -q
pip install -q -r requirements.txt
echo "Dependencies installed."

# Point HF cache at a project-local folder instead of the default
# ~/.cache/huggingface, and persist it for every future `source .venv/bin/activate`.
if ! grep -q "HF_HOME" .venv/bin/activate; then
    echo "export HF_HOME=\"$SCRIPT_DIR/.hf_home\"" >> .venv/bin/activate
fi
mkdir -p .hf_home
export HF_HOME="$SCRIPT_DIR/.hf_home"

echo
echo "== Hugging Face access =="
if python -c "from huggingface_hub import whoami; whoami()" >/dev/null 2>&1; then
    WHOAMI="$(python -c "from huggingface_hub import whoami; print(whoami()['name'])")"
    echo "Already logged in as: $WHOAMI"
else
    cat <<'EOF'
Not logged in yet. This model is gated, so you need to:
  1. Visit https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
     while logged in, and click "Agree and access repository" (auto-approved).
  2. Run:  source .venv/bin/activate && hf auth login
     and paste an access token from https://huggingface.co/settings/tokens
EOF
fi

echo
echo "Setup complete. Next steps:"
echo "  ./start.sh                 # start the server (loads the model once)"
echo "  python transcribe.py test_audio.wav hi both   # or run one-shot CLI"
echo "  ./stop.sh                  # stop the server"
