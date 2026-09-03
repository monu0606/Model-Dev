#!/usr/bin/env bash
# Starts the IndicConformer ASR server in the background.
# Safe to re-run: does nothing if the server is already up.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8420}"
RUN_DIR="$SCRIPT_DIR/run"
PID_FILE="$RUN_DIR/server.pid"
LOG_FILE="$RUN_DIR/server.log"
HEALTH_URL="http://127.0.0.1:${PORT}/health"

mkdir -p "$RUN_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Server already running (PID $(cat "$PID_FILE")) at $HEALTH_URL"
    exit 0
fi

if [[ ! -d "$SCRIPT_DIR/.venv" ]]; then
    echo "No .venv found. Run setup.sh first." >&2
    exit 1
fi

source "$SCRIPT_DIR/.venv/bin/activate"

echo "Starting IndicConformer server on port $PORT ..."
nohup python -m uvicorn server:app --host 127.0.0.1 --port "$PORT" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "Waiting for model to load (first load after a fresh cache can take a few minutes; warm loads take ~10-15s) ..."
for _ in $(seq 1 180); do
    if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "Server process died during startup. Check $LOG_FILE for details." >&2
        rm -f "$PID_FILE"
        exit 1
    fi
    if curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null | grep -q "200"; then
        echo "Server is up: $HEALTH_URL"
        echo "API docs:      http://127.0.0.1:${PORT}/docs"
        echo "PID:           $(cat "$PID_FILE")  (logs: $LOG_FILE)"
        exit 0
    fi
    sleep 1
done

echo "Timed out waiting for the server to become healthy. Check $LOG_FILE." >&2
exit 1
