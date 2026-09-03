#!/usr/bin/env bash
# Stops the IndicConformer ASR server started by start.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/run/server.pid"

if [[ ! -f "$PID_FILE" ]]; then
    echo "No PID file found — server does not appear to be running."
    exit 0
fi

PID="$(cat "$PID_FILE")"

if ! kill -0 "$PID" 2>/dev/null; then
    echo "Process $PID is not running. Cleaning up stale PID file."
    rm -f "$PID_FILE"
    exit 0
fi

echo "Stopping server (PID $PID) ..."
kill -TERM "$PID"

for _ in $(seq 1 20); do
    if ! kill -0 "$PID" 2>/dev/null; then
        rm -f "$PID_FILE"
        echo "Stopped."
        exit 0
    fi
    sleep 0.5
done

echo "Server did not stop gracefully, forcing ..."
kill -KILL "$PID" 2>/dev/null || true
rm -f "$PID_FILE"
echo "Stopped."
