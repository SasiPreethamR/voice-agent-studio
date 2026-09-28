#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$ROOT_DIR/logs/gemma_vllm.pid"

if systemctl --user is-active --quiet gemma4-vllm 2>/dev/null; then
    echo "Stopping Gemma 4 31B systemd unit (gemma4-vllm)..."
    systemctl --user stop gemma4-vllm
    rm -f "$PID_FILE" "$ROOT_DIR/logs/gemma4_31b.pid"
    echo "Stopped."
    exit 0
fi

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping Gemma 4 31B server (PID $PID)..."
        kill "$PID"
        rm -f "$PID_FILE"
        echo "Stopped."
    else
        echo "Process $PID not running. Cleaning up PID file."
        rm -f "$PID_FILE"
    fi
else
    echo "No PID file found. Trying to find vllm process..."
    pkill -f "gemma-4-31B-it" && echo "Killed." || echo "No process found."
fi
