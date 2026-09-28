#!/bin/bash
# Stop STT server
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/logs/stt_server.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    kill "$PID" 2>/dev/null && echo "Stopped STT server (PID $PID)" || echo "Not running"
    rm -f "$PID_FILE"
else
    pkill -f "stt_server.py" && echo "Killed." || echo "No STT server found."
fi
