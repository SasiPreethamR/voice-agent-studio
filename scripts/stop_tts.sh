#!/bin/bash
# Stop TTS server
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$ROOT_DIR/logs/tts_server.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    kill "$PID" 2>/dev/null && echo "Stopped TTS server (PID $PID)" || echo "Not running"
    rm -f "$PID_FILE"
else
    pkill -f "tts_server.py" && echo "Killed." || echo "No TTS server found."
fi
