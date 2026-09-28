#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/logs/orchestrator.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    kill "$PID" 2>/dev/null && echo "Stopped orchestrator (PID $PID)" || echo "Not running"
    rm -f "$PID_FILE"
else
    pkill -f "orchestrator.py" && echo "Killed." || echo "No orchestrator found."
fi
