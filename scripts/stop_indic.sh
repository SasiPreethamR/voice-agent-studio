#!/bin/bash
# Stop Indic Speech server
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
systemctl --user stop voice-indic 2>/dev/null && echo "Stopped voice-indic" || true
PID_FILE="$ROOT_DIR/logs/indic_server.pid"
[ -f "$PID_FILE" ] && rm -f "$PID_FILE"
pkill -f "indic_server.py" 2>/dev/null || true
echo "Done."
