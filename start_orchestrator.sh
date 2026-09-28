#!/bin/bash
# Voice Agent Orchestrator | Port 8080
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
. "$SCRIPT_DIR/_env.sh"
mkdir -p data/documents

echo "Starting Voice Agent Orchestrator on port 8080..."
echo "  UI:  http://localhost:8080"
echo "  API: http://localhost:8080/api/documents"
echo "  WS:  ws://localhost:8080/ws/voice"

systemd-run --user --unit=voice-orchestrator --collect \
  -p Restart=always -p RestartSec=3 \
  --working-directory="$SCRIPT_DIR" \
  -p StandardOutput=append:"$SCRIPT_DIR/logs/orchestrator.log" \
  -p StandardError=append:"$SCRIPT_DIR/logs/orchestrator.log" \
  "$PYTHON" "$SCRIPT_DIR/orchestrator.py"

sleep 1
PID=$(systemctl --user show -p MainPID --value voice-orchestrator 2>/dev/null)
echo "$PID" > logs/orchestrator.pid
echo "Started as systemd --user unit 'voice-orchestrator' (MainPID=$PID)"
echo "Logs:   tail -f $SCRIPT_DIR/logs/orchestrator.log"
echo "Status: systemctl --user status voice-orchestrator"
echo "Stop:   systemctl --user stop voice-orchestrator"
