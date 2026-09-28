#!/bin/bash
# STT Server: Faster-Whisper large-v3-turbo | GPU $STT_GPU (default 7) | Port 8010
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
. "$SCRIPT_DIR/_env.sh"

echo "Starting STT server (Faster-Whisper large-v3-turbo), port 8010..."
systemd-run --user --unit=voice-stt --collect \
  -p Restart=always -p RestartSec=5 \
  --setenv=CUDA_VISIBLE_DEVICES="${STT_GPU:-7}" \
  --working-directory="$ROOT_DIR" \
  -p StandardOutput=append:"$ROOT_DIR/logs/stt_server.log" \
  -p StandardError=append:"$ROOT_DIR/logs/stt_server.log" \
  "$PYTHON" "$ROOT_DIR/services/stt_server.py"

sleep 1
PID=$(systemctl --user show -p MainPID --value voice-stt 2>/dev/null)
echo "$PID" > logs/stt_server.pid
echo "Started as systemd --user unit 'voice-stt' (MainPID=$PID)"
echo "Logs:   tail -f $ROOT_DIR/logs/stt_server.log"
echo "Status: systemctl --user status voice-stt"
echo "Stop:   systemctl --user stop voice-stt"
echo "Test:   curl http://localhost:8010/health"
