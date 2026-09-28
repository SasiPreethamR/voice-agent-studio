#!/bin/bash
# TTS Server: Kokoro-82M | GPU $TTS_GPU (default 7) | Port 8011
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
. "$SCRIPT_DIR/_env.sh"
TTS_GPU="${TTS_GPU:-7}"

echo "Starting TTS server (Kokoro-82M) on GPU $TTS_GPU, port 8011..."
# Resolve the CUDA 13 libs bundled with torch 2.11+cu130 so torch's inductor/nvrtc
# JIT can find libnvrtc-builtins.so.13.0 (otherwise /v1/audio/speech 500s).
TORCH_CUDA_LIBS=$("$PYTHON" -c 'import nvidia.cu13 as m; print(list(m.__path__)[0] + "/lib")' 2>/dev/null || true)
[ -n "$TORCH_CUDA_LIBS" ] && [ -d "$TORCH_CUDA_LIBS" ] && echo "Using CUDA 13 libs at: $TORCH_CUDA_LIBS"
systemd-run --user --unit=voice-tts --collect \
  -p Restart=always -p RestartSec=5 \
  --setenv=CUDA_VISIBLE_DEVICES="$TTS_GPU" \
  --setenv=LD_LIBRARY_PATH="${TORCH_CUDA_LIBS}:${LD_LIBRARY_PATH}" \
  --working-directory="$SCRIPT_DIR" \
  -p StandardOutput=append:"$SCRIPT_DIR/logs/tts_server.log" \
  -p StandardError=append:"$SCRIPT_DIR/logs/tts_server.log" \
  "$PYTHON" "$SCRIPT_DIR/tts_server.py"

sleep 1
PID=$(systemctl --user show -p MainPID --value voice-tts 2>/dev/null)
echo "$PID" > logs/tts_server.pid
echo "Started as systemd --user unit 'voice-tts' (MainPID=$PID)"
echo "Logs:   tail -f $SCRIPT_DIR/logs/tts_server.log"
echo "Status: systemctl --user status voice-tts"
echo "Stop:   systemctl --user stop voice-tts"
echo "Test:   curl http://localhost:8011/health"
