#!/bin/bash
# Indic Speech Server: AI4Bharat IndicConformer ASR + IndicParler TTS | GPU $INDIC_GPU (default 6) | Port 7862
# Handles 22 Indic languages (as, bn, brx, doi, gu, hi, kn, kok, ks, mai, ml,
# mni, mr, ne, or, pa, sa, sat, sd, ta, te, ur).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INDIC_DIR="$ROOT_DIR/services/indic"
cd "$ROOT_DIR"
. "$SCRIPT_DIR/_env.sh"

# Resolve NVRTC libs the same way services/indic/run.sh does, so torch can find
# libnvrtc-builtins matching the installed CUDA wheels.
NVRTC_DIR=$("$PYTHON" -c "import os,glob; \
hits=glob.glob(os.path.expanduser('~/.local/lib/python*/site-packages/nvidia/cu13/lib')) \
     + glob.glob(os.path.expanduser('~/.local/lib/python*/site-packages/nvidia/cuda_nvrtc/lib')); \
print(hits[0] if hits else '')" 2>/dev/null || true)

PORT="${INDIC_PORT:-7862}"

# Pinned-deps shim: parler-tts requires transformers==4.46.1, but vLLM/Gemma
# need transformers>=4.56 in the global env. Keep an isolated copy in indic/pyenv/
# and prepend it to PYTHONPATH so only this server sees the older pin.
PYENV_DIR="$INDIC_DIR/pyenv"

echo "Starting Indic Speech server (IndicConformer ASR + IndicParler TTS), port $PORT..."
SYSTEMD_ENV=(-E "INDIC_PORT=$PORT")
[ -n "$HF_TOKEN" ] && SYSTEMD_ENV+=(-E "HF_TOKEN=$HF_TOKEN")
if [[ -n "$NVRTC_DIR" ]]; then
  SYSTEMD_ENV+=(-E "LD_LIBRARY_PATH=$NVRTC_DIR")
fi
if [[ -d "$PYENV_DIR" ]]; then
  SYSTEMD_ENV+=(-E "PYTHONPATH=$PYENV_DIR")
fi
# IndicParler TTS runs on this GPU; IndicConformer ASR runs on CPU (see services/indic_server.py).
SYSTEMD_ENV+=(-E "CUDA_VISIBLE_DEVICES=${INDIC_GPU:-6}")

systemd-run --user --unit=voice-indic --collect \
  -p Restart=always -p RestartSec=5 \
  --working-directory="$ROOT_DIR" \
  -p StandardOutput=append:"$ROOT_DIR/logs/indic_server.log" \
  -p StandardError=append:"$ROOT_DIR/logs/indic_server.log" \
  "${SYSTEMD_ENV[@]}" \
  "$PYTHON" "$ROOT_DIR/services/indic_server.py"

sleep 1
PID=$(systemctl --user show -p MainPID --value voice-indic 2>/dev/null)
echo "$PID" > logs/indic_server.pid
echo "Started as systemd --user unit 'voice-indic' (MainPID=$PID)"
echo "Logs:   tail -f $ROOT_DIR/logs/indic_server.log"
echo "Status: systemctl --user status voice-indic"
echo "Stop:   systemctl --user stop voice-indic"
echo "Test:   curl http://localhost:$PORT/health"
