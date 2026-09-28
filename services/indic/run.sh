#!/usr/bin/env bash
set -e
# Standalone playground mode (HTTPS on :7860). The orchestrator stack uses
# scripts/start_indic.sh instead. Server code lives in ../indic_server.py.
cd "$(dirname "$0")"
ROOT_DIR="$(cd .. && pwd)"
set -a; [ -f .env ] && . .env; set +a

# Ensure torch can locate libnvrtc-builtins (CUDA 13) shipped via pip wheels.
NVRTC_DIR=$(python3 -c "import os,glob; \
hits=glob.glob(os.path.expanduser('~/.local/lib/python*/site-packages/nvidia/cu13/lib')) \
     + glob.glob(os.path.expanduser('~/.local/lib/python*/site-packages/nvidia/cuda_nvrtc/lib')); \
print(hits[0] if hits else '')" 2>/dev/null || true)
if [[ -n "$NVRTC_DIR" ]]; then
  export LD_LIBRARY_PATH="$NVRTC_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  echo "Using NVRTC libs from: $NVRTC_DIR"
fi

PORT="${PORT:-7860}"
CERT="${SSL_CERT:-certs/cert.pem}"
KEY="${SSL_KEY:-certs/key.pem}"
if [[ -f "$CERT" && -f "$KEY" ]]; then
  echo "Starting HTTPS on :$PORT (cert=$CERT)"
  exec python3 -m uvicorn indic_server:app --app-dir "$ROOT_DIR" --host 0.0.0.0 --port "$PORT" \
      --ssl-certfile "$CERT" --ssl-keyfile "$KEY" --log-level info
else
  echo "Starting HTTP on :$PORT (no certs found at $CERT)"
  exec python3 -m uvicorn indic_server:app --app-dir "$ROOT_DIR" --host 0.0.0.0 --port "$PORT" --log-level info
fi
