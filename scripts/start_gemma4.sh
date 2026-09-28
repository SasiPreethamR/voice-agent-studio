#!/bin/bash
# Gemma 4 31B vLLM Server | Port: 8003
# Runs on two H200s by default to support long-context serving. Override
# LLM_GPUS/TENSOR_PARALLEL_SIZE if you need a different layout.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
. "$SCRIPT_DIR/_env.sh"

MODEL="${LLM_MODEL:-google/gemma-4-31B-it}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
GPU_DEVICES="${LLM_GPUS:-${CUDA_VISIBLE_DEVICES:-4,5}}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-200000}"
HF_ACCESS_TOKEN="${HF_TOKEN:-}"
if [ -z "$HF_ACCESS_TOKEN" ]; then
  echo "Note: HF_TOKEN is not set; gated models download only if already cached."
fi

TOOL_ARGS=()
if [[ "$MODEL" == *gemma-4* ]]; then
  TOOL_ARGS=(
    --enable-auto-tool-choice
    --tool-call-parser gemma4
    --language-model-only
    --skip-mm-profiling
  )
fi

echo "Starting $MODEL vLLM server on GPU(s) $GPU_DEVICES..."

EXTRA_ENV=()
# Optional NVML shim: if the userspace libnvidia-ml.so is newer than the loaded
# kernel module (e.g. after a driver upgrade without reboot), vLLM's CUDA probe
# fails. Put a matching libnvidia-ml.so in NVML_SHIM (default
# ~/.local/lib/nvml580126, used only if it exists) to prepend it.
NVML_SHIM="${NVML_SHIM:-$HOME/.local/lib/nvml580126}"
[ -d "$NVML_SHIM" ] && EXTRA_ENV+=(--setenv=LD_LIBRARY_PATH="$NVML_SHIM")
# Optional Python headers for Triton/JIT builds on hosts without python3-dev.
PYDEV_INCLUDE="${PYDEV_INCLUDE:-/tmp/pydev/usr/include}"
if [ -d "$PYDEV_INCLUDE" ]; then
  PYVER=$("$PYTHON" -c 'import sys; print(f"python{sys.version_info[0]}.{sys.version_info[1]}")')
  EXTRA_ENV+=(--setenv=CPATH="$PYDEV_INCLUDE/$PYVER:$PYDEV_INCLUDE" --setenv=C_INCLUDE_PATH="$PYDEV_INCLUDE/$PYVER:$PYDEV_INCLUDE")
fi
[ -n "$HF_ACCESS_TOKEN" ] && EXTRA_ENV+=(--setenv=HF_TOKEN="$HF_ACCESS_TOKEN" --setenv=HUGGING_FACE_HUB_TOKEN="$HF_ACCESS_TOKEN")

# Run under systemd --user so it survives SSH/VS Code disconnects (requires `loginctl enable-linger`).
systemd-run --user --unit=gemma4-vllm --collect \
  -p Restart=always -p RestartSec=5 -p SuccessExitStatus=SIGKILL \
  --setenv=CUDA_VISIBLE_DEVICES="$GPU_DEVICES" \
  --setenv=VLLM_USE_DEEP_GEMM=0 \
  "${EXTRA_ENV[@]}" \
  --working-directory="$ROOT_DIR" \
  -p StandardOutput=append:"$ROOT_DIR/logs/gemma_vllm.log" \
  -p StandardError=append:"$ROOT_DIR/logs/gemma_vllm.log" \
  "$PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  "${TOOL_ARGS[@]}" \
  --host 0.0.0.0 \
  --port 8003 \
  --trust-remote-code

sleep 2
PID=$(systemctl --user show -p MainPID --value gemma4-vllm 2>/dev/null)
echo "$PID" > logs/gemma_vllm.pid
echo "Server started as systemd --user unit 'gemma4-vllm' (MainPID=$PID)"
echo "Logs: tail -f $ROOT_DIR/logs/gemma_vllm.log"
echo "Status: systemctl --user status gemma4-vllm"
echo "Stop:   systemctl --user stop gemma4-vllm"
echo "Test:   curl http://localhost:8003/v1/models"
