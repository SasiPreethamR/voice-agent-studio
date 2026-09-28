#!/bin/bash
# Stop ALL voice agent services
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

echo "Stopping all services..."
bash stop_orchestrator.sh
bash stop_indic.sh
bash stop_stt.sh
bash stop_tts.sh
bash stop_gemma4.sh
echo "All stopped."
