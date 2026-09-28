#!/bin/bash
# Start ALL voice agent services
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════╗"
echo "║     Voice Agent — Full Stack         ║"
echo "╚══════════════════════════════════════╝"
echo ""

echo "[1/5] Gemma 4 31B (LLM) — :8003"
bash start_gemma4.sh
echo ""

echo "[2/5] Faster-Whisper (STT) — :8010"
bash start_stt.sh
echo ""

echo "[3/5] Kokoro-82M (TTS) — :8011"
bash start_tts.sh
echo ""

echo "[4/5] Indic Speech (AI4Bharat) — :7862"
bash start_indic.sh
echo ""

echo "[5/5] Orchestrator — :8080"
bash start_orchestrator.sh
echo ""

echo "════════════════════════════════════════"
echo "  UI ready at  →  http://localhost:8080"
echo "════════════════════════════════════════"
