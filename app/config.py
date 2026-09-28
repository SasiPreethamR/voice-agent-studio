"""
Shared configuration for the voice agent stack.

Service endpoints, the local LLM model name, audio formats and Gemini (BYOK)
defaults live here so the orchestrator, document store and provider layer all
agree. Every value can be overridden through the environment.
"""

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent                 # repo root: .env, data/, logs/, cert.pem live here
STATIC_DIR = APP_DIR / "static"           # Studio UI + operator console
DATA_DIR = ROOT_DIR / "data"              # agents, tools, handoffs, documents (runtime state)
DATA_DIR.mkdir(exist_ok=True)


def _load_dotenv(path: Path) -> None:
    """Minimal KEY=VALUE .env loader (no dependency); real env vars win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip("'\""))


_load_dotenv(ROOT_DIR / ".env")

# -- Deployment mode --
# "local"  : self-hosted GPU services below; browsers may switch stages to Gemini (BYOK).
# "gemini" : Gemini-only deployment (no GPU services installed). Every call uses
#            Gemini with the browser's key, or GEMINI_API_KEY as a fallback.
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "local").strip().lower()
if MODEL_PROVIDER not in ("local", "gemini"):
    MODEL_PROVIDER = "local"
GEMINI_ONLY = MODEL_PROVIDER == "gemini"

# -- Local self-hosted services (see start_*.sh) --
STT_URL = os.getenv("STT_URL", "http://localhost:8010")      # Faster-Whisper
LLM_URL = os.getenv("LLM_URL", "http://localhost:8003")      # vLLM / Gemma 4
TTS_URL = os.getenv("TTS_URL", "http://localhost:8011")      # Kokoro-82M
INDIC_URL = os.getenv("INDIC_URL", "http://localhost:7862")  # AI4Bharat IndicConformer ASR + IndicParler TTS
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemma-4-31B-it")

# -- Audio formats on the browser <-> orchestrator WebSocket --
INPUT_SAMPLE_RATE = 16000   # mic PCM16 mono
OUTPUT_SAMPLE_RATE = 24000  # playback PCM16 mono

# -- Gemini API (bring-your-own-key alternative to the local stack) --
GEMINI_API_BASE = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
GEMINI_LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-3.8-flash")
# Flash-Lite supports "minimal" thinking, which keeps transcription fast.
GEMINI_STT_MODEL = os.getenv("GEMINI_STT_MODEL", "gemini-3.5-flash-lite")
GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts")
# Gemini 3 models accept low/medium/high; "none" only works on Gemini 2.5 Flash.
GEMINI_REASONING_EFFORT = os.getenv("GEMINI_REASONING_EFFORT", "low")
# Used only when a browser selects Gemini without supplying its own key.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
