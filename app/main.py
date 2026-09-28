"""
Voice Agent Studio - orchestrator entry point (port 8080).

Serves the Studio UI and operator console, the REST API and the /ws/voice
real-time voice WebSocket. Run from the repo root:

    python -m app.main            # or: uvicorn app.main:app

Pipeline:  Mic -> STT (8010) -> vLLM/Gemma (8003) + RAG -> TTS (8011) -> Speaker
Indic languages use the AI4Bharat server (7862) for STT/TTS. Any stage can be
switched to the Gemini API per call (BYOK) - see app/providers.py.

Modules: voice.py (call loop), tools.py, agents.py, handoff.py, api.py (REST),
providers.py (model calls), documents.py (RAG store), config.py.
"""

import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import ROOT_DIR, STATIC_DIR, MODEL_PROVIDER, GEMINI_ONLY
from app import api, handoff, voice

# Raise Starlette's multipart limits so large folder uploads (up to 100k files) work.
try:
    from starlette.formparsers import MultiPartParser
    MultiPartParser.max_files = 100_000
    MultiPartParser.max_fields = 100_000
except Exception:
    pass


# -- App --
app = FastAPI(title="Voice Agent Orchestrator", version="2.0.0")
app.include_router(voice.router)
app.include_router(handoff.router)
app.include_router(api.router)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_ui():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/human")
async def serve_human_console():
    return FileResponse(str(STATIC_DIR / "human.html"))


if __name__ == "__main__":
    cert = ROOT_DIR / "cert.pem"
    key = ROOT_DIR / "key.pem"
    ssl_kw = {}
    # HTTPS=auto (default) uses cert.pem/key.pem when present; HTTPS=0 forces plain HTTP.
    if os.getenv("HTTPS", "auto").lower() not in ("0", "false", "no", "off") and cert.exists() and key.exists():
        ssl_kw = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
        print("[ORCH] HTTPS enabled (self-signed cert)")
    print(f"[ORCH] Model provider mode: {MODEL_PROVIDER}" + (" (Gemini only, no local GPU services)" if GEMINI_ONLY else ""))
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8080")), **ssl_kw)
