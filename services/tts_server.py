"""
Real-time Text-to-Speech API Server
Model: Kokoro-82M (Apache-licensed, #1 TTS Arena)
Port: 8011
OpenAI-compatible /v1/audio/speech endpoint with streaming support
"""
import io
import os
import struct
from typing import Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")  # scripts/start_tts.sh sets TTS_GPU

import torch
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[TTS] Using device: {DEVICE} (CUDA available: {torch.cuda.is_available()})")

app = FastAPI(title="TTS API - Kokoro-82M", version="1.0.0")

_shared_model = None  # single KModel shared across all lang pipelines
pipelines = {}  # lang_code -> KPipeline

# Default English voice map (OpenAI name -> Kokoro voice ID)
VOICE_MAP = {
    "alloy": "af_heart",
    "echo": "am_adam",
    "fable": "bf_emma",
    "onyx": "am_michael",
    "nova": "af_bella",
    "shimmer": "af_sky",
}

# Per-language voice maps: lang_code -> {openai_name -> kokoro_voice_id}
# Falls back to the first available voice for that language if no match
LANG_VOICE_MAPS = {
    "a": VOICE_MAP,  # American English
    "b": {"alloy": "bf_emma", "echo": "bm_daniel", "fable": "bf_emma", "onyx": "bm_george", "nova": "bf_isabella", "shimmer": "bf_lily"},
    "e": {"alloy": "ef_dora", "echo": "em_alex", "fable": "ef_dora", "onyx": "em_alex", "nova": "ef_dora", "shimmer": "ef_dora"},  # Spanish
    "f": {"alloy": "ff_siwis", "echo": "ff_siwis", "fable": "ff_siwis", "onyx": "ff_siwis", "nova": "ff_siwis", "shimmer": "ff_siwis"},  # French (only 1 voice)
    "h": {"alloy": "hf_alpha", "echo": "hm_omega", "fable": "hf_alpha", "onyx": "hm_omega", "nova": "hf_beta", "shimmer": "hf_beta"},  # Hindi
    "i": {"alloy": "if_sara", "echo": "im_nicola", "fable": "if_sara", "onyx": "im_nicola", "nova": "if_sara", "shimmer": "if_sara"},  # Italian
    "j": {"alloy": "jf_alpha", "echo": "jm_kumo", "fable": "jf_gongitsune", "onyx": "jm_kumo", "nova": "jf_nezumi", "shimmer": "jf_tebukuro"},  # Japanese
    "p": {"alloy": "pf_dora", "echo": "pm_alex", "fable": "pf_dora", "onyx": "pm_alex", "nova": "pf_dora", "shimmer": "pf_dora"},  # Portuguese
    "z": {"alloy": "zf_xiaobei", "echo": "zm_yunjian", "fable": "zf_xiaoni", "onyx": "zm_yunxi", "nova": "zf_xiaoxiao", "shimmer": "zf_xiaoyi"},  # Chinese
}

SAMPLE_RATE = 24000


def get_pipeline(lang_code: str = 'a'):
    global pipelines, _shared_model
    if lang_code not in pipelines:
        from kokoro import KPipeline
        from kokoro.model import KModel
        # Load model explicitly on CUDA once, share across all pipelines
        if _shared_model is None:
            print(f"[TTS] Loading KModel on {DEVICE}...")
            _shared_model = KModel(repo_id='hexgrad/Kokoro-82M').to(DEVICE).eval()
            print(f"[TTS] KModel loaded on {next(_shared_model.parameters()).device}")
        print(f"[TTS] Creating pipeline for lang_code='{lang_code}'...")
        pipelines[lang_code] = KPipeline(lang_code=lang_code, model=_shared_model)
        print(f"[TTS] Pipeline ready for lang_code='{lang_code}'.")
    return pipelines[lang_code]


@app.on_event("startup")
async def startup():
    get_pipeline('a')  # preload English


@app.get("/health")
async def health():
    return {"status": "ok", "model": "kokoro-82m"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "kokoro", "object": "model", "owned_by": "hexgrad"}]
    }


@app.get("/v1/voices")
async def list_voices():
    return {
        "voices": list(VOICE_MAP.keys()),
        "voice_map": VOICE_MAP,
    }


class SpeechRequest(BaseModel):
    model: str = "kokoro"
    input: str
    voice: str = "alloy"
    response_format: str = "wav"
    speed: float = 1.0
    lang_code: str = "a"


def create_wav_bytes(audio_np: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio_np, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def to_numpy(audio) -> np.ndarray:
    """Convert audio (torch.Tensor or np.ndarray) to numpy float32. Returns empty array for None."""
    if audio is None:
        return np.array([], dtype=np.float32)
    if isinstance(audio, torch.Tensor):
        return audio.detach().cpu().float().numpy()
    return np.asarray(audio, dtype=np.float32)


def create_pcm_bytes(audio) -> bytes:
    audio_np = to_numpy(audio)
    pcm = (audio_np * 32767).astype(np.int16)
    return pcm.tobytes()


@app.post("/v1/audio/speech")
async def text_to_speech(request: SpeechRequest):
    """OpenAI-compatible TTS endpoint."""
    p = get_pipeline(request.lang_code)

    vmap = LANG_VOICE_MAPS.get(request.lang_code, VOICE_MAP)
    voice_id = vmap.get(request.voice, request.voice)

    try:
        audio_chunks = []
        generator = p(request.input, voice=voice_id, speed=request.speed)
        for gs, ps, audio in generator:
            np_audio = to_numpy(audio)
            if np_audio.size > 0:
                audio_chunks.append(np_audio)

        if not audio_chunks:
            raise HTTPException(status_code=500, detail="No audio generated")

        full_audio = np.concatenate(audio_chunks)

        if request.response_format == "pcm":
            return Response(
                content=create_pcm_bytes(full_audio),
                media_type="audio/pcm",
                headers={"X-Sample-Rate": str(SAMPLE_RATE)},
            )
        else:
            return Response(
                content=create_wav_bytes(full_audio),
                media_type="audio/wav",
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/speech/stream")
async def text_to_speech_stream(request: SpeechRequest):
    """Streaming TTS - returns audio chunks as they're generated."""
    p = get_pipeline(request.lang_code)
    vmap = LANG_VOICE_MAPS.get(request.lang_code, VOICE_MAP)
    voice_id = vmap.get(request.voice, request.voice)

    def audio_stream():
        generator = p(request.input, voice=voice_id, speed=request.speed)
        for _, _, audio in generator:
            yield create_pcm_bytes(audio)

    return StreamingResponse(
        audio_stream(),
        media_type="audio/pcm",
        headers={"X-Sample-Rate": str(SAMPLE_RATE)},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8011)
