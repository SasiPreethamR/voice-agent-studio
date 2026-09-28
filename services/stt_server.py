"""
Real-time Speech-to-Text API Server
Model: Faster-Whisper large-v3-turbo (FP16, GPU)
Port: 8010
OpenAI-compatible /v1/audio/transcriptions endpoint + WebSocket streaming
"""
import io
import os
import json
import wave
import tempfile
import asyncio
import numpy as np
from typing import Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")  # scripts/start_stt.sh sets STT_GPU

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

app = FastAPI(title="STT API - Faster-Whisper", version="1.0.0")

MODEL_SIZE = "large-v3-turbo"
model: Optional[WhisperModel] = None


def get_model() -> WhisperModel:
    global model
    if model is None:
        print(f"Loading Faster-Whisper model: {MODEL_SIZE} on GPU...")
        model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
        print("Model loaded.")
    return model


@app.on_event("startup")
async def startup():
    get_model()


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_SIZE}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_SIZE, "object": "model", "owned_by": "faster-whisper"}]
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
):
    """OpenAI-compatible transcription endpoint."""
    m = get_model()

    # Save uploaded file to temp
    content = await file.read()
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        segments, info = m.transcribe(
            tmp_path,
            beam_size=1,
            language=language,
            temperature=temperature,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=200,
                threshold=0.4,
            ),
            no_speech_threshold=0.5,
            log_prob_threshold=-0.8,
            condition_on_previous_text=False,
            suppress_blank=True,
        )
        segments_list = list(segments)
        full_text = " ".join(seg.text.strip() for seg in segments_list if seg.text.strip())

        if response_format == "verbose_json":
            return JSONResponse({
                "task": "transcribe",
                "language": info.language,
                "duration": info.duration,
                "text": full_text,
                "segments": [
                    {
                        "id": i,
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text.strip(),
                    }
                    for i, seg in enumerate(segments_list)
                ],
            })
        elif response_format == "text":
            return JSONResponse(content=full_text, media_type="text/plain")
        else:
            return JSONResponse({"text": full_text})
    finally:
        os.unlink(tmp_path)


@app.websocket("/v1/audio/transcriptions/stream")
async def stream_transcribe(ws: WebSocket):
    """
    WebSocket streaming transcription.
    Send raw PCM16 audio chunks (16kHz, mono) as binary messages.
    Send JSON {"action": "stop"} to finalize.
    Receives JSON {"text": "...", "is_final": bool} messages.
    """
    await ws.accept()
    m = get_model()
    audio_buffer = bytearray()

    try:
        while True:
            data = await ws.receive()

            if "bytes" in data and data["bytes"]:
                audio_buffer.extend(data["bytes"])

                # Process every ~1 second of audio (32000 bytes = 1s at 16kHz 16bit mono)
                if len(audio_buffer) >= 32000:
                    audio_np = np.frombuffer(bytes(audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0
                    segments, _ = m.transcribe(
                        audio_np,
                        beam_size=1,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=300),
                    )
                    text = " ".join(seg.text.strip() for seg in segments)
                    if text.strip():
                        await ws.send_json({"text": text, "is_final": False})
                    audio_buffer.clear()

            elif "text" in data and data["text"]:
                msg = json.loads(data["text"])
                if msg.get("action") == "stop":
                    # Final transcription of remaining buffer
                    if audio_buffer:
                        audio_np = np.frombuffer(bytes(audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0
                        segments, _ = m.transcribe(audio_np, beam_size=1, vad_filter=True)
                        text = " ".join(seg.text.strip() for seg in segments)
                        if text.strip():
                            await ws.send_json({"text": text, "is_final": True})
                    await ws.close()
                    break

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)
