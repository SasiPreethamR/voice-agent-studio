"""
Model providers for the voice pipeline: speech-to-text, chat LLM, text-to-speech.

Two backends share one interface:
  local   - self-hosted Faster-Whisper (8010), vLLM/Gemma (8003), Kokoro (8011)
            and the AI4Bharat Indic stack (7862) for Indic languages.
  gemini  - Google Gemini API with a bring-your-own key. Chat and STT use the
            OpenAI-compatible endpoint (so streaming + tool calls keep the same
            shape); TTS uses the Interactions API and returns 24 kHz PCM16.

Each stage can be switched to Gemini independently via ModelSettings, so e.g.
Gemini LLM + local Kokoro TTS is a valid combination.
"""

import io
import json
import base64
import struct
from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np
import httpx

from config import (
    STT_URL, LLM_URL, TTS_URL, INDIC_URL, LLM_MODEL,
    INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE,
    GEMINI_API_BASE, GEMINI_LLM_MODEL, GEMINI_STT_MODEL, GEMINI_TTS_MODEL,
    GEMINI_REASONING_EFFORT, GEMINI_API_KEY,
)

MIN_AUDIO_ENERGY = 400

HALLUCINATION_PHRASES = {
    "thank you", "thanks", "thank you.", "thanks.", "thank you very much",
    "thanks for watching", "thanks for watching.", "thank you for watching",
    "bye", "bye.", "bye bye", "goodbye", "goodbye.",
    "you", "you.", "you\'re", "the end", "the end.",
    "so", "so.", "hmm", "hmm.", "uh", "uh.", "um", "um.",
    "subtitles by", "subtitles", "amara.org",
    "...", "..", ".",
    "", " ",
}

LANG_NAMES = {
    "en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French",
    "de": "German", "ja": "Japanese", "zh": "Chinese",
    # AI4Bharat Indic languages (routed to INDIC_URL on the local provider)
    "as": "Assamese", "bn": "Bengali", "brx": "Bodo", "doi": "Dogri",
    "gu": "Gujarati", "kn": "Kannada", "kok": "Konkani", "ks": "Kashmiri",
    "mai": "Maithili", "ml": "Malayalam", "mni": "Manipuri", "mr": "Marathi",
    "ne": "Nepali", "or": "Odia", "pa": "Punjabi", "sa": "Sanskrit",
    "sat": "Santali", "sd": "Sindhi", "ta": "Tamil", "te": "Telugu",
    "ur": "Urdu",
}

KOKORO_LANG_CODES = {
    "en": "a", "hi": "h", "es": "e", "fr": "f",
    "ja": "j", "zh": "z",
}

# Languages routed to the AI4Bharat indic stack (IndicConformer + IndicParler).
# Hindi stays on Kokoro/Whisper by default for speed; flip the flag to route it
# to the indic stack instead.
INDIC_LANGS = {
    "as", "bn", "brx", "doi", "gu", "kn", "kok", "ks", "mai", "ml",
    "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur",
}

# Optional per-voice style descriptions for IndicParler. The model expects a
# free-text English style prompt, not a voice ID. Voices map to Kokoro names so
# the agent UI keeps a single "voice" selector.
INDIC_VOICE_DESCRIPTIONS = {
    "alloy":   "A clear female voice speaks at a moderate pace with a neutral, warm tone. The recording is of very high quality, with the speaker's voice sounding close up.",
    "nova":    "A bright female voice speaks energetically and clearly at a moderate pace. The recording is of very high quality, with the speaker's voice sounding close up.",
    "shimmer": "A soft, gentle female voice speaks at a moderate pace with a calm tone. The recording is of very high quality, with the speaker's voice sounding close up.",
    "echo":    "A clear male voice speaks at a moderate pace with a neutral, friendly tone. The recording is of very high quality, with the speaker's voice sounding close up.",
    "onyx":    "A deep, confident male voice speaks at a moderate pace with an authoritative tone. The recording is of very high quality, with the speaker's voice sounding close up.",
    "fable":   "An expressive male voice narrates at a moderate pace with a warm, engaging tone. The recording is of very high quality, with the speaker's voice sounding close up.",
}
INDIC_DEFAULT_DESCRIPTION = INDIC_VOICE_DESCRIPTIONS["alloy"]
INDIC_VOICE_SEEDS = {
    "alloy": 147011,
    "nova": 247013,
    "shimmer": 347029,
    "echo": 447041,
    "onyx": 547069,
    "fable": 647083,
}

# Agent voice names -> Gemini prebuilt voices of a similar character.
GEMINI_VOICES = {
    "alloy":   "Kore",     # firm female
    "nova":    "Zephyr",   # bright female
    "shimmer": "Aoede",    # breezy female
    "echo":    "Puck",     # upbeat male
    "onyx":    "Charon",   # informative male
    "fable":   "Fenrir",   # excitable male
}

REASONING_EFFORTS = {"", "none", "minimal", "low", "medium", "high"}


# ================================================================
# Settings
# ================================================================
class ProviderError(RuntimeError):
    """A model provider rejected a request; the message is safe to show users."""


@dataclass
class ModelSettings:
    provider: str = "local"                      # "local" | "gemini"
    api_key: str = field(default="", repr=False)
    llm_model: str = GEMINI_LLM_MODEL
    stt_model: str = GEMINI_STT_MODEL
    tts_model: str = GEMINI_TTS_MODEL
    reasoning_effort: str = GEMINI_REASONING_EFFORT
    use_llm: bool = True
    use_stt: bool = True
    use_tts: bool = True
    live_captions: bool = False                  # partial STT while speaking (extra API calls)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "ModelSettings":
        if not isinstance(data, dict):
            return cls()
        known = {f.name: f for f in fields(cls)}
        kwargs = {}
        for key, value in data.items():
            if key not in known or value is None:
                continue
            if known[key].type is bool:
                kwargs[key] = value is True or str(value).lower() in ("true", "1", "yes", "on")
            else:
                kwargs[key] = str(value).strip()
        settings = cls(**kwargs)
        if settings.provider not in ("local", "gemini"):
            settings.provider = "local"
        if settings.reasoning_effort not in REASONING_EFFORTS:
            settings.reasoning_effort = GEMINI_REASONING_EFFORT
        settings.llm_model = settings.llm_model or GEMINI_LLM_MODEL
        settings.stt_model = settings.stt_model or settings.llm_model
        settings.tts_model = settings.tts_model or GEMINI_TTS_MODEL
        return settings

    @classmethod
    def from_header(cls, raw: Optional[str]) -> "ModelSettings":
        if not raw:
            return cls()
        try:
            return cls.from_dict(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            return cls()

    def gemini(self, stage: str) -> bool:
        """True when `stage` ("llm" | "stt" | "tts") should use Gemini."""
        return self.provider == "gemini" and bool(getattr(self, f"use_{stage}", False))

    @property
    def key(self) -> str:
        key = self.api_key or GEMINI_API_KEY
        if not key:
            raise ProviderError("Gemini API key missing - add it in Model settings.")
        return key

    def describe(self) -> str:
        if self.provider != "gemini":
            return "local"
        stages = [s for s in ("llm", "stt", "tts") if self.gemini(s)]
        return f"gemini[{','.join(stages) or 'none'}] llm={self.llm_model} tts={self.tts_model}"


LOCAL = ModelSettings()


def _gemini_error(resp: httpx.Response) -> ProviderError:
    detail = resp.text[:300]
    try:
        body = resp.json()
        if isinstance(body, list) and body:
            body = body[0]
        if isinstance(body, dict):
            err = body.get("error", body)
            detail = err.get("message") or detail if isinstance(err, dict) else str(err)
    except Exception:
        pass
    if resp.status_code in (401, 403):
        return ProviderError(f"Gemini rejected the API key ({resp.status_code}): {detail}")
    if resp.status_code == 429:
        return ProviderError(f"Gemini rate limit / quota exceeded: {detail}")
    return ProviderError(f"Gemini error {resp.status_code}: {detail}")


# ================================================================
# Audio helpers
# ================================================================
def pcm_to_wav_bytes(pcm: bytes, sample_rate: int = INPUT_SAMPLE_RATE) -> bytes:
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    buf.seek(0)
    return buf.read()


def is_hallucination(text: str) -> bool:
    clean = text.strip().lower().rstrip('.!?,;:')
    if clean in HALLUCINATION_PHRASES:
        return True
    if len(clean.split()) <= 1 and len(clean) < 6:
        return True
    return False


def audio_has_speech(pcm: bytes, threshold: float = MIN_AUDIO_ENERGY) -> bool:
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if len(samples) == 0:
        return False
    rms = float(np.sqrt(np.mean(samples ** 2)))
    return rms > threshold


def _float_to_pcm24k(data: np.ndarray, sr: int) -> bytes:
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != OUTPUT_SAMPLE_RATE and len(data) > 0:
        # Simple linear resample - good enough for speech and avoids extra deps.
        n_out = int(round(len(data) * OUTPUT_SAMPLE_RATE / sr))
        if n_out > 0:
            x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False, dtype=np.float32)
            x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float32)
            data = np.interp(x_new, x_old, data).astype(np.float32)
    pcm = np.clip(data, -1.0, 1.0)
    return (pcm * 32767.0).astype(np.int16).tobytes()


def _pcm16_wav_frames(wav_bytes: bytes) -> Optional[tuple]:
    """Parse a PCM16 RIFF/WAV blob -> (int16 samples, sample_rate, channels).

    Tolerates streaming-style headers whose size fields are 0 / 0xFFFFFFFF,
    which the stdlib `wave` module rejects.
    """
    if len(wav_bytes) < 12 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        return None
    pos, fmt = 12, None
    while pos + 8 <= len(wav_bytes):
        cid, size = wav_bytes[pos:pos + 4], struct.unpack("<I", wav_bytes[pos + 4:pos + 8])[0]
        body = pos + 8
        if cid == b"fmt ":
            audio_fmt, channels, rate = struct.unpack("<HHI", wav_bytes[body:body + 8])
            bits = struct.unpack("<H", wav_bytes[body + 14:body + 16])[0]
            fmt = (audio_fmt, channels, rate, bits)
        elif cid == b"data":
            if not fmt or fmt[0] != 1 or fmt[3] != 16:
                return None
            end = len(wav_bytes) if size in (0, 0xFFFFFFFF) else min(len(wav_bytes), body + size)
            data = wav_bytes[body:end]
            data = data[: len(data) - (len(data) % 2)]
            return np.frombuffer(data, dtype="<i2"), fmt[2], fmt[1]
        pos = body + size + (size & 1)
    return None


def _wav_to_pcm24k(wav_bytes: bytes) -> Optional[bytes]:
    """Decode a WAV blob (any sr / float or int) to mono int16 PCM @ 24 kHz."""
    parsed = _pcm16_wav_frames(wav_bytes)
    if parsed:
        samples, sr, channels = parsed
        data = samples.astype(np.float32) / 32768.0
        if channels > 1:
            data = data[: len(data) - len(data) % channels].reshape(-1, channels)
        if sr == OUTPUT_SAMPLE_RATE and channels == 1:
            return samples.astype(np.int16).tobytes()
        return _float_to_pcm24k(data, sr)
    try:
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    except Exception as e:
        print(f"[PROVIDERS] Failed to decode TTS WAV: {e}")
        return None
    return _float_to_pcm24k(data, sr)


# ================================================================
# Speech-to-text
# ================================================================
async def transcribe_audio(audio_pcm: bytes, language: str = "en", model: ModelSettings = LOCAL) -> str:
    if not audio_has_speech(audio_pcm):
        print("[ORCH] Skipped transcription - low energy audio")
        return ""
    wav = pcm_to_wav_bytes(audio_pcm)

    if model.gemini("stt"):
        text = await _gemini_transcribe(wav, language, model)
        if is_hallucination(text):
            print(f"[ORCH] Filtered hallucination (gemini): '{text}'")
            return ""
        return text

    # Route Indic languages to AI4Bharat IndicConformer.
    if language in INDIC_LANGS:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{INDIC_URL}/api/asr",
                files={"audio": ("audio.wav", wav, "audio/wav")},
                data={"lang": language, "decoder": "ctc"},
            )
            resp.raise_for_status()
            text = (resp.json().get("text") or "").strip()
            if is_hallucination(text):
                print(f"[ORCH] Filtered hallucination (indic): '{text}'")
                return ""
            return text

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{STT_URL}/v1/audio/transcriptions",
            files={"file": ("audio.wav", wav, "audio/wav")},
            data={"response_format": "json", "language": language},
        )
        resp.raise_for_status()
        text = resp.json().get("text", "").strip()
        if is_hallucination(text):
            print(f"[ORCH] Filtered hallucination: '{text}'")
            return ""
        return text


async def _gemini_transcribe(wav: bytes, language: str, model: ModelSettings) -> str:
    lang_name = LANG_NAMES.get(language, language)
    prompt = (
        f"Transcribe this {lang_name} speech verbatim, in the language's native script. "
        "Reply with only the transcript - no quotes, labels, translation or commentary. "
        "If there is no intelligible speech, reply with an empty message."
    )
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}},
    ]}]
    body = _chat_body(messages, model, stage="stt", max_tokens=1024, temperature=0.0)
    async with httpx.AsyncClient(timeout=60.0) as client:
        url, headers = _chat_target(model, "stt")
        resp = await client.post(url, json=body, headers=headers)
        if resp.status_code != 200:
            raise _gemini_error(resp)
        content = resp.json()["choices"][0]["message"].get("content") or ""
    return content.strip().strip('"').strip()


# ================================================================
# Chat LLM
# ================================================================
def _chat_target(model: ModelSettings, stage: str = "llm") -> tuple:
    """(url, headers) for the chat-completions endpoint serving `stage`."""
    if model.gemini(stage):
        return f"{GEMINI_API_BASE}/openai/chat/completions", {"Authorization": f"Bearer {model.key}"}
    return f"{LLM_URL}/v1/chat/completions", {}


def _chat_body(messages: list, model: ModelSettings, stage: str = "llm", max_tokens: int = 512,
               temperature: float = 0.7, **extra) -> dict:
    if model.gemini(stage):
        body = {
            "model": model.stt_model if stage == "stt" else model.llm_model,
            "messages": messages,
            # Gemini counts thinking tokens against the output budget.
            "max_tokens": max(max_tokens, 2048),
            "temperature": temperature,
        }
        if model.reasoning_effort:
            body["reasoning_effort"] = model.reasoning_effort
    else:
        body = {"model": LLM_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    body.update({k: v for k, v in extra.items() if v is not None})
    return body


async def stream_llm_tokens(messages: list, tools: list = None, model: ModelSettings = LOCAL,
                            tool_choice: Optional[str] = None):
    body = _chat_body(messages, model, stream=True, tools=tools or None,
                      tool_choice=tool_choice if tools else None)
    url, headers = _chat_target(model)
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code != 200:
                await resp.aread()
                if model.gemini("llm"):
                    raise _gemini_error(resp)
                resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"]
                        if delta.get("tool_calls"):
                            yield {"tool_calls": delta["tool_calls"]}
                        elif delta.get("content"):
                            yield delta["content"]
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass


async def call_llm_with_tools(messages: list, tools: list, model: ModelSettings = LOCAL) -> dict:
    body = _chat_body(messages, model, tools=tools)
    async with httpx.AsyncClient(timeout=60.0) as client:
        url, headers = _chat_target(model)
        resp = await client.post(url, json=body, headers=headers)
        if resp.status_code != 200 and model.gemini("llm"):
            raise _gemini_error(resp)
        resp.raise_for_status()
        return resp.json()


async def chat_completion(messages: list, model: ModelSettings = LOCAL, max_tokens: int = 512,
                          temperature: float = 0.7, timeout: float = 180.0) -> str:
    """Single non-streaming completion; returns the assistant text."""
    body = _chat_body(messages, model, max_tokens=max_tokens, temperature=temperature)
    async with httpx.AsyncClient(timeout=timeout) as client:
        url, headers = _chat_target(model)
        resp = await client.post(url, json=body, headers=headers)
        if resp.status_code != 200 and model.gemini("llm"):
            raise _gemini_error(resp)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"].get("content") or ""


# ================================================================
# Text-to-speech
# ================================================================
async def synthesize_speech(text: str, voice: str = "alloy", speed: float = 1.0, lang_code: str = "a",
                            language: str = "en", model: ModelSettings = LOCAL) -> Optional[bytes]:
    """Returns mono int16 PCM @ OUTPUT_SAMPLE_RATE, or None."""
    if not text or not text.strip():
        return None

    if model.gemini("tts"):
        return await _gemini_tts(text, voice, speed, model)

    # Route Indic languages to AI4Bharat IndicParler. Returns WAV; decode and
    # convert to 24 kHz int16 PCM so the client path stays uniform.
    if language in INDIC_LANGS:
        description = INDIC_VOICE_DESCRIPTIONS.get(voice, INDIC_DEFAULT_DESCRIPTION)
        seed = INDIC_VOICE_SEEDS.get(voice, INDIC_VOICE_SEEDS["alloy"])
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{INDIC_URL}/api/tts",
                data={"prompt": text, "description": description, "seed": str(seed)},
            )
            if resp.status_code != 200:
                print(f"[ORCH] Indic TTS failed: {resp.status_code} {resp.text[:200]}")
                return None
            return _wav_to_pcm24k(resp.content)

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{TTS_URL}/v1/audio/speech",
            json={"input": text, "voice": voice, "response_format": "pcm", "speed": speed, "lang_code": lang_code},
        )
        if resp.status_code == 200:
            return resp.content
    return None


def _speed_style(speed: float) -> Optional[str]:
    # Gemini TTS has no numeric rate; pace is steered with a style annotation.
    if speed >= 1.35:
        return "speaking rapidly"
    if speed >= 1.15:
        return "speaking a little faster than usual"
    if speed <= 0.65:
        return "speaking slowly"
    if speed <= 0.85:
        return "speaking a little slower than usual"
    return None


def _find_audio_blob(node) -> Optional[tuple]:
    """Depth-first search for the last base64 audio part -> (bytes, mime_type)."""
    found = None
    if isinstance(node, dict):
        mime = node.get("mime_type") or node.get("mimeType") or ""
        data = node.get("data")
        if isinstance(data, str) and data and (node.get("type") == "audio" or mime.startswith("audio")):
            found = (base64.b64decode(data), mime)
        for value in node.values():
            found = _find_audio_blob(value) or found
    elif isinstance(node, list):
        for value in node:
            found = _find_audio_blob(value) or found
    return found


def _audio_blob_to_pcm24k(blob: bytes, mime: str) -> Optional[bytes]:
    if blob[:4] == b"RIFF" or "wav" in mime.lower():
        return _wav_to_pcm24k(blob)
    # Raw little-endian PCM16 ("audio/l16; rate=24000" or "audio/pcm;rate=24000").
    rate = OUTPUT_SAMPLE_RATE
    for part in mime.replace(" ", "").split(";"):
        if part.lower().startswith("rate="):
            try:
                rate = int(part[5:])
            except ValueError:
                pass
    blob = blob[: len(blob) - (len(blob) % 2)]
    if rate == OUTPUT_SAMPLE_RATE:
        return blob
    samples = np.frombuffer(blob, dtype="<i2").astype(np.float32) / 32768.0
    return _float_to_pcm24k(samples, rate)


async def _gemini_tts(text: str, voice: str, speed: float, model: ModelSettings) -> Optional[bytes]:
    voice_name = GEMINI_VOICES.get(voice, voice if voice and voice[0].isupper() else "Kore")
    style = _speed_style(speed)
    headers = {"x-goog-api-key": model.key}
    part = {"type": "text", "text": text}
    if style:
        part["annotations"] = [{"type": "speech_metadata", "style": style}]
    body = {
        "model": model.tts_model,
        "input": [{"type": "user_input", "content": [part]}],
        "response_format": {"type": "audio"},
        "generation_config": {"speech_config": [{"voice": voice_name}]},
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{GEMINI_API_BASE}/interactions", json=body, headers=headers)
        if resp.status_code in (400, 404):
            # Fall back to the generateContent TTS API for models/keys where the
            # Interactions API is not available.
            legacy = await client.post(
                f"{GEMINI_API_BASE}/models/{model.tts_model}:generateContent",
                headers=headers,
                json={
                    "contents": [{"role": "user", "parts": [{"text": text}]}],
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice_name}}},
                    },
                },
            )
            if legacy.status_code == 200:
                resp = legacy
        if resp.status_code != 200:
            raise _gemini_error(resp)
        blob = _find_audio_blob(resp.json())
    if not blob:
        raise ProviderError("Gemini TTS returned no audio")
    return _audio_blob_to_pcm24k(*blob)


# ================================================================
# Key check / model discovery
# ================================================================
_NON_CHAT_MARKERS = ("embedding", "image", "tts", "live", "native-audio", "veo", "imagen", "aqa", "robotics", "computer-use")


async def gemini_list_models(model: ModelSettings) -> dict:
    """Validate the key and list usable chat + TTS models."""
    names, token = [], ""
    async with httpx.AsyncClient(timeout=15.0) as client:
        for _ in range(5):
            params = {"pageSize": 1000}
            if token:
                params["pageToken"] = token
            resp = await client.get(f"{GEMINI_API_BASE}/models", params=params,
                                    headers={"x-goog-api-key": model.key})
            if resp.status_code != 200:
                raise _gemini_error(resp)
            payload = resp.json()
            names += [m.get("name", "").removeprefix("models/") for m in payload.get("models", [])]
            token = payload.get("nextPageToken") or ""
            if not token:
                break
    gemini = sorted({n for n in names if n.startswith("gemini")}, reverse=True)
    return {
        "chat_models": [n for n in gemini if not any(m in n for m in _NON_CHAT_MARKERS)],
        "tts_models": [n for n in gemini if "tts" in n],
    }
