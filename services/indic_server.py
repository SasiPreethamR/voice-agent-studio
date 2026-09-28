"""
Indic Speech API Server (AI4Bharat)
  - ASR:  ai4bharat/indic-conformer-600m-multilingual  (POST /api/asr)
  - TTS:  ai4bharat/indic-parler-tts                   (POST /api/tts)
Port: 7862 (INDIC_PORT)
Serves a small playground UI at /. Model weights, the pinned-deps pyenv/ and
the .env with HF_TOKEN live in indic/.
"""
import io
import os
import logging
import subprocess
import tempfile
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

INDIC_DIR = Path(__file__).parent / "indic"

# Force HF caches into a writable, workspace-local dir BEFORE importing transformers.
_HF_CACHE = INDIC_DIR / "hf_cache"
(_HF_CACHE / "transformers").mkdir(parents=True, exist_ok=True)
(_HF_CACHE / "modules").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_HF_CACHE / "transformers"))
os.environ["HF_MODULES_CACHE"] = str(_HF_CACHE / "modules")  # force-override system default

import torch

# Disable torch JIT fusers to avoid NVRTC compiling invalid CUDA for the
# IndicConformer STFT preprocessing (complex tensors break TensorExpr fuser).
# NOTE: don't set PYTORCH_JIT=0 — it disables torch.jit.load() which the model needs.
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
for fn_name in ("_jit_set_texpr_fuser_enabled", "_jit_set_nvfuser_enabled", "_jit_override_can_fuse_on_cpu", "_jit_override_can_fuse_on_gpu"):
    try:
        getattr(torch._C, fn_name)(False)
    except Exception:
        pass
try:
    torch.jit.set_fusion_strategy([("STATIC", 0), ("DYNAMIC", 0)])
except Exception:
    pass

import torchaudio
import soundfile as sf
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

# Pick an ffmpeg binary: system, or the one bundled with imageio-ffmpeg.
def _find_ffmpeg() -> str:
    from shutil import which
    p = which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return ""

FFMPEG = _find_ffmpeg()

load_dotenv(INDIC_DIR.parent.parent / ".env")  # repo-root .env
load_dotenv(INDIC_DIR / ".env")  # legacy location; the root .env takes precedence

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ai4b-server")

ASR_MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
TTS_MODEL_ID = "ai4bharat/indic-parler-tts"

# Pin large models to separate GPUs if available.
# NOTE: the IndicConformer ONNX model runs on CPU here (no CUDAExecutionProvider
# in this onnxruntime build). Keeping it on CPU also avoids torch NVRTC fusion
# of its preprocessing ops, which needs libnvrtc-builtins matching the torch build.
DEVICE_ASR = "cpu"
DEVICE_TTS = "cuda:1" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DESC_CACHE_MAX = int(os.getenv("INDIC_TTS_DESC_CACHE", "64"))
TTS_ATTN_IMPLS = [i.strip() for i in os.getenv("INDIC_TTS_ATTN_IMPLS", "flash_attention_2,sdpa,eager").split(",") if i.strip()]

ASR_LANGS = {
    "as": "Assamese", "bn": "Bengali", "brx": "Bodo", "doi": "Dogri",
    "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada", "kok": "Konkani",
    "ks": "Kashmiri", "mai": "Maithili", "ml": "Malayalam", "mni": "Manipuri",
    "mr": "Marathi", "ne": "Nepali", "or": "Odia", "pa": "Punjabi",
    "sa": "Sanskrit", "sat": "Santali", "sd": "Sindhi", "ta": "Tamil",
    "te": "Telugu", "ur": "Urdu",
}

STATE: dict = {}


def _prepare_cuda_tts_runtime() -> None:
    if not torch.cuda.is_available():
        return
    # H200 supports fast TF32/BF16 paths; enabling these improves real decode speed.
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def _load_tts_model():
    from parler_tts import ParlerTTSForConditionalGeneration

    last_error = None
    for attn_impl in TTS_ATTN_IMPLS:
        try:
            kwargs = {"torch_dtype": DTYPE}
            if attn_impl and attn_impl != "eager":
                kwargs["attn_implementation"] = attn_impl
            model = ParlerTTSForConditionalGeneration.from_pretrained(TTS_MODEL_ID, **kwargs).to(DEVICE_TTS)
            return model, attn_impl
        except Exception as exc:
            last_error = exc
            log.warning("TTS load with attn_impl=%s failed: %s", attn_impl, exc)
    raise RuntimeError(f"Unable to load TTS model {TTS_MODEL_ID}: {last_error}")


def _get_cached_description_tokens(description: str):
    cache: OrderedDict = STATE.setdefault("tts_desc_cache", OrderedDict())
    key = (description or "").strip()
    cached = cache.get(key)
    if cached is None:
        tok = STATE["tts_desc_tok"](key, return_tensors="pt")
        cached = (tok.input_ids.cpu(), tok.attention_mask.cpu())
        cache[key] = cached
        if len(cache) > DESC_CACHE_MAX:
            cache.popitem(last=False)
    else:
        cache.move_to_end(key)
    input_ids, attention_mask = cached
    return input_ids.to(DEVICE_TTS), attention_mask.to(DEVICE_TTS)


def _tts_rng_devices() -> list[int]:
    if not DEVICE_TTS.startswith("cuda") or not torch.cuda.is_available():
        return []
    if ":" in DEVICE_TTS:
        try:
            return [int(DEVICE_TTS.split(":", 1)[1])]
        except ValueError:
            pass
    return [torch.cuda.current_device()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading ASR model %s on %s ...", ASR_MODEL_ID, DEVICE_ASR)
    from transformers import AutoModel
    asr = AutoModel.from_pretrained(ASR_MODEL_ID, trust_remote_code=True)
    try:
        asr = asr.to(DEVICE_ASR)
    except Exception as e:  # some custom models don't follow .to()
        log.warning("ASR .to(%s) failed (%s); leaving on default device", DEVICE_ASR, e)
    STATE["asr"] = asr
    log.info("ASR loaded.")

    _prepare_cuda_tts_runtime()
    log.info("Loading TTS model %s on %s ...", TTS_MODEL_ID, DEVICE_TTS)
    from transformers import AutoTokenizer
    tts, attn_impl = _load_tts_model()
    prompt_tok = AutoTokenizer.from_pretrained(TTS_MODEL_ID)
    desc_tok = AutoTokenizer.from_pretrained(tts.config.text_encoder._name_or_path)
    STATE["tts"] = tts
    STATE["tts_prompt_tok"] = prompt_tok
    STATE["tts_desc_tok"] = desc_tok
    STATE["tts_desc_cache"] = OrderedDict()
    STATE["tts_sr"] = tts.config.sampling_rate
    STATE["tts_dtype"] = str(next(tts.parameters()).dtype)
    STATE["tts_attn_impl"] = attn_impl
    log.info("TTS loaded. sampling_rate=%s dtype=%s attn_impl=%s", STATE["tts_sr"], STATE["tts_dtype"], attn_impl)

    # Optional warmup compiles/loads kernels up-front so first real request is faster.
    if torch.cuda.is_available() and os.getenv("INDIC_TTS_WARMUP", "1") == "1":
        try:
            wd = desc_tok("A clear and natural speaking style.", return_tensors="pt").to(DEVICE_TTS)
            wp = prompt_tok("నమస్కారం", return_tensors="pt").to(DEVICE_TTS)
            with torch.inference_mode():
                _ = tts.generate(
                    input_ids=wd.input_ids,
                    attention_mask=wd.attention_mask,
                    prompt_input_ids=wp.input_ids,
                    prompt_attention_mask=wp.attention_mask,
                    use_cache=True,
                )
            torch.cuda.synchronize()
            log.info("TTS warmup complete")
        except Exception as exc:
            log.warning("TTS warmup skipped: %s", exc)

    yield

    STATE.clear()


app = FastAPI(title="AI4Bharat Indic Speech Stack", lifespan=lifespan)


# ---------- ASR ----------
@app.post("/api/asr")
async def asr(
    audio: UploadFile = File(...),
    lang: str = Form("hi"),
    decoder: str = Form("ctc"),
):
    if lang not in ASR_LANGS:
        raise HTTPException(400, f"Unsupported lang '{lang}'. Allowed: {sorted(ASR_LANGS)}")
    if decoder not in {"ctc", "rnnt"}:
        raise HTTPException(400, "decoder must be 'ctc' or 'rnnt'")

    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "Empty audio upload")

    # Decode anything → mono 16 kHz float32 WAV via ffmpeg, then load with soundfile.
    if not FFMPEG:
        raise HTTPException(500, "ffmpeg not available on server")
    suffix = Path(audio.filename or "audio").suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix) as inf, \
         tempfile.NamedTemporaryFile(suffix=".wav") as outf:
        inf.write(raw); inf.flush()
        cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", inf.name,
               "-ac", "1", "-ar", "16000", "-f", "wav", outf.name]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise HTTPException(400, f"Could not decode audio: {proc.stderr.decode(errors='ignore')[:300]}")
        try:
            data, sr = sf.read(outf.name, dtype="float32", always_2d=False)
        except Exception as e:
            raise HTTPException(400, f"Could not read decoded WAV: {e}")

    wav = torch.from_numpy(data).unsqueeze(0)  # [1, T]
    assert sr == 16000

    model = STATE["asr"]
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            text = model(wav, lang, decoder)
    except Exception as e:
        log.exception("ASR inference failed")
        raise HTTPException(500, f"ASR failed: {e}")

    log.info("ASR completed lang=%s decoder=%s seconds=%.3f", lang, decoder, time.perf_counter() - started)

    return {"text": text, "lang": lang, "decoder": decoder}


# ---------- TTS ----------
@app.post("/api/tts")
async def tts(
    prompt: str = Form(...),
    description: str = Form(
        "A clear female voice speaks at a moderate pace with neutral tone. "
        "The recording is of very high quality, with the speaker's voice sounding close up."
    ),
    seed: Optional[int] = Form(None),
):
    if not prompt.strip():
        raise HTTPException(400, "prompt must not be empty")

    model = STATE["tts"]
    p_tok = STATE["tts_prompt_tok"]
    sr = STATE["tts_sr"]

    desc_input_ids, desc_attention_mask = _get_cached_description_tokens(description)
    prom = p_tok(prompt, return_tensors="pt").to(DEVICE_TTS)
    generate_kwargs = {
        "input_ids": desc_input_ids,
        "attention_mask": desc_attention_mask,
        "prompt_input_ids": prom.input_ids,
        "prompt_attention_mask": prom.attention_mask,
        "use_cache": True,
    }
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            if seed is None:
                gen = model.generate(**generate_kwargs)
            else:
                seed_value = int(seed) % (2**31)
                with torch.random.fork_rng(devices=_tts_rng_devices(), enabled=True):
                    torch.manual_seed(seed_value)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed_value)
                    gen = model.generate(**generate_kwargs)
    except Exception as e:
        log.exception("TTS inference failed")
        raise HTTPException(500, f"TTS failed: {e}")

    audio_arr = gen.float().cpu().numpy().squeeze()
    buf = io.BytesIO()
    sf.write(buf, audio_arr, sr, format="WAV")
    log.info("TTS completed chars=%s seed=%s seconds=%.3f", len(prompt), seed, time.perf_counter() - started)
    return Response(content=buf.getvalue(), media_type="audio/wav")


@app.get("/api/langs")
async def langs():
    return ASR_LANGS


# ---------- UI ----------
STATIC_DIR = INDIC_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")
async def health():
    return {
        "asr_loaded": "asr" in STATE,
        "tts_loaded": "tts" in STATE,
        "asr_device": DEVICE_ASR,
        "tts_device": DEVICE_TTS,
        "tts_dtype": STATE.get("tts_dtype"),
        "tts_attn_impl": STATE.get("tts_attn_impl"),
        "tts_desc_cache_items": len(STATE.get("tts_desc_cache", {})),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("INDIC_PORT", "7862")), log_level="info")
