# AI4Bharat Indic Speech Stack — Instructions

Self-hosted FastAPI server exposing two AI4Bharat models behind one HTTPS endpoint plus a small web UI.

| Endpoint           | Model                                              | Task |
|--------------------|----------------------------------------------------|------|
| `POST /api/asr`    | `ai4bharat/indic-conformer-600m-multilingual`      | Speech → text (22 Indic langs, CTC or RNN-T) |
| `POST /api/tts`    | `ai4bharat/indic-parler-tts`                       | Text → speech (21 langs + style prompt) |
| `GET  /`           | —                                                  | Web UI (mic record + file upload + TTS form) |
| `GET  /health`     | —                                                  | Status JSON |
| `GET  /api/langs`  | —                                                  | Supported ASR language codes |

---

## 1. Prerequisites

- Python 3.10+
- A CUDA-capable GPU (TTS runs on `cuda:1`; ASR runs on CPU via ONNX)
- ffmpeg either on `$PATH` or via the `imageio-ffmpeg` pip wheel (fallback already wired in)
- HuggingFace account that has **accepted the model terms** for:
  - https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
  - https://huggingface.co/ai4bharat/indic-parler-tts
- An HF access token in `.env` as `HF_TOKEN=hf_...`

---

## 2. One-time setup

```bash
cd indic

# Install Python deps
pip install -r requirements.txt
pip install "git+https://github.com/huggingface/parler-tts.git"

# Optional but recommended (provides ffmpeg without sudo)
pip install imageio-ffmpeg
```

If the same Python environment also runs vLLM (which needs a newer
transformers), install the pinned copy into `pyenv/` instead. `scripts/start_indic.sh`
puts it first on `PYTHONPATH` for this server only:

```bash
pip install --target pyenv "transformers==4.46.1" "tokenizers==0.20.3"
```

Model weights (~6 GB) download into `hf_cache/` on the first launch. Put
`HF_TOKEN=hf_...` in the repo-root `.env` (see `.env.example`) after accepting
the model terms.

### 2a. TLS certificates

Only needed for standalone playground mode over HTTPS. Generate a self-signed cert into `certs/`:

```bash
HOST=$(hostname -f) IP=$(hostname -I | awk '{print $1}')
openssl req -x509 -newkey rsa:4096 -sha256 -days 825 -nodes \
  -keyout certs/key.pem -out certs/cert.pem \
  -subj "/CN=$HOST" \
  -addext "subjectAltName=DNS:$HOST,DNS:localhost,IP:$IP,IP:127.0.0.1"
chmod 600 certs/key.pem
```

For a trusted certificate, swap in Let's Encrypt:
```bash
export SSL_CERT=/etc/letsencrypt/live/<domain>/fullchain.pem
export SSL_KEY=/etc/letsencrypt/live/<domain>/privkey.pem
```

### 2b. Environment file

The `.env` should contain at least:

```
HF_TOKEN=hf_xxx
HUGGINGFACE_HUB_TOKEN=hf_xxx
```

---

## 3. Launch the server

```bash
cd indic
./run.sh
```

`run.sh` auto-detects:
- The bundled NVRTC libs (prepends `~/.local/.../nvidia/cu13/lib` to `LD_LIBRARY_PATH`) — required so torch's TorchScript preprocessor can load on the GPU build.
- TLS certs in `certs/` (falls back to plain HTTP if missing).
- `PORT` (default `7860`) and `SSL_CERT` / `SSL_KEY` env vars.

You should see:
```
Using NVRTC libs from: ~/.local/lib/python3.10/site-packages/nvidia/cu13/lib
Starting HTTPS on :7860 (cert=certs/cert.pem)
INFO:     Application startup complete.
INFO:     Uvicorn running on https://0.0.0.0:7860
```

To run in the background:

```bash
nohup ./run.sh > server.log 2>&1 &
disown
```

To check health:

```bash
curl -k https://127.0.0.1:7860/health
# {"asr_loaded":true,"tts_loaded":true,"asr_device":"cpu","tts_device":"cuda:1","gpu_count":8}
```

---

## 4. Open the UI

Browse to `https://<host-ip>:7860/`.

Because the cert is self-signed, Chrome/Firefox will show *"Your connection is not private"* — click **Advanced → Proceed**. Mic recording inside the UI requires this secure context to work.

---

## 5. cURL recipes

```bash
# ASR (speech -> text)
curl -k -X POST https://127.0.0.1:7860/api/asr \
  -F "audio=@/path/to/clip.wav" -F "lang=hi" -F "decoder=ctc"

# ASR with RNN-T decoder (often more accurate)
curl -k -X POST https://127.0.0.1:7860/api/asr \
  -F "audio=@/path/to/clip.wav" -F "lang=ta" -F "decoder=rnnt"

# TTS (text -> speech)
curl -k -X POST https://127.0.0.1:7860/api/tts \
  -F 'prompt=नमस्ते, आप कैसे हैं?' \
  -F 'description=Rohit speaks clearly at a moderate pace, very high quality recording.' \
  -o hello.wav

# Round-trip (TTS -> ASR)
curl -k -X POST https://127.0.0.1:7860/api/tts \
  -F 'prompt=नमस्ते, आप कैसे हैं?' \
  -F 'description=Rohit speaks clearly at a moderate pace, very high quality.' \
  -o /tmp/rt.wav && \
curl -k -X POST https://127.0.0.1:7860/api/asr \
  -F 'audio=@/tmp/rt.wav' -F 'lang=hi' -F 'decoder=ctc'
```

Supported `lang` codes (ASR — 22): `as bn brx doi gu hi kn kok ks mai ml mni mr ne or pa sa sat sd ta te ur`. Get the full name map via `GET /api/langs`.

TTS accepts free-form text in 21 languages (`description` controls voice/style — speakers, emotion, pitch, pace, background noise, accent). See the model card for named speakers (Rohit, Divya, Aditi, Jaya, Anjali, …) and emotion vocabulary.

---

## 6. Stopping / restarting

```bash
# stop
pkill -f "indic_server"

# restart
./run.sh
```

---

## 7. Common pitfalls

| Symptom | Fix |
|---|---|
| `PermissionError: ~/.cache/huggingface/modules/...` | Already handled — `../indic_server.py` redirects all HF caches to the local `hf_cache/` dir. |
| `Could not decode audio: TorchCodec is required...` | Install ffmpeg or `pip install imageio-ffmpeg` (the app's decoder uses one of these). |
| `nvrtc: error: failed to open libnvrtc-builtins.so.13.0` | Make sure you launch via `./run.sh` (which sets `LD_LIBRARY_PATH` to the bundled `nvidia/cu13/lib`). |
| `c10::complex` NVRTC compilation errors during ASR | Already handled — `../indic_server.py` disables the TensorExpr / nvFuser JIT fusers on startup. |
| `RecursiveScriptModule has no attribute _construct` | You set `PYTORCH_JIT=0` somewhere. **Don't** — it breaks `torch.jit.load`. Unset and re-launch. |
| Browser blocks mic | Self-signed HTTPS still counts as a secure context, but you must "Proceed" past the cert warning first. |
| Gated model 401 / 403 | Re-accept the terms on the HF model page while logged in, and verify `HF_TOKEN` in `.env`. |

---

## 8. File layout

```
voice-agent-studio/
├── scripts/start_indic.sh   # systemd launcher used by scripts/start_all.sh
└── services/
    ├── indic_server.py      # FastAPI server (ASR + TTS routes + UI), port 7862
    └── indic/
        ├── run.sh           # standalone playground launcher (HTTPS :7860)
        ├── requirements.txt
        ├── README.md
        ├── instructions.md  # ← you are here
        ├── certs/           # self-signed TLS material (playground only)
        ├── pyenv/           # pinned transformers==4.46.1 shim (PYTHONPATH)
        ├── static/
        │   └── index.html   # playground web UI
        └── hf_cache/        # HuggingFace cache (~5 GB of model weights)
```
