# AI4Bharat Indic Speech Stack

Self-hosted playground for two AI4Bharat models behind one FastAPI server + UI.

| Endpoint   | Model                                              | Task |
|------------|----------------------------------------------------|------|
| `POST /api/asr` | `ai4bharat/indic-conformer-600m-multilingual` | Speech → text (22 Indic langs, CTC/RNN-T) |
| `POST /api/tts` | `ai4bharat/indic-parler-tts`                  | Text → speech (21 langs + style prompt + optional seed) |
| `GET  /`        | —                                              | Web UI |

## Run

```bash
# 1. install (one-time)
pip install -r requirements.txt
pip install "git+https://github.com/huggingface/parler-tts.git"

# 2. accept the gated models on huggingface.co (logged-in browser):
#    https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
#    https://huggingface.co/ai4bharat/indic-parler-tts

# 3. add your token to .env  (HF_TOKEN=hf_xxx)

# 4. launch
./run.sh           # or:  python3 -m uvicorn indic_server:app --app-dir .. --host 0.0.0.0 --port 7860
```

Open `http://<host>:7860/`.

GPUs: ASR pinned to `cuda:0`, TTS to `cuda:1` (falls back to `cuda:0` if single-GPU).

`/api/tts` accepts an optional `seed` form field. Use the same seed with the same voice description to keep separately generated sentences closer to the same sampled speaker.
