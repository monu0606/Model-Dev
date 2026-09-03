# IndicConformer ASR (local, macOS)

Runs [ai4bharat/indic-conformer-600m-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
locally — a 600M-parameter speech-to-text model covering 22 Indian languages,
with both CTC and RNNT decoding.

Two ways to use it:
- **Server mode** (recommended): start it once, transcribe many files fast (~1-2s each).
- **One-shot CLI**: simplest, but reloads the model every run (~10-15s overhead each time).

## Prerequisites

- macOS (tested on Apple Silicon; should also work on Intel Macs, just slower)
- [Homebrew](https://brew.sh) with Python 3.11+ (`brew install python@3.11`)
- A free [Hugging Face](https://huggingface.co) account
- ~3 GB free disk space

## Setup

On a fresh clone (new machine), run the full bootstrap instead — it does
everything below plus llama.cpp + Gemma 4 weights (see
[Gemma 4 audio](#gemma-4-audio-experimental-e2b--e4b-only)):

```bash
./bootstrap.sh
```

Or, for just the IndicConformer path:

```bash
./setup.sh
```

This creates a `.venv`, installs pinned dependencies from `requirements.txt`,
and points the Hugging Face cache at a project-local `.hf_home/` folder
(rather than the default `~/.cache/huggingface` — see [Troubleshooting](#troubleshooting)
if that folder has permission issues on your machine).

Model weights (IndicConformer, Gemma 4 GGUF) are never committed to git —
they're downloaded fresh by these scripts on whatever machine needs them, so
the repo itself stays small and cloning is fast.

### Getting model access (one-time, per person)

This model is **gated** — a free Hugging Face account is required, and you
must accept its terms once:

1. Visit the [model page](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
   while logged in, and click **"Agree and access repository"** (access is
   granted automatically).
2. Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   (the default "read" permission is enough).
3. Log in from the terminal so the token never has to be pasted into chat/scripts:
   ```bash
   source .venv/bin/activate
   hf auth login
   ```
   Paste the token when prompted.

`setup.sh` tells you if this step is still needed.

## Starting and stopping the server

```bash
./start.sh    # loads the model into memory, starts a local API on :8420
./stop.sh     # shuts it down
```

- `start.sh` is safe to run repeatedly — it no-ops if the server is already up.
- First start after a fresh setup downloads ~2.56 GB (one-time); after that,
  starting takes ~10-15s (model load time), and every start after that reuses
  the cache in `.hf_home/`.
- Logs: `run/server.log`. PID: `run/server.pid`.
- Change the port with `PORT=9000 ./start.sh`.

Check it's alive:
```bash
curl http://127.0.0.1:8420/health
```

Interactive API docs (Swagger UI) while the server is running:
[http://127.0.0.1:8420/docs](http://127.0.0.1:8420/docs)

## Transcribing audio

### Option A — web UI (easiest)

With the server running, open **[http://127.0.0.1:8420](http://127.0.0.1:8420)**
in your browser. Drag in an audio file (or click to choose one), pick a
language and decoding mode, and hit Transcribe. Runs entirely on your
machine — no audio is uploaded anywhere external.

### Option B — server + client script (fast, once the server is running)

```bash
python client.py path/to/audio.wav hi both
```

Args: `<audio_file> [language_code] [ctc|rnnt|both]`. Defaults: `hi`, `both`.

Or with curl directly:
```bash
curl -X POST "http://127.0.0.1:8420/transcribe?language=hi&mode=both" \
  -F "file=@path/to/audio.wav"
```

### Option C — one-shot CLI (no server needed, but slower)

```bash
python transcribe.py path/to/audio.wav hi both
```

Same arguments as `client.py`. Loads the model fresh every run.

### Supported language codes

`hi` Hindi · `bn` Bengali · `ta` Tamil · `te` Telugu · `kn` Kannada ·
`ml` Malayalam · `mr` Marathi · `gu` Gujarati · `pa` Punjabi · `ur` Urdu ·
`or` Odia · `as` Assamese · `mai` Maithili · `kok` Konkani · `sa` Sanskrit ·
`brx` Bodo · `doi` Dogri · `sat` Santali · `ks` Kashmiri · `sd` Sindhi ·
`mni` Manipuri · `ne` Nepali

### Audio requirements

Any format `torchaudio`/`torchcodec` can decode (wav, flac, mp3, ...). Audio
is automatically converted to mono and resampled to 16kHz if needed — you
don't need to pre-process it.

## Gemma 4 audio (experimental, E2B / E4B only)

`gemma4_transcribe.sh` tests Google's Gemma 4 audio-capable variants
(E2B, E4B — the 12B variant is skipped, it needs more RAM than this
machine has) via [llama.cpp](https://github.com/ggml-org/llama.cpp)'s
`llama-mtmd-cli`, using quantized GGUF weights from
[unsloth](https://huggingface.co/unsloth).

```bash
./bootstrap.sh                                 # one-time: installs llama.cpp + downloads E2B weights
./gemma4_transcribe.sh test_audio.wav e2b      # or e4b
```

(`bootstrap.sh` handles `brew install llama.cpp` and the weight download;
run `./bootstrap.sh e2b e4b` to fetch both sizes up front, or do it manually
with `brew install llama.cpp` + `./gemma4_transcribe.sh --download-only e2b`.)

First run per size downloads the quantized model + a required BF16
multimodal projector (`mmproj`) into `gguf_models/<size>/` (~4.1GB for
E2B, ~6GB for E4B) — no HF login needed (Apache 2.0, unlike
IndicConformer). Runs CPU-only (`-ngl 0`): on an 8GB-RAM Mac, GPU
(Metal) offload overflows Metal's buffer allocator during audio
decoding, even with free system RAM — CPU inference doesn't hit that
ceiling, it's just slower. If you have more RAM/VRAM to spare, dropping
`-ngl 0` (edit the script) lets more layers run on GPU.

There's also `gemma4_transcribe.py`, a transformers-based one-shot CLI
for the same models — it loads full-precision safetensors weights
(E2B: 10.25GB, E4B: 16GB) and needs far more RAM than this machine has,
so it's untested here. Prefer the `.sh`/GGUF path on similarly
low-spec hardware.

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI for uploading and transcribing audio |
| `/health` | GET | `{"status": "ok", "model_loaded": true}` once ready |
| `/languages` | GET | List of supported language codes |
| `/transcribe` | POST | Multipart file upload (`file`), query params `language` (default `hi`), `mode` (`ctc`\|`rnnt`\|`both`, default `both`) |
| `/docs` | GET | Interactive Swagger UI |

## Project layout

```
bootstrap.sh      One-shot setup for a fresh clone: env + IndicConformer deps + llama.cpp + Gemma 4 weights
setup.sh          Python env + IndicConformer dependencies only
start.sh          Start the server (idempotent)
stop.sh           Stop the server (idempotent)
server.py         FastAPI app — loads the model once, serves the web UI + /transcribe
static/index.html Web UI (upload audio, pick language/mode, view transcript)
client.py         CLI client that calls the running server
transcribe.py     Standalone one-shot CLI (no server)
gemma4_transcribe.sh  Gemma 4 E2B/E4B audio test via llama.cpp + GGUF (recommended, low-RAM friendly)
gemma4_transcribe.py  Gemma 4 E2B/E4B audio test via transformers (needs much more RAM)
requirements.txt  Pinned dependency versions
.venv/            Python virtualenv (created by setup.sh, git-ignored)
.hf_home/         Project-local HF cache + auth token (created by setup.sh, git-ignored)
gguf_models/      Downloaded GGUF weights for gemma4_transcribe.sh (git-ignored)
run/              PID file + server logs (created by start.sh, git-ignored)
```

## Troubleshooting

**"Access to model ... is restricted"** — you haven't accepted the model's
terms yet, or aren't logged in. See [Getting model access](#getting-model-access-one-time-per-person).

**`ModuleNotFoundError: No module named 'torchcodec'`** — `torchaudio` 2.11+
delegates audio decoding to `torchcodec`; it's in `requirements.txt`, so this
should only happen if you installed packages manually instead of via
`setup.sh`/`requirements.txt`.

**`PermissionError` writing to `~/.cache/huggingface`** — on some machines
`~/.cache` ends up owned by `root` (check with `ls -ld ~/.cache`). This
project avoids that by using `.hf_home/` inside the project folder instead.
If you want to use the default global HF cache instead, fix ownership with
`sudo chown -R "$(whoami)" ~/.cache` and remove the `HF_HOME` line from
`.venv/bin/activate`.

**Slow inference / CPU only** — the model's own code only requests
`CUDAExecutionProvider`/`CPUExecutionProvider` from ONNX Runtime, so on a Mac
it always runs on CPU even though `onnxruntime` reports `CoreMLExecutionProvider`
as available. This is fine for short clips (~1-2s each once the server is
warm) but is a known limit if you need to process long audio faster.

**Port already in use** — another process is on 8420. Run with
`PORT=9000 ./start.sh` instead.
