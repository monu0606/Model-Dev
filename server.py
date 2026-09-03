#!/usr/bin/env python3
"""Local HTTP server for ai4bharat/indic-conformer-600m-multilingual.

Loads the model once at startup and keeps it warm in memory, so repeated
transcription requests skip the ~10s model-load cost and only pay the
~1-2s inference cost.

Run via start.sh / stop.sh rather than directly, so the process is tracked
with a PID file and logs go to run/server.log. See README.md.
"""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
import torchaudio
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
TARGET_SAMPLE_RATE = 16000
SUPPORTED_LANGUAGES = {
    "hi", "bn", "ta", "te", "kn", "ml", "mr", "gu", "pa", "ur", "or",
    "as", "mai", "kok", "sa", "brx", "doi", "sat", "ks", "sd", "mni", "ne",
}
STATIC_DIR = Path(__file__).parent / "static"

# Gemma 4 audio (E2B / E4B) via llama.cpp + GGUF - see gemma4_transcribe.sh
# for why: full-precision transformers weights don't fit in RAM on this
# machine, and GPU (Metal) offload overflows its buffer allocator during
# audio decoding, so we shell out to the same CPU-only llama-mtmd-cli
# invocation that gemma4_transcribe.sh uses, rather than loading a second
# model into this process.
GGUF_DIR = Path(__file__).parent / "gguf_models"
GEMMA4_MODELS = {
    "e2b": GGUF_DIR / "e2b" / "gemma-4-E2B-it-Q4_K_M.gguf",
    "e4b": GGUF_DIR / "e4b" / "gemma-4-E4B-it-Q4_K_M.gguf",
}
GEMMA4_DEFAULT_PROMPT = "Transcribe this audio."
GEMMA4_SYSTEM_BASE = (
    "You are a speech transcription engine. Transcribe the given audio "
    "exactly as spoken, in its original language. Write numbers as digits "
    "(e.g. 1.7, not one point seven)."
)
GEMMA4_SYSTEM_NO_REASONING = (
    " Output only the transcription - no commentary, no chain-of-thought, "
    "no preamble, no newlines."
)

app = FastAPI(
    title="IndicConformer ASR",
    description="Local speech-to-text server for ai4bharat/indic-conformer-600m-multilingual",
    version="1.0.0",
)

_model = None


class TranscriptionResponse(BaseModel):
    language: str
    mode: str
    ctc: str | None = None
    rnnt: str | None = None


class Gemma4Response(BaseModel):
    size: str
    transcript: str


@app.on_event("startup")
def load_model():
    global _model
    from transformers import AutoModel

    print(f"Loading {MODEL_ID} ...")
    _model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("Model loaded and ready.")


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/languages")
def languages():
    return {"supported_languages": sorted(SUPPORTED_LANGUAGES)}


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    language: str = "hi",
    mode: str = "both",
):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly.")
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language '{language}'. See /languages.")
    if mode not in ("ctc", "rnnt", "both"):
        raise HTTPException(status_code=400, detail="mode must be one of: ctc, rnnt, both")

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    audio_bytes = await file.read()

    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        wav, sr = torchaudio.load(tmp.name)

    wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
        wav = resampler(wav)

    result = TranscriptionResponse(language=language, mode=mode)
    if mode in ("ctc", "both"):
        result.ctc = _model(wav, language, "ctc")
    if mode in ("rnnt", "both"):
        result.rnnt = _model(wav, language, "rnnt")
    return result


@app.get("/gemma4/status")
def gemma4_status():
    return {size: path.exists() for size, path in GEMMA4_MODELS.items()}


@app.get("/gemma4/defaults")
def gemma4_defaults():
    return {
        "prompt": GEMMA4_DEFAULT_PROMPT,
        "system_prompt_reasoning": GEMMA4_SYSTEM_BASE,
        "system_prompt_no_reasoning": GEMMA4_SYSTEM_BASE + GEMMA4_SYSTEM_NO_REASONING,
        "temperature": 1.0,
        "top_k": 64,
        "top_p": 0.95,
        "max_tokens": 700,
    }


@app.post("/gemma4/transcribe", response_model=Gemma4Response)
async def gemma4_transcribe(
    file: UploadFile = File(...),
    size: str = "e2b",
    reasoning: bool = False,
    system_prompt: str | None = None,
    prompt: str = GEMMA4_DEFAULT_PROMPT,
    temperature: float = 1.0,
    top_k: int = 64,
    top_p: float = 0.95,
    max_tokens: int = 700,
):
    if size not in GEMMA4_MODELS:
        raise HTTPException(status_code=400, detail=f"size must be one of: {', '.join(GEMMA4_MODELS)}")
    if shutil.which("llama-mtmd-cli") is None:
        raise HTTPException(status_code=503, detail="llama-mtmd-cli not found. Install with: brew install llama.cpp")

    model_path = GEMMA4_MODELS[size]
    mmproj_path = model_path.parent / "mmproj-BF16.gguf"
    if not model_path.exists() or not mmproj_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"{size.upper()} weights not downloaded yet. Run: ./gemma4_transcribe.sh test_audio.wav {size}",
        )

    if system_prompt is None:
        system_prompt = GEMMA4_SYSTEM_BASE + ("" if reasoning else GEMMA4_SYSTEM_NO_REASONING)

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    audio_bytes = await file.read()

    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        try:
            proc = subprocess.run(
                [
                    "llama-mtmd-cli",
                    "-m", str(model_path),
                    "--mmproj", str(mmproj_path),
                    "--audio", tmp.name,
                    "-sys", system_prompt,
                    "-p", prompt,
                    "-ngl", "0", "--jinja",
                    "-n", str(max_tokens),
                    "--temp", str(temperature),
                    "--top-k", str(top_k),
                    "--top-p", str(top_p),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="Gemma 4 inference timed out after 300s.")

    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"llama-mtmd-cli failed:\n{proc.stderr[-2000:]}")

    # The model streams its chain-of-thought before the final answer, separated
    # by a "channel" marker (observed empirically - see gemma4_transcribe.sh).
    # Take everything after the last such marker as the transcript.
    match = re.search(r"channel\|>(.*)$", proc.stdout, re.DOTALL)
    transcript = (match.group(1) if match else proc.stdout).strip()
    if not transcript:
        raise HTTPException(status_code=500, detail=f"Empty transcript. Raw output:\n{proc.stdout[-2000:]}")

    return Gemma4Response(size=size, transcript=transcript)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8420)
