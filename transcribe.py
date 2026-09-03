#!/usr/bin/env python3
"""Transcribe an audio file with ai4bharat/indic-conformer-600m-multilingual.

Usage:
    python transcribe.py <audio_file> [language_code] [ctc|rnnt|both]

Example:
    python transcribe.py test_audio.wav hi both
"""
import sys

import torch
import torchaudio
from transformers import AutoModel

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
TARGET_SAMPLE_RATE = 16000


def load_audio(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    wav = torch.mean(wav, dim=0, keepdim=True)  # stereo -> mono
    if sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
        wav = resampler(wav)
    return wav


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    audio_path = sys.argv[1]
    language_code = sys.argv[2] if len(sys.argv) > 2 else "hi"
    mode = sys.argv[3] if len(sys.argv) > 3 else "both"

    print(f"Loading model {MODEL_ID} ...")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)

    print(f"Loading audio: {audio_path}")
    wav = load_audio(audio_path)

    if mode in ("ctc", "both"):
        transcription_ctc = model(wav, language_code, "ctc")
        print("CTC Transcription: ", transcription_ctc)

    if mode in ("rnnt", "both"):
        transcription_rnnt = model(wav, language_code, "rnnt")
        print("RNNT Transcription:", transcription_rnnt)


if __name__ == "__main__":
    main()
