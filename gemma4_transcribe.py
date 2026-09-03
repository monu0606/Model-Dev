#!/usr/bin/env python3
"""Transcribe audio with Gemma 4's audio-capable variants (E2B / E4B).

Only E2B and E4B are wired up here — the 12B variant also supports audio
but is deliberately excluded, it's too large for this machine.

Usage:
    python gemma4_transcribe.py <audio_file> [e2b|e4b]

Example:
    python gemma4_transcribe.py test_audio.wav e2b

Notes:
    - Audio clips must be <= 30 seconds (a Gemma 4 audio-encoder limit).
    - Models download from Hugging Face on first use (E2B ~10GB, E4B ~16GB)
      and are cached under .hf_home/. There's no gating/login needed
      (Apache 2.0, unlike the IndicConformer model).
"""
import sys

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

MODEL_IDS = {
    "e2b": "google/gemma-4-E2B-it",
    "e4b": "google/gemma-4-E4B-it",
}

PROMPT = (
    "Transcribe the following speech segment in its original language. "
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits, i.e. write 1.7 and not "
    "one point seven, and write 3 instead of three."
)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    audio_path = sys.argv[1]
    size = sys.argv[2].lower() if len(sys.argv) > 2 else "e2b"

    if size not in MODEL_IDS:
        print(f"Unknown size '{size}'. Choose one of: {', '.join(MODEL_IDS)}")
        sys.exit(1)

    model_id = MODEL_IDS[size]

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading {model_id} onto {device} ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        dtype="auto",
        device_map=device,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "audio", "audio": audio_path},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    print(f"Transcribing {audio_path} with {size.upper()} ...")
    outputs = model.generate(**inputs, max_new_tokens=512)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    transcript = processor.parse_response(response, prefix=inputs["input_ids"])

    print(f"{size.upper()} transcription:", transcript)


if __name__ == "__main__":
    main()
