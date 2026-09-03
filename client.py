#!/usr/bin/env python3
"""Send an audio file to a running IndicConformer server (see start.sh).

Usage:
    python client.py <audio_file> [language_code] [ctc|rnnt|both] [port]

Example:
    python client.py test_audio.wav hi both
"""
import sys

import requests

DEFAULT_PORT = 8420


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    audio_path = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else "hi"
    mode = sys.argv[3] if len(sys.argv) > 3 else "both"
    port = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_PORT

    url = f"http://127.0.0.1:{port}/transcribe"
    with open(audio_path, "rb") as f:
        response = requests.post(
            url,
            params={"language": language, "mode": mode},
            files={"file": (audio_path, f)},
        )

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}", file=sys.stderr)
        sys.exit(1)

    data = response.json()
    if data.get("ctc") is not None:
        print("CTC Transcription: ", data["ctc"])
    if data.get("rnnt") is not None:
        print("RNNT Transcription:", data["rnnt"])


if __name__ == "__main__":
    main()
