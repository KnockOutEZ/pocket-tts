#!/usr/bin/env python3
"""WhisperX alignment sidecar for pocket-tts.

Usage: whisperx_align.py <audio.wav> <output.json> [--text "known text"]

Produces word-level timestamps using WhisperX (Whisper + wav2vec2 forced alignment).
Output is a JSON array of {word, start, end} objects.
"""

import argparse
import json
import sys
import warnings
warnings.filterwarnings("ignore")

# Force imports that PyInstaller can't trace through transformers' lazy __getattr__
import transformers.pipelines  # noqa: F401
from transformers import Pipeline  # noqa: F401

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Input WAV file")
    parser.add_argument("output", help="Output JSON file")
    parser.add_argument("--text", help="Known text (unused for now, WhisperX transcribes)")
    parser.add_argument("--model", default="tiny.en", help="Whisper model size")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import whisperx

    # Load and transcribe
    model = whisperx.load_model(args.model, args.device, compute_type="float32")
    audio = whisperx.load_audio(args.audio)
    result = model.transcribe(audio, batch_size=16, language="en")

    # Force align
    model_a, metadata = whisperx.load_align_model(language_code="en", device=args.device)
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, args.device,
        return_char_alignments=False,
    )

    # Extract word timestamps
    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append({
                    "word": w["word"],
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                })

    with open(args.output, "w") as f:
        json.dump(words, f)

    # Also print summary to stderr for debugging
    print(f"Aligned {len(words)} words", file=sys.stderr)

if __name__ == "__main__":
    main()
