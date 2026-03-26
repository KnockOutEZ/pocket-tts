"""Compare WhisperX word timestamps vs our whisper-cli sidecar on the same TTS audio."""

import json
import sys
import whisperx
import torch

AUDIO_FILE = "/tmp/tts_symbols.wav"
OURS_FILE = "/tmp/ours_symbols.json"

def main():
    device = "cpu"
    compute_type = "float32"

    print("Loading WhisperX model (small.en)...")
    model = whisperx.load_model("small.en", device, compute_type=compute_type)

    print(f"Transcribing {AUDIO_FILE}...")
    audio = whisperx.load_audio(AUDIO_FILE)
    result = model.transcribe(audio, batch_size=16, language="en")

    print("Loading alignment model...")
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)

    print("Aligning...")
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, device,
        return_char_alignments=False
    )

    # Extract word timestamps
    wx_words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                wx_words.append({
                    "word": w["word"],
                    "start": w["start"],
                    "end": w["end"],
                })

    # Save WhisperX results
    with open("/tmp/whisperx_symbols.json", "w") as f:
        json.dump(wx_words, f, indent=2)

    print(f"\nWhisperX: {len(wx_words)} words")
    print(f"{'WORD':<22} {'START':>8} {'END':>8} {'DUR':>6}")
    print("-" * 48)
    for w in wx_words:
        dur = (w["end"] - w["start"]) * 1000
        print(f"{w['word']:<22} {w['start']:>6.3f}s {w['end']:>6.3f}s {dur:>4.0f}ms")

    # Compare with ours if available
    try:
        ours = json.load(open(OURS_FILE))
        print(f"\n{'='*62}")
        print(f"COMPARISON: WhisperX vs Our whisper-cli ({len(wx_words)} vs {len(ours)} words)")
        print(f"{'='*62}")
        print(f"{'WORD':<20} {'WHISPERX':>14} {'OURS':>14} {'DELTA':>8}")
        print("-" * 60)

        oi = 0
        deltas = []
        for w in wx_words:
            wt = w["word"].strip().lower()
            wt_clean = "".join(c for c in wt if c.isalnum())
            if not wt_clean:
                continue
            while oi < len(ours):
                ot_clean = "".join(c for c in ours[oi]["word"].lower() if c.isalnum())
                if ot_clean == wt_clean or wt_clean.startswith(ot_clean) or ot_clean.startswith(wt_clean):
                    d = abs(ours[oi]["start"] - w["start"]) * 1000
                    deltas.append(d)
                    flag = " !!!" if d > 200 else ""
                    print(f"{wt:<20} {w['start']:>6.3f}-{w['end']:>6.3f}s {ours[oi]['start']:>6.3f}-{ours[oi]['end']:>6.3f}s {d:>5.0f}ms{flag}")
                    oi += 1
                    break
                oi += 1

        if deltas:
            print(f"\n=== STATS ({len(deltas)} matched) ===")
            print(f"Average delta: {sum(deltas)/len(deltas):.0f}ms")
            print(f"Median delta:  {sorted(deltas)[len(deltas)//2]:.0f}ms")
            print(f"Max delta:     {max(deltas):.0f}ms")
            print(f"Within 50ms:   {sum(1 for d in deltas if d<=50)}/{len(deltas)} ({100*sum(1 for d in deltas if d<=50)//len(deltas)}%)")
            print(f"Within 100ms:  {sum(1 for d in deltas if d<=100)}/{len(deltas)} ({100*sum(1 for d in deltas if d<=100)//len(deltas)}%)")
            print(f"Within 200ms:  {sum(1 for d in deltas if d<=200)}/{len(deltas)} ({100*sum(1 for d in deltas if d<=200)//len(deltas)}%)")
    except FileNotFoundError:
        print("\nNo ours file found — run the Rust timestamp_compare example first.")

if __name__ == "__main__":
    main()
