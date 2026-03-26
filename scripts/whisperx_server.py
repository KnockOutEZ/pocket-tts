#!/usr/bin/env python3
"""WhisperX alignment server — keeps models loaded in memory.

Starts an HTTP server that accepts WAV files and returns word timestamps.
Models load once at startup (~10s), then each alignment takes ~2-4s.

Usage: python3 whisperx_server.py [--port 9876] [--model tiny.en]
"""

import argparse
import json
import tempfile
import os
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--model", default="tiny.en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preload", action="store_true",
                        help="Download all models and exit (no server)")
    args = parser.parse_args()

    import whisperx
    from http.server import HTTPServer, BaseHTTPRequestHandler

    print(f"Loading WhisperX model ({args.model})...", flush=True)
    model = whisperx.load_model(args.model, args.device, compute_type="float32")

    print("Loading alignment model (wav2vec2-base)...", flush=True)
    model_a, metadata = whisperx.load_align_model(language_code="en", device=args.device)

    if args.preload:
        print("All models downloaded and verified.", flush=True)
        return

    print(f"Ready on port {args.port}", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            # Parse request: JSON with wav_path and optional text
            try:
                req = json.loads(body)
                wav_path = req["wav_path"]
                # text = req.get("text", "")  # unused for now
            except (json.JSONDecodeError, KeyError):
                self.send_error(400, "Expected JSON with wav_path")
                return

            try:
                audio = whisperx.load_audio(wav_path)
                result = model.transcribe(audio, batch_size=16, language="en")
                result = whisperx.align(
                    result["segments"], model_a, metadata, audio, args.device,
                    return_char_alignments=False,
                )

                words = []
                for seg in result["segments"]:
                    for w in seg.get("words", []):
                        if "start" in w and "end" in w:
                            words.append({
                                "word": w["word"],
                                "start": round(w["start"], 3),
                                "end": round(w["end"], 3),
                            })

                resp = json.dumps(words).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            except Exception as e:
                self.send_error(500, str(e))

        def log_message(self, format, *args):
            pass  # Suppress request logs

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    server.serve_forever()

if __name__ == "__main__":
    main()
