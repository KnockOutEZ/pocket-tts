#!/usr/bin/env python3
"""WhisperX alignment server — keeps models loaded in memory.

Starts an HTTP server that accepts WAV files and returns word timestamps.
Models load once at startup (~10s), then each alignment takes ~2-4s.

Usage:
  python3 whisperx_server.py [--port 9876] [--model tiny.en]
  python3 whisperx_server.py --check     # verify models cached, no loading
  python3 whisperx_server.py --preload   # download + load models, then exit
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

# Patch: torchcodec may not be available on all platforms (especially in PyInstaller).
# transformers.audio_utils checks importlib.metadata.version("torchcodec") and crashes.
import importlib.metadata as _md
_orig_version = _md.version
def _patched_version(name):
    if name == "torchcodec":
        return "0.0.0"
    return _orig_version(name)
_md.version = _patched_version
try:
    _orig_dist = _md.distribution
    def _patched_dist(name):
        if name == "torchcodec":
            return "0.0.0"
        return _orig_dist(name)
    _md.distribution = _patched_dist
except AttributeError:
    pass

# Pin model versions — never auto-update, stay stable
WHISPER_MODEL = "tiny.en"
# wav2vec2 alignment model is pinned by whisperx internally
# (wav2vec2_fairseq_base_ls960_asr_ls960.pth)

def check_models_cached():
    """Verify model files exist in cache without loading them. Instant."""
    import torch
    cache_dir = torch.hub.get_dir()
    checkpoints_dir = os.path.join(cache_dir, "checkpoints")

    # Check wav2vec2 alignment model
    wav2vec2_file = os.path.join(
        checkpoints_dir, "wav2vec2_fairseq_base_ls960_asr_ls960.pth"
    )
    if not os.path.exists(wav2vec2_file):
        return False, f"wav2vec2 model not cached: {wav2vec2_file}"

    # Check whisper model — faster-whisper stores in HF cache
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    whisper_dir = os.path.join(hf_cache, f"models--Systran--faster-whisper-{WHISPER_MODEL}")
    if not os.path.exists(whisper_dir):
        return False, f"Whisper model not cached: {whisper_dir}"

    return True, "All models cached"


def download_models(args):
    """Download models by loading them once. Slow (~15-20s) but only needed first time."""
    import whisperx

    print(f"Downloading WhisperX model ({args.model})...", flush=True)
    model = whisperx.load_model(args.model, args.device, compute_type="float32")
    del model

    print("Downloading alignment model (wav2vec2-base)...", flush=True)
    model_a, metadata = whisperx.load_align_model(language_code="en", device=args.device)
    del model_a, metadata

    print("All models downloaded.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--model", default=WHISPER_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--check", action="store_true",
                        help="Check if models are cached (instant, no loading)")
    parser.add_argument("--preload", action="store_true",
                        help="Download models if not cached, then exit")
    args = parser.parse_args()

    # --check: instant cache verification
    if args.check:
        cached, msg = check_models_cached()
        if cached:
            print("OK", flush=True)
        else:
            print(f"MISSING: {msg}", flush=True)
            exit(1)
        return

    # --preload: download if needed, then exit
    if args.preload:
        cached, _ = check_models_cached()
        if cached:
            print("All models already cached.", flush=True)
        else:
            download_models(args)
        return

    # Server mode: load models and serve
    import whisperx
    from http.server import HTTPServer, BaseHTTPRequestHandler

    print(f"Loading WhisperX model ({args.model})...", flush=True)
    model = whisperx.load_model(args.model, args.device, compute_type="float32")

    print("Loading alignment model (wav2vec2-base)...", flush=True)
    model_a, metadata = whisperx.load_align_model(language_code="en", device=args.device)

    print(f"Ready on port {args.port}", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                req = json.loads(body)
                wav_path = req["wav_path"]
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

    import socket
    class ReusableHTTPServer(HTTPServer):
        allow_reuse_address = True
        def server_bind(self):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            super().server_bind()

    server = ReusableHTTPServer(("127.0.0.1", args.port), Handler)
    server.serve_forever()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
