#!/usr/bin/env python3
"""Export wav2vec2-large-960h-lv60-self to ONNX with INT8 quantization.

Run once. Produces:
  - wav2vec2-large-int8.onnx (~320MB)
  - vocab.json (character vocabulary)

Requirements:
  pip install optimum[exporters] onnxruntime transformers
"""
import json
import shutil
from pathlib import Path
from transformers import Wav2Vec2CTCTokenizer

MODEL_ID = "facebook/wav2vec2-large-960h-lv60-self"
OUT_DIR = Path("wav2vec2-large-onnx")
PREPROCESSED_DIR = Path("preprocessed")
FINAL_MODEL = Path("wav2vec2-large-int8.onnx")

def main():
    import subprocess
    import sys

    # 1. Export to ONNX
    print(f"[1/3] Exporting {MODEL_ID} to ONNX...")
    subprocess.run([
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model", MODEL_ID,
        "--task", "ctc",
        str(OUT_DIR),
    ], check=True)

    # 2. Preprocess for quantization
    print("[2/3] Preprocessing for quantization...")
    subprocess.run([
        sys.executable, "-m", "onnxruntime.quantization.preprocess",
        "--input", str(OUT_DIR / "model.onnx"),
        "--output", str(PREPROCESSED_DIR),
    ], check=True)

    # 3. Quantize to INT8
    print("[3/3] Quantizing to INT8...")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        str(PREPROCESSED_DIR / "model.onnx"),
        str(FINAL_MODEL),
        weight_type=QuantType.QInt8,
    )

    # 4. Extract vocab.json
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(MODEL_ID)
    vocab = tokenizer.get_vocab()
    with open("vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"\nDone! Files:")
    print(f"  {FINAL_MODEL} ({FINAL_MODEL.stat().st_size / 1e6:.0f} MB)")
    print(f"  vocab.json ({len(vocab)} tokens)")

    # Cleanup
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    shutil.rmtree(PREPROCESSED_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()
