#!/usr/bin/env python3
"""Export wav2vec2-large-960h-lv60-self to ONNX.

Run once. Produces:
  - model.onnx (FP32, ~1.26GB) or quantized via optimum
  - vocab.json (character vocabulary)

Requirements:
  pip install "optimum[onnxruntime]" onnxruntime transformers torch
"""
import json
import shutil
from pathlib import Path

MODEL_ID = "facebook/wav2vec2-large-960h-lv60-self"
OUT_DIR = Path("wav2vec2-large-onnx")
FINAL_DIR = Path("wav2vec2-onnx-output")

def main():
    # 1. Export to ONNX + quantize via optimum (handles weight-norm layers correctly)
    print(f"[1/2] Exporting {MODEL_ID} to ONNX (quantized)...")
    from optimum.onnxruntime import ORTModelForCTC, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    # First export FP32
    if not (OUT_DIR / "model.onnx").exists():
        from optimum.exporters.onnx import main_export
        main_export(
            model_name_or_path=MODEL_ID,
            output=OUT_DIR,
            task="automatic-speech-recognition",
        )
    else:
        print("  (FP32 ONNX already exported)")

    # Quantize with optimum's built-in quantizer (handles weight-norm correctly)
    print("  Quantizing with optimum...")
    FINAL_DIR.mkdir(exist_ok=True)
    try:
        quantizer = ORTQuantizer.from_pretrained(str(OUT_DIR))
        qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
        quantizer.quantize(save_dir=str(FINAL_DIR), quantization_config=qconfig)
        final_model = FINAL_DIR / "model_quantized.onnx"
    except Exception as e:
        print(f"  Quantization failed ({e}), using FP32 model instead")
        shutil.copy(OUT_DIR / "model.onnx", FINAL_DIR / "model.onnx")
        final_model = FINAL_DIR / "model.onnx"

    # 2. Extract vocab.json
    print("[2/2] Extracting vocabulary...")
    from transformers import Wav2Vec2CTCTokenizer
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(MODEL_ID)
    vocab = tokenizer.get_vocab()
    with open(str(FINAL_DIR / "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)

    # Also copy to repo root for easy upload
    shutil.copy(final_model, "wav2vec2-large-int8.onnx")
    shutil.copy(FINAL_DIR / "vocab.json", "vocab.json")

    size_mb = Path("wav2vec2-large-int8.onnx").stat().st_size / 1e6
    print(f"\nDone! Files:")
    print(f"  wav2vec2-large-int8.onnx ({size_mb:.0f} MB)")
    print(f"  vocab.json ({len(vocab)} tokens)")

    # Cleanup intermediates
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    shutil.rmtree(FINAL_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()
