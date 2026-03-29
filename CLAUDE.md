# CLAUDE.md

## What This Is

A fork of the Rust/Candle port of Kyutai's pocket-tts. We added word-level timestamp support (native ONNX alignment) for **Vordetta** — a local-first Speechify-like ebook reader built with Rust/Tauri that reads aloud while highlighting words in sync.

Fork chain: `kyutai/pocket-tts` (Python) → `babybirdprd/pocket-tts` (Rust port) → `KnockOutEZ/pocket-tts` (this repo, word timestamps)

Active branch: `feat/word-timestamps` — the Tauri app depends on this branch as a git crate dependency.

## Repo Layout

```
crates/pocket-tts/          Core TTS library (the crate consumers import)
  src/tts_model.rs           Main orchestrator — TTSModel + all public API
  src/alignment/             Native ONNX word-timestamp alignment (new)
    aligner.rs               NativeAligner — ONNX inference pipeline
    forced_align.rs          Viterbi algorithm + CTC target builder + WordTimestamp
  src/models/                FlowLM, Mimi, SEANet, Transformer
  src/modules/               Attention, MLP, RoPE, SDPA, Conv
  src/audio.rs               WAV I/O, resampling (Rubato + Cubic polynomial)
  src/conditioners/          SentencePiece text tokenizer
  config/                    YAML model configs

crates/pocket-tts-cli/      HTTP server (Axum) + React web UI
  src/server/                Routes, handlers, state
  web/                       React 19 + Tailwind + shadcn/ui (Vite)

crates/pocket-tts-bindings/  Python bindings (PyO3, not used by Vordetta)
python-reference/            Original Python impl (parity testing only)
scripts/export_wav2vec2_onnx.py  One-time ONNX model export script
.github/workflows/           CI
```

## Build Commands

```bash
# Always build in release mode — debug is unusably slow for ML inference
cargo build --release

# With Metal acceleration (macOS Apple Silicon)
cargo build --release --features metal

# Run tests (needs HF_TOKEN for gated model weights)
HF_TOKEN=hf_xxx cargo test --release --all-targets

# Run the dev server with web UI
cargo run --release -p pocket-tts-cli -- serve

# Benchmarks
cargo bench --release
```

## Architecture: TTS Pipeline

```
Text → SentencePiece tokenizer → FlowLM (LSD decoding) → Mimi codec (SEANet) → 24kHz audio
```

## Architecture: Word Timestamps (feat/word-timestamps)

```
Generated audio (24kHz) → resample to 16kHz (Cubic polynomial)
  → wav2vec2-large-960h-lv60-self INT8 ONNX Runtime inference
  → CTC emission probabilities
  → Viterbi forced alignment against known text (pure Rust)
  → word timestamps (~20-50ms accuracy, ~300-500ms total)
```

`NativeAligner` in `src/alignment/aligner.rs` manages the ONNX session. Models load once (~5-8s), alignment is ~300-500ms per sentence. No external process or Python dependency.

## Public API (what the Tauri app consumes)

```rust
// Download all models on app install/first launch (~450MB, cached forever)
TTSModel::preload_weights("b6369a24", TTSModel::VOICES)?;

// Load TTS + load wav2vec2 ONNX model when user opens a book (~5-8s)
let model = TTSModel::load_with_alignment("b6369a24")?;

// Per sentence: generate audio + get word timestamps (~1.5-3s)
let result = model.generate_sentence_with_timestamps("Hello world.", &voice_state)?;
// result.audio: Tensor, result.word_timestamps: Vec<WordTimestamp>

// Full text with automatic sentence splitting
let result = model.generate_with_timestamps("Long text here...", &voice_state)?;

// Voices: alba, marius, javert, jean, fantine, cosette, eponine, azelma
```

## Key Types

- `TTSModel` — main model, holds FlowLM + Mimi + optional NativeAligner
- `ModelState` — voice conditioning state (from predefined voice or cloned audio)
- `GenerationResult` — `{ audio: Tensor, word_timestamps: Vec<WordTimestamp> }`
- `WordTimestamp` — `{ word: String, start_sec: f32, end_sec: f32 }`
- `NativeAligner` — owns ONNX Runtime session for wav2vec2-large, runs CTC + Viterbi alignment

## Things to Know

- **Always `--release`**. Debug builds are 10-50x slower — never test audio quality or performance without it.
- **`HF_TOKEN` required** for downloading gated model weights from `kyutai/pocket-tts`.
- **Config discovery**: `find_config_path()` looks in `crates/pocket-tts/config/` then `python-reference/pocket_tts/config/`.
- **WASM gating**: Alignment code is `#[cfg(not(target_arch = "wasm32"))]` — it only compiles on native targets.
- **Sample rates**: TTS outputs 24kHz, wav2vec2 expects 16kHz. Resampling via `resample_for_alignment()` in `audio.rs` (Cubic polynomial).
- **`ort` crate**: ONNX Runtime bindings. Pulls in `ort` + `ndarray`. The ONNX model is downloaded from HF Hub and cached alongside TTS weights.
- **Numerical parity**: Parity tests in `crates/pocket-tts/tests/parity_tests.rs` compare against reference tensors in `assets/`.
- **No sentencepiece crate**: Removed due to protobuf conflicts with onnxruntime. Uses `tokenizers` crate's built-in SentencePiece support instead.
- **Pinned model versions**: wav2vec2-large-960h-lv60-self (INT8 ONNX). Never auto-update or re-export without re-validating alignment accuracy.

## What NOT to Touch

- `python-reference/` — read-only reference impl, do not modify
- `assets/*.safetensors` — reference tensors for parity tests
- Model architecture files (`models/`, `modules/`) — unless specifically working on model changes
- Pinned dependency versions in `Cargo.toml` (especially `candle-*` at 0.9.1)

## CI

Standard Rust CI (build + test) on push/PR. No special sidecar build step — alignment is fully in-process via ONNX Runtime. No PyInstaller, no Python packaging.
