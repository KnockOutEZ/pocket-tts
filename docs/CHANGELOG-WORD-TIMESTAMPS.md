## Native ONNX Alignment (replacing WhisperX)

**What changed:** Replaced the WhisperX Python sidecar server with in-process ONNX inference (wav2vec2-large-960h-lv60-self INT8) + pure Rust Viterbi forced alignment.

**Why:** WhisperX ran Whisper transcription (redundant — we already know the text), took 2-4s per sentence, required Python/PyTorch (~500MB process), and needed a PyInstaller binary for distribution.

**New approach:** Since we know the text, skip ASR entirely. Run wav2vec2-large through ONNX Runtime to get CTC emission probabilities, then Viterbi-align against known text.

| Metric | Before (WhisperX) | After (Native ONNX) |
|--------|-------------------|---------------------|
| Alignment latency | 2-4s | ~300-500ms |
| Download size | ~530MB | ~450MB |
| Model load time | ~15s | ~5-8s |
| Memory | ~800MB | ~450-550MB |
| Python required | Yes | No |

**Files added/changed:**
- `crates/pocket-tts/src/alignment/aligner.rs` — rewritten: NativeAligner with ONNX inference
- `crates/pocket-tts/src/alignment/forced_align.rs` — Viterbi algorithm + CTC target builder + path-to-word-timestamps
- `crates/pocket-tts/src/audio.rs` — added `resample_for_alignment` (Cubic polynomial)
- `scripts/export_wav2vec2_onnx.py` — one-time ONNX export script

**Files deleted:**
- `scripts/whisperx_server.py`
- `.github/workflows/build-whisperx.yml`

---

# Word-Level Timestamps — Branch Summary

**Branch:** `feat/word-timestamps`
**Base:** `main`
**Net changes:** 9 files added/modified, ~1090 lines added

## What Was Added

Word-level timestamp support for read-aloud applications. Given text, the library generates audio AND returns per-word start/end times with ~20-50ms accuracy using WhisperX (Whisper transcription + wav2vec2 CTC forced alignment).

## Files

| File | Purpose |
|------|---------|
| `crates/pocket-tts/src/alignment/mod.rs` | Module declarations |
| `crates/pocket-tts/src/alignment/aligner.rs` | `WhisperAligner` — manages WhisperX server, sends audio, parses word timestamps |
| `crates/pocket-tts/src/alignment/forced_align.rs` | `WordTimestamp` type definition |
| `crates/pocket-tts/src/tts_model.rs` | `+preload_weights()`, `+load_with_alignment()`, `+generate_sentence_with_timestamps()`, `+generate_with_timestamps()`, `+VOICES`, `+GenerationResult` |
| `crates/pocket-tts/src/lib.rs` | Re-exports: `WhisperAligner`, `WordTimestamp`, `GenerationResult` |
| `scripts/whisperx_server.py` | Persistent WhisperX HTTP server (sidecar) |
| `.github/workflows/build-whisperx.yml` | CI: builds PyInstaller sidecar binaries for macOS/Windows |
| `docs/WORD_TIMESTAMPS_INTEGRATION.md` | Full integration guide for Tauri apps |

## API Surface

```rust
// Types
pub struct GenerationResult { pub audio: Tensor, pub word_timestamps: Vec<WordTimestamp> }
pub struct WordTimestamp { pub word: String, pub start_sec: f32, pub end_sec: f32 }

// Methods on TTSModel
pub const VOICES: &[&str];  // ["alba", "marius", "javert", ...]
pub fn preload_weights(variant: &str, voices: &[&str]) -> Result<()>;
pub fn load_with_alignment(variant: &str) -> Result<Self>;
pub fn generate_sentence_with_timestamps(&self, sentence: &str, voice: &ModelState) -> Result<GenerationResult>;
pub fn generate_with_timestamps(&self, text: &str, voice: &ModelState) -> Result<GenerationResult>;
```

## How It Works

1. **WhisperX server** (`scripts/whisperx_server.py`) runs as a child process, loaded once with Whisper tiny.en + wav2vec2-base models
2. **Per sentence:** TTS generates audio → audio resampled 16kHz → POST to server → WhisperX transcribes + force-aligns → JSON word timestamps returned
3. **Server lifecycle:** starts on `load_with_alignment()`, killed on `drop(model)`

## Approaches Tried & Discarded

| Approach | Why Discarded |
|----------|---------------|
| wav2vec2 CTC forced alignment (native Candle) | CTC probabilities garbage on synthetic speech — words compressed into early frames |
| Candle Whisper with timestamp tokens | ~200ms average offset, proportional splitting within segments was inaccurate |
| Candle Whisper + DTW on cross-attention (teacher forcing) | Cross-attention diffuse during teacher forcing — fundamentally wrong approach |
| Candle Whisper + autoregressive DTW | Better but still ~177ms drift, pauses compressed |
| whisper-rs (whisper.cpp native Rust binding) | cmake TryCompile binaries blocked by Kandji endpoint security on dev machine |
| whisper.cpp CLI sidecar | Good speed (~2s) but slightly less accurate than WhisperX |
| PyInstaller standalone binary | Builds succeed on CI but Kandji blocks execution on dev machine |

**Final approach:** WhisperX Python server as sidecar. Production-grade accuracy (~20-50ms). PyInstaller binary built by CI for customer machines (no Python needed).

## Performance

| Phase | Time |
|-------|------|
| First app launch (download models) | ~60s (530MB, cached forever) |
| Open book (load models into memory) | ~15s |
| Per sentence (generate + align) | ~3-5s |
| Close book | Instant (server killed) |

## CI

GitHub Actions workflow builds WhisperX sidecar binaries on `v*` tags:
- macOS ARM64 ✓
- macOS x64 ✓
- Windows x64 ✓
- Linux x64 — disabled (exceeds GitHub 2GB release limit, needs separate solution)

Release: https://github.com/KnockOutEZ/pocket-tts/releases/tag/v0.7.0-timestamps
