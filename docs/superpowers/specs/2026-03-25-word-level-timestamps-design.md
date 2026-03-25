# Word-Level Timestamps via CTC Forced Alignment

**Date:** 2026-03-25
**Status:** Approved

## Problem

Pocket TTS generates audio but provides no word-level timing information. The product (a Speechify competitor) requires precise word-highlight sync during playback. Target accuracy: ~20-40ms boundary precision, matching or exceeding Speechify.

## Approach

CTC forced alignment using a wav2vec2-base-960h model (~90MB) running natively in Candle. The TTS generates audio, then a separate alignment pass maps known text to audio frame boundaries using Viterbi decoding on CTC log-probabilities.

This is the industry-standard approach for TTS word timestamps. It outperforms attention-based methods (Whisper DTW, model-internal attention extraction) because it uses a principled alignment algorithm on known text rather than heuristic attention weight analysis.

## Constraints

- Tauri desktop only (CPU + Metal). No WASM.
- Batch mode per sentence: generate full audio, align, return together.
- The product buffers ~2s before playback — initial alignment latency is acceptable.
- No changes to existing TTS pipeline modules (sdpa, attention, flow_lm, mimi).

## Public API

```rust
pub struct GenerationResult {
    pub audio: Tensor,                      // [C, T] at 24kHz
    pub word_timestamps: Vec<WordTimestamp>,
}

pub struct WordTimestamp {
    pub word: String,
    pub start_sec: f32,
    pub end_sec: f32,
}
```

New methods on `TTSModel`:

```rust
impl TTSModel {
    /// Load TTS model + alignment model together
    pub fn load_with_alignment(variant: &str) -> Result<Self>;

    /// Generate audio with word timestamps (batch, non-streaming)
    pub fn generate_with_timestamps(
        &self,
        text: &str,
        voice_state: &ModelState,
    ) -> Result<GenerationResult>;
}
```

The alignment model is stored as `Option<Wav2Vec2Aligner>` on `TTSModel`. Loaded via `load_with_alignment()`. Existing `load()` and `generate()` methods remain unchanged.

## Module Structure

```
crates/pocket-tts/src/
├── alignment/
│   ├── mod.rs              # Re-exports
│   ├── wav2vec2.rs         # Wav2Vec2 model (CNN + Transformer + CTC head)
│   ├── forced_align.rs     # Viterbi CTC forced alignment algorithm
│   └── aligner.rs          # Wav2Vec2Aligner orchestrator
├── tts_model.rs            # Extended with aligner field + generate_with_timestamps
└── lib.rs                  # Re-export GenerationResult, WordTimestamp
```

No changes to existing modules.

## wav2vec2 Model Architecture

Model: `facebook/wav2vec2-base-960h` (~90MB safetensors, downloaded via hf-hub).

### CNN Feature Extractor

7 Conv1d layers that downsample raw 16kHz audio to 50 frames/sec:

| Layer | Channels | Kernel | Stride |
|-------|----------|--------|--------|
| 0     | 512      | 10     | 5      |
| 1     | 512      | 3      | 2      |
| 2     | 512      | 3      | 2      |
| 3     | 512      | 3      | 2      |
| 4     | 512      | 3      | 2      |
| 5     | 512      | 2      | 2      |
| 6     | 512      | 2      | 2      |

Total downsampling factor: 320x. Input: `[B, 1, samples]`. Output: `[B, 512, frames]`.

Group normalization after the first layer. GELU activation after all layers.

### Feature Projection

Linear 512 → 768 with layer norm. Projects CNN features to transformer dimension.

### Positional Encoding

Convolutional position embedding: grouped Conv1d (128 groups, kernel 128, padding 64) on the projected features. This is NOT RoPE — it's a learned convolutional positional encoding specific to wav2vec2.

### Transformer Encoder

12 layers, 768-dim, 12 heads. Standard pre-norm transformer (LayerNorm → self-attention → residual → LayerNorm → FFN → residual). FFN dim: 3072. No KV caching needed — this runs in batch mode on complete audio.

### CTC Head

Linear 768 → 32 (26 lowercase letters + space + apostrophe + pipe/blank + padding + special tokens). Outputs log-probabilities per frame via log-softmax.

### Sample Rate

wav2vec2 expects 16kHz input. Pocket TTS generates at 24kHz. Resample via `rubato` (already a dependency) before running alignment.

## CTC Forced Alignment Algorithm

### Step 1: Text to CTC targets

Convert text to lowercase character sequence. Map each character to wav2vec2's vocabulary index. Insert CTC blank tokens between every pair:

```
"Hello world" → [blank, h, blank, e, blank, l, blank, l, blank, o, blank, ' ', blank, w, blank, o, blank, r, blank, l, blank, d, blank]
```

### Step 2: Viterbi forced alignment

Dynamic programming over `[num_frames, num_targets]` trellis.

For each frame `t` and target position `s`:
- `score[t][s] = log_prob[t][target[s]] + max(score[t-1][s], score[t-1][s-1], score[t-1][s-2])`
- The s-2 transition is allowed only when skipping a blank (standard CTC topology)

Backtrack to find the optimal path assigning every frame to a target position.

Complexity: O(T x N) where T = audio frames (~150 for 3s), N = targets (~100 for a sentence). Trivial computation.

### Step 3: Character boundaries to word boundaries

The Viterbi path assigns frames to characters. Group consecutive non-blank characters into words (split at space characters). Word start = first frame of first character. Word end = last frame of last character.

Convert: `timestamp_sec = frame_idx / 50.0` (wav2vec2 frame rate).

### Step 4: Timeline mapping

Timestamps are in seconds, which map directly to the 24kHz TTS output. No additional conversion needed.

## Aligner Orchestrator

```rust
pub struct Wav2Vec2Aligner {
    model: Wav2Vec2Model,
    vocab: HashMap<char, usize>,  // char → token ID
    vocab_inv: Vec<char>,         // token ID → char
    blank_id: usize,
    device: Device,
}

impl Wav2Vec2Aligner {
    pub fn load(device: &Device) -> Result<Self>;

    pub fn align(
        &self,
        audio: &Tensor,   // [C, T] at 24kHz
        text: &str,
    ) -> Result<Vec<WordTimestamp>>;
}
```

`align()` pipeline:
1. Resample 24kHz → 16kHz (rubato)
2. Normalize audio (zero mean, unit variance — wav2vec2 convention)
3. Forward pass → `[frames, vocab]` log-probs
4. Convert text → CTC target sequence
5. Viterbi forced alignment → frame-to-character mapping
6. Character boundaries → word boundaries → `Vec<WordTimestamp>`

## Integration with TTSModel

```rust
pub struct TTSModel {
    // ... existing fields unchanged ...
    pub aligner: Option<Wav2Vec2Aligner>,
}
```

`generate_with_timestamps()` implementation:
1. Split text into sentences via existing `split_into_best_sentences()`
2. For each sentence:
   a. Generate full audio via existing batch generation path
   b. Run `aligner.align(&audio, &sentence_text)`
   c. Offset timestamps by cumulative duration of prior sentences
3. Concatenate audio tensors
4. Return `GenerationResult { audio, word_timestamps }`

## Performance Estimate

For a typical sentence (~3 seconds of audio):
- wav2vec2 forward pass: ~30-50ms on CPU, ~10-20ms on Metal
- Viterbi alignment: <1ms (trivial DP)
- Resampling: ~5ms (rubato)
- Total alignment overhead: ~40-60ms per sentence on CPU

Well within the 2s buffering window.

## Model Download

wav2vec2-base-960h weights downloaded from HuggingFace Hub via `hf-hub` on first use, same pattern as TTS model weights. Cached locally after first download.
