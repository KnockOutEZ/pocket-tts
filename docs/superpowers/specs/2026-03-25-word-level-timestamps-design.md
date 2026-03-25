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
- `Wav2Vec2Aligner` must implement/derive `Clone` since `TTSModel` derives `Clone` and actively clones itself (e.g., `generate_stream_owned`). Candle tensors are reference-counted so this is cheap.

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
    /// Splits text into sentences internally, generates + aligns each.
    pub fn generate_with_timestamps(
        &self,
        text: &str,
        voice_state: &ModelState,
    ) -> Result<GenerationResult>;

    /// Generate + align a single pre-split sentence.
    /// No internal text splitting — caller controls chunking.
    /// Use this when the product already splits sentences itself.
    pub fn generate_sentence_with_timestamps(
        &self,
        sentence: &str,
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

Model: `facebook/wav2vec2-base-960h` from HuggingFace Hub. File: `model.safetensors` (~90MB). Downloaded via `hf-hub` on first use, cached locally. Architecture and vocabulary are hardcoded (fixed well-known model, no config file needed).

### CNN Feature Extractor

7 Conv1d layers that downsample raw 16kHz audio to 50 frames/sec (constant: `WAV2VEC2_FRAME_RATE_HZ = 50`):

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

Convolutional position embedding: grouped Conv1d (16 groups, kernel 128, padding 64) on the projected features. This is NOT RoPE — it's a learned convolutional positional encoding specific to wav2vec2.

### Transformer Encoder

12 layers, 768-dim, 12 heads. Standard pre-norm transformer (LayerNorm → self-attention → residual → LayerNorm → FFN → residual). FFN dim: 3072. No KV caching needed — this runs in batch mode on complete audio.

### CTC Head

Linear 768 → 32. Outputs log-probabilities per frame via log-softmax.

### Vocabulary

wav2vec2-base-960h has exactly 32 tokens:

| ID | Token | Meaning |
|----|-------|---------|
| 0  | `<pad>` | CTC blank token |
| 1-26 | `a`-`z` | Lowercase letters |
| 27 | `\|` | Word boundary (represents space between words) |
| 28 | `'` | Apostrophe |
| 29 | `<unk>` | Unknown |
| 30 | `<s>` | Start of sequence |
| 31 | `</s>` | End of sequence |

The CTC blank token is ID 0 (`<pad>`). The space character in text maps to `|` (ID 27), NOT to a literal space token. This vocabulary is hardcoded in the aligner.

### Sample Rate

wav2vec2 expects 16kHz mono input. Pocket TTS generates at 24kHz mono. Resample via `rubato` (already a dependency) before alignment. If multi-channel audio is ever produced, mix to mono first.

## CTC Forced Alignment Algorithm

### Step 1: Text to CTC targets

The aligner receives the **TTS-prepared text** (output of `prepare_text_prompt()`, lowercase). Processing:

1. Lowercase the text
2. Strip all characters not in the wav2vec2 vocabulary (drop digits, punctuation except apostrophe)
3. Map space → `|` (token 27), letters → their token IDs, apostrophe → token 28
4. Insert CTC blank (token 0) between every pair, and at start/end

Example:
```
"hello world" → [blank, h, blank, e, blank, l, blank, l, blank, o, blank, |, blank, w, blank, o, blank, r, blank, l, blank, d, blank]
```

Characters outside the vocabulary (digits, punctuation like `.`, `!`, `,`) are dropped before building the target sequence. Their corresponding words are still tracked for the word mapping (see Step 3).

**Edge cases:**
- Empty text after filtering → return empty `Vec<WordTimestamp>` (not an error)
- Single word → valid, produces a single `WordTimestamp`
- All-punctuation/digit input → empty after filtering → return empty timestamps

### Step 2: Viterbi forced alignment

Dynamic programming over `[num_frames, num_targets]` trellis.

**Initialization:**
- `score[0][0] = log_prob[0][target[0]]` (start at first blank)
- `score[0][1] = log_prob[0][target[1]]` (or start at first character)
- `score[0][s] = -infinity` for all s >= 2

**Recurrence** for each frame `t > 0` and target position `s`:
- `score[t][s] = log_prob[t][target[s]] + max(score[t-1][s], score[t-1][s-1], maybe score[t-1][s-2])`

**The s-2 skip transition** is allowed ONLY when BOTH conditions are met:
1. `target[s-1]` is a blank token (we're skipping over a blank)
2. `target[s] != target[s-2]` (the characters on either side of the blank are different)

This second condition is critical: for repeated adjacent characters (e.g., the two `l`s in "hello"), the blank between them is mandatory to distinguish the repetitions. Without this constraint, "ll" could collapse to a single "l".

**Termination:**
The final frame must end at position `N-1` (last blank) or `N-2` (last character). Take the max of these two and backtrack.

**Backtracking:**
Store `backpointer[t][s]` alongside scores. Trace back from the terminal position to produce the frame-to-target assignment.

Complexity: O(T x N) where T = audio frames (~150 for 3s), N = targets (~100 for a sentence). Trivial computation.

### Step 3: Character boundaries to word boundaries

The Viterbi path assigns frames to target positions. To produce word timestamps:

1. Walk the path, collecting frame ranges for each non-blank character
2. Group characters into words by splitting at `|` (word boundary) tokens
3. Word start = first frame of its first character. Word end = last frame of its last character.
4. Convert: `timestamp_sec = frame_idx as f32 / WAV2VEC2_FRAME_RATE_HZ as f32`

**Word text mapping:** The `WordTimestamp.word` field should contain the **original user-facing word** from the input text, not the lowercased/stripped alignment version. The aligner maintains a parallel mapping from alignment-text word positions back to original-text words. This is done by splitting both the original and filtered text by whitespace and correlating by position (skipping words that were entirely filtered out).

### Step 4: Timeline mapping

Timestamps are in seconds, which map directly to the 24kHz TTS output. No additional conversion needed.

## Aligner Orchestrator

```rust
#[derive(Clone)]
pub struct Wav2Vec2Aligner {
    model: Wav2Vec2Model,
    device: Device,
}

impl Wav2Vec2Aligner {
    /// Load wav2vec2-base-960h from HuggingFace Hub.
    /// Downloads on first use, cached locally via hf-hub.
    pub fn load(device: &Device) -> Result<Self>;

    /// Align known text to audio, returning word timestamps.
    /// Audio: [C, T] at 24kHz. Text: the original user-facing text.
    pub fn align(
        &self,
        audio: &Tensor,
        text: &str,
    ) -> Result<Vec<WordTimestamp>>;
}
```

Vocabulary mapping is a module-level constant (hardcoded `[char; 32]` array), not stored on the struct.

`align()` pipeline:
1. Mix to mono if needed, resample 24kHz → 16kHz (rubato)
2. Normalize audio (zero mean, unit variance — wav2vec2 convention)
3. Forward pass → `[frames, vocab]` log-probs
4. Lowercase text, strip non-vocab chars, build CTC target sequence
5. Viterbi forced alignment → frame-to-character mapping
6. Character boundaries → word boundaries → `Vec<WordTimestamp>`

**Error conditions** (all surfaced via `Result`):
- Alignment model not loaded → error from `generate_with_timestamps` (checks `Option`)
- Audio too short for target sequence (fewer frames than required target positions) → error
- Empty audio tensor → error
- Device mismatch (audio on different device than model) → move audio to model device before forward pass
- Model download failure → propagated from `hf-hub`

## Integration with TTSModel

```rust
pub struct TTSModel {
    // ... existing fields unchanged ...
    pub aligner: Option<Wav2Vec2Aligner>,
}
```

`generate_with_timestamps()` implementation:
1. Verify `self.aligner.is_some()`, error if not
2. Split text into sentences via existing `split_into_best_sentences()`
3. Track `cumulative_offset_sec: f32 = 0.0`
4. For each sentence:
   a. Generate full audio via existing batch generation path (reuse `generate()` internals)
   b. Run `aligner.align(&audio, &original_sentence_text)`
   c. Offset each word's `start_sec` and `end_sec` by `cumulative_offset_sec`
   d. `cumulative_offset_sec += sentence_audio.dim(1) as f32 / 24000.0` (from actual sample count)
5. Concatenate audio tensors
6. Return `GenerationResult { audio, word_timestamps }`

The cumulative offset is derived from the actual audio tensor sample count (`dim(1) / sample_rate`), not from alignment frame boundaries, to avoid rounding drift.

**Pause marker interaction:** The initial implementation passes text through `prepare_text_prompt()` which strips `[pause:Xms]` markers. Silence samples inserted by pause handling contribute to `cumulative_offset_sec` via the audio sample count. Words before a pause end at their acoustic boundary; words after start after the silence. This falls out naturally from the per-sentence alignment + offset accumulation.

## Performance Estimate

For a typical sentence (~3 seconds of audio):
- wav2vec2 forward pass: ~30-50ms on CPU, ~10-20ms on Metal
- Viterbi alignment: <1ms (trivial DP)
- Resampling: ~5ms (rubato)
- Total alignment overhead: ~40-60ms per sentence on CPU

Well within the 2s buffering window.

## Model Download

Repo: `facebook/wav2vec2-base-960h` on HuggingFace Hub. File: `model.safetensors`. Downloaded via `hf-hub` on first call to `Wav2Vec2Aligner::load()`, cached in the standard hf-hub cache directory. Same pattern as existing TTS model weight downloads in `weights.rs`.
