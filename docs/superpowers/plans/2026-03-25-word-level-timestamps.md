# Word-Level Timestamps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CTC forced alignment via wav2vec2-base-960h to pocket-tts, enabling production-grade word-level timestamps for a Speechify competitor.

**Architecture:** A self-contained `alignment/` module implements wav2vec2 in Candle + Viterbi CTC forced alignment. The `Wav2Vec2Aligner` is stored as `Option` on `TTSModel`. New `generate_with_timestamps()` and `generate_sentence_with_timestamps()` methods generate audio then align in a single call, returning `GenerationResult { audio, word_timestamps }`.

**Tech Stack:** Candle (candle-core, candle-nn), rubato (resampling), hf-hub (model download). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-25-word-level-timestamps-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `crates/pocket-tts/src/alignment/mod.rs` | Create | Module declarations, re-exports of public types |
| `crates/pocket-tts/src/alignment/wav2vec2.rs` | Create | Wav2Vec2 model architecture (CNN + Transformer + CTC head) |
| `crates/pocket-tts/src/alignment/forced_align.rs` | Create | Viterbi CTC forced alignment algorithm |
| `crates/pocket-tts/src/alignment/aligner.rs` | Create | `Wav2Vec2Aligner` orchestrator (resample, forward, align, word-map) |
| `crates/pocket-tts/src/tts_model.rs` | Modify | Add `aligner` field, `load_with_alignment()`, `generate_with_timestamps()`, `generate_sentence_with_timestamps()` |
| `crates/pocket-tts/src/lib.rs` | Modify | Add `pub mod alignment` (WASM-gated), re-export `GenerationResult`, `WordTimestamp` |

---

### Task 1: Module Wiring + Stubs

Module declarations must exist BEFORE any other task, otherwise `cargo test` won't compile the new files.

**Files:**
- Create: `crates/pocket-tts/src/alignment/mod.rs`
- Create: `crates/pocket-tts/src/alignment/forced_align.rs` (empty stub)
- Create: `crates/pocket-tts/src/alignment/wav2vec2.rs` (empty stub)
- Create: `crates/pocket-tts/src/alignment/aligner.rs` (empty stub)
- Modify: `crates/pocket-tts/src/lib.rs`

- [ ] **Step 1: Create directory and stub files**

```bash
mkdir -p crates/pocket-tts/src/alignment
```

Create `crates/pocket-tts/src/alignment/mod.rs`:
```rust
#[cfg(not(target_arch = "wasm32"))]
pub mod aligner;
pub mod forced_align;
#[cfg(not(target_arch = "wasm32"))]
pub mod wav2vec2;

#[cfg(not(target_arch = "wasm32"))]
pub use aligner::Wav2Vec2Aligner;
pub use forced_align::WordTimestamp;
```

Note: `wav2vec2` and `aligner` are gated behind `#[cfg(not(target_arch = "wasm32"))]` because they depend on `hf-hub` for model download. `forced_align` is pure logic and compiles everywhere.

Create empty stubs for each submodule (just enough to compile):

`forced_align.rs`:
```rust
/// A word with its start and end time in seconds.
#[derive(Debug, Clone)]
pub struct WordTimestamp {
    pub word: String,
    pub start_sec: f32,
    pub end_sec: f32,
}
```

`wav2vec2.rs`:
```rust
// Wav2Vec2 model - implemented in Task 3
```

`aligner.rs`:
```rust
// Wav2Vec2Aligner - implemented in Task 4
```

- [ ] **Step 2: Add to `lib.rs`**

Add to the module list:
```rust
pub mod alignment;
```

Add re-exports:
```rust
pub use alignment::WordTimestamp;
#[cfg(not(target_arch = "wasm32"))]
pub use alignment::Wav2Vec2Aligner;
```

- [ ] **Step 3: Verify crate compiles**

Run: `cargo check -p pocket-tts`
Expected: compiles with no errors (stubs are valid Rust)

- [ ] **Step 4: Commit**

```bash
git add crates/pocket-tts/src/alignment/ crates/pocket-tts/src/lib.rs
git commit -m "feat: scaffold alignment module with stubs"
```

---

### Task 2: Viterbi CTC Forced Alignment

The alignment algorithm is pure logic with no ML dependencies — test it first in isolation.

**Files:**
- Modify: `crates/pocket-tts/src/alignment/forced_align.rs`

- [ ] **Step 1: Write failing tests for CTC target building**

In `forced_align.rs`, add a `#[cfg(test)] mod tests` block with:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_to_ctc_targets_simple() {
        // "hi" -> [blank, h, blank, i, blank]
        let (targets, word_spans) = text_to_ctc_targets("hi");
        assert_eq!(targets, vec![0, 8, 0, 9, 0]); // blank=0, h=8, i=9
        assert_eq!(word_spans.len(), 1);
        assert_eq!(word_spans[0].word, "hi");
    }

    #[test]
    fn test_text_to_ctc_targets_two_words() {
        let (targets, word_spans) = text_to_ctc_targets("hi there");
        assert_eq!(word_spans.len(), 2);
        assert_eq!(word_spans[0].word, "hi");
        assert_eq!(word_spans[1].word, "there");
        // Contains word boundary token (27) for space
        assert!(targets.contains(&27));
    }

    #[test]
    fn test_text_to_ctc_targets_strips_punctuation() {
        let (targets, word_spans) = text_to_ctc_targets("Hello, World!");
        assert_eq!(word_spans.len(), 2);
        assert_eq!(word_spans[0].word, "Hello,");  // original word preserved
        assert_eq!(word_spans[1].word, "World!");
    }

    #[test]
    fn test_text_to_ctc_targets_preserves_apostrophe() {
        let (targets, word_spans) = text_to_ctc_targets("don't");
        assert_eq!(word_spans.len(), 1);
        assert!(targets.contains(&28)); // apostrophe token
    }

    #[test]
    fn test_text_to_ctc_targets_empty_after_filter() {
        let (targets, word_spans) = text_to_ctc_targets("123 !!!");
        assert!(targets.is_empty());
        assert!(word_spans.is_empty());
    }
}
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `cargo test -p pocket-tts forced_align -- --nocapture 2>&1 | head -20`
Expected: FAIL (function `text_to_ctc_targets` not defined)

- [ ] **Step 3: Implement vocabulary constants and `text_to_ctc_targets`**

Replace the stub content with:

```rust
/// wav2vec2-base-960h vocabulary.
/// ID 0 = <pad> (CTC blank), 1-26 = a-z, 27 = | (word boundary), 28 = ' (apostrophe)
const BLANK_ID: usize = 0;
const WORD_BOUNDARY_ID: usize = 27;
const APOSTROPHE_ID: usize = 28;
pub const VOCAB_SIZE: usize = 32;
pub const WAV2VEC2_FRAME_RATE_HZ: f32 = 50.0;

/// A word with its start and end time in seconds.
#[derive(Debug, Clone)]
pub struct WordTimestamp {
    pub word: String,
    pub start_sec: f32,
    pub end_sec: f32,
}

fn char_to_token_id(c: char) -> Option<usize> {
    match c {
        'a'..='z' => Some((c as usize) - ('a' as usize) + 1),
        '\'' => Some(APOSTROPHE_ID),
        _ => None,
    }
}

/// Span tracking which target positions belong to which original word.
#[derive(Debug, Clone)]
pub struct WordSpan {
    pub word: String,           // original user-facing word
    pub target_start: usize,    // first non-blank target index for this word
    pub target_end: usize,      // last non-blank target index (inclusive)
}

/// Convert text to CTC target sequence with blank insertions.
/// Returns (target_ids, word_spans).
/// Characters outside vocab are dropped. Space maps to | (token 27).
/// WordSpan.word contains the ORIGINAL word (preserving case/punctuation).
pub fn text_to_ctc_targets(text: &str) -> (Vec<usize>, Vec<WordSpan>) {
    let original_words: Vec<&str> = text.split_whitespace().collect();
    if original_words.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Phase 1: Build flat list of (token_id, original_word_index) for non-blank chars
    // Also track which original words produced at least one valid character
    let mut char_tokens: Vec<(usize, usize)> = Vec::new(); // (token_id, word_idx)
    let mut words_with_chars: Vec<bool> = vec![false; original_words.len()];

    for (word_idx, word) in original_words.iter().enumerate() {
        for c in word.to_lowercase().chars() {
            if let Some(id) = char_to_token_id(c) {
                char_tokens.push((id, word_idx));
                words_with_chars[word_idx] = true;
            }
        }
    }

    if char_tokens.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Phase 2: Build target sequence with blanks and word boundaries
    let mut targets = vec![BLANK_ID]; // leading blank
    let mut word_spans: Vec<WordSpan> = Vec::new();
    let mut current_word_idx: Option<usize> = None;
    let mut current_span_start: usize = 0;

    for (token_id, word_idx) in &char_tokens {
        // Insert word boundary when switching to a new word
        if let Some(prev_idx) = current_word_idx {
            if *word_idx != prev_idx {
                // Close previous word span
                let last_char_target = targets.len() - 1; // last pushed char
                word_spans.push(WordSpan {
                    word: original_words[prev_idx].to_string(),
                    target_start: current_span_start,
                    target_end: last_char_target,
                });
                // Insert word boundary: blank, |, blank
                targets.push(WORD_BOUNDARY_ID);
                targets.push(BLANK_ID);
                current_word_idx = Some(*word_idx);
                // Push char
                targets.push(*token_id);
                current_span_start = targets.len() - 1;
                continue;
            }
        } else {
            current_word_idx = Some(*word_idx);
            current_span_start = targets.len(); // next push position
        }

        // Same word — push blank + char
        targets.push(*token_id);
        targets.push(BLANK_ID);
    }

    // Close last word
    if let Some(idx) = current_word_idx {
        // The last target is a trailing blank from the loop; the last char is at len-2
        let last_char_target = targets.len() - 2;
        word_spans.push(WordSpan {
            word: original_words[idx].to_string(),
            target_start: current_span_start,
            target_end: last_char_target,
        });
    }

    (targets, word_spans)
}
```

**Important:** This is a sketch. The word span index tracking is subtle — the tests define the contract. If the indices are off, iterate on the implementation until all tests pass. Consider adding a debug print of `(targets, word_spans)` during development.

- [ ] **Step 4: Run tests — verify they pass**

Run: `cargo test -p pocket-tts forced_align -- --nocapture`
Expected: all 5 tests PASS. If word span indices are wrong, debug and fix.

- [ ] **Step 5: Write failing tests for Viterbi alignment**

```rust
#[test]
fn test_viterbi_align_simple() {
    let num_frames = 10;
    let targets = vec![BLANK_ID, 8, BLANK_ID, 9, BLANK_ID]; // blank, h, blank, i, blank

    let mut log_probs = vec![vec![-10.0f32; VOCAB_SIZE]; num_frames];
    log_probs[0][8] = -0.1;
    log_probs[1][8] = -0.1;
    log_probs[2][0] = -0.1;
    log_probs[3][0] = -0.1;
    for f in 4..9 { log_probs[f][9] = -0.1; }
    log_probs[9][0] = -0.1;

    let path = viterbi_forced_align(&log_probs, &targets).unwrap();
    assert_eq!(path.len(), num_frames);
    assert!(path[0] <= 1); // h or leading blank
    assert!(path[7] >= 3); // i or trailing blank
}

#[test]
fn test_viterbi_repeated_chars() {
    // "ll" = targets [blank, l, blank, l, blank]
    // The blank between the two l's is MANDATORY (repeated char constraint)
    let targets = vec![BLANK_ID, 12, BLANK_ID, 12, BLANK_ID];
    let num_frames = 10;
    let mut log_probs = vec![vec![-10.0f32; VOCAB_SIZE]; num_frames];
    for f in 0..3 { log_probs[f][12] = -0.1; }
    for f in 3..5 { log_probs[f][0] = -0.1; }
    for f in 5..8 { log_probs[f][12] = -0.1; }
    for f in 8..10 { log_probs[f][0] = -0.1; }

    let path = viterbi_forced_align(&log_probs, &targets).unwrap();
    assert!(path.iter().any(|&s| s == 2),
        "Must traverse mandatory blank between repeated chars");
}

#[test]
fn test_viterbi_too_few_frames() {
    let targets = vec![BLANK_ID, 8, BLANK_ID];
    let log_probs = vec![vec![-1.0f32; VOCAB_SIZE]; 1];
    assert!(viterbi_forced_align(&log_probs, &targets).is_err());
}
```

- [ ] **Step 6: Run tests — verify they fail**

Run: `cargo test -p pocket-tts forced_align -- --nocapture 2>&1 | head -20`
Expected: FAIL (function `viterbi_forced_align` not defined)

- [ ] **Step 7: Implement `viterbi_forced_align`**

```rust
/// Run Viterbi forced alignment on CTC log-probabilities.
/// Returns path[t] = target index assigned to frame t.
pub fn viterbi_forced_align(
    log_probs: &[Vec<f32>],
    targets: &[usize],
) -> anyhow::Result<Vec<usize>> {
    let num_frames = log_probs.len();
    let num_targets = targets.len();

    if num_targets == 0 {
        anyhow::bail!("Empty target sequence");
    }
    if num_frames < (num_targets + 1) / 2 {
        anyhow::bail!(
            "Audio too short: {} frames for {} targets (need at least {})",
            num_frames, num_targets, (num_targets + 1) / 2
        );
    }

    let neg_inf = f32::NEG_INFINITY;
    let mut score = vec![vec![neg_inf; num_targets]; num_frames];
    let mut backptr = vec![vec![0usize; num_targets]; num_frames];

    // Initialization
    score[0][0] = log_probs[0][targets[0]];
    if num_targets > 1 {
        score[0][1] = log_probs[0][targets[1]];
    }

    // Fill trellis
    for t in 1..num_frames {
        for s in 0..num_targets {
            let emit = log_probs[t][targets[s]];
            let mut best = score[t - 1][s];
            let mut best_from = s;

            if s >= 1 && score[t - 1][s - 1] > best {
                best = score[t - 1][s - 1];
                best_from = s - 1;
            }

            if s >= 2 {
                // Skip allowed ONLY if: s-1 is blank AND target[s] != target[s-2]
                let skip_ok = targets[s - 1] == BLANK_ID && targets[s] != targets[s - 2];
                if skip_ok && score[t - 1][s - 2] > best {
                    best = score[t - 1][s - 2];
                    best_from = s - 2;
                }
            }

            score[t][s] = best + emit;
            backptr[t][s] = best_from;
        }
    }

    // Termination: best of last two positions
    let mut end_s = num_targets - 1;
    if num_targets >= 2 && score[num_frames - 1][num_targets - 2] > score[num_frames - 1][end_s] {
        end_s = num_targets - 2;
    }

    // Backtrack
    let mut path = vec![0usize; num_frames];
    path[num_frames - 1] = end_s;
    for t in (0..num_frames - 1).rev() {
        path[t] = backptr[t + 1][path[t + 1]];
    }

    Ok(path)
}
```

- [ ] **Step 8: Run tests — verify they pass**

Run: `cargo test -p pocket-tts forced_align -- --nocapture`
Expected: all 8 tests PASS

- [ ] **Step 9: Write failing test for `path_to_word_timestamps`**

```rust
#[test]
fn test_path_to_word_timestamps() {
    // Build targets for "hi there" and a synthetic path
    let (targets, word_spans) = text_to_ctc_targets("hi there");
    // Create a fake path with 20 frames. Exact values depend on target indices
    // from text_to_ctc_targets — print them during debugging if needed.
    let num_frames = 20;
    // Assign frames roughly: first half to "hi", second half to "there"
    let mut path = Vec::new();
    for _ in 0..10 {
        // Use target indices within first word's span
        path.push(word_spans[0].target_start);
    }
    for _ in 10..20 {
        path.push(word_spans[1].target_start);
    }

    let timestamps = path_to_word_timestamps(&path, &targets, &word_spans);
    assert_eq!(timestamps.len(), 2);
    assert_eq!(timestamps[0].word, "hi");
    assert_eq!(timestamps[1].word, "there");
    assert!(timestamps[0].start_sec < timestamps[0].end_sec);
    assert!(timestamps[1].start_sec >= timestamps[0].end_sec);
}
```

- [ ] **Step 10: Implement `path_to_word_timestamps`**

```rust
/// Convert Viterbi path + word spans to word-level timestamps.
pub fn path_to_word_timestamps(
    path: &[usize],
    targets: &[usize],
    word_spans: &[WordSpan],
) -> Vec<WordTimestamp> {
    let mut timestamps = Vec::with_capacity(word_spans.len());

    for span in word_spans {
        let mut first_frame: Option<usize> = None;
        let mut last_frame: Option<usize> = None;

        for (frame_idx, &target_idx) in path.iter().enumerate() {
            if target_idx >= span.target_start
                && target_idx <= span.target_end
                && targets.get(target_idx).copied() != Some(BLANK_ID)
            {
                if first_frame.is_none() {
                    first_frame = Some(frame_idx);
                }
                last_frame = Some(frame_idx);
            }
        }

        if let (Some(first), Some(last)) = (first_frame, last_frame) {
            timestamps.push(WordTimestamp {
                word: span.word.clone(),
                start_sec: first as f32 / WAV2VEC2_FRAME_RATE_HZ,
                end_sec: (last + 1) as f32 / WAV2VEC2_FRAME_RATE_HZ,
            });
        }
    }

    timestamps
}
```

- [ ] **Step 11: Run all forced_align tests — verify they pass**

Run: `cargo test -p pocket-tts forced_align -- --nocapture`
Expected: all 9 tests PASS

- [ ] **Step 12: Commit**

```bash
git add crates/pocket-tts/src/alignment/forced_align.rs
git commit -m "feat: add CTC forced alignment with Viterbi decoding"
```

---

### Task 3: Wav2Vec2 Model Architecture

Implement wav2vec2-base-960h in Candle. CNN feature extractor → feature projection → convolutional position embedding → 12-layer transformer → CTC head.

**Files:**
- Modify: `crates/pocket-tts/src/alignment/wav2vec2.rs`

**Note on GroupNorm:** The feature extractor uses GroupNorm after the first conv layer. Check if `candle_nn` provides `group_norm`. If not, implement a simple GroupNorm (reshape to groups, normalize, reshape back). The existing codebase does NOT use GroupNorm anywhere — this will be new.

- [ ] **Step 1: Implement config, model struct, and `load()`**

```rust
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Linear, Module, VarBuilder};

#[derive(Clone)]
pub struct Wav2Vec2Config {
    pub conv_channels: Vec<usize>,    // [512; 7]
    pub conv_kernels: Vec<usize>,     // [10, 3, 3, 3, 3, 2, 2]
    pub conv_strides: Vec<usize>,     // [5, 2, 2, 2, 2, 2, 2]
    pub hidden_size: usize,           // 768
    pub num_attention_heads: usize,   // 12
    pub num_hidden_layers: usize,     // 12
    pub intermediate_size: usize,     // 3072
    pub vocab_size: usize,            // 32
    pub pos_conv_kernel: usize,       // 128
    pub pos_conv_groups: usize,       // 16
}

impl Wav2Vec2Config {
    pub fn base_960h() -> Self {
        Self {
            conv_channels: vec![512; 7],
            conv_kernels: vec![10, 3, 3, 3, 3, 2, 2],
            conv_strides: vec![5, 2, 2, 2, 2, 2, 2],
            hidden_size: 768,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            intermediate_size: 3072,
            vocab_size: 32,
            pos_conv_kernel: 128,
            pos_conv_groups: 16,
        }
    }
}

#[derive(Clone)]
pub struct Wav2Vec2Model {
    feature_extractor: Wav2Vec2FeatureExtractor,
    feature_projection: Wav2Vec2FeatureProjection,
    encoder: Wav2Vec2Encoder,
    lm_head: Linear,
}

impl Wav2Vec2Model {
    pub fn load(device: &Device) -> anyhow::Result<Self> {
        let config = Wav2Vec2Config::base_960h();

        // Download weights from HuggingFace Hub
        let weights_path = crate::weights::download_if_necessary(
            "hf://facebook/wav2vec2-base-960h/model.safetensors"
        )?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)?
        };

        // Build model components using VarBuilder path prefixes
        // Weight names: wav2vec2.feature_extractor.conv_layers.{i}.{conv,layer_norm}.{weight,bias}
        //               wav2vec2.feature_projection.{projection,layer_norm}.{weight,bias}
        //               wav2vec2.encoder.pos_conv_embed.conv.{weight,bias}
        //               wav2vec2.encoder.layers.{i}.attention.{k,q,v,out}_proj.{weight,bias}
        //               wav2vec2.encoder.layers.{i}.feed_forward.{intermediate_dense,output_dense}.{weight,bias}
        //               wav2vec2.encoder.layers.{i}.layer_norm.{weight,bias}
        //               wav2vec2.encoder.layers.{i}.final_layer_norm.{weight,bias}
        //               wav2vec2.encoder.layer_norm.{weight,bias}
        //               lm_head.{weight,bias}

        let feature_extractor = Wav2Vec2FeatureExtractor::new(
            &config, vb.pp("wav2vec2.feature_extractor")
        )?;
        let feature_projection = Wav2Vec2FeatureProjection::new(
            &config, vb.pp("wav2vec2.feature_projection")
        )?;
        let encoder = Wav2Vec2Encoder::new(
            &config, vb.pp("wav2vec2.encoder")
        )?;
        let lm_head = candle_nn::linear(
            config.hidden_size, config.vocab_size, vb.pp("lm_head")
        )?;

        Ok(Self { feature_extractor, feature_projection, encoder, lm_head })
    }

    pub fn forward(&self, audio: &Tensor) -> Result<Tensor> {
        // audio: [B, 1, samples] at 16kHz
        let features = self.feature_extractor.forward(audio)?;   // [B, 512, frames]
        let features = features.transpose(1, 2)?;                // [B, frames, 512]
        let hidden = self.feature_projection.forward(&features)?; // [B, frames, 768]
        let encoded = self.encoder.forward(&hidden)?;              // [B, frames, 768]
        let logits = self.lm_head.forward(&encoded)?;              // [B, frames, 32]
        candle_nn::ops::log_softmax(&logits, 2)                   // [B, frames, 32]
    }
}
```

Then implement the sub-structs:

- `Wav2Vec2FeatureExtractor`: 7 Conv1d layers. First layer: Conv1d(1→512, k=10, s=5) + GroupNorm(512 groups, 512 channels) + GELU. Layers 1-6: Conv1d(512→512, k/s per table) + GELU (no norm).
- `Wav2Vec2FeatureProjection`: LayerNorm(512) + Linear(512→768). VarBuilder prefix: `projection` for linear, `layer_norm` for norm.
- `Wav2Vec2Encoder`: Convolutional position embedding (Conv1d with 16 groups, kernel 128, padding 64) + 12 transformer layers + final LayerNorm. Each layer: LayerNorm → SelfAttention → residual → LayerNorm → FFN → residual.

**Weight name debugging:** If the model fails to load, dump all tensor names:
```rust
let tensors = candle_core::safetensors::load(&weights_path, &Device::Cpu)?;
for name in tensors.keys().sorted() { eprintln!("{}: {:?}", name, tensors[name].shape()); }
```

- [ ] **Step 2: Implement transformer encoder layers**

Each `Wav2Vec2EncoderLayer` has:
- `attention`: 4 linear projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`) each 768→768
- `layer_norm`: LayerNorm(768) before attention
- `feed_forward.intermediate_dense`: Linear(768→3072)
- `feed_forward.output_dense`: Linear(3072→768)
- `final_layer_norm`: LayerNorm(768) after FFN

The attention is standard multi-head:
```rust
fn self_attention(q: &Tensor, k: &Tensor, v: &Tensor, num_heads: usize) -> Result<Tensor> {
    let (b, t, d) = q.dims3()?;
    let head_dim = d / num_heads;
    let scale = (head_dim as f64).sqrt();
    let q = q.reshape((b, t, num_heads, head_dim))?.transpose(1, 2)?;
    let k = k.reshape((b, t, num_heads, head_dim))?.transpose(1, 2)?;
    let v = v.reshape((b, t, num_heads, head_dim))?.transpose(1, 2)?;
    let scores = (q.matmul(&k.transpose(2, 3)?)? / scale)?;
    let probs = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;
    let out = probs.matmul(&v)?;
    out.transpose(1, 2)?.reshape((b, t, d))
}
```

No causal mask — wav2vec2 is bidirectional. No KV cache — batch mode.

- [ ] **Step 3: Write ignored load+forward test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // requires ~90MB model download
    fn test_wav2vec2_loads_and_runs() {
        let device = Device::Cpu;
        let model = Wav2Vec2Model::load(&device).unwrap();
        let audio = Tensor::zeros((1, 1, 16000), DType::F32, &device).unwrap();
        let log_probs = model.forward(&audio).unwrap();
        assert_eq!(log_probs.dims()[0], 1);   // batch
        assert_eq!(log_probs.dims()[2], 32);  // vocab
        // Exact frame count depends on conv padding; expect ~49-50 for 16000 samples
        assert!(log_probs.dims()[1] >= 40 && log_probs.dims()[1] <= 55);
    }
}
```

Run: `cargo test -p pocket-tts test_wav2vec2_loads_and_runs -- --ignored --nocapture`

This validates architecture + weight loading in one shot. If weight names are wrong, it fails here.

- [ ] **Step 4: Commit**

```bash
git add crates/pocket-tts/src/alignment/wav2vec2.rs
git commit -m "feat: add wav2vec2-base-960h model in Candle"
```

---

### Task 4: Aligner Orchestrator

Ties wav2vec2 + forced alignment together. Handles resampling, normalization, and the full pipeline.

**Files:**
- Modify: `crates/pocket-tts/src/alignment/aligner.rs`

- [ ] **Step 1: Write failing integration test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    #[ignore] // requires model download
    fn test_aligner_on_silence() {
        let device = Device::Cpu;
        let aligner = Wav2Vec2Aligner::load(&device).unwrap();

        // 1 second of silence at 24kHz
        let audio = Tensor::zeros((1, 24000), DType::F32, &device).unwrap();
        let timestamps = aligner.align(&audio, "hello world").unwrap();

        assert_eq!(timestamps.len(), 2);
        assert_eq!(timestamps[0].word, "hello");
        assert_eq!(timestamps[1].word, "world");
    }
}
```

- [ ] **Step 2: Implement `Wav2Vec2Aligner`**

```rust
use crate::alignment::forced_align::{
    WordTimestamp, text_to_ctc_targets, viterbi_forced_align, path_to_word_timestamps,
};
use crate::alignment::wav2vec2::Wav2Vec2Model;
use crate::audio::resample;
use candle_core::{DType, Device, Tensor};

const TTS_SAMPLE_RATE: u32 = 24000;
const WAV2VEC2_SAMPLE_RATE: u32 = 16000;

#[derive(Clone)]
pub struct Wav2Vec2Aligner {
    model: Wav2Vec2Model,
    device: Device,
}

impl Wav2Vec2Aligner {
    pub fn load(device: &Device) -> anyhow::Result<Self> {
        let model = Wav2Vec2Model::load(device)?;
        Ok(Self { model, device: device.clone() })
    }

    pub fn align(&self, audio: &Tensor, text: &str) -> anyhow::Result<Vec<WordTimestamp>> {
        // 1. Build CTC targets
        let (targets, word_spans) = text_to_ctc_targets(text);
        if targets.is_empty() {
            return Ok(Vec::new());
        }

        // 2. Prepare audio: ensure [1, T], resample, normalize
        let audio = match audio.dims().len() {
            1 => audio.unsqueeze(0)?,                     // [T] -> [1, T]
            2 if audio.dims()[0] == 1 => audio.clone(),   // [1, T] already
            2 => audio.mean(0)?.unsqueeze(0)?,            // [C, T] -> mono -> [1, T]
            _ => anyhow::bail!("Unexpected audio shape: {:?}", audio.dims()),
        };

        let audio_16k = resample(&audio, TTS_SAMPLE_RATE, WAV2VEC2_SAMPLE_RATE)?;

        // Normalize: zero mean, unit variance (wav2vec2 convention)
        let mean = audio_16k.mean_all()?;
        let centered = audio_16k.broadcast_sub(&mean)?;
        let var = (&centered * &centered)?.mean_all()?;
        let std = (var + 1e-7)?.sqrt()?;
        let normalized = centered.broadcast_div(&std)?;

        // Move to model device if needed
        let normalized = if !normalized.device().same_device(&self.device) {
            normalized.to_device(&self.device)?
        } else {
            normalized
        };

        // 3. Forward pass: [1, 1, samples] -> [1, frames, 32]
        let input = normalized.unsqueeze(0)?;
        let log_probs = self.model.forward(&input)?;

        // 4. Extract as Vec<Vec<f32>> for Viterbi
        let log_probs_2d = log_probs.squeeze(0)?;
        let flat = log_probs_2d.to_vec2::<f32>()?;

        // 5. Viterbi forced alignment
        let path = viterbi_forced_align(&flat, &targets)?;

        // 6. Convert to word timestamps
        Ok(path_to_word_timestamps(&path, &targets, &word_spans))
    }
}
```

- [ ] **Step 3: Run integration test**

Run: `cargo test -p pocket-tts test_aligner_on_silence -- --ignored --nocapture`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/pocket-tts/src/alignment/aligner.rs
git commit -m "feat: add Wav2Vec2Aligner orchestrator"
```

---

### Task 5: TTSModel Integration

Add `aligner` field, `generate_with_timestamps()`, and `generate_sentence_with_timestamps()`.

**Files:**
- Modify: `crates/pocket-tts/src/tts_model.rs`
- Modify: `crates/pocket-tts/src/lib.rs`

- [ ] **Step 1: Add `aligner` field to `TTSModel` struct**

At `tts_model.rs:22`, add to the struct:
```rust
    /// Optional alignment model for word timestamps
    #[cfg(not(target_arch = "wasm32"))]
    pub aligner: Option<crate::alignment::Wav2Vec2Aligner>,
```

Add `aligner: None` to the `Self { ... }` return in `from_config_and_vb` (around line 411) and `load_from_bytes` (around line 276). Example for `from_config_and_vb`:

```rust
Ok(Self {
    flow_lm,
    mimi,
    conditioner,
    speaker_proj_weight,
    temp,
    lsd_decode_steps,
    eos_threshold,
    noise_clamp,
    voice_prompt_chunk_frames: None,
    sample_rate: config.mimi.sample_rate,
    dim,
    ldim,
    device,
    #[cfg(not(target_arch = "wasm32"))]
    aligner: None,
})
```

Do the same for `load_from_bytes`.

- [ ] **Step 2: Add `GenerationResult` struct**

In `tts_model.rs`:
```rust
/// Result of generation with word-level timestamps.
pub struct GenerationResult {
    pub audio: Tensor,
    pub word_timestamps: Vec<crate::alignment::WordTimestamp>,
}
```

In `lib.rs`, add:
```rust
pub use tts_model::GenerationResult;
```

- [ ] **Step 3: Implement `load_with_alignment`**

```rust
#[cfg(not(target_arch = "wasm32"))]
pub fn load_with_alignment(variant: &str) -> Result<Self> {
    let mut model = Self::load(variant)?;
    let aligner = crate::alignment::Wav2Vec2Aligner::load(&model.device)?;
    model.aligner = Some(aligner);
    Ok(model)
}
```

- [ ] **Step 4: Implement `generate_sentence_with_timestamps`**

**Critical:** `self.generate()` internally calls `split_into_best_sentences()` which re-splits text. For single-sentence generation, use the lower-level path that generates without re-splitting.

```rust
#[cfg(not(target_arch = "wasm32"))]
pub fn generate_sentence_with_timestamps(
    &self,
    sentence: &str,
    voice_state: &ModelState,
) -> Result<GenerationResult> {
    let aligner = self.aligner.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Alignment model not loaded. Use load_with_alignment()"))?;

    // Generate audio — generate() is fine here since for a single sentence
    // the internal split_into_best_sentences will return it as one chunk
    // (sentences are already <=50 tokens from the caller's splitting)
    let audio = self.generate(sentence, voice_state)?;

    // Align using the ORIGINAL sentence text (not prepare_text_prompt output)
    // so that WordTimestamp.word contains the original user-facing words
    let word_timestamps = aligner.align(&audio, sentence)?;

    Ok(GenerationResult { audio, word_timestamps })
}
```

- [ ] **Step 5: Implement `generate_with_timestamps`**

```rust
#[cfg(not(target_arch = "wasm32"))]
pub fn generate_with_timestamps(
    &self,
    text: &str,
    voice_state: &ModelState,
) -> Result<GenerationResult> {
    let aligner = self.aligner.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Alignment model not loaded. Use load_with_alignment()"))?;

    // Split text into sentences — same logic as generate_stream
    let chunks = self.split_into_best_sentences(text);

    if chunks.is_empty() {
        anyhow::bail!("No text to generate");
    }

    let mut all_audio = Vec::new();
    let mut all_timestamps = Vec::new();
    let mut cumulative_offset_sec: f32 = 0.0;

    // We need to recover original words for each chunk.
    // split_into_best_sentences runs prepare_text_prompt internally (lowercases, etc.)
    // For alignment, we pass the CHUNK text (which is already prepared) to generate,
    // but we pass the chunk to the aligner which will lowercase internally.
    // The aligner's text_to_ctc_targets preserves original words from its input.
    for chunk_text in &chunks {
        let audio = self.generate(chunk_text, voice_state)?;

        let mut timestamps = aligner.align(&audio, chunk_text)?;
        for ts in &mut timestamps {
            ts.start_sec += cumulative_offset_sec;
            ts.end_sec += cumulative_offset_sec;
        }

        // Duration from actual sample count (not alignment frames)
        let num_samples = audio.dims().last().copied().unwrap_or(0);
        cumulative_offset_sec += num_samples as f32 / self.sample_rate as f32;

        all_audio.push(audio);
        all_timestamps.extend(timestamps);
    }

    let audio = if all_audio.len() == 1 {
        all_audio.into_iter().next().unwrap()
    } else {
        Tensor::cat(&all_audio, 1)?
    };

    Ok(GenerationResult {
        audio,
        word_timestamps: all_timestamps,
    })
}
```

- [ ] **Step 6: Verify crate compiles**

Run: `cargo check -p pocket-tts`
Expected: compiles with no errors

- [ ] **Step 7: Write ignored integration test**

```rust
#[test]
#[ignore] // requires both TTS and wav2vec2 model downloads
fn test_generate_with_timestamps() {
    let model = TTSModel::load_with_alignment("b6369a24").unwrap();
    // Use default voice or skip voice state if test_data unavailable
    let voice_state = crate::voice_state::init_states(1, 1000);

    let result = model.generate_with_timestamps("Hello world.", &voice_state).unwrap();

    assert!(!result.word_timestamps.is_empty());
    assert!(result.word_timestamps.len() >= 2);
    // Monotonically increasing
    for window in result.word_timestamps.windows(2) {
        assert!(window[1].start_sec >= window[0].start_sec);
    }
    assert!(result.audio.dims().last().unwrap() > &0);
}
```

- [ ] **Step 8: Commit**

```bash
git add crates/pocket-tts/src/tts_model.rs crates/pocket-tts/src/lib.rs
git commit -m "feat: add generate_with_timestamps to TTSModel"
```

---

### Task 6: End-to-End Validation

Run the full pipeline and verify timestamps are sane on real speech.

**Files:** No new files — validation only.

- [ ] **Step 1: Run ignored integration tests**

```bash
cargo test -p pocket-tts -- --ignored --nocapture 2>&1
```

All ignored tests should pass: model loading, alignment on silence, full generation with timestamps.

- [ ] **Step 2: Manual validation**

Write a quick test or example:

```rust
let model = TTSModel::load_with_alignment("b6369a24")?;
let voice_state = model.get_voice_state("path/to/voice.wav")?;
let result = model.generate_with_timestamps(
    "The quick brown fox jumps over the lazy dog.",
    &voice_state
)?;

for ts in &result.word_timestamps {
    println!("{:.3}s - {:.3}s: {}", ts.start_sec, ts.end_sec, ts.word);
}
```

Verify: timestamps monotonically increasing, ~50-200ms per word, covering full audio duration.

- [ ] **Step 3: Commit any test additions**

```bash
git add -A && git commit -m "test: add end-to-end timestamp validation"
```

---

## Implementation Notes

**Weight name debugging:** The most likely failure point is wav2vec2 weight names not matching VarBuilder paths. If `Wav2Vec2Model::load()` fails, list all tensor names from the safetensors file to find the correct mapping. Use: `VarBuilder::pp("wav2vec2").pp("feature_extractor").pp("conv_layers").pp("0").pp("conv")` to match `wav2vec2.feature_extractor.conv_layers.0.conv.weight`.

**Conv1d in wav2vec2:** Use plain `candle_nn::conv1d` — NOT the project's `StreamingConv1d`. wav2vec2 runs in batch mode.

**Attention in wav2vec2:** Use standard `Q @ K.T` matmul — NOT the project's `StreamingMultiheadAttention`. No KV cache, no causal mask. Bidirectional.

**GroupNorm:** Check `candle_nn` for `group_norm`. If unavailable, implement as: reshape `[B, C, T]` to `[B, G, C/G, T]`, normalize over dims 2+3, reshape back, apply affine.

**`generate()` returns `[C, T]`** after squeeze. The aligner handles this shape.

**`text_to_ctc_targets` word span tracking** is the trickiest pure-logic part. The sketch is approximate — iterate on it until the tests pass. Print intermediate state during debugging.

**WASM gating:** `wav2vec2.rs` and `aligner.rs` are gated behind `#[cfg(not(target_arch = "wasm32"))]` in `alignment/mod.rs`. The `aligner` field on `TTSModel` and all timestamp methods are similarly gated. `forced_align.rs` compiles on all targets (pure logic).
