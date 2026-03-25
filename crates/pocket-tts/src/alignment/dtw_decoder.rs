//! Custom Whisper decoder that captures cross-attention weights for DTW alignment.
//!
//! After the first decoding pass (using candle-transformers' built-in Whisper model),
//! this module runs a second teacher-forced pass through a lightweight decoder replica
//! that exposes cross-attention softmax weights. Those weights are then aligned via
//! Dynamic Time Warping (DTW) to produce per-token audio-frame mappings, which are
//! finally grouped into word-level timestamps.

use crate::alignment::forced_align::WordTimestamp;
use candle_core::{Device, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder};

// Whisper mel-spectrogram constants (re-exported from candle_transformers::models::whisper).
const HOP_LENGTH: usize = 160;
const SAMPLE_RATE: usize = 16000;

// ---------------------------------------------------------------------------
// Multi-Head Attention (captures softmax weights for cross-attention)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: usize,
}

impl MultiHeadAttention {
    fn load(n_state: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let query = candle_nn::linear(n_state, n_state, vb.pp("q_proj"))?;
        let key = candle_nn::linear_no_bias(n_state, n_state, vb.pp("k_proj"))?;
        let value = candle_nn::linear(n_state, n_state, vb.pp("v_proj"))?;
        let out = candle_nn::linear(n_state, n_state, vb.pp("out_proj"))?;
        Ok(Self {
            query,
            key,
            value,
            out,
            n_head,
        })
    }

    /// Reshape `[B, T, D]` to `[B, n_head, T, head_dim]`.
    fn reshape_head(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let head_dim = d / self.n_head;
        x.reshape((b, t, self.n_head, head_dim))?.transpose(1, 2)
    }

    /// Standard forward — returns the output projection only.
    fn forward(
        &self,
        x: &Tensor,
        xa: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (out, _) = self.forward_with_weights(x, xa, mask)?;
        Ok(out)
    }

    /// Forward that also returns the softmax attention weights `[B, n_head, T_q, T_kv]`.
    fn forward_with_weights(
        &self,
        x: &Tensor,
        xa: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let q = self.query.forward(x)?;
        let kv_input = xa.unwrap_or(x);
        let k = self.key.forward(kv_input)?;
        let v = self.value.forward(kv_input)?;

        let (_, n_ctx, n_state) = q.dims3()?;
        let scale = ((n_state / self.n_head) as f64).powf(-0.25);

        let q = (self.reshape_head(&q)? * scale)?;
        let k = (self.reshape_head(&k)?.transpose(2, 3)? * scale)?;
        let v = self.reshape_head(&v)?.contiguous()?;

        let mut qk = q.matmul(&k)?;
        if let Some(mask) = mask {
            let mask = mask.i((0..n_ctx, 0..n_ctx))?;
            qk = qk.broadcast_add(&mask)?;
        }

        let w = candle_nn::ops::softmax_last_dim(&qk)?;

        let wv = w.matmul(&v)?.transpose(1, 2)?.flatten_from(2)?;
        let out = self.out.forward(&wv)?;
        Ok((out, w))
    }
}

// We need IndexOp for the mask slicing `mask.i(...)`.
use candle_core::IndexOp;

// ---------------------------------------------------------------------------
// Residual Attention Block (decoder layer)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct ResidualAttentionBlock {
    self_attn: MultiHeadAttention,
    self_attn_ln: LayerNorm,
    cross_attn: MultiHeadAttention,
    cross_attn_ln: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_ln: LayerNorm,
}

impl ResidualAttentionBlock {
    fn load(n_state: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let self_attn = MultiHeadAttention::load(n_state, n_head, vb.pp("self_attn"))?;
        let self_attn_ln = candle_nn::layer_norm(n_state, 1e-5, vb.pp("self_attn_layer_norm"))?;
        let cross_attn = MultiHeadAttention::load(n_state, n_head, vb.pp("encoder_attn"))?;
        let cross_attn_ln =
            candle_nn::layer_norm(n_state, 1e-5, vb.pp("encoder_attn_layer_norm"))?;
        let fc1 = candle_nn::linear(n_state, n_state * 4, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(n_state * 4, n_state, vb.pp("fc2"))?;
        let final_ln = candle_nn::layer_norm(n_state, 1e-5, vb.pp("final_layer_norm"))?;
        Ok(Self {
            self_attn,
            self_attn_ln,
            cross_attn,
            cross_attn_ln,
            fc1,
            fc2,
            final_ln,
        })
    }

    /// Forward through one decoder layer, returning `(hidden, cross_attn_weights)`.
    fn forward(
        &self,
        x: &Tensor,
        xa: &Tensor,
        mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // 1. Self-attention (causal)
        let sa_out = self.self_attn.forward(
            &self.self_attn_ln.forward(x)?,
            None,
            Some(mask),
        )?;
        let x = (x + sa_out)?;

        // 2. Cross-attention (captures weights)
        let (ca_out, ca_weights) = self.cross_attn.forward_with_weights(
            &self.cross_attn_ln.forward(&x)?,
            Some(xa),
            None,
        )?;
        let x = (&x + ca_out)?;

        // 3. FFN
        let ffn_out = self.fc2.forward(&self.fc1.forward(&self.final_ln.forward(&x)?)?.gelu()?)?;
        let x = (x + ffn_out)?;

        Ok((x, ca_weights))
    }
}

// ---------------------------------------------------------------------------
// CrossAttentionDecoder
// ---------------------------------------------------------------------------

/// Lightweight Whisper decoder replica that captures cross-attention weights.
///
/// Loads the same safetensors weights as candle-transformers' built-in decoder
/// but exposes the cross-attention softmax output from every layer.
#[derive(Clone, Debug)]
pub struct CrossAttentionDecoder {
    token_embedding: Embedding,
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln: LayerNorm,
    mask: Tensor,
}

impl CrossAttentionDecoder {
    /// Load decoder weights from safetensors via `VarBuilder`.
    ///
    /// The `vb` must be rooted at `model.decoder` (i.e. call `vb.pp("model.decoder")`
    /// before passing it here).
    pub fn load(vb: VarBuilder, config: &super::DecoderConfig) -> Result<Self> {
        let n_state = config.d_model;
        let n_head = config.decoder_attention_heads;
        let n_ctx = config.max_target_positions;

        let token_embedding =
            candle_nn::embedding(config.vocab_size, n_state, vb.pp("embed_tokens"))?;
        let positional_embedding = vb.get((n_ctx, n_state), "embed_positions.weight")?;

        let blocks = (0..config.decoder_layers)
            .map(|i| ResidualAttentionBlock::load(n_state, n_head, vb.pp(format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;

        let ln = candle_nn::layer_norm(n_state, 1e-5, vb.pp("layer_norm"))?;

        // Causal mask: lower-triangular with -inf above diagonal
        let mask_data: Vec<f32> = (0..n_ctx)
            .flat_map(|i| {
                (0..n_ctx).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 })
            })
            .collect();
        let mask = Tensor::from_vec(mask_data, (n_ctx, n_ctx), vb.device())?;

        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            mask,
        })
    }

    /// Run a teacher-forced forward pass and return cross-attention weights from all layers.
    ///
    /// - `tokens`: `[B, T]` — token IDs from the first decoding pass
    /// - `encoder_output`: `[B, S, D]` — audio features from the encoder
    ///
    /// Returns `Vec<Tensor>` where each element is `[B, n_heads, T, S]` (one per layer).
    pub fn forward(&self, tokens: &Tensor, encoder_output: &Tensor) -> Result<Vec<Tensor>> {
        let seq_len = tokens.dim(candle_core::D::Minus1)?;
        let token_emb = self.token_embedding.forward(tokens)?;
        let pos_emb = self.positional_embedding.narrow(0, 0, seq_len)?;
        let mut x = token_emb.broadcast_add(&pos_emb)?;

        let mut all_cross_attn_weights = Vec::with_capacity(self.blocks.len());

        for block in &self.blocks {
            let (hidden, ca_weights) = block.forward(&x, encoder_output, &self.mask)?;
            x = hidden;
            all_cross_attn_weights.push(ca_weights);
        }

        // Apply final layer norm (not strictly needed for attention capture,
        // but keeps the representation faithful).
        let _final_hidden = self.ln.forward(&x)?;

        Ok(all_cross_attn_weights)
    }
}

// ---------------------------------------------------------------------------
// DTW alignment
// ---------------------------------------------------------------------------

/// Align text tokens to audio frames using Dynamic Time Warping on attention weights.
///
/// `attention`: `[n_text, n_audio]` — averaged cross-attention matrix (higher = more aligned).
///
/// Returns `Vec<(token_idx, start_frame, end_frame)>` where each text token is assigned
/// a contiguous range of audio frames. `end_frame` is exclusive.
pub fn dtw_alignment(attention: &[Vec<f32>]) -> Vec<(usize, usize, usize)> {
    let n_text = attention.len();
    if n_text == 0 {
        return Vec::new();
    }
    let n_audio = attention[0].len();
    if n_audio == 0 {
        return Vec::new();
    }

    // Build cost matrix: cost[i][j] = -attention[i][j] + min(up, left, diag)
    let mut cost = vec![vec![f64::MAX; n_audio]; n_text];
    cost[0][0] = -(attention[0][0] as f64);

    // First row: can only come from the left
    for j in 1..n_audio {
        cost[0][j] = -(attention[0][j] as f64) + cost[0][j - 1];
    }

    // First column: can only come from above
    for i in 1..n_text {
        cost[i][0] = -(attention[i][0] as f64) + cost[i - 1][0];
    }

    // Fill interior
    for i in 1..n_text {
        for j in 1..n_audio {
            let prev = cost[i - 1][j]
                .min(cost[i][j - 1])
                .min(cost[i - 1][j - 1]);
            cost[i][j] = -(attention[i][j] as f64) + prev;
        }
    }

    // Backtrack to recover the optimal path
    let mut path: Vec<(usize, usize)> = Vec::new();
    let mut i = n_text - 1;
    let mut j = n_audio - 1;
    path.push((i, j));

    while i > 0 || j > 0 {
        if i == 0 {
            j -= 1;
        } else if j == 0 {
            i -= 1;
        } else {
            let diag = cost[i - 1][j - 1];
            let up = cost[i - 1][j];
            let left = cost[i][j - 1];
            if diag <= up && diag <= left {
                i -= 1;
                j -= 1;
            } else if up <= left {
                i -= 1;
            } else {
                j -= 1;
            }
        }
        path.push((i, j));
    }

    path.reverse();

    // Convert path to per-token frame ranges.
    // Each text token i maps to a contiguous range [start_frame, end_frame).
    let mut token_ranges: Vec<(usize, usize)> = vec![(usize::MAX, 0); n_text];
    for &(ti, aj) in &path {
        if token_ranges[ti].0 == usize::MAX {
            token_ranges[ti].0 = aj;
        }
        // end_frame is exclusive, so always update to aj + 1
        token_ranges[ti].1 = aj + 1;
    }

    // Fill in any tokens that were skipped (shouldn't happen with DTW, but be safe)
    for idx in 0..n_text {
        if token_ranges[idx].0 == usize::MAX {
            // Inherit from previous token
            if idx > 0 {
                token_ranges[idx] = (token_ranges[idx - 1].1, token_ranges[idx - 1].1);
            } else {
                token_ranges[idx] = (0, 0);
            }
        }
    }

    token_ranges
        .into_iter()
        .enumerate()
        .map(|(tok_idx, (start, end))| (tok_idx, start, end))
        .collect()
}

// ---------------------------------------------------------------------------
// Word-level timestamp extraction
// ---------------------------------------------------------------------------

/// Group per-token DTW alignments into word-level timestamps.
///
/// Whisper BPE tokens that start with a space (or the decoded text starts with a space)
/// mark word boundaries. This function decodes each token, detects boundaries, and
/// converts frame indices to seconds.
///
/// `alignment`: output of [`dtw_alignment`] — `Vec<(token_idx, start_frame, end_frame)>`
/// `tokens`: the raw token IDs from decoding (only the text tokens, no special tokens)
/// `tokenizer`: the Whisper tokenizer
pub fn extract_word_timestamps(
    alignment: &[(usize, usize, usize)],
    tokens: &[u32],
    tokenizer: &tokenizers::Tokenizer,
) -> Vec<WordTimestamp> {
    if alignment.is_empty() || tokens.is_empty() {
        return Vec::new();
    }

    // Decode each token individually to detect word boundaries
    struct TokenInfo {
        text: String,
        start_frame: usize,
        end_frame: usize,
    }

    let mut token_infos: Vec<TokenInfo> = Vec::new();
    for &(tok_idx, start_frame, end_frame) in alignment {
        if tok_idx >= tokens.len() {
            continue;
        }
        let tok_id = tokens[tok_idx];
        if let Ok(text) = tokenizer.decode(&[tok_id], false) {
            if !text.is_empty() {
                token_infos.push(TokenInfo {
                    text,
                    start_frame,
                    end_frame,
                });
            }
        }
    }

    if token_infos.is_empty() {
        return Vec::new();
    }

    // Group tokens into words.
    // A new word starts when a decoded token begins with a space character.
    let mut words: Vec<WordTimestamp> = Vec::new();
    let mut current_text = String::new();
    let mut word_start_frame: usize = 0;
    let mut word_end_frame: usize = 0;
    let mut in_word = false;

    for info in &token_infos {
        let starts_with_space = info.text.starts_with(' ');

        if starts_with_space && in_word {
            // Emit previous word
            let trimmed = current_text.trim().to_string();
            if !trimmed.is_empty() {
                words.push(WordTimestamp {
                    word: trimmed,
                    start_sec: frame_to_sec(word_start_frame),
                    end_sec: frame_to_sec(word_end_frame),
                });
            }
            current_text.clear();
        }

        if !in_word || starts_with_space {
            word_start_frame = info.start_frame;
            in_word = true;
        }

        current_text.push_str(&info.text);
        word_end_frame = info.end_frame;
    }

    // Emit the last word
    let trimmed = current_text.trim().to_string();
    if !trimmed.is_empty() {
        words.push(WordTimestamp {
            word: trimmed,
            start_sec: frame_to_sec(word_start_frame),
            end_sec: frame_to_sec(word_end_frame),
        });
    }

    words
}

/// Convert a mel-spectrogram frame index to seconds.
fn frame_to_sec(frame: usize) -> f32 {
    (frame as f32) * (HOP_LENGTH as f32) / (SAMPLE_RATE as f32)
}

// ---------------------------------------------------------------------------
// Helper: average cross-attention weights across heads and layers
// ---------------------------------------------------------------------------

/// Average cross-attention weights across all layers and heads.
///
/// Input: `Vec<Tensor>` where each is `[B, n_heads, T, S]` (one per layer).
/// Output: `[T, S]` (averaged, squeezed from batch dim; batch is assumed to be 1).
pub fn average_cross_attention_weights(
    layer_weights: &[Tensor],
    device: &Device,
) -> Result<Tensor> {
    // Stack along a new dim → [n_layers, B, n_heads, T, S]
    let stacked = Tensor::stack(layer_weights, 0)?;
    // Mean over layers (dim 0) and heads (dim 2, but after squeezing layers it shifts)
    // Shape after mean(0): [B, n_heads, T, S]
    let avg_layers = (stacked.sum(0)? / (layer_weights.len() as f64))?;
    // Shape: [B, n_heads, T, S] → mean over heads (dim 1) → [B, T, S]
    let n_heads = avg_layers.dim(1)? as f64;
    let avg_heads = (avg_layers.sum(1)? / n_heads)?;
    // Squeeze batch dim → [T, S]
    let result = avg_heads.squeeze(0)?;
    // Make sure we're on the right device and contiguous
    let result = result.to_device(device)?;
    Ok(result)
}

// ---------------------------------------------------------------------------
// DecoderConfig — extracted subset of Whisper Config needed by this module
// ---------------------------------------------------------------------------

// The DecoderConfig type is defined in the parent module (mod.rs) so that
// aligner.rs can construct it from the full whisper Config.

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtw_alignment_basic() {
        // 3 text tokens, 5 audio frames
        // Strong diagonal attention pattern
        let attention = vec![
            vec![0.8, 0.1, 0.05, 0.03, 0.02],
            vec![0.05, 0.1, 0.7, 0.1, 0.05],
            vec![0.02, 0.03, 0.05, 0.1, 0.8],
        ];

        let result = dtw_alignment(&attention);
        assert_eq!(result.len(), 3);

        // Token 0 should start at frame 0
        assert_eq!(result[0].0, 0); // token_idx
        assert_eq!(result[0].1, 0); // start_frame

        // Token 2 should end at frame 5 (exclusive)
        assert_eq!(result[2].0, 2);
        assert_eq!(result[2].2, 5);

        // Ranges must be monotonically non-decreasing
        for i in 1..result.len() {
            assert!(result[i].1 >= result[i - 1].1, "start frames must be non-decreasing");
        }
    }

    #[test]
    fn test_dtw_alignment_empty() {
        let result = dtw_alignment(&[]);
        assert!(result.is_empty());

        let result = dtw_alignment(&[vec![]]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dtw_alignment_single_token() {
        let attention = vec![vec![0.3, 0.5, 0.2]];
        let result = dtw_alignment(&attention);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], (0, 0, 3)); // single token spans all frames
    }

    #[test]
    fn test_frame_to_sec() {
        // frame 0 → 0.0s
        assert!((frame_to_sec(0) - 0.0).abs() < 1e-6);
        // frame 100 → 100 * 160 / 16000 = 1.0s
        assert!((frame_to_sec(100) - 1.0).abs() < 1e-6);
        // frame 50 → 50 * 160 / 16000 = 0.5s
        assert!((frame_to_sec(50) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_dtw_alignment_uniform_attention() {
        // When attention is uniform, DTW should still produce a valid monotonic alignment
        let attention = vec![
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.25, 0.25, 0.25, 0.25],
        ];
        let result = dtw_alignment(&attention);
        assert_eq!(result.len(), 2);
        // Must cover all frames
        assert_eq!(result[0].1, 0);
        assert_eq!(result[result.len() - 1].2, 4);
    }
}
