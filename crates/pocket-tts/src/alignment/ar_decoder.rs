//! Autoregressive Whisper decoder with cross-attention weight capture.
//!
//! Processes tokens one at a time with KV caching, capturing cross-attention
//! softmax weights at each step. This is the correct approach for extracting
//! meaningful attention weights — teacher forcing (all tokens at once) produces
//! diffuse/garbage cross-attention because the model never learns to attend
//! properly in that mode.
//!
//! After generating all tokens, the accumulated attention weights are averaged
//! across layers and heads to produce a `[n_tokens, n_audio_frames]` matrix
//! suitable for DTW alignment.

use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder};

// ---------------------------------------------------------------------------
// Public config
// ---------------------------------------------------------------------------

/// Config subset needed by the autoregressive decoder.
#[derive(Debug, Clone)]
pub struct ARDecoderConfig {
    pub d_model: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub n_vocab: usize,
    pub n_ctx: usize, // max_target_positions (448)
}

// ---------------------------------------------------------------------------
// KV caches
// ---------------------------------------------------------------------------

/// Self-attention KV cache — grows by one position each step.
#[derive(Debug, Clone)]
struct SelfAttnCache {
    /// `[1, n_head, seq_so_far, head_dim]`
    k: Option<Tensor>,
    /// `[1, n_head, seq_so_far, head_dim]`
    v: Option<Tensor>,
}

impl SelfAttnCache {
    fn new() -> Self {
        Self { k: None, v: None }
    }

    fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }
}

/// Cross-attention KV cache — computed once from encoder output, then reused.
#[derive(Debug, Clone)]
struct CrossAttnCache {
    /// `[1, n_head, n_audio, head_dim]`
    k: Option<Tensor>,
    /// `[1, n_head, n_audio, head_dim]`
    v: Option<Tensor>,
}

impl CrossAttnCache {
    fn new() -> Self {
        Self { k: None, v: None }
    }

    fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }
}

// ---------------------------------------------------------------------------
// Multi-Head Attention (self-attention variant with KV cache)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct SelfAttention {
    q_proj: Linear,
    k_proj: Linear, // no bias
    v_proj: Linear,
    out_proj: Linear,
    n_head: usize,
    head_dim: usize,
}

impl SelfAttention {
    fn load(d_model: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = d_model / n_head;
        let q_proj = candle_nn::linear(d_model, d_model, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear_no_bias(d_model, d_model, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(d_model, d_model, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(d_model, d_model, vb.pp("out_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            n_head,
            head_dim,
        })
    }

    /// Reshape `[1, 1, d_model]` -> `[1, n_head, 1, head_dim]`.
    fn reshape_head(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;
        x.reshape((b, t, self.n_head, self.head_dim))?
            .transpose(1, 2)
    }

    /// Forward one step with KV caching.
    ///
    /// `x`: `[1, 1, d_model]` — current token hidden state.
    /// Returns `[1, 1, d_model]`.
    fn forward_one(&self, x: &Tensor, cache: &mut SelfAttnCache) -> Result<Tensor> {
        let q = self.q_proj.forward(x)?;
        let k_new = self.k_proj.forward(x)?;
        let v_new = self.v_proj.forward(x)?;

        // Reshape to head form: [1, n_head, 1, head_dim]
        let q = self.reshape_head(&q)?;
        let k_new = self.reshape_head(&k_new)?;
        let v_new = self.reshape_head(&v_new)?;

        // Concatenate with cache
        let k = match &cache.k {
            Some(prev) => Tensor::cat(&[prev, &k_new], 2)?,
            None => k_new,
        };
        let v = match &cache.v {
            Some(prev) => Tensor::cat(&[prev, &v_new], 2)?,
            None => v_new,
        };

        // Update cache
        cache.k = Some(k.clone());
        cache.v = Some(v.clone());

        // Scaled dot-product attention
        // q: [1, n_head, 1, head_dim], k: [1, n_head, seq_len, head_dim]
        let scale = (self.head_dim as f64).powf(-0.5);
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        // Causal masking is implicit: since we only have query for the current
        // position and K/V for positions 0..=current, the attention is already
        // correctly causal. No explicit mask needed.

        let attn_probs = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_probs.matmul(&v)?;

        // [1, n_head, 1, head_dim] -> [1, 1, d_model]
        let attn_output = attn_output.transpose(1, 2)?.flatten_from(2)?;
        self.out_proj.forward(&attn_output)
    }
}

// ---------------------------------------------------------------------------
// Multi-Head Cross-Attention (captures softmax weights)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct CrossAttention {
    q_proj: Linear,
    k_proj: Linear, // no bias
    v_proj: Linear,
    out_proj: Linear,
    n_head: usize,
    head_dim: usize,
}

impl CrossAttention {
    fn load(d_model: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = d_model / n_head;
        let q_proj = candle_nn::linear(d_model, d_model, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear_no_bias(d_model, d_model, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(d_model, d_model, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(d_model, d_model, vb.pp("out_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            n_head,
            head_dim,
        })
    }

    /// Reshape `[1, T, d_model]` -> `[1, n_head, T, head_dim]`.
    fn reshape_head(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;
        x.reshape((b, t, self.n_head, self.head_dim))?
            .transpose(1, 2)
    }

    /// Forward one step. K/V are computed from encoder output on first call,
    /// then cached for subsequent calls.
    ///
    /// `x`: `[1, 1, d_model]` — current decoder hidden state.
    /// `encoder_output`: `[1, n_audio, d_model]` — encoder features.
    ///
    /// Returns `(output [1, 1, d_model], attn_weights [n_head, n_audio])`.
    fn forward_one(
        &self,
        x: &Tensor,
        encoder_output: &Tensor,
        cache: &mut CrossAttnCache,
    ) -> Result<(Tensor, Tensor)> {
        let q = self.q_proj.forward(x)?;
        let q = self.reshape_head(&q)?;

        // Compute or reuse cached K/V from encoder output
        let (k, v) = match (&cache.k, &cache.v) {
            (Some(k), Some(v)) => (k.clone(), v.clone()),
            _ => {
                let k = self.reshape_head(&self.k_proj.forward(encoder_output)?)?;
                let v = self.reshape_head(&self.v_proj.forward(encoder_output)?)?;
                cache.k = Some(k.clone());
                cache.v = Some(v.clone());
                (k, v)
            }
        };

        // Scaled dot-product attention
        // q: [1, n_head, 1, head_dim], k: [1, n_head, n_audio, head_dim]
        let scale = (self.head_dim as f64).powf(-0.5);
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        // attn_weights: [1, n_head, 1, n_audio]

        let attn_probs = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_probs.matmul(&v)?;

        // [1, n_head, 1, head_dim] -> [1, 1, d_model]
        let attn_output = attn_output.transpose(1, 2)?.flatten_from(2)?;
        let output = self.out_proj.forward(&attn_output)?;

        // Extract attention weights for capture: [1, n_head, 1, n_audio] -> [n_head, n_audio]
        let captured_weights = attn_probs.squeeze(0)?.squeeze(1)?;

        Ok((output, captured_weights))
    }
}

// ---------------------------------------------------------------------------
// Decoder Layer
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct DecoderLayer {
    self_attn: SelfAttention,
    self_attn_ln: LayerNorm,
    cross_attn: CrossAttention,
    cross_attn_ln: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_ln: LayerNorm,
    self_attn_cache: SelfAttnCache,
    cross_attn_cache: CrossAttnCache,
}

impl DecoderLayer {
    fn load(d_model: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let self_attn = SelfAttention::load(d_model, n_head, vb.pp("self_attn"))?;
        let self_attn_ln =
            candle_nn::layer_norm(d_model, 1e-5, vb.pp("self_attn_layer_norm"))?;
        let cross_attn = CrossAttention::load(d_model, n_head, vb.pp("encoder_attn"))?;
        let cross_attn_ln =
            candle_nn::layer_norm(d_model, 1e-5, vb.pp("encoder_attn_layer_norm"))?;
        let fc1 = candle_nn::linear(d_model, d_model * 4, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(d_model * 4, d_model, vb.pp("fc2"))?;
        let final_ln = candle_nn::layer_norm(d_model, 1e-5, vb.pp("final_layer_norm"))?;
        Ok(Self {
            self_attn,
            self_attn_ln,
            cross_attn,
            cross_attn_ln,
            fc1,
            fc2,
            final_ln,
            self_attn_cache: SelfAttnCache::new(),
            cross_attn_cache: CrossAttnCache::new(),
        })
    }

    fn reset_cache(&mut self) {
        self.self_attn_cache.reset();
        self.cross_attn_cache.reset();
    }

    /// Forward one step through this layer.
    ///
    /// Returns `(hidden [1, 1, d_model], cross_attn_weights [n_head, n_audio])`.
    fn forward_one(
        &mut self,
        x: &Tensor,
        encoder_output: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // 1. LayerNorm -> Self-Attention (causal, with KV cache) -> residual
        let sa_out = self
            .self_attn
            .forward_one(&self.self_attn_ln.forward(x)?, &mut self.self_attn_cache)?;
        let x = (x + sa_out)?;

        // 2. LayerNorm -> Cross-Attention (to encoder, cache K/V once) -> residual
        let (ca_out, ca_weights) = self.cross_attn.forward_one(
            &self.cross_attn_ln.forward(&x)?,
            encoder_output,
            &mut self.cross_attn_cache,
        )?;
        let x = (&x + ca_out)?;

        // 3. LayerNorm -> FFN (Linear -> GELU -> Linear) -> residual
        let ffn_out = self
            .fc2
            .forward(&self.fc1.forward(&self.final_ln.forward(&x)?)?.gelu()?)?;
        let x = (x + ffn_out)?;

        Ok((x, ca_weights))
    }
}

// ---------------------------------------------------------------------------
// ARDecoder
// ---------------------------------------------------------------------------

/// Autoregressive Whisper decoder with cross-attention capture.
///
/// Processes tokens one at a time, maintaining KV caches for self-attention
/// (grows each step) and cross-attention (computed once from encoder output).
/// Cross-attention softmax weights are captured at each step for later
/// extraction as a `[n_tokens, n_audio_frames]` alignment matrix.
#[derive(Clone, Debug)]
pub struct ARDecoder {
    token_embedding: Embedding,
    positional_embedding: Tensor,
    layers: Vec<DecoderLayer>,
    ln: LayerNorm,
    /// Per-layer, per-step attention weights. `attn_weights[layer][step]` is `[n_head, n_audio]`.
    attn_weights: Vec<Vec<Tensor>>,
    n_layer: usize,
}

impl ARDecoder {
    /// Load decoder weights from VarBuilder rooted at `model.decoder`.
    pub fn load(vb: VarBuilder, config: &ARDecoderConfig) -> Result<Self> {
        let d_model = config.d_model;

        let token_embedding =
            candle_nn::embedding(config.n_vocab, d_model, vb.pp("embed_tokens"))?;
        let positional_embedding = vb.get((config.n_ctx, d_model), "embed_positions.weight")?;

        let layers = (0..config.n_layer)
            .map(|i| DecoderLayer::load(d_model, config.n_head, vb.pp(format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;

        let ln = candle_nn::layer_norm(d_model, 1e-5, vb.pp("layer_norm"))?;

        let attn_weights = vec![Vec::new(); config.n_layer];

        Ok(Self {
            token_embedding,
            positional_embedding,
            layers,
            ln,
            attn_weights,
            n_layer: config.n_layer,
        })
    }

    /// Reset all KV caches and accumulated attention weights.
    /// Call before each new audio.
    pub fn reset_cache(&mut self) {
        for layer in &mut self.layers {
            layer.reset_cache();
        }
        self.attn_weights = vec![Vec::new(); self.n_layer];
    }

    /// Process one token autoregressively.
    ///
    /// Returns logits `[vocab_size]` for the next token.
    /// Cross-attention weights are captured internally.
    pub fn forward_one(
        &mut self,
        token: u32,
        encoder_output: &Tensor,
        position: usize,
    ) -> Result<Tensor> {
        let device = encoder_output.device();

        // Token + positional embedding: [1, 1, d_model]
        let token_tensor = Tensor::new(&[token], device)?.unsqueeze(0)?;
        let tok_emb = self.token_embedding.forward(&token_tensor)?;
        let pos_emb = self.positional_embedding.i((position..position + 1, ..))?;
        let mut hidden = tok_emb.broadcast_add(&pos_emb.unsqueeze(0)?)?;
        // hidden: [1, 1, d_model]

        // Run through each layer, capturing cross-attention weights
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let (h, ca_weights) = layer.forward_one(&hidden, encoder_output)?;
            hidden = h;
            self.attn_weights[layer_idx].push(ca_weights);
        }

        // Final layer norm
        let hidden = self.ln.forward(&hidden)?;

        // Project to vocab (weight tying with token embeddings)
        // hidden: [1, 1, d_model], embed: [vocab_size, d_model]
        let logits = hidden.matmul(&self.token_embedding.embeddings().t()?)?;

        // Squeeze to [vocab_size]
        logits.squeeze(0)?.squeeze(0)
    }

    /// Get the accumulated cross-attention weights as `[n_tokens, n_audio_frames]`.
    ///
    /// Averaged across layers and heads. Call after all tokens have been processed.
    pub fn get_cross_attention_matrix(&self) -> Result<Tensor> {
        if self.attn_weights.is_empty() || self.attn_weights[0].is_empty() {
            return Err(candle_core::Error::Msg(
                "No attention weights captured. Call forward_one first.".to_string(),
            ));
        }

        let n_steps = self.attn_weights[0].len();
        let n_layers = self.attn_weights.len();

        // For each step, average attention across all layers and heads.
        // Each attn_weights[layer][step] is [n_head, n_audio].
        let mut step_tensors: Vec<Tensor> = Vec::with_capacity(n_steps);

        for step in 0..n_steps {
            // Collect this step's weights from all layers: each is [n_head, n_audio]
            let layer_tensors: Vec<&Tensor> = (0..n_layers)
                .map(|layer| &self.attn_weights[layer][step])
                .collect();

            // Stack across layers -> [n_layers, n_head, n_audio]
            let stacked = Tensor::stack(&layer_tensors, 0)?;

            // Mean over layers (dim 0) -> [n_head, n_audio]
            let avg_layers = (stacked.sum(0)? / (n_layers as f64))?;

            // Mean over heads (dim 0) -> [n_audio]
            let n_heads = avg_layers.dim(0)? as f64;
            let avg_heads = (avg_layers.sum(0)? / n_heads)?;

            step_tensors.push(avg_heads);
        }

        // Stack all steps -> [n_tokens, n_audio]
        Tensor::stack(&step_tensors, 0)
    }
}
