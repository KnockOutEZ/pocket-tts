//! Wav2Vec2-base-960h model implemented in Candle.
//!
//! This module provides a batch-mode (non-streaming) implementation of the
//! wav2vec2-base-960h architecture for producing CTC log-probabilities from
//! 16 kHz audio.  The log-probs are consumed by the forced-alignment module
//! to generate word-level timestamps.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{
    conv1d_no_bias, group_norm, layer_norm, linear, Conv1d, Conv1dConfig, GroupNorm,
    LayerNorm, Linear, Module, VarBuilder,
};

// ---------------------------------------------------------------------------
// Constants (wav2vec2-base-960h — fixed architecture)
// ---------------------------------------------------------------------------

const HIDDEN_SIZE: usize = 768;
const FEAT_DIM: usize = 512;
const NUM_HEADS: usize = 12;
const HEAD_DIM: usize = HIDDEN_SIZE / NUM_HEADS; // 64
const INTERMEDIATE_SIZE: usize = 3072;
const NUM_LAYERS: usize = 12;
const VOCAB_SIZE: usize = 32;
const POS_CONV_KERNEL: usize = 128;
const POS_CONV_GROUPS: usize = 16;
const LAYER_NORM_EPS: f64 = 1e-5;

// CNN feature extractor layer specs: (in_ch, out_ch, kernel, stride)
const CNN_LAYERS: [(usize, usize, usize, usize); 7] = [
    (1, 512, 10, 5),
    (512, 512, 3, 2),
    (512, 512, 3, 2),
    (512, 512, 3, 2),
    (512, 512, 3, 2),
    (512, 512, 2, 2),
    (512, 512, 2, 2),
];

// ---------------------------------------------------------------------------
// CNN Feature Extractor
// ---------------------------------------------------------------------------

/// One layer of the CNN feature extractor.
#[derive(Clone, Debug)]
struct CnnLayer {
    conv: Conv1d,
    group_norm: Option<GroupNorm>,
}

impl CnnLayer {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.conv.forward(x)?;
        let x = match &self.group_norm {
            Some(gn) => gn.forward(&x)?,
            None => x,
        };
        x.gelu()
    }
}

/// 7-layer CNN feature extractor producing [B, 512, frames] from raw audio.
#[derive(Clone, Debug)]
struct FeatureExtractor {
    layers: Vec<CnnLayer>,
}

impl FeatureExtractor {
    fn load(vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(7);
        for (i, &(in_ch, out_ch, kernel, stride)) in CNN_LAYERS.iter().enumerate() {
            let vb_conv = vb.pp(format!("conv_layers.{}.conv", i));
            let cfg = Conv1dConfig {
                stride,
                ..Default::default()
            };
            let conv = conv1d_no_bias(in_ch, out_ch, kernel, cfg, vb_conv)
                .with_context(|| format!("loading CNN layer {} conv", i))?;

            let group_norm = if i == 0 {
                // Layer 0 has GroupNorm(512, 512)
                let vb_gn = vb.pp(format!("conv_layers.{}.layer_norm", i));
                Some(
                    group_norm(out_ch, out_ch, LAYER_NORM_EPS, vb_gn)
                        .with_context(|| "loading CNN layer 0 group_norm")?,
                )
            } else {
                None
            };

            layers.push(CnnLayer { conv, group_norm });
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// Feature Projection
// ---------------------------------------------------------------------------

/// LayerNorm(512) + Linear(512 -> 768).
#[derive(Clone, Debug)]
struct FeatureProjection {
    layer_norm: LayerNorm,
    projection: Linear,
}

impl FeatureProjection {
    fn load(vb: VarBuilder) -> Result<Self> {
        let layer_norm = layer_norm(FEAT_DIM, LAYER_NORM_EPS, vb.pp("layer_norm"))
            .context("loading feature_projection layer_norm")?;
        let projection = linear(FEAT_DIM, HIDDEN_SIZE, vb.pp("projection"))
            .context("loading feature_projection projection")?;
        Ok(Self {
            layer_norm,
            projection,
        })
    }

    /// Input: [B, frames, 512] -> Output: [B, frames, 768]
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.layer_norm.forward(x)?;
        self.projection.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// Positional Convolutional Embedding
// ---------------------------------------------------------------------------

/// Conv1d(768 -> 768, kernel=128, padding=64, groups=16) + GELU, trimmed by 1.
#[derive(Clone, Debug)]
struct PosConvEmbed {
    conv: Conv1d,
}

impl PosConvEmbed {
    fn load(vb: VarBuilder) -> Result<Self> {
        // The positional conv in wav2vec2-base-960h uses PyTorch weight
        // normalization, so the safetensors file stores `weight_g` and
        // `weight_v` instead of a plain `weight` tensor.  We reconstruct
        // the effective weight here:
        //   weight = weight_g * weight_v / ||weight_v||_2
        let pos_vb = vb.pp("conv");

        let weight_g = pos_vb
            .get((1, 1, POS_CONV_KERNEL), "weight_g")
            .context("loading pos_conv_embed weight_g")?;
        let weight_v = pos_vb
            .get(
                (HIDDEN_SIZE, HIDDEN_SIZE / POS_CONV_GROUPS, POS_CONV_KERNEL),
                "weight_v",
            )
            .context("loading pos_conv_embed weight_v")?;
        let bias = pos_vb
            .get(HIDDEN_SIZE, "bias")
            .context("loading pos_conv_embed bias")?;

        // L2 norm of weight_v over dims [0, 1], keepdim -> [1, 1, kernel]
        let norm = weight_v
            .sqr()?
            .sum_keepdim((0usize, 1usize))?
            .sqrt()
            .context("computing weight_v L2 norm")?;
        let weight = weight_v
            .broadcast_mul(&weight_g)?
            .broadcast_div(&norm)
            .context("reconstructing weight-normed conv weight")?;

        let cfg = Conv1dConfig {
            padding: POS_CONV_KERNEL / 2, // 64
            groups: POS_CONV_GROUPS,
            ..Default::default()
        };
        let conv = Conv1d::new(weight, Some(bias), cfg);
        Ok(Self { conv })
    }

    /// Input: [B, 768, frames] -> Output: [B, 768, frames] (same length).
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let out = self.conv.forward(x)?;
        // Padding 64 on each side with kernel 128 adds 1 extra time step.
        // Trim the last time step to restore original length.
        let seq_len = x.dim(2)?;
        let out = out.narrow(2, 0, seq_len)?;
        out.gelu()
    }
}

// ---------------------------------------------------------------------------
// Self-Attention (bidirectional, no KV cache)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct SelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
}

impl SelfAttention {
    fn load(vb: VarBuilder) -> Result<Self> {
        let q_proj = linear(HIDDEN_SIZE, HIDDEN_SIZE, vb.pp("q_proj"))
            .context("loading attention q_proj")?;
        let k_proj = linear(HIDDEN_SIZE, HIDDEN_SIZE, vb.pp("k_proj"))
            .context("loading attention k_proj")?;
        let v_proj = linear(HIDDEN_SIZE, HIDDEN_SIZE, vb.pp("v_proj"))
            .context("loading attention v_proj")?;
        let out_proj = linear(HIDDEN_SIZE, HIDDEN_SIZE, vb.pp("out_proj"))
            .context("loading attention out_proj")?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        })
    }

    /// Input: [B, T, 768] -> Output: [B, T, 768]
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [B, num_heads, T, head_dim]
        let q = q
            .reshape((b, t, NUM_HEADS, HEAD_DIM))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, t, NUM_HEADS, HEAD_DIM))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, t, NUM_HEADS, HEAD_DIM))?
            .transpose(1, 2)?;

        // Scaled dot-product attention (no causal mask)
        let scale = (HEAD_DIM as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? / scale)?;
        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back to [B, T, 768]
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, HIDDEN_SIZE))?;

        self.out_proj.forward(&attn_output)
    }
}

// ---------------------------------------------------------------------------
// Feed-Forward Network
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct FeedForward {
    intermediate_dense: Linear,
    output_dense: Linear,
}

impl FeedForward {
    fn load(vb: VarBuilder) -> Result<Self> {
        let intermediate_dense =
            linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, vb.pp("intermediate_dense"))
                .context("loading ffn intermediate_dense")?;
        let output_dense = linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, vb.pp("output_dense"))
            .context("loading ffn output_dense")?;
        Ok(Self {
            intermediate_dense,
            output_dense,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.intermediate_dense.forward(x)?.gelu()?;
        self.output_dense.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// Transformer Encoder Layer
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct EncoderLayer {
    attention: SelfAttention,
    layer_norm: LayerNorm,
    feed_forward: FeedForward,
    final_layer_norm: LayerNorm,
}

impl EncoderLayer {
    fn load(vb: VarBuilder) -> Result<Self> {
        let attention =
            SelfAttention::load(vb.pp("attention")).context("loading encoder attention")?;
        let ln = candle_nn::layer_norm(HIDDEN_SIZE, LAYER_NORM_EPS, vb.pp("layer_norm"))
            .context("loading encoder layer_norm")?;
        let feed_forward =
            FeedForward::load(vb.pp("feed_forward")).context("loading encoder ffn")?;
        let final_ln =
            candle_nn::layer_norm(HIDDEN_SIZE, LAYER_NORM_EPS, vb.pp("final_layer_norm"))
                .context("loading encoder final_layer_norm")?;
        Ok(Self {
            attention,
            layer_norm: ln,
            feed_forward,
            final_layer_norm: final_ln,
        })
    }

    /// Pre-norm transformer: LN -> Attn -> residual, LN -> FFN -> residual.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // Self-attention block
        let residual = x;
        let x = self.layer_norm.forward(x)?;
        let x = self.attention.forward(&x)?;
        let x = (x + residual)?;

        // Feed-forward block
        let residual = &x;
        let h = self.final_layer_norm.forward(&x)?;
        let h = self.feed_forward.forward(&h)?;
        h + residual
    }
}

// ---------------------------------------------------------------------------
// Full Wav2Vec2 Model
// ---------------------------------------------------------------------------

/// Wav2Vec2-base-960h model for producing CTC log-probabilities.
///
/// Architecture:
/// - 7-layer CNN feature extractor (16 kHz audio -> 50 fps features)
/// - Feature projection (512 -> 768)
/// - Convolutional positional embedding
/// - 12 transformer encoder layers
/// - CTC head (768 -> 32 vocab, log-softmax)
#[derive(Clone, Debug)]
pub struct Wav2Vec2Model {
    feature_extractor: FeatureExtractor,
    feature_projection: FeatureProjection,
    pos_conv_embed: PosConvEmbed,
    encoder_layer_norm: LayerNorm,
    encoder_layers: Vec<EncoderLayer>,
    lm_head: Linear,
}

impl Wav2Vec2Model {
    /// Load the wav2vec2-base-960h model from HuggingFace Hub.
    ///
    /// Downloads ~360 MB of weights on first call (cached thereafter).
    pub fn load(device: &Device) -> Result<Self> {
        let weights_path = crate::weights::download_if_necessary(
            "hf://facebook/wav2vec2-base-960h/model.safetensors",
        )
        .context("downloading wav2vec2-base-960h weights")?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)
                .context("memory-mapping wav2vec2 safetensors")?
        };

        Self::load_from_vb(vb)
    }

    /// Build the model from a `VarBuilder` (useful for testing with custom weights).
    fn load_from_vb(vb: VarBuilder) -> Result<Self> {
        let vb_w2v = vb.pp("wav2vec2");

        let feature_extractor =
            FeatureExtractor::load(vb_w2v.pp("feature_extractor"))
                .context("loading feature_extractor")?;

        let feature_projection =
            FeatureProjection::load(vb_w2v.pp("feature_projection"))
                .context("loading feature_projection")?;

        let pos_conv_embed =
            PosConvEmbed::load(vb_w2v.pp("encoder.pos_conv_embed"))
                .context("loading pos_conv_embed")?;

        let encoder_layer_norm =
            layer_norm(HIDDEN_SIZE, LAYER_NORM_EPS, vb_w2v.pp("encoder.layer_norm"))
                .context("loading encoder layer_norm")?;

        let mut encoder_layers = Vec::with_capacity(NUM_LAYERS);
        for i in 0..NUM_LAYERS {
            let layer = EncoderLayer::load(vb_w2v.pp(format!("encoder.layers.{}", i)))
                .with_context(|| format!("loading encoder layer {}", i))?;
            encoder_layers.push(layer);
        }

        let lm_head =
            linear(HIDDEN_SIZE, VOCAB_SIZE, vb.pp("lm_head")).context("loading lm_head")?;

        Ok(Self {
            feature_extractor,
            feature_projection,
            pos_conv_embed,
            encoder_layer_norm,
            encoder_layers,
            lm_head,
        })
    }

    /// Run the full forward pass.
    ///
    /// - `audio`: `[B, 1, samples]` — raw 16 kHz waveform.
    /// - Returns: `[B, frames, 32]` — CTC log-probabilities (log-softmax over vocab).
    pub fn forward(&self, audio: &Tensor) -> Result<Tensor> {
        // CNN feature extraction: [B, 1, T] -> [B, 512, F]
        let features = self
            .feature_extractor
            .forward(audio)
            .context("CNN feature extraction")?;

        // Transpose to [B, F, 512] for the projection
        let features = features.transpose(1, 2).context("transpose after CNN")?;

        // Feature projection: [B, F, 512] -> [B, F, 768]
        let hidden = self
            .feature_projection
            .forward(&features)
            .context("feature projection")?;

        // Positional conv embedding operates on [B, 768, F]
        let hidden_t = hidden.transpose(1, 2).context("transpose for pos_conv")?;
        let pos_embed = self
            .pos_conv_embed
            .forward(&hidden_t)
            .context("positional conv embedding")?;
        // Add positional embedding and transpose back to [B, F, 768]
        let hidden = (hidden_t + pos_embed)
            .context("add pos embed")?
            .transpose(1, 2)
            .context("transpose after pos_conv")?;

        // Encoder layer norm
        let mut hidden = self
            .encoder_layer_norm
            .forward(&hidden)
            .context("encoder layer_norm")?;

        // Transformer encoder layers
        for (i, layer) in self.encoder_layers.iter().enumerate() {
            hidden = layer
                .forward(&hidden)
                .with_context(|| format!("encoder layer {}", i))?;
        }

        // CTC head: [B, F, 768] -> [B, F, 32]
        let logits = self.lm_head.forward(&hidden).context("lm_head")?;

        // Log-softmax over vocabulary dimension
        let log_probs =
            candle_nn::ops::log_softmax(&logits, D::Minus1).context("log_softmax")?;

        Ok(log_probs)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

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
        assert_eq!(log_probs.dims()[0], 1); // batch
        assert_eq!(log_probs.dims()[2], 32); // vocab
        // Frame count depends on conv padding; expect ~49-50
        let frames = log_probs.dims()[1];
        assert!(
            frames >= 40 && frames <= 55,
            "unexpected frame count: {}",
            frames
        );
    }
}
