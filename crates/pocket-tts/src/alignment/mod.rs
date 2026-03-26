#[cfg(not(target_arch = "wasm32"))]
pub mod aligner;
#[cfg(not(target_arch = "wasm32"))]
pub mod ar_decoder;
#[cfg(not(target_arch = "wasm32"))]
pub mod dtw_decoder;
pub mod forced_align;

#[cfg(not(target_arch = "wasm32"))]
pub use aligner::WhisperAligner;
pub use forced_align::WordTimestamp;

/// Subset of the Whisper config needed by the cross-attention decoder.
///
/// Constructed from `candle_transformers::models::whisper::Config` in `aligner.rs`.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    pub d_model: usize,
    pub decoder_attention_heads: usize,
    pub decoder_layers: usize,
    pub vocab_size: usize,
    pub max_target_positions: usize,
}
