#[cfg(not(target_arch = "wasm32"))]
pub mod aligner;
pub mod forced_align;
#[cfg(not(target_arch = "wasm32"))]
pub mod wav2vec2;

#[cfg(not(target_arch = "wasm32"))]
pub use aligner::Wav2Vec2Aligner;
pub use forced_align::WordTimestamp;
