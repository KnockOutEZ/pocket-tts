#[cfg(not(target_arch = "wasm32"))]
pub mod aligner;
pub mod forced_align;

#[cfg(not(target_arch = "wasm32"))]
pub use aligner::WhisperAligner;
pub use forced_align::WordTimestamp;
