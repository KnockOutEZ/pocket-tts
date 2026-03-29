#[cfg(not(target_arch = "wasm32"))]
pub mod aligner;
pub(crate) mod forced_align;

#[cfg(not(target_arch = "wasm32"))]
pub use aligner::NativeAligner;
pub use forced_align::WordTimestamp;
