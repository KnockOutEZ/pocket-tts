//! Native ONNX-based aligner for word-level timestamps.
//!
//! Uses wav2vec2 (quantized ONNX) + Viterbi forced alignment for
//! word-level timestamps (~20-50ms accuracy). No Python dependency.

use crate::alignment::forced_align::{
    path_to_word_timestamps, text_to_ctc_targets, viterbi_forced_align, WordTimestamp,
};
use candle_core::{Device, Tensor};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor as OrtTensor;
use std::sync::{Arc, Mutex};

const TTS_SAMPLE_RATE: u32 = 24000;
const WHISPER_SAMPLE_RATE: u32 = 16000;
const FRAME_DURATION_SEC: f32 = 0.02;
const ALIGNMENT_REPO: &str = "KnockOutEZ/pocket-tts-alignment";
const ONNX_FILENAME: &str = "wav2vec2-large-int8.onnx";
const VOCAB_FILENAME: &str = "vocab.json";

/// Native ONNX-based aligner for production-grade word timestamps.
///
/// Uses a quantized wav2vec2 model for CTC emissions and Viterbi
/// forced alignment to map text characters to audio frames.
#[derive(Clone)]
pub struct NativeAligner {
    session: Arc<Mutex<Session>>,
    vocab: Vec<String>,
    blank_id: usize,
}

impl NativeAligner {
    /// Load the aligner. Downloads the ONNX model + vocab if not cached,
    /// then creates an ORT inference session.
    pub fn load(_device: &Device) -> anyhow::Result<Self> {
        let onnx_path = crate::weights::download_if_necessary(&format!(
            "hf://{}/{}",
            ALIGNMENT_REPO, ONNX_FILENAME
        ))?;
        let vocab_path = crate::weights::download_if_necessary(&format!(
            "hf://{}/{}",
            ALIGNMENT_REPO, VOCAB_FILENAME
        ))?;

        let mut builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create ORT session builder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {}", e))?;
        let session = builder
            .commit_from_file(&onnx_path)
            .map_err(|e| anyhow::anyhow!("Failed to load ONNX model: {}", e))?;

        let vocab_json: serde_json::Value =
            serde_json::from_reader(std::io::BufReader::new(std::fs::File::open(&vocab_path)?))?;

        // vocab.json is { "a": 1, "b": 2, ... } -- invert to index->token
        let map = vocab_json
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("vocab.json is not a JSON object"))?;

        let max_id = map.values().filter_map(|v| v.as_u64()).max().unwrap_or(0) as usize;
        let mut vocab = vec![String::new(); max_id + 1];
        let mut blank_id = 0usize;

        for (token, id_val) in map {
            let id = id_val.as_u64().ok_or_else(|| {
                anyhow::anyhow!("vocab.json value for '{}' is not an integer", token)
            })? as usize;
            // Normalize special tokens
            let normalized = match token.as_str() {
                "|" => " ".to_string(),
                t => t.to_lowercase(),
            };
            if token == "<pad>" {
                blank_id = id;
            }
            if id < vocab.len() {
                vocab[id] = normalized;
            }
        }

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            vocab,
            blank_id,
        })
    }

    /// Check if the alignment model files are already cached locally.
    /// Does NOT download anything or create an ORT session.
    pub fn check_available() -> anyhow::Result<()> {
        let home = dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
        let cache_root = home.join(".cache").join("huggingface").join("hub");
        let repo_dir = cache_root.join(format!(
            "models--{}",
            ALIGNMENT_REPO.replace('/', "--")
        ));

        if !repo_dir.exists() {
            anyhow::bail!(
                "Alignment model not cached. Run preload_models() first. Expected: {:?}",
                repo_dir
            );
        }
        Ok(())
    }

    /// Download model + vocab if not already cached. Instant if cached.
    pub fn preload_models() -> anyhow::Result<()> {
        crate::weights::download_if_necessary(&format!(
            "hf://{}/{}",
            ALIGNMENT_REPO, ONNX_FILENAME
        ))?;
        crate::weights::download_if_necessary(&format!(
            "hf://{}/{}",
            ALIGNMENT_REPO, VOCAB_FILENAME
        ))?;
        Ok(())
    }

    /// Align audio to produce word-level timestamps.
    ///
    /// Pipeline: normalize mono -> resample 24kHz->16kHz -> ONNX inference ->
    /// log-softmax -> CTC targets -> Viterbi -> word timestamps.
    pub fn align(&self, audio: &Tensor, text: &str) -> anyhow::Result<Vec<WordTimestamp>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        // 1. Normalize to mono [1, N]
        let audio = match audio.dims().len() {
            1 => audio.unsqueeze(0)?,
            2 if audio.dims()[0] == 1 => audio.clone(),
            2 => audio.mean(0)?.unsqueeze(0)?,
            _ => anyhow::bail!("Unexpected audio shape: {:?}", audio.dims()),
        };

        // 2. Ensure CPU
        let audio = audio.to_device(&Device::Cpu)?;

        // 3. Resample 24kHz -> 16kHz
        let audio_16k =
            crate::audio::resample_for_alignment(&audio, TTS_SAMPLE_RATE, WHISPER_SAMPLE_RATE)?;

        // 4. Extract f32 samples
        let samples = audio_16k.flatten_all()?.to_vec1::<f32>()?;
        let num_samples = samples.len();

        // 5. Build ONNX input [1, num_samples]
        let input_value = OrtTensor::from_array(([1usize, num_samples], samples.into_boxed_slice()))
            .map_err(|e| anyhow::anyhow!("Failed to create ORT input tensor: {}", e))?;

        // 6. Run inference + extract logits (scoped to release session lock)
        let emissions = {
            let mut session = self.session.lock().unwrap();
            let outputs = session
                .run(ort::inputs![input_value])
                .map_err(|e| anyhow::anyhow!("ONNX inference failed: {}", e))?;

            // 7. Extract logits and apply log-softmax per frame
            let (shape, logits_data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("Failed to extract logits: {}", e))?;

            // shape is [1, T, vocab_size]
            let t_len = shape[1] as usize;
            let vocab_size = shape[2] as usize;

            let mut emissions: Vec<Vec<f32>> = Vec::with_capacity(t_len);
            for t in 0..t_len {
                let offset = t * vocab_size;
                let mut frame: Vec<f32> = logits_data[offset..offset + vocab_size].to_vec();

                // Log-softmax: log(exp(x_i) / sum(exp(x_j)))
                let max_val = frame.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum_exp: f32 = frame.iter().map(|&x| (x - max_val).exp()).sum();
                let log_sum_exp = max_val + sum_exp.ln();
                for val in &mut frame {
                    *val -= log_sum_exp;
                }
                emissions.push(frame);
            }
            emissions
        };

        // 8. Normalize text for alignment
        let normalized = normalize_for_alignment(text);
        if normalized.is_empty() {
            return Ok(vec![]);
        }

        // 9. CTC targets -> Viterbi -> word timestamps
        let targets = text_to_ctc_targets(&normalized, &self.vocab, self.blank_id);
        let path = viterbi_forced_align(&emissions, &targets)?;
        let words = path_to_word_timestamps(
            &path,
            &normalized,
            &self.vocab,
            self.blank_id,
            FRAME_DURATION_SEC,
        );

        Ok(words)
    }
}

/// Normalize text for CTC alignment: lowercase, keep only a-z/space/apostrophe,
/// collapse whitespace.
fn normalize_for_alignment(text: &str) -> String {
    let filtered: String = text
        .chars()
        .filter_map(|c| {
            if c.is_ascii_alphabetic() {
                Some(c.to_ascii_lowercase())
            } else if c == '\'' {
                Some('\'')
            } else if c.is_whitespace() || c == '-' {
                Some(' ')
            } else {
                None
            }
        })
        .collect();

    // Collapse multiple spaces into one, trim
    filtered.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_for_alignment() {
        assert_eq!(normalize_for_alignment("Hello, World!"), "hello world");
        assert_eq!(normalize_for_alignment("I'm fine."), "i'm fine");
        assert_eq!(
            normalize_for_alignment("  multiple   spaces  "),
            "multiple spaces"
        );
        assert_eq!(normalize_for_alignment("self-driving"), "self driving");
        assert_eq!(normalize_for_alignment("123!@#"), "");
        assert_eq!(normalize_for_alignment(""), "");
        assert_eq!(
            normalize_for_alignment("UPPER lower MiXeD"),
            "upper lower mixed"
        );
    }
}
