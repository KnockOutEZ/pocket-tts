//! Native Whisper aligner for word-level timestamps.
//!
//! Uses candle-transformers' Whisper encoder + a custom decoder that captures
//! cross-attention weights. DTW on those weights gives true per-token alignment,
//! which is then grouped into word-level timestamps.
//!
//! No Python, no C++ FFI, no external processes. Pure Rust.

use crate::alignment::dtw_decoder::{
    CrossAttentionDecoder, average_cross_attention_weights, dtw_alignment,
    extract_word_timestamps,
};
use crate::alignment::forced_align::WordTimestamp;
use crate::alignment::DecoderConfig;
use crate::audio::resample;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as w, Config};
use std::sync::Arc;
use tokenizers::Tokenizer;

const TTS_SAMPLE_RATE: u32 = 24000;
const WHISPER_SAMPLE_RATE: u32 = 16000;
const WHISPER_REPO: &str = "openai/whisper-tiny.en";

// Mel filter bank (80 bins x 201 freq bins), extracted from OpenAI Whisper.
const MEL_FILTERS: &[u8] = include_bytes!("melfilters.bytes");

/// Native Whisper-based aligner with DTW cross-attention for word-level timestamps.
#[derive(Clone)]
pub struct WhisperAligner {
    model: Arc<std::sync::Mutex<w::model::Whisper>>,
    dtw_decoder: Arc<CrossAttentionDecoder>,
    tokenizer: Arc<Tokenizer>,
    config: Config,
    device: Device,
    mel_filters: Vec<f32>,
    // Special token IDs
    sot_token: u32,
    eot_token: u32,
    no_timestamps_token: u32,
    suppress_tokens: Tensor,
}

impl WhisperAligner {
    /// Load whisper-tiny.en (~75MB). Downloads from HuggingFace on first use.
    pub fn load(device: &Device) -> anyhow::Result<Self> {
        // Download model files
        let weights_path = crate::weights::download_if_necessary(
            &format!("hf://{}/model.safetensors", WHISPER_REPO),
        )?;
        let config_path = crate::weights::download_if_necessary(
            &format!("hf://{}/config.json", WHISPER_REPO),
        )?;
        let tokenizer_path = crate::weights::download_if_necessary(
            &format!("hf://{}/tokenizer.json", WHISPER_REPO),
        )?;

        // Load config
        let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;

        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)?
        };

        // Load candle-transformers encoder + decoder (for greedy decoding)
        let model = w::model::Whisper::load(&vb, config.clone())?;

        // Load our custom decoder (for cross-attention capture)
        let decoder_config = DecoderConfig {
            d_model: config.d_model,
            decoder_attention_heads: config.decoder_attention_heads,
            decoder_layers: config.decoder_layers,
            vocab_size: config.vocab_size,
            max_target_positions: config.max_target_positions,
        };
        let dtw_decoder =
            CrossAttentionDecoder::load(vb.pp("model.decoder"), &decoder_config)?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Resolve special token IDs
        let sot_token = token_id(&tokenizer, w::SOT_TOKEN)?;
        let eot_token = token_id(&tokenizer, w::EOT_TOKEN)?;
        let no_timestamps_token = token_id(&tokenizer, w::NO_TIMESTAMPS_TOKEN)?;

        // Build suppress mask
        let suppress_tokens = build_suppress_mask(&config, device)?;

        // Load mel filters
        let mut mel_filters = vec![0f32; MEL_FILTERS.len() / 4];
        byteorder::LittleEndian::read_f32_into(MEL_FILTERS, &mut mel_filters);

        Ok(Self {
            model: Arc::new(std::sync::Mutex::new(model)),
            dtw_decoder: Arc::new(dtw_decoder),
            tokenizer: Arc::new(tokenizer),
            config,
            device: device.clone(),
            mel_filters,
            sot_token,
            eot_token,
            no_timestamps_token,
            suppress_tokens,
        })
    }

    /// Align audio, producing word-level timestamps via DTW on cross-attention.
    /// `audio`: Tensor [C, T] at 24kHz
    pub fn align(&self, audio: &Tensor, _text: &str) -> anyhow::Result<Vec<WordTimestamp>> {
        // 1. Prepare audio: mono, 16kHz
        let audio = match audio.dims().len() {
            1 => audio.unsqueeze(0)?,
            2 if audio.dims()[0] == 1 => audio.clone(),
            2 => audio.mean(0)?.unsqueeze(0)?,
            _ => anyhow::bail!("Unexpected audio shape: {:?}", audio.dims()),
        };
        let audio_16k = resample(&audio, TTS_SAMPLE_RATE, WHISPER_SAMPLE_RATE)?;
        let pcm: Vec<f32> = audio_16k.flatten_all()?.to_vec1()?;

        // 2. Compute mel spectrogram
        let mel = w::audio::pcm_to_mel(&self.config, &pcm, &self.mel_filters);
        let mel_len = mel.len() / self.config.num_mel_bins;
        let mel_tensor = Tensor::from_vec(
            mel,
            (1, self.config.num_mel_bins, mel_len),
            &self.device,
        )?;

        // 3. Encode audio (shared between greedy decode and DTW pass)
        let mut model = self.model.lock().unwrap();
        model.reset_kv_cache();
        let audio_features = model.encoder.forward(&mel_tensor, true)?;

        // 4. Greedy decode to get token sequence
        let mut tokens: Vec<u32> = vec![
            self.sot_token,
            self.no_timestamps_token + 1, // <|0.00|>
        ];
        let max_tokens = self.config.max_target_positions / 2;

        for i in 0..max_tokens {
            let tokens_tensor =
                Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let ys = model
                .decoder
                .forward(&tokens_tensor, &audio_features, i == 0)?;
            let (_, seq_len, _) = ys.dims3()?;
            let logits = model
                .decoder
                .final_linear(&ys.i((.., seq_len - 1.., ..))?)?
                .squeeze(0)?
                .squeeze(0)?;
            let logits = logits.broadcast_add(&self.suppress_tokens)?;

            let logits_vec: Vec<f32> = logits.to_vec1()?;
            let next_token = logits_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i as u32)
                .unwrap_or(self.eot_token);

            tokens.push(next_token);
            if next_token == self.eot_token {
                break;
            }
        }

        // Drop the model lock before the DTW pass
        drop(model);

        // 5. Filter to text-only tokens (no special/timestamp tokens)
        let timestamp_begin = self.no_timestamps_token + 1;
        let text_tokens: Vec<u32> = tokens
            .iter()
            .copied()
            .filter(|&t| t < self.sot_token && t != self.eot_token)
            .collect();

        if text_tokens.is_empty() {
            return Ok(Vec::new());
        }

        // 6. DTW pass: run custom decoder with text tokens to get cross-attention weights
        let text_tensor =
            Tensor::new(text_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let cross_attn_weights = self.dtw_decoder.forward(&text_tensor, &audio_features)?;

        // 7. Average attention across layers and heads → [n_text, n_audio]
        let avg_attn = average_cross_attention_weights(&cross_attn_weights, &self.device)?;

        // Trim attention to actual audio frames (pcm_to_mel pads to 30s = 1500 frames)
        // Actual frames = ceil(pcm_samples / HOP_LENGTH) after encoder's 2x downsampling
        let actual_audio_frames = (pcm.len() / 160 + 1) / 2; // HOP=160, encoder stride=2
        let avg_attn = if avg_attn.dim(1)? > actual_audio_frames && actual_audio_frames > 0 {
            avg_attn.narrow(1, 0, actual_audio_frames)?
        } else {
            avg_attn
        };
        let attn_2d: Vec<Vec<f32>> = avg_attn.to_vec2()?;

        // 8. DTW alignment → per-token frame ranges
        let token_alignments = dtw_alignment(&attn_2d);

        // 9. Convert to word timestamps
        let timestamps =
            extract_word_timestamps(&token_alignments, &text_tokens, &self.tokenizer);

        Ok(timestamps)
    }
}

use byteorder::ByteOrder;

fn token_id(tokenizer: &Tokenizer, token: &str) -> anyhow::Result<u32> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| anyhow::anyhow!("Token '{}' not found in vocabulary", token))
}

fn build_suppress_mask(config: &Config, device: &Device) -> anyhow::Result<Tensor> {
    let vocab_size = config.vocab_size;
    let mut mask = vec![0f32; vocab_size];
    for &t in &config.suppress_tokens {
        if (t as usize) < vocab_size {
            mask[t as usize] = f32::NEG_INFINITY;
        }
    }
    Ok(Tensor::new(mask, device)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // requires model download (~75MB)
    fn test_whisper_aligner_loads() {
        let device = Device::Cpu;
        let _aligner = WhisperAligner::load(&device).unwrap();
    }

    #[test]
    #[ignore] // requires model download
    fn test_whisper_aligner_on_silence() {
        let device = Device::Cpu;
        let aligner = WhisperAligner::load(&device).unwrap();
        let audio = Tensor::zeros((1, 24000), DType::F32, &device).unwrap();
        let ts = aligner.align(&audio, "hello").unwrap();
        println!("Silence timestamps: {:?}", ts);
    }
}
