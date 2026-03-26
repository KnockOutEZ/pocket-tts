//! Native Whisper aligner for word-level timestamps.
//!
//! Uses candle-transformers' Whisper encoder + our custom autoregressive decoder
//! that captures cross-attention weights during token-by-token generation.
//! DTW on those weights gives true per-word alignment.
//!
//! No Python, no C++ FFI, no external processes. Pure Rust.

use crate::alignment::ar_decoder::{ARDecoder, ARDecoderConfig};
use crate::alignment::dtw_decoder::{dtw_alignment, extract_word_timestamps};
use crate::alignment::forced_align::WordTimestamp;
use crate::audio::resample;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as w, Config};
use std::sync::Arc;
use tokenizers::Tokenizer;

const TTS_SAMPLE_RATE: u32 = 24000;
const WHISPER_SAMPLE_RATE: u32 = 16000;
const WHISPER_REPO: &str = "openai/whisper-base.en";

const MEL_FILTERS: &[u8] = include_bytes!("melfilters.bytes");

/// Native Whisper-based aligner with autoregressive cross-attention capture.
#[derive(Clone)]
pub struct WhisperAligner {
    encoder: Arc<std::sync::Mutex<w::model::Whisper>>,
    decoder: Arc<std::sync::Mutex<ARDecoder>>,
    tokenizer: Arc<Tokenizer>,
    config: Config,
    device: Device,
    mel_filters: Vec<f32>,
    sot_token: u32,
    eot_token: u32,
    no_timestamps_token: u32,
    suppress_tokens: Vec<f32>,
}

impl WhisperAligner {
    /// Load whisper-base.en (~140MB). Downloads from HuggingFace on first use.
    pub fn load(device: &Device) -> anyhow::Result<Self> {
        let weights_path = crate::weights::download_if_necessary(
            &format!("hf://{}/model.safetensors", WHISPER_REPO),
        )?;
        let config_path = crate::weights::download_if_necessary(
            &format!("hf://{}/config.json", WHISPER_REPO),
        )?;
        let tokenizer_path = crate::weights::download_if_necessary(
            &format!("hf://{}/tokenizer.json", WHISPER_REPO),
        )?;

        let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)?
        };

        // Load candle-transformers encoder (we only use the encoder from it)
        let encoder_model = w::model::Whisper::load(&vb, config.clone())?;

        // Load our custom autoregressive decoder with attention capture
        let ar_config = ARDecoderConfig {
            d_model: config.d_model,
            n_head: config.decoder_attention_heads,
            n_layer: config.decoder_layers,
            n_vocab: config.vocab_size,
            n_ctx: config.max_target_positions,
        };
        let decoder = ARDecoder::load(vb.pp("model.decoder"), &ar_config)?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let sot_token = token_id(&tokenizer, w::SOT_TOKEN)?;
        let eot_token = token_id(&tokenizer, w::EOT_TOKEN)?;
        let no_timestamps_token = token_id(&tokenizer, w::NO_TIMESTAMPS_TOKEN)?;

        // Build suppress mask as Vec<f32> (applied manually to logits)
        let mut suppress_tokens = vec![0f32; config.vocab_size];
        for &t in &config.suppress_tokens {
            if (t as usize) < config.vocab_size {
                suppress_tokens[t as usize] = f32::NEG_INFINITY;
            }
        }

        let mut mel_filters = vec![0f32; MEL_FILTERS.len() / 4];
        byteorder::LittleEndian::read_f32_into(MEL_FILTERS, &mut mel_filters);

        Ok(Self {
            encoder: Arc::new(std::sync::Mutex::new(encoder_model)),
            decoder: Arc::new(std::sync::Mutex::new(decoder)),
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

        // 2. Mel spectrogram
        let mel = w::audio::pcm_to_mel(&self.config, &pcm, &self.mel_filters);
        let mel_len = mel.len() / self.config.num_mel_bins;
        let mel_tensor = Tensor::from_vec(
            mel,
            (1, self.config.num_mel_bins, mel_len),
            &self.device,
        )?;

        // 3. Encode audio using candle-transformers encoder
        let mut encoder = self.encoder.lock().unwrap();
        encoder.reset_kv_cache();
        let audio_features = encoder.encoder.forward(&mel_tensor, true)?;
        drop(encoder);

        // 4. Autoregressive decode with our custom decoder (captures attention)
        let mut decoder = self.decoder.lock().unwrap();
        decoder.reset_cache();

        // Seed with SOT token
        let mut tokens: Vec<u32> = vec![self.sot_token];
        let max_tokens = self.config.max_target_positions / 2;

        // First step: SOT
        let _ = decoder.forward_one(self.sot_token, &audio_features, 0)?;

        for step in 1..max_tokens {
            let logits = decoder.forward_one(
                *tokens.last().unwrap(),
                &audio_features,
                step,
            )?;

            // Apply suppress mask
            let mut logits_vec: Vec<f32> = logits.to_vec1()?;
            for (i, mask) in self.suppress_tokens.iter().enumerate() {
                logits_vec[i] += mask;
            }

            // Greedy argmax
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

        // 5. Get cross-attention matrix [n_tokens, n_audio_frames]
        let attn_matrix = decoder.get_cross_attention_matrix()?;
        drop(decoder);

        // 6. Trim to actual audio frames (encoder pads to 1500 frames for 30s)
        let raw_mel_frames = (pcm.len() + 159) / 160;
        let actual_frames = (raw_mel_frames + 1) / 2; // encoder 2x downsampling
        let n_audio = attn_matrix.dim(1)?;
        let trim = actual_frames.min(n_audio);
        let attn_matrix = if trim < n_audio {
            attn_matrix.narrow(1, 0, trim)?
        } else {
            attn_matrix
        };

        // 7. Filter to text-only tokens (skip SOT, timestamps, EOT)
        let text_token_indices: Vec<usize> = tokens
            .iter()
            .enumerate()
            .filter(|&(_, &t)| t < self.sot_token && t != self.eot_token)
            .map(|(i, _)| i)
            .collect();

        if text_token_indices.is_empty() {
            return Ok(Vec::new());
        }

        let text_tokens: Vec<u32> = text_token_indices
            .iter()
            .map(|&i| tokens[i])
            .collect();

        // Extract attention rows for text tokens only
        // attn_matrix is [n_all_tokens, n_audio], we want [n_text_tokens, n_audio]
        let text_attn_rows: Vec<Tensor> = text_token_indices
            .iter()
            .filter_map(|&i| {
                // Token at position i corresponds to attention captured at step i
                // (step 0 = SOT, step 1 = first generated token, etc.)
                if i < attn_matrix.dim(0).unwrap_or(0) {
                    attn_matrix.get(i).ok()
                } else {
                    None
                }
            })
            .collect();

        if text_attn_rows.is_empty() {
            return Ok(Vec::new());
        }

        let text_attn = Tensor::stack(&text_attn_rows, 0)?;
        let attn_2d: Vec<Vec<f32>> = text_attn.to_vec2()?;

        // 8. DTW alignment
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_whisper_aligner_loads() {
        let device = Device::Cpu;
        let _aligner = WhisperAligner::load(&device).unwrap();
    }
}
