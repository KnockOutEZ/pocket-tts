//! Native Whisper aligner for word-level timestamps.
//!
//! Combines timestamp tokens (for segment boundaries / pause preservation) with
//! DTW on cross-attention weights (for per-word timing within segments). This
//! matches the approach used by OpenAI's Whisper `--word_timestamps True`.
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
const HOP_LENGTH: usize = 160;
const ENCODER_DOWNSAMPLE: usize = 2;

const MEL_FILTERS: &[u8] = include_bytes!("melfilters.bytes");

/// A decoded segment: timestamp-bounded group of text tokens.
struct Segment {
    start_sec: f32,
    end_sec: f32,
    /// Indices into the full token list (positions of text tokens in this segment)
    token_positions: Vec<usize>,
    /// The text token IDs themselves
    token_ids: Vec<u32>,
}

/// Native Whisper-based aligner. Segment timestamps + per-segment DTW.
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

        let encoder_model = w::model::Whisper::load(&vb, config.clone())?;

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

    pub fn align(&self, audio: &Tensor, _text: &str) -> anyhow::Result<Vec<WordTimestamp>> {
        // 1. Prepare audio
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
            mel, (1, self.config.num_mel_bins, mel_len), &self.device,
        )?;

        // 3. Encode
        let mut encoder = self.encoder.lock().unwrap();
        encoder.reset_kv_cache();
        let audio_features = encoder.encoder.forward(&mel_tensor, true)?;
        drop(encoder);

        // 4. Autoregressive decode (captures cross-attention at each step)
        let mut decoder = self.decoder.lock().unwrap();
        decoder.reset_cache();

        let timestamp_begin = self.no_timestamps_token + 1;
        let first_timestamp = timestamp_begin; // <|0.00|>
        let mut tokens: Vec<u32> = vec![self.sot_token, first_timestamp];

        // Seed SOT + <|0.00|>
        let _ = decoder.forward_one(self.sot_token, &audio_features, 0)?;
        let _ = decoder.forward_one(first_timestamp, &audio_features, 1)?;

        let max_tokens = self.config.max_target_positions / 2;
        // Force shorter segments: bias toward timestamp tokens after a few text tokens
        const MAX_TEXT_PER_SEG: usize = 3;
        const TS_BIAS: f32 = 6.0;
        let mut text_since_ts: usize = 0;

        for step in 2..max_tokens {
            let logits = decoder.forward_one(
                *tokens.last().unwrap(), &audio_features, step,
            )?;

            let mut logits_vec: Vec<f32> = logits.to_vec1()?;
            for (i, mask) in self.suppress_tokens.iter().enumerate() {
                logits_vec[i] += mask;
            }

            // Bias toward timestamp tokens when segment is getting long
            if text_since_ts >= MAX_TEXT_PER_SEG {
                for tok_id in (timestamp_begin as usize)..logits_vec.len() {
                    logits_vec[tok_id] += TS_BIAS;
                }
            }

            let next_token = logits_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i as u32)
                .unwrap_or(self.eot_token);

            tokens.push(next_token);

            if next_token >= timestamp_begin {
                text_since_ts = 0;
            } else if next_token != self.eot_token {
                text_since_ts += 1;
            }

            if next_token == self.eot_token {
                break;
            }
        }

        // 5. Get full attention matrix [n_steps, n_audio_frames]
        let attn_matrix = decoder.get_cross_attention_matrix()?;
        drop(decoder);

        // Trim to actual audio frames
        let raw_mel_frames = (pcm.len() + HOP_LENGTH - 1) / HOP_LENGTH;
        let actual_frames = (raw_mel_frames + ENCODER_DOWNSAMPLE - 1) / ENCODER_DOWNSAMPLE;
        let n_audio = attn_matrix.dim(1)?;
        let trim = actual_frames.min(n_audio);
        let attn_matrix = if trim < n_audio {
            attn_matrix.narrow(1, 0, trim)?
        } else {
            attn_matrix
        };

        // 6. Parse tokens into segments using timestamp tokens
        let segments = self.parse_segments(&tokens, timestamp_begin);

        // 7. For each segment, run DTW on its portion of the attention matrix
        let mut all_timestamps = Vec::new();

        for seg in &segments {
            if seg.token_ids.is_empty() {
                continue;
            }

            // Convert segment time to encoder frames
            let start_frame = sec_to_frame(seg.start_sec);
            let end_frame = sec_to_frame(seg.end_sec).min(trim);

            if end_frame <= start_frame {
                continue;
            }

            // Extract attention sub-matrix for this segment:
            // rows = text token positions, cols = segment's audio frame range
            let mut seg_attn_rows: Vec<Vec<f32>> = Vec::new();
            for &pos in &seg.token_positions {
                if pos < attn_matrix.dim(0)? {
                    let row = attn_matrix.get(pos)?;
                    // Narrow to segment's frame range
                    let seg_row = row.narrow(0, start_frame, end_frame - start_frame)?;
                    seg_attn_rows.push(seg_row.to_vec1()?);
                }
            }

            if seg_attn_rows.is_empty() {
                continue;
            }

            // DTW within this segment
            let token_alignments = dtw_alignment(&seg_attn_rows);

            // Convert frame offsets to absolute timestamps
            let seg_timestamps = extract_word_timestamps(
                &token_alignments, &seg.token_ids, &self.tokenizer,
            );

            // Offset timestamps by segment start + frame offset
            for mut ts in seg_timestamps {
                ts.start_sec += seg.start_sec;
                ts.end_sec += seg.start_sec;
                all_timestamps.push(ts);
            }
        }

        Ok(all_timestamps)
    }

    /// Parse decoded token sequence into segments bounded by timestamp tokens.
    fn parse_segments(&self, tokens: &[u32], timestamp_begin: u32) -> Vec<Segment> {
        let mut segments = Vec::new();
        let mut current_start: Option<f32> = None;
        let mut current_positions: Vec<usize> = Vec::new();
        let mut current_ids: Vec<u32> = Vec::new();

        for (pos, &tok) in tokens.iter().enumerate() {
            if tok == self.sot_token || tok == self.eot_token {
                continue;
            }

            if tok >= timestamp_begin {
                let time_sec = (tok - timestamp_begin) as f32 * 0.02;

                if let Some(start) = current_start {
                    if !current_ids.is_empty() {
                        segments.push(Segment {
                            start_sec: start,
                            end_sec: time_sec,
                            token_positions: std::mem::take(&mut current_positions),
                            token_ids: std::mem::take(&mut current_ids),
                        });
                    } else {
                        current_positions.clear();
                        current_ids.clear();
                    }
                }
                current_start = Some(time_sec);
            } else {
                // Text token
                current_positions.push(pos);
                current_ids.push(tok);
            }
        }

        segments
    }
}

/// Convert seconds to encoder frame index.
fn sec_to_frame(sec: f32) -> usize {
    // Each encoder frame = HOP_LENGTH * ENCODER_DOWNSAMPLE samples at 16kHz
    // = 320 samples = 20ms
    let samples = (sec * WHISPER_SAMPLE_RATE as f32) as usize;
    samples / (HOP_LENGTH * ENCODER_DOWNSAMPLE)
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
