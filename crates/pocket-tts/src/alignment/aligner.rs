//! Native Whisper aligner for word-level timestamps.
//!
//! Uses candle-transformers' Whisper model (already a dependency) with a greedy
//! decoding loop. Parses timestamp tokens to extract segment-level timing, then
//! maps segments to words.
//!
//! No Python, no C++ FFI, no external processes. Pure Rust.

use crate::alignment::forced_align::WordTimestamp;
use crate::audio::resample;
use candle_core::{Device, IndexOp, Tensor, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as w, Config};
use std::sync::Arc;
use tokenizers::Tokenizer;

const TTS_SAMPLE_RATE: u32 = 24000;
const WHISPER_SAMPLE_RATE: u32 = 16000;
const WHISPER_REPO: &str = "openai/whisper-base.en";

// Mel filter bank (80 bins x 201 freq bins), extracted from OpenAI Whisper.
const MEL_FILTERS: &[u8] = include_bytes!("melfilters.bytes");

/// Native Whisper-based aligner. Pure Rust via candle-transformers.
#[derive(Clone)]
pub struct WhisperAligner {
    model: Arc<std::sync::Mutex<w::model::Whisper>>,
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
    /// Load whisper-tiny.en model (~75MB). Downloads from HuggingFace on first use.
    pub fn load(device: &Device) -> anyhow::Result<Self> {
        // Download model files
        let weights_path = crate::weights::download_if_necessary(
            &format!("hf://{}/model.safetensors", WHISPER_REPO)
        )?;
        let config_path = crate::weights::download_if_necessary(
            &format!("hf://{}/config.json", WHISPER_REPO)
        )?;
        let tokenizer_path = crate::weights::download_if_necessary(
            &format!("hf://{}/tokenizer.json", WHISPER_REPO)
        )?;

        // Load config
        let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;

        // Load model
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)?
        };
        let model = w::model::Whisper::load(&vb, config.clone())?;

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

    /// Align audio, producing word-level timestamps.
    /// `audio`: Tensor [C, T] at 24kHz
    pub fn align(&self, audio: &Tensor, _text: &str) -> anyhow::Result<Vec<WordTimestamp>> {
        // 1. Prepare audio: mono, 16kHz, f32 samples
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

        // 3. Encode audio
        let mut model = self.model.lock().unwrap();
        model.reset_kv_cache();

        let audio_features = model.encoder.forward(&mel_tensor, true)?;

        // 4. Greedy decode with timestamps
        // For .en models: [SOT, <|0.00|>]
        // Seeding <|0.00|> forces the model into timestamp mode
        let mut tokens: Vec<u32> = vec![
            self.sot_token,
            self.no_timestamps_token + 1, // <|0.00|>
        ];

        let max_tokens = self.config.max_target_positions / 2;
        let timestamp_begin = self.no_timestamps_token + 1;

        // Track text tokens since last timestamp to force shorter segments.
        // After MAX_TEXT_TOKENS_PER_SEGMENT text tokens, bias heavily toward
        // timestamp tokens so each segment is ~1-2 words.
        const MAX_TEXT_TOKENS_PER_SEGMENT: usize = 2; // ~1 word
        const TIMESTAMP_BIAS: f32 = 8.0; // very strong bias toward emitting timestamp
        let mut text_tokens_since_timestamp: usize = 0;

        for i in 0..max_tokens {
            let tokens_tensor = Tensor::new(tokens.as_slice(), &self.device)?
                .unsqueeze(0)?;

            let ys = model.decoder.forward(&tokens_tensor, &audio_features, i == 0)?;
            let (_, seq_len, _) = ys.dims3()?;
            let logits = model.decoder.final_linear(&ys.i((.., seq_len - 1.., ..))?)?
                .squeeze(0)?
                .squeeze(0)?;

            // Apply suppress mask
            let logits = logits.broadcast_add(&self.suppress_tokens)?;
            let mut logits_vec: Vec<f32> = logits.to_vec1()?;

            // Bias toward timestamp tokens when segment is getting long
            if text_tokens_since_timestamp >= MAX_TEXT_TOKENS_PER_SEGMENT {
                for tok_id in (timestamp_begin as usize)..logits_vec.len() {
                    logits_vec[tok_id] += TIMESTAMP_BIAS;
                }
            }

            // Greedy: argmax
            let next_token = logits_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i as u32)
                .unwrap_or(self.eot_token);

            tokens.push(next_token);

            if next_token >= timestamp_begin {
                text_tokens_since_timestamp = 0;
            } else if next_token != self.eot_token {
                text_tokens_since_timestamp += 1;
            }

            if next_token == self.eot_token {
                break;
            }
        }

        // 5. Parse tokens into word timestamps
        let timestamps = self.parse_timestamps(&tokens, timestamp_begin)?;

        Ok(timestamps)
    }

    /// Parse decoded tokens into word timestamps using timestamp token pairs.
    ///
    /// Words are identified by BPE token boundaries (Whisper tokens start with
    /// a space character for word beginnings). Duration within a segment is
    /// distributed proportionally by token count per word, which tracks
    /// pronunciation duration better than character count.
    fn parse_timestamps(
        &self,
        tokens: &[u32],
        timestamp_begin: u32,
    ) -> anyhow::Result<Vec<WordTimestamp>> {
        let mut result = Vec::new();
        let mut current_start: Option<f32> = Some(0.0);
        // Accumulate (decoded_text, token_count) per word in current segment
        let mut current_words: Vec<(String, usize)> = Vec::new();

        // Skip [SOT, <|0.00|>]
        for &tok in tokens.iter().skip(2) {
            if tok == self.eot_token {
                break;
            }

            if tok >= timestamp_begin {
                let time_sec = (tok - timestamp_begin) as f32 * 0.02;

                if current_start.is_none() {
                    current_start = Some(time_sec);
                } else {
                    // Closing timestamp — emit words
                    let start = current_start.unwrap();
                    let end = time_sec;

                    if !current_words.is_empty() {
                        let total_tokens: usize =
                            current_words.iter().map(|(_, count)| count).sum();
                        let duration = end - start;
                        let mut offset = start;

                        for (word, token_count) in &current_words {
                            let frac = *token_count as f32 / total_tokens as f32;
                            let word_dur = duration * frac;
                            result.push(WordTimestamp {
                                word: word.clone(),
                                start_sec: offset,
                                end_sec: offset + word_dur,
                            });
                            offset += word_dur;
                        }
                    }

                    current_words.clear();
                    current_start = Some(time_sec);
                }
            } else {
                // Text token — decode and group into words.
                // Whisper BPE tokens that start a new word begin with a space.
                if let Ok(text) = self.tokenizer.decode(&[tok], false) {
                    let starts_new_word = text.starts_with(' ');
                    let clean = text.trim().to_string();
                    if clean.is_empty() {
                        continue;
                    }

                    if starts_new_word || current_words.is_empty() {
                        // New word
                        current_words.push((clean, 1));
                    } else {
                        // Continue current word (subword token)
                        if let Some(last) = current_words.last_mut() {
                            last.0.push_str(&clean);
                            last.1 += 1;
                        }
                    }
                }
            }
        }

        Ok(result)
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
