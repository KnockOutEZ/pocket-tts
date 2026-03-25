//! Whisper-based speech aligner for word-level timestamps.
//!
//! Shells out to the `whisper` CLI for transcription with word timestamps.
//! Requires `whisper` (OpenAI) installed: `pip install openai-whisper`

use crate::alignment::forced_align::WordTimestamp;
use crate::audio::resample;
use candle_core::Tensor;
use std::process::Command;

const TTS_SAMPLE_RATE: u32 = 24000;
const WHISPER_SAMPLE_RATE: u32 = 16000;

/// Speech aligner using Whisper CLI for word-level timestamps.
#[derive(Clone)]
pub struct WhisperAligner {
    model: String,
}

impl WhisperAligner {
    /// Create aligner with specified Whisper model size.
    pub fn new(model: &str) -> Self {
        Self { model: model.to_string() }
    }

    /// Create aligner with default model (base.en — fast + accurate on English).
    pub fn load() -> anyhow::Result<Self> {
        // Verify whisper is installed
        let check = Command::new("whisper").arg("--help").output();
        if check.is_err() {
            anyhow::bail!(
                "Whisper CLI not found. Install with: pip install openai-whisper"
            );
        }
        Ok(Self::new("base.en"))
    }

    /// Align audio to produce word-level timestamps.
    ///
    /// `audio`: Tensor [C, T] at 24kHz (TTS output)
    /// `_text`: Original text (unused — Whisper transcribes freely)
    pub fn align(&self, audio: &Tensor, _text: &str) -> anyhow::Result<Vec<WordTimestamp>> {
        // 1. Prepare audio: mono 16kHz WAV in temp file
        let audio = match audio.dims().len() {
            1 => audio.unsqueeze(0)?,
            2 if audio.dims()[0] == 1 => audio.clone(),
            2 => audio.mean(0)?.unsqueeze(0)?,
            _ => anyhow::bail!("Unexpected audio shape: {:?}", audio.dims()),
        };

        let audio_16k = resample(&audio, TTS_SAMPLE_RATE, WHISPER_SAMPLE_RATE)?;
        let samples = audio_16k.flatten_all()?.to_vec1::<f32>()?;

        // Write temp WAV
        let tmp_dir = std::env::temp_dir();
        let wav_path = tmp_dir.join("pocket_tts_align.wav");
        let json_dir = tmp_dir.join("pocket_tts_whisper");
        std::fs::create_dir_all(&json_dir)?;

        {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: WHISPER_SAMPLE_RATE,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(&wav_path, spec)?;
            for &s in &samples {
                writer.write_sample((s * 32767.0).clamp(-32768.0, 32767.0) as i16)?;
            }
            writer.finalize()?;
        }

        // 2. Run Whisper CLI
        let output = Command::new("whisper")
            .arg(wav_path.to_str().unwrap())
            .arg("--model").arg(&self.model)
            .arg("--language").arg("en")
            .arg("--word_timestamps").arg("True")
            .arg("--output_format").arg("json")
            .arg("--output_dir").arg(json_dir.to_str().unwrap())
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Whisper failed: {}", stderr);
        }

        // 3. Parse JSON output
        let json_path = json_dir.join("pocket_tts_align.json");
        let json_str = std::fs::read_to_string(&json_path)?;
        let json: serde_json::Value = serde_json::from_str(&json_str)?;

        let mut timestamps = Vec::new();

        if let Some(segments) = json.get("segments").and_then(|s| s.as_array()) {
            for seg in segments {
                if let Some(words) = seg.get("words").and_then(|w| w.as_array()) {
                    for w in words {
                        let word = w.get("word")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .trim()
                            .to_string();
                        let start = w.get("start")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0) as f32;
                        let end = w.get("end")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0) as f32;

                        if !word.is_empty() {
                            timestamps.push(WordTimestamp { word, start_sec: start, end_sec: end });
                        }
                    }
                }
            }
        }

        // Cleanup temp files
        let _ = std::fs::remove_file(&wav_path);
        let _ = std::fs::remove_dir_all(&json_dir);

        Ok(timestamps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    #[ignore] // requires whisper CLI installed
    fn test_whisper_aligner_loads() {
        let _aligner = WhisperAligner::load().unwrap();
    }
}
