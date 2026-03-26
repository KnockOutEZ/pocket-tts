//! Whisper-based aligner for word-level timestamps.
//!
//! Uses a bundled whisper-cli binary (whisper.cpp) for accurate word-level
//! timestamps via DTW on cross-attention weights.
//!
//! No Python required. The whisper-cli binary is a self-contained 2.3MB
//! native executable. The ggml model (~140MB) downloads on first use.

use crate::alignment::forced_align::WordTimestamp;
use crate::audio::resample;
use candle_core::{Device, Tensor};
use std::path::PathBuf;
use std::process::Command;

const TTS_SAMPLE_RATE: u32 = 24000;
const WHISPER_SAMPLE_RATE: u32 = 16000;
const WHISPER_MODEL_REPO: &str = "ggerganov/whisper.cpp";
const WHISPER_MODEL_FILE: &str = "ggml-base.en.bin";

/// Whisper-based aligner using bundled whisper-cli (whisper.cpp).
///
/// Produces word-level timestamps with ~20-50ms accuracy by running
/// the Whisper model's DTW on cross-attention weights — same algorithm
/// as OpenAI's Whisper `--word_timestamps True`.
#[derive(Clone)]
pub struct WhisperAligner {
    whisper_cli: PathBuf,
    model_path: PathBuf,
}

impl WhisperAligner {
    /// Load the aligner. Finds the whisper-cli binary and downloads the model.
    ///
    /// The `_device` parameter is accepted for API compatibility with TTSModel
    /// but is unused — whisper-cli manages its own compute.
    pub fn load(_device: &Device) -> anyhow::Result<Self> {
        let whisper_cli = Self::find_whisper_cli()?;
        let model_path = Self::ensure_model()?;

        // Verify the binary runs
        let test = Command::new(&whisper_cli).arg("--help").output();
        if test.is_err() {
            anyhow::bail!(
                "whisper-cli at {:?} failed to execute. \
                 Rebuild with: cd whisper.cpp && cmake -B build \
                 -DCMAKE_CROSSCOMPILING=TRUE && cmake --build build",
                whisper_cli
            );
        }

        Ok(Self {
            whisper_cli,
            model_path,
        })
    }

    /// Find whisper-cli binary. Checks:
    /// 1. Bundled in crate's bin/ directory
    /// 2. Next to the running executable
    /// 3. In PATH
    fn find_whisper_cli() -> anyhow::Result<PathBuf> {
        let candidates = [
            // Bundled (workspace root)
            PathBuf::from("crates/pocket-tts/bin/whisper-cli"),
            // Bundled (crate root)
            PathBuf::from("bin/whisper-cli"),
            // Next to the running binary (Tauri sidecar)
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.join("whisper-cli")))
                .unwrap_or_default(),
        ];

        for path in &candidates {
            if path.exists() {
                return Ok(path.clone());
            }
        }

        // Check PATH
        if let Ok(output) = Command::new("which").arg("whisper-cli").output() {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !path.is_empty() {
                    return Ok(PathBuf::from(path));
                }
            }
        }

        anyhow::bail!(
            "whisper-cli not found. Place it in bin/ or add to PATH.\n\
             Build: git clone https://github.com/ggerganov/whisper.cpp && \
             cd whisper.cpp && cmake -B build -DCMAKE_CROSSCOMPILING=TRUE && \
             cmake --build build --config Release"
        )
    }

    fn ensure_model() -> anyhow::Result<PathBuf> {
        crate::weights::download_if_necessary(&format!(
            "hf://{}/{}",
            WHISPER_MODEL_REPO, WHISPER_MODEL_FILE
        ))
    }

    /// Align audio to produce word-level timestamps.
    ///
    /// `audio`: Tensor `[C, T]` at 24kHz (TTS output).
    /// `text`: The known text — passed as `--prompt` to guide Whisper's
    ///         transcription for better word recognition accuracy.
    pub fn align(&self, audio: &Tensor, text: &str) -> anyhow::Result<Vec<WordTimestamp>> {
        // 1. Mono, resample 24kHz → 16kHz
        let audio = match audio.dims().len() {
            1 => audio.unsqueeze(0)?,
            2 if audio.dims()[0] == 1 => audio.clone(),
            2 => audio.mean(0)?.unsqueeze(0)?,
            _ => anyhow::bail!("Unexpected audio shape: {:?}", audio.dims()),
        };
        let audio_16k = resample(&audio, TTS_SAMPLE_RATE, WHISPER_SAMPLE_RATE)?;
        let samples = audio_16k.flatten_all()?.to_vec1::<f32>()?;

        // 2. Write temp WAV (whisper-cli reads from file)
        let tmp_dir = std::env::temp_dir();
        let wav_path = tmp_dir.join("pocket_tts_align.wav");
        let json_stem = tmp_dir.join("pocket_tts_align");
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

        // 3. Run whisper-cli with --prompt for guided transcription
        let mut cmd = Command::new(&self.whisper_cli);
        cmd.arg("-m").arg(&self.model_path)
            .arg("-f").arg(&wav_path)
            .arg("--output-json-full")
            .arg("-of").arg(&json_stem)
            .arg("-l").arg("en");

        // Use known text as prompt — improves word recognition accuracy
        if !text.is_empty() {
            cmd.arg("--prompt").arg(text);
        }

        let output = cmd.output()?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("whisper-cli failed: {}", stderr);
        }

        // 4. Parse per-token timestamps from JSON, group into words
        let json_file = json_stem.with_extension("json");
        let timestamps = Self::parse_word_timestamps(&json_file)?;

        // 5. Cleanup
        let _ = std::fs::remove_file(&wav_path);
        let _ = std::fs::remove_file(&json_file);

        Ok(timestamps)
    }

    /// Parse whisper.cpp's JSON output into word timestamps.
    /// Groups BPE tokens into words using space-prefix convention.
    fn parse_word_timestamps(json_path: &PathBuf) -> anyhow::Result<Vec<WordTimestamp>> {
        let json_str = std::fs::read_to_string(json_path)?;
        let json: serde_json::Value = serde_json::from_str(&json_str)?;

        let mut timestamps = Vec::new();

        let transcription = json
            .get("transcription")
            .and_then(|t| t.as_array())
            .map(|a| a.as_slice())
            .unwrap_or(&[]);

        for segment in transcription {
            let tokens = segment
                .get("tokens")
                .and_then(|t| t.as_array())
                .map(|a| a.as_slice())
                .unwrap_or(&[]);

            let mut word = String::new();
            let mut word_start: Option<f32> = None;
            let mut word_end: f32 = 0.0;

            for token in tokens {
                let text = token.get("text").and_then(|t| t.as_str()).unwrap_or("");
                if text.is_empty() || text.starts_with('[') {
                    continue;
                }

                let offsets = token.get("offsets");
                let t0 = offsets
                    .and_then(|o| o.get("from"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32
                    / 1000.0;
                let t1 = offsets
                    .and_then(|o| o.get("to"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32
                    / 1000.0;

                // Space prefix = new word in Whisper's BPE
                if text.starts_with(' ') && !word.is_empty() {
                    if let Some(start) = word_start {
                        timestamps.push(WordTimestamp {
                            word: word.clone(),
                            start_sec: start,
                            end_sec: word_end,
                        });
                    }
                    word.clear();
                    word_start = None;
                }

                let clean = text.trim();
                if !clean.is_empty() {
                    if word.is_empty() {
                        word = clean.to_string();
                        word_start = Some(t0);
                    } else {
                        word.push_str(clean);
                    }
                    word_end = t1;
                }
            }

            // Emit final word in segment
            if !word.is_empty() {
                if let Some(start) = word_start {
                    timestamps.push(WordTimestamp {
                        word,
                        start_sec: start,
                        end_sec: word_end,
                    });
                }
            }
        }

        Ok(timestamps)
    }
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
