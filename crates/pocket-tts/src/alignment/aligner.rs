//! WhisperX-based aligner for word-level timestamps.
//!
//! Uses WhisperX (Whisper + wav2vec2 forced alignment) via a Python sidecar
//! script for production-grade word-level timestamps (~20-50ms accuracy).
//!
//! For shipping: compile whisperx_align.py with PyInstaller to a standalone
//! binary — no Python needed on customer machines.

use crate::alignment::forced_align::WordTimestamp;
use crate::audio::resample;
use candle_core::{Device, Tensor};
use std::path::PathBuf;
use std::process::Command;

const TTS_SAMPLE_RATE: u32 = 24000;
const WHISPER_SAMPLE_RATE: u32 = 16000;

/// WhisperX-based aligner for production-grade word timestamps.
///
/// Uses Whisper for transcription + wav2vec2 CTC forced alignment for
/// phoneme-level word boundary precision. ~20-50ms accuracy.
#[derive(Clone)]
pub struct WhisperAligner {
    script_path: PathBuf,
}

impl WhisperAligner {
    /// Load the aligner. Finds the whisperx_align script/binary.
    pub fn load(_device: &Device) -> anyhow::Result<Self> {
        let script_path = Self::find_script()?;
        Ok(Self { script_path })
    }

    /// Find the alignment script/binary. Checks:
    /// 1. scripts/whisperx_align.py (dev)
    /// 2. Next to running executable as whisperx_align (PyInstaller binary)
    /// 3. In PATH
    fn find_script() -> anyhow::Result<PathBuf> {
        let candidates = [
            // Dev: script in workspace
            PathBuf::from("scripts/whisperx_align.py"),
            // Bundled PyInstaller binary next to exe
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.join("whisperx_align")))
                .unwrap_or_default(),
            // In crate bin/
            PathBuf::from("crates/pocket-tts/bin/whisperx_align"),
            PathBuf::from("bin/whisperx_align"),
        ];

        for path in &candidates {
            if path.exists() {
                return Ok(path.clone());
            }
        }

        // Check PATH
        if let Ok(output) = Command::new("which").arg("whisperx_align").output() {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !path.is_empty() {
                    return Ok(PathBuf::from(path));
                }
            }
        }

        anyhow::bail!(
            "WhisperX aligner not found. Expected scripts/whisperx_align.py \
             or a compiled whisperx_align binary in PATH."
        )
    }

    /// Align audio to produce word-level timestamps.
    ///
    /// `audio`: Tensor `[C, T]` at 24kHz (TTS output).
    /// `text`: Known text — passed to WhisperX for guided alignment.
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

        // 2. Write temp WAV
        let tmp_dir = std::env::temp_dir();
        let wav_path = tmp_dir.join("pocket_tts_align.wav");
        let json_path = tmp_dir.join("pocket_tts_align.json");
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

        // 3. Run WhisperX alignment
        let mut cmd = if self.script_path.extension().map_or(false, |e| e == "py") {
            let mut c = Command::new("python3");
            c.arg(&self.script_path);
            c
        } else {
            Command::new(&self.script_path)
        };

        cmd.arg(&wav_path).arg(&json_path);
        if !text.is_empty() {
            cmd.arg("--text").arg(text);
        }

        let output = cmd.output()?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("WhisperX alignment failed: {}", stderr);
        }

        // 4. Parse JSON output
        let json_str = std::fs::read_to_string(&json_path)?;
        let words: Vec<serde_json::Value> = serde_json::from_str(&json_str)?;

        let timestamps: Vec<WordTimestamp> = words
            .iter()
            .filter_map(|w| {
                let word = w.get("word")?.as_str()?.to_string();
                let start = w.get("start")?.as_f64()? as f32;
                let end = w.get("end")?.as_f64()? as f32;
                Some(WordTimestamp {
                    word,
                    start_sec: start,
                    end_sec: end,
                })
            })
            .collect();

        // 5. Cleanup
        let _ = std::fs::remove_file(&wav_path);
        let _ = std::fs::remove_file(&json_path);

        Ok(timestamps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_whisperx_aligner_loads() {
        let device = Device::Cpu;
        let _aligner = WhisperAligner::load(&device).unwrap();
    }
}
