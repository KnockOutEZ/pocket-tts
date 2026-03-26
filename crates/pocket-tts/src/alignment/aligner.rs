//! WhisperX-based aligner for word-level timestamps.
//!
//! Uses WhisperX (Whisper + wav2vec2 forced alignment) for production-grade
//! word-level timestamps (~20-50ms accuracy).
//!
//! Backends (auto-detected in order):
//! 1. Python script (dev: scripts/whisperx_align.py)
//! 2. Standalone binary (shipped: PyInstaller-compiled whisperx_align)
//! 3. Docker container (fallback: whisperx-align image)

use crate::alignment::forced_align::WordTimestamp;
use crate::audio::resample;
use candle_core::{Device, Tensor};
use std::path::PathBuf;
use std::process::Command;

const TTS_SAMPLE_RATE: u32 = 24000;
const WHISPER_SAMPLE_RATE: u32 = 16000;

#[derive(Clone, Debug)]
enum Backend {
    /// Python script (requires python3 + whisperx installed)
    PythonScript(PathBuf),
    /// Standalone binary (PyInstaller-compiled, no Python needed)
    Binary(PathBuf),
    /// Docker container (no local deps needed)
    Docker(String),
}

/// WhisperX-based aligner for production-grade word timestamps.
#[derive(Clone)]
pub struct WhisperAligner {
    backend: Backend,
}

impl WhisperAligner {
    /// Load the aligner. Auto-detects the best available backend.
    pub fn load(_device: &Device) -> anyhow::Result<Self> {
        let backend = Self::detect_backend()?;
        Ok(Self { backend })
    }

    /// Detect available backend in priority order.
    fn detect_backend() -> anyhow::Result<Backend> {
        // 1. Python script (dev mode)
        let py_candidates = [
            PathBuf::from("scripts/whisperx_align.py"),
            PathBuf::from("crates/pocket-tts/scripts/whisperx_align.py"),
        ];
        for path in &py_candidates {
            if path.exists() {
                // Verify python3 can import whisperx
                if Command::new("python3")
                    .args(["-c", "import whisperx"])
                    .output()
                    .map(|o| o.status.success())
                    .unwrap_or(false)
                {
                    return Ok(Backend::PythonScript(path.clone()));
                }
            }
        }

        // 2. Standalone binary
        let bin_candidates = [
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.join("whisperx_align")))
                .unwrap_or_default(),
            PathBuf::from("bin/whisperx_align"),
            PathBuf::from("crates/pocket-tts/bin/whisperx_align"),
        ];
        for path in &bin_candidates {
            if path.exists() {
                return Ok(Backend::Binary(path.clone()));
            }
        }
        // Check PATH
        if let Ok(output) = Command::new("which").arg("whisperx_align").output() {
            if output.status.success() {
                let p = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !p.is_empty() {
                    return Ok(Backend::Binary(PathBuf::from(p)));
                }
            }
        }

        // 3. Docker
        if Command::new("docker")
            .args(["image", "inspect", "whisperx-align"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return Ok(Backend::Docker("whisperx-align".to_string()));
        }

        anyhow::bail!(
            "No WhisperX backend found. Options:\n\
             1. Install whisperx: pip install whisperx (+ scripts/whisperx_align.py)\n\
             2. Place compiled whisperx_align binary next to your app\n\
             3. Build Docker image: cd scripts && docker build -f Dockerfile.whisperx -t whisperx-align ."
        )
    }

    /// Align audio to produce word-level timestamps.
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

        // 3. Run alignment via detected backend
        let output = match &self.backend {
            Backend::PythonScript(script) => {
                let mut cmd = Command::new("python3");
                cmd.arg(script).arg(&wav_path).arg(&json_path);
                if !text.is_empty() {
                    cmd.arg("--text").arg(text);
                }
                cmd.output()?
            }
            Backend::Binary(bin) => {
                let mut cmd = Command::new(bin);
                cmd.arg(&wav_path).arg(&json_path);
                if !text.is_empty() {
                    cmd.arg("--text").arg(text);
                }
                cmd.output()?
            }
            Backend::Docker(image) => {
                let mut cmd = Command::new("docker");
                cmd.args(["run", "--rm", "-v"]);
                cmd.arg(format!("{}:{}", tmp_dir.display(), tmp_dir.display()));
                cmd.arg(image);
                cmd.arg(&wav_path).arg(&json_path);
                if !text.is_empty() {
                    cmd.arg("--text").arg(text);
                }
                cmd.output()?
            }
        };

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("WhisperX alignment failed: {}", stderr);
        }

        // 4. Parse JSON
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
        let aligner = WhisperAligner::load(&device).unwrap();
        println!("Backend: {:?}", aligner.backend);
    }
}
