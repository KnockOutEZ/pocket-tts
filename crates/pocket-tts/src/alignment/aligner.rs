//! WhisperX-based aligner for word-level timestamps.
//!
//! Uses WhisperX (Whisper + wav2vec2 forced alignment) for production-grade
//! word-level timestamps (~20-50ms accuracy).
//!
//! Primary backend: HTTP server (models loaded once, ~2-4s per alignment).
//! Auto-starts the server on first use. Fallback: subprocess per call.

use crate::alignment::forced_align::WordTimestamp;
use crate::audio::resample;
use candle_core::{Device, Tensor};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::io::Read;

const TTS_SAMPLE_RATE: u32 = 24000;
const WHISPER_SAMPLE_RATE: u32 = 16000;
const SERVER_PORT: u16 = 9876;

/// WhisperX-based aligner for production-grade word timestamps.
///
/// Manages a persistent WhisperX server process. Models load once (~10s),
/// then each alignment takes ~2-4s regardless of hardware.
#[derive(Clone)]
pub struct WhisperAligner {
    server_script: PathBuf,
    server_process: std::sync::Arc<Mutex<Option<Child>>>,
}

impl WhisperAligner {
    /// Load the aligner. Finds the server script and starts the server.
    pub fn load(_device: &Device) -> anyhow::Result<Self> {
        let server_script = Self::find_server_script()?;

        let aligner = Self {
            server_script,
            server_process: std::sync::Arc::new(Mutex::new(None)),
        };

        // Start server eagerly so models load during app startup
        aligner.ensure_server()?;

        Ok(aligner)
    }

    /// Check if the WhisperX backend is available (script or binary exists).
    /// Does NOT start the server or load models.
    pub fn check_available() -> anyhow::Result<()> {
        Self::find_server_script()?;
        Ok(())
    }

    fn find_server_script() -> anyhow::Result<PathBuf> {
        let candidates = [
            PathBuf::from("scripts/whisperx_server.py"),
            PathBuf::from("crates/pocket-tts/scripts/whisperx_server.py"),
            // Standalone binary (PyInstaller)
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.join("whisperx_server")))
                .unwrap_or_default(),
        ];

        for path in &candidates {
            if path.exists() {
                return Ok(path.clone());
            }
        }

        anyhow::bail!(
            "WhisperX server script not found. Expected scripts/whisperx_server.py"
        )
    }

    /// Start the server if not already running.
    fn ensure_server(&self) -> anyhow::Result<()> {
        // Check if server is already responding
        if self.server_healthy() {
            return Ok(());
        }

        let mut proc = self.server_process.lock().unwrap();

        // Kill stale process if any
        if let Some(ref mut child) = *proc {
            let _ = child.kill();
        }

        // Start server
        let is_py = self.server_script.extension().map_or(false, |e| e == "py");
        let child = if is_py {
            Command::new("python3")
                .arg(&self.server_script)
                .arg("--port")
                .arg(SERVER_PORT.to_string())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()?
        } else {
            Command::new(&self.server_script)
                .arg("--port")
                .arg(SERVER_PORT.to_string())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()?
        };

        *proc = Some(child);
        drop(proc);

        // Wait for server to be ready (max 60s for model loading)
        for _ in 0..120 {
            std::thread::sleep(std::time::Duration::from_millis(500));
            if self.server_healthy() {
                return Ok(());
            }
        }

        anyhow::bail!("WhisperX server failed to start within 60s")
    }

    fn server_healthy(&self) -> bool {
        std::net::TcpStream::connect(format!("127.0.0.1:{}", SERVER_PORT)).is_ok()
    }

    /// Align audio to produce word-level timestamps.
    pub fn align(&self, audio: &Tensor, _text: &str) -> anyhow::Result<Vec<WordTimestamp>> {
        self.ensure_server()?;

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

        // 3. POST to server
        let body = serde_json::json!({ "wav_path": wav_path.to_str() });
        let body_bytes = serde_json::to_vec(&body)?;

        let mut stream = std::net::TcpStream::connect(format!("127.0.0.1:{}", SERVER_PORT))?;
        stream.set_read_timeout(Some(std::time::Duration::from_secs(120)))?;

        use std::io::Write;
        write!(
            stream,
            "POST /align HTTP/1.1\r\n\
             Host: 127.0.0.1\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             \r\n",
            body_bytes.len()
        )?;
        stream.write_all(&body_bytes)?;
        stream.flush()?;

        // Read response
        let mut response = Vec::new();
        stream.read_to_end(&mut response)?;
        let response_str = String::from_utf8_lossy(&response);

        // Parse HTTP response — find JSON body after headers
        let body_start = response_str
            .find("\r\n\r\n")
            .map(|i| i + 4)
            .unwrap_or(0);
        let json_body = &response_str[body_start..];

        let words: Vec<serde_json::Value> = serde_json::from_str(json_body)?;

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

        // Cleanup
        let _ = std::fs::remove_file(&wav_path);

        Ok(timestamps)
    }
}

impl Drop for WhisperAligner {
    fn drop(&mut self) {
        // Only kill if we're the last reference
        if std::sync::Arc::strong_count(&self.server_process) == 1 {
            if let Ok(mut proc) = self.server_process.lock() {
                if let Some(ref mut child) = *proc {
                    let _ = child.kill();
                }
            }
        }
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
        println!("Server running on port {}", SERVER_PORT);
        drop(aligner);
    }
}
