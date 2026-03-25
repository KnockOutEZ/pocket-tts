use crate::alignment::forced_align::{
    WordTimestamp, text_to_ctc_targets, viterbi_forced_align, path_to_word_timestamps,
};
use crate::alignment::wav2vec2::Wav2Vec2Model;
use crate::audio::resample;
use candle_core::{Device, Tensor};

const TTS_SAMPLE_RATE: u32 = 24000;
const WAV2VEC2_SAMPLE_RATE: u32 = 16000;

#[derive(Clone)]
pub struct Wav2Vec2Aligner {
    model: Wav2Vec2Model,
    device: Device,
}

impl Wav2Vec2Aligner {
    pub fn load(device: &Device) -> anyhow::Result<Self> {
        let model = Wav2Vec2Model::load(device)?;
        Ok(Self { model, device: device.clone() })
    }

    pub fn align(&self, audio: &Tensor, text: &str) -> anyhow::Result<Vec<WordTimestamp>> {
        // 1. Build CTC targets from text
        let (targets, word_spans) = text_to_ctc_targets(text);
        if targets.is_empty() {
            return Ok(Vec::new());
        }

        // 2. Prepare audio: ensure [1, T], resample 24kHz → 16kHz, normalize
        let audio = match audio.dims().len() {
            1 => audio.unsqueeze(0)?,                     // [T] → [1, T]
            2 if audio.dims()[0] == 1 => audio.clone(),   // [1, T]
            2 => audio.mean(0)?.unsqueeze(0)?,            // [C, T] → mono → [1, T]
            _ => anyhow::bail!("Unexpected audio shape: {:?}", audio.dims()),
        };

        let audio_16k = resample(&audio, TTS_SAMPLE_RATE, WAV2VEC2_SAMPLE_RATE)?;

        // Normalize: zero mean, unit variance (wav2vec2 convention)
        let mean = audio_16k.mean_all()?;
        let centered = audio_16k.broadcast_sub(&mean)?;
        let var = (&centered * &centered)?.mean_all()?;
        let std = (var + 1e-7)?.sqrt()?;
        let normalized = centered.broadcast_div(&std)?;

        // Move to model device if needed
        let normalized = if !normalized.device().same_device(&self.device) {
            normalized.to_device(&self.device)?
        } else {
            normalized
        };

        // 3. Forward pass: [1, 1, samples] → [1, frames, 32]
        let input = normalized.unsqueeze(0)?;
        let log_probs = self.model.forward(&input)?;

        // 4. Extract as Vec<Vec<f32>> for Viterbi
        let log_probs_2d = log_probs.squeeze(0)?; // [frames, 32]
        let flat = log_probs_2d.to_vec2::<f32>()?;

        // 5. Viterbi forced alignment
        let path = viterbi_forced_align(&flat, &targets)?;

        // 6. Convert to word timestamps
        Ok(path_to_word_timestamps(&path, &targets, &word_spans))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    #[ignore] // requires ~90MB model download
    fn test_aligner_on_silence() {
        let device = Device::Cpu;
        let aligner = Wav2Vec2Aligner::load(&device).unwrap();

        // 1 second of silence at 24kHz
        let audio = Tensor::zeros((1, 24000), DType::F32, &device).unwrap();
        let timestamps = aligner.align(&audio, "hello world").unwrap();

        assert_eq!(timestamps.len(), 2);
        assert_eq!(timestamps[0].word, "hello");
        assert_eq!(timestamps[1].word, "world");
    }
}
