//! Minimal demo: generate speech with word timestamps using the "alba" voice.
//! Prints each word with its time range, then saves audio to output.wav.
//!
//! Run: cargo run -p pocket-tts --example timestamp_demo

use anyhow::Result;
use pocket_tts::TTSModel;

const VOICE_REPO: &str = "kyutai/pocket-tts-without-voice-cloning";

fn main() -> Result<()> {
    let text = "The quick brown fox jumps over the lazy dog.";

    eprintln!("Loading TTS + alignment models...");
    let model = TTSModel::load_with_alignment("b6369a24")?;

    eprintln!("Loading alba voice...");
    let voice_path = pocket_tts::weights::download_if_necessary(&format!(
        "hf://{}/embeddings/alba.safetensors",
        VOICE_REPO
    ))?;
    let voice_state = model.get_voice_state_from_prompt_file(&voice_path)?;

    eprintln!("Generating...");
    let result = model.generate_with_timestamps(text, &voice_state)?;

    // Print timestamps
    println!("\n  Word Timestamps:");
    println!("  {:-<50}", "");
    for ts in &result.word_timestamps {
        println!("  {:>6.3}s - {:>6.3}s  {}", ts.start_sec, ts.end_sec, ts.word);
    }
    println!("  {:-<50}", "");

    let total_samples = result.audio.dims().last().copied().unwrap_or(0);
    let duration = total_samples as f32 / 24000.0;
    println!("  Audio duration: {:.3}s ({} samples)", duration, total_samples);

    // Save audio
    let audio_data = result.audio.flatten_all()?.to_vec1::<f32>()?;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create("output.wav", spec)?;
    for sample in &audio_data {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;
    eprintln!("\nSaved to output.wav");

    Ok(())
}
