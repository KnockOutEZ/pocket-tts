//! Compare our whisper-cli timestamps vs Python whisper CLI on the same audio.
//! cargo run -p pocket-tts --example timestamp_compare --release

use anyhow::Result;
use pocket_tts::TTSModel;

const VOICE_REPO: &str = "kyutai/pocket-tts-without-voice-cloning";

fn main() -> Result<()> {
    let text = "Dr. Smith & his team of 42 researchers — working at MIT's \
                AI lab — published 3 papers in 2024. They found that GPT-4's \
                accuracy was 97.8%, which is remarkable! \"We're thrilled,\" \
                said Dr. Smith. The cost? Only $1.2 million per year... \
                not bad for cutting-edge research.";

    eprintln!("Loading models...");
    let model = TTSModel::load_with_alignment("b6369a24")?;
    let voice_path = pocket_tts::weights::download_if_necessary(&format!(
        "hf://{}/embeddings/alba.safetensors", VOICE_REPO
    ))?;
    let voice_state = model.get_voice_state_from_prompt_file(&voice_path)?;

    eprintln!("Generating...");
    let result = model.generate_with_timestamps(text, &voice_state)?;

    let samples = result.audio.dims().last().copied().unwrap_or(0);
    let duration = samples as f32 / 24000.0;

    // Save audio
    let audio_data = result.audio.flatten_all()?.to_vec1::<f32>()?;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create("/tmp/tts_symbols.wav", spec)?;
    for &s in &audio_data {
        writer.write_sample((s * 32767.0).clamp(-32768.0, 32767.0) as i16)?;
    }
    writer.finalize()?;

    // Save our timestamps
    let ours: Vec<serde_json::Value> = result.word_timestamps.iter().map(|t| {
        serde_json::json!({ "word": t.word, "start": t.start_sec, "end": t.end_sec })
    }).collect();
    std::fs::write("/tmp/ours_symbols.json", serde_json::to_string_pretty(&ours)?)?;

    println!("Audio: {:.3}s ({} samples, {} words)", duration, samples, result.word_timestamps.len());
    println!("\n=== OUR TIMESTAMPS ===");
    for ts in &result.word_timestamps {
        let dur = (ts.end_sec - ts.start_sec) * 1000.0;
        println!("  {:>6.3}s - {:>6.3}s ({:>4.0}ms) {}", ts.start_sec, ts.end_sec, dur, ts.word);
    }

    println!("\nSaved /tmp/tts_symbols.wav and /tmp/ours_symbols.json");
    println!("\nRun: whisper /tmp/tts_symbols.wav --model small.en --language en --word_timestamps True --output_format json --output_dir /tmp/whisper_symbols");

    Ok(())
}
