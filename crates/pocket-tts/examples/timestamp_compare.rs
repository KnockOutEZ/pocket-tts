//! Compare our CTC alignment vs Whisper word timestamps on longer text.
//!
//! cargo run -p pocket-tts --example timestamp_compare --release

use anyhow::Result;
use pocket_tts::TTSModel;

const VOICE_REPO: &str = "kyutai/pocket-tts-without-voice-cloning";

fn main() -> Result<()> {
    let text = "In the beginning, there was silence. Then came the machines, humming and \
                whirring, processing language at speeds no human could match. They didn't \
                just understand words. They felt the rhythm, the cadence, the subtle pauses \
                between thoughts. And when they spoke, it was almost indistinguishable from \
                a real human voice. Almost.";

    eprintln!("Loading models...");
    let model = TTSModel::load_with_alignment("b6369a24")?;
    let voice_path = pocket_tts::weights::download_if_necessary(&format!(
        "hf://{}/embeddings/alba.safetensors", VOICE_REPO
    ))?;
    let voice_state = model.get_voice_state_from_prompt_file(&voice_path)?;

    eprintln!("Generating: \"{}\"", &text[..60]);
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
    let mut writer = hound::WavWriter::create("/tmp/tts_compare2.wav", spec)?;
    for &s in &audio_data {
        writer.write_sample((s * 32767.0).clamp(-32768.0, 32767.0) as i16)?;
    }
    writer.finalize()?;

    // Save our timestamps
    let ours: Vec<serde_json::Value> = result.word_timestamps.iter().map(|t| {
        serde_json::json!({ "word": t.word, "start": t.start_sec, "end": t.end_sec })
    }).collect();
    std::fs::write("/tmp/ours2.json", serde_json::to_string_pretty(&ours)?)?;

    println!("Audio: {:.3}s ({} samples, {} words)", duration, samples, result.word_timestamps.len());
    println!("\n=== OUR TIMESTAMPS ===");
    for ts in &result.word_timestamps {
        let dur = (ts.end_sec - ts.start_sec) * 1000.0;
        let flag = if dur > 500.0 { " <<< LONG" } else if dur < 30.0 { " <<< SHORT" } else { "" };
        println!("  {:>6.3}s - {:>6.3}s  ({:>4.0}ms)  {:20}{}", ts.start_sec, ts.end_sec, dur, ts.word, flag);
    }

    eprintln!("\nSaved /tmp/tts_compare2.wav and /tmp/ours2.json");
    println!("\nRun: whisper /tmp/tts_compare2.wav --model large-v3 --language en --word_timestamps True --output_format json --output_dir /tmp/whisper_out2");

    Ok(())
}
