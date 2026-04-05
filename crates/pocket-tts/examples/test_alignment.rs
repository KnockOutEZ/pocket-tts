//! Visual + programmatic alignment tester.
//!
//! Generates speech for test sentences, aligns them, and:
//!   1. Prints a terminal table with per-word timestamps
//!   2. Runs programmatic assertions (gaps, ordering, word count)
//!   3. Optionally writes an HTML file that plays audio + highlights words in sync
//!
//! Usage:
//!   HF_TOKEN=hf_xxx cargo run --release -p pocket-tts --example test_alignment
//!   HF_TOKEN=hf_xxx cargo run --release -p pocket-tts --example test_alignment -- --html out.html
//!   HF_TOKEN=hf_xxx cargo run --release -p pocket-tts --example test_alignment -- --voice alba

use pocket_tts::audio::write_wav;
use pocket_tts::tts_model::TTSModel;
use std::path::PathBuf;

const SAMPLE_RATE: u32 = 24000;

/// Test sentences covering edge cases that trigger the offset bug.
const TEST_SENTENCES: &[&str] = &[
    // Basic
    "Hello world, this is a test.",
    // Numbers — normalize to empty, used to cause offset drift
    "Chapter 1: The Beginning",
    "There are 3 cats and 12 dogs.",
    "In 2024, everything changed.",
    // Punctuation-heavy
    "Well... I don't know — really!",
    // Symbols that vanish
    "Tom & Jerry are friends.",
    "Price is $100 per unit.",
    // Hyphenated words (hyphen becomes space in normalization)
    "This is a well-known fact.",
    // Apostrophes
    "I'm sure they're coming, aren't they?",
    // Short sentence
    "Yes.",
    // Long sentence
    "The quick brown fox jumps over the lazy dog while the cat watches from the windowsill.",
];

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let html_path = args
        .windows(2)
        .find(|w| w[0] == "--html")
        .map(|w| PathBuf::from(&w[1]));
    let voice_name = args
        .windows(2)
        .find(|w| w[0] == "--voice")
        .map(|w| w[1].as_str())
        .unwrap_or("alba");

    println!("[1/3] Loading TTS model (may download ~450MB on first run)...");
    let mut model = TTSModel::load("b6369a24")?;

    println!("[2/3] Loading alignment model (may download ~300MB on first run)...");
    let aligner = pocket_tts::NativeAligner::load(&candle_core::Device::Cpu)?;
    model.aligner = Some(aligner);

    println!("[3/3] Loading voice: {voice_name}...");
    let voice_path = pocket_tts::weights::download_if_necessary(&format!(
        "hf://kyutai/pocket-tts-without-voice-cloning/embeddings/{voice_name}.safetensors"
    ))?;
    let voice_state = model.get_voice_state_from_prompt_file(&voice_path)?;
    println!("Ready! Running {} test sentences...\n", TEST_SENTENCES.len());

    let mut all_passed = true;
    let mut html_entries: Vec<HtmlEntry> = Vec::new();
    let out_dir = std::env::temp_dir().join("pocket_tts_alignment_test");
    std::fs::create_dir_all(&out_dir)?;

    for (i, &sentence) in TEST_SENTENCES.iter().enumerate() {
        println!("\n{}", "=".repeat(70));
        println!("TEST {}: {:?}", i + 1, sentence);
        println!("{}", "=".repeat(70));

        let result = model.generate_sentence_with_timestamps(sentence, &voice_state)?;
        let timestamps = &result.word_timestamps;
        let audio_samples = result.audio.dims().last().copied().unwrap_or(0);
        let audio_duration = audio_samples as f32 / SAMPLE_RATE as f32;

        // -- Terminal table --
        println!(
            "{:<4} {:<20} {:>8} {:>8} {:>8}",
            "#", "Word", "Start", "End", "Dur(ms)"
        );
        println!("{}", "-".repeat(52));
        for (j, ts) in timestamps.iter().enumerate() {
            let dur_ms = (ts.end_sec - ts.start_sec) * 1000.0;
            println!(
                "{:<4} {:<20} {:>8.3} {:>8.3} {:>8.1}",
                j, ts.word, ts.start_sec, ts.end_sec, dur_ms
            );
        }
        println!(
            "\nAudio duration: {:.3}s | Words: {} | Original words: {}",
            audio_duration,
            timestamps.len(),
            sentence.split_whitespace().count()
        );

        // -- Programmatic assertions --
        let mut errors: Vec<String> = Vec::new();

        // 1. Word count must match original text
        let orig_count = sentence.split_whitespace().count();
        if timestamps.len() != orig_count {
            errors.push(format!(
                "WORD COUNT MISMATCH: got {} timestamps, expected {} words",
                timestamps.len(),
                orig_count
            ));
        }

        // 2. Original words preserved (not normalized forms)
        let orig_words: Vec<&str> = sentence.split_whitespace().collect();
        for (j, (ts, &orig)) in timestamps.iter().zip(orig_words.iter()).enumerate() {
            if ts.word != orig {
                errors.push(format!(
                    "WORD MISMATCH at index {}: got {:?}, expected {:?}",
                    j, ts.word, orig
                ));
            }
        }

        // 3. start <= end for every word
        for ts in timestamps {
            if ts.start_sec > ts.end_sec {
                errors.push(format!(
                    "INVALID SPAN: '{}' start ({}) > end ({})",
                    ts.word, ts.start_sec, ts.end_sec
                ));
            }
        }

        // 4. Monotonically non-decreasing start times
        for j in 1..timestamps.len() {
            if timestamps[j].start_sec < timestamps[j - 1].start_sec {
                errors.push(format!(
                    "NOT MONOTONIC: '{}' starts at {:.3} before '{}' at {:.3}",
                    timestamps[j].word,
                    timestamps[j].start_sec,
                    timestamps[j - 1].word,
                    timestamps[j - 1].start_sec
                ));
            }
        }

        // 5. No huge gaps between consecutive words (>500ms suggests misalignment)
        for j in 1..timestamps.len() {
            let gap = timestamps[j].start_sec - timestamps[j - 1].end_sec;
            if gap > 0.5 {
                errors.push(format!(
                    "LARGE GAP: {:.0}ms between '{}' and '{}'",
                    gap * 1000.0,
                    timestamps[j - 1].word,
                    timestamps[j].word
                ));
            }
        }

        // 6. Last word shouldn't end way past audio duration
        if let Some(last) = timestamps.last() {
            if last.end_sec > audio_duration + 0.3 {
                errors.push(format!(
                    "OVERRUN: last word ends at {:.3}s but audio is {:.3}s",
                    last.end_sec, audio_duration
                ));
            }
        }

        if errors.is_empty() {
            println!("PASS: All assertions OK");
        } else {
            all_passed = false;
            for e in &errors {
                println!("FAIL: {}", e);
            }
        }

        // Save WAV + collect for HTML
        let wav_name = format!("test_{}.wav", i + 1);
        let wav_path = out_dir.join(&wav_name);
        write_wav(&wav_path, &result.audio, SAMPLE_RATE)?;

        html_entries.push(HtmlEntry {
            _sentence: sentence.to_string(),
            wav_path: wav_path.clone(),
            timestamps: timestamps.clone(),
            _audio_duration: audio_duration,
            errors,
        });
    }

    // -- Generate HTML if requested --
    if let Some(path) = &html_path {
        write_html(path, &html_entries)?;
        println!("\n\nHTML written to: {}", path.display());
        println!("Open it in a browser to visually verify word highlighting sync.");
    } else {
        println!("\n\nTip: add --html out.html to generate a visual test page");
    }

    println!(
        "\nWAV files saved in: {}",
        out_dir.display()
    );

    if all_passed {
        println!("\nALL TESTS PASSED");
        Ok(())
    } else {
        anyhow::bail!("Some tests failed — see FAIL lines above");
    }
}

struct HtmlEntry {
    _sentence: String,
    wav_path: PathBuf,
    timestamps: Vec<pocket_tts::alignment::WordTimestamp>,
    _audio_duration: f32,
    errors: Vec<String>,
}

fn write_html(path: &PathBuf, entries: &[HtmlEntry]) -> anyhow::Result<()> {
    use std::io::Write;

    let mut f = std::fs::File::create(path)?;

    write!(
        f,
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Alignment Test — pocket-tts</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; background: #0a0a0a; color: #e0e0e0; padding: 2rem; }}
  h1 {{ margin-bottom: 1.5rem; color: #fff; }}
  .test-card {{ background: #1a1a1a; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #333; }}
  .test-card.has-errors {{ border-color: #e74c3c; }}
  .test-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }}
  .test-num {{ font-weight: 700; color: #888; }}
  .badge {{ padding: 2px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; }}
  .badge.pass {{ background: #2ecc4033; color: #2ecc40; }}
  .badge.fail {{ background: #e74c3c33; color: #e74c3c; }}
  .sentence {{ font-size: 1.3rem; line-height: 2; margin-bottom: 1rem; min-height: 2.6rem; }}
  .word {{ display: inline-block; padding: 2px 4px; border-radius: 4px; transition: background 0.08s, color 0.08s; cursor: default; }}
  .word.active {{ background: #3b82f6; color: #fff; }}
  .word.past {{ color: #888; }}
  .word.interpolated {{ border-bottom: 2px dashed #f59e0b; }}
  audio {{ width: 100%; margin-bottom: 0.5rem; }}
  .controls {{ display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.5rem; }}
  .controls button {{ background: #333; color: #fff; border: none; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 0.85rem; }}
  .controls button:hover {{ background: #555; }}
  .speed-label {{ font-size: 0.8rem; color: #888; }}
  .timestamps {{ font-family: monospace; font-size: 0.75rem; color: #666; margin-top: 0.75rem; white-space: pre-wrap; }}
  .errors {{ color: #e74c3c; font-size: 0.85rem; margin-top: 0.5rem; }}
  .legend {{ margin-bottom: 1.5rem; font-size: 0.85rem; color: #888; }}
  .legend span {{ margin-right: 1.5rem; }}
  .legend .swatch {{ display: inline-block; width: 14px; height: 14px; border-radius: 3px; vertical-align: middle; margin-right: 4px; }}
</style>
</head>
<body>
<h1>pocket-tts Alignment Test</h1>
<div class="legend">
  <span><span class="swatch" style="background:#3b82f6"></span> Active word</span>
  <span><span class="swatch" style="background:transparent; border-bottom: 2px dashed #f59e0b; height: 0; width: 14px;"></span> Interpolated (no audio chars)</span>
</div>
"#
    )?;

    for (i, entry) in entries.iter().enumerate() {
        let pass = entry.errors.is_empty();
        let card_class = if pass { "test-card" } else { "test-card has-errors" };
        let badge = if pass {
            r#"<span class="badge pass">PASS</span>"#
        } else {
            r#"<span class="badge fail">FAIL</span>"#
        };

        // Base64-encode the WAV for inline playback
        let wav_bytes = std::fs::read(&entry.wav_path)?;
        let wav_b64 = base64_encode(&wav_bytes);

        // Build word spans
        let mut word_spans = String::new();
        for (j, ts) in entry.timestamps.iter().enumerate() {
            let is_interpolated = (ts.end_sec - ts.start_sec).abs() < 0.001;
            let class = if is_interpolated { "word interpolated" } else { "word" };
            word_spans.push_str(&format!(
                r#"<span class="{class}" data-idx="{j}" data-start="{:.4}" data-end="{:.4}">{}</span> "#,
                ts.start_sec,
                ts.end_sec,
                html_escape(&ts.word)
            ));
        }

        // Timestamp dump
        let mut ts_dump = String::new();
        for (j, ts) in entry.timestamps.iter().enumerate() {
            ts_dump.push_str(&format!(
                "{:>3}  {:<20} {:.3}s - {:.3}s  ({:.0}ms)\n",
                j,
                ts.word,
                ts.start_sec,
                ts.end_sec,
                (ts.end_sec - ts.start_sec) * 1000.0
            ));
        }

        let error_html = if entry.errors.is_empty() {
            String::new()
        } else {
            let mut s = String::from(r#"<div class="errors">"#);
            for e in &entry.errors {
                s.push_str(&format!("{}<br>", html_escape(e)));
            }
            s.push_str("</div>");
            s
        };

        write!(
            f,
            r#"<div class="{card_class}" id="card-{i}">
  <div class="test-header">
    <span class="test-num">Test {num}</span>
    {badge}
  </div>
  <div class="sentence" id="sentence-{i}">{word_spans}</div>
  <audio id="audio-{i}" src="data:audio/wav;base64,{wav_b64}" preload="auto"></audio>
  <div class="controls">
    <button onclick="playTest({i})">Play</button>
    <button onclick="playTest({i}, 0.5)">0.5x</button>
    <button onclick="playTest({i}, 0.75)">0.75x</button>
    <button onclick="document.getElementById('audio-{i}').pause()">Pause</button>
    <span class="speed-label" id="speed-{i}"></span>
  </div>
  {error_html}
  <details><summary style="color:#666;cursor:pointer;font-size:0.8rem;margin-top:0.5rem">Timestamps</summary>
    <div class="timestamps">{ts_dump}</div>
  </details>
</div>
"#,
            num = i + 1,
        )?;
    }

    write!(
        f,
        r#"
<script>
function playTest(idx, rate) {{
  // Reset all words in this card
  const container = document.getElementById('sentence-' + idx);
  const words = container.querySelectorAll('.word');
  words.forEach(w => w.classList.remove('active', 'past'));

  const audio = document.getElementById('audio-' + idx);
  audio.playbackRate = rate || 1.0;
  document.getElementById('speed-' + idx).textContent = (rate || 1.0) + 'x';
  audio.currentTime = 0;
  audio.play();

  function update() {{
    if (audio.paused) return;
    const t = audio.currentTime;
    words.forEach(w => {{
      const start = parseFloat(w.dataset.start);
      const end = parseFloat(w.dataset.end);
      w.classList.remove('active', 'past');
      if (t >= start && t < end) {{
        w.classList.add('active');
      }} else if (t >= end) {{
        w.classList.add('past');
      }}
    }});
    requestAnimationFrame(update);
  }}
  requestAnimationFrame(update);
}}
</script>
</body>
</html>"#
    )?;

    Ok(())
}

fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
