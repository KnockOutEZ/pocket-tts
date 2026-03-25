# Word-Level Timestamps — Integration Guide

## What Was Added

The `pocket-tts` fork on branch `feat/word-timestamps` adds word-level timestamp support for read-aloud / word-highlight sync. It uses a native Whisper model (via candle-transformers, pure Rust) to align generated audio with text. No Python, no C++ FFI, no external dependencies — ships as part of the crate.

## Branch

```
repo: KnockOutEZ/pocket-tts
branch: feat/word-timestamps
```

## New Public API

### Types

```rust
use pocket_tts::{TTSModel, GenerationResult, WordTimestamp, ModelState};

/// Returned by generate_with_timestamps / generate_sentence_with_timestamps
pub struct GenerationResult {
    pub audio: Tensor,                      // [C, T] audio at 24kHz
    pub word_timestamps: Vec<WordTimestamp>, // one per word
}

pub struct WordTimestamp {
    pub word: String,     // the word as transcribed
    pub start_sec: f32,   // start time in seconds
    pub end_sec: f32,     // end time in seconds
}
```

### Loading (with alignment model)

```rust
// Loads TTS model (~90MB) + Whisper-tiny.en alignment model (~75MB)
// Both downloaded from HuggingFace on first use, cached locally
let model = TTSModel::load_with_alignment("b6369a24")?;
```

This replaces `TTSModel::load()` when you need timestamps. The existing `load()` still works for audio-only generation without the alignment model.

### Voice State (unchanged)

```rust
// Predefined voice (downloads from HF)
let voice_path = pocket_tts::weights::download_if_necessary(
    "hf://kyutai/pocket-tts-without-voice-cloning/embeddings/alba.safetensors"
)?;
let voice_state = model.get_voice_state_from_prompt_file(&voice_path)?;

// Or from bytes (e.g., user-uploaded voice)
let voice_state = model.get_voice_state_from_bytes(&wav_bytes)?;
```

### Generating Audio + Timestamps

**Option A: Full text (library splits into sentences internally)**

```rust
let result = model.generate_with_timestamps("Hello world. This is a test.", &voice_state)?;
// result.audio: Tensor [C, T] at 24kHz
// result.word_timestamps: [
//   WordTimestamp { word: "Hello", start_sec: 0.0, end_sec: 0.24 },
//   WordTimestamp { word: "world.", start_sec: 0.24, end_sec: 0.52 },
//   WordTimestamp { word: "This", start_sec: 0.80, end_sec: 1.02 },
//   WordTimestamp { word: "is", start_sec: 1.02, end_sec: 1.14 },
//   ...
// ]
```

**Option B: Single sentence (caller controls chunking — recommended for your pipeline)**

```rust
let result = model.generate_sentence_with_timestamps("Hello world.", &voice_state)?;
```

Use this when your Tauri app already splits text into sentences. Avoids double-splitting.

### Converting Audio to WAV Bytes

```rust
let audio_data: Vec<f32> = result.audio.flatten_all()?.to_vec1()?;
let spec = hound::WavSpec {
    channels: 1,
    sample_rate: 24000,
    bits_per_sample: 16,
    sample_format: hound::SampleFormat::Int,
};
let mut cursor = std::io::Cursor::new(Vec::new());
let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
for &s in &audio_data {
    writer.write_sample((s * 32767.0).clamp(-32768.0, 32767.0) as i16)?;
}
writer.finalize()?;
let wav_bytes = cursor.into_inner();
```

## Recommended Architecture for Tauri App

```
Frontend (React/Svelte)          Tauri Backend (Rust)
─────────────────────           ────────────────────────
                                On startup:
                                  model = TTSModel::load_with_alignment("b6369a24")
                                  voice_state = load voice

User clicks "Read Aloud"
  → send paragraph text  ──→   Split into sentences
                                For sentence N:
                                  result = model.generate_sentence_with_timestamps(
                                    sentence, &voice_state
                                  )
                                  wav_bytes = encode_wav(result.audio)
  ←── { wav_bytes_base64,  ←──   return audio + timestamps
        timestamps: [{
          word, start, end
        }, ...] }

Play audio via Web Audio API
Start highlight loop:
  on each animationFrame:
    currentTime = audio.currentTime
    find word where start <= currentTime < end
    highlight that word
```

### Key Points for the Frontend

1. **Sentence pipeline**: Generate sentence N, start playing it, pre-generate N+1 in background
2. **Word matching**: The `word` field in timestamps comes from Whisper's transcription, not the original text. Do fuzzy matching (lowercase, strip punctuation) to map timestamps to the displayed words
3. **Timing**: `start_sec` and `end_sec` are relative to each sentence's audio. If you concatenate sentences, offset by cumulative duration
4. **Buffer**: The first sentence takes ~2-5s to generate + align. Buffer before playback starts
5. **Sample rate**: Audio is 24kHz mono. The frontend plays it via `<audio>` element or Web Audio API

### Tauri Command Example

```rust
#[tauri::command]
async fn generate_speech(
    text: String,
    state: tauri::State<'_, AppState>,
) -> Result<SpeechResult, String> {
    let model = &state.model;
    let voice = &state.voice_state;

    let result = tokio::task::spawn_blocking(move || {
        model.generate_sentence_with_timestamps(&text, voice)
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;

    let audio_data: Vec<f32> = result.audio
        .flatten_all().map_err(|e| e.to_string())?
        .to_vec1().map_err(|e| e.to_string())?;

    // Encode to WAV base64...

    Ok(SpeechResult {
        audio_base64,
        timestamps: result.word_timestamps.into_iter().map(|t| WordTs {
            word: t.word,
            start: t.start_sec,
            end: t.end_sec,
        }).collect(),
    })
}
```

## Performance

- **TTS generation**: ~1-3s per sentence on CPU (release mode)
- **Whisper alignment**: ~2-5s per sentence on CPU
- **Total per sentence**: ~3-8s
- **First load**: Downloads ~165MB of models on first run (cached after)
- **Memory**: ~500MB for both models loaded

## Dependencies Added

None new. Uses `candle-transformers` (already in dependency tree) for Whisper model. Mel filterbank (64KB) is embedded in the binary.

## Files Changed

```
crates/pocket-tts/src/
├── alignment/
│   ├── mod.rs              # Module declarations
│   ├── aligner.rs          # WhisperAligner (native Whisper via candle)
│   ├── forced_align.rs     # WordTimestamp type + Viterbi utils
│   └── melfilters.bytes    # Mel filterbank (64KB, embedded)
├── tts_model.rs            # +GenerationResult, +load_with_alignment,
│                           #  +generate_with_timestamps,
│                           #  +generate_sentence_with_timestamps
└── lib.rs                  # Re-exports: WhisperAligner, WordTimestamp,
                            #  GenerationResult
```
