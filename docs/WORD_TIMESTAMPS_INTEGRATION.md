# Word-Level Timestamps — Integration Guide

## Overview

pocket-tts on branch `feat/word-timestamps` provides word-level timestamps for read-aloud apps. Uses WhisperX (Whisper + wav2vec2 forced alignment) for ~20-50ms accuracy.

## Architecture

```
App startup:         TTSModel::preload_weights()  → download/check all models
User opens book:     TTSModel::load_with_alignment()  → load TTS + start WhisperX server
User reads:          generate_sentence_with_timestamps()  → ~3-5s per sentence
User closes book:    drop(model)  → kill server, free memory
```

The WhisperX server loads models once (~15s), then each alignment is ~2s. The server runs as a child process and dies automatically when the model is dropped.

## Setup

### 1. For Development (your machine)

```bash
pip install whisperx
```

The aligner calls `scripts/whisperx_server.py` directly via Python.

### 2. For Shipping (customer machines)

Download the pre-built `whisperx_server` binary from GitHub Releases (built by CI). Place it next to your Tauri app binary. No Python needed — it's self-contained.

```
MyApp.app/Contents/MacOS/
  my-tauri-app          ← Tauri binary
  whisperx_server       ← WhisperX sidecar (~400-500MB)
```

CI builds binaries for: macOS ARM64, macOS Intel, Linux x64, Windows x64.

Trigger a build: push a `v*` tag or use "Run workflow" on GitHub Actions.

## API

### Types

```rust
use pocket_tts::{TTSModel, GenerationResult, WordTimestamp, ModelState};

pub struct GenerationResult {
    pub audio: Tensor,
    pub word_timestamps: Vec<WordTimestamp>,
}

pub struct WordTimestamp {
    pub word: String,
    pub start_sec: f32,
    pub end_sec: f32,
}
```

### Lifecycle

```rust
// === App first launch: download all weights ===
TTSModel::preload_weights("b6369a24", Some("alba"))?;
// Downloads: TTS model (~90MB), tokenizer, voice embeddings
// Checks: WhisperX script/binary exists
// Subsequent launches: instant (cached)

// === User opens a book ===
let model = TTSModel::load_with_alignment("b6369a24")?;
let voice_state = model.get_voice_state_from_prompt_file(&voice_path)?;
// Loads TTS into memory + starts WhisperX server (~15s)

// === User reads (per sentence) ===
let result = model.generate_sentence_with_timestamps(sentence, &voice_state)?;
// result.audio: [C, T] at 24kHz
// result.word_timestamps: [{word, start_sec, end_sec}, ...]
// ~3-5s total: 1-3s TTS + 2s alignment

// === User closes book ===
drop(model);
// Server process killed, memory freed
```

### WAV Encoding

```rust
let audio_data: Vec<f32> = result.audio.flatten_all()?.to_vec1()?;
let spec = hound::WavSpec {
    channels: 1, sample_rate: 24000,
    bits_per_sample: 16, sample_format: hound::SampleFormat::Int,
};
let mut cursor = std::io::Cursor::new(Vec::new());
let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
for &s in &audio_data {
    writer.write_sample((s * 32767.0).clamp(-32768.0, 32767.0) as i16)?;
}
writer.finalize()?;
let wav_bytes = cursor.into_inner();
```

## Tauri Integration

### Tauri Commands

```rust
struct AppState {
    model: Option<TTSModel>,
    voice_state: Option<ModelState>,
}

#[tauri::command]
async fn preload(state: tauri::State<'_, Mutex<AppState>>) -> Result<(), String> {
    tokio::task::spawn_blocking(|| {
        TTSModel::preload_weights("b6369a24", Some("alba"))
    }).await.map_err(|e| e.to_string())?.map_err(|e| e.to_string())
}

#[tauri::command]
async fn open_book(state: tauri::State<'_, Mutex<AppState>>) -> Result<(), String> {
    let model = tokio::task::spawn_blocking(|| {
        TTSModel::load_with_alignment("b6369a24")
    }).await.map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;

    let voice_path = pocket_tts::weights::download_if_necessary(
        "hf://kyutai/pocket-tts-without-voice-cloning/embeddings/alba.safetensors"
    ).map_err(|e| e.to_string())?;
    let voice_state = model.get_voice_state_from_prompt_file(&voice_path)
        .map_err(|e| e.to_string())?;

    let mut s = state.lock().unwrap();
    s.model = Some(model);
    s.voice_state = Some(voice_state);
    Ok(())
}

#[tauri::command]
async fn speak_sentence(
    sentence: String,
    state: tauri::State<'_, Mutex<AppState>>,
) -> Result<SpeechResult, String> {
    let s = state.lock().unwrap();
    let model = s.model.as_ref().ok_or("Book not open")?;
    let voice = s.voice_state.as_ref().ok_or("Voice not loaded")?;

    // Clone for spawn_blocking
    let model = model.clone();
    let voice = voice.clone();

    let result = tokio::task::spawn_blocking(move || {
        model.generate_sentence_with_timestamps(&sentence, &voice)
    }).await.map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;

    // Encode audio to base64 WAV...
    Ok(SpeechResult { audio_base64, timestamps })
}

#[tauri::command]
async fn close_book(state: tauri::State<'_, Mutex<AppState>>) -> Result<(), String> {
    let mut s = state.lock().unwrap();
    s.model = None;       // Drop kills WhisperX server
    s.voice_state = None;
    Ok(())
}
```

### Frontend Highlight Loop

```javascript
const audio = new Audio(audioUrl);
audio.play();

requestAnimationFrame(function tick() {
    const t = audio.currentTime;
    for (const word of timestamps) {
        if (t >= word.start && t < word.end) {
            highlightWord(word);
            break;
        }
    }
    if (!audio.paused) requestAnimationFrame(tick);
});
```

## Performance

| Phase | Time | Notes |
|-------|------|-------|
| First launch (download) | ~30-60s | TTS weights + voice (~90MB) |
| Open book (load models) | ~15s | TTS + WhisperX models into memory |
| Per sentence | ~3-5s | 1-3s TTS + 2s alignment |
| Close book | instant | Server killed, memory freed |
| Memory (book open) | ~800MB | TTS (~300MB) + WhisperX (~500MB) |

## CI Build

GitHub Actions builds WhisperX sidecar binaries on push to `v*` tags.

Trigger manually: Actions → "Build WhisperX Sidecar" → "Run workflow"

Artifacts are uploaded to GitHub Releases as platform-specific binaries.

## Files

```
pocket-tts/
├── .github/workflows/
│   └── build-whisperx.yml          # CI: builds PyInstaller binaries
├── scripts/
│   ├── whisperx_server.py           # WhisperX HTTP server (dev)
│   ├── whisperx_align.py            # Standalone alignment script
│   ├── Dockerfile.whisperx          # Docker image for alignment
│   └── test_whisperx.py             # Comparison test script
├── crates/pocket-tts/
│   ├── src/alignment/
│   │   ├── mod.rs                   # Module declarations
│   │   ├── aligner.rs               # WhisperAligner (server client)
│   │   └── forced_align.rs          # WordTimestamp type
│   ├── src/tts_model.rs             # +preload_weights, +load_with_alignment,
│   │                                #  +generate_with_timestamps
│   └── src/lib.rs                   # Re-exports
└── docs/
    └── WORD_TIMESTAMPS_INTEGRATION.md  # This file
```
