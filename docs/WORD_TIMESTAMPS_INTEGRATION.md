# Word-Level Timestamps — Integration Guide

## Overview

pocket-tts fork (`feat/word-timestamps` branch) provides TTS audio generation with word-level timestamps for read-aloud apps. Uses WhisperX (Whisper transcription + wav2vec2-base forced alignment) for ~20-50ms accuracy. No customer-facing Python dependency — the WhisperX server is compiled into a standalone sidecar binary via CI.

## Repository Setup

```
Private repos, same owner:
  KnockOutEZ/pocket-tts          ← TTS fork (this repo)
  KnockOutEZ/vordetta-tauri      ← Tauri app (consumer)
```

### Dependency in Tauri's Cargo.toml

```toml
[dependencies]
pocket-tts = { git = "ssh://git@github.com/KnockOutEZ/pocket-tts.git", branch = "feat/word-timestamps" }
```

SSH works for private repos with your SSH key. For CI, add a deploy key or use a GitHub PAT:

```toml
# CI-friendly (uses GITHUB_TOKEN or PAT)
pocket-tts = { git = "https://github.com/KnockOutEZ/pocket-tts.git", branch = "feat/word-timestamps" }
```

### Release Workflow (Both Repos)

```
pocket-tts (TTS fork):
  1. Make changes, push to feat/word-timestamps
  2. Tag: git tag v0.7.x && git push origin v0.7.x
  3. CI builds WhisperX sidecar binaries (macOS ARM64/x64, Windows, Linux)
  4. Download artifacts from GitHub Actions

vordetta-tauri (Tauri app):
  1. cargo update -p pocket-tts  (pulls latest from git)
  2. Place whisperx_server binary in src-tauri/bin/
  3. Build Tauri app: cargo tauri build
```

When pocket-tts is stable, you can pin to a specific commit:
```toml
pocket-tts = { git = "ssh://git@github.com/KnockOutEZ/pocket-tts.git", rev = "abc1234" }
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Tauri App                         │
│                                                     │
│  App Startup                                        │
│  ├─ preload_weights()  → download all models once   │
│  └─ Show "Setting up..." UI with progress           │
│                                                     │
│  User Opens Book                                    │
│  ├─ load_with_alignment()  → load TTS + start       │
│  │   WhisperX server (~15s)                         │
│  └─ Show "Loading..." until ready                   │
│                                                     │
│  User Reads (per sentence pipeline)                 │
│  ├─ generate_sentence_with_timestamps()  (~3-5s)    │
│  ├─ Play sentence N audio                           │
│  ├─ Highlight words using timestamps                │
│  └─ While N plays, generate N+1 in background       │
│                                                     │
│  User Closes Book                                   │
│  └─ drop(model)  → kills server, frees memory       │
└─────────────────────────────────────────────────────┘

Internal flow of generate_sentence_with_timestamps():
  1. Pocket TTS generates audio (Candle, ~1-3s)
  2. Audio resampled 24kHz → 16kHz
  3. Written as temp WAV
  4. POST to WhisperX server (localhost:9876)
  5. WhisperX: Whisper transcribes → wav2vec2 force-aligns
  6. Server returns JSON word timestamps (~2s)
  7. Temp WAV cleaned up
  8. Return GenerationResult { audio, word_timestamps }
```

---

## App Lifecycle — Full Detail

### Phase 1: First App Launch (Downloads)

```rust
// Call on EVERY app startup. First run downloads ~530MB.
// Subsequent runs: instant cache check.
// Show progress UI during first run.

TTSModel::preload_weights("b6369a24", TTSModel::VOICES)?;

// This downloads (all cached after first time):
//   [1/4] TTS model weights        ~90MB   (hf-hub cache)
//   [2/4] TTS tokenizer            ~2MB    (hf-hub cache)
//   [3/4] ALL voice embeddings     ~40MB   (8 voices × ~5MB each)
//   [4/4] WhisperX models          ~435MB  (Whisper tiny.en + wav2vec2-base)
//         - whisper tiny.en         ~75MB  (torch hub cache)
//         - wav2vec2-base           ~360MB (torch hub cache)
```

Available voices: `TTSModel::VOICES` = `["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]`

### Phase 2: User Opens a Book

```rust
// Load TTS into memory + start WhisperX HTTP server.
// Takes ~15s (model loading). Show "Opening book..." UI.
// ZERO downloads — everything cached from Phase 1.

let model = TTSModel::load_with_alignment("b6369a24")?;

// Load voice for this book
let voice_path = pocket_tts::weights::download_if_necessary(
    "hf://kyutai/pocket-tts-without-voice-cloning/embeddings/alba.safetensors"
)?;
let voice_state = model.get_voice_state_from_prompt_file(&voice_path)?;
```

### Phase 3: Reading (Per-Sentence Pipeline)

```rust
// ~3-5s per sentence. Play while generating next.
let result = model.generate_sentence_with_timestamps(sentence, &voice_state)?;

// result.audio: candle_core::Tensor [C, T] at 24kHz
// result.word_timestamps: Vec<WordTimestamp>
//   each: { word: String, start_sec: f32, end_sec: f32 }
```

**Pipeline pattern:**
```
Sentence 1: generate (3s) → play + highlight
Sentence 2: generate (3s, in background while 1 plays) → play + highlight
Sentence 3: generate (3s, in background while 2 plays) → play + highlight
...
```

User hears continuous audio with synchronized word highlighting. The ~3-5s generation time is hidden by the pipeline.

### Phase 4: User Closes Book

```rust
drop(model);
// Automatically:
//   - Kills WhisperX server process
//   - Frees TTS model memory (~300MB)
//   - WhisperX memory freed when process dies (~500MB)
```

---

## Public API Reference

### Types

```rust
use pocket_tts::{TTSModel, GenerationResult, WordTimestamp, ModelState};

pub struct GenerationResult {
    pub audio: Tensor,                      // [C, T] at 24kHz
    pub word_timestamps: Vec<WordTimestamp>,
}

pub struct WordTimestamp {
    pub word: String,     // Word as transcribed by Whisper
    pub start_sec: f32,   // Start time in seconds
    pub end_sec: f32,     // End time in seconds
}
```

### Methods

```rust
impl TTSModel {
    /// All predefined voice names.
    pub const VOICES: &[&str];

    /// Download/verify all models. Call on every app startup.
    /// First run: ~530MB download. Subsequent: instant cache check.
    pub fn preload_weights(variant: &str, voices: &[&str]) -> Result<()>;

    /// Load TTS + start WhisperX server. Call when user opens a book.
    /// ~15s for model loading. Zero downloads.
    pub fn load_with_alignment(variant: &str) -> Result<Self>;

    /// Generate audio for one sentence with word timestamps. ~3-5s.
    pub fn generate_sentence_with_timestamps(
        &self, sentence: &str, voice_state: &ModelState
    ) -> Result<GenerationResult>;

    /// Generate audio for full text (splits into sentences internally).
    pub fn generate_with_timestamps(
        &self, text: &str, voice_state: &ModelState
    ) -> Result<GenerationResult>;

    // Existing methods (unchanged):
    pub fn load(variant: &str) -> Result<Self>;
    pub fn generate(&self, text: &str, voice_state: &ModelState) -> Result<Tensor>;
    pub fn get_voice_state_from_prompt_file<P>(path: P) -> Result<ModelState>;
    pub fn get_voice_state_from_bytes(bytes: &[u8]) -> Result<ModelState>;
}
```

### Audio Encoding (Tensor → WAV bytes)

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
let wav_bytes = cursor.into_inner(); // Ready for base64 or playback
```

---

## Tauri Commands (Copy-Paste Ready)

```rust
use pocket_tts::{TTSModel, ModelState};
use std::sync::Mutex;
use tauri::State;

struct AppState {
    model: Option<TTSModel>,
    voice_state: Option<ModelState>,
}

#[tauri::command]
async fn preload(state: State<'_, Mutex<AppState>>) -> Result<String, String> {
    tokio::task::spawn_blocking(|| {
        TTSModel::preload_weights("b6369a24", TTSModel::VOICES)
    }).await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;
    Ok("ready".into())
}

#[tauri::command]
async fn open_book(
    voice: String,
    state: State<'_, Mutex<AppState>>,
) -> Result<(), String> {
    let model = tokio::task::spawn_blocking(|| {
        TTSModel::load_with_alignment("b6369a24")
    }).await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;

    let voice_path = pocket_tts::weights::download_if_necessary(
        &format!("hf://kyutai/pocket-tts-without-voice-cloning/embeddings/{}.safetensors", voice)
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
    state: State<'_, Mutex<AppState>>,
) -> Result<SpeechResult, String> {
    let (model, voice) = {
        let s = state.lock().unwrap();
        (
            s.model.clone().ok_or("Book not open")?,
            s.voice_state.clone().ok_or("Voice not loaded")?,
        )
    };

    let result = tokio::task::spawn_blocking(move || {
        model.generate_sentence_with_timestamps(&sentence, &voice)
    }).await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;

    // Encode audio to WAV base64
    let audio_data: Vec<f32> = result.audio.flatten_all()
        .and_then(|t| t.to_vec1()).map_err(|e| e.to_string())?;
    let mut wav_buf = std::io::Cursor::new(Vec::new());
    let spec = hound::WavSpec {
        channels: 1, sample_rate: 24000,
        bits_per_sample: 16, sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::new(&mut wav_buf, spec).map_err(|e| e.to_string())?;
    for &s in &audio_data {
        writer.write_sample((s * 32767.0).clamp(-32768.0, 32767.0) as i16)
            .map_err(|e| e.to_string())?;
    }
    writer.finalize().map_err(|e| e.to_string())?;
    let audio_base64 = base64::engine::general_purpose::STANDARD
        .encode(wav_buf.into_inner());

    Ok(SpeechResult {
        audio_base64,
        sample_rate: 24000,
        timestamps: result.word_timestamps.into_iter().map(|t| WordTs {
            word: t.word, start: t.start_sec, end: t.end_sec,
        }).collect(),
    })
}

#[tauri::command]
async fn close_book(state: State<'_, Mutex<AppState>>) -> Result<(), String> {
    let mut s = state.lock().unwrap();
    s.model = None;       // Drops model, kills WhisperX server
    s.voice_state = None;
    Ok(())
}

#[derive(serde::Serialize)]
struct SpeechResult {
    audio_base64: String,
    sample_rate: u32,
    timestamps: Vec<WordTs>,
}

#[derive(serde::Serialize)]
struct WordTs {
    word: String,
    start: f32,
    end: f32,
}
```

### Frontend Word Highlighting

```javascript
async function readSentence(sentence) {
  const { audio_base64, timestamps } = await invoke('speak_sentence', { sentence });

  const wavBytes = Uint8Array.from(atob(audio_base64), c => c.charCodeAt(0));
  const blob = new Blob([wavBytes], { type: 'audio/wav' });
  const audio = new Audio(URL.createObjectURL(blob));
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
}
```

---

## WhisperX Sidecar Binary

### What It Is

A standalone executable (~400-500MB) containing Python + PyTorch + WhisperX. No Python installation needed on customer machines. Built by CI via PyInstaller.

### How It's Found

The aligner auto-detects backends in order:
1. `scripts/whisperx_server.py` — dev mode (requires Python + whisperx pip package)
2. `whisperx_server` binary next to running executable — shipped Tauri app
3. `whisperx_server` in PATH

### CI Build

GitHub Actions workflow: `.github/workflows/build-whisperx.yml`

Builds for: macOS ARM64, macOS x64, Windows x64, Linux x64

Trigger: push a `v*` tag or manual dispatch from Actions page.

Download artifacts from: GitHub Actions → workflow run → Artifacts section.

### Tauri Sidecar Placement

```
src-tauri/
  bin/
    whisperx_server-aarch64-apple-darwin    ← macOS ARM64
    whisperx_server-x86_64-apple-darwin     ← macOS Intel
    whisperx_server-x86_64-pc-windows-msvc.exe  ← Windows
    whisperx_server-x86_64-unknown-linux-gnu    ← Linux
```

Tauri's `tauri.conf.json`:
```json
{
  "bundle": {
    "externalBin": ["bin/whisperx_server"]
  }
}
```

---

## Performance

| Phase | Time | Memory | Notes |
|-------|------|--------|-------|
| App startup (preload) | First: ~60s download. Subsequent: <1s | Minimal | Cache check only |
| Open book | ~15s | +800MB | TTS (~300MB) + WhisperX server (~500MB) |
| Per sentence | ~3-5s | No change | 1-3s TTS + 2s alignment |
| Close book | Instant | -800MB | Server killed, models freed |

### Customer Machine Requirements

- macOS 12+, Windows 10+, or Linux (glibc 2.31+)
- 4GB RAM minimum (8GB recommended)
- 1.5GB disk for models (downloaded once, cached)
- Internet for first-run model download

---

## Files in This Fork

```
pocket-tts/
├── .github/workflows/
│   └── build-whisperx.yml              # CI: PyInstaller builds
├── scripts/
│   ├── whisperx_server.py              # WhisperX HTTP server (dev + PyInstaller source)
│   ├── whisperx_align.py               # Standalone alignment script
│   ├── Dockerfile.whisperx             # Docker fallback
│   └── test_whisperx.py                # Comparison test
├── crates/pocket-tts/
│   ├── src/alignment/
│   │   ├── mod.rs                      # Module declarations
│   │   ├── aligner.rs                  # WhisperAligner (server client)
│   │   └── forced_align.rs             # WordTimestamp type
│   ├── src/tts_model.rs                # +preload_weights, +VOICES,
│   │                                   #  +load_with_alignment,
│   │                                   #  +generate_sentence_with_timestamps,
│   │                                   #  +generate_with_timestamps
│   └── src/lib.rs                      # Re-exports
└── docs/
    └── WORD_TIMESTAMPS_INTEGRATION.md  # This file
```
