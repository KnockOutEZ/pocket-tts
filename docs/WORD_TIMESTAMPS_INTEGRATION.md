# Word-Level Timestamps — Integration Guide

## What Was Added

The `pocket-tts` fork on branch `feat/word-timestamps` adds word-level timestamp support for read-aloud / word-highlight sync. It uses a bundled `whisper-cli` binary (whisper.cpp) for alignment — same algorithm as OpenAI's `whisper --word_timestamps True`, ~20-50ms accuracy. No Python required.

## Branch

```
repo: KnockOutEZ/pocket-tts
branch: feat/word-timestamps
```

## Setup: whisper-cli Binary

The aligner needs a `whisper-cli` binary (from whisper.cpp). Build once:

```bash
git clone --depth 1 https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
cmake -B build -DCMAKE_CROSSCOMPILING=TRUE -DGGML_METAL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

Place the binary where pocket-tts can find it:
```bash
# Option A: In the pocket-tts crate
cp build/bin/whisper-cli /path/to/pocket-tts/crates/pocket-tts/bin/

# Option B: Next to your Tauri app binary (sidecar)
cp build/bin/whisper-cli /path/to/tauri-app/src-tauri/bin/

# Option C: In PATH
cp build/bin/whisper-cli /usr/local/bin/
```

The ggml model (~140MB) downloads automatically from HuggingFace on first use.

## Public API

### Types

```rust
use pocket_tts::{TTSModel, GenerationResult, WordTimestamp, ModelState};

pub struct GenerationResult {
    pub audio: Tensor,                      // [C, T] audio at 24kHz
    pub word_timestamps: Vec<WordTimestamp>, // one per word
}

pub struct WordTimestamp {
    pub word: String,     // the word as transcribed by Whisper
    pub start_sec: f32,   // start time in seconds
    pub end_sec: f32,     // end time in seconds
}
```

### Loading

```rust
// Loads TTS model (~90MB) + Whisper aligner (finds whisper-cli binary + downloads ~140MB ggml model)
let model = TTSModel::load_with_alignment("b6369a24")?;
```

### Voice State (unchanged)

```rust
let voice_path = pocket_tts::weights::download_if_necessary(
    "hf://kyutai/pocket-tts-without-voice-cloning/embeddings/alba.safetensors"
)?;
let voice_state = model.get_voice_state_from_prompt_file(&voice_path)?;
```

### Generating Audio + Timestamps

**Option A: Full text (splits into sentences internally)**
```rust
let result = model.generate_with_timestamps("Hello world. This is a test.", &voice_state)?;
```

**Option B: Single sentence (caller controls chunking — recommended)**
```rust
let result = model.generate_sentence_with_timestamps("Hello world.", &voice_state)?;
```

### Converting Audio to WAV Bytes

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

### Sidecar Setup

Place `whisper-cli` in `src-tauri/bin/`:
```
src-tauri/
  bin/
    whisper-cli          # macOS ARM64
    whisper-cli.exe      # Windows (build separately)
```

The aligner auto-discovers the binary next to the running executable.

### Tauri Command

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

    // Encode audio + return timestamps...
    Ok(SpeechResult { audio_base64, timestamps })
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

- **TTS generation**: ~1-3s per sentence (CPU release mode)
- **Whisper alignment**: ~1-2s per sentence (whisper.cpp is fast)
- **Total**: ~2-5s per sentence
- **First run**: Downloads ~140MB ggml model (cached after)
- **Memory**: ~300MB for TTS + ~200MB for Whisper model

## Architecture

```
pocket-tts crate
├── TTSModel (Pocket TTS via Candle)
│   └── generate() → audio Tensor
├── WhisperAligner (whisper-cli sidecar)
│   ├── whisper-cli binary (2.3MB, bundled)
│   ├── ggml-base.en model (140MB, downloaded)
│   └── align(audio, text) → Vec<WordTimestamp>
└── generate_with_timestamps()
    ├── generate audio
    ├── resample 24kHz → 16kHz
    ├── write temp WAV
    ├── run whisper-cli --output-json-full --prompt "known text"
    ├── parse per-token timestamps from JSON
    ├── group BPE tokens into words
    └── return GenerationResult { audio, word_timestamps }
```

## Files

```
crates/pocket-tts/
├── bin/
│   └── whisper-cli          # Bundled binary (not in git, build from whisper.cpp)
├── src/alignment/
│   ├── mod.rs               # Module declarations
│   ├── aligner.rs           # WhisperAligner (whisper-cli sidecar)
│   └── forced_align.rs      # WordTimestamp type
├── src/tts_model.rs         # +GenerationResult, +load_with_alignment,
│                            #  +generate_with_timestamps,
│                            #  +generate_sentence_with_timestamps
└── src/lib.rs               # Re-exports
```
