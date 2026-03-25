//! Read-aloud demo: rich text editor with word-level highlight sync.
//!
//! Run: cargo run -p pocket-tts-cli --example readaloud_demo
//! Then open http://localhost:3033

use anyhow::Result;
use axum::{Json, Router, extract::State, http::StatusCode, response::Html, routing::{get, post}};
use base64::Engine;
use pocket_tts::TTSModel;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::CorsLayer;

const VOICE_REPO: &str = "kyutai/pocket-tts-without-voice-cloning";

struct AppState {
    model: TTSModel,
    voice_state: pocket_tts::ModelState,
}

#[derive(Deserialize)]
struct TtsRequest {
    text: String,
}

#[derive(Serialize)]
struct TtsResponse {
    audio_base64: String,
    timestamps: Vec<WordTs>,
    sample_rate: u32,
}

#[derive(Serialize)]
struct WordTs {
    word: String,
    start: f32,
    end: f32,
}

#[tokio::main]
async fn main() -> Result<()> {
    eprintln!("Loading TTS + alignment models...");
    let model = TTSModel::load_with_alignment("b6369a24")?;

    eprintln!("Loading alba voice...");
    let voice_path = pocket_tts::weights::download_if_necessary(&format!(
        "hf://{}/embeddings/alba.safetensors",
        VOICE_REPO
    ))?;
    let voice_state = model.get_voice_state_from_prompt_file(&voice_path)?;

    let state = Arc::new(AppState { model, voice_state });

    let app = Router::new()
        .route("/", get(index))
        .route("/api/tts", post(generate))
        .layer(CorsLayer::permissive())
        .with_state(state);

    eprintln!("\n  Ready at http://localhost:3033\n");
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3033").await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TtsRequest>,
) -> Result<Json<TtsResponse>, StatusCode> {
    let text = req.text.trim().to_string();
    if text.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Run generation on blocking thread (CPU-bound)
    let state = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        state.model.generate_with_timestamps(&text, &state.voice_state)
    })
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    .map_err(|e| {
        eprintln!("Generation error: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Encode audio as WAV base64
    let audio_data = result.audio.flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let mut wav_buf = std::io::Cursor::new(Vec::new());
    {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 24000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::new(&mut wav_buf, spec)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        for &s in &audio_data {
            writer.write_sample(s).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        }
        writer.finalize().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    }

    let audio_base64 = base64::engine::general_purpose::STANDARD.encode(wav_buf.into_inner());

    let timestamps: Vec<WordTs> = result.word_timestamps.into_iter().map(|t| WordTs {
        word: t.word,
        start: t.start_sec,
        end: t.end_sec,
    }).collect();

    Ok(Json(TtsResponse {
        audio_base64,
        timestamps,
        sample_rate: 24000,
    }))
}

async fn index() -> Html<&'static str> {
    Html(include_str!("readaloud_ui.html"))
}
