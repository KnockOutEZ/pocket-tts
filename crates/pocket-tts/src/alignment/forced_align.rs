//! CTC forced alignment via Viterbi decoding.
//!
//! Pure-logic module — no ML frameworks, no tensors, no external deps beyond `anyhow`.

use anyhow::{bail, Result};

// ---------------------------------------------------------------------------
// wav2vec2-base-960h vocabulary constants
// ---------------------------------------------------------------------------

/// CTC blank token (padding).
pub const BLANK: usize = 0;

/// Token IDs 1–26 map to lowercase a–z.
const LETTER_A: usize = 1;

/// Word-boundary token (represents space between words).
pub const WORD_BOUNDARY: usize = 27;

/// Apostrophe token.
pub const APOSTROPHE: usize = 28;

/// Total vocabulary size for wav2vec2-base-960h.
pub const VOCAB_SIZE: usize = 32;

/// wav2vec2 produces 50 frames per second of audio.
pub const WAV2VEC2_FRAME_RATE_HZ: f32 = 50.0;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A word with its start and end time in seconds.
#[derive(Debug, Clone)]
pub struct WordTimestamp {
    pub word: String,
    pub start_sec: f32,
    pub end_sec: f32,
}

/// Tracks which slice of the CTC target sequence belongs to a single word.
#[derive(Debug, Clone, PartialEq)]
pub struct WordSpan {
    /// The *original* word from the input text (preserving case/punctuation).
    pub word: String,
    /// Inclusive start index into the target sequence.
    pub target_start: usize,
    /// Exclusive end index into the target sequence.
    pub target_end: usize,
}

// ---------------------------------------------------------------------------
// text_to_ctc_targets
// ---------------------------------------------------------------------------

/// Map a single character to its CTC token ID, or `None` if it should be dropped.
fn char_to_token(c: char) -> Option<usize> {
    match c {
        'a'..='z' => Some(LETTER_A + (c as usize - 'a' as usize)),
        'A'..='Z' => Some(LETTER_A + (c as usize - 'A' as usize)),
        '\'' => Some(APOSTROPHE),
        _ => None,
    }
}

/// Convert input text to a CTC target sequence with interleaved blanks.
///
/// Returns `(targets, word_spans)` where:
/// - `targets` is the full CTC token sequence (blanks between every pair of tokens,
///   plus leading and trailing blank).
/// - `word_spans` maps each original word to its range within `targets`.
///
/// Words whose characters are all filtered out are skipped entirely.
/// If nothing remains after filtering, both vecs are empty.
pub fn text_to_ctc_targets(text: &str) -> (Vec<usize>, Vec<WordSpan>) {
    let original_words: Vec<&str> = text.split_whitespace().collect();

    // Build per-word token lists (char_to_token handles case-insensitivity directly,
    // avoiding a to_lowercase() heap allocation per call).
    let mut word_tokens: Vec<(/* original */ &str, Vec<usize>)> = Vec::new();
    for &orig in &original_words {
        let tokens: Vec<usize> = orig.chars().filter_map(char_to_token).collect();
        if !tokens.is_empty() {
            word_tokens.push((orig, tokens));
        }
    }

    if word_tokens.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Build the flat token list with word-boundary (|) between words.
    let mut flat_tokens: Vec<usize> = Vec::new();
    // Also record (word_idx, start_in_flat, end_in_flat) for span tracking.
    let mut span_ranges: Vec<(usize, usize, usize)> = Vec::new();

    for (i, (_orig, tokens)) in word_tokens.iter().enumerate() {
        if i > 0 {
            flat_tokens.push(WORD_BOUNDARY);
        }
        let start = flat_tokens.len();
        flat_tokens.extend(tokens.iter());
        let end = flat_tokens.len();
        span_ranges.push((i, start, end));
    }

    // Now interleave blanks: blank before every token and trailing blank.
    // final length = 2 * flat_tokens.len() + 1
    let mut targets: Vec<usize> = Vec::with_capacity(2 * flat_tokens.len() + 1);
    // Map from flat_tokens index → target index (position of the actual token).
    let mut flat_to_target: Vec<usize> = Vec::with_capacity(flat_tokens.len());

    for &tok in &flat_tokens {
        targets.push(BLANK); // blank before token
        flat_to_target.push(targets.len());
        targets.push(tok);
    }
    targets.push(BLANK); // trailing blank

    // Build word spans using the flat-to-target mapping.
    let mut word_spans: Vec<WordSpan> = Vec::new();
    for &(word_idx, flat_start, flat_end) in &span_ranges {
        // target_start = index of the *blank before* the first token of this word
        let target_start = flat_to_target[flat_start] - 1;
        // target_end = index *after* the blank that follows the last token
        let last_flat = flat_end - 1;
        let target_end = flat_to_target[last_flat] + 2; // token + 1 blank after
        word_spans.push(WordSpan {
            word: word_tokens[word_idx].0.to_string(),
            target_start,
            target_end,
        });
    }

    (targets, word_spans)
}

// ---------------------------------------------------------------------------
// viterbi_forced_align
// ---------------------------------------------------------------------------

/// Viterbi DP for CTC forced alignment.
///
/// - `log_probs[t][v]` — log probability of vocabulary token `v` at time frame `t`.
/// - `targets` — CTC target sequence (with interleaved blanks).
///
/// Returns `path[t]` — the target index assigned to frame `t`.
pub fn viterbi_forced_align(log_probs: &[Vec<f32>], targets: &[usize]) -> Result<Vec<usize>> {
    let num_frames = log_probs.len();
    let num_targets = targets.len();

    if num_targets == 0 {
        bail!("target sequence is empty");
    }

    let min_frames = (num_targets + 1) / 2;
    if num_frames < min_frames {
        bail!(
            "audio too short: need at least {} frames for {} targets, got {}",
            min_frames,
            num_targets,
            num_frames,
        );
    }

    let neg_inf: f32 = f32::NEG_INFINITY;

    // score[t][s] = best log-prob ending at frame t, target index s.
    let mut score = vec![vec![neg_inf; num_targets]; num_frames];
    // back[t][s] = previous target index at t-1 that led to score[t][s].
    let mut back = vec![vec![0usize; num_targets]; num_frames];

    // --- Initialization (t = 0) ---
    score[0][0] = log_probs[0][targets[0]];
    if num_targets > 1 {
        score[0][1] = log_probs[0][targets[1]];
    }

    // --- Recurrence ---
    for t in 1..num_frames {
        for s in 0..num_targets {
            // Stay on same target.
            let mut best = score[t - 1][s];
            let mut best_prev = s;

            // Step from s-1.
            if s >= 1 && score[t - 1][s - 1] > best {
                best = score[t - 1][s - 1];
                best_prev = s - 1;
            }

            // Skip transition from s-2 (only when the intermediate is blank
            // AND the current token differs from the token two steps back).
            if s >= 2 && targets[s - 1] == BLANK && targets[s] != targets[s - 2] {
                if score[t - 1][s - 2] > best {
                    best = score[t - 1][s - 2];
                    best_prev = s - 2;
                }
            }

            score[t][s] = best + log_probs[t][targets[s]];
            back[t][s] = best_prev;
        }
    }

    // --- Termination ---
    // targets always ends with BLANK (guaranteed by text_to_ctc_targets),
    // so the final valid states are the last token (num_targets-2) or the
    // trailing blank (num_targets-1).
    let final_s = if num_targets >= 2
        && score[num_frames - 1][num_targets - 2] > score[num_frames - 1][num_targets - 1]
    {
        num_targets - 2
    } else {
        num_targets - 1
    };

    // --- Backtrack ---
    let mut path = vec![0usize; num_frames];
    path[num_frames - 1] = final_s;
    for t in (1..num_frames).rev() {
        path[t - 1] = back[t][path[t]];
    }

    Ok(path)
}

// ---------------------------------------------------------------------------
// path_to_word_timestamps
// ---------------------------------------------------------------------------

/// Convert a Viterbi path to per-word timestamps.
///
/// For each `WordSpan`, finds the first and last frames whose path target
/// index falls within `[target_start, target_end)` and points to a non-blank
/// token. Converts frame indices to seconds using `WAV2VEC2_FRAME_RATE_HZ`.
pub fn path_to_word_timestamps(
    path: &[usize],
    targets: &[usize],
    word_spans: &[WordSpan],
) -> Vec<WordTimestamp> {
    let mut timestamps: Vec<WordTimestamp> = Vec::new();

    for span in word_spans {
        let mut first_frame: Option<usize> = None;
        let mut last_frame: Option<usize> = None;

        for (frame_idx, &target_idx) in path.iter().enumerate() {
            // Path is monotonically non-decreasing — no future frame can match this span.
            if target_idx >= span.target_end {
                break;
            }
            if target_idx >= span.target_start && targets[target_idx] != BLANK {
                if first_frame.is_none() {
                    first_frame = Some(frame_idx);
                }
                last_frame = Some(frame_idx);
            }
        }

        if let (Some(first), Some(last)) = (first_frame, last_frame) {
            timestamps.push(WordTimestamp {
                word: span.word.clone(),
                start_sec: first as f32 / WAV2VEC2_FRAME_RATE_HZ,
                end_sec: (last + 1) as f32 / WAV2VEC2_FRAME_RATE_HZ,
            });
        }
    }

    timestamps
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- text_to_ctc_targets tests ------------------------------------------

    #[test]
    fn test_text_to_ctc_targets_simple() {
        let (targets, spans) = text_to_ctc_targets("hi");
        // h=8, i=9 → [blank, h, blank, i, blank]
        assert_eq!(targets, vec![BLANK, 8, BLANK, 9, BLANK]);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].word, "hi");
    }

    #[test]
    fn test_text_to_ctc_targets_two_words() {
        let (targets, spans) = text_to_ctc_targets("hi there");
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].word, "hi");
        assert_eq!(spans[1].word, "there");
        // Must contain the word-boundary token (|).
        assert!(targets.contains(&WORD_BOUNDARY));
    }

    #[test]
    fn test_text_to_ctc_targets_strips_punctuation() {
        let (targets, spans) = text_to_ctc_targets("Hello, World!");
        assert_eq!(spans.len(), 2);
        // Original words preserve case and punctuation.
        assert_eq!(spans[0].word, "Hello,");
        assert_eq!(spans[1].word, "World!");
        // No token for comma or exclamation mark — only letters/blank/boundary.
        for &t in &targets {
            assert!(t == BLANK || (t >= 1 && t <= 26) || t == WORD_BOUNDARY);
        }
    }

    #[test]
    fn test_text_to_ctc_targets_preserves_apostrophe() {
        let (targets, spans) = text_to_ctc_targets("don't");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].word, "don't");
        // Must contain the apostrophe token.
        assert!(targets.contains(&APOSTROPHE));
    }

    #[test]
    fn test_text_to_ctc_targets_empty_after_filter() {
        let (targets, spans) = text_to_ctc_targets("123 !!!");
        assert!(targets.is_empty());
        assert!(spans.is_empty());
    }

    // -- Viterbi tests ------------------------------------------------------

    /// Build synthetic log-probs where one token per frame has high probability.
    fn make_log_probs(num_frames: usize, targets: &[usize]) -> Vec<Vec<f32>> {
        let mut log_probs = vec![vec![-10.0_f32; VOCAB_SIZE]; num_frames];
        // Spread target tokens across frames uniformly.
        let step = num_frames as f32 / targets.len() as f32;
        for (s, &tok) in targets.iter().enumerate() {
            let frame = ((s as f32) * step) as usize;
            let frame = frame.min(num_frames - 1);
            log_probs[frame][tok] = -0.1; // much higher than -10
        }
        log_probs
    }

    #[test]
    fn test_viterbi_align_simple() {
        let (targets, _spans) = text_to_ctc_targets("hi");
        let log_probs = make_log_probs(10, &targets);
        let path = viterbi_forced_align(&log_probs, &targets).unwrap();
        assert_eq!(path.len(), 10);
        // Path values must be valid target indices.
        for &p in &path {
            assert!(p < targets.len());
        }
        // Path must be non-decreasing (monotonic).
        for w in path.windows(2) {
            assert!(w[1] >= w[0]);
        }
    }

    #[test]
    fn test_viterbi_repeated_chars() {
        // "ll" has targets: [blank, l, blank, l, blank]
        //                       0   1    2    3    4
        // The blank at index 2 is *mandatory* because targets[1] == targets[3] (both 'l').
        // So the s-2 skip must NOT be allowed at s=3, forcing traversal through the blank.
        let (targets, _spans) = text_to_ctc_targets("ll");
        // Verify target structure.
        assert_eq!(targets, vec![BLANK, 12, BLANK, 12, BLANK]); // l = 12
        let log_probs = make_log_probs(10, &targets);
        let path = viterbi_forced_align(&log_probs, &targets).unwrap();
        assert_eq!(path.len(), 10);
        // The blank at position 2 must appear in the path at least once,
        // because skipping it is not allowed for repeated characters.
        assert!(
            path.contains(&2),
            "mandatory blank at target index 2 must be traversed for repeated chars"
        );
    }

    #[test]
    fn test_viterbi_too_few_frames() {
        // targets = [blank, h, blank, i, blank] → 5 targets → need ceil(5/2) = 3 frames
        let (targets, _spans) = text_to_ctc_targets("hi");
        let log_probs = make_log_probs(1, &targets); // only 1 frame
        let result = viterbi_forced_align(&log_probs, &targets);
        assert!(result.is_err());
    }

    // -- Timestamp test -----------------------------------------------------

    #[test]
    fn test_path_to_word_timestamps() {
        let (targets, spans) = text_to_ctc_targets("hi there");
        let log_probs = make_log_probs(20, &targets);
        let path = viterbi_forced_align(&log_probs, &targets).unwrap();
        let timestamps = path_to_word_timestamps(&path, &targets, &spans);

        assert_eq!(timestamps.len(), 2);
        assert_eq!(timestamps[0].word, "hi");
        assert_eq!(timestamps[1].word, "there");
        // Timestamps must be non-negative and ordered.
        assert!(timestamps[0].start_sec >= 0.0);
        assert!(timestamps[0].end_sec > timestamps[0].start_sec);
        assert!(timestamps[1].start_sec >= timestamps[0].start_sec);
        assert!(timestamps[1].end_sec > timestamps[1].start_sec);
    }
}
