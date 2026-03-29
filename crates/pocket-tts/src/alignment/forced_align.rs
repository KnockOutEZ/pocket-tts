/// A word with its start and end time in seconds.
#[derive(Debug, Clone)]
pub struct WordTimestamp {
    pub word: String,
    pub start_sec: f32,
    pub end_sec: f32,
}

/// Convert text to CTC target token IDs with interleaved blanks.
///
/// "hi" with blank_id=0 → [0, h_id, 0, i_id, 0]
/// Blanks between every character allow the model to absorb pauses/silence.
pub fn text_to_ctc_targets(text: &str, vocab: &[String], blank_id: usize) -> Vec<usize> {
    // Build char-to-index lookup
    let char_to_id: std::collections::HashMap<String, usize> = vocab
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i))
        .collect();

    let mut targets = vec![blank_id];

    for ch in text.chars() {
        let key = ch.to_lowercase().to_string();
        if let Some(&id) = char_to_id.get(&key) {
            targets.push(id);
            targets.push(blank_id);
        }
        // Unknown chars are silently skipped
    }

    // Ensure at least one blank if text was empty
    if targets.is_empty() {
        targets.push(blank_id);
    }

    targets
}

/// Viterbi CTC forced alignment.
///
/// Given emission log-probabilities (T frames x vocab_size) and a CTC target
/// sequence (with interleaved blanks), finds the optimal frame-to-token alignment.
///
/// Returns a Vec of length T where each element is the target token ID at that frame.
pub fn viterbi_forced_align(
    emissions: &[Vec<f32>],
    targets: &[usize],
) -> anyhow::Result<Vec<usize>> {
    let t_len = emissions.len();
    let s_len = targets.len();

    if t_len < s_len {
        anyhow::bail!(
            "Cannot align: {} audio frames < {} target tokens. Audio too short for text.",
            t_len, s_len
        );
    }
    if s_len == 0 {
        return Ok(vec![]);
    }

    // Trellis: trellis[t][s] = best log-prob of being at target s at frame t
    let neg_inf = f32::NEG_INFINITY;
    let mut trellis = vec![vec![neg_inf; s_len]; t_len];

    // Initialize: first frame can only be at target 0 or 1
    trellis[0][0] = emissions[0][targets[0]];
    if s_len > 1 {
        trellis[0][1] = emissions[0][targets[1]];
    }

    // Forward pass
    for t in 1..t_len {
        for s in 0..s_len {
            let emit = emissions[t][targets[s]];

            // Transition 1: stay at same target
            let stay = trellis[t - 1][s];

            // Transition 2: advance from previous target
            let advance = if s > 0 { trellis[t - 1][s - 1] } else { neg_inf };

            // Transition 3: skip blank (s-2 → s), only when:
            // - s >= 2
            // - current target is not blank
            // - current target != target at s-2 (prevents skipping blank between repeated chars)
            let skip = if s >= 2
                && targets[s] != targets[0]  // current is not blank (targets[0] is always blank_id)
                && targets[s] != targets[s - 2]
            {
                trellis[t - 1][s - 2]
            } else {
                neg_inf
            };

            trellis[t][s] = emit + stay.max(advance).max(skip);
        }
    }

    // Backtrack from the best of the last two targets (last blank or last char)
    let mut path = vec![0usize; t_len];
    let mut s = if s_len >= 2 && trellis[t_len - 1][s_len - 1] >= trellis[t_len - 1][s_len - 2] {
        s_len - 1
    } else if s_len >= 2 {
        s_len - 2
    } else {
        s_len - 1
    };
    path[t_len - 1] = targets[s];

    for t in (0..t_len - 1).rev() {
        let stay = trellis[t][s];
        let advance = if s > 0 { trellis[t][s - 1] } else { neg_inf };
        let skip = if s >= 2
            && targets[s] != targets[0]
            && targets[s] != targets[s - 2]
        {
            trellis[t][s - 2]
        } else {
            neg_inf
        };

        // Priority must mirror forward pass: stay.max(advance).max(skip)
        // means skip wins ties with advance, advance wins ties with stay.
        // Guard against NEG_INFINITY to avoid false transitions.
        if skip > advance && skip > stay && skip != neg_inf && s >= 2 {
            s -= 2;
        } else if advance > stay && advance != neg_inf && s > 0 {
            s -= 1;
        }
        // else: stay (s unchanged)

        path[t] = targets[s];
    }

    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_vocab() -> Vec<String> {
        // Minimal vocab: <pad>=0, a-z=1-26, space=27, apostrophe=28
        let mut v = vec!["<pad>".to_string()];
        for c in 'a'..='z' {
            v.push(c.to_string());
        }
        v.push(" ".to_string()); // index 27
        v.push("'".to_string()); // index 28
        v
    }

    #[test]
    fn test_ctc_targets_simple() {
        let vocab = test_vocab();
        let targets = text_to_ctc_targets("hi", &vocab, 0);
        // h=8, i=9 → [0, 8, 0, 9, 0]
        assert_eq!(targets, vec![0, 8, 0, 9, 0]);
    }

    #[test]
    fn test_ctc_targets_with_space() {
        let vocab = test_vocab();
        let targets = text_to_ctc_targets("a b", &vocab, 0);
        // a=1, space=27, b=2 → [0, 1, 0, 27, 0, 2, 0]
        assert_eq!(targets, vec![0, 1, 0, 27, 0, 2, 0]);
    }

    #[test]
    fn test_ctc_targets_repeated_chars() {
        let vocab = test_vocab();
        let targets = text_to_ctc_targets("ll", &vocab, 0);
        // l=12, l=12 → [0, 12, 0, 12, 0]
        assert_eq!(targets, vec![0, 12, 0, 12, 0]);
    }

    #[test]
    fn test_ctc_targets_apostrophe() {
        let vocab = test_vocab();
        let targets = text_to_ctc_targets("i'm", &vocab, 0);
        // i=9, apostrophe=28, m=13 → [0, 9, 0, 28, 0, 13, 0]
        assert_eq!(targets, vec![0, 9, 0, 28, 0, 13, 0]);
    }

    #[test]
    fn test_ctc_targets_single_char() {
        let vocab = test_vocab();
        let targets = text_to_ctc_targets("a", &vocab, 0);
        assert_eq!(targets, vec![0, 1, 0]);
    }

    #[test]
    fn test_ctc_targets_empty() {
        let vocab = test_vocab();
        let targets = text_to_ctc_targets("", &vocab, 0);
        assert_eq!(targets, vec![0]); // just one blank
    }

    #[test]
    fn test_ctc_targets_unknown_char_skipped() {
        let vocab = test_vocab();
        // '!' is not in vocab — should be skipped
        let targets = text_to_ctc_targets("a!b", &vocab, 0);
        assert_eq!(targets, vec![0, 1, 0, 2, 0]);
    }

    #[test]
    fn test_viterbi_simple_two_chars() {
        // Synthetic emissions for "ab": targets = [blank, a, blank, b, blank]
        // 10 frames, vocab_size = 3 (blank=0, a=1, b=2)
        let t = 10;
        let vocab_size = 3;
        let mut emissions = vec![vec![-10.0f32; vocab_size]; t];

        // Make blank dominant for frames 0-2
        for f in 0..3 { emissions[f][0] = -0.1; }
        // Make 'a' dominant for frames 3-5
        for f in 3..6 { emissions[f][1] = -0.1; }
        // Make blank dominant for frames 6-7
        for f in 6..8 { emissions[f][0] = -0.1; }
        // Make 'b' dominant for frames 8-9
        for f in 8..10 { emissions[f][2] = -0.1; }

        let targets = vec![0, 1, 0, 2, 0]; // blank, a, blank, b, blank
        let path = viterbi_forced_align(&emissions, &targets).unwrap();

        assert_eq!(path.len(), t);
        assert_eq!(path[0], 0); // blank
        assert_eq!(path[4], 1); // 'a' region
        assert_eq!(path[7], 0); // blank
        assert_eq!(path[9], 2); // 'b' region
    }

    #[test]
    fn test_viterbi_repeated_chars() {
        // "ll": targets = [blank, l, blank, l, blank]
        // Must pass through the intervening blank between the two l's
        let t = 8;
        let vocab_size = 2; // blank=0, l=1
        let mut emissions = vec![vec![-10.0f32; vocab_size]; t];

        // First 'l': frames 0-2
        for f in 0..3 { emissions[f][1] = -0.1; }
        // Blank: frames 3-4
        for f in 3..5 { emissions[f][0] = -0.1; }
        // Second 'l': frames 5-7
        for f in 5..8 { emissions[f][1] = -0.1; }

        let targets = vec![0, 1, 0, 1, 0];
        let path = viterbi_forced_align(&emissions, &targets).unwrap();

        assert_eq!(path.len(), t);
        let has_blank_between = path[3..5].iter().any(|&p| p == 0);
        assert!(has_blank_between, "Must pass through blank between repeated chars");
    }

    #[test]
    fn test_viterbi_too_few_frames() {
        // 2 frames but 5 targets — impossible
        let emissions = vec![vec![-1.0f32; 3]; 2];
        let targets = vec![0, 1, 0, 2, 0];
        let result = viterbi_forced_align(&emissions, &targets);
        assert!(result.is_err());
    }
}
