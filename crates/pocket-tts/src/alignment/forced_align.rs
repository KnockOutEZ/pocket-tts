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
}
