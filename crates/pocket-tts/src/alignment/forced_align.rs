/// A word with its start and end time in seconds.
#[derive(Debug, Clone)]
pub struct WordTimestamp {
    pub word: String,
    pub start_sec: f32,
    pub end_sec: f32,
}
