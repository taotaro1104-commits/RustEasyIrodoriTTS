use thiserror::Error;

/// ライブラリ共通エラー。各モジュールが variant を追加する。
#[derive(Debug, Error)]
pub enum IrodoriError {
    #[error("tensor error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("text is empty after normalization")]
    EmptyText,
    #[error("text too long: {tokens} tokens (including BOS) > max {max}")]
    TextTooLong { tokens: usize, max: usize },
    #[error("golden mismatch: {0}")]
    GoldenMismatch(String),
    #[error("golden file not found: {0}")]
    GoldenMissing(String),
}

pub type Result<T> = std::result::Result<T, IrodoriError>;
