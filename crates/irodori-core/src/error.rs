use thiserror::Error;

/// ライブラリ共通エラー。各モジュールが variant を追加する。
#[derive(Debug, Error)]
pub enum IrodoriError {
    #[error("tensor error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("golden mismatch: {0}")]
    GoldenMismatch(String),
    #[error("golden file not found: {0}")]
    GoldenMissing(String),
}

pub type Result<T> = std::result::Result<T, IrodoriError>;
