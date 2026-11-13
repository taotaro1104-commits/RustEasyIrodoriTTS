//! Irodori-TTS の Candle 実装（CPU）。
pub mod error;
pub mod golden;
pub mod normalize;
pub mod text_tokenizer;

pub use error::{IrodoriError, Result};
