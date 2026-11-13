//! Irodori-TTS の Candle 実装（CPU）。
pub mod error;
pub mod config;
pub mod golden;
pub mod loader;
pub mod nn;
pub mod text_encoder;
pub mod normalize;
pub mod text_tokenizer;

pub use error::{IrodoriError, Result};
