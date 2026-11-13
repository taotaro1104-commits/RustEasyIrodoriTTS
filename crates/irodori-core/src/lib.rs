//! Irodori-TTS の Candle 実装（CPU）。
pub mod error;
pub mod attention;
pub mod codec;
pub mod conditioner;
pub mod config;
pub mod dit;
pub mod duration;
pub mod golden;
pub mod joint_attention;
pub mod loader;
pub mod nn;
pub mod speaker_encoder;
pub mod text_encoder;
pub mod normalize;
pub mod text_tokenizer;

pub use error::{IrodoriError, Result};
