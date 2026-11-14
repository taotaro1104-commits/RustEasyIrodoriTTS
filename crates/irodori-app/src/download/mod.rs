//! モデルの自動ダウンロード（確認のあとにだけ使う）。許可リストのモデルを Hugging Face から取得し、
//! HF キャッシュ互換のフォルダへ置く。

mod error;
mod fetch;
mod hub;
mod install;
mod installer;
mod real;

pub use error::DownloadError;
pub use fetch::{partial_path, Transfer};
pub use hub::{host_allowed, HostPolicy, Hub, PlannedFile, RepoPlan, HF_BASE};
pub use install::{
    approximate_bytes, install_model, install_with_space, APPROX_CODEC_BYTES, APPROX_MODEL_BYTES,
    CODEC_FILES, MODEL_FILES,
};
pub use installer::{DownloadControl, ModelInstaller};
pub use real::RealInstaller;
