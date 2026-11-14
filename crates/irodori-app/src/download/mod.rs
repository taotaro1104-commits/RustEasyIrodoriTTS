//! モデルの自動ダウンロード（確認のあとにだけ使う）。許可リストのモデルを Hugging Face から取得し、
//! HF キャッシュ互換のフォルダへ置く。

mod error;
mod fetch;
mod hub;

pub use error::DownloadError;
pub use fetch::{partial_path, Transfer};
pub use hub::{host_allowed, HostPolicy, Hub, PlannedFile, RepoPlan, HF_BASE};
