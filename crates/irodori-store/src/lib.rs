//! v2.0 互換の保存層・台本操作・辞書・分割・署名（HTTP 非依存）。
pub mod assets;
pub mod dictionary;
pub mod error;
pub mod generation;
pub mod history;
pub mod model;
pub mod naming;
pub mod ops;
pub mod py_json;
pub mod signature;
pub mod splitter;
pub mod state;
pub mod storage;
pub mod unicode_tables;

#[cfg(test)]
pub(crate) mod testutil;

pub use error::{Result, StoreError};
pub use storage::{ProjectSummary, Store};
