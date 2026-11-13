//! v2.0 互換の保存層・台本操作・辞書・分割・署名（HTTP 非依存）。
pub mod error;

#[cfg(test)]
pub(crate) mod testutil;

pub use error::{Result, StoreError};
