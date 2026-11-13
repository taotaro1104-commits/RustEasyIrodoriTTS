//! 保存層のエラー。HTTP の状態コードへの対応は仕様 §5 に従う。

use thiserror::Error;

#[derive(Debug, Error)]
pub enum StoreError {
    /// 不正な入力（400）。メッセージは利用者向けの日本語。
    #[error("{0}")]
    Invalid(String),
    /// 古い revision など競合（409）。
    #[error("{0}")]
    Conflict(String),
    /// ファイル操作の失敗（400）。詳細なパスは含めない。
    #[error("{0}")]
    Io(String),
    /// 想定外（500）。詳細はログにだけ出す。
    #[error("{0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, StoreError>;

impl StoreError {
    pub fn message(&self) -> &str {
        match self {
            StoreError::Invalid(m)
            | StoreError::Conflict(m)
            | StoreError::Io(m)
            | StoreError::Internal(m) => m,
        }
    }

    pub fn status(&self) -> u16 {
        match self {
            StoreError::Invalid(_) | StoreError::Io(_) => 400,
            StoreError::Conflict(_) => 409,
            StoreError::Internal(_) => 500,
        }
    }
}

impl From<std::io::Error> for StoreError {
    fn from(_: std::io::Error) -> Self {
        StoreError::Io(
            "ファイルを読み書きできません。保存先の接続・空き容量・アクセス権を確認してください。"
                .into(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_mapping_follows_the_spec() {
        assert_eq!(StoreError::Invalid("x".into()).status(), 400);
        assert_eq!(StoreError::Conflict("x".into()).status(), 409);
        assert_eq!(StoreError::Io("x".into()).status(), 400);
        assert_eq!(StoreError::Internal("x".into()).status(), 500);
    }

    #[test]
    fn message_returns_the_original_text() {
        assert_eq!(
            StoreError::Conflict("台本が更新されています。".into()).message(),
            "台本が更新されています。"
        );
        assert_eq!(StoreError::Invalid("a".into()).to_string(), "a");
    }

    #[test]
    fn io_errors_convert_without_leaking_paths_in_the_user_message() {
        let e: StoreError = std::io::Error::new(std::io::ErrorKind::NotFound, "C:/secret/x").into();
        assert_eq!(e.status(), 400);
        assert!(!e.message().contains("secret"), "{}", e.message());
    }
}
