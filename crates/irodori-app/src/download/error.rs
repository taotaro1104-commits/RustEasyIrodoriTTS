//! 取得の失敗。利用者に見せる日本語は `message()`、詳細は `Display`（ログ用）。

use std::fmt;

/// 想定外の失敗に画面へ出す文（詳細はログにだけ出す）。
const GENERIC: &str = "処理に失敗しました。logs/editor.logを確認してください。";

#[derive(Debug, Clone, PartialEq)]
pub enum DownloadError {
    /// 接続できない（DNS・接続拒否・タイムアウト・TLS）。
    Offline,
    /// 取得元が返した HTTP の状態コード。
    Http(u16),
    /// サイズまたは SHA-256 が合わない。
    Verification,
    /// 空き容量が足りない（`needed` は必要なバイト数、`available` は空き）。
    DiskSpace { needed: u64, available: u64 },
    /// 利用者が中止した。
    Cancelled,
    /// 同じモデル置き場へ、ほかの取得が進んでいる。
    Locked,
    /// 取得元の応答が想定と違う（詳細はログ用）。
    Protocol(String),
    /// ファイルの読み書きの失敗（詳細はログ用）。
    Io(String),
}

impl DownloadError {
    /// 画面に出す日本語。`Protocol` と `Io` の詳細は含めない。
    pub fn message(&self) -> String {
        match self {
            DownloadError::Offline => {
                "ネットワークに接続できません。接続を確認してもう一度お試しください。".into()
            }
            DownloadError::Http(code) => {
                format!("モデルの取得元から取得できませんでした（HTTP {code}）。")
            }
            DownloadError::Verification => {
                "ダウンロードしたファイルの検証に失敗しました。もう一度お試しください。".into()
            }
            DownloadError::DiskSpace { needed, available } => {
                let shortfall = needed.saturating_sub(*available) as f64 / 1e9;
                format!("空き容量が足りません。あと約{shortfall:.1}GB 必要です。")
            }
            DownloadError::Locked => {
                "ほかの取得がこのモデルの置き場所で進んでいます。終わってからもう一度お試しください。"
                    .into()
            }
            DownloadError::Cancelled => "取得を中止しました。".into(),
            DownloadError::Protocol(_) | DownloadError::Io(_) => GENERIC.into(),
        }
    }
}

impl fmt::Display for DownloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DownloadError::Protocol(d) => write!(f, "protocol error: {d}"),
            DownloadError::Io(d) => write!(f, "io error: {d}"),
            other => write!(f, "{}", other.message()),
        }
    }
}

impl std::error::Error for DownloadError {}

impl From<std::io::Error> for DownloadError {
    fn from(e: std::io::Error) -> Self {
        DownloadError::Io(format!("{:?}: {e}", e.kind()))
    }
}
