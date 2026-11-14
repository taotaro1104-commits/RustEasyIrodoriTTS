//! モデル導入の境界。本物（`RealInstaller`）と、テスト用の擬似（`FakeInstaller`）が同じ形で差し替わる。

use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::Arc;

use super::DownloadError;

/// 実行中の取得と、画面（ジョブ）をつなぐ共有の値。
#[derive(Clone)]
pub struct DownloadControl {
    /// 立てると、次のかたまりの前で止まる（書きかけは再開用に残る）。
    pub cancel: Arc<AtomicBool>,
    /// これまでに取得したバイト数。
    pub done: Arc<AtomicU64>,
    /// 取得するバイト数の合計（計画が分かるまでは 0）。
    pub total: Arc<AtomicU64>,
}

impl DownloadControl {
    pub fn new() -> DownloadControl {
        DownloadControl {
            cancel: Arc::new(AtomicBool::new(false)),
            done: Arc::new(AtomicU64::new(0)),
            total: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl Default for DownloadControl {
    fn default() -> Self {
        DownloadControl::new()
    }
}

pub trait ModelInstaller: Send + Sync {
    /// モデル（と、無ければ共通の codec）を取得して使える状態にする。確認はここより前に済んでいること。
    fn install(&self, model: &str, control: &DownloadControl) -> Result<(), DownloadError>;
    /// これから取得するバイト数の概算（画面の確認用）。取得済みでも 0 とは限らない（呼び出し側が取得済みかを見る）。
    fn approximate_bytes(&self, model: &str) -> u64;
}
