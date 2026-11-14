//! 本物の導入。Hugging Face から取得し、HF キャッシュ互換のフォルダへ置く。

use std::path::PathBuf;
use std::sync::atomic::Ordering;

use super::{
    approximate_bytes, install_model, DownloadControl, DownloadError, Hub, ModelInstaller, Transfer,
};

pub struct RealInstaller {
    hub_dir: PathBuf,
    hub: Hub,
}

impl RealInstaller {
    pub fn new(hub_dir: PathBuf) -> RealInstaller {
        RealInstaller::with_hub(hub_dir, Hub::huggingface())
    }

    /// 取得元を差し替える（テスト用）。
    pub fn with_hub(hub_dir: PathBuf, hub: Hub) -> RealInstaller {
        RealInstaller { hub_dir, hub }
    }
}

impl ModelInstaller for RealInstaller {
    fn install(&self, model: &str, control: &DownloadControl) -> Result<(), DownloadError> {
        let done = &control.done;
        let progress = |n: u64| {
            done.fetch_add(n, Ordering::SeqCst);
        };
        let on_total = |n: u64| control.total.store(n, Ordering::SeqCst);
        install_model(
            &self.hub,
            &self.hub_dir,
            model,
            &Transfer {
                cancel: &control.cancel,
                progress: &progress,
            },
            &on_total,
        )
    }

    fn approximate_bytes(&self, model: &str) -> u64 {
        approximate_bytes(&self.hub_dir, model)
    }
}
