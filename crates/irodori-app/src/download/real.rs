//! 本物の導入。Hugging Face から取得し、HF キャッシュ互換のフォルダへ置く。

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::{
    approximate_bytes, install_model, DownloadControl, DownloadError, Hub, ModelInstaller, Transfer,
};

/// 概算を使い回す時間（画面の状態の取得のたびに、フォルダを調べ直さないため）。
const ESTIMATE_TTL: Duration = Duration::from_secs(5);

pub struct RealInstaller {
    hub_dir: PathBuf,
    hub: Hub,
    estimates: Mutex<HashMap<String, (Instant, u64)>>,
}

impl RealInstaller {
    pub fn new(hub_dir: PathBuf) -> RealInstaller {
        RealInstaller::with_hub(hub_dir, Hub::huggingface())
    }

    /// 取得元を差し替える（テスト用）。
    pub fn with_hub(hub_dir: PathBuf, hub: Hub) -> RealInstaller {
        RealInstaller {
            hub_dir,
            hub,
            estimates: Mutex::new(HashMap::new()),
        }
    }
}

impl ModelInstaller for RealInstaller {
    fn install(&self, model: &str, control: &DownloadControl) -> Result<(), DownloadError> {
        let done = &control.done;
        let progress = |n: u64| {
            done.fetch_add(n, Ordering::SeqCst);
        };
        let on_total = |n: u64| control.total.store(n, Ordering::SeqCst);
        let outcome = install_model(
            &self.hub,
            &self.hub_dir,
            model,
            &Transfer {
                cancel: &control.cancel,
                progress: &progress,
            },
            &on_total,
        );
        // 結果にかかわらず（途中まで置けた分も）、次の概算は調べ直す。
        self.estimates
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        outcome
    }

    fn approximate_bytes(&self, model: &str) -> u64 {
        let mut cache = self.estimates.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((at, bytes)) = cache.get(model) {
            if at.elapsed() < ESTIMATE_TTL {
                return *bytes;
            }
        }
        let bytes = approximate_bytes(&self.hub_dir, model);
        cache.insert(model.to_string(), (Instant::now(), bytes));
        bytes
    }
}
