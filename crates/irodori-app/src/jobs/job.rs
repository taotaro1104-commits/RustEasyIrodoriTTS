//! ジョブ 1 件の状態（生成・声の試作・モデルの取得で共通）。

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize};
use std::sync::Arc;
use std::time::Instant;

use serde_json::Value;

/// ジョブの種類。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Generate,
    Design,
    Download,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Generate => "generate",
            Kind::Design => "design",
            Kind::Download => "download",
        }
    }
}

pub(crate) struct Job {
    pub id: String,
    pub kind: Kind,
    /// 生成ジョブの台本の ID。声の試作・モデルの取得では `None`。
    pub project: Option<String>,
    pub status: String,
    pub done: usize,
    pub total: usize,
    pub error: String,
    pub steps: usize,
    pub cancel: Arc<AtomicBool>,
    pub progress: Arc<AtomicUsize>,
    pub stage: Arc<AtomicUsize>,
    /// 止まり（長い間、ステップも段階も変わらない）を見つけるための記録。
    pub watch: Watch,
    /// まだ処理していないカードの ID（順番どおり）。
    pub pending: Vec<String>,
    /// 声の試作の候補（登録簿の鍵）。
    pub assets: Vec<String>,
    /// 声の試作の依頼（画面から受け取った値）。
    pub creation: Value,
    /// 声の試作の記録（完了後）。
    pub record: Option<Value>,
    /// モデルの取得の対象（それ以外では空）。
    pub model: String,
    /// モデルの取得のバイト数（それ以外では 0 のまま）。
    pub bytes_done: Arc<AtomicU64>,
    pub bytes_total: Arc<AtomicU64>,
}

impl Job {
    /// 待機中の新しいジョブ。種類に必要な項目は、呼び出し側が `..Job::new(..)` で残りを使って埋める。
    pub fn new(id: String, kind: Kind) -> Job {
        Job {
            id,
            kind,
            project: None,
            status: "待機中".into(),
            done: 0,
            total: 1,
            error: String::new(),
            steps: 0,
            cancel: Arc::new(AtomicBool::new(false)),
            progress: Arc::new(AtomicUsize::new(0)),
            stage: Arc::new(AtomicUsize::new(0)),
            watch: Watch::new(),
            pending: Vec::new(),
            assets: Vec::new(),
            creation: Value::Null,
            record: None,
            model: String::new(),
            bytes_done: Arc::new(AtomicU64::new(0)),
            bytes_total: Arc::new(AtomicU64::new(0)),
        }
    }
}

/// 最後に見た `(ステップ, 段階)` と、それが変わらなくなってからの時間。
pub(crate) struct Watch {
    pub seen: (usize, usize),
    pub since: Instant,
    pub warned: bool,
}

impl Watch {
    pub fn new() -> Watch {
        Watch {
            seen: (0, 0),
            since: Instant::now(),
            warned: false,
        }
    }
}
