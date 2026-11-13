//! 音声合成エンジンとの境界。本物（`real_synth`）と、テスト用の擬似（`fake`）が同じ形で差し替わる。

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::Arc;

use irodori_store::generation::Generation;
use serde_json::Value;

/// 1 回の合成の依頼。
#[derive(Debug, Clone)]
pub struct SynthJob {
    pub text: String,
    /// 許可リストのモデル ID。
    pub model: String,
    pub caption: String,
    /// 参照音声（話者の WAV）。無ければ参照なし。
    pub references: Vec<PathBuf>,
    /// 検証済みの生成設定（`validate_generation` の結果）。
    pub generation: Generation,
    /// 候補の数（1〜8）。シードを変えて順に作る。
    pub candidates: usize,
    /// 立てると、次のステップの先頭で止まる。
    pub cancel: Arc<AtomicBool>,
    /// 実行中の候補の完了ステップ数。
    pub progress: Arc<AtomicUsize>,
    /// いまの段階（`irodori_core::control::Stage` の値）。止まったときにどこかを知るために使う。
    pub stage: Arc<AtomicUsize>,
}

impl SynthJob {
    pub fn new(text: &str, model: &str, generation: Generation) -> SynthJob {
        SynthJob {
            text: text.into(),
            model: model.into(),
            caption: String::new(),
            references: Vec::new(),
            generation,
            candidates: 1,
            cancel: Arc::new(AtomicBool::new(false)),
            progress: Arc::new(AtomicUsize::new(0)),
            stage: Arc::new(AtomicUsize::new(0)),
        }
    }
}

/// 合成の結果。`paths` は絶対パスの 24bit WAV（話速・無音は未適用）。
#[derive(Debug, Clone)]
pub struct SynthRecord {
    /// `outputs/internal/<id>/` のフォルダ名。
    pub id: String,
    pub model: String,
    pub text: String,
    pub caption: String,
    pub references: Vec<String>,
    pub generation: Value,
    pub seed: u64,
    pub paths: Vec<PathBuf>,
    pub seconds: f64,
    pub messages: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SynthError {
    /// 取消が要求された。
    Cancelled,
    /// 利用者に見せてよい日本語の理由。
    Invalid(String),
    /// 想定外の失敗（詳細はログにだけ出す）。
    Failed(String),
}

pub trait Synthesizer: Send + Sync {
    fn synthesize(&self, job: &SynthJob) -> Result<SynthRecord, SynthError>;
    /// 読み込み済みのモデルを解放する。
    fn unload(&self);
    fn is_loaded(&self) -> bool;
    /// そのモデルのファイルがそろっていて、すぐ使えるか。
    fn model_available(&self, model: &str) -> bool;
}
