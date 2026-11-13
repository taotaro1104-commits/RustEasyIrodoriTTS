//! 生成ジョブ（v2.0 の `Service.submit / run / stop` の移植）。
//!
//! 生成は専用のワーカースレッド 1 本で直列に実行する。要求は提出時にスナップショットし、カードごとに状態を保存する。
//! ロックの規則: ジョブ表の `Mutex` を持ったまま、`Store` や `Synthesizer` を呼ばない。

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use irodori_store::assets::Assets;
use irodori_store::generation::Generation;
use irodori_store::{Result, Store, StoreError};
use serde_json::{json, Value};

use irodori_core::control::Stage;

use crate::synth::Synthesizer;

pub use design::{DesignMode, DesignRequest, REGISTRATION_SAMPLE_TEXT};

mod design;
#[cfg(test)]
mod design_tests;
mod run;
mod submit;
#[cfg(test)]
mod tests;

/// 画面に返す履歴として残す、終わったジョブの上限。
const KEEP_JOBS: usize = 50;
/// これだけの間ステップも段階も変わらない生成を「止まり」としてログに出す。
const STALL_AFTER: Duration = Duration::from_secs(120);
const WATCH_EVERY: Duration = Duration::from_secs(15);
/// まだ終わっていないジョブの状態。
pub(crate) const BUSY: [&str; 2] = ["待機中", "生成中"];
/// 想定外の失敗でカードに表示する文（詳細はログにだけ出す）。
pub(crate) const GENERIC_FAILURE: &str = "処理に失敗しました。logs/editor.logを確認してください。";

#[derive(Debug, Clone, PartialEq)]
pub enum SubmitOutcome {
    Started,
    /// 25 秒超（推定）または 180 文字超のカードがあるので、確認が要る（カードの番号は 1 始まり）。
    NeedsLongConfirmation {
        long_cards: Vec<usize>,
    },
}

/// 提出時点のスナップショット 1 件（カード 1 枚分）。
pub(crate) struct Request {
    pub card_id: String,
    pub signature: String,
    /// 読み辞書を適用したあとの、合成する文章。
    pub text: String,
    pub model: String,
    pub caption: String,
    pub reference: PathBuf,
    pub generation: Generation,
    pub silence: f64,
    /// `result.request` にそのまま保存する、v2.0 と同じ形の記録。
    pub record: Value,
}

/// ジョブの種類。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Generate,
    Design,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Generate => "generate",
            Kind::Design => "design",
        }
    }
}

pub(crate) struct Job {
    pub id: String,
    pub kind: Kind,
    /// 生成ジョブの台本の ID。声の試作では `None`。
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
}

/// 最後に見た `(ステップ, 段階)` と、それが変わらなくなってからの時間。
pub(crate) struct Watch {
    seen: (usize, usize),
    since: Instant,
    warned: bool,
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

pub(crate) enum Work {
    Cards {
        job_id: String,
        project: String,
        requests: Vec<Request>,
    },
    Design {
        job_id: String,
        request: DesignRequest,
    },
}

pub(crate) struct Inner {
    pub store: Arc<Store>,
    pub synth: Arc<dyn Synthesizer>,
    /// 声の試作の候補を載せる登録簿。
    pub assets: Arc<Assets>,
    jobs: Mutex<Vec<Job>>,
    /// カードの状態・結果を保存できなかったときの理由（ジョブが「完了」と報告しないために、ワーカーが拾う）。
    save_error: Mutex<Option<String>>,
}

impl Inner {
    pub fn jobs(&self) -> MutexGuard<'_, Vec<Job>> {
        self.jobs.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// 生成中のジョブのうち、`threshold` 以上ステップも段階も変わっていないものを、止まりとして 1 度だけ知らせる。
    pub fn check_stalls(&self, now: Instant, threshold: Duration) -> Vec<String> {
        let mut found = Vec::new();
        for job in self.jobs().iter_mut().filter(|j| j.status == "生成中") {
            let current = (
                job.progress.load(Ordering::SeqCst),
                job.stage.load(Ordering::SeqCst),
            );
            if current != job.watch.seen {
                job.watch = Watch {
                    seen: current,
                    since: now,
                    warned: false,
                };
            } else if !job.watch.warned
                && now.saturating_duration_since(job.watch.since) >= threshold
            {
                job.watch.warned = true;
                found.push(format!(
                    "generation job {} seems stalled: stage={} step {}/{} unchanged for {}s",
                    job.id,
                    Stage::from_usize(current.1).name(),
                    current.0,
                    job.steps,
                    now.saturating_duration_since(job.watch.since).as_secs()
                ));
            }
        }
        found
    }

    pub fn note_save_error(&self, message: String) {
        *self.save_error.lock().unwrap_or_else(|e| e.into_inner()) = Some(message);
    }

    pub fn take_save_error(&self) -> Option<String> {
        self.save_error
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
    }

    /// 該当のジョブがあれば `f` を適用する。
    pub fn update_job(&self, id: &str, f: impl FnOnce(&mut Job)) {
        if let Some(job) = self.jobs().iter_mut().find(|j| j.id == id) {
            f(job);
        }
    }
}

pub struct JobManager {
    inner: Arc<Inner>,
    tx: Mutex<Option<Sender<Work>>>,
    /// 提出の同時実行を直列にする（空きの確認から登録までの間に割り込まれないように）。
    gate: Mutex<()>,
}

impl JobManager {
    pub fn new(store: Arc<Store>, synth: Arc<dyn Synthesizer>, assets: Arc<Assets>) -> JobManager {
        let inner = Arc::new(Inner {
            store,
            synth,
            assets,
            jobs: Mutex::new(Vec::new()),
            save_error: Mutex::new(None),
        });
        let (tx, rx) = mpsc::channel();
        // 長い間なにも進まない生成を、ログに残す（固まったときに、どの段階かを後から知るため）。
        let weak = Arc::downgrade(&inner);
        std::thread::Builder::new()
            .name("easy-synthesis-watchdog".into())
            .spawn(move || loop {
                std::thread::sleep(WATCH_EVERY);
                let Some(inner) = weak.upgrade() else { break };
                for message in inner.check_stalls(Instant::now(), STALL_AFTER) {
                    tracing::warn!("{message}");
                }
            })
            .expect("spawn the watchdog");
        let worker = Arc::clone(&inner);
        std::thread::Builder::new()
            .name("easy-synthesis".into())
            .spawn(move || run::worker(worker, rx))
            .expect("spawn the synthesis worker");
        JobManager {
            inner,
            tx: Mutex::new(Some(tx)),
            gate: Mutex::new(()),
        }
    }

    /// 待機中・生成中のジョブがあるか。
    pub fn busy(&self) -> bool {
        self.inner
            .jobs()
            .iter()
            .any(|j| BUSY.contains(&j.status.as_str()))
    }

    /// 動いているジョブすべてに停止を要求する。1 ステップ以内に止まる。
    pub fn stop(&self) {
        for job in self
            .inner
            .jobs()
            .iter()
            .filter(|j| BUSY.contains(&j.status.as_str()))
        {
            job.cancel.store(true, Ordering::SeqCst);
        }
    }

    /// これから処理される（未完了の）カードの ID。元に戻した後に「停止」へ変えない対象。
    pub fn active_card_ids(&self, pid: &str) -> HashSet<String> {
        self.inner
            .jobs()
            .iter()
            .filter(|j| j.project.as_deref() == Some(pid) && BUSY.contains(&j.status.as_str()))
            .flat_map(|j| j.pending.iter().cloned())
            .collect()
    }

    /// `/api/state` の `jobs`。内部の取消フラグなどは含めない。
    pub fn jobs_json(&self) -> Vec<Value> {
        self.inner
            .jobs()
            .iter()
            .map(|j| {
                json!({
                    "id": j.id, "project": j.project, "status": j.status, "done": j.done, "total": j.total, "error": j.error,
                    "stopping": j.cancel.load(Ordering::SeqCst), "kind": j.kind.name(), "assets": j.assets,
                    "step": j.progress.load(Ordering::SeqCst), "steps": j.steps,
                    "stage": Stage::from_usize(j.stage.load(Ordering::SeqCst)).name(),
                })
            })
            .collect()
    }

    /// 読み込み済みのモデルを解放する。生成中は断る。
    pub fn unload(&self) -> Result<()> {
        if self.busy() {
            return Err(StoreError::Invalid("生成完了後に実行してください。".into()));
        }
        self.inner.synth.unload();
        Ok(())
    }

    /// 止まりの検出（`Inner::check_stalls`）。ワーカーとは別のスレッドが定期的に呼ぶ。
    pub fn check_stalls(&self, now: Instant, threshold: Duration) -> Vec<String> {
        self.inner.check_stalls(now, threshold)
    }

    /// 履歴が増えすぎないよう、終わった古いジョブから捨てる。
    fn trim(jobs: &mut Vec<Job>) {
        while jobs.len() > KEEP_JOBS {
            match jobs.iter().position(|j| !BUSY.contains(&j.status.as_str())) {
                Some(i) => {
                    jobs.remove(i);
                }
                None => break,
            }
        }
    }
}

impl Drop for JobManager {
    fn drop(&mut self) {
        self.stop();
        self.tx.lock().unwrap_or_else(|e| e.into_inner()).take();
    }
}
