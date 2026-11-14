//! モデルの取得ジョブ。確認（`confirmed`）のあとにだけ始め、既存のワーカー・取消・進捗の仕組みで動かす。

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use irodori_store::model::uid;
use irodori_store::{Result, StoreError};

use super::{Inner, Job, JobManager, Kind, Work, BUSY, GENERIC_FAILURE};
use crate::download::{DownloadControl, DownloadError};
use crate::models::ALLOW_LIST;

fn invalid(message: &str) -> StoreError {
    StoreError::Invalid(message.into())
}

impl JobManager {
    /// モデルの取得を始める。`confirmed` は、利用者が画面で確認したことの印（偽なら何も始めない）。戻り値はジョブの ID。
    pub fn submit_download(&self, model: &str, confirmed: bool) -> Result<String> {
        let _gate = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        if !ALLOW_LIST.contains(&model) {
            return Err(invalid("使用するモデルを選んでください。"));
        }
        if !confirmed {
            return Err(invalid("取得の確認が必要です。"));
        }
        if self.busy() {
            return Err(invalid("現在の処理が終わってから実行してください。"));
        }
        if self.inner.synth.model_available(model) {
            return Err(invalid("このモデルはすでに取得済みです。"));
        }
        let job = Job {
            model: model.to_string(),
            ..Job::new(uid(), Kind::Download)
        };
        let id = job.id.clone();
        {
            let mut jobs = self.inner.jobs();
            if jobs.iter().any(|j| BUSY.contains(&j.status.as_str())) {
                return Err(invalid("現在の処理が終わってから実行してください。"));
            }
            jobs.push(job);
            JobManager::trim(&mut jobs);
        }
        let sender = self.tx.lock().unwrap_or_else(|e| e.into_inner());
        sender
            .as_ref()
            .ok_or_else(|| StoreError::Internal("worker stopped".into()))?
            .send(Work::Download {
                job_id: id.clone(),
                model: model.to_string(),
            })
            .map_err(|_| StoreError::Internal("worker stopped".into()))?;
        Ok(id)
    }

    /// 動いている取得にだけ中止を要求する（生成・試作は止めない）。書きかけは再開用に残る。
    pub fn cancel_downloads(&self) {
        for job in self
            .inner
            .jobs()
            .iter()
            .filter(|j| j.kind == Kind::Download && BUSY.contains(&j.status.as_str()))
        {
            job.cancel.store(true, Ordering::SeqCst);
        }
    }
}

/// ワーカー側。取得を実行し、結果をジョブへ書く。
pub(super) fn run(inner: &Arc<Inner>, job_id: &str, model: &str) {
    let control = {
        let jobs = inner.jobs();
        let Some(job) = jobs.iter().find(|j| j.id == job_id) else {
            return;
        };
        DownloadControl {
            cancel: Arc::clone(&job.cancel),
            done: Arc::clone(&job.bytes_done),
            total: Arc::clone(&job.bytes_total),
        }
    };
    if control.cancel.load(Ordering::SeqCst) {
        inner.update_job(job_id, |j| j.status = "停止".into());
        return;
    }
    inner.update_job(job_id, |j| j.status = "生成中".into());
    tracing::info!("model download started: {model}");
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        inner.installer.install(model, &control)
    }));
    match outcome {
        Ok(Ok(())) => {
            tracing::info!("model download finished: {model}");
            inner.update_job(job_id, |j| {
                j.status = "完了".into();
                j.done = 1;
            });
        }
        Ok(Err(DownloadError::Cancelled)) => {
            tracing::info!("model download cancelled: {model}");
            inner.update_job(job_id, |j| j.status = "停止".into());
        }
        Ok(Err(error)) => {
            tracing::warn!("model download failed: {error}");
            let message = error.message();
            inner.update_job(job_id, |j| {
                j.status = "失敗".into();
                j.error = message;
            });
        }
        Err(_) => {
            tracing::error!("the installer panicked");
            inner.update_job(job_id, |j| {
                j.status = "失敗".into();
                j.error = GENERIC_FAILURE.into();
            });
        }
    }
}
