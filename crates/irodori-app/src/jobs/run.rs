//! ワーカー側（`easy_service.run` の移植）。カードを 1 枚ずつ合成し、結果と状態を保存する。

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::mpsc::Receiver;
use std::sync::Arc;

use irodori_store::model::Project;
use irodori_store::postprocess::add_silence;
use irodori_store::storage::atomic_json;
use serde_json::{json, Value};

use super::{Inner, Request, Work, BUSY, GENERIC_FAILURE};
use crate::synth::{SynthError, SynthJob, SynthRecord};

pub(super) fn worker(inner: Arc<Inner>, rx: Receiver<Work>) {
    while let Ok(work) = rx.recv() {
        run_work(&inner, work);
    }
}

fn run_work(inner: &Arc<Inner>, work: Work) {
    let Work {
        job_id,
        project,
        requests,
    } = work;
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        run_requests(inner, &job_id, &project, &requests)
    }));
    if outcome.is_err() {
        tracing::error!("generation job panicked outside a request");
        inner.update_job(&job_id, |j| {
            j.status = "失敗".into();
            j.error = GENERIC_FAILURE.into();
        });
    }
    // どの経路でも、待機中・生成中のまま残ったカードは「停止」にする。
    let reset = inner
        .store
        .internal_update(&project, &mut |d: &mut Project| {
            for c in d
                .cards
                .iter_mut()
                .filter(|c| BUSY.contains(&c.status.as_str()))
            {
                c.status = "停止".into();
            }
        });
    if let Err(e) = reset {
        tracing::error!("could not reset card statuses: {}", e.message());
        inner.note_save_error(
            "結果を保存できませんでした。保存先の空き容量とアクセス権を確認してください。"
                .to_string(),
        );
    }
    // 最後の保存の失敗も、ジョブの結果に反映する（「完了」のままにしない）。
    if let Some(message) = inner.take_save_error() {
        inner.update_job(&job_id, |j| {
            j.error = message;
            if j.status == "完了" {
                j.status = "完了（一部失敗）".into();
            }
        });
    }
}

/// カード 1 枚の状態（と結果）を、ID で探して更新する。文章など入力は触らない。
fn mark(
    inner: &Inner,
    project: &str,
    card_id: &str,
    status: &str,
    error: &str,
    result: Option<Value>,
) {
    let outcome = inner
        .store
        .internal_update(project, &mut |d: &mut Project| {
            if let Some(c) = d.cards.iter_mut().find(|c| c.id == card_id) {
                c.status = status.to_string();
                c.error = error.to_string();
                if let Some(r) = &result {
                    c.result = Some(r.clone());
                }
            }
        });
    if let Err(e) = outcome {
        tracing::error!("could not save the card status: {}", e.message());
        inner.note_save_error(format!(
            "結果を保存できませんでした。保存先の空き容量とアクセス権を確認してください。"
        ));
    }
}

fn run_requests(inner: &Arc<Inner>, job_id: &str, project: &str, requests: &[Request]) {
    let (cancel, progress) = {
        let jobs = inner.jobs();
        let job = jobs
            .iter()
            .find(|j| j.id == job_id)
            .expect("job registered before work is sent");
        (Arc::clone(&job.cancel), Arc::clone(&job.progress))
    };
    let mut last_error = String::new();
    for request in requests {
        if cancel.load(Ordering::SeqCst) {
            break;
        }
        inner.update_job(job_id, |j| j.status = "生成中".into());
        mark(inner, project, &request.card_id, "生成中", "", None);
        progress.store(0, Ordering::SeqCst);
        let job = SynthJob {
            text: request.text.clone(),
            model: request.model.clone(),
            caption: request.caption.clone(),
            references: vec![request.reference.clone()],
            generation: request.generation.clone(),
            candidates: 1,
            cancel: Arc::clone(&cancel),
            progress: Arc::clone(&progress),
        };
        let synthesized = catch_unwind(AssertUnwindSafe(|| inner.synth.synthesize(&job)));
        match synthesized {
            Ok(Ok(record)) => match finish(request, &record) {
                Ok(result) => mark(inner, project, &request.card_id, "完了", "", Some(result)),
                Err(message) => {
                    last_error = message.clone();
                    mark(inner, project, &request.card_id, "失敗", &message, None);
                }
            },
            Ok(Err(SynthError::Cancelled)) => {
                mark(inner, project, &request.card_id, "停止", "", None)
            }
            Ok(Err(SynthError::Invalid(message))) => {
                last_error = message.clone();
                mark(inner, project, &request.card_id, "失敗", &message, None);
            }
            Ok(Err(SynthError::Failed(detail))) => {
                tracing::error!("generation failed: {detail}");
                last_error = GENERIC_FAILURE.into();
                mark(
                    inner,
                    project,
                    &request.card_id,
                    "失敗",
                    GENERIC_FAILURE,
                    None,
                );
            }
            Err(_) => {
                tracing::error!("the synthesizer panicked on a card");
                last_error = GENERIC_FAILURE.into();
                mark(
                    inner,
                    project,
                    &request.card_id,
                    "失敗",
                    GENERIC_FAILURE,
                    None,
                );
            }
        }
        if let Some(message) = inner.take_save_error() {
            last_error = message;
        }
        inner.update_job(job_id, |j| {
            j.done += 1;
            if !j.pending.is_empty() {
                j.pending.remove(0);
            }
            if !last_error.is_empty() {
                j.error = last_error.clone();
            }
        });
    }
    let cancelled = cancel.load(Ordering::SeqCst);
    inner.update_job(job_id, |j| {
        j.status = if cancelled {
            "停止"
        } else if j.error.is_empty() {
            "完了"
        } else {
            "完了（一部失敗）"
        }
        .into();
        j.pending.clear();
    });
}

/// 合成できた WAV に終了無音を足し、v2.0 と同じ形の `result` を作って `card.json` にも保存する。失敗はカードに出す文を返す。
fn finish(request: &Request, record: &SynthRecord) -> Result<Value, String> {
    let first = record
        .paths
        .first()
        .ok_or_else(|| GENERIC_FAILURE.to_string())?;
    let wav = add_silence(first, request.silence).map_err(|e| e.message().to_string())?;
    let path: PathBuf = std::path::absolute(&wav).unwrap_or(wav);
    let result = json!({
        "path": path.display().to_string(), "signature": request.signature, "seed": record.seed,
        "request": request.record, "record": record.id, "export": {},
    });
    let folder: &Path = path.parent().unwrap_or(Path::new("."));
    atomic_json(&folder.join("card.json"), &result).map_err(|e| {
        tracing::error!("could not write card.json: {}", e.message());
        GENERIC_FAILURE.to_string()
    })?;
    Ok(result)
}
