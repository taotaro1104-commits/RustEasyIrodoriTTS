//! 声の試作ジョブ（`easy_service.design` の移植）。生成ジョブと同じワーカー・取消・進捗を使い、カードには触らない。

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use irodori_store::dictionary::Dictionary;
use irodori_store::generation::{validate_generation, Generation};
use irodori_store::model::uid;
use irodori_store::splitter::split_text;
use irodori_store::{Result, StoreError};
use serde_json::{json, Map, Value};

use super::submit::from_synth;
use super::{Inner, Job, JobManager, Kind, Watch, Work, BUSY, GENERIC_FAILURE};
use crate::models::ALLOW_LIST;
use crate::settings_map::{effective_steps, options_from};
use crate::synth::{SynthError, SynthJob};

/// 登録用の定型文（v2.0 の `REGISTRATION_SAMPLE_TEXT`）。
pub const REGISTRATION_SAMPLE_TEXT: &str = "おはようございます。今日は空が晴れて気持ちのよい朝ですね。窓を開けると涼しい風が入ってきました。温かいお茶を飲んでゆっくり出かける準備をしましょう。";

const TRIAL_LENGTH: &str = "声の試作には推定25秒以内の短い文章を入力してください。";
const MAX_CANDIDATES: f64 = 8.0;

fn invalid(message: &str) -> StoreError {
    StoreError::Invalid(message.into())
}

/// 声の作り方。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DesignMode {
    /// 説明だけで作る。
    Design,
    /// お手本の音声だけで作る。
    Clone,
    /// 説明とお手本の両方。
    Both,
    /// どちらも使わない。
    Auto,
}

impl DesignMode {
    fn parse(value: Option<&Value>) -> Result<DesignMode> {
        match value {
            None => Ok(DesignMode::Design),
            Some(Value::String(s)) => match s.as_str() {
                "design" => Ok(DesignMode::Design),
                "clone" => Ok(DesignMode::Clone),
                "both" => Ok(DesignMode::Both),
                "auto" => Ok(DesignMode::Auto),
                _ => Err(invalid("声の作り方が不正です。")),
            },
            Some(_) => Err(invalid("声の作り方が不正です。")),
        }
    }

    fn uses_caption(self) -> bool {
        matches!(self, DesignMode::Design | DesignMode::Both)
    }

    fn uses_references(self) -> bool {
        matches!(self, DesignMode::Clone | DesignMode::Both)
    }
}

/// 提出時点の試作の依頼。
#[derive(Debug, Clone)]
pub struct DesignRequest {
    pub text: String,
    pub model: String,
    pub mode: DesignMode,
    pub caption: String,
    pub references: Vec<PathBuf>,
    pub generation: Generation,
    pub candidates: usize,
    /// 画面から受け取った値（`text` は実際に使う文章に置き換え済み）。採用時に話者の `creation` になる。
    pub creation: Value,
}

/// 1〜8 の整数（小数は切り捨て）。数値か数字の文字列を受け付ける。
fn candidates_of(value: Option<&Value>) -> Result<usize> {
    let message = "候補数は1～8で指定してください。";
    let x = match value {
        None => 1.0,
        Some(Value::Number(n)) => n.as_f64().unwrap_or(f64::NAN),
        Some(Value::String(s)) => s.trim().parse::<f64>().unwrap_or(f64::NAN),
        Some(_) => f64::NAN,
    };
    if x.is_finite() && (1.0..=MAX_CANDIDATES).contains(&x) {
        Ok(x as usize)
    } else {
        Err(invalid(message))
    }
}

impl JobManager {
    /// 声の試作を受け付ける。`registration` なら、文章は登録用の定型文に置き換える。戻り値はジョブの ID。
    pub fn submit_design(&self, values: &Map<String, Value>, registration: bool) -> Result<String> {
        let _gate = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        if self.busy() {
            return Err(invalid("現在の生成が終わってから実行してください。"));
        }
        let text = if registration {
            REGISTRATION_SAMPLE_TEXT.to_string()
        } else {
            values
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        let generation = validate_generation(
            &values
                .get("generation")
                .cloned()
                .unwrap_or_else(|| json!({})),
        )?;
        let scale = generation
            .get("duration_scale")
            .and_then(Value::as_f64)
            .unwrap_or(1.0);
        if text.trim().is_empty() || split_text(&text, &Dictionary::new(), scale)?.len() > 1 {
            return Err(invalid(TRIAL_LENGTH));
        }
        let mode = DesignMode::parse(values.get("mode"))?;
        let mut references = Vec::new();
        if mode.uses_references() {
            let keys = values.get("references").and_then(Value::as_array);
            for key in keys.map(|a| a.as_slice()).unwrap_or_default() {
                let key = key
                    .as_str()
                    .ok_or_else(|| invalid("音声が見つかりません。もう一度追加してください。"))?;
                references.push(self.inner.assets.source(key)?);
            }
            if references.is_empty() {
                return Err(invalid("お手本の音声を追加してください。"));
            }
            if references.len() > 1 {
                return Err(invalid("お手本の音声は1つだけ使えます。"));
            }
        }
        let model = values
            .get("model")
            .and_then(Value::as_str)
            .filter(|m| ALLOW_LIST.contains(m))
            .ok_or_else(|| invalid("使用するモデルを選んでください。"))?
            .to_string();
        if !self.inner.synth.model_available(&model) {
            return Err(invalid("モデルがまだ取得されていません。"));
        }
        let candidates = candidates_of(values.get("candidates"))?;
        let steps = effective_steps(&options_from(&generation).map_err(from_synth)?, &model);
        let caption = if mode.uses_caption() {
            values
                .get("caption")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        } else {
            String::new()
        };
        let mut creation = values.clone();
        creation.insert("text".into(), json!(text));
        let request = DesignRequest {
            text,
            model,
            mode,
            caption,
            references,
            generation,
            candidates,
            creation: Value::Object(creation),
        };

        let job = Job {
            id: uid(),
            kind: Kind::Design,
            project: None,
            status: "待機中".into(),
            done: 0,
            total: 1,
            error: String::new(),
            steps,
            cancel: Arc::new(AtomicBool::new(false)),
            progress: Arc::new(AtomicUsize::new(0)),
            stage: Arc::new(AtomicUsize::new(0)),
            watch: Watch::new(),
            pending: Vec::new(),
            assets: Vec::new(),
            creation: request.creation.clone(),
            record: None,
        };
        let id = job.id.clone();
        {
            let mut jobs = self.inner.jobs();
            if jobs.iter().any(|j| BUSY.contains(&j.status.as_str())) {
                return Err(invalid("現在の生成が終わってから実行してください。"));
            }
            jobs.push(job);
            JobManager::trim(&mut jobs);
        }
        let sender = self.tx.lock().unwrap_or_else(|e| e.into_inner());
        sender
            .as_ref()
            .ok_or_else(|| StoreError::Internal("worker stopped".into()))?
            .send(Work::Design {
                job_id: id.clone(),
                request,
            })
            .map_err(|_| StoreError::Internal("worker stopped".into()))?;
        Ok(id)
    }

    /// 完了した試作ジョブの `(候補の鍵, 話者の creation にする記録)`。完了していなければ `None`。
    pub fn design_result(&self, job_id: &str) -> Option<(Vec<String>, Value)> {
        self.inner
            .jobs()
            .iter()
            .find(|j| j.id == job_id && j.kind == Kind::Design && j.status == "完了")
            .map(|j| {
                (
                    j.assets.clone(),
                    j.record.clone().unwrap_or_else(|| j.creation.clone()),
                )
            })
    }
}

/// ワーカー側。試作 1 件を合成し、候補を登録簿に載せて結果をジョブへ書く。
pub(super) fn run(inner: &Arc<Inner>, job_id: &str, request: DesignRequest) {
    let (cancel, progress, stage) = {
        let jobs = inner.jobs();
        let Some(job) = jobs.iter().find(|j| j.id == job_id) else {
            return;
        };
        (
            Arc::clone(&job.cancel),
            Arc::clone(&job.progress),
            Arc::clone(&job.stage),
        )
    };
    if cancel.load(Ordering::SeqCst) {
        inner.update_job(job_id, |j| j.status = "停止".into());
        return;
    }
    inner.update_job(job_id, |j| j.status = "生成中".into());
    let synth_job = SynthJob {
        text: request.text.clone(),
        model: request.model.clone(),
        caption: request.caption.clone(),
        references: request.references.clone(),
        generation: request.generation.clone(),
        candidates: request.candidates,
        cancel,
        progress,
        stage,
    };
    let outcome = catch_unwind(AssertUnwindSafe(|| inner.synth.synthesize(&synth_job)));
    match outcome {
        Ok(Ok(record)) => {
            let assets: Vec<String> = record
                .paths
                .iter()
                .filter_map(|p| inner.assets.register(p))
                .collect();
            if assets.len() != record.paths.len() {
                tracing::error!("design candidates could not be registered");
                fail(inner, job_id, GENERIC_FAILURE);
                return;
            }
            let paths: Vec<String> = record
                .paths
                .iter()
                .map(|p| p.display().to_string())
                .collect();
            let record_json = json!({
                "id": record.id, "model": record.model, "text": record.text, "caption": record.caption,
                "references": record.references, "generation": record.generation, "seed": record.seed,
                "speed": 1.0, "silence": 0.0, "paths": paths, "seconds": record.seconds,
                "messages": record.messages,
            });
            inner.update_job(job_id, |j| {
                j.status = "完了".into();
                j.done = 1;
                j.assets = assets;
                j.record = Some(record_json);
            });
        }
        Ok(Err(SynthError::Cancelled)) => inner.update_job(job_id, |j| j.status = "停止".into()),
        Ok(Err(SynthError::Invalid(message))) => fail(inner, job_id, &message),
        Ok(Err(SynthError::Failed(detail))) => {
            tracing::error!("voice design failed: {detail}");
            fail(inner, job_id, GENERIC_FAILURE);
        }
        Err(_) => {
            tracing::error!("the synthesizer panicked during a voice design");
            fail(inner, job_id, GENERIC_FAILURE);
        }
    }
}

fn fail(inner: &Inner, job_id: &str, message: &str) {
    inner.update_job(job_id, |j| {
        j.status = "失敗".into();
        j.error = message.to_string();
    });
}
