//! 生成の受け付け（`easy_service.submit` の移植）。確認・検証・スナップショットまでを行い、ワーカーへ渡す。

use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::Arc;

use irodori_store::dictionary::{self, Dictionary};
use irodori_store::model::{uid, Project};
use irodori_store::signature::{reading_text, result_matches, signature};
use irodori_store::splitter::estimate;
use irodori_store::{Result, StoreError};
use serde_json::{json, Value};

use super::{Job, JobManager, Request, SubmitOutcome, Watch, Work, BUSY};
use crate::settings_map::{effective_steps, options_from};
use crate::synth::SynthError;

/// 推定でこれを超える、または文字数がこれを超えるカードは確認を求める。
const LONG_SECONDS: f64 = 25.0;
const LONG_CHARS: usize = 180;

fn invalid(message: &str) -> StoreError {
    StoreError::Invalid(message.into())
}

fn from_synth(e: SynthError) -> StoreError {
    match e {
        SynthError::Invalid(m) => StoreError::Invalid(m),
        SynthError::Cancelled => StoreError::Invalid("停止しました。".into()),
        SynthError::Failed(d) => StoreError::Internal(d),
    }
}

impl JobManager {
    /// 台本のカードを生成に出す。`cid` を指定すればそのカードだけ。`changed` なら、すでに現行の結果があるカードは除く。
    pub fn submit(
        &self,
        pid: &str,
        revision: i64,
        cid: Option<&str>,
        changed: bool,
        allow_long: bool,
    ) -> Result<SubmitOutcome> {
        let _gate = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        if self.busy() {
            return Err(invalid(
                "現在の生成が終わるか、停止してから実行してください。",
            ));
        }
        let store = &self.inner.store;
        let doc = store.load(Some(pid))?;
        if doc.revision != revision {
            return Err(StoreError::Conflict(
                "台本が更新されています。読み直してください。".into(),
            ));
        }
        let dict = dictionary::load_lenient(store.root())?;
        let options = options_from(&doc.generation).map_err(from_synth)?;
        let requests = self.snapshot(&doc, &dict, cid, changed)?;
        if requests.is_empty() {
            return Err(invalid(
                "生成対象がありません。文章と生成状態を確認してください。",
            ));
        }
        let long_cards: Vec<usize> = requests
            .iter()
            .filter(|(_, r, original_len)| {
                *original_len > LONG_CHARS
                    || estimate(
                        &r.text,
                        &Dictionary::new(),
                        f64::from(options.duration_scale),
                    ) > LONG_SECONDS
            })
            .map(|(ordinal, _, _)| *ordinal)
            .collect();
        if !long_cards.is_empty() && !allow_long {
            return Ok(SubmitOutcome::NeedsLongConfirmation { long_cards });
        }

        let requests: Vec<Request> = requests.into_iter().map(|(_, r, _)| r).collect();
        let ids: HashSet<String> = requests.iter().map(|r| r.card_id.clone()).collect();
        let job = Job {
            id: uid(),
            project: pid.to_string(),
            status: "待機中".into(),
            done: 0,
            total: requests.len(),
            error: String::new(),
            steps: effective_steps(&options, &requests[0].model),
            cancel: Arc::new(AtomicBool::new(false)),
            progress: Arc::new(AtomicUsize::new(0)),
            stage: Arc::new(AtomicUsize::new(0)),
            watch: Watch::new(),
            pending: requests.iter().map(|r| r.card_id.clone()).collect(),
        };
        let work = Work {
            job_id: job.id.clone(),
            project: pid.to_string(),
            requests,
        };
        store.internal_update(pid, &mut |d: &mut Project| {
            for c in d.cards.iter_mut().filter(|c| ids.contains(&c.id)) {
                c.status = "待機中".into();
                c.error.clear();
            }
        })?;
        {
            let mut jobs = self.inner.jobs();
            if jobs.iter().any(|j| BUSY.contains(&j.status.as_str())) {
                return Err(invalid(
                    "現在の生成が終わるか、停止してから実行してください。",
                ));
            }
            jobs.push(job);
            JobManager::trim(&mut jobs);
        }
        let sender = self.tx.lock().unwrap_or_else(|e| e.into_inner());
        sender
            .as_ref()
            .ok_or_else(|| StoreError::Internal("worker stopped".into()))?
            .send(work)
            .map_err(|_| StoreError::Internal("worker stopped".into()))?;
        Ok(SubmitOutcome::Started)
    }

    /// 対象カードごとの要求を作る。`(カードの番号, 要求, 元の文章の文字数)`。
    fn snapshot(
        &self,
        doc: &Project,
        dict: &Dictionary,
        cid: Option<&str>,
        changed: bool,
    ) -> Result<Vec<(usize, Request, usize)>> {
        let store = &self.inner.store;
        let (voices, settings) = (store.voices(), store.settings());
        let settings_json =
            serde_json::to_value(&settings).map_err(|e| StoreError::Internal(e.to_string()))?;
        let rules: Dictionary = if doc.dictionary_enabled {
            dict.clone()
        } else {
            Dictionary::new()
        };
        let date = chrono::Local::now().format("%Y%m%d").to_string();
        let mut out = Vec::new();
        for (index, card) in doc.cards.iter().enumerate() {
            let ordinal = index + 1;
            if cid.is_some_and(|c| c != card.id) || card.text.trim().is_empty() {
                continue;
            }
            let voice = voices.iter().find(|v| v.id == card.voice_id);
            let current = result_matches(doc, card, dict, voice)
                && card
                    .result
                    .as_ref()
                    .and_then(|r| r["path"].as_str())
                    .is_some_and(|p| Path::new(p).exists());
            if changed && card.status != "失敗" && current {
                continue;
            }
            let voice = voice.ok_or_else(|| {
                StoreError::Invalid(format!("{ordinal}番のカードの話者を選んでください。"))
            })?;
            if !self.inner.synth.model_available(&voice.model) {
                return Err(invalid("モデルがまだ取得されていません。"));
            }
            if (card.speed - 1.0).abs() > 1e-9 {
                return Err(invalid(
                    "話速の変更はこの版ではまだ使えません。話速を1.0倍にしてください。",
                ));
            }
            let sig = signature(doc, card, dict, Some(voice));
            let text = reading_text(doc, card, dict);
            let voice_json =
                serde_json::to_value(voice).map_err(|e| StoreError::Internal(e.to_string()))?;
            let record = json!({
                "card": {"id": card.id, "text": card.text, "voice_id": card.voice_id, "speed": card.speed, "silence": card.silence, "filename": card.filename},
                "voice": voice_json, "signature": sig, "text": text, "generation": Value::Object(doc.generation.clone()),
                "ordinal": ordinal, "settings": settings_json, "date": date, "dictionary": serde_json::to_value(&rules).map_err(|e| StoreError::Internal(e.to_string()))?,
            });
            let audio = Path::new(&voice.audio);
            let reference = if audio.is_absolute() {
                audio.to_path_buf()
            } else {
                store.root().join(audio)
            };
            out.push((
                ordinal,
                Request {
                    card_id: card.id.clone(),
                    signature: sig,
                    text,
                    model: voice.model.clone(),
                    caption: voice.caption.clone(),
                    reference,
                    generation: doc.generation.clone(),
                    silence: card.silence,
                    record,
                },
                card.text.chars().count(),
            ));
        }
        Ok(out)
    }
}
