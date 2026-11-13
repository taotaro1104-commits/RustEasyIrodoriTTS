//! 本物の合成器：`irodori-core` の `Pipeline` を、モデルと精度の組ごとに 1 つだけ保持して使う。
//!
//! 別の組に切り替えるときは前のを解放してから読み込む（段階ロードなので読み込みは速く、ピークメモリは最大の部品 1 つ分）。

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle_core::{DType, Device};
use irodori_core::control::Control;
use irodori_core::pipeline::{MemoryMode, Pipeline, PipelineOptions, SynthRequest};
use irodori_core::runtime::Runtime;
use irodori_core::IrodoriError;
use irodori_store::model::uid;
use irodori_store::storage::atomic_json;
use serde_json::{json, Value};

use crate::models::{self, ModelFiles};
use crate::settings_map::options_from;
use crate::synth::{SynthError, SynthJob, SynthRecord, Synthesizer};

/// 上流（v2.0）と同じ、生成する音声の最大の長さ（秒）。
const MAX_SECONDS: f32 = 30.0;

/// 読み込み済みのモデルの識別: `(モデル ID, モデルを bf16 で持つか, codec を bf16 で持つか)`。
type Key = (String, bool, bool);

struct Loaded {
    key: Key,
    runtime: Arc<Runtime<Pipeline>>,
}

pub struct RealSynth {
    root: PathBuf,
    hub: PathBuf,
    loaded: Mutex<Option<Loaded>>,
}

fn invalid(message: &str) -> SynthError {
    SynthError::Invalid(message.to_string())
}

/// コアのエラーを、利用者向けの文（`Invalid`）か内部の失敗（`Failed`）に分ける。
pub fn map_error(error: IrodoriError, has_reference: bool) -> SynthError {
    match error {
        IrodoriError::Cancelled => SynthError::Cancelled,
        IrodoriError::EmptyText => invalid("モデルと読み上げ文章を確認してください。"),
        IrodoriError::TextTooLong { .. } => {
            invalid("文章がモデルの処理上限を超えています。カードをさらに短くしてください。")
        }
        IrodoriError::SilentReference | IrodoriError::ReferenceTooShort { .. } => {
            invalid("参照音声を確認してください。")
        }
        // 入出力の失敗（モデルファイルの欠損・破損など）は参照音声のせいにしない。参照の有無は事前に確かめてある。
        IrodoriError::Audio(_) if has_reference => {
            invalid("参照音声を読み込めません。形式を確認してください。")
        }
        IrodoriError::InvalidInput(detail) => {
            tracing::warn!("invalid generation input: {detail}");
            invalid("生成設定を確認してください。")
        }
        other => SynthError::Failed(other.to_string()),
    }
}

impl RealSynth {
    pub fn new(root: PathBuf, hub: PathBuf) -> RealSynth {
        RealSynth {
            root,
            hub,
            loaded: Mutex::new(None),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<Loaded>> {
        self.loaded.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// 組が同じなら読み込み済みを使い回し、違えば前のを解放して新しく用意する。
    fn runtime_for(&self, key: Key, files: ModelFiles) -> Arc<Runtime<Pipeline>> {
        let mut slot = self.lock();
        if let Some(loaded) = slot.as_ref().filter(|l| l.key == key) {
            return Arc::clone(&loaded.runtime);
        }
        *slot = None; // 先に解放してから読み込む（一時的にメモリが 2 倍にならないように）
        let dtype = |bf16: bool| if bf16 { DType::BF16 } else { DType::F32 };
        let options = PipelineOptions {
            dtype: dtype(key.1),
            codec_dtype: dtype(key.2),
            memory: MemoryMode::Staged,
        };
        let runtime = Arc::new(Runtime::new(move || {
            Pipeline::load_with(&files.model, &files.codec, &Device::Cpu, options.clone())
        }));
        *slot = Some(Loaded {
            key,
            runtime: Arc::clone(&runtime),
        });
        runtime
    }
}

impl Synthesizer for RealSynth {
    fn synthesize(&self, job: &SynthJob) -> Result<SynthRecord, SynthError> {
        if job.references.len() > 1 {
            return Err(invalid("お手本の音声は1つだけ使えます。"));
        }
        let options = options_from(&job.generation)?;
        let files = models::locate(&self.hub, &job.model)
            .ok_or_else(|| invalid("モデルがまだ取得されていません。"))?;
        // 参照音声のファイルが無い・開けないことは、モデルを読み込む前に分かる。
        if let Some(reference) = job.references.first() {
            if std::fs::File::open(reference).is_err() {
                return Err(invalid(
                    "参照音声を読み込めません。話者を登録し直してください。",
                ));
            }
        }
        let runtime = self.runtime_for(
            (job.model.clone(), options.model_bf16, options.codec_bf16),
            files,
        );

        let started = Instant::now();
        let id = uid();
        tracing::info!(
            "generation {id} started: model={} candidates={} chars={}",
            job.model,
            job.candidates,
            job.text.chars().count()
        );
        let folder = std::path::absolute(self.root.join("outputs").join("internal").join(&id))
            .map_err(|e| SynthError::Failed(e.to_string()))?;
        let control = Control {
            cancel: Some(Arc::clone(&job.cancel)),
            progress: Some(Arc::clone(&job.progress)),
            stage: Some(Arc::clone(&job.stage)),
        };
        let has_reference = !job.references.is_empty();
        let (mut paths, mut messages, mut first_seed) = (Vec::new(), Vec::new(), None);
        for candidate in 0..job.candidates.max(1) {
            let request = SynthRequest {
                text: job.text.clone(),
                caption: Some(job.caption.clone()).filter(|c| !c.trim().is_empty()),
                ref_wav: job.references.first().cloned(),
                seed: options.seed.map(|s| s + candidate as u64),
                seconds: options.seconds,
                duration_scale: options.duration_scale,
                max_seconds: MAX_SECONDS,
                num_steps: options.steps,
                sampler: options.sampler.clone(),
                control: control.clone(),
                max_text_len: options.max_text_len,
                max_caption_len: options.max_caption_len,
                ..SynthRequest::default()
            };
            let output = runtime
                .run(&request)
                .map_err(|e| map_error(e, has_reference))?;
            tracing::info!(
                "generation {id}: candidate {} synthesized at {:.1}s (stages: {:?})",
                candidate + 1,
                started.elapsed().as_secs_f64(),
                output.timings
            );
            std::fs::create_dir_all(&folder).map_err(|e| SynthError::Failed(e.to_string()))?;
            let path = folder.join(format!("{:02}.wav", candidate + 1));
            irodori_core::audio::write_wav(&path, &output.samples, output.sample_rate)
                .map_err(|e| SynthError::Failed(e.to_string()))?;
            first_seed.get_or_insert(output.seed);
            messages.extend(output.warnings);
            paths.push(path);
        }
        let record = SynthRecord {
            id: id.clone(),
            model: job.model.clone(),
            text: job.text.clone(),
            caption: job.caption.clone(),
            references: job
                .references
                .iter()
                .map(|p| p.display().to_string())
                .collect(),
            generation: Value::Object(job.generation.clone()),
            seed: first_seed.unwrap_or_default(),
            paths,
            seconds: (started.elapsed().as_secs_f64() * 100.0).round() / 100.0,
            messages,
        };
        let meta = json!({
            "id": record.id, "model": record.model, "text": record.text, "caption": record.caption,
            "references": record.references, "generation": record.generation, "seed": record.seed,
            "paths": record.paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
            "seconds": record.seconds, "messages": record.messages,
        });
        tracing::info!(
            "generation {id}: {} file(s) written at {:.1}s",
            record.paths.len(),
            started.elapsed().as_secs_f64()
        );
        atomic_json(&folder.join("generation.json"), &meta)
            .map_err(|e| SynthError::Failed(e.message().to_string()))?;
        Ok(record)
    }

    fn unload(&self) {
        *self.lock() = None;
    }

    fn is_loaded(&self) -> bool {
        self.lock().is_some()
    }

    fn model_available(&self, model: &str) -> bool {
        models::locate(&self.hub, model).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{SynthError, SynthJob, Synthesizer};
    use irodori_core::IrodoriError;
    use irodori_store::generation::default_generation;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    const MF: &str = "Aratako/Irodori-TTS-v4.1-Small-MF";

    #[test]
    fn core_errors_map_to_japanese_messages() {
        let invalid = |m: &str| SynthError::Invalid(m.to_string());
        assert_eq!(
            map_error(IrodoriError::Cancelled, false),
            SynthError::Cancelled
        );
        assert_eq!(
            map_error(IrodoriError::EmptyText, false),
            invalid("モデルと読み上げ文章を確認してください。")
        );
        assert_eq!(
            map_error(IrodoriError::TextTooLong { tokens: 9, max: 4 }, false),
            invalid("文章がモデルの処理上限を超えています。カードをさらに短くしてください。")
        );
        assert_eq!(
            map_error(IrodoriError::SilentReference, true),
            invalid("参照音声を確認してください。")
        );
        assert_eq!(
            map_error(IrodoriError::ReferenceTooShort { seconds: 0.2 }, true),
            invalid("参照音声を確認してください。")
        );
        assert_eq!(
            map_error(IrodoriError::Audio("decode".into()), true),
            invalid("参照音声を読み込めません。形式を確認してください。")
        );
        assert_eq!(
            map_error(IrodoriError::InvalidInput("seconds".into()), false),
            invalid("生成設定を確認してください。")
        );
        assert!(
            matches!(
                map_error(IrodoriError::Audio("decode".into()), false),
                SynthError::Failed(_)
            ),
            "no reference: an audio error is ours"
        );
        assert!(
            matches!(map_error(IrodoriError::MissingKey("w".into()), false), SynthError::Failed(d) if d.contains("w"))
        );
    }

    #[test]
    fn a_missing_model_is_reported_before_anything_is_loaded() {
        let dir = tempfile::tempdir().unwrap();
        let synth = RealSynth::new(dir.path().to_path_buf(), dir.path().join("hub"));
        assert!(!synth.model_available(MF));
        let err = synth
            .synthesize(&SynthJob::new("こんにちは", MF, default_generation()))
            .err()
            .unwrap();
        assert_eq!(
            err,
            SynthError::Invalid("モデルがまだ取得されていません。".into())
        );
        assert!(!synth.is_loaded());
        let err = synth
            .synthesize(&SynthJob::new(
                "こんにちは",
                "someone/else",
                default_generation(),
            ))
            .err()
            .unwrap();
        assert_eq!(
            err,
            SynthError::Invalid("モデルがまだ取得されていません。".into()),
            "ids outside the allow list are never loaded"
        );
    }

    #[test]
    fn unsupported_settings_are_refused_before_loading() {
        let dir = tempfile::tempdir().unwrap();
        let synth = RealSynth::new(dir.path().to_path_buf(), dir.path().join("hub"));
        let mut g = default_generation();
        g.insert("model_precision".into(), serde_json::json!("fp16"));
        let err = synth
            .synthesize(&SynthJob::new("こんにちは", MF, g))
            .err()
            .unwrap();
        assert_eq!(
            err,
            SynthError::Invalid("この版では fp32 か bf16 を選んでください。".into())
        );
    }

    #[test]
    fn two_references_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let synth = RealSynth::new(dir.path().to_path_buf(), dir.path().join("hub"));
        let mut job = SynthJob::new("こんにちは", MF, default_generation());
        job.references = vec!["a.wav".into(), "b.wav".into()];
        assert_eq!(
            synth.synthesize(&job).err().unwrap(),
            SynthError::Invalid("お手本の音声は1つだけ使えます。".into())
        );
    }

    #[test]
    fn model_file_io_errors_are_not_blamed_on_the_reference_voice() {
        let io = || IrodoriError::Io(std::io::Error::other("truncated model.safetensors"));
        assert!(
            matches!(map_error(io(), true), SynthError::Failed(d) if d.contains("truncated")),
            "an IO error with a reference is still ours"
        );
        assert!(matches!(map_error(io(), false), SynthError::Failed(_)));
    }

    /// 偽のモデル置き場（ファイルは空でよい。読み込みまで進まないテスト用）。
    fn fake_hub(dir: &std::path::Path) -> std::path::PathBuf {
        let hub = dir.join("hub");
        for (id, sha, files) in [
            (
                MF,
                "aaaa1111",
                vec!["model.safetensors", "tokenizer/tokenizer.json"],
            ),
            (crate::models::CODEC_ID, "cccc3333", vec!["weights.pth"]),
        ] {
            for f in files {
                let path = hub
                    .join(format!("models--{}", id.replace('/', "--")))
                    .join("snapshots")
                    .join(sha)
                    .join(f);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(path, b"x").unwrap();
            }
        }
        hub
    }

    #[test]
    fn a_missing_reference_file_is_reported_before_the_model_is_loaded() {
        let dir = tempfile::tempdir().unwrap();
        let synth = RealSynth::new(dir.path().to_path_buf(), fake_hub(dir.path()));
        assert!(synth.model_available(MF));
        let mut job = SynthJob::new("こんにちは", MF, default_generation());
        job.references = vec![dir.path().join("voices/none/ref.wav")];
        let err = synth.synthesize(&job).err().unwrap();
        assert_eq!(
            err,
            SynthError::Invalid("参照音声を読み込めません。話者を登録し直してください。".into())
        );
        assert!(
            !synth.is_loaded(),
            "nothing was loaded for a request that cannot run"
        );
    }

    /// 実モデル（`IRODORI_HUB_DIR` が HF 構造のキャッシュを指すときだけ）。`--release` で。
    #[test]
    fn real_meanflow_generation_writes_a_24bit_wav_and_can_be_cancelled() {
        let Some(hub) = std::env::var_os("IRODORI_HUB_DIR") else {
            eprintln!("skip: IRODORI_HUB_DIR not set");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let synth = RealSynth::new(dir.path().to_path_buf(), hub.into());
        assert!(
            synth.model_available(MF),
            "the MF model must be in IRODORI_HUB_DIR"
        );

        let mut g = default_generation();
        g.insert("seed".into(), serde_json::json!(1));
        g.insert("seconds".into(), serde_json::json!(1.0));
        let job = SynthJob::new("こんにちは、世界。", MF, g.clone());
        let record = synth.synthesize(&job).unwrap();
        assert_eq!(job.progress.load(Ordering::SeqCst), 4);
        assert!(synth.is_loaded());
        let path = &record.paths[0];
        assert!(path.is_absolute() && path.starts_with(dir.path().join("outputs/internal")));
        let reader = hound::WavReader::open(path).unwrap();
        assert_eq!(
            (reader.spec().bits_per_sample, reader.spec().channels),
            (24, 1)
        );
        assert!(
            reader.len() as f32 / reader.spec().sample_rate as f32 > 0.5,
            "about one second of audio"
        );
        let meta: serde_json::Value = serde_json::from_slice(
            &std::fs::read(path.parent().unwrap().join("generation.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            (meta["id"].as_str(), meta["seed"].as_u64()),
            (Some(record.id.as_str()), Some(1))
        );

        // 実行中の取消
        let job = SynthJob::new("止めてください。", MF, g);
        let (cancel, progress) = (Arc::clone(&job.cancel), Arc::clone(&job.progress));
        let watcher = std::thread::spawn(move || {
            while progress.load(Ordering::SeqCst) < 1 {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            cancel.store(true, Ordering::SeqCst);
        });
        let result = synth.synthesize(&job);
        watcher.join().unwrap();
        assert_eq!(result.err(), Some(SynthError::Cancelled));

        synth.unload();
        assert!(!synth.is_loaded());
    }

    /// 参照音声つき（`IRODORI_REF_WAV` が参照用の WAV を指すとき）。話者を使う通常の生成と同じ経路。
    #[test]
    fn real_generation_with_a_reference_voice_finishes() {
        let (Some(hub), Some(reference)) = (
            std::env::var_os("IRODORI_HUB_DIR"),
            std::env::var_os("IRODORI_REF_WAV"),
        ) else {
            eprintln!("skip: IRODORI_HUB_DIR / IRODORI_REF_WAV not set");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let synth = RealSynth::new(dir.path().to_path_buf(), hub.into());
        let mut job = SynthJob::new(
            "こんにちは。今日はいい天気ですね。",
            MF,
            default_generation(),
        );
        job.caption = "落ち着いた女性の声".into();
        job.references = vec![reference.into()];
        let record = synth.synthesize(&job).unwrap();
        assert!(record.paths[0].exists());
        assert_eq!(job.progress.load(Ordering::SeqCst), 4);
    }
}
