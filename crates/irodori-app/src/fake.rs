//! テストと開発用の擬似エンジン。モデルを読まずに、決まった長さのサイン波を書く。
//!
//! 取消・進捗・失敗・panic・モデル未取得を再現できるので、ジョブ管理と HTTP の全経路をモデル無しで試せる。

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use irodori_store::model::uid;
use serde_json::Value;

use crate::synth::{SynthError, SynthJob, SynthRecord, Synthesizer};

const SAMPLE_RATE: u32 = 24_000;
/// 0.2 秒。
const CLIP_SAMPLES: usize = 4_800;

pub struct FakeSynth {
    root: PathBuf,
    steps: usize,
    step_delay: Duration,
    failures: HashMap<String, SynthError>,
    panics: HashSet<String>,
    missing: HashSet<String>,
    calls: Mutex<Vec<String>>,
    loaded: AtomicBool,
    on_success: Option<Box<dyn Fn(&Path) + Send + Sync>>,
}

impl FakeSynth {
    pub fn new(root: &Path) -> FakeSynth {
        FakeSynth {
            root: root.to_path_buf(),
            steps: 4,
            step_delay: Duration::ZERO,
            failures: HashMap::new(),
            panics: HashSet::new(),
            missing: HashSet::new(),
            calls: Mutex::new(Vec::new()),
            loaded: AtomicBool::new(false),
            on_success: None,
        }
    }

    pub fn with_steps(mut self, steps: usize) -> FakeSynth {
        self.steps = steps;
        self
    }

    pub fn with_step_delay(mut self, delay: Duration) -> FakeSynth {
        self.step_delay = delay;
        self
    }

    pub fn fail_on(mut self, text: &str, error: SynthError) -> FakeSynth {
        self.failures.insert(text.to_string(), error);
        self
    }

    pub fn panic_on(mut self, text: &str) -> FakeSynth {
        self.panics.insert(text.to_string());
        self
    }

    pub fn missing_models(mut self, models: &[&str]) -> FakeSynth {
        self.missing.extend(models.iter().map(|m| m.to_string()));
        self
    }

    /// 合成に成功して結果を返す直前に、データのルートを渡して呼ぶ（保存先を壊すなど、失敗の再現に使う）。
    pub fn on_success(mut self, hook: impl Fn(&Path) + Send + Sync + 'static) -> FakeSynth {
        self.on_success = Some(Box::new(hook));
        self
    }

    /// これまでに合成を頼まれた文章（呼ばれた順）。
    pub fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

fn sine_clip() -> Vec<f32> {
    (0..CLIP_SAMPLES)
        .map(|i| 0.2 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin())
        .collect()
}

impl Synthesizer for FakeSynth {
    fn synthesize(&self, job: &SynthJob) -> Result<SynthRecord, SynthError> {
        self.calls
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(job.text.clone());
        if self.panics.contains(&job.text) {
            panic!("fake synthesizer panic");
        }
        if let Some(error) = self.failures.get(&job.text) {
            return Err(error.clone());
        }
        if self.missing.contains(&job.model) {
            return Err(SynthError::Invalid(
                "モデルがまだ取得されていません。".into(),
            ));
        }
        self.loaded.store(true, Ordering::SeqCst);
        let seed = job
            .generation
            .get("seed")
            .and_then(Value::as_u64)
            .unwrap_or(1);
        let id = uid();
        let folder = self.root.join("outputs").join("internal").join(&id);
        let mut paths = Vec::with_capacity(job.candidates.max(1));
        for candidate in 0..job.candidates.max(1) {
            job.progress.store(0, Ordering::SeqCst);
            job.stage.store(
                irodori_core::control::Stage::Sampling as usize,
                Ordering::SeqCst,
            );
            for step in 0..self.steps {
                if job.cancel.load(Ordering::SeqCst) {
                    return Err(SynthError::Cancelled);
                }
                if !self.step_delay.is_zero() {
                    std::thread::sleep(self.step_delay);
                }
                job.progress.store(step + 1, Ordering::SeqCst);
            }
            std::fs::create_dir_all(&folder).map_err(|e| SynthError::Failed(e.to_string()))?;
            let path = folder.join(format!("{:02}.wav", candidate + 1));
            irodori_core::audio::write_wav(&path, &sine_clip(), SAMPLE_RATE)
                .map_err(|e| SynthError::Failed(e.to_string()))?;
            paths.push(path);
        }
        if let Some(hook) = &self.on_success {
            hook(&self.root);
        }
        Ok(SynthRecord {
            id,
            model: job.model.clone(),
            text: job.text.clone(),
            caption: job.caption.clone(),
            references: job
                .references
                .iter()
                .map(|p| p.display().to_string())
                .collect(),
            generation: Value::Object(job.generation.clone()),
            seed,
            paths,
            seconds: 0.0,
            messages: Vec::new(),
        })
    }

    fn unload(&self) {
        self.loaded.store(false, Ordering::SeqCst);
    }

    fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::SeqCst)
    }

    fn model_available(&self, model: &str) -> bool {
        !self.missing.contains(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{SynthError, SynthJob, Synthesizer};
    use irodori_store::generation::default_generation;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;
    use std::time::Duration;

    fn job(text: &str) -> SynthJob {
        SynthJob::new(
            text,
            "Aratako/Irodori-TTS-v4.1-Small-MF",
            default_generation(),
        )
    }

    #[test]
    fn fake_synth_writes_24bit_wavs_and_reports_progress() {
        let dir = tempfile::tempdir().unwrap();
        let synth = FakeSynth::new(dir.path()).with_steps(4);
        let j = job("こんにちは");
        let record = synth.synthesize(&j).unwrap();
        assert_eq!(j.progress.load(Ordering::SeqCst), 4);
        assert_eq!(record.paths.len(), 1);
        let path = &record.paths[0];
        assert!(path.is_absolute() && path.exists(), "{path:?}");
        assert_eq!(
            path.parent()
                .unwrap()
                .file_name()
                .unwrap()
                .to_string_lossy(),
            record.id
        );
        assert!(path.starts_with(dir.path().join("outputs/internal")));
        let spec = hound::WavReader::open(path).unwrap().spec();
        assert_eq!(
            (spec.bits_per_sample, spec.channels, spec.sample_rate),
            (24, 1, 24000)
        );
        assert_eq!(
            hound::WavReader::open(path).unwrap().len(),
            4800,
            "0.2 s of audio"
        );
        assert_eq!(
            (record.text.as_str(), record.model.as_str()),
            ("こんにちは", "Aratako/Irodori-TTS-v4.1-Small-MF")
        );
        assert_eq!(synth.calls(), ["こんにちは"]);
    }

    #[test]
    fn fake_synth_stops_between_steps_when_cancelled() {
        let dir = tempfile::tempdir().unwrap();
        let synth = FakeSynth::new(dir.path())
            .with_steps(50)
            .with_step_delay(Duration::from_millis(10));
        let j = job("止める");
        let (cancel, progress) = (Arc::clone(&j.cancel), Arc::clone(&j.progress));
        let watcher = std::thread::spawn(move || {
            while progress.load(Ordering::SeqCst) < 2 {
                std::thread::sleep(Duration::from_millis(1));
            }
            cancel.store(true, Ordering::SeqCst);
        });
        let result = synth.synthesize(&j);
        watcher.join().unwrap();
        assert_eq!(result.err(), Some(SynthError::Cancelled));
        let done = j.progress.load(Ordering::SeqCst);
        assert!((2..50).contains(&done), "stopped after {done} steps");
        assert!(
            !dir.path().join("outputs").exists(),
            "no WAV is written for a cancelled job"
        );
    }

    #[test]
    fn fake_synth_candidates_produce_one_file_each() {
        let dir = tempfile::tempdir().unwrap();
        let synth = FakeSynth::new(dir.path()).with_steps(2);
        let mut j = job("候補");
        j.candidates = 3;
        let record = synth.synthesize(&j).unwrap();
        let names: Vec<String> = record
            .paths
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["01.wav", "02.wav", "03.wav"]);
        assert!(record.paths.iter().all(|p| p.exists()));
    }

    #[test]
    fn fake_synth_can_fail_or_panic_on_a_text() {
        let dir = tempfile::tempdir().unwrap();
        let synth = FakeSynth::new(dir.path())
            .fail_on(
                "だめ",
                SynthError::Invalid("文章を確認してください。".into()),
            )
            .panic_on("爆発");
        assert_eq!(
            synth.synthesize(&job("だめ")).err(),
            Some(SynthError::Invalid("文章を確認してください。".into()))
        );
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            synth.synthesize(&job("爆発"))
        }));
        assert!(caught.is_err(), "the fake panics on request");
        assert!(
            synth.synthesize(&job("普通")).is_ok(),
            "other texts still work"
        );
    }

    #[test]
    fn fake_synth_tracks_loading_and_model_availability() {
        let dir = tempfile::tempdir().unwrap();
        let synth =
            FakeSynth::new(dir.path()).missing_models(&["phasefield-audio/Irodori-TTS-v4.1-Anime"]);
        assert!(!synth.is_loaded());
        synth.synthesize(&job("読み込む")).unwrap();
        assert!(synth.is_loaded());
        synth.unload();
        assert!(!synth.is_loaded());
        assert!(synth.model_available("Aratako/Irodori-TTS-v4.1-Small"));
        assert!(!synth.model_available("phasefield-audio/Irodori-TTS-v4.1-Anime"));
    }
}
