//! テキスト → 波形の合成パイプライン。
//!
//! 上流 `InferenceRuntime.synthesize`（inference_runtime.py 1028–1495）の単一候補・透かしなしの経路。
//! 流れ: 正規化・トークナイズ → 参照音声のラウドネス正規化と encode → 条件エンコード →
//! 長さ決定（手動 / duration 予測）→ サンプリング → 末尾トリム → codec decode。

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};

use crate::audio::loudness::normalize_loudness;
use crate::audio::load_reference;
use crate::codec::{Codec, DECODE_CHUNK_FRAMES, DECODE_OVERLAP_FRAMES, ENCODE_CHUNK_FRAMES, ENCODE_OVERLAP_FRAMES};
use crate::conditioner::{ConditionInputs, Conditioner};
use crate::config::{ModelConfig, TextEncoderConfig};
use crate::dit::Dit;
use crate::duration::DurationPredictor;
use crate::loader::Weights;
use crate::normalize::normalize_text;
use crate::sampler::{initial_noise, sample_euler_cfg, sample_meanflow, SamplerParams};
use crate::text_tokenizer::TextTokenizer;
use crate::{IrodoriError, Result};

/// 参照音声のラウドネス目標（上流 `ref_normalize_db` の既定値）。
const REF_TARGET_DB: f64 = -16.0;
/// duration 予測を持たないモデルの既定長（秒）。
const FALLBACK_SECONDS: f32 = 30.0;

#[derive(Clone)]
pub struct SynthRequest {
    pub text: String,
    pub caption: Option<String>,
    pub ref_wav: Option<PathBuf>,
    /// `None` なら時刻から決める。
    pub seed: Option<u64>,
    /// 指定すると duration 予測を使わずこの長さにする（`min/max_seconds` でクランプ）。
    pub seconds: Option<f32>,
    pub duration_scale: f32,
    pub min_seconds: f32,
    pub max_seconds: f32,
    /// `None` なら RF は `sampler.num_steps`、MeanFlow は 4。
    pub num_steps: Option<usize>,
    pub sampler: SamplerParams,
    pub trim_tail: bool,
    pub tail_window: usize,
    pub tail_std_threshold: f32,
    pub tail_mean_threshold: f32,
    /// テスト用: 初期ノイズを外から与える（形状 `(1, T_patched, patched_latent_dim)`）。
    pub noise: Option<Tensor>,
}

impl Default for SynthRequest {
    fn default() -> Self {
        Self {
            text: String::new(),
            caption: None,
            ref_wav: None,
            seed: None,
            seconds: None,
            duration_scale: 1.0,
            min_seconds: 0.5,
            max_seconds: 30.0,
            num_steps: None,
            sampler: SamplerParams::default(),
            trim_tail: true,
            tail_window: 20,
            tail_std_threshold: 0.05,
            tail_mean_threshold: 0.1,
            noise: None,
        }
    }
}

impl SynthRequest {
    pub fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(IrodoriError::InvalidInput(m));
        if let Some(s) = self.seconds {
            if !(s > 0.0) {
                return bad(format!("seconds must be > 0, got {s}"));
            }
        }
        if !(self.duration_scale > 0.0) {
            return bad(format!("duration_scale must be > 0, got {}", self.duration_scale));
        }
        if !(self.min_seconds > 0.0) || self.max_seconds < self.min_seconds {
            return bad(format!("need 0 < min_seconds <= max_seconds, got {} / {}", self.min_seconds, self.max_seconds));
        }
        if self.num_steps == Some(0) {
            return bad("num_steps must be > 0".into());
        }
        Ok(())
    }
}

pub struct SynthOutput {
    /// モノラル波形（`sample_rate` Hz、末尾トリム済み）。
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub seed: u64,
    /// サンプリング直後の潜在 `(1, latent_steps, latent_dim)`。
    pub latent: Tensor,
    pub latent_steps: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub struct FramePlan {
    pub latent_steps: usize,
    pub target_samples: usize,
}

/// 手動の長さ: `[min, max]` にクランプし、フレーム数は切り上げ。
pub fn plan_manual(seconds: f32, min_s: f32, max_s: f32, sr: u32, hop: usize) -> FramePlan {
    let clamped = seconds.max(min_s).min(max_s);
    let target = ((clamped * sr as f32) as usize).max(1);
    FramePlan { latent_steps: target.div_ceil(hop), target_samples: target }
}

/// duration 予測（`log1p(frames)`）から長さを決める。
pub fn plan_predicted(log_frames: f32, scale: f32, min_s: f32, max_s: f32, sr: u32, hop: usize) -> FramePlan {
    let frames = log_frames.exp_m1() * scale;
    let min_frames = ((min_s * sr as f32 / hop as f32).ceil() as usize).max(1);
    let max_frames = ((max_s * sr as f32 / hop as f32).floor() as usize).max(1);
    let steps = (frames.round().max(0.0) as usize).clamp(min_frames, max_frames.max(min_frames));
    FramePlan { latent_steps: steps, target_samples: steps * hop }
}

/// 末尾が平坦（ほぼ無音）になる最初の位置。見つからなければ全長。
pub fn find_flattening_point(latent: &[Vec<f32>], window: usize, std_threshold: f32, mean_threshold: f32) -> usize {
    let total = latent.len();
    if total == 0 || window == 0 {
        return total;
    }
    let dim = latent[0].len();
    let zero = vec![0f32; dim];
    let row = |i: usize| -> &Vec<f32> { if i < total { &latent[i] } else { &zero } };
    for i in 0..total {
        let n = (window * dim) as f32;
        let mean = (i..i + window).flat_map(|r| row(r).iter()).sum::<f32>() / n;
        let var = (i..i + window).flat_map(|r| row(r).iter()).map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        if var.sqrt() < std_threshold && mean.abs() < mean_threshold {
            return i;
        }
    }
    total
}

/// トリム後の最大サンプル数。
pub fn trim_limit(target_samples: usize, flattening_frames: usize, hop: usize, trim_tail: bool) -> usize {
    let flat = flattening_frames * hop;
    if trim_tail && flat > 0 {
        target_samples.min(flat)
    } else {
        target_samples
    }
}

fn random_seed() -> u64 {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
    nanos ^ ((std::process::id() as u64) << 32)
}

/// 重みの置き方。`Staged` は段階（参照 encode → 条件 → サンプリング → decode）ごとに必要な部品だけを
/// mmap から読み、使い終えたら解放する（ピークメモリは最大の部品 1 つ分。呼び出しごとに再読み込みする）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryMode {
    Resident,
    Staged,
}

#[derive(Clone, Debug)]
pub struct PipelineOptions {
    /// DiT / テキストエンコーダ / speaker / duration の重みの dtype。
    pub dtype: DType,
    /// codec の重みの dtype（入出力は常に F32）。
    pub codec_dtype: DType,
    pub memory: MemoryMode,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self { dtype: DType::F32, codec_dtype: DType::F32, memory: MemoryMode::Resident }
    }
}

struct Resident {
    conditioner: Conditioner,
    dit: Dit,
    duration: Option<DurationPredictor>,
    codec: Codec,
}

/// 常駐部品の借用か、段階ロードした所有物。
enum Held<'a, T> {
    Ref(&'a T),
    Own(T),
}

impl<T> std::ops::Deref for Held<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        match self {
            Held::Ref(r) => r,
            Held::Own(o) => o,
        }
    }
}

pub struct Pipeline {
    cfg: ModelConfig,
    te: TextEncoderConfig,
    tokenizer: TextTokenizer,
    weights: Weights,
    codec_pth: PathBuf,
    device: Device,
    opts: PipelineOptions,
    sample_rate: u32,
    hop: usize,
    resident: Option<Resident>,
}

impl Pipeline {
    /// F32・常駐で読む。
    pub fn load(model_safetensors: &Path, codec_pth: &Path, device: &Device) -> Result<Self> {
        Self::load_with(model_safetensors, codec_pth, device, PipelineOptions::default())
    }

    /// `model_safetensors` と同じフォルダの `tokenizer/tokenizer.json`、codec の `weights.pth` を読む。
    pub fn load_with(model_safetensors: &Path, codec_pth: &Path, device: &Device, opts: PipelineOptions) -> Result<Self> {
        let weights = Weights::open(model_safetensors, device)?;
        let (cfg, te) = weights.config()?;
        let tok_path = model_safetensors.parent().unwrap_or(Path::new(".")).join("tokenizer").join("tokenizer.json");
        let tokenizer = TextTokenizer::from_file(&tok_path, te.bos_token_id, te.pad_token_id)?;
        let codec = Codec::load_with_dtype(codec_pth, device, opts.codec_dtype)?;
        let (sample_rate, hop) = (codec.sample_rate(), codec.hop());
        let mut p = Self { cfg, te, tokenizer, weights, codec_pth: codec_pth.to_path_buf(), device: device.clone(), opts, sample_rate, hop, resident: None };
        if p.opts.memory == MemoryMode::Resident {
            p.resident = Some(Resident { conditioner: p.load_conditioner()?, dit: p.load_dit()?, duration: p.load_duration()?, codec });
        }
        Ok(p)
    }

    fn load_conditioner(&self) -> Result<Conditioner> {
        Conditioner::load(self.weights.var_builder(self.opts.dtype)?, &self.cfg, &self.te)
    }

    fn load_dit(&self) -> Result<Dit> {
        Dit::load(self.weights.var_builder(self.opts.dtype)?, &self.cfg)
    }

    fn load_duration(&self) -> Result<Option<DurationPredictor>> {
        if !self.cfg.use_duration_predictor {
            return Ok(None);
        }
        let vb = self.weights.var_builder(self.opts.dtype)?.pp("duration_predictor");
        Ok(Some(DurationPredictor::load(vb, &self.cfg)?))
    }

    fn conditioner(&self) -> Result<Held<'_, Conditioner>> {
        Ok(match &self.resident {
            Some(r) => Held::Ref(&r.conditioner),
            None => Held::Own(self.load_conditioner()?),
        })
    }

    fn dit(&self) -> Result<Held<'_, Dit>> {
        Ok(match &self.resident {
            Some(r) => Held::Ref(&r.dit),
            None => Held::Own(self.load_dit()?),
        })
    }

    fn duration(&self) -> Result<Held<'_, Option<DurationPredictor>>> {
        Ok(match &self.resident {
            Some(r) => Held::Ref(&r.duration),
            None => Held::Own(self.load_duration()?),
        })
    }

    fn codec(&self) -> Result<Held<'_, Codec>> {
        Ok(match &self.resident {
            Some(r) => Held::Ref(&r.codec),
            None => Held::Own(Codec::load_with_dtype(&self.codec_pth, &self.device, self.opts.codec_dtype)?),
        })
    }

    pub fn memory_mode(&self) -> MemoryMode {
        self.opts.memory
    }

    fn token_tensors(&self, text: &str, max_len: usize, force_empty: bool) -> Result<(Tensor, Tensor)> {
        let n = self.tokenizer.encode(text)?.len().min(max_len).max(1);
        let (ids, mask) = self.tokenizer.batch_encode(&[text], n)?;
        if force_empty {
            return Ok((ids, mask.zeros_like()?));
        }
        Ok((ids, mask))
    }

    /// 参照音声 → パッチ化済み潜在とマスク。無ければ「参照なし」（全マスク）。
    fn reference(&self, path: Option<&Path>) -> Result<(Tensor, Tensor)> {
        if self.cfg.latent_patch_size != 1 {
            return Err(IrodoriError::InvalidConfig("latent_patch_size != 1 is not supported".into()));
        }
        let Some(path) = path else {
            let len = self.cfg.speaker_patch_size.max(1);
            let z = Tensor::zeros((1, len, self.cfg.latent_dim), DType::F32, &self.device)?;
            return Ok((z, Tensor::zeros((1, len), DType::U8, &self.device)?));
        };
        let sr = self.sample_rate;
        let mut wav = load_reference(path, sr, self.cfg.ref_max_seconds as f32)?;
        normalize_loudness(&mut wav, sr, REF_TARGET_DB);
        let n = wav.len();
        let wav = Tensor::from_vec(wav, (1, 1, n), &self.device)?;
        let mut z = self.codec()?.encode_chunked(&wav, ENCODE_CHUNK_FRAMES, ENCODE_OVERLAP_FRAMES)?;
        let max_steps = ((self.cfg.ref_max_seconds * sr as f64 / self.hop as f64).ceil() as usize).max(1);
        if z.dim(1)? > max_steps {
            z = z.narrow(1, 0, max_steps)?;
        }
        let t = z.dim(1)?;
        Ok((z, Tensor::ones((1, t), DType::U8, &self.device)?))
    }

    pub fn synthesize(&self, req: &SynthRequest) -> Result<SynthOutput> {
        req.validate()?;
        let cfg = &self.cfg;
        let seed = req.seed.unwrap_or_else(random_seed);
        let (sr, hop) = (self.sample_rate, self.hop);

        let text = normalize_text(&req.text).trim().to_string();
        if text.is_empty() {
            return Err(IrodoriError::EmptyText);
        }
        let n_tokens = self.tokenizer.encode(&text)?.len();
        if n_tokens > cfg.max_text_len {
            return Err(IrodoriError::TextTooLong { tokens: n_tokens, max: cfg.max_text_len });
        }
        let (text_ids, text_mask) = self.token_tensors(&text, cfg.max_text_len, false)?;
        let caption_text = req.caption.as_deref().map(str::trim).unwrap_or("");
        let has_caption = cfg.use_caption_condition && !caption_text.is_empty();
        let caption = cfg
            .use_caption_condition
            .then(|| self.token_tensors(caption_text, cfg.max_caption_len.unwrap_or(512), caption_text.is_empty()))
            .transpose()?;

        // 段階 1: 参照音声の encode（codec）
        let reference = cfg.use_speaker_condition_resolved().then(|| self.reference(req.ref_wav.as_deref())).transpose()?;
        let has_speaker = req.ref_wav.is_some() && reference.is_some();

        // 段階 2: 条件エンコードと長さ決定（テキストエンコーダ・speaker・duration）
        let (enc, plan) = {
            let conditioner = self.conditioner()?;
            let enc = conditioner.encode(&ConditionInputs {
                text_ids: &text_ids,
                text_mask: &text_mask,
                caption_ids: caption.as_ref().map(|c| &c.0),
                caption_mask: caption.as_ref().map(|c| &c.1),
                ref_latent: reference.as_ref().map(|r| &r.0),
                ref_mask: reference.as_ref().map(|r| &r.1),
            })?;
            let plan = match req.seconds {
                Some(s) => plan_manual(s, req.min_seconds, req.max_seconds, sr, hop),
                None => match &*self.duration()? {
                    Some(d) => {
                        let log_frames = d.predict_log_frames(&enc, &[has_speaker], &[has_caption])?.to_vec1::<f32>()?[0];
                        plan_predicted(log_frames, req.duration_scale, req.min_seconds, req.max_seconds, sr, hop)
                    }
                    None => plan_manual(FALLBACK_SECONDS, req.min_seconds, req.max_seconds.max(FALLBACK_SECONDS), sr, hop),
                },
            };
            (enc, plan)
        };

        // 段階 3: サンプリング（DiT）
        let dim = cfg.patched_latent_dim();
        let noise = match &req.noise {
            Some(n) => {
                if n.dims() != [1, plan.latent_steps, dim] {
                    return Err(IrodoriError::InvalidInput(format!("noise must have shape (1, {}, {dim}), got {:?}", plan.latent_steps, n.dims())));
                }
                n.clone()
            }
            None => initial_noise(seed, 1, plan.latent_steps, dim, &self.device)?,
        };
        let z = {
            let dit = self.dit()?;
            if cfg.flow_parameterization.trim().eq_ignore_ascii_case("meanflow") {
                sample_meanflow(&dit, &enc, req.num_steps.unwrap_or(4), &noise)?
            } else {
                let params = SamplerParams { num_steps: req.num_steps.unwrap_or(req.sampler.num_steps), ..req.sampler.clone() };
                sample_euler_cfg(&dit, &enc, &params, &noise)?
            }
        };
        drop(enc);
        let z = z.narrow(1, 0, plan.latent_steps)?;

        // 段階 4: decode（codec）
        let rows = z.squeeze(0)?.to_vec2::<f32>()?;
        let flat = find_flattening_point(&rows, req.tail_window.max(1), req.tail_std_threshold, req.tail_mean_threshold);
        let limit = trim_limit(plan.target_samples, flat, hop, req.trim_tail);
        let mut samples = self.codec()?.decode_chunked(&z, DECODE_CHUNK_FRAMES, DECODE_OVERLAP_FRAMES)?.flatten_all()?.to_vec1::<f32>()?;
        samples.truncate(limit);
        Ok(SynthOutput { samples, sample_rate: sr, seed, latent: z, latent_steps: plan.latent_steps })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48000;
    const HOP: usize = 1920;

    #[test]
    fn manual_seconds_round_up_to_whole_frames() {
        let p = plan_manual(1.0, 0.5, 30.0, SR, HOP);
        assert_eq!((p.latent_steps, p.target_samples), (25, 48000));
        let p = plan_manual(0.9, 0.5, 30.0, SR, HOP);
        assert_eq!((p.latent_steps, p.target_samples), (23, 43200)); // ceil(43200/1920)=23
    }

    #[test]
    fn manual_seconds_are_clamped_to_the_allowed_range() {
        assert_eq!(plan_manual(100.0, 0.5, 30.0, SR, HOP).latent_steps, 750);
        let p = plan_manual(0.1, 0.5, 30.0, SR, HOP);
        assert_eq!((p.latent_steps, p.target_samples), (13, 24000));
    }

    #[test]
    fn predicted_frames_are_scaled_rounded_and_clamped() {
        let log1p = |frames: f32| (1.0 + frames).ln();
        assert_eq!(plan_predicted(log1p(100.0), 1.0, 0.5, 30.0, SR, HOP).latent_steps, 100);
        assert_eq!(plan_predicted(log1p(100.0), 2.0, 0.5, 30.0, SR, HOP).latent_steps, 200);
        assert_eq!(plan_predicted(log1p(5000.0), 1.0, 0.5, 30.0, SR, HOP).latent_steps, 750, "upper clamp");
        assert_eq!(plan_predicted(0.0, 1.0, 0.5, 30.0, SR, HOP).latent_steps, 13, "lower clamp");
        let p = plan_predicted(log1p(100.0), 1.0, 0.5, 30.0, SR, HOP);
        assert_eq!(p.target_samples, 100 * HOP);
    }

    fn latent(rows: &[[f32; 2]]) -> Vec<Vec<f32>> {
        rows.iter().map(|r| r.to_vec()).collect()
    }

    #[test]
    fn flattening_point_is_where_the_tail_goes_quiet() {
        let mut rows = vec![[1.0, -1.0]; 6];
        rows.extend([[0.0, 0.0]; 4]);
        assert_eq!(find_flattening_point(&latent(&rows), 3, 0.05, 0.1), 6);
    }

    #[test]
    fn flattening_point_defaults_to_the_full_length_when_nothing_is_flat() {
        let rows = vec![[1.0, -1.0]; 8];
        assert_eq!(find_flattening_point(&latent(&rows), 3, 0.05, 0.1), 8);
        assert_eq!(find_flattening_point(&[], 3, 0.05, 0.1), 0);
    }

    #[test]
    fn trim_limit_uses_the_earlier_of_target_and_flattening() {
        assert_eq!(trim_limit(48000, 10, HOP, true), 10 * HOP);
        assert_eq!(trim_limit(48000, 100, HOP, true), 48000);
        assert_eq!(trim_limit(48000, 0, HOP, true), 48000, "no flattening found");
        assert_eq!(trim_limit(48000, 10, HOP, false), 48000, "trim disabled");
    }

    #[test]
    fn invalid_requests_are_rejected_before_any_work() {
        let ok = SynthRequest { text: "a".into(), ..SynthRequest::default() };
        assert!(ok.validate().is_ok());
        for bad in [
            SynthRequest { seconds: Some(0.0), ..ok.clone() },
            SynthRequest { duration_scale: 0.0, ..ok.clone() },
            SynthRequest { min_seconds: 5.0, max_seconds: 1.0, ..ok.clone() },
            SynthRequest { num_steps: Some(0), ..ok.clone() },
        ] {
            assert!(bad.validate().is_err());
        }
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    /// 誤差の二乗平均平方根 / 正解の二乗平均平方根。
    fn rel_rms(a: &[f32], b: &[f32]) -> f32 {
        let err: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        let sig: f32 = b.iter().map(|y| y * y).sum();
        (err / sig.max(1e-12)).sqrt()
    }

    /// 参照なし・1 秒・8 ステップの合成を指定オプションで実行し、(潜在の最大誤差, 波形の最大誤差) を返す。
    fn run_e2e(opts: PipelineOptions) -> Option<(f32, f32, f32)> {
        use candle_core::Device;
        let (Some(model_dir), Some(codec_dir), Some(_)) =
            (std::env::var_os("IRODORI_MODEL_DIR"), std::env::var_os("IRODORI_CODEC_DIR"), crate::golden::dir())
        else {
            eprintln!("skip: IRODORI_MODEL_DIR / IRODORI_CODEC_DIR / IRODORI_GOLDEN_DIR not set");
            return None;
        };
        let p = Pipeline::load_with(
            &std::path::Path::new(&model_dir).join("model.safetensors"),
            &std::path::Path::new(&codec_dir).join("weights.pth"),
            &Device::Cpu,
            opts,
        )
        .unwrap();
        let g = crate::golden::load("e2e").unwrap();
        let req = SynthRequest {
            text: "こんにちは、世界。".into(),
            caption: Some("落ち着いた女性の声".into()),
            seconds: Some(1.0),
            num_steps: Some(8),
            trim_tail: false,
            noise: Some(g["noise"].clone()),
            ..SynthRequest::default()
        };
        let out = p.synthesize(&req).unwrap();
        let lat: Vec<f32> = out.latent.flatten_all().unwrap().to_vec1().unwrap();
        let want_lat: Vec<f32> = g["latent"].flatten_all().unwrap().to_vec1().unwrap();
        let want_audio: Vec<f32> = g["audio"].flatten_all().unwrap().to_vec1().unwrap();
        let (lat_rel, audio_rel) = (rel_rms(&lat, &want_lat), rel_rms(&out.samples, &want_audio[..out.samples.len()]));
        eprintln!("relative rms error: latent {lat_rel:.3e}, audio {audio_rel:.3e}");
        Some((max_abs(&lat, &want_lat), max_abs(&out.samples, &want_audio), lat_rel))
    }

    #[test]
    fn real_meanflow_model_matches_python() {
        use candle_core::Device;
        let (Some(mf_dir), Some(codec_dir), Some(_)) =
            (std::env::var_os("IRODORI_MF_DIR"), std::env::var_os("IRODORI_CODEC_DIR"), crate::golden::dir())
        else {
            eprintln!("skip: IRODORI_MF_DIR / IRODORI_CODEC_DIR / IRODORI_GOLDEN_DIR not set");
            return;
        };
        let opts = PipelineOptions { memory: MemoryMode::Staged, ..PipelineOptions::default() };
        let p = Pipeline::load_with(
            &std::path::Path::new(&mf_dir).join("model.safetensors"),
            &std::path::Path::new(&codec_dir).join("weights.pth"),
            &Device::Cpu,
            opts,
        )
        .unwrap();
        let g = crate::golden::load("e2e_mf").unwrap();
        let req = SynthRequest {
            text: "こんにちは、世界。".into(),
            caption: Some("落ち着いた女性の声".into()),
            seconds: Some(1.0),
            trim_tail: false,
            noise: Some(g["noise"].clone()),
            ..SynthRequest::default()
        };
        let out = p.synthesize(&req).unwrap(); // num_steps 未指定 → MeanFlow の既定 4
        let lat: Vec<f32> = out.latent.flatten_all().unwrap().to_vec1().unwrap();
        let want: Vec<f32> = g["latent"].flatten_all().unwrap().to_vec1().unwrap();
        eprintln!("meanflow: latent max diff {:.2e}", max_abs(&lat, &want));
        assert!(max_abs(&lat, &want) < 1e-3, "meanflow latent max diff {}", max_abs(&lat, &want));
    }

    #[test]
    fn real_model_staged_f32_matches_python() {
        let Some((lat, audio, _)) = run_e2e(PipelineOptions { memory: MemoryMode::Staged, ..PipelineOptions::default() }) else { return };
        eprintln!("staged f32: latent {lat:.2e}, audio {audio:.2e}");
        assert!(lat < 1e-3 && audio < 1e-3, "latent {lat}, audio {audio}");
    }

    #[test]
    fn real_model_bf16_stays_close_to_python() {
        let opts = PipelineOptions { dtype: DType::BF16, codec_dtype: DType::F32, memory: MemoryMode::Staged };
        let Some((lat, audio, lat_rel)) = run_e2e(opts) else { return };
        eprintln!("bf16 weights: latent {lat:.2e}, audio {audio:.2e}");
        // 計画の基準: 潜在の相対誤差 < 5e-2（実測 4.1e-2）。
        assert!(lat_rel < 5e-2, "latent relative rms error {lat_rel}");
    }

    #[test]
    fn real_model_matches_python_end_to_end() {
        use candle_core::Device;
        let (Some(model_dir), Some(codec_dir), Some(_)) =
            (std::env::var_os("IRODORI_MODEL_DIR"), std::env::var_os("IRODORI_CODEC_DIR"), crate::golden::dir())
        else {
            eprintln!("skip: IRODORI_MODEL_DIR / IRODORI_CODEC_DIR / IRODORI_GOLDEN_DIR not set");
            return;
        };
        let p = Pipeline::load(
            &std::path::Path::new(&model_dir).join("model.safetensors"),
            &std::path::Path::new(&codec_dir).join("weights.pth"),
            &Device::Cpu,
        )
        .unwrap();
        let g = crate::golden::load("e2e").unwrap();
        let req = SynthRequest {
            text: "こんにちは、世界。".into(),
            caption: Some("落ち着いた女性の声".into()),
            seconds: Some(1.0),
            num_steps: Some(8),
            trim_tail: false,
            noise: Some(g["noise"].clone()),
            ..SynthRequest::default()
        };
        let out = p.synthesize(&req).unwrap();
        assert_eq!(out.latent_steps, 25);
        crate::golden::assert_close(&out.latent, &g["latent"], 1e-3, 1e-3).unwrap_or_else(|e| panic!("latent: {e}"));
        let want: Vec<f32> = g["audio"].flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(out.samples.len(), want.len().min(48000));
        let max_diff = out.samples.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        let lat_diff = {
            let a: Vec<f32> = out.latent.flatten_all().unwrap().to_vec1().unwrap();
            let b: Vec<f32> = g["latent"].flatten_all().unwrap().to_vec1().unwrap();
            a.iter().zip(&b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
        };
        eprintln!("e2e max diff: latent {lat_diff:.2e}, audio {max_diff:.2e}");
        assert!(max_diff < 1e-3, "audio max diff {max_diff}");
    }
}
