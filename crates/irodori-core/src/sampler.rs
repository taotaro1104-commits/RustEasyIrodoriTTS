//! Euler サンプラ（RF + CFG）と MeanFlow サンプラ。
//!
//! 上流 `rf.py::sample_euler_rf_cfg` と `meanflow.py::sample_euler_meanflow` の推論経路。
//! CFG の各分岐は直列に評価してメモリを抑える（上流はバッチ連結）。無条件分岐は長さ 1・マスク 0 の
//! ダミー条件で表す（マスクされた文脈は出力に影響しないため、全長ゼロ詰めと等価で安い）。

use candle_core::{Device, Tensor};

use crate::conditioner::EncodedConditions;
use crate::control::Control;
use crate::dit::{ContextKv, Dit};
use crate::rng::normal_vec;
use crate::{IrodoriError, Result};

/// 初期ノイズに掛ける係数の上限（`t_schedule` の先頭値）。
const INIT_SCALE: f64 = 0.999;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuidanceMode {
    Independent,
    Joint,
    Alternating,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TSchedule {
    Linear,
    Sway,
}

#[derive(Clone, Debug)]
pub struct SamplerParams {
    pub num_steps: usize,
    pub cfg_text: f32,
    pub cfg_caption: f32,
    pub cfg_speaker: f32,
    pub guidance: GuidanceMode,
    pub cfg_min_t: f32,
    pub cfg_max_t: f32,
    pub truncation_factor: Option<f32>,
    /// `(rescale_k, rescale_sigma)`。両方そろったときだけ有効。
    pub rescale: Option<(f32, f32)>,
    pub schedule: TSchedule,
    pub sway_coeff: f32,
    pub speaker_kv_scale: Option<f32>,
    pub speaker_kv_min_t: f32,
    pub speaker_kv_max_layers: Option<usize>,
}

impl Default for SamplerParams {
    /// Easy 版 `DEFAULT_GENERATION` と同じ既定値。ただしステップ数だけは 40 でなく 24（docs/bench.md）。
    fn default() -> Self {
        Self {
            num_steps: 24,
            cfg_text: 3.0,
            cfg_caption: 4.0,
            cfg_speaker: 5.0,
            guidance: GuidanceMode::Independent,
            cfg_min_t: 0.5,
            cfg_max_t: 1.0,
            truncation_factor: None,
            rescale: None,
            schedule: TSchedule::Linear,
            sway_coeff: -1.0,
            speaker_kv_scale: None,
            speaker_kv_min_t: 0.9,
            speaker_kv_max_layers: None,
        }
    }
}

/// `t_schedule[i] = (1 - u_i) * 0.999`。長さ `num_steps + 1`、厳密に単調減少でなければエラー。
pub fn t_schedule(num_steps: usize, mode: TSchedule, sway_coeff: f32) -> Result<Vec<f32>> {
    if num_steps == 0 {
        return Err(IrodoriError::InvalidInput("num_steps must be > 0".into()));
    }
    if !sway_coeff.is_finite() {
        return Err(IrodoriError::InvalidInput(format!("sway_coeff must be finite, got {sway_coeff}")));
    }
    let t: Vec<f32> = (0..=num_steps)
        .map(|i| {
            let mut u = i as f64 / num_steps as f64;
            if mode == TSchedule::Sway {
                u = (u + sway_coeff as f64 * ((0.5 * std::f64::consts::PI * u).cos() + u - 1.0)).clamp(0.0, 1.0);
            }
            ((1.0 - u) * INIT_SCALE) as f32
        })
        .collect();
    if !t.windows(2).all(|w| w[0] > w[1]) {
        return Err(IrodoriError::InvalidInput("t_schedule must be strictly decreasing; adjust num_steps or sway_coeff".into()));
    }
    Ok(t)
}

/// Temporal score rescaling（arXiv:2510.01184）。`t >= 1` はそのまま返す。
pub fn temporal_score_rescale(v: &Tensor, x_t: &Tensor, t: f32, k: f32, sigma: f32) -> Result<Tensor> {
    if t >= 1.0 {
        return Ok(v.clone());
    }
    let (t, k, sigma) = (t as f64, k as f64, sigma as f64);
    let om = 1.0 - t;
    let snr = om * om / (t * t);
    let s2 = sigma * sigma;
    let ratio = (snr * s2 + 1.0) / (snr * s2 / k + 1.0);
    let inner = (v.affine(om, 0.0)? + x_t)?.affine(ratio, 0.0)?;
    Ok(((inner - x_t)? / om)?)
}

pub fn apply_truncation(noise: &Tensor, factor: Option<f32>) -> Result<Tensor> {
    match factor {
        Some(f) => Ok(noise.affine(f as f64, 0.0)?),
        None => Ok(noise.clone()),
    }
}

/// `(batch, len, dim)` の標準正規ノイズ（seed 固定）。
pub fn initial_noise(seed: u64, batch: usize, len: usize, dim: usize, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(normal_vec(seed, batch * len * dim), (batch, len, dim), device)?)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Text,
    Speaker,
    Caption,
}

/// 無条件の分岐: 長さ 1・マスク 0 のゼロ状態。
fn zero_branch(state: &Tensor) -> Result<(Tensor, Tensor)> {
    let (b, _, d) = state.dims3()?;
    Ok((Tensor::zeros((b, 1, d), state.dtype(), state.device())?, Tensor::zeros((b, 1), candle_core::DType::U8, state.device())?))
}

fn uncond(cond: &EncodedConditions, kinds: &[Kind]) -> Result<EncodedConditions> {
    let mut c = cond.clone();
    if kinds.contains(&Kind::Text) {
        (c.text, c.text_mask) = zero_branch(&cond.text)?;
    }
    if kinds.contains(&Kind::Speaker) {
        if let Some((s, _)) = &cond.speaker {
            c.speaker = Some(zero_branch(s)?);
        }
    }
    if kinds.contains(&Kind::Caption) {
        if let Some((s, _)) = &cond.caption {
            c.caption = Some(zero_branch(s)?);
        }
    }
    Ok(c)
}

fn any_valid(mask: &Tensor) -> Result<bool> {
    Ok(mask.to_dtype(candle_core::DType::F32)?.sum_all()?.to_scalar::<f32>()? > 0.0)
}

/// 各層の speaker K/V（`[k_text, v_text, k_spk, v_spk, ..]` の 2, 3 番目）を定数倍する。
fn scale_speaker_kv(cache: &mut ContextKv, scale: f64, max_layers: Option<usize>) -> Result<()> {
    let n = max_layers.map_or(cache.len(), |m| m.min(cache.len()));
    for layer in cache.iter_mut().take(n) {
        if layer.len() < 4 {
            return Err(IrodoriError::InvalidInput(format!("expected at least 4 tensors in a context KV entry, got {}", layer.len())));
        }
        layer[2] = layer[2].affine(scale, 0.0)?;
        layer[3] = layer[3].affine(scale, 0.0)?;
    }
    Ok(())
}

struct Branch {
    scale: f32,
    cond: EncodedConditions,
    cache: ContextKv,
}

fn velocity(dit: &Dit, x: &Tensor, tt: &Tensor, c: &EncodedConditions, cache: &ContextKv) -> Result<Tensor> {
    dit.forward_with_encoded_conditions(x, tt, c, None, Some(cache))
}

/// ガイダンスを掛ける項目（順序は text, speaker, caption）。speaker / caption は**有効なトークンが
/// 1 つでもあるとき**だけ。参照なし（全マスク）の speaker に無条件分岐を走らせても差分は 0 で、
/// forward が無駄になる（上流 `resolve_cfg_scales` も参照なしでは speaker のスケールを 0 にする）。
fn enabled_guidance(cond: &EncodedConditions, p: &SamplerParams) -> Result<Vec<(Kind, f32)>> {
    let valid = |c: &Option<(Tensor, Tensor)>| -> Result<bool> {
        match c {
            Some((_, m)) => any_valid(m),
            None => Ok(false),
        }
    };
    let mut enabled: Vec<(Kind, f32)> = Vec::new();
    if p.cfg_text > 0.0 {
        enabled.push((Kind::Text, p.cfg_text));
    }
    if p.cfg_speaker > 0.0 && valid(&cond.speaker)? {
        enabled.push((Kind::Speaker, p.cfg_speaker));
    }
    if p.cfg_caption > 0.0 && valid(&cond.caption)? {
        enabled.push((Kind::Caption, p.cfg_caption));
    }
    Ok(enabled)
}

/// RF の Euler サンプリング。`noise: (B, T, patched_latent_dim)`。戻り値は同形のパッチ化済み潜在。
pub fn sample_euler_cfg(dit: &Dit, cond: &EncodedConditions, p: &SamplerParams, noise: &Tensor) -> Result<Tensor> {
    sample_euler_cfg_ctl(dit, cond, p, noise, &Control::default())
}

/// `sample_euler_cfg` に取消・進捗のフックを付けたもの。各ステップの先頭で取消を確認し、終わりで進捗を進める。
pub fn sample_euler_cfg_ctl(dit: &Dit, cond: &EncodedConditions, p: &SamplerParams, noise: &Tensor, ctl: &Control) -> Result<Tensor> {
    let ts = t_schedule(p.num_steps, p.schedule, p.sway_coeff)?;
    let has_speaker = match &cond.speaker {
        Some((_, m)) => any_valid(m)?,
        None => false,
    };
    let enabled = enabled_guidance(cond, p)?;
    if p.guidance == GuidanceMode::Joint && enabled.len() > 1 {
        let (lo, hi) = enabled.iter().fold((f32::MAX, f32::MIN), |(lo, hi), (_, s)| (lo.min(*s), hi.max(*s)));
        if hi - lo > 1e-6 {
            return Err(IrodoriError::InvalidInput("cfg_guidance_mode='joint' requires equal enabled guidance scales".into()));
        }
    }

    let mut cond_cache = dit.build_context_kv_cache(cond)?;
    // independent / alternating は項目ごとの無条件分岐、joint は全項目を落とした 1 分岐。
    let mut branches: Vec<Branch> = Vec::new();
    if !enabled.is_empty() {
        match p.guidance {
            GuidanceMode::Joint => {
                let kinds: Vec<Kind> = [Kind::Text, Kind::Speaker, Kind::Caption].to_vec();
                let c = uncond(cond, &kinds)?;
                branches.push(Branch { scale: enabled[0].1, cache: dit.build_context_kv_cache(&c)?, cond: c });
            }
            _ => {
                for (kind, scale) in &enabled {
                    let c = uncond(cond, &[*kind])?;
                    branches.push(Branch { scale: *scale, cache: dit.build_context_kv_cache(&c)?, cond: c });
                }
            }
        }
    }
    let kv_scale = p.speaker_kv_scale.filter(|_| has_speaker).map(f64::from);
    if let Some(s) = kv_scale {
        scale_speaker_kv(&mut cond_cache, s, p.speaker_kv_max_layers)?;
        for b in &mut branches {
            scale_speaker_kv(&mut b.cache, s, p.speaker_kv_max_layers)?;
        }
    }
    let mut kv_active = kv_scale.is_some();

    let mut x = apply_truncation(noise, p.truncation_factor)?;
    let batch = x.dim(0)?;
    for i in 0..p.num_steps {
        ctl.check()?;
        let (t, t_next) = (ts[i], ts[i + 1]);
        let tt = Tensor::full(t, batch, x.device())?.to_dtype(x.dtype())?;
        let v_c = velocity(dit, &x, &tt, cond, &cond_cache)?;
        let use_cfg = !branches.is_empty() && p.cfg_min_t <= t && t <= p.cfg_max_t;
        let mut v = v_c.clone();
        if use_cfg {
            let picked: Vec<&Branch> = match p.guidance {
                GuidanceMode::Alternating => vec![&branches[i % branches.len()]],
                _ => branches.iter().collect(),
            };
            for b in picked {
                let v_u = velocity(dit, &x, &tt, &b.cond, &b.cache)?;
                v = (v + (&v_c - &v_u)?.affine(b.scale as f64, 0.0)?)?;
            }
        }
        if let Some((k, sigma)) = p.rescale {
            v = temporal_score_rescale(&v, &x, t, k, sigma)?;
        }
        if let (true, Some(s)) = (kv_active, kv_scale) {
            if t_next < p.speaker_kv_min_t && t >= p.speaker_kv_min_t {
                scale_speaker_kv(&mut cond_cache, 1.0 / s, p.speaker_kv_max_layers)?;
                for b in &mut branches {
                    scale_speaker_kv(&mut b.cache, 1.0 / s, p.speaker_kv_max_layers)?;
                }
                kv_active = false;
            }
        }
        x = (x + v.affine((t_next - t) as f64, 0.0)?)?;
        ctl.set_progress(i + 1);
    }
    Ok(x)
}

/// MeanFlow の線形サンプラ（`t: 1 → 0`、CFG なし、既定 4 ステップ）。
pub fn sample_meanflow(dit: &Dit, cond: &EncodedConditions, num_steps: usize, noise: &Tensor) -> Result<Tensor> {
    sample_meanflow_ctl(dit, cond, num_steps, noise, &Control::default())
}

/// `sample_meanflow` に取消・進捗のフックを付けたもの。
pub fn sample_meanflow_ctl(dit: &Dit, cond: &EncodedConditions, num_steps: usize, noise: &Tensor, ctl: &Control) -> Result<Tensor> {
    if num_steps == 0 {
        return Err(IrodoriError::InvalidInput("MeanFlow num_steps must be positive".into()));
    }
    let cache = dit.build_context_kv_cache(cond)?;
    let mut x = noise.clone();
    let batch = x.dim(0)?;
    let sched: Vec<f32> = (0..=num_steps).map(|i| 1.0 - i as f32 / num_steps as f32).collect();
    for i in 0..num_steps {
        ctl.check()?;
        let (t, next) = (sched[i], sched[i + 1]);
        let tt = Tensor::full(t, batch, x.device())?.to_dtype(x.dtype())?;
        let delta = Tensor::full(t - next, batch, x.device())?.to_dtype(x.dtype())?;
        let v = dit.forward_meanflow(&x, &tt, &delta, cond, None, Some(&cache))?;
        x = (x + v.affine((next - t) as f64, 0.0)?)?;
        ctl.set_progress(i + 1);
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conditioner::EncodedConditions;
    use crate::dit::tests::{conds, dit, dit_meanflow, mask, LATENT};
    use crate::golden::assert_close;
    use crate::nn::test_util::fake_tensor;
    use candle_core::{Device, Tensor};

    fn noise(len: usize, seed: &str) -> Tensor {
        fake_tensor(seed, &[1, len, LATENT])
    }

    fn quick(steps: usize) -> SamplerParams {
        SamplerParams { num_steps: steps, ..SamplerParams::default() }
    }

    #[test]
    fn the_default_step_count_is_24() {
        // 40 step の出力に最も近く、CER も同等だった最小の値（docs/bench.md）。MeanFlow は別に 4 step。
        assert_eq!(SamplerParams::default().num_steps, 24);
    }

    #[test]
    fn linear_schedule_runs_from_0_999_down_to_zero() {
        let t = t_schedule(4, TSchedule::Linear, -1.0).unwrap();
        let want = [0.999, 0.74925, 0.4995, 0.24975, 0.0];
        assert_eq!(t.len(), 5);
        for (a, b) in t.iter().zip(want) {
            assert!((a - b).abs() < 1e-6, "{t:?}");
        }
    }

    #[test]
    fn sway_schedule_is_monotonic_and_ends_at_zero() {
        let t = t_schedule(40, TSchedule::Sway, -1.0).unwrap();
        assert!((t[0] - 0.999).abs() < 1e-6 && t[40].abs() < 1e-6);
        assert!(t.windows(2).all(|w| w[0] > w[1]), "{t:?}");
    }

    #[test]
    fn invalid_schedules_are_rejected() {
        assert!(t_schedule(0, TSchedule::Linear, -1.0).is_err());
        assert!(t_schedule(4, TSchedule::Sway, f32::NAN).is_err());
        // 係数が極端だと単調減少でなくなる。
        assert!(t_schedule(40, TSchedule::Sway, 8.0).is_err());
    }

    #[test]
    fn rescale_with_k_one_is_identity() {
        let v = fake_tensor("v", &[1, 3, 2]);
        let x = fake_tensor("x", &[1, 3, 2]);
        assert_close(&temporal_score_rescale(&v, &x, 0.5, 1.0, 0.7).unwrap(), &v, 1e-5, 0.0).unwrap();
    }

    #[test]
    fn rescale_is_skipped_at_t_one() {
        let v = fake_tensor("v", &[1, 3, 2]);
        let x = fake_tensor("x", &[1, 3, 2]);
        assert_close(&temporal_score_rescale(&v, &x, 1.0, 2.0, 0.7).unwrap(), &v, 0.0, 0.0).unwrap();
    }

    #[test]
    fn euler_cfg_is_deterministic_and_keeps_shape() {
        let (d, c, n) = (dit(), conds("txt"), noise(6, "n"));
        let a = sample_euler_cfg(&d, &c, &quick(3), &n).unwrap();
        let b = sample_euler_cfg(&d, &c, &quick(3), &n).unwrap();
        assert_eq!(a.dims(), &[1, 6, LATENT]);
        assert_close(&a, &b, 0.0, 0.0).unwrap();
    }

    #[test]
    fn guidance_changes_the_result_unless_the_cfg_window_is_never_hit() {
        let (d, c, n) = (dit(), conds("txt"), noise(6, "n"));
        let with_cfg = sample_euler_cfg(&d, &c, &quick(4), &n).unwrap();
        let off = SamplerParams { cfg_text: 0.0, cfg_caption: 0.0, cfg_speaker: 0.0, ..quick(4) };
        let without = sample_euler_cfg(&d, &c, &off, &n).unwrap();
        assert!(assert_close(&with_cfg, &without, 1e-4, 0.0).is_err());
        // 最初のステップ t=0.999 より大きい cfg_min_t なら CFG は一度も使われず、無効化と一致する。
        let never = SamplerParams { cfg_min_t: 0.9999, ..quick(4) };
        assert_close(&sample_euler_cfg(&d, &c, &never, &n).unwrap(), &without, 1e-6, 0.0).unwrap();
    }

    #[test]
    fn joint_guidance_requires_equal_scales() {
        let (d, c, n) = (dit(), conds("txt"), noise(4, "n"));
        let p = SamplerParams { guidance: GuidanceMode::Joint, ..quick(2) }; // 3 / 4 / 5 は不揃い
        assert!(sample_euler_cfg(&d, &c, &p, &n).is_err());
        let ok = SamplerParams { guidance: GuidanceMode::Joint, cfg_text: 3.0, cfg_caption: 3.0, cfg_speaker: 3.0, ..quick(2) };
        assert!(sample_euler_cfg(&d, &c, &ok, &n).is_ok());
    }

    #[test]
    fn alternating_guidance_runs() {
        let (d, c, n) = (dit(), conds("txt"), noise(4, "n"));
        let p = SamplerParams { guidance: GuidanceMode::Alternating, ..quick(3) };
        assert_eq!(sample_euler_cfg(&d, &c, &p, &n).unwrap().dims(), &[1, 4, LATENT]);
    }

    #[test]
    fn caption_guidance_is_skipped_when_the_caption_is_fully_masked() {
        let (d, n) = (dit(), noise(5, "n"));
        let mut c: EncodedConditions = conds("txt");
        c.caption = Some((fake_tensor("cap", &[1, 2, 5]), mask(&[0, 0])));
        let only_caption = SamplerParams { cfg_text: 0.0, cfg_speaker: 0.0, ..quick(4) };
        let off = SamplerParams { cfg_text: 0.0, cfg_speaker: 0.0, cfg_caption: 0.0, ..quick(4) };
        assert_close(&sample_euler_cfg(&d, &c, &only_caption, &n).unwrap(), &sample_euler_cfg(&d, &c, &off, &n).unwrap(), 1e-6, 0.0).unwrap();
    }

    #[test]
    fn truncation_factor_scales_the_initial_noise() {
        let n = noise(4, "n");
        let scaled = apply_truncation(&n, Some(0.5)).unwrap();
        assert_close(&scaled, &n.affine(0.5, 0.0).unwrap(), 1e-7, 0.0).unwrap();
        assert_close(&apply_truncation(&n, None).unwrap(), &n, 0.0, 0.0).unwrap();
    }

    #[test]
    fn meanflow_sampler_uses_delta_t_and_is_deterministic() {
        let (d, c, n) = (dit_meanflow(), conds("txt"), noise(5, "n"));
        let a = sample_meanflow(&d, &c, 4, &n).unwrap();
        let b = sample_meanflow(&d, &c, 4, &n).unwrap();
        assert_eq!(a.dims(), &[1, 5, LATENT]);
        assert_close(&a, &b, 0.0, 0.0).unwrap();
        assert!(sample_meanflow(&dit(), &c, 4, &n).is_err(), "RF model must be rejected");
        assert!(sample_meanflow(&d, &c, 0, &n).is_err());
    }

    #[test]
    fn initial_noise_has_requested_shape_and_is_seeded() {
        let a = initial_noise(5, 1, 7, 3, &Device::Cpu).unwrap();
        assert_eq!(a.dims(), &[1, 7, 3]);
        assert_close(&a, &initial_noise(5, 1, 7, 3, &Device::Cpu).unwrap(), 0.0, 0.0).unwrap();
    }

    #[test]
    fn all_masked_speaker_does_not_enable_speaker_guidance() {
        let mut c = conds("txt");
        c.speaker = Some((fake_tensor("spk", &[1, 3, 6]), mask(&[0, 0, 0])));
        let kinds = enabled_guidance(&c, &quick(2)).unwrap();
        assert!(kinds.iter().all(|(k, _)| *k != Kind::Speaker), "no-reference run must not pay for a speaker branch");
        assert!(enabled_guidance(&conds("txt"), &quick(2)).unwrap().iter().any(|(k, _)| *k == Kind::Speaker));
    }
}
