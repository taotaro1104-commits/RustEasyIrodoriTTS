//! 発話長（フレーム数）予測。
//!
//! 上流 `DurationPredictor`（model.py 990–1452）のうち、v4.1 が使う
//! `token_sum_adarn_zero_no_aux` / `token_sum_dual_adarn_zero_no_aux` のみを実装する。
//! 出力は `log1p(総フレーム数)`、総フレーム数 = Σ 有効トークンの softplus(logit)。

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{linear, Linear, Module, VarBuilder};

use crate::conditioner::EncodedConditions;
use crate::config::ModelConfig;
use crate::nn::norm::RmsNorm;
use crate::nn::swiglu::SwiGlu;
use crate::{IrodoriError, Result};

struct DurationBlock {
    norm: RmsNorm,
    mlp: SwiGlu,
    modulation: Linear,
    caption_modulation: Option<Linear>,
}

impl DurationBlock {
    fn load(dim: usize, speaker_dim: usize, caption_dim: Option<usize>, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm: RmsNorm::new(&[dim], eps, vb.pp("norm"))?,
            mlp: SwiGlu::new(dim, dim, vb.pp("mlp"))?,
            modulation: linear(speaker_dim, dim * 3, vb.pp("modulation"))?,
            caption_modulation: caption_dim.map(|d| linear(d, dim * 3, vb.pp("caption_modulation"))).transpose()?,
        })
    }

    fn forward(&self, x: &Tensor, speaker: &Tensor, caption: Option<&Tensor>) -> Result<Tensor> {
        let h = self.norm.forward(x)?;
        let mut m = self.modulation.forward(&candle_nn::ops::silu(speaker)?)?.chunk(3, D::Minus1)?;
        if let Some(cm) = &self.caption_modulation {
            let c = caption.ok_or_else(|| IrodoriError::InvalidInput("caption_cond is required for caption AdaRN-Zero duration blocks".into()))?;
            let cc = cm.forward(&candle_nn::ops::silu(c)?)?.chunk(3, D::Minus1)?;
            for (a, b) in m.iter_mut().zip(cc) {
                *a = (&*a + b)?;
            }
        }
        let (shift, scale, gate) = (m[0].unsqueeze(1)?, m[1].unsqueeze(1)?, m[2].unsqueeze(1)?);
        let h = h.broadcast_mul(&scale.affine(1.0, 1.0)?)?.broadcast_add(&shift)?;
        Ok((x + gate.tanh()?.broadcast_mul(&self.mlp.forward(&h)?)?)?)
    }
}

pub struct DurationPredictor {
    null_speaker: Tensor,
    null_caption: Option<Tensor>,
    input_proj: Linear,
    blocks: Vec<DurationBlock>,
    out_norm: RmsNorm,
    out_proj: Linear,
}

/// `(B,)` の bool を `(B, 1)` の F32 選択マスクにする。
fn select_mask(flags: &[bool], device: &Device) -> Result<Tensor> {
    let v: Vec<f32> = flags.iter().map(|&b| f32::from(u8::from(b))).collect();
    Ok(Tensor::from_vec(v, (flags.len(), 1), device)?)
}

/// `vec * sel + null * (1 - sel)`。
fn choose(vec: &Tensor, null: &Tensor, sel: &Tensor) -> Result<Tensor> {
    Ok(vec.broadcast_mul(sel)?.broadcast_add(&null.unsqueeze(0)?.broadcast_mul(&sel.affine(-1.0, 1.0)?)?)?)
}

impl DurationPredictor {
    /// `vb` は `duration_predictor` 接頭辞まで進めた VarBuilder。
    pub fn load(vb: VarBuilder, cfg: &ModelConfig) -> Result<Self> {
        let arch = cfg.duration_architecture.trim().to_ascii_lowercase();
        let dual = match arch.as_str() {
            "token_sum_adarn_zero_no_aux" => false,
            "token_sum_dual_adarn_zero_no_aux" => true,
            other => {
                return Err(IrodoriError::InvalidConfig(format!("unsupported duration architecture: {other:?}")))
            }
        };
        if cfg.duration_speaker_fusion.trim().to_ascii_lowercase() != "adarn_zero" {
            return Err(IrodoriError::InvalidConfig("token-sum duration requires speaker_fusion='adarn_zero'".into()));
        }
        if !cfg.use_speaker_condition_resolved() || (dual && !cfg.use_caption_condition) {
            return Err(IrodoriError::InvalidConfig(format!("duration architecture {arch:?} needs speaker (and caption if dual) conditioning")));
        }
        let caption_dim = dual.then(|| cfg.caption_dim_resolved());
        let hidden = cfg.duration_hidden_dim;
        let blocks = (0..cfg.duration_layers)
            .map(|i| DurationBlock::load(hidden, cfg.speaker_dim, caption_dim, cfg.norm_eps, vb.pp(format!("token_blocks.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            null_speaker: vb.get(cfg.speaker_dim, "null_speaker")?,
            null_caption: caption_dim.map(|d| vb.get(d, "null_caption")).transpose()?,
            input_proj: linear(cfg.text_dim, hidden, vb.pp("token_input_proj"))?,
            blocks,
            out_norm: RmsNorm::new(&[hidden], cfg.norm_eps, vb.pp("token_out_norm"))?,
            out_proj: linear(hidden, 1, vb.pp("token_out_proj"))?,
        })
    }

    /// 条件から `log1p(総フレーム数)` を `(B,)` で返す。`has_*` は行ごとにその条件を使うか。
    pub fn predict_log_frames(&self, c: &EncodedConditions, has_speaker: &[bool], has_caption: &[bool]) -> Result<Tensor> {
        let (b, s, _) = c.text.dims3()?;
        if has_speaker.len() != b || has_caption.len() != b {
            return Err(IrodoriError::InvalidInput(format!("has_speaker/has_caption must have length {b}")));
        }
        let dev = c.text.device();
        // 全マスク行は x をゼロにして位置 0 を有効にする（上流 `_safe_attention_mask`）。
        let mask_f = c.text_mask.to_dtype(DType::F32)?.gt(0.0f32)?.to_dtype(DType::F32)?;
        let empty = mask_f.sum_keepdim(1)?.eq(0.0f32)?.to_dtype(DType::F32)?;
        let mut first = vec![0f32; s];
        first[0] = 1.0;
        let safe_mask = (&mask_f + empty.broadcast_mul(&Tensor::from_vec(first, (1, s), dev)?)?)?;
        let x = c.text.broadcast_mul(&empty.affine(-1.0, 1.0)?.unsqueeze(2)?)?;

        let sel_spk = select_mask(has_speaker, dev)?;
        let speaker = match &c.speaker {
            Some((state, _)) => choose(&state.narrow(1, 0, 1)?.squeeze(1)?, &self.null_speaker, &sel_spk)?,
            None => self.null_speaker.unsqueeze(0)?.broadcast_as((b, self.null_speaker.dim(0)?))?.contiguous()?,
        };
        let caption = match (&self.null_caption, &c.caption) {
            (None, _) => None,
            (Some(null), None) => Some(null.unsqueeze(0)?.broadcast_as((b, null.dim(0)?))?.contiguous()?),
            (Some(null), Some((state, mask))) => {
                let m = mask.to_dtype(DType::F32)?.gt(0.0f32)?.to_dtype(DType::F32)?.broadcast_mul(&select_mask(has_caption, dev)?)?;
                let denom = m.sum_keepdim(1)?.clamp(1.0, f64::MAX)?;
                let mean = state.broadcast_mul(&m.unsqueeze(2)?)?.sum(1)?.broadcast_div(&denom)?;
                let any = m.sum_keepdim(1)?.gt(0.0f32)?.to_dtype(DType::F32)?;
                Some(choose(&mean, null, &any)?)
            }
        };

        let mut h = self.input_proj.forward(&x)?;
        for block in &self.blocks {
            h = block.forward(&h, &speaker, caption.as_ref())?;
        }
        let logits = self.out_proj.forward(&self.out_norm.forward(&h)?)?.squeeze(2)?.to_dtype(DType::F32)?;
        // softplus(x) = relu(x) + ln(1 + exp(-|x|))（オーバーフローしない形）
        let frames = (logits.relu()? + (logits.abs()?.neg()?.exp()? + 1.0)?.log()?)?;
        let total = frames.broadcast_mul(&safe_mask)?.sum(1)?;
        Ok((total.clamp(0.0, f64::MAX)? + 1.0)?.log()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conditioner::EncodedConditions;
    use crate::config::ModelConfig;
    use crate::nn::test_util::{fake_tensor, vb_from_spec, zeros};
    use candle_core::{Device, Tensor};

    const TEXT: usize = 4;
    const HID: usize = 6;
    const SPK: usize = 5;
    const CAP: usize = 3;
    const LAYERS: usize = 2;

    fn cfg(arch: &str) -> ModelConfig {
        ModelConfig {
            text_dim: TEXT,
            speaker_dim: SPK,
            caption_dim: Some(CAP),
            use_caption_condition: true,
            use_speaker_condition: Some(true),
            use_duration_predictor: true,
            duration_hidden_dim: HID,
            duration_layers: LAYERS,
            duration_architecture: arch.into(),
            ..ModelConfig::default()
        }
    }

    /// token_out_proj の重みを `out_weight` で埋め、bias を `out_bias` にする。
    /// `out_weight=0` なら出力フレーム数は bias だけで決まる（解析的に検算できる）。
    fn predictor(arch: &str, out_bias: f32, out_weight: f32) -> DurationPredictor {
        let dual = arch.contains("dual");
        let mut sp: Vec<(String, Vec<usize>)> = vec![
            ("null_speaker".into(), vec![SPK]),
            ("token_input_proj.weight".into(), vec![HID, TEXT]),
            ("token_input_proj.bias".into(), vec![HID]),
            ("token_out_norm.weight".into(), vec![HID]),
        ];
        if dual {
            sp.push(("null_caption".into(), vec![CAP]));
        }
        for i in 0..LAYERS {
            let p = format!("token_blocks.{i}.");
            sp.push((format!("{p}norm.weight"), vec![HID]));
            for w in ["w1", "w3"] {
                sp.push((format!("{p}mlp.{w}.weight"), vec![HID, HID]));
            }
            sp.push((format!("{p}mlp.w2.weight"), vec![HID, HID]));
            sp.push((format!("{p}modulation.weight"), vec![3 * HID, SPK]));
            sp.push((format!("{p}modulation.bias"), vec![3 * HID]));
            if dual {
                sp.push((format!("{p}caption_modulation.weight"), vec![3 * HID, CAP]));
                sp.push((format!("{p}caption_modulation.bias"), vec![3 * HID]));
            }
        }
        // 上の fake 重みに out_proj だけ解析的な値を足す。
        let mut map = std::collections::HashMap::new();
        for (n, s) in sp {
            let t = fake_tensor(&n, &s);
            map.insert(n, t);
        }
        map.insert("token_out_proj.weight".into(), zeros(1, HID).affine(1.0, out_weight as f64).unwrap());
        map.insert("token_out_proj.bias".into(), Tensor::from_slice(&[out_bias], 1, &Device::Cpu).unwrap());
        let vb = candle_nn::VarBuilder::from_tensors(map, candle_core::DType::F32, &Device::Cpu);
        DurationPredictor::load(vb, &cfg(arch)).unwrap()
    }

    fn mask(v: &[u8]) -> Tensor {
        Tensor::from_slice(v, (1, v.len()), &Device::Cpu).unwrap()
    }

    fn conds(text_mask: &[u8], spk_seed: &str, cap_mask: &[u8]) -> EncodedConditions {
        EncodedConditions {
            text: fake_tensor("txt", &[1, text_mask.len(), TEXT]),
            text_mask: mask(text_mask),
            speaker: Some((fake_tensor(spk_seed, &[1, 3, SPK]), mask(&[1, 1, 1]))),
            caption: Some((fake_tensor("cap", &[1, cap_mask.len(), CAP]), mask(cap_mask))),
        }
    }

    const DUAL: &str = "token_sum_dual_adarn_zero_no_aux";

    fn value(t: &Tensor) -> f32 {
        t.to_vec1::<f32>().unwrap()[0]
    }

    #[test]
    fn total_is_log1p_of_summed_token_frames() {
        // bias = ln(expm1(9)) → softplus = 9 frames/token。有効 4 トークン → 36 フレーム。
        let p = predictor(DUAL, (9f32.exp() - 1.0).ln(), 0.0);
        let out = p.predict_log_frames(&conds(&[1, 1, 1, 1, 0], "s", &[1, 1]), &[true], &[true]).unwrap();
        assert!((value(&out) - 37f32.ln()).abs() < 1e-4, "{}", value(&out));
    }

    #[test]
    fn masked_tokens_do_not_count() {
        let p = predictor(DUAL, (9f32.exp() - 1.0).ln(), 0.0);
        let out = p.predict_log_frames(&conds(&[1, 0, 0], "s", &[1, 1]), &[true], &[true]).unwrap();
        assert!((value(&out) - 10f32.ln()).abs() < 1e-4);
    }

    #[test]
    fn speaker_state_is_ignored_when_has_speaker_is_false() {
        let p = predictor(DUAL, 0.3, 0.0);
        let a = p.predict_log_frames(&conds(&[1, 1, 1], "s1", &[1, 1]), &[false], &[true]).unwrap();
        let b = p.predict_log_frames(&conds(&[1, 1, 1], "s2", &[1, 1]), &[false], &[true]).unwrap();
        assert_eq!(value(&a), value(&b));
    }

    #[test]
    fn speaker_state_matters_when_has_speaker_is_true() {
        // out_proj 重みが 0 だと条件が効かないので、重みを持つ予測器で確かめる。
        let p = predictor(DUAL, 0.3, 0.7);
        let a = p.predict_log_frames(&conds(&[1, 1, 1], "s1", &[1, 1]), &[true], &[true]).unwrap();
        let b = p.predict_log_frames(&conds(&[1, 1, 1], "s2", &[1, 1]), &[true], &[true]).unwrap();
        assert!((value(&a) - value(&b)).abs() > 1e-6);
    }

    #[test]
    fn fully_masked_caption_falls_back_to_null_caption() {
        let p = predictor(DUAL, 0.3, 0.7);
        let a = p.predict_log_frames(&conds(&[1, 1, 1], "s", &[0, 0]), &[true], &[true]).unwrap();
        let mut c = conds(&[1, 1, 1], "s", &[0, 0]);
        c.caption = Some((fake_tensor("different", &[1, 2, CAP]), mask(&[0, 0])));
        let b = p.predict_log_frames(&c, &[true], &[true]).unwrap();
        assert_eq!(value(&a), value(&b));
    }

    #[test]
    fn fully_masked_text_stays_finite() {
        let p = predictor(DUAL, 0.3, 0.0);
        let out = p.predict_log_frames(&conds(&[0, 0, 0], "s", &[1, 1]), &[true], &[true]).unwrap();
        assert!(value(&out).is_finite());
    }

    #[test]
    fn unsupported_architecture_is_rejected() {
        let vb = vb_from_spec(vec![]);
        assert!(DurationPredictor::load(vb, &cfg("pooled")).is_err());
    }

    #[test]
    fn real_model_matches_python_golden() {
        use crate::conditioner::{ConditionInputs, Conditioner};
        use candle_core::DType;
        let (Some(model_dir), Some(_)) = (std::env::var_os("IRODORI_MODEL_DIR"), crate::golden::dir()) else {
            eprintln!("skip: IRODORI_MODEL_DIR / IRODORI_GOLDEN_DIR not set");
            return;
        };
        let w = crate::loader::Weights::open(&std::path::Path::new(&model_dir).join("model.safetensors"), &Device::Cpu).unwrap();
        let (cfg, te) = w.config().unwrap();
        let vb = w.var_builder(DType::F32).unwrap();
        let cond = Conditioner::load(vb.clone(), &cfg, &te).unwrap();
        let dur = DurationPredictor::load(vb.pp("duration_predictor"), &cfg).unwrap();
        let g = crate::golden::load("dit_step").unwrap();
        let u32t = |k: &str| g[k].to_dtype(DType::U32).unwrap();
        let u8t = |k: &str| g[k].to_dtype(DType::U8).unwrap();
        let (tid, tm, cid, cm, rm) = (u32t("text_ids"), u8t("text_mask"), u32t("caption_ids"), u8t("caption_mask"), u8t("ref_mask"));
        let enc = cond
            .encode(&ConditionInputs { text_ids: &tid, text_mask: &tm, caption_ids: Some(&cid), caption_mask: Some(&cm), ref_latent: Some(&g["ref_latent"]), ref_mask: Some(&rm) })
            .unwrap();
        let out = dur.predict_log_frames(&enc, &[true], &[true]).unwrap();
        crate::golden::assert_close(&out, &g["duration"], 1e-3, 1e-3).unwrap();
    }
}
