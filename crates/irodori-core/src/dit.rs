//! 拡散 Transformer（RF velocity 予測）。
//!
//! 上流 `DiffusionBlock`（model.py 927）と `TextToLatentRFDiT.forward_with_encoded_conditions`（1877）。

use candle_core::Tensor;
use candle_nn::{linear, linear_no_bias, Linear, Module, VarBuilder};

use crate::conditioner::EncodedConditions;
use crate::config::ModelConfig;
use crate::joint_attention::{concat_kv_mask, JointAttention};
use crate::nn::adaln::LowRankAdaLn;
use crate::nn::norm::RmsNorm;
use crate::nn::rope::FreqsCis;
use crate::nn::swiglu::SwiGlu;
use crate::nn::timestep::timestep_embedding;
use crate::{IrodoriError, Result};

/// ブロックごとの文脈 K/V（`JointAttention::project_context_kv` の出力）。
pub type ContextKv = Vec<Vec<Tensor>>;

struct DiffusionBlock {
    attention: JointAttention,
    mlp: SwiGlu,
    attention_adaln: LowRankAdaLn,
    mlp_adaln: LowRankAdaLn,
}

impl DiffusionBlock {
    fn load(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let speaker = cfg.use_speaker_condition_resolved().then_some(cfg.speaker_dim);
        let caption = cfg.use_caption_condition.then(|| cfg.caption_dim_resolved());
        Ok(Self {
            attention: JointAttention::load(cfg.model_dim, cfg.num_heads, cfg.text_dim, speaker, caption, cfg.norm_eps, vb.pp("attention"))?,
            mlp: SwiGlu::new(cfg.model_dim, (cfg.model_dim as f64 * cfg.mlp_ratio) as usize, vb.pp("mlp"))?,
            attention_adaln: LowRankAdaLn::new(cfg.model_dim, cfg.adaln_rank, cfg.norm_eps, vb.pp("attention_adaln"))?,
            mlp_adaln: LowRankAdaLn::new(cfg.model_dim, cfg.adaln_rank, cfg.norm_eps, vb.pp("mlp_adaln"))?,
        })
    }

    fn forward(&self, x: &Tensor, cond: &Tensor, kv: &[Tensor], kv_mask: &Tensor, freqs: &FreqsCis) -> Result<Tensor> {
        let (h, gate) = self.attention_adaln.forward(x, cond)?;
        let x = (x + gate.broadcast_mul(&self.attention.forward(&h, kv, kv_mask, freqs)?)?)?;
        let (h, gate) = self.mlp_adaln.forward(&x, cond)?;
        Ok((&x + gate.broadcast_mul(&self.mlp.forward(&h)?)?)?)
    }
}

pub struct Dit {
    cond: [Linear; 3],
    /// MeanFlow の `delta_t`（区間幅）用の条件 MLP。RF モデルでは `None`。
    delta_cond: Option<[Linear; 3]>,
    in_proj: Linear,
    blocks: Vec<DiffusionBlock>,
    out_norm: RmsNorm,
    out_proj: Linear,
    head_dim: usize,
    timestep_dim: usize,
    /// 重みの dtype。入力・条件は演算前にこれへ揃え、出力は呼び出し側の dtype に戻す。
    dtype: candle_core::DType,
    has_speaker: bool,
    has_caption: bool,
}

impl Dit {
    pub fn load(vb: VarBuilder, cfg: &ModelConfig) -> Result<Self> {
        let d = cfg.model_dim;
        let c = vb.pp("cond_module");
        let blocks = (0..cfg.num_layers)
            .map(|i| DiffusionBlock::load(cfg, vb.pp(format!("blocks.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let mlp3 = |vb: &VarBuilder| -> Result<[Linear; 3]> {
            Ok([
                linear_no_bias(cfg.timestep_embed_dim, d, vb.pp("0"))?,
                linear_no_bias(d, d, vb.pp("2"))?,
                linear_no_bias(d, d * 3, vb.pp("4"))?,
            ])
        };
        let flow = cfg.flow_parameterization.trim().to_ascii_lowercase();
        let delta_cond = match flow.as_str() {
            "rf_velocity" => None,
            "meanflow" => Some(mlp3(&vb.pp("delta_cond_module"))?),
            other => {
                return Err(IrodoriError::InvalidConfig(format!(
                    "flow_parameterization must be 'rf_velocity' or 'meanflow', got {other:?}"
                )))
            }
        };
        Ok(Self {
            cond: mlp3(&c)?,
            delta_cond,
            in_proj: linear(cfg.patched_latent_dim(), d, vb.pp("in_proj"))?,
            blocks,
            out_norm: RmsNorm::new(&[d], cfg.norm_eps, vb.pp("out_norm"))?,
            out_proj: linear(d, cfg.patched_latent_dim(), vb.pp("out_proj"))?,
            head_dim: d / cfg.num_heads,
            timestep_dim: cfg.timestep_embed_dim,
            dtype: vb.dtype(),
            has_speaker: cfg.use_speaker_condition_resolved(),
            has_caption: cfg.use_caption_condition,
        })
    }

    /// サンプリング中は条件が不変なので、全ブロックの文脈 K/V を一度だけ射影して使い回せる。
    pub fn build_context_kv_cache(&self, c: &EncodedConditions) -> Result<ContextKv> {
        let c = &c.to_dtype(self.dtype)?;
        self.blocks
            .iter()
            .map(|b| {
                b.attention.project_context_kv(&c.text, c.speaker.as_ref().map(|s| &s.0), c.caption.as_ref().map(|s| &s.0))
            })
            .collect()
    }

    fn cond_mlp(&self, mlp: &[Linear; 3], t: &Tensor, dtype: candle_core::DType) -> Result<Tensor> {
        let e = timestep_embedding(t, self.timestep_dim)?.to_dtype(dtype)?;
        let h = candle_nn::ops::silu(&mlp[0].forward(&e)?)?;
        let h = candle_nn::ops::silu(&mlp[1].forward(&h)?)?;
        Ok(mlp[2].forward(&h)?)
    }

    /// `(B, 1, 3*model_dim)`。MeanFlow では区間幅 `delta_t` の条件を足す（RF では `delta_t` を渡せない）。
    fn time_cond(&self, t: &Tensor, delta_t: Option<&Tensor>, dtype: candle_core::DType) -> Result<Tensor> {
        let mut c = self.cond_mlp(&self.cond, t, dtype)?;
        match (&self.delta_cond, delta_t) {
            (Some(m), Some(d)) => c = (c + self.cond_mlp(m, d, dtype)?)?,
            (Some(_), None) => return Err(IrodoriError::InvalidInput("delta_t is required for a MeanFlow forward pass".into())),
            (None, Some(_)) => return Err(IrodoriError::InvalidInput("delta_t was provided to an RF velocity model".into())),
            (None, None) => {}
        }
        Ok(c.unsqueeze(1)?)
    }

    /// `x_t: (B, T, patched_latent_dim)`, `t: (B,)` → 速度 `(B, T, patched_latent_dim)`（RF モデル）。
    pub fn forward_with_encoded_conditions(
        &self,
        x_t: &Tensor,
        t: &Tensor,
        c: &EncodedConditions,
        latent_mask: Option<&Tensor>,
        kv_cache: Option<&ContextKv>,
    ) -> Result<Tensor> {
        self.forward_inner(x_t, t, None, c, latent_mask, kv_cache)
    }

    /// MeanFlow モデルの順伝播。`delta_t: (B,)` は区間幅 `t - t_next`。
    pub fn forward_meanflow(
        &self,
        x_t: &Tensor,
        t: &Tensor,
        delta_t: &Tensor,
        c: &EncodedConditions,
        latent_mask: Option<&Tensor>,
        kv_cache: Option<&ContextKv>,
    ) -> Result<Tensor> {
        self.forward_inner(x_t, t, Some(delta_t), c, latent_mask, kv_cache)
    }

    fn forward_inner(
        &self,
        x_t: &Tensor,
        t: &Tensor,
        delta_t: Option<&Tensor>,
        c: &EncodedConditions,
        latent_mask: Option<&Tensor>,
        kv_cache: Option<&ContextKv>,
    ) -> Result<Tensor> {
        let (b, s, _) = x_t.dims3()?;
        let (out_dtype, c) = (x_t.dtype(), &c.to_dtype(self.dtype)?);
        let (x_t, t, delta_t) = (&x_t.to_dtype(self.dtype)?, t, delta_t);
        if let Some(cache) = kv_cache {
            if cache.len() != self.blocks.len() {
                return Err(IrodoriError::InvalidInput(format!("kv cache has {} layers, model has {}", cache.len(), self.blocks.len())));
            }
        }
        let cond = self.time_cond(t, delta_t, x_t.dtype())?;
        let mut x = self.in_proj.forward(x_t)?;
        let freqs = FreqsCis::new(self.head_dim, s, 10000.0, x.device())?;

        let missing = |what: &str| IrodoriError::InvalidInput(format!("{what} state is required by this model"));
        let mut ctx_masks: Vec<&Tensor> = vec![&c.text_mask];
        if self.has_speaker {
            ctx_masks.push(&c.speaker.as_ref().ok_or_else(|| missing("speaker"))?.1);
        }
        if self.has_caption {
            ctx_masks.push(&c.caption.as_ref().ok_or_else(|| missing("caption"))?.1);
        }
        let kv_mask = concat_kv_mask(b, s, latent_mask, &ctx_masks, x.device())?;

        for (i, block) in self.blocks.iter().enumerate() {
            let owned;
            let kv: &[Tensor] = match kv_cache {
                Some(cache) => &cache[i],
                None => {
                    owned = block.attention.project_context_kv(&c.text, c.speaker.as_ref().map(|s| &s.0), c.caption.as_ref().map(|s| &s.0))?;
                    &owned
                }
            };
            x = block.forward(&x, &cond, kv, &kv_mask, &freqs)?;
        }
        Ok(self.out_proj.forward(&self.out_norm.forward(&x)?)?.to_dtype(out_dtype)?)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::conditioner::EncodedConditions;
    use crate::config::ModelConfig;
    use crate::joint_attention::test_spec::spec;
    use crate::nn::test_util::fake_tensor;
    use candle_core::{Device, Tensor};

    pub(crate) const DIM: usize = 8;
    const HEADS: usize = 2;
    const LAYERS: usize = 2;
    pub(crate) const LATENT: usize = 3;

    pub(crate) fn cfg() -> ModelConfig {
        ModelConfig {
            latent_dim: LATENT,
            model_dim: DIM,
            num_layers: LAYERS,
            num_heads: HEADS,
            mlp_ratio: 2.0,
            text_dim: 4,
            use_caption_condition: true,
            caption_dim: Some(5),
            use_speaker_condition: Some(true),
            speaker_dim: 6,
            timestep_embed_dim: 4,
            adaln_rank: 2,
            ..ModelConfig::default()
        }
    }

    fn adaln(prefix: &str, rank: usize) -> Vec<(String, Vec<usize>)> {
        let mut s = Vec::new();
        for n in ["shift", "scale", "gate"] {
            s.push((format!("{prefix}.{n}_down.weight"), vec![rank, DIM]));
            s.push((format!("{prefix}.{n}_up.weight"), vec![DIM, rank]));
            s.push((format!("{prefix}.{n}_up.bias"), vec![DIM]));
        }
        s
    }

    pub(crate) fn dit() -> Dit {
        dit_with(false, candle_core::DType::F32)
    }

    pub(crate) fn dit_dtype(dtype: candle_core::DType) -> Dit {
        dit_with(false, dtype)
    }

    pub(crate) fn dit_meanflow() -> Dit {
        dit_with(true, candle_core::DType::F32)
    }

    fn dit_with(meanflow: bool, dtype: candle_core::DType) -> Dit {
        let mut sp: Vec<(String, Vec<usize>)> = vec![
            ("cond_module.0.weight".into(), vec![DIM, 4]),
            ("cond_module.2.weight".into(), vec![DIM, DIM]),
            ("cond_module.4.weight".into(), vec![3 * DIM, DIM]),
            ("in_proj.weight".into(), vec![DIM, LATENT]),
            ("in_proj.bias".into(), vec![DIM]),
            ("out_norm.weight".into(), vec![DIM]),
            ("out_proj.weight".into(), vec![LATENT, DIM]),
            ("out_proj.bias".into(), vec![LATENT]),
        ];
        for i in 0..LAYERS {
            let p = format!("blocks.{i}.");
            sp.extend(spec(&format!("{p}attention."), DIM, HEADS, 4, Some(6), Some(5)));
            sp.push((format!("{p}mlp.w1.weight"), vec![16, DIM]));
            sp.push((format!("{p}mlp.w3.weight"), vec![16, DIM]));
            sp.push((format!("{p}mlp.w2.weight"), vec![DIM, 16]));
            sp.extend(adaln(&format!("{p}attention_adaln"), 2));
            sp.extend(adaln(&format!("{p}mlp_adaln"), 2));
        }
        let mut c = cfg();
        if meanflow {
            c.flow_parameterization = "meanflow".into();
            sp.push(("delta_cond_module.0.weight".into(), vec![DIM, 4]));
            sp.push(("delta_cond_module.2.weight".into(), vec![DIM, DIM]));
            sp.push(("delta_cond_module.4.weight".into(), vec![3 * DIM, DIM]));
        }
        Dit::load(crate::nn::test_util::vb_from_spec_dtype(sp, dtype), &c).unwrap()
    }

    pub(crate) fn mask(v: &[u8]) -> Tensor {
        Tensor::from_slice(v, (1, v.len()), &Device::Cpu).unwrap()
    }

    pub(crate) fn conds(text_seed: &str) -> EncodedConditions {
        EncodedConditions {
            text: fake_tensor(text_seed, &[1, 3, 4]),
            text_mask: mask(&[1, 1, 1]),
            speaker: Some((fake_tensor("spk", &[1, 3, 6]), mask(&[1, 1, 1]))),
            caption: Some((fake_tensor("cap", &[1, 2, 5]), mask(&[1, 1]))),
        }
    }

    pub(crate) fn x_t() -> Tensor {
        fake_tensor("xt", &[1, 5, LATENT])
    }

    pub(crate) fn t() -> Tensor {
        Tensor::from_slice(&[0.5f32], 1, &Device::Cpu).unwrap()
    }

    #[test]
    fn output_matches_latent_shape() {
        let y = dit().forward_with_encoded_conditions(&x_t(), &t(), &conds("txt"), None, None).unwrap();
        assert_eq!(y.dims(), &[1, 5, LATENT]);
        assert!(y.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn kv_cache_equals_uncached() {
        let d = dit();
        let c = conds("txt");
        let cache = d.build_context_kv_cache(&c).unwrap();
        let a = d.forward_with_encoded_conditions(&x_t(), &t(), &c, None, None).unwrap();
        let b = d.forward_with_encoded_conditions(&x_t(), &t(), &c, None, Some(&cache)).unwrap();
        crate::golden::assert_close(&a, &b, 1e-6, 0.0).unwrap();
    }

    #[test]
    fn masked_text_tokens_do_not_change_output() {
        let d = dit();
        let mut c1 = conds("txt");
        let mut c2 = conds("txt");
        c1.text_mask = mask(&[1, 1, 0]);
        c2.text_mask = mask(&[1, 1, 0]);
        c2.text = Tensor::cat(&[&c2.text.narrow(1, 0, 2).unwrap(), &Tensor::full(9.0f32, (1, 1, 4), &Device::Cpu).unwrap()], 1).unwrap();
        let a = d.forward_with_encoded_conditions(&x_t(), &t(), &c1, None, None).unwrap();
        let b = d.forward_with_encoded_conditions(&x_t(), &t(), &c2, None, None).unwrap();
        crate::golden::assert_close(&a, &b, 1e-5, 0.0).unwrap();
    }

    #[test]
    fn speaker_state_influences_output() {
        let d = dit();
        let c1 = conds("txt");
        let mut c2 = conds("txt");
        c2.speaker = Some((fake_tensor("other", &[1, 3, 6]), mask(&[1, 1, 1])));
        let a = d.forward_with_encoded_conditions(&x_t(), &t(), &c1, None, None).unwrap();
        let b = d.forward_with_encoded_conditions(&x_t(), &t(), &c2, None, None).unwrap();
        assert!(crate::golden::assert_close(&a, &b, 1e-4, 0.0).is_err());
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
        let dit = Dit::load(vb, &cfg).unwrap();
        let g = crate::golden::load("dit_step").unwrap();
        let u32t = |k: &str| g[k].to_dtype(DType::U32).unwrap();
        let u8t = |k: &str| g[k].to_dtype(DType::U8).unwrap();
        let (tid, tm, cid, cm, rm) = (u32t("text_ids"), u8t("text_mask"), u32t("caption_ids"), u8t("caption_mask"), u8t("ref_mask"));
        let enc = cond
            .encode(&ConditionInputs { text_ids: &tid, text_mask: &tm, caption_ids: Some(&cid), caption_mask: Some(&cm), ref_latent: Some(&g["ref_latent"]), ref_mask: Some(&rm) })
            .unwrap();
        let close = |a: &Tensor, k: &str| crate::golden::assert_close(a, &g[k], 2e-3, 2e-3).unwrap_or_else(|e| panic!("{k}: {e}"));
        close(&enc.text, "text_state");
        let (ss, sm) = enc.speaker.clone().unwrap();
        close(&ss, "speaker_state");
        assert_eq!(sm.to_vec2::<u8>().unwrap(), g["speaker_mask"].to_vec2::<u8>().unwrap());
        close(&enc.caption.clone().unwrap().0, "caption_state");

        let v = dit.forward_with_encoded_conditions(&g["x_t"], &g["t"], &enc, None, None).unwrap();
        close(&v, "velocity");
        let cache = dit.build_context_kv_cache(&enc).unwrap();
        let vc = dit.forward_with_encoded_conditions(&g["x_t"], &g["t"], &enc, None, Some(&cache)).unwrap();
        close(&vc, "velocity");
    }

    #[test]
    fn bf16_weights_accept_f32_inputs_and_stay_close_to_f32() {
        let c = conds("txt");
        let a = dit().forward_with_encoded_conditions(&x_t(), &t(), &c, None, None).unwrap();
        let d16 = dit_dtype(candle_core::DType::BF16);
        let b = d16.forward_with_encoded_conditions(&x_t(), &t(), &c, None, None).unwrap();
        assert_eq!(b.dtype(), candle_core::DType::F32, "output keeps the caller's dtype");
        crate::golden::assert_close(&a, &b, 8e-2, 8e-2).unwrap();
        let cache = d16.build_context_kv_cache(&c).unwrap();
        let bc = d16.forward_with_encoded_conditions(&x_t(), &t(), &c, None, Some(&cache)).unwrap();
        crate::golden::assert_close(&b, &bc, 1e-6, 0.0).unwrap();
    }

    fn delta(v: f32) -> Tensor {
        Tensor::from_slice(&[v], 1, &Device::Cpu).unwrap()
    }

    #[test]
    fn meanflow_model_requires_delta_t() {
        let d = dit_meanflow();
        assert!(d.forward_with_encoded_conditions(&x_t(), &t(), &conds("txt"), None, None).is_err());
    }

    #[test]
    fn rf_model_rejects_delta_t() {
        let d = dit();
        assert!(d.forward_meanflow(&x_t(), &t(), &delta(0.25), &conds("txt"), None, None).is_err());
    }

    #[test]
    fn meanflow_delta_changes_the_output() {
        let d = dit_meanflow();
        let c = conds("txt");
        let a = d.forward_meanflow(&x_t(), &t(), &delta(0.25), &c, None, None).unwrap();
        let b = d.forward_meanflow(&x_t(), &t(), &delta(0.75), &c, None, None).unwrap();
        assert!(crate::golden::assert_close(&a, &b, 1e-4, 0.0).is_err());
    }

    #[test]
    fn missing_speaker_state_is_an_error_when_model_expects_it() {
        let mut c = conds("txt");
        c.speaker = None;
        assert!(dit().forward_with_encoded_conditions(&x_t(), &t(), &c, None, None).is_err());
    }

}
