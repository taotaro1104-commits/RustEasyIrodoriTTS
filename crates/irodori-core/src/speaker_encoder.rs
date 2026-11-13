//! 参照潜在エンコーダ（speaker 条件）と、それが使う `SelfAttention` / `TextBlock`。
//!
//! 上流 model.py の `patch_sequence_with_mask`(121), `SelfAttention`(159), `TextBlock`(608),
//! `ReferenceLatentEncoder`(883) の移植。

use candle_core::{DType, Tensor};
use candle_nn::{linear, linear_no_bias, Linear, Module, VarBuilder};

use crate::attention::masked_sdpa;
use crate::config::ModelConfig;
use crate::nn::norm::RmsNorm;
use crate::nn::rope::{apply_rotary_emb, FreqsCis};
use crate::nn::swiglu::SwiGlu;
use crate::{IrodoriError, Result};

/// `(B, S, D)` を系列方向に `patch` 個ずつ束ねて `(B, S/patch, D*patch)` にする。端数は捨てる。
/// マスクは窓内がすべて有効のときだけ有効。
pub fn patch_sequence_with_mask(seq: &Tensor, mask: &Tensor, patch: usize) -> Result<(Tensor, Tensor)> {
    if patch <= 1 {
        return Ok((seq.clone(), mask.clone()));
    }
    let (b, s, d) = seq.dims3()?;
    if mask.dims() != [b, s] {
        return Err(IrodoriError::InvalidInput(format!(
            "sequence/mask shape mismatch: seq={:?}, mask={:?}",
            seq.dims(),
            mask.dims()
        )));
    }
    let usable = (s / patch) * patch;
    if usable == 0 {
        return Err(IrodoriError::InvalidInput(format!(
            "reference sequence too short for speaker_patch_size={patch}: seq_len={s}"
        )));
    }
    let n = usable / patch;
    let seq = seq.narrow(1, 0, usable)?.reshape((b, n, d * patch))?;
    let mask = mask
        .narrow(1, 0, usable)?
        .to_dtype(DType::U32)?
        .reshape((b, n, patch))?
        .sum(2)?
        .ge(patch as u32)?;
    Ok((seq, mask))
}

struct SelfAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    gate: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    head_dim: usize,
}

impl SelfAttention {
    fn load(dim: usize, heads: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        if heads == 0 || dim % heads != 0 || (dim / heads) % 2 != 0 {
            return Err(IrodoriError::InvalidConfig(format!(
                "dim={dim} must be divisible by heads={heads} with an even head_dim"
            )));
        }
        let head_dim = dim / heads;
        Ok(Self {
            wq: linear_no_bias(dim, dim, vb.pp("wq"))?,
            wk: linear_no_bias(dim, dim, vb.pp("wk"))?,
            wv: linear_no_bias(dim, dim, vb.pp("wv"))?,
            wo: linear_no_bias(dim, dim, vb.pp("wo"))?,
            gate: linear_no_bias(dim, dim, vb.pp("gate"))?,
            q_norm: RmsNorm::new(&[heads, head_dim], eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(&[heads, head_dim], eps, vb.pp("k_norm"))?,
            heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor, key_mask: Option<&Tensor>, freqs: &FreqsCis) -> Result<Tensor> {
        let (b, s, dim) = x.dims3()?;
        let shape = (b, s, self.heads, self.head_dim);
        let q = self.q_norm.forward(&self.wq.forward(x)?.reshape(shape)?)?;
        let k = self.k_norm.forward(&self.wk.forward(x)?.reshape(shape)?)?;
        let v = self.wv.forward(x)?.reshape(shape)?;
        let f = freqs.narrow(s)?;
        let (q, k) = (apply_rotary_emb(&q, &f)?, apply_rotary_emb(&k, &f)?);
        let y = masked_sdpa(&q, &k, &v, key_mask)?.reshape((b, s, dim))?;
        let y = (y * candle_nn::ops::sigmoid(&self.gate.forward(x)?)?)?;
        Ok(self.wo.forward(&y)?)
    }
}

struct TextBlock {
    attention_norm: RmsNorm,
    attention: SelfAttention,
    mlp_norm: RmsNorm,
    mlp: SwiGlu,
}

impl TextBlock {
    fn load(dim: usize, heads: usize, mlp_ratio: f64, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attention_norm: RmsNorm::new(&[dim], eps, vb.pp("attention_norm"))?,
            attention: SelfAttention::load(dim, heads, eps, vb.pp("attention"))?,
            mlp_norm: RmsNorm::new(&[dim], eps, vb.pp("mlp_norm"))?,
            // 上流は `int(dim * mlp_ratio)`（切り捨て）。
            mlp: SwiGlu::new(dim, (dim as f64 * mlp_ratio) as usize, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor, freqs: &FreqsCis) -> Result<Tensor> {
        let x = (x + self.attention.forward(&self.attention_norm.forward(x)?, Some(mask), freqs)?)?;
        Ok((&x + self.mlp.forward(&self.mlp_norm.forward(&x)?)?)?)
    }
}

/// 参照音声の潜在列（パッチ化済み）→ speaker 状態。マスク位置は常に 0。
pub struct ReferenceLatentEncoder {
    in_proj: Linear,
    blocks: Vec<TextBlock>,
    head_dim: usize,
}

impl ReferenceLatentEncoder {
    /// `vb` は `speaker_encoder` 接頭辞まで進めた VarBuilder。
    pub fn load(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let in_proj = linear(cfg.speaker_patched_latent_dim(), cfg.speaker_dim, vb.pp("in_proj"))?;
        let blocks = (0..cfg.speaker_layers)
            .map(|i| {
                TextBlock::load(
                    cfg.speaker_dim,
                    cfg.speaker_heads,
                    cfg.speaker_mlp_ratio_resolved(),
                    cfg.norm_eps,
                    vb.pp(format!("blocks.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { in_proj, blocks, head_dim: cfg.speaker_dim / cfg.speaker_heads })
    }

    /// `latent: (B, S, speaker_patched_latent_dim)`, `mask: (B, S)` → `(B, S, speaker_dim)`。
    pub fn forward(&self, latent: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (_, s, _) = latent.dims3()?;
        let mut x = (self.in_proj.forward(latent)? / 6.0)?;
        let mask_f = mask.to_dtype(x.dtype())?.unsqueeze(2)?;
        x = x.broadcast_mul(&mask_f)?;
        let freqs = FreqsCis::new(self.head_dim, s, 10000.0, x.device())?;
        for block in &self.blocks {
            x = block.forward(&x, mask, &freqs)?.broadcast_mul(&mask_f)?;
        }
        Ok(x.broadcast_mul(&mask_f)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    const DIM: usize = 8;
    const HEADS: usize = 2;
    const LAYERS: usize = 2;
    const PATCHED: usize = 6; // latent_dim 3 * speaker_patch 2
    const HIDDEN: usize = 16; // int(8 * 2.0)

    fn cfg() -> ModelConfig {
        ModelConfig {
            latent_dim: 3,
            speaker_dim: DIM,
            speaker_heads: HEADS,
            speaker_layers: LAYERS,
            speaker_patch_size: 2,
            speaker_mlp_ratio: Some(2.0),
            ..ModelConfig::default()
        }
    }

    fn fake(name: &str, shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        let seed = name.bytes().map(|b| b as f32).sum::<f32>();
        let norm = name.contains("norm");
        let v: Vec<f32> = (0..n)
            .map(|k| {
                let s = (0.9 * k as f32 + seed).sin();
                if norm { 1.0 + 0.1 * s } else { 0.3 * s }
            })
            .collect();
        Tensor::from_vec(v, shape.to_vec(), &Device::Cpu).unwrap()
    }

    fn fake_vb() -> VarBuilder<'static> {
        let mut spec: Vec<(String, Vec<usize>)> = vec![
            ("in_proj.weight".into(), vec![DIM, PATCHED]),
            ("in_proj.bias".into(), vec![DIM]),
        ];
        for i in 0..LAYERS {
            let p = format!("blocks.{i}.");
            for w in ["wq", "wk", "wv", "wo", "gate"] {
                spec.push((format!("{p}attention.{w}.weight"), vec![DIM, DIM]));
            }
            spec.push((format!("{p}attention.q_norm.weight"), vec![HEADS, DIM / HEADS]));
            spec.push((format!("{p}attention.k_norm.weight"), vec![HEADS, DIM / HEADS]));
            spec.push((format!("{p}attention_norm.weight"), vec![DIM]));
            spec.push((format!("{p}mlp_norm.weight"), vec![DIM]));
            spec.push((format!("{p}mlp.w1.weight"), vec![HIDDEN, DIM]));
            spec.push((format!("{p}mlp.w3.weight"), vec![HIDDEN, DIM]));
            spec.push((format!("{p}mlp.w2.weight"), vec![DIM, HIDDEN]));
        }
        let map: HashMap<String, Tensor> = spec.into_iter().map(|(n, s)| { let t = fake(&n, &s); (n, t) }).collect();
        VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
    }

    fn latent(rows: usize, bias: f32) -> Tensor {
        let v: Vec<f32> = (0..rows * PATCHED).map(|k| (k as f32 * 0.37).sin() + bias).collect();
        Tensor::from_vec(v, (1, rows, PATCHED), &Device::Cpu).unwrap()
    }

    fn m(v: &[u8]) -> Tensor {
        Tensor::from_slice(v, (1, v.len()), &Device::Cpu).unwrap()
    }

    #[test]
    fn patch_sequence_merges_patches_and_drops_the_tail() {
        let seq = Tensor::from_vec((0..12).map(|v| v as f32).collect::<Vec<_>>(), (1, 6, 2), &Device::Cpu).unwrap();
        let (p, pm) = patch_sequence_with_mask(&seq, &m(&[1, 1, 1, 1, 1, 0]), 4).unwrap();
        assert_eq!(p.dims(), &[1, 1, 8]);
        assert_eq!(p.flatten_all().unwrap().to_vec1::<f32>().unwrap(), (0..8).map(|v| v as f32).collect::<Vec<_>>());
        assert_eq!(pm.to_vec2::<u8>().unwrap(), vec![vec![1]]);
    }

    #[test]
    fn patch_mask_is_all_over_the_window() {
        let seq = Tensor::zeros((1, 4, 2), DType::F32, &Device::Cpu).unwrap();
        let (_, pm) = patch_sequence_with_mask(&seq, &m(&[1, 1, 1, 0]), 2).unwrap();
        assert_eq!(pm.to_vec2::<u8>().unwrap(), vec![vec![1, 0]]);
    }

    #[test]
    fn patch_size_one_is_identity() {
        let seq = Tensor::zeros((1, 3, 2), DType::F32, &Device::Cpu).unwrap();
        let (p, _) = patch_sequence_with_mask(&seq, &m(&[1, 1, 1]), 1).unwrap();
        assert_eq!(p.dims(), &[1, 3, 2]);
    }

    #[test]
    fn too_short_or_mismatched_reference_is_an_error() {
        let seq = Tensor::zeros((1, 3, 2), DType::F32, &Device::Cpu).unwrap();
        assert!(patch_sequence_with_mask(&seq, &m(&[1, 1, 1]), 4).is_err());
        assert!(patch_sequence_with_mask(&seq, &m(&[1, 1]), 2).is_err());
    }

    #[test]
    fn output_has_speaker_dim_and_zero_masked_rows() {
        let enc = ReferenceLatentEncoder::load(&cfg(), fake_vb()).unwrap();
        let y = enc.forward(&latent(5, 0.0), &m(&[1, 1, 1, 0, 0])).unwrap();
        assert_eq!(y.dims(), &[1, 5, DIM]);
        let tail = y.narrow(1, 3, 2).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(tail.iter().all(|v| *v == 0.0), "{tail:?}");
    }

    #[test]
    fn padding_content_does_not_change_valid_rows() {
        let enc = ReferenceLatentEncoder::load(&cfg(), fake_vb()).unwrap();
        let mk = m(&[1, 1, 1, 0]);
        let a = latent(4, 0.0);
        // 末尾行だけ別の値に差し替える。
        let tail = Tensor::full(5.0f32, (1, 1, PATCHED), &Device::Cpu).unwrap();
        let b = Tensor::cat(&[&a.narrow(1, 0, 3).unwrap(), &tail], 1).unwrap();
        let ya = enc.forward(&a, &mk).unwrap().narrow(1, 0, 3).unwrap();
        let yb = enc.forward(&b, &mk).unwrap().narrow(1, 0, 3).unwrap();
        crate::golden::assert_close(&ya, &yb, 1e-5, 0.0).unwrap();
    }

    #[test]
    fn real_model_matches_python_golden() {
        let (Some(model_dir), Some(_)) = (std::env::var_os("IRODORI_MODEL_DIR"), crate::golden::dir()) else {
            eprintln!("skip: IRODORI_MODEL_DIR / IRODORI_GOLDEN_DIR not set");
            return;
        };
        let w = crate::loader::Weights::open(&std::path::Path::new(&model_dir).join("model.safetensors"), &Device::Cpu).unwrap();
        let (cfg, _) = w.config().unwrap();
        let enc = ReferenceLatentEncoder::load(&cfg, w.var_builder(DType::F32).unwrap().pp("speaker_encoder")).unwrap();
        let g = crate::golden::load("speaker_enc").unwrap();
        let mask = g["mask"].to_dtype(DType::U8).unwrap();
        let (patched, pmask) = patch_sequence_with_mask(&g["latent"], &mask, cfg.speaker_patch_size).unwrap();
        crate::golden::assert_close(&patched, &g["patched"], 1e-6, 0.0).unwrap();
        let y = enc.forward(&patched, &pmask).unwrap();
        crate::golden::assert_close(&y, &g["state"], 2e-3, 2e-3).unwrap();
    }
}
