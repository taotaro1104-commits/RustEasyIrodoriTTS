//! テキスト／caption 条件エンコーダ: 共有 ModernBERT → projector(residual_mlp) → RMSNorm。
//!
//! 上流 `PretrainedTextBackbone` / `PretrainedConditionProjector` と
//! `TextToLatentRFDiT.encode_conditions` のテキスト・caption 経路（model.py 673–880, 1825–1874）。
//! 出力は `text_norm` / `caption_norm` 適用後で、マスク位置は 0。

use candle_core::Tensor;
use candle_nn::{linear, Linear, Module, VarBuilder};
use candle_transformers::models::modernbert::ModernBert;

use crate::config::{ModelConfig, TextEncoderConfig};
use crate::nn::norm::RmsNorm;
use crate::{IrodoriError, Result};

struct Residual {
    norm: RmsNorm,
    up: Linear,
    down: Linear,
}

/// バックボーン出力を条件空間へ射影する（`linear` または `residual_mlp`）。
struct ConditionProjector {
    projector: Linear,
    residual: Option<Residual>,
}

impl ConditionProjector {
    fn load(backbone_dim: usize, out_dim: usize, cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let projector = linear(backbone_dim, out_dim, vb.pp("projector"))?;
        let kind = cfg.pretrained_projector_type.trim().to_ascii_lowercase();
        let residual = match kind.as_str() {
            "linear" => None,
            "residual_mlp" => {
                let hidden =
                    ((out_dim as f64 * cfg.pretrained_projector_hidden_ratio).round() as usize).max(1);
                Some(Residual {
                    norm: RmsNorm::new(&[backbone_dim], cfg.norm_eps, vb.pp("residual_norm"))?,
                    up: linear(backbone_dim, hidden, vb.pp("residual_up"))?,
                    down: linear(hidden, out_dim, vb.pp("residual_down"))?,
                })
            }
            other => {
                return Err(IrodoriError::InvalidConfig(format!(
                    "pretrained projector type must be 'linear' or 'residual_mlp', got {other:?}"
                )))
            }
        };
        Ok(Self { projector, residual })
    }

    fn forward(&self, state: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let mut out = self.projector.forward(state)?;
        if let Some(r) = &self.residual {
            let h = candle_nn::ops::silu(&r.up.forward(&r.norm.forward(state)?)?)?;
            out = (out + r.down.forward(&h)?)?;
        }
        Ok(out.broadcast_mul(mask)?)
    }
}

pub struct TextEncoder {
    backbone: ModernBert,
    text: ConditionProjector,
    text_norm: RmsNorm,
    caption: Option<(ConditionProjector, RmsNorm)>,
}

impl TextEncoder {
    /// `vb` はチェックポイント全体の VarBuilder（`pretrained_text_backbone.*` 等を含む）。
    pub fn load(vb: VarBuilder, cfg: &ModelConfig, te: &TextEncoderConfig) -> Result<Self> {
        if !cfg.use_pretrained_text_encoder() {
            return Err(IrodoriError::InvalidConfig(
                "only text_encoder_type='pretrained' is supported".into(),
            ));
        }
        te.validate()?;
        // candle の ModernBert は `model.*` を探すが、チェックポイントは `backbone.*`（HF ModernBertModel）。
        let bb_vb = vb
            .clone()
            .rename_f(|n| n.replacen("backbone.model.", "backbone.", 1))
            .pp("pretrained_text_backbone")
            .pp("backbone");
        let backbone = ModernBert::load(bb_vb, &te.into_candle())?;
        let text = ConditionProjector::load(te.hidden_size, cfg.text_dim, cfg, vb.pp("text_encoder"))?;
        let text_norm = RmsNorm::new(&[cfg.text_dim], cfg.norm_eps, vb.pp("text_norm"))?;
        let caption = if cfg.use_caption_condition {
            let dim = cfg.caption_dim_resolved();
            Some((
                ConditionProjector::load(te.hidden_size, dim, cfg, vb.pp("caption_encoder"))?,
                RmsNorm::new(&[dim], cfg.norm_eps, vb.pp("caption_norm"))?,
            ))
        } else {
            None
        };
        Ok(Self { backbone, text, text_norm, caption })
    }

    /// `ids: (B, L) u32`, `mask: (B, L)`（1=有効）→ `(B, L, dim)`。
    fn run(&self, proj: &ConditionProjector, norm: &RmsNorm, ids: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let state = self.backbone.forward(ids, mask)?;
        let mask_f = mask.to_dtype(state.dtype())?.unsqueeze(2)?;
        let state = state.broadcast_mul(&mask_f)?;
        norm.forward(&proj.forward(&state, &mask_f)?)
    }

    pub fn encode_text(&self, ids: &Tensor, mask: &Tensor) -> Result<Tensor> {
        self.run(&self.text, &self.text_norm, ids, mask)
    }

    pub fn encode_caption(&self, ids: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (proj, norm) = self
            .caption
            .as_ref()
            .ok_or_else(|| IrodoriError::InvalidInput("caption condition is disabled for this model".into()))?;
        self.run(proj, norm, ids, mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelConfig, RopeParameters, RopeParams, TextEncoderConfig};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    const V: usize = 20;
    const H: usize = 8;
    const I: usize = 16;
    const OUT: usize = 4;
    const HID: usize = 8; // round(OUT * 2.0)
    const B: &str = "pretrained_text_backbone.backbone.";

    fn te_cfg() -> TextEncoderConfig {
        TextEncoderConfig {
            vocab_size: V,
            hidden_size: H,
            intermediate_size: I,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            max_position_embeddings: 64,
            norm_eps: 1e-5,
            pad_token_id: 3,
            bos_token_id: 1,
            local_attention: 4,
            global_attn_every_n_layers: 3,
            rope_parameters: RopeParameters {
                sliding_attention: RopeParams { rope_theta: 10000.0 },
                full_attention: RopeParams { rope_theta: 160000.0 },
            },
            layer_types: vec![
                "full_attention".into(),
                "sliding_attention".into(),
                "sliding_attention".into(),
                "full_attention".into(),
            ],
        }
    }

    fn model_cfg() -> ModelConfig {
        ModelConfig {
            text_dim: OUT,
            use_caption_condition: true,
            text_encoder_type: "pretrained".into(),
            pretrained_projector_type: "residual_mlp".into(),
            pretrained_projector_hidden_ratio: 2.0,
            ..ModelConfig::default()
        }
    }

    /// 決定的な擬似乱数重み。norm 系は 1 付近。
    fn fake(name: &str, shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        let seed = name.bytes().map(|b| b as f32).sum::<f32>();
        let is_norm = name.contains("norm");
        let v: Vec<f32> = (0..n)
            .map(|k| {
                let s = (0.7 * k as f32 + seed).sin();
                if is_norm { 1.0 + 0.1 * s } else { 0.3 * s }
            })
            .collect();
        Tensor::from_vec(v, shape.to_vec(), &Device::Cpu).unwrap()
    }

    fn fake_vb(with_caption: bool) -> VarBuilder<'static> {
        let mut spec: Vec<(String, Vec<usize>)> = vec![
            (format!("{B}embeddings.tok_embeddings.weight"), vec![V, H]),
            (format!("{B}embeddings.norm.weight"), vec![H]),
            (format!("{B}final_norm.weight"), vec![H]),
            ("text_norm.weight".into(), vec![OUT]),
        ];
        for i in 0..4 {
            spec.push((format!("{B}layers.{i}.attn.Wqkv.weight"), vec![3 * H, H]));
            spec.push((format!("{B}layers.{i}.attn.Wo.weight"), vec![H, H]));
            spec.push((format!("{B}layers.{i}.mlp.Wi.weight"), vec![2 * I, H]));
            spec.push((format!("{B}layers.{i}.mlp.Wo.weight"), vec![H, I]));
            spec.push((format!("{B}layers.{i}.mlp_norm.weight"), vec![H]));
            if i > 0 {
                spec.push((format!("{B}layers.{i}.attn_norm.weight"), vec![H]));
            }
        }
        let mut heads = vec!["text_encoder"];
        if with_caption {
            heads.push("caption_encoder");
            spec.push(("caption_norm.weight".into(), vec![OUT]));
        }
        for h in heads {
            spec.push((format!("{h}.projector.weight"), vec![OUT, H]));
            spec.push((format!("{h}.projector.bias"), vec![OUT]));
            spec.push((format!("{h}.residual_norm.weight"), vec![H]));
            spec.push((format!("{h}.residual_up.weight"), vec![HID, H]));
            spec.push((format!("{h}.residual_up.bias"), vec![HID]));
            spec.push((format!("{h}.residual_down.weight"), vec![OUT, HID]));
            spec.push((format!("{h}.residual_down.bias"), vec![OUT]));
        }
        let map: HashMap<String, Tensor> = spec.into_iter().map(|(n, s)| { let t = fake(&n, &s); (n, t) }).collect();
        VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
    }

    fn ids(rows: &[&[u32]]) -> Tensor {
        let l = rows[0].len();
        Tensor::from_vec(rows.concat(), (rows.len(), l), &Device::Cpu).unwrap()
    }

    fn mask(rows: &[&[u8]]) -> Tensor {
        let l = rows[0].len();
        Tensor::from_vec(rows.concat(), (rows.len(), l), &Device::Cpu).unwrap()
    }

    fn load(with_caption: bool) -> TextEncoder {
        let mut cfg = model_cfg();
        cfg.use_caption_condition = with_caption;
        TextEncoder::load(fake_vb(with_caption), &cfg, &te_cfg()).unwrap()
    }

    #[test]
    fn output_shape_is_batch_len_text_dim() {
        let enc = load(true);
        let out = enc.encode_text(&ids(&[&[1, 5, 6, 3]]), &mask(&[&[1, 1, 1, 0]])).unwrap();
        assert_eq!(out.dims(), &[1, 4, OUT]);
    }

    #[test]
    fn masked_positions_are_exactly_zero() {
        let enc = load(true);
        let out = enc.encode_text(&ids(&[&[1, 5, 6, 3]]), &mask(&[&[1, 1, 1, 0]])).unwrap();
        let last = out.narrow(1, 3, 1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(last.iter().all(|v| *v == 0.0), "{last:?}");
    }

    #[test]
    fn padding_does_not_change_valid_tokens() {
        let enc = load(true);
        let m = mask(&[&[1, 1, 1, 0, 0]]);
        let a = enc.encode_text(&ids(&[&[1, 5, 6, 3, 3]]), &m).unwrap();
        let b = enc.encode_text(&ids(&[&[1, 5, 6, 17, 9]]), &m).unwrap();
        let (a, b) = (a.narrow(1, 0, 3).unwrap(), b.narrow(1, 0, 3).unwrap());
        crate::golden::assert_close(&a, &b, 1e-5, 0.0).unwrap();
    }

    #[test]
    fn fully_masked_input_gives_zeros_without_nan() {
        let enc = load(true);
        let out = enc.encode_text(&ids(&[&[1, 5, 6]]), &mask(&[&[0, 0, 0]])).unwrap();
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| *x == 0.0), "{v:?}");
    }

    #[test]
    fn caption_uses_the_same_backbone_but_its_own_projector() {
        let enc = load(true);
        let (i, m) = (ids(&[&[1, 5, 6]]), mask(&[&[1, 1, 1]]));
        let t = enc.encode_text(&i, &m).unwrap();
        let c = enc.encode_caption(&i, &m).unwrap();
        assert_eq!(t.dims(), c.dims());
        assert!(crate::golden::assert_close(&t, &c, 1e-4, 0.0).is_err(), "projectors should differ");
    }

    #[test]
    fn caption_without_caption_condition_is_an_error() {
        let enc = load(false);
        assert!(enc.encode_caption(&ids(&[&[1]]), &mask(&[&[1]])).is_err());
    }

    #[test]
    fn real_model_matches_python_golden() {
        let (Some(model_dir), Some(_)) = (std::env::var_os("IRODORI_MODEL_DIR"), crate::golden::dir()) else {
            eprintln!("skip: IRODORI_MODEL_DIR / IRODORI_GOLDEN_DIR not set");
            return;
        };
        let w = crate::loader::Weights::open(&std::path::Path::new(&model_dir).join("model.safetensors"), &Device::Cpu).unwrap();
        let (cfg, te) = w.config().unwrap();
        let enc = TextEncoder::load(w.var_builder(DType::F32).unwrap(), &cfg, &te).unwrap();
        let g = crate::golden::load("text_enc").unwrap();
        let ids = g["ids"].to_dtype(DType::U32).unwrap();
        let mask = g["mask"].to_dtype(DType::U8).unwrap();
        let text = enc.encode_text(&ids, &mask).unwrap();
        crate::golden::assert_close(&text, &g["text_state"], 2e-3, 2e-3).unwrap();
        let cap = enc.encode_caption(&ids, &mask).unwrap();
        crate::golden::assert_close(&cap, &g["caption_state"], 2e-3, 2e-3).unwrap();
    }
}
