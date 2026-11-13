//! 条件エンコード: テキスト／caption／speaker をまとめて DiT 用の `EncodedConditions` にする。
//!
//! 上流 `TextToLatentRFDiT.encode_conditions`（model.py 1768–1875）の推論経路
//! （条件ドロップアウトは CFG 側でマスクとして表現するため含めない）。

use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;

use crate::config::{ModelConfig, TextEncoderConfig};
use crate::nn::norm::RmsNorm;
use crate::speaker_encoder::{patch_sequence_with_mask, ReferenceLatentEncoder};
use crate::text_encoder::TextEncoder;
use crate::{IrodoriError, Result};

/// DiT に渡す条件。マスクは `(B, S)` u8（1=有効）。
#[derive(Debug, Clone)]
pub struct EncodedConditions {
    pub text: Tensor,
    pub text_mask: Tensor,
    pub speaker: Option<(Tensor, Tensor)>,
    pub caption: Option<(Tensor, Tensor)>,
}

pub struct ConditionInputs<'a> {
    pub text_ids: &'a Tensor,
    pub text_mask: &'a Tensor,
    pub caption_ids: Option<&'a Tensor>,
    pub caption_mask: Option<&'a Tensor>,
    /// `(B, S, latent_dim * latent_patch_size)`。
    pub ref_latent: Option<&'a Tensor>,
    pub ref_mask: Option<&'a Tensor>,
}

/// 先頭に有効トークンの平均（1 トークン）を付ける。全マスク行の平均は 0、有効フラグは 0。
pub fn prepend_masked_mean_token(state: &Tensor, mask: &Tensor) -> Result<(Tensor, Tensor)> {
    let mask_f = mask.to_dtype(state.dtype())?.unsqueeze(2)?;
    let denom = mask_f.sum_keepdim(1)?.clamp(1.0, f64::MAX)?;
    let mean = state.broadcast_mul(&mask_f)?.sum_keepdim(1)?.broadcast_div(&denom)?;
    let has_any = mask.to_dtype(DType::U32)?.sum_keepdim(1)?.gt(0u32)?;
    Ok((Tensor::cat(&[&mean, state], 1)?, Tensor::cat(&[&has_any, &mask.to_dtype(DType::U8)?], 1)?))
}

pub struct Conditioner {
    text: TextEncoder,
    speaker: Option<(ReferenceLatentEncoder, RmsNorm)>,
    speaker_patch: usize,
    needs_caption: bool,
}

impl Conditioner {
    /// `vb` はチェックポイント全体の VarBuilder。
    pub fn load(vb: VarBuilder, cfg: &ModelConfig, te: &TextEncoderConfig) -> Result<Self> {
        let speaker = if cfg.use_speaker_condition_resolved() {
            Some((
                ReferenceLatentEncoder::load(cfg, vb.pp("speaker_encoder"))?,
                RmsNorm::new(&[cfg.speaker_dim], cfg.norm_eps, vb.pp("speaker_norm"))?,
            ))
        } else {
            None
        };
        Ok(Self {
            text: TextEncoder::load(vb, cfg, te)?,
            speaker,
            speaker_patch: cfg.speaker_patch_size,
            needs_caption: cfg.use_caption_condition,
        })
    }

    pub fn encode(&self, inp: &ConditionInputs) -> Result<EncodedConditions> {
        let text = self.text.encode_text(inp.text_ids, inp.text_mask)?;
        let speaker = match &self.speaker {
            None => None,
            Some((enc, norm)) => {
                let (latent, mask) = inp.ref_latent.zip(inp.ref_mask).ok_or_else(|| {
                    IrodoriError::InvalidInput("ref_latent and ref_mask are required when speaker conditioning is enabled".into())
                })?;
                let (latent, mask) = patch_sequence_with_mask(latent, mask, self.speaker_patch)?;
                let state = norm.forward(&enc.forward(&latent, &mask)?)?;
                Some(prepend_masked_mean_token(&state, &mask)?)
            }
        };
        let caption = if self.needs_caption {
            let (ids, mask) = inp.caption_ids.zip(inp.caption_mask).ok_or_else(|| {
                IrodoriError::InvalidInput("caption_ids and caption_mask are required when caption conditioning is enabled".into())
            })?;
            Some((self.text.encode_caption(ids, mask)?, mask.to_dtype(DType::U8)?))
        } else {
            None
        };
        Ok(EncodedConditions { text, text_mask: inp.text_mask.to_dtype(DType::U8)?, speaker, caption })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn mean_token_averages_valid_tokens_and_is_prepended() {
        let state = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 100.0, 100.0], (1, 3, 2), &Device::Cpu).unwrap();
        let mask = Tensor::from_slice(&[1u8, 1, 0], (1, 3), &Device::Cpu).unwrap();
        let (s, m) = prepend_masked_mean_token(&state, &mask).unwrap();
        assert_eq!(s.dims(), &[1, 4, 2]);
        let first = s.narrow(1, 0, 1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(first, vec![2.0, 3.0]);
        assert_eq!(m.to_vec2::<u8>().unwrap(), vec![vec![1, 1, 1, 0]]);
    }

    #[test]
    fn mean_token_of_fully_masked_row_is_zero_and_invalid() {
        let state = Tensor::from_slice(&[5.0f32, 6.0], (1, 1, 2), &Device::Cpu).unwrap();
        let mask = Tensor::from_slice(&[0u8], (1, 1), &Device::Cpu).unwrap();
        let (s, m) = prepend_masked_mean_token(&state, &mask).unwrap();
        let first = s.narrow(1, 0, 1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(first, vec![0.0, 0.0]);
        assert_eq!(m.to_vec2::<u8>().unwrap(), vec![vec![0, 0]]);
    }
}
