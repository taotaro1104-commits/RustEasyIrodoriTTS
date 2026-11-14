//! Echo 式 JointAttention: 潜在トークン（self）と text/speaker/caption 文脈を 1 つの KV に連結して注意する。
//!
//! 上流 model.py の `JointAttention`（202–426）。self の q/k だけ half-RoPE、文脈 k は `k_norm` のみ。
//! KV の連結順は self, text[, speaker][, caption]。

use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::attention::masked_sdpa;
use crate::nn::norm::RmsNorm;
use crate::nn::qlinear::{linear_no_bias, QLinear, QuantKind};
use crate::nn::rope::{apply_rotary_half, FreqsCis};
use crate::{IrodoriError, Result};

pub struct JointAttention {
    wq: QLinear,
    wk: QLinear,
    wv: QLinear,
    wk_text: QLinear,
    wv_text: QLinear,
    speaker: Option<(QLinear, QLinear)>,
    caption: Option<(QLinear, QLinear)>,
    gate: QLinear,
    wo: QLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    head_dim: usize,
    dim: usize,
}

fn kv_pair(in_dim: usize, out_dim: usize, vb: &VarBuilder, k: &str, v: &str, quant: Option<QuantKind>) -> Result<(QLinear, QLinear)> {
    Ok((linear_no_bias(in_dim, out_dim, vb.pp(k), quant)?, linear_no_bias(in_dim, out_dim, vb.pp(v), quant)?))
}

/// `(B, S_self + ctx...)` の u8 キーマスクを作る。`self_mask=None` は全有効。
pub fn concat_kv_mask(batch: usize, query_len: usize, self_mask: Option<&Tensor>, ctx_masks: &[&Tensor], device: &Device) -> Result<Tensor> {
    let own = match self_mask {
        Some(m) => m.to_dtype(DType::U8)?,
        None => Tensor::ones((batch, query_len), DType::U8, device)?,
    };
    let mut parts: Vec<Tensor> = vec![own];
    for m in ctx_masks {
        parts.push(m.to_dtype(DType::U8)?);
    }
    Ok(Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1)?)
}

impl JointAttention {
    pub fn load(
        dim: usize,
        heads: usize,
        text_ctx_dim: usize,
        speaker_ctx_dim: Option<usize>,
        caption_ctx_dim: Option<usize>,
        eps: f64,
        vb: VarBuilder,
        quant: Option<QuantKind>,
    ) -> Result<Self> {
        if heads == 0 || dim % heads != 0 || (dim / heads) % 2 != 0 {
            return Err(IrodoriError::InvalidConfig(format!(
                "dim={dim} must be divisible by heads={heads} with an even head_dim"
            )));
        }
        let head_dim = dim / heads;
        let (wk_text, wv_text) = kv_pair(text_ctx_dim, dim, &vb, "wk_text", "wv_text", quant)?;
        let speaker = speaker_ctx_dim.map(|d| kv_pair(d, dim, &vb, "wk_speaker", "wv_speaker", quant)).transpose()?;
        let caption = caption_ctx_dim.map(|d| kv_pair(d, dim, &vb, "wk_caption", "wv_caption", quant)).transpose()?;
        Ok(Self {
            wq: linear_no_bias(dim, dim, vb.pp("wq"), quant)?,
            wk: linear_no_bias(dim, dim, vb.pp("wk"), quant)?,
            wv: linear_no_bias(dim, dim, vb.pp("wv"), quant)?,
            wk_text,
            wv_text,
            speaker,
            caption,
            gate: linear_no_bias(dim, dim, vb.pp("gate"), quant)?,
            wo: linear_no_bias(dim, dim, vb.pp("wo"), quant)?,
            q_norm: RmsNorm::new(&[heads, head_dim], eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(&[heads, head_dim], eps, vb.pp("k_norm"))?,
            heads,
            head_dim,
            dim,
        })
    }

    fn project(&self, pair: &(QLinear, QLinear), ctx: &Tensor) -> Result<[Tensor; 2]> {
        let (b, s, _) = ctx.dims3()?;
        let shape = (b, s, self.heads, self.head_dim);
        let k = self.k_norm.forward(&pair.0.forward(ctx)?.reshape(shape)?)?;
        let v = pair.1.forward(ctx)?.reshape(shape)?;
        Ok([k, v])
    }

    /// 静的な条件の K/V 射影。`[k_text, v_text, (k_spk, v_spk), (k_cap, v_cap)]`（有効な条件のみ）。
    pub fn project_context_kv(&self, text: &Tensor, speaker: Option<&Tensor>, caption: Option<&Tensor>) -> Result<Vec<Tensor>> {
        let mut out: Vec<Tensor> = self.project(&(self.wk_text.clone(), self.wv_text.clone()), text)?.into();
        if let Some(pair) = &self.speaker {
            let ctx = speaker.ok_or_else(|| IrodoriError::InvalidInput("speaker_context is required when speaker conditioning is enabled".into()))?;
            out.extend(self.project(pair, ctx)?);
        }
        if let Some(pair) = &self.caption {
            let ctx = caption.ok_or_else(|| IrodoriError::InvalidInput("caption_context is required when caption conditioning is enabled".into()))?;
            out.extend(self.project(pair, ctx)?);
        }
        Ok(out)
    }

    /// `x: (B, S, dim)`, `kv`: [`project_context_kv`] の出力, `kv_mask: (B, S + ctx)`。
    /// `freqs` は head_dim 用のテーブル（長さ ≥ S）。
    pub fn forward(&self, x: &Tensor, kv: &[Tensor], kv_mask: &Tensor, freqs: &FreqsCis) -> Result<Tensor> {
        let expected = 2 + 2 * usize::from(self.speaker.is_some()) + 2 * usize::from(self.caption.is_some());
        if kv.len() != expected {
            return Err(IrodoriError::InvalidInput(format!("expected {expected} context kv tensors, got {}", kv.len())));
        }
        let (b, s, _) = x.dims3()?;
        let shape = (b, s, self.heads, self.head_dim);
        let f = freqs.narrow(s)?;
        let q = apply_rotary_half(&self.q_norm.forward(&self.wq.forward(x)?.reshape(shape)?)?, &f)?;
        let k_self = apply_rotary_half(&self.k_norm.forward(&self.wk.forward(x)?.reshape(shape)?)?, &f)?;
        let v_self = self.wv.forward(x)?.reshape(shape)?;
        let mut ks: Vec<&Tensor> = vec![&k_self];
        let mut vs: Vec<&Tensor> = vec![&v_self];
        for pair in kv.chunks(2) {
            ks.push(&pair[0]);
            vs.push(&pair[1]);
        }
        let (k, v) = (Tensor::cat(&ks, 1)?, Tensor::cat(&vs, 1)?);
        let y = masked_sdpa(&q, &k, &v, Some(kv_mask))?.reshape((b, s, self.dim))?;
        let y = (y * candle_nn::ops::sigmoid(&self.gate.forward(x)?)?)?;
        Ok(self.wo.forward(&y)?)
    }
}

#[cfg(test)]
pub(crate) mod test_spec {
    /// JointAttention の合成重み仕様（`prefix` 付き）。
    pub fn spec(prefix: &str, dim: usize, heads: usize, text: usize, spk: Option<usize>, cap: Option<usize>) -> Vec<(String, Vec<usize>)> {
        let mut s: Vec<(String, Vec<usize>)> = Vec::new();
        for n in ["wq", "wk", "wv", "gate", "wo"] {
            s.push((format!("{prefix}{n}.weight"), vec![dim, dim]));
        }
        s.push((format!("{prefix}wk_text.weight"), vec![dim, text]));
        s.push((format!("{prefix}wv_text.weight"), vec![dim, text]));
        if let Some(d) = spk {
            s.push((format!("{prefix}wk_speaker.weight"), vec![dim, d]));
            s.push((format!("{prefix}wv_speaker.weight"), vec![dim, d]));
        }
        if let Some(d) = cap {
            s.push((format!("{prefix}wk_caption.weight"), vec![dim, d]));
            s.push((format!("{prefix}wv_caption.weight"), vec![dim, d]));
        }
        s.push((format!("{prefix}q_norm.weight"), vec![heads, dim / heads]));
        s.push((format!("{prefix}k_norm.weight"), vec![heads, dim / heads]));
        s
    }
}

#[cfg(test)]
mod tests {
    use super::test_spec::spec;
    use super::*;
    use crate::nn::rope::FreqsCis;
    use crate::nn::test_util::{fake_tensor, vb_from_spec};
    use candle_core::{Device, Tensor};

    const DIM: usize = 8;
    const HEADS: usize = 2;

    fn attn(spk: bool, cap: bool) -> JointAttention {
        let vb = vb_from_spec(spec("", DIM, HEADS, 4, spk.then_some(6), cap.then_some(5)));
        JointAttention::load(DIM, HEADS, 4, spk.then_some(6), cap.then_some(5), 1e-5, vb, None).unwrap()
    }

    fn x(s: usize) -> Tensor {
        fake_tensor("x", &[1, s, DIM])
    }

    fn ctx(name: &str, s: usize, d: usize) -> Tensor {
        fake_tensor(name, &[1, s, d])
    }

    fn ones(n: usize) -> Tensor {
        Tensor::ones((1, n), candle_core::DType::U8, &Device::Cpu).unwrap()
    }

    #[test]
    fn project_context_kv_returns_pairs_per_enabled_condition() {
        let t = ctx("t", 3, 4);
        assert_eq!(attn(false, false).project_context_kv(&t, None, None).unwrap().len(), 2);
        let s = ctx("s", 2, 6);
        assert_eq!(attn(true, false).project_context_kv(&t, Some(&s), None).unwrap().len(), 4);
        let c = ctx("c", 2, 5);
        assert_eq!(attn(true, true).project_context_kv(&t, Some(&s), Some(&c)).unwrap().len(), 6);
    }

    #[test]
    fn missing_required_context_is_an_error() {
        let t = ctx("t", 3, 4);
        assert!(attn(true, false).project_context_kv(&t, None, None).is_err());
        assert!(attn(false, true).project_context_kv(&t, None, None).is_err());
    }

    #[test]
    fn output_keeps_query_shape() {
        let a = attn(true, true);
        let kv = a.project_context_kv(&ctx("t", 3, 4), Some(&ctx("s", 2, 6)), Some(&ctx("c", 2, 5))).unwrap();
        let f = FreqsCis::new(DIM / HEADS, 8, 10000.0, &Device::Cpu).unwrap();
        let mask = ones(5 + 3 + 2 + 2);
        let y = a.forward(&x(5), &kv, &mask, &f).unwrap();
        assert_eq!(y.dims(), &[1, 5, DIM]);
    }

    #[test]
    fn masked_context_tokens_do_not_influence_output() {
        let a = attn(false, false);
        let f = FreqsCis::new(DIM / HEADS, 8, 10000.0, &Device::Cpu).unwrap();
        // kv 順は self(4) + text(3)。text の最後の 1 トークンをマスクし、その値だけ変える。
        let mask = Tensor::from_slice(&[1u8, 1, 1, 1, 1, 1, 0], (1, 7), &Device::Cpu).unwrap();
        let t1 = ctx("t", 3, 4);
        let noise = Tensor::full(7.0f32, (1, 1, 4), &Device::Cpu).unwrap();
        let t2 = Tensor::cat(&[&t1.narrow(1, 0, 2).unwrap(), &noise], 1).unwrap();
        let y1 = a.forward(&x(4), &a.project_context_kv(&t1, None, None).unwrap(), &mask, &f).unwrap();
        let y2 = a.forward(&x(4), &a.project_context_kv(&t2, None, None).unwrap(), &mask, &f).unwrap();
        crate::golden::assert_close(&y1, &y2, 1e-5, 0.0).unwrap();
    }
}
