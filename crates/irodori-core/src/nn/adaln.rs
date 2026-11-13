//! Low-rank AdaLN（シフト・スケール・ゲートを低ランクで変調）。

use candle_core::{DType, Tensor, D};
use candle_nn::{linear, linear_no_bias, Linear, Module, VarBuilder};

use crate::Result;

#[derive(Debug, Clone)]
pub struct LowRankAdaLn {
    eps: f64,
    shift_down: Linear,
    scale_down: Linear,
    gate_down: Linear,
    shift_up: Linear,
    scale_up: Linear,
    gate_up: Linear,
}

impl LowRankAdaLn {
    /// `rank` は `[1, model_dim]` に丸める（上流と同じ）。
    pub fn new(model_dim: usize, rank: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let rank = rank.min(model_dim).max(1);
        Ok(Self {
            eps,
            shift_down: linear_no_bias(model_dim, rank, vb.pp("shift_down"))?,
            scale_down: linear_no_bias(model_dim, rank, vb.pp("scale_down"))?,
            gate_down: linear_no_bias(model_dim, rank, vb.pp("gate_down"))?,
            shift_up: linear(rank, model_dim, vb.pp("shift_up"))?,
            scale_up: linear(rank, model_dim, vb.pp("scale_up"))?,
            gate_up: linear(rank, model_dim, vb.pp("gate_up"))?,
        })
    }

    fn modulate(down: &Linear, up: &Linear, v: &Tensor) -> Result<Tensor> {
        let low = up.forward(&down.forward(&candle_nn::ops::silu(v)?)?)?;
        Ok((low + v)?)
    }

    /// `cond: (..., 3*model_dim)` を shift/scale/gate に分け、`(x_modulated, tanh(gate))` を返す。
    pub fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<(Tensor, Tensor)> {
        let parts = cond.chunk(3, D::Minus1)?;
        let shift = Self::modulate(&self.shift_down, &self.shift_up, &parts[0])?;
        let scale = Self::modulate(&self.scale_down, &self.scale_up, &parts[1])?;
        let gate = Self::modulate(&self.gate_down, &self.gate_up, &parts[2])?;

        let x_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let var = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x = x.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let scale = scale.to_dtype(DType::F32)?.affine(1.0, 1.0)?;
        let x = x.broadcast_mul(&scale)?.broadcast_add(&shift.to_dtype(DType::F32)?)?;
        Ok((x.to_dtype(x_dtype)?, gate.tanh()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::golden::assert_close;
    use crate::nn::test_util::{t, vb, zeros};

    fn zero_vb(rank: usize, dim: usize) -> candle_nn::VarBuilder<'static> {
        vb(vec![
            ("shift_down.weight", zeros(rank, dim)),
            ("scale_down.weight", zeros(rank, dim)),
            ("gate_down.weight", zeros(rank, dim)),
            ("shift_up.weight", zeros(dim, rank)),
            ("scale_up.weight", zeros(dim, rank)),
            ("gate_up.weight", zeros(dim, rank)),
            ("shift_up.bias", zeros(1, dim).flatten_all().unwrap()),
            ("scale_up.bias", zeros(1, dim).flatten_all().unwrap()),
            ("gate_up.bias", zeros(1, dim).flatten_all().unwrap()),
        ])
    }

    #[test]
    fn zero_up_projections_reduce_to_plain_modulation() {
        // model_dim=2, rank=1。up が全て 0 のとき shift/scale/gate は cond の各チャンクそのもの。
        let ada = LowRankAdaLn::new(2, 1, 0.0, zero_vb(1, 2)).unwrap();
        let x = t(&[1.0, 1.0], &[1, 1, 2]); // rms=1 → 正規化後も 1
        let cond = t(&[0.5, -0.5, 1.0, 2.0, 0.3, -0.3], &[1, 1, 6]);
        let (y, gate) = ada.forward(&x, &cond).unwrap();
        assert_close(&y, &t(&[1.0 * 2.0 + 0.5, 1.0 * 3.0 - 0.5], &[1, 1, 2]), 1e-6, 0.0).unwrap();
        let want_gate = t(&[0.3f32.tanh(), (-0.3f32).tanh()], &[1, 1, 2]);
        assert_close(&gate, &want_gate, 1e-6, 0.0).unwrap();
    }

    #[test]
    fn rank_is_clamped_to_model_dim() {
        // rank=8 > model_dim=2 でも重み形状は (2, 2)（上流 `min(rank, model_dim)`）。
        assert!(LowRankAdaLn::new(2, 8, 1e-5, zero_vb(2, 2)).is_ok());
    }
}
