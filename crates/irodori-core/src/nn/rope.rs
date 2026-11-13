//! RoPE（隣接ペアを複素数とみなす interleaved 方式）。
//!
//! 上流 `precompute_freqs_cis` / `apply_rotary_emb` と同じ式を F32 で計算する。
//! 入力 `x` は `(B, S, H, Dh)`。

use candle_core::{DType, Device, Tensor};

use crate::{IrodoriError, Result};

/// 位置ごとの回転角の cos/sin テーブル。各 `(end, dim/2)` の F32。
#[derive(Debug, Clone)]
pub struct FreqsCis {
    cos: Tensor,
    sin: Tensor,
}

impl FreqsCis {
    pub fn new(dim: usize, end: usize, theta: f32, device: &Device) -> Result<Self> {
        if dim == 0 || dim % 2 != 0 {
            return Err(IrodoriError::InvalidInput(format!("rope dim must be even and > 0, got {dim}")));
        }
        let half = dim / 2;
        let inv: Vec<f32> = (0..half)
            .map(|i| 1.0f32 / theta.powf((2 * i) as f32 / dim as f32))
            .collect();
        let mut cos = Vec::with_capacity(end * half);
        let mut sin = Vec::with_capacity(end * half);
        for pos in 0..end {
            for f in &inv {
                let angle = pos as f32 * f;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        Ok(Self {
            cos: Tensor::from_vec(cos, (end, half), device)?,
            sin: Tensor::from_vec(sin, (end, half), device)?,
        })
    }

    pub fn len(&self) -> usize {
        self.cos.dim(0).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 先頭 `len` 位置だけを取り出す。テーブルより長ければエラー。
    pub fn narrow(&self, len: usize) -> Result<Self> {
        if len > self.len() {
            return Err(IrodoriError::InvalidInput(format!(
                "sequence length {len} exceeds rope table length {}",
                self.len()
            )));
        }
        Ok(Self { cos: self.cos.narrow(0, 0, len)?, sin: self.sin.narrow(0, 0, len)? })
    }
}

/// `x: (B, S, H, Dh)` の隣接ペア `(x[2i], x[2i+1])` を位置ごとの角度で回転する。
pub fn apply_rotary_emb(x: &Tensor, freqs: &FreqsCis) -> Result<Tensor> {
    let (b, s, h, dh) = x.dims4()?;
    if dh % 2 != 0 || freqs.len() != s {
        return Err(IrodoriError::InvalidInput(format!(
            "rope shape mismatch: x={:?}, table_len={}",
            x.dims(),
            freqs.len()
        )));
    }
    let half = dh / 2;
    let pairs = x.to_dtype(DType::F32)?.reshape((b, s, h, half, 2))?;
    let x0 = pairs.narrow(4, 0, 1)?.squeeze(4)?;
    let x1 = pairs.narrow(4, 1, 1)?.squeeze(4)?;
    let cos = freqs.cos.reshape((1, s, 1, half))?;
    let sin = freqs.sin.reshape((1, s, 1, half))?;
    let re = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let im = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
    let out = Tensor::stack(&[re, im], 4)?.reshape((b, s, h, dh))?;
    Ok(out.to_dtype(x.dtype())?)
}

/// half-RoPE: ヘッド軸の前半分（`chunk(2, dim=-2)` の先頭側）だけを回転し、残りはそのまま。
pub fn apply_rotary_half(x: &Tensor, freqs: &FreqsCis) -> Result<Tensor> {
    let (_, _, h, _) = x.dims4()?;
    let first = h.div_ceil(2);
    let rotated = apply_rotary_emb(&x.narrow(2, 0, first)?, freqs)?;
    if first == h {
        return Ok(rotated);
    }
    Ok(Tensor::cat(&[&rotated, &x.narrow(2, first, h - first)?], 2)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::golden::assert_close;
    use crate::nn::test_util::t;
    use candle_core::Device;

    #[test]
    fn position_zero_is_identity() {
        let f = FreqsCis::new(4, 3, 10000.0, &Device::Cpu).unwrap();
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 4]);
        assert_close(&apply_rotary_emb(&x, &f.narrow(1).unwrap()).unwrap(), &x, 1e-6, 0.0).unwrap();
    }

    #[test]
    fn first_pair_rotates_by_position_radians() {
        // dim=2 → 周波数は 1/theta^0 = 1。位置 1 で角度 1 rad。
        let f = FreqsCis::new(2, 2, 10000.0, &Device::Cpu).unwrap();
        let x = t(&[1.0, 0.0, 1.0, 0.0], &[1, 2, 1, 2]);
        let y = apply_rotary_emb(&x, &f.narrow(2).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let want = [1.0, 0.0, 1f32.cos(), 1f32.sin()];
        for (a, b) in y.iter().zip(want) {
            assert!((a - b).abs() < 1e-6, "{y:?}");
        }
    }

    #[test]
    fn half_rope_leaves_second_half_of_heads_untouched() {
        let f = FreqsCis::new(2, 2, 10000.0, &Device::Cpu).unwrap();
        // (B=1, S=2, H=2, Dh=2): head0 は回転、head1 はそのまま。
        let x = t(&[1.0, 0.0, 5.0, 6.0, 1.0, 0.0, 7.0, 8.0], &[1, 2, 2, 2]);
        let y = apply_rotary_half(&x, &f.narrow(2).unwrap()).unwrap();
        let head1 = y.narrow(2, 1, 1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(head1, vec![5.0, 6.0, 7.0, 8.0]);
        let h0p1 = y
            .narrow(2, 0, 1)
            .unwrap()
            .narrow(1, 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!((h0p1[0] - 1f32.cos()).abs() < 1e-6 && (h0p1[1] - 1f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn narrow_longer_than_table_is_an_error() {
        let f = FreqsCis::new(2, 2, 10000.0, &Device::Cpu).unwrap();
        assert!(f.narrow(3).is_err());
    }
}
