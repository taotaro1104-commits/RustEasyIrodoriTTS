//! 正弦波タイムステップ埋め込み。

use candle_core::{DType, Tensor};

use crate::{IrodoriError, Result};

/// `t: (B,)` → `(B, dim)`（`[cos | sin]`、`t` と同じ dtype）。周波数は `1000*exp(-ln(1e4)*i/half)`。
pub fn timestep_embedding(t: &Tensor, dim: usize) -> Result<Tensor> {
    if dim == 0 || dim % 2 != 0 {
        return Err(IrodoriError::InvalidInput(format!("timestep dim must be even and > 0, got {dim}")));
    }
    let half = dim / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| 1000.0 * (-(10000f32.ln()) * i as f32 / half as f32).exp())
        .collect();
    let freqs = Tensor::from_vec(freqs, (1, half), t.device())?;
    let args = t.to_dtype(DType::F32)?.unsqueeze(1)?.broadcast_mul(&freqs)?;
    let emb = Tensor::cat(&[args.cos()?, args.sin()?], 1)?;
    Ok(emb.to_dtype(t.dtype())?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::test_util::t;

    #[test]
    fn t_zero_gives_ones_then_zeros() {
        let e = timestep_embedding(&t(&[0.0], &[1]), 4).unwrap();
        assert_eq!(e.dims(), &[1, 4]);
        assert_eq!(e.to_vec2::<f32>().unwrap(), vec![vec![1.0, 1.0, 0.0, 0.0]]);
    }

    #[test]
    fn first_frequency_is_1000() {
        let e = timestep_embedding(&t(&[1.0], &[1]), 4).unwrap().to_vec2::<f32>().unwrap();
        assert!((e[0][0] - 1000f32.cos()).abs() < 1e-3);
        assert!((e[0][2] - 1000f32.sin()).abs() < 1e-3);
    }

    #[test]
    fn odd_dim_is_rejected() {
        assert!(timestep_embedding(&t(&[0.0], &[1]), 3).is_err());
    }
}
