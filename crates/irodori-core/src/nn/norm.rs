//! RMSNorm（F32 で正規化し入力 dtype に戻す）。

use candle_core::{DType, Tensor, D};
use candle_nn::VarBuilder;

use crate::Result;

/// `weight` の形は `(dim,)` または per-head の `(heads, head_dim)`。正規化は常に最後の軸。
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(weight_shape: &[usize], eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self { weight: vb.get(weight_shape.to_vec(), "weight")?, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let var = x.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let out = normed.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?;
        Ok(out.to_dtype(x_dtype)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::golden::assert_close;
    use crate::nn::test_util::{t, vb};

    #[test]
    fn normalizes_last_dim_by_rms() {
        let norm = RmsNorm::new(&[2], 0.0, vb(vec![("weight", t(&[1.0, 1.0], &[2]))])).unwrap();
        let y = norm.forward(&t(&[3.0, 4.0], &[1, 2])).unwrap();
        let rms = (12.5f32).sqrt();
        assert_close(&y, &t(&[3.0 / rms, 4.0 / rms], &[1, 2]), 1e-6, 0.0).unwrap();
    }

    #[test]
    fn per_head_weight_broadcasts_over_batch_and_sequence() {
        let w = t(&[1.0, 1.0, 2.0, 2.0], &[2, 2]); // (heads=2, head_dim=2)
        let norm = RmsNorm::new(&[2, 2], 0.0, vb(vec![("weight", w)])).unwrap();
        let x = t(&[1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2]);
        let want = t(&[1.0, 1.0, 2.0, 2.0], &[1, 1, 2, 2]);
        assert_close(&norm.forward(&x).unwrap(), &want, 1e-6, 0.0).unwrap();
    }

    #[test]
    fn missing_weight_error_names_the_key() {
        let err = RmsNorm::new(&[2], 1e-6, vb(vec![])).err().unwrap().to_string();
        assert!(err.contains("weight"), "{err}");
    }
}
