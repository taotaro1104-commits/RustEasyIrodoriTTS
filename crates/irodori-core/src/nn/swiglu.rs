//! SwiGLU MLP: `w2(silu(w1 x) * w3 x)`。

use candle_core::Tensor;
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

use crate::Result;

#[derive(Debug, Clone)]
pub struct SwiGlu {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl SwiGlu {
    pub fn new(dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w1: linear_no_bias(dim, hidden_dim, vb.pp("w1"))?,
            w2: linear_no_bias(hidden_dim, dim, vb.pp("w2"))?,
            w3: linear_no_bias(dim, hidden_dim, vb.pp("w3"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gated = (candle_nn::ops::silu(&self.w1.forward(x)?)? * self.w3.forward(x)?)?;
        Ok(self.w2.forward(&gated)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::golden::assert_close;
    use crate::nn::test_util::{t, vb};

    #[test]
    fn identity_weights_give_silu_x_times_x() {
        let eye = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let m = SwiGlu::new(
            2,
            2,
            vb(vec![("w1.weight", eye.clone()), ("w2.weight", eye.clone()), ("w3.weight", eye)]),
        )
        .unwrap();
        let y = m.forward(&t(&[1.0, 2.0], &[1, 2])).unwrap();
        let silu = |v: f32| v / (1.0 + (-v).exp());
        assert_close(&y, &t(&[silu(1.0) * 1.0, silu(2.0) * 2.0], &[1, 2]), 1e-6, 0.0).unwrap();
    }
}
