//! DACVAE の基本層: Snake 活性、weight-norm 畳み込み、reflect パディング、残差ユニット。
//!
//! 参照: `DACVAE-selfcontained`（Apache-2.0）の `snake` / `NormConv1d` / `NormConvTranspose1d` /
//! `ResidualUnit`。weight-norm はロード時に `w = g * v / ||v||` へ畳み込む。

use candle_core::{DType, Tensor};

use crate::Result;

/// 重み取得関数（テンソル名 → テンソル）。不足時は `IrodoriError::MissingKey`。
pub type Fetch<'a> = &'a dyn Fn(&str) -> Result<Tensor>;

/// `x + sin²(αx) / (α + 1e-9)`。`x: (B, C, T)`, `alpha: (1, C, 1)`。
pub fn snake(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let s = x.broadcast_mul(alpha)?.sin()?.sqr()?;
    Ok((x + s.broadcast_div(&(alpha + 1e-9)?)?)?)
}

/// weight-norm の畳み込み: 出力チャンネル（第 0 軸）ごとに `w = g * v / ||v||`。
pub fn fold_weight_norm(g: &Tensor, v: &Tensor) -> Result<Tensor> {
    let norm = v.sqr()?.sum_keepdim(vec![1usize, 2])?.sqrt()?;
    Ok(v.broadcast_mul(&g.broadcast_div(&norm)?)?)
}

/// 末尾に reflect パディング（端の値を繰り返さない）。`pad >= T` はエラー。
pub fn reflect_pad_right(x: &Tensor, pad: usize) -> Result<Tensor> {
    if pad == 0 {
        return Ok(x.clone());
    }
    let len = x.dim(2)?;
    if pad >= len {
        return Err(crate::IrodoriError::InvalidInput(format!("reflect pad {pad} needs a signal longer than {pad}, got {len}")));
    }
    let idx: Vec<u32> = (0..pad).map(|i| (len - 2 - i) as u32).collect();
    let idx = Tensor::from_vec(idx, pad, x.device())?;
    Ok(Tensor::cat(&[x, &x.index_select(&idx, 2)?], 2)?)
}

pub struct Conv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    padding: usize,
    stride: usize,
    dilation: usize,
}

impl Conv1d {
    pub fn new(weight: Tensor, bias: Option<Tensor>, padding: usize, stride: usize, dilation: usize) -> Self {
        Self { weight, bias, padding, stride, dilation }
    }

    /// `{prefix}.weight_g/.weight_v/.bias` を読み、weight-norm を畳み込んで `dtype` に変換する。
    pub fn load_wn(fetch: Fetch, prefix: &str, padding: usize, stride: usize, dilation: usize, dtype: DType) -> Result<Self> {
        let w = fold_weight_norm(&fetch(&format!("{prefix}.weight_g"))?, &fetch(&format!("{prefix}.weight_v"))?)?;
        let b = fetch(&format!("{prefix}.bias"))?;
        Ok(Self::new(w.to_dtype(dtype)?, Some(b.to_dtype(dtype)?), padding, stride, dilation))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv1d(&self.weight, self.padding, self.stride, self.dilation, 1)?;
        match &self.bias {
            Some(b) => Ok(y.broadcast_add(&b.reshape((1, b.dim(0)?, 1))?)?),
            None => Ok(y),
        }
    }
}

pub struct ConvT1d {
    weight: Tensor,
    bias: Option<Tensor>,
    padding: usize,
    stride: usize,
}

impl ConvT1d {
    pub fn new(weight: Tensor, bias: Option<Tensor>, padding: usize, stride: usize) -> Self {
        Self { weight, bias, padding, stride }
    }

    pub fn load_wn(fetch: Fetch, prefix: &str, padding: usize, stride: usize, dtype: DType) -> Result<Self> {
        let w = fold_weight_norm(&fetch(&format!("{prefix}.weight_g"))?, &fetch(&format!("{prefix}.weight_v"))?)?;
        let b = fetch(&format!("{prefix}.bias"))?;
        Ok(Self::new(w.to_dtype(dtype)?, Some(b.to_dtype(dtype)?), padding, stride))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv_transpose1d(&self.weight, self.padding, 0, self.stride, 1, 1)?;
        match &self.bias {
            Some(b) => Ok(y.broadcast_add(&b.reshape((1, b.dim(0)?, 1))?)?),
            None => Ok(y),
        }
    }
}

/// `Snake → Conv(k=7, dilation) → Snake → Conv(k=1)` + 恒等の残差。
pub struct ResidualUnit {
    alpha0: Tensor,
    conv0: Conv1d,
    alpha1: Tensor,
    conv1: Conv1d,
}

impl ResidualUnit {
    pub fn load(fetch: Fetch, prefix: &str, dilation: usize, dtype: DType) -> Result<Self> {
        Ok(Self {
            alpha0: fetch(&format!("{prefix}.block.0.alpha"))?.to_dtype(dtype)?,
            conv0: Conv1d::load_wn(fetch, &format!("{prefix}.block.1"), 3 * dilation, 1, dilation, dtype)?,
            alpha1: fetch(&format!("{prefix}.block.2.alpha"))?.to_dtype(dtype)?,
            conv1: Conv1d::load_wn(fetch, &format!("{prefix}.block.3"), 0, 1, 1, dtype)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.conv0.forward(&snake(x, &self.alpha0)?)?;
        let y = self.conv1.forward(&snake(&y, &self.alpha1)?)?;
        Ok((y + x)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::golden::assert_close;
    use crate::nn::test_util::t;
    use candle_core::Tensor;

    #[test]
    fn snake_matches_formula() {
        // x + sin^2(alpha*x) / alpha  （alpha=2）
        let x = t(&[0.5, -1.0], &[1, 1, 2]);
        let alpha = t(&[2.0], &[1, 1, 1]);
        let want: Vec<f32> = [0.5f32, -1.0].iter().map(|v| v + (2.0 * v).sin().powi(2) / 2.0).collect();
        assert_close(&snake(&x, &alpha).unwrap(), &t(&want, &[1, 1, 2]), 1e-6, 0.0).unwrap();
    }

    #[test]
    fn weight_norm_folds_to_g_times_unit_v() {
        // v=[3,4] → ||v||=5、g=10 → w=[6,8]。出力チャンネルごとに独立。
        let g = t(&[10.0, 1.0], &[2, 1, 1]);
        let v = t(&[3.0, 4.0, 0.0, 2.0], &[2, 1, 2]);
        assert_close(&fold_weight_norm(&g, &v).unwrap(), &t(&[6.0, 8.0, 0.0, 1.0], &[2, 1, 2]), 1e-6, 0.0).unwrap();
    }

    #[test]
    fn reflect_pad_right_mirrors_without_repeating_the_edge() {
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
        assert_eq!(reflect_pad_right(&x, 2).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
    }

    #[test]
    fn reflect_pad_longer_than_signal_is_an_error() {
        let x = t(&[1.0, 2.0], &[1, 1, 2]);
        assert!(reflect_pad_right(&x, 2).is_err());
        assert!(reflect_pad_right(&x, 0).is_ok());
    }

    #[test]
    fn conv_keeps_length_with_same_padding() {
        let w = Tensor::ones((1, 1, 7), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
        let c = Conv1d::new(w, None, 3, 1, 1);
        let y = c.forward(&Tensor::ones((1, 1, 20), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap()).unwrap();
        assert_eq!(y.dims(), &[1, 1, 20]);
    }

    #[test]
    fn transposed_conv_multiplies_length_by_stride() {
        // k=2*stride, padding=stride/2 で長さがちょうど stride 倍になる。
        let w = Tensor::ones((1, 1, 8), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
        let c = ConvT1d::new(w, None, 2, 4);
        let y = c.forward(&Tensor::ones((1, 1, 5), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap()).unwrap();
        assert_eq!(y.dims(), &[1, 1, 20]);
    }
}
