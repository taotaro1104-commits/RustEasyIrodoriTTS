//! DACVAE の基本層: Snake 活性、weight-norm 畳み込み、reflect パディング、残差ユニット。
//!
//! 参照: `DACVAE-selfcontained`（Apache-2.0）の `snake` / `NormConv1d` / `NormConvTranspose1d` /
//! `ResidualUnit`。weight-norm はロード時に `w = g * v / ||v||` へ畳み込む。

use candle_core::{DType, Tensor};
use rayon::prelude::*;

use crate::Result;

/// 重み取得関数（テンソル名 → テンソル）。不足時は `IrodoriError::MissingKey`。
pub type Fetch<'a> = &'a dyn Fn(&str) -> Result<Tensor>;

/// `x + sin²(αx) / (α + 1e-9)`。`x: (B, C, T)`, `alpha: (1, C, 1)`。
///
/// F32 の CPU では 1 パスのスレッド並列で計算する（汎用経路はテンソル演算を約 7 回に分けて毎回確保し、
/// 高解像度層で 1 要素 7ns かかっていた）。
pub fn snake(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    if let Some(y) = snake_fused_f32(x, alpha)? {
        return Ok(y);
    }
    let s = x.broadcast_mul(alpha)?.sin()?.sqr()?;
    Ok((x + s.broadcast_div(&(alpha + 1e-9)?)?)?)
}

fn snake_fused_f32(x: &Tensor, alpha: &Tensor) -> Result<Option<Tensor>> {
    let (Ok((b, c, l)), true) = (x.dims3(), x.dtype() == DType::F32 && alpha.dtype() == DType::F32 && x.device().is_cpu()) else {
        return Ok(None);
    };
    if alpha.elem_count() != c || l == 0 {
        return Ok(None);
    }
    let a = alpha.flatten_all()?.to_vec1::<f32>()?;
    let mut out = x.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    out.par_chunks_mut(l).enumerate().for_each(|(row, chunk)| {
        let al = a[row % c];
        let inv = 1.0 / (al + 1e-9);
        for v in chunk {
            let s = (al * *v).sin();
            *v += inv * s * s;
        }
    });
    Ok(Some(Tensor::from_vec(out, (b, c, l), x.device())?))
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

/// `forward_conv2d` を使い始める `c_in * len`（`Conv1d::prefers_conv2d`）。
const CONV2D_MIN_ELEMS: usize = 4_000_000;

pub struct Conv1d {
    weight: Tensor,
    /// k=1・stride 1・padding 0 の畳み込み用に `(c_out, c_in)` へ整形した重み（im2col を避けて行列積にする）。
    pointwise: Option<Tensor>,
    bias: Option<Tensor>,
    padding: usize,
    stride: usize,
    dilation: usize,
}

impl Conv1d {
    pub fn new(weight: Tensor, bias: Option<Tensor>, padding: usize, stride: usize, dilation: usize) -> Self {
        let pointwise = match weight.dims3() {
            Ok((c_out, c_in, 1)) if padding == 0 && stride == 1 => weight.reshape((c_out, c_in)).ok(),
            _ => None,
        };
        Self { weight, pointwise, bias, padding, stride, dilation }
    }

    /// `{prefix}.weight_g/.weight_v/.bias` を読み、weight-norm を畳み込んで `dtype` に変換する。
    pub fn load_wn(fetch: Fetch, prefix: &str, padding: usize, stride: usize, dilation: usize, dtype: DType) -> Result<Self> {
        let w = fold_weight_norm(&fetch(&format!("{prefix}.weight_g"))?, &fetch(&format!("{prefix}.weight_v"))?)?;
        let b = fetch(&format!("{prefix}.bias"))?;
        Ok(Self::new(w.to_dtype(dtype)?, Some(b.to_dtype(dtype)?), padding, stride, dilation))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(w) = &self.pointwise {
            return self.with_bias(w.broadcast_matmul(x)?);
        }
        let (_, c_in, len) = x.dims3()?;
        // conv2d の経路は F32 の大きい活性だけ（BF16 などは従来どおり im2col）。
        if x.dtype() == DType::F32 && Self::prefers_conv2d(c_in, len) {
            self.forward_conv2d(x)
        } else {
            self.forward_im2col(x)
        }
    }

    /// conv2d の経路（タイル化 im2col を rayon で並列）を使うほうが速い大きさか。
    /// 実測（Ryzen 5 3400G、8 スレッド）: `c_in * len` が約 900 万（長さ 48000 / 96000）の層は約 2 倍速く、
    /// 約 46 万〜230 万（長さ 600 / 6000）の層は im2col のほうが速い。
    pub(crate) fn prefers_conv2d(c_in: usize, len: usize) -> bool {
        c_in * len >= CONV2D_MIN_ELEMS
    }

    /// Candle の conv1d（全体を展開する im2col と行列積 1 回）。
    pub(crate) fn forward_im2col(&self, x: &Tensor) -> Result<Tensor> {
        self.with_bias(x.conv1d(&self.weight, self.padding, self.stride, self.dilation, 1)?)
    }

    /// `(b, c, len)` を `(b, c, 1, len)` として conv2d（タイル化 im2col）へ載せる。padding は先に自前で入れる
    /// （conv2d の padding は高さにも掛かるため）。出力は `forward_im2col` と同じ（浮動小数の丸め以外）。
    pub(crate) fn forward_conv2d(&self, x: &Tensor) -> Result<Tensor> {
        let padded;
        let x = if self.padding > 0 {
            padded = x.pad_with_zeros(2, self.padding, self.padding)?;
            &padded
        } else {
            x
        };
        let y = x
            .unsqueeze(2)?
            .conv2d(&self.weight.unsqueeze(2)?, 0, self.stride, self.dilation, 1)?
            .squeeze(2)?;
        self.with_bias(y)
    }

    fn with_bias(&self, y: Tensor) -> Result<Tensor> {
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
        // col2im の高速経路は、連続した重みでしか使われない。
        let weight = weight.contiguous().expect("a weight tensor can be made contiguous");
        Self { weight, bias, padding, stride }
    }

    pub fn load_wn(fetch: Fetch, prefix: &str, padding: usize, stride: usize, dtype: DType) -> Result<Self> {
        let w = fold_weight_norm(&fetch(&format!("{prefix}.weight_g"))?, &fetch(&format!("{prefix}.weight_v"))?)?;
        let b = fetch(&format!("{prefix}.bias"))?;
        Ok(Self::new(w.to_dtype(dtype)?, Some(b.to_dtype(dtype)?), padding, stride))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Candle 0.11.1 の col2im の経路（padding 0）はバッチが 2 以上だと結果が誤るので、1 件ずつ処理する。
        let batch = x.dim(0)?;
        let y = if batch == 1 {
            self.forward_one(x)?
        } else {
            let items = (0..batch)
                .map(|i| self.forward_one(&x.narrow(0, i, 1)?))
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&items, 0)?
        };
        match &self.bias {
            Some(b) => Ok(y.broadcast_add(&b.reshape((1, b.dim(0)?, 1))?)?),
            None => Ok(y),
        }
    }

    /// バッチ 1 の転置畳み込み（bias なし）。Candle は padding が 0 のときだけ高速な col2im の経路を通る。
    /// padding 0 で計算してから両端を `padding` ずつ切り落とすと、同じ出力が約 2 倍速く得られる（実測）。
    fn forward_one(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv_transpose1d(&self.weight, 0, 0, self.stride, 1, 1)?;
        if self.padding == 0 {
            return Ok(y);
        }
        let len = y.dim(2)?;
        Ok(y.narrow(2, self.padding, len - 2 * self.padding)?.contiguous()?)
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
    use candle_core::{Device, Tensor};

    #[test]
    fn snake_matches_formula() {
        // x + sin^2(alpha*x) / alpha  （alpha=2）
        let x = t(&[0.5, -1.0], &[1, 1, 2]);
        let alpha = t(&[2.0], &[1, 1, 1]);
        let want: Vec<f32> = [0.5f32, -1.0].iter().map(|v| v + (2.0 * v).sin().powi(2) / 2.0).collect();
        assert_close(&snake(&x, &alpha).unwrap(), &t(&want, &[1, 1, 2]), 1e-6, 0.0).unwrap();
    }

    #[test]
    fn snake_uses_a_separate_alpha_per_channel_for_every_batch_item() {
        // (B=2, C=3, T=4)。高速経路のチャンネル添字（行 % C）の取り違えを検出する。
        let x: Vec<f32> = (0..24).map(|i| (i as f32 - 11.0) * 0.17).collect();
        let alpha = [0.5f32, 1.5, 2.5];
        let got = snake(&t(&x, &[2, 3, 4]), &t(&alpha, &[1, 3, 1])).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, (g, v)) in got.iter().zip(&x).enumerate() {
            let a = alpha[(i / 4) % 3];
            let want = v + (a * v).sin().powi(2) / (a + 1e-9);
            assert!((g - want).abs() < 1e-6, "index {i}: {g} vs {want}");
        }
    }

    #[test]
    fn pointwise_conv_equals_the_plain_matrix_product() {
        // k=1, stride 1, padding 0 は im2col を通さず行列積にする。bias 付き・バッチ 2 で検算する。
        let (c_in, c_out, len) = (3usize, 2usize, 5usize);
        let w: Vec<f32> = (0..c_out * c_in).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let b = t(&[0.5, -0.5], &[2]);
        let x: Vec<f32> = (0..2 * c_in * len).map(|i| (i as f32 * 0.3).sin()).collect();
        let conv = Conv1d::new(t(&w, &[c_out, c_in, 1]), Some(b), 0, 1, 1);
        let y = conv.forward(&t(&x, &[2, c_in, len])).unwrap();
        assert_eq!(y.dims(), &[2, c_out, len]);
        let got = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for bi in 0..2 {
            for o in 0..c_out {
                for l in 0..len {
                    let acc: f32 = (0..c_in).map(|c| w[o * c_in + c] * x[bi * c_in * len + c * len + l]).sum();
                    let want = acc + if o == 0 { 0.5 } else { -0.5 };
                    let g = got[bi * c_out * len + o * len + l];
                    assert!((g - want).abs() < 1e-5, "b{bi} o{o} l{l}: {g} vs {want}");
                }
            }
        }
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

    /// 代表的な codec 層の速度を測る手動ベンチ（`cargo test --release -- --ignored --nocapture conv_speed`）。
    #[test]
    #[ignore]
    fn conv_speed_report() {
        use candle_core::{DType, Device};
        let dev = Device::Cpu;
        let time = |name: &str, flops: f64, f: &dyn Fn()| {
            f(); // ウォームアップ
            let t0 = std::time::Instant::now();
            let n = 3;
            for _ in 0..n {
                f();
            }
            let secs = t0.elapsed().as_secs_f64() / n as f64;
            eprintln!("{name:<46} {:>8.1} ms  {:>6.1} GFLOP/s", secs * 1e3, flops / secs / 1e9);
        };
        // (c, len, k, dilation): 実 decoder の RU の畳み込み
        for (c, len, k, dil) in [(768usize, 600usize, 7usize, 1usize), (384, 6000, 7, 3), (192, 48000, 7, 3), (96, 96000, 7, 3), (192, 48000, 1, 1)] {
            let x = Tensor::randn(0f32, 1f32, (1, c, len), &dev).unwrap();
            let w = Tensor::randn(0f32, 0.05f32, (c, c, k), &dev).unwrap();
            let conv = Conv1d::new(w, None, if k == 1 { 0 } else { 3 * dil }, 1, dil);
            let flops = 2.0 * (c * c * k * len) as f64;
            time(&format!("conv1d c={c} len={len} k={k} dil={dil}"), flops, &|| {
                conv.forward(&x).unwrap();
            });
        }
        // snake（要素ごとの演算）
        let x = Tensor::randn(0f32, 1f32, (1, 192, 48000), &dev).unwrap();
        let alpha = Tensor::ones((1, 192, 1), DType::F32, &dev).unwrap();
        time("snake c=192 len=48000", 0.0, &|| {
            snake(&x, &alpha).unwrap();
        });
        // 転置畳み込み: 実 decoder の up 層
        for (cin, cout, len, stride) in [(1536usize, 768usize, 50usize, 12usize), (768, 384, 600, 10), (384, 192, 6000, 8), (192, 96, 48000, 2)] {
            let x = Tensor::randn(0f32, 1f32, (1, cin, len), &dev).unwrap();
            let w = Tensor::randn(0f32, 0.05f32, (cin, cout, 2 * stride), &dev).unwrap();
            let ct = ConvT1d::new(w, None, stride.div_ceil(2), stride);
            let flops = 2.0 * (cin * cout * 2 * stride * len) as f64;
            time(&format!("convT cin={cin} cout={cout} len={len} s={stride}"), flops, &|| {
                ct.forward(&x).unwrap();
            });
        }
    }

    /// 転置畳み込みの参照実装（Candle の padding 付き経路そのまま）。
    fn convt_reference(x: &Tensor, w: &Tensor, b: Option<&Tensor>, padding: usize, stride: usize) -> Tensor {
        let y = x.conv_transpose1d(w, padding, 0, stride, 1, 1).unwrap();
        match b {
            Some(b) => y.broadcast_add(&b.reshape((1, b.dim(0).unwrap(), 1)).unwrap()).unwrap(),
            None => y,
        }
    }

    #[test]
    fn transposed_conv_matches_the_padded_reference_for_every_decoder_shape() {
        // 実 decoder の stride（12, 10, 8, 2）と、padding = ceil(stride / 2)。バッチ 2・bias あり／なし。
        let dev = Device::Cpu;
        for (cin, cout, len, stride) in [(6usize, 4usize, 5usize, 12usize), (5, 3, 7, 10), (4, 6, 9, 8), (3, 2, 11, 2)] {
            for with_bias in [true, false] {
                let x = Tensor::randn(0f32, 1f32, (2, cin, len), &dev).unwrap();
                let w = Tensor::randn(0f32, 0.3f32, (cin, cout, 2 * stride), &dev).unwrap();
                let b = Tensor::randn(0f32, 1f32, cout, &dev).unwrap();
                let padding = stride.div_ceil(2);
                let conv = ConvT1d::new(w.clone(), with_bias.then(|| b.clone()), padding, stride);
                let got = conv.forward(&x).unwrap();
                let want = convt_reference(&x, &w, with_bias.then_some(&b), padding, stride);
                assert_eq!(got.dims(), want.dims(), "s={stride}");
                crate::golden::assert_close(&got, &want, 1e-4, 1e-5).unwrap();
            }
        }
    }

    #[test]
    fn transposed_conv_is_correct_for_batches_larger_than_one() {
        // Candle の col2im の経路はバッチ 2 以上で誤る（0.11.1）。バッチごとに素朴な実装と照合する。
        let dev = Device::Cpu;
        let (cin, cout, len, stride, pad) = (4usize, 3usize, 5usize, 2usize, 1usize);
        let k = 2 * stride;
        let x = Tensor::randn(0f32, 1f32, (3, cin, len), &dev).unwrap();
        let w = Tensor::randn(0f32, 0.3f32, (cin, cout, k), &dev).unwrap();
        let xv = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let wv = w.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let full = (len - 1) * stride + k;
        let mut naive = vec![0f32; 3 * cout * full];
        for b in 0..3 {
            for ci in 0..cin {
                for l in 0..len {
                    for co in 0..cout {
                        for kk in 0..k {
                            naive[b * cout * full + co * full + l * stride + kk] += xv[b * cin * len + ci * len + l] * wv[ci * cout * k + co * k + kk];
                        }
                    }
                }
            }
        }
        let want = Tensor::from_vec(naive, (3, cout, full), &dev).unwrap().narrow(2, pad, full - 2 * pad).unwrap();
        let got = ConvT1d::new(w, None, pad, stride).forward(&x).unwrap();
        crate::golden::assert_close(&got, &want, 1e-5, 1e-5).unwrap();
    }

    #[test]
    fn transposed_conv_without_padding_is_unchanged() {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1f32, (1, 4, 6), &dev).unwrap();
        let w = Tensor::randn(0f32, 0.3f32, (4, 3, 4), &dev).unwrap();
        let got = ConvT1d::new(w.clone(), None, 0, 2).forward(&x).unwrap();
        crate::golden::assert_close(&got, &convt_reference(&x, &w, None, 0, 2), 1e-5, 0.0).unwrap();
    }

    #[test]
    fn conv2d_path_matches_the_im2col_path() {
        // 実 decoder・encoder の形（k=7、dilation 1/3、stride 1/2）と、小さな形。
        let dev = Device::Cpu;
        for (c_in, c_out, len, k, dil, stride) in [(8usize, 8usize, 40usize, 7usize, 1usize, 1usize), (6, 6, 64, 7, 3, 1), (4, 8, 50, 7, 1, 2), (5, 5, 33, 3, 2, 1)] {
            for with_bias in [true, false] {
                let x = Tensor::randn(0f32, 1f32, (2, c_in, len), &dev).unwrap();
                let w = Tensor::randn(0f32, 0.2f32, (c_out, c_in, k), &dev).unwrap();
                let b = Tensor::randn(0f32, 1f32, c_out, &dev).unwrap();
                let padding = dil * (k - 1) / 2;
                let conv = Conv1d::new(w, with_bias.then_some(b), padding, stride, dil);
                let want = conv.forward_im2col(&x).unwrap();
                let got = conv.forward_conv2d(&x).unwrap();
                assert_eq!(got.dims(), want.dims(), "c={c_in} k={k} dil={dil} s={stride}");
                crate::golden::assert_close(&got, &want, 1e-4, 1e-5).unwrap();
            }
        }
    }

    #[test]
    fn conv2d_path_is_chosen_only_for_large_activations() {
        // 実測: 長さ 48000 / 96000 の層は conv2d が約 2 倍速く、長さ 600 / 6000 の層は im2col が速い。
        assert!(Conv1d::prefers_conv2d(192, 48000));
        assert!(Conv1d::prefers_conv2d(96, 96000));
        assert!(!Conv1d::prefers_conv2d(768, 600));
        assert!(!Conv1d::prefers_conv2d(384, 6000));
    }

    #[test]
    fn forward_picks_the_same_result_on_either_side_of_the_threshold() {
        let dev = Device::Cpu;
        // 閾値の上（c * len が大きい）でも下でも、出力は im2col と一致する。
        for (c, len) in [(4usize, 64usize), (16, 300_000)] {
            let x = Tensor::randn(0f32, 1f32, (1, c, len), &dev).unwrap();
            let w = Tensor::randn(0f32, 0.1f32, (c, c, 7), &dev).unwrap();
            let conv = Conv1d::new(w, None, 9, 1, 3);
            let want = conv.forward_im2col(&x).unwrap();
            crate::golden::assert_close(&conv.forward(&x).unwrap(), &want, 1e-4, 1e-5).unwrap();
        }
    }
}
