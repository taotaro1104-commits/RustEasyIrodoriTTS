//! K-quants（Q6_K / Q5_K / Q4_K）で量子化した線形層。
//!
//! 実測（Ryzen 5 3400G、AVX2、`m=150` 行）: Candle の K-quants の行列積は f32 の行列積と同じかやや速く
//! （0.8〜1.1 倍の時間）、重みは Q6_K で約 1/5、Q4_K で約 1/7 になる。Q8_0 / Q5_0 / Q4_0 は 4〜17 倍遅いので使わない。
//! K-quants は入力次元が 256 の倍数であることが必要なので、足りない分は重みの列と入力を 0 で詰める
//! （0 を掛けるだけなので結果は変わらない）。

use std::borrow::Cow;

use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Linear, VarBuilder};
use rayon::prelude::*;

/// K-quants のブロック長（入力次元をこの倍数にそろえる）。
const BLOCK: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantKind {
    Q6K,
    Q5K,
    Q4K,
}

impl QuantKind {
    pub fn dtype(self) -> GgmlDType {
        match self {
            QuantKind::Q6K => GgmlDType::Q6K,
            QuantKind::Q5K => GgmlDType::Q5K,
            QuantKind::Q4K => GgmlDType::Q4K,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            QuantKind::Q6K => "q6k",
            QuantKind::Q5K => "q5k",
            QuantKind::Q4K => "q4k",
        }
    }
}

impl std::str::FromStr for QuantKind {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "q6k" | "q6_k" => Ok(QuantKind::Q6K),
            "q5k" | "q5_k" => Ok(QuantKind::Q5K),
            "q4k" | "q4_k" => Ok(QuantKind::Q4K),
            other => Err(format!("unknown quantization {other:?} (use q6k, q5k or q4k)")),
        }
    }
}

/// `(rows, cols)` の重みを行ごとに並列に量子化する。`cols` は 256 の倍数。
/// （`QTensor::quantize` は 1 スレッドで約 24MB/s と遅い。K-quants は行ごとに独立なので、行を分けても同じバイト列になる。）
pub fn quantize_rows(w: &Tensor, kind: QuantKind) -> Result<QTensor> {
    let (rows, cols) = w.dims2()?;
    if cols % BLOCK != 0 {
        candle_core::bail!("the input dimension {cols} must be a multiple of {BLOCK} to quantize");
    }
    // `QTensor::quantize` はテンソルの開始位置を見ずにストレージを直接読む（`narrow` の結果を `copy` しても別の行を
    // 量子化する）ので、値を `Vec` に取り出して行ごとのテンソルを作り直して渡す。
    let values = w.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let chunk = rows.div_ceil(rayon::current_num_threads().max(1)).max(1);
    let starts: Vec<usize> = (0..rows).step_by(chunk).collect();
    let parts: Vec<Vec<u8>> = starts
        .par_iter()
        .map(|&r0| {
            let len = chunk.min(rows - r0);
            let part = Tensor::from_slice(&values[r0 * cols..(r0 + len) * cols], (len, cols), &Device::Cpu)?;
            Ok(QTensor::quantize(&part, kind.dtype())?.data()?.into_owned())
        })
        .collect::<Result<_>>()?;
    let storage = QStorage::from_data(Cow::Owned(parts.concat()), &Device::Cpu, kind.dtype())?;
    QTensor::new(storage, (rows, cols))
}

/// 通常の線形層か、量子化した線形層。
#[derive(Clone, Debug)]
pub enum QLinear {
    Dense(Linear),
    Quant {
        matmul: QMatMul,
        bias: Option<Tensor>,
        /// 入力の末尾に足す 0 の数（入力次元を 256 の倍数にそろえるため）。
        pad: usize,
    },
}

impl QLinear {
    /// `vb` の `weight`（と `bias`）を読む。`quant` があれば量子化する（重みは f32 に直して量子化する）。
    pub fn load(in_dim: usize, out_dim: usize, with_bias: bool, vb: VarBuilder, quant: Option<QuantKind>) -> Result<Self> {
        let Some(kind) = quant else {
            let linear = if with_bias { candle_nn::linear(in_dim, out_dim, vb)? } else { candle_nn::linear_no_bias(in_dim, out_dim, vb)? };
            return Ok(QLinear::Dense(linear));
        };
        let weight = vb.get((out_dim, in_dim), "weight")?.to_dtype(DType::F32)?;
        let pad = (BLOCK - in_dim % BLOCK) % BLOCK;
        let weight = if pad > 0 { weight.pad_with_zeros(1, 0, pad)? } else { weight };
        let bias = if with_bias { Some(vb.get(out_dim, "bias")?.to_dtype(DType::F32)?) } else { None };
        Ok(QLinear::Quant { matmul: QMatMul::from_qtensor(quantize_rows(&weight, kind)?)?, bias, pad })
    }
}

impl Module for QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            QLinear::Dense(linear) => linear.forward(x),
            QLinear::Quant { matmul, bias, pad } => {
                let x = if *pad > 0 { x.pad_with_zeros(D::Minus1, 0, *pad)? } else { x.contiguous()? };
                let y = matmul.forward(&x)?;
                match bias {
                    Some(b) => y.broadcast_add(b),
                    None => Ok(y),
                }
            }
        }
    }
}

pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder, quant: Option<QuantKind>) -> Result<QLinear> {
    QLinear::load(in_dim, out_dim, false, vb, quant)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::test_util::vb;

    fn rel_err(a: &Tensor, b: &Tensor) -> f32 {
        let d = (a - b).unwrap().sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
        d / b.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt()
    }

    fn weights(out: usize, inp: usize, bias: bool) -> VarBuilder<'static> {
        let dev = Device::Cpu;
        let mut items = vec![("w.weight", Tensor::randn(0f32, 0.05f32, (out, inp), &dev).unwrap())];
        if bias {
            items.push(("w.bias", Tensor::randn(0f32, 1f32, out, &dev).unwrap()));
        }
        vb(items)
    }

    fn dense_and_quant(out: usize, inp: usize, bias: bool, kind: QuantKind) -> (QLinear, QLinear) {
        let v = weights(out, inp, bias);
        (QLinear::load(inp, out, bias, v.pp("w"), None).unwrap(), QLinear::load(inp, out, bias, v.pp("w"), Some(kind)).unwrap())
    }

    #[test]
    fn parallel_quantization_is_byte_identical_to_the_single_threaded_one() {
        let w = Tensor::randn(0f32, 0.05f32, (37usize, 512usize), &Device::Cpu).unwrap();
        for kind in [QuantKind::Q6K, QuantKind::Q5K, QuantKind::Q4K] {
            let single = QTensor::quantize(&w, kind.dtype()).unwrap();
            let parallel = quantize_rows(&w, kind).unwrap();
            let (a, b) = (single.data().unwrap(), parallel.data().unwrap());
            let first = a.iter().zip(b.iter()).position(|(x, y)| x != y);
            assert_eq!((a.len(), first), (b.len(), None), "{kind:?}: lengths {} vs {}, first difference at {first:?}", a.len(), b.len());
            assert_eq!(parallel.shape().dims(), &[37, 512]);
        }
    }

    #[test]
    fn quantization_requires_a_block_multiple_input_dimension() {
        let w = Tensor::zeros((4usize, 300usize), DType::F32, &Device::Cpu).unwrap();
        assert!(quantize_rows(&w, QuantKind::Q6K).is_err());
    }

    #[test]
    fn quantized_output_stays_close_to_the_dense_output() {
        let x = Tensor::randn(0f32, 1f32, (2usize, 5usize, 512usize), &Device::Cpu).unwrap();
        for (kind, limit) in [(QuantKind::Q6K, 0.03f32), (QuantKind::Q5K, 0.06), (QuantKind::Q4K, 0.10)] {
            let (dense, quant) = dense_and_quant(64, 512, false, kind);
            let (want, got) = (dense.forward(&x).unwrap(), quant.forward(&x).unwrap());
            assert_eq!(got.dims(), want.dims());
            let e = rel_err(&got, &want);
            assert!(e < limit, "{kind:?}: relative error {e}");
        }
    }

    #[test]
    fn an_input_dimension_that_is_not_a_block_multiple_is_padded_with_zeros() {
        // 実モデルの mlp.w2 は入力次元が 3680（256 の倍数でない）。
        let x = Tensor::randn(0f32, 1f32, (1usize, 3usize, 300usize), &Device::Cpu).unwrap();
        let (dense, quant) = dense_and_quant(48, 300, false, QuantKind::Q6K);
        let (want, got) = (dense.forward(&x).unwrap(), quant.forward(&x).unwrap());
        assert_eq!(got.dims(), want.dims());
        assert!(rel_err(&got, &want) < 0.03);
    }

    #[test]
    fn the_bias_is_added_after_the_matrix_product() {
        let x = Tensor::randn(0f32, 1f32, (1usize, 4usize, 256usize), &Device::Cpu).unwrap();
        let (dense, quant) = dense_and_quant(32, 256, true, QuantKind::Q6K);
        assert!(rel_err(&quant.forward(&x).unwrap(), &dense.forward(&x).unwrap()) < 0.03);
    }

    #[test]
    fn without_a_quantization_the_layer_is_the_dense_one() {
        let v = weights(8, 16, false);
        let layer = QLinear::load(16, 8, false, v.pp("w"), None).unwrap();
        assert!(matches!(layer, QLinear::Dense(_)));
    }

    #[test]
    fn quantization_names_round_trip() {
        for kind in [QuantKind::Q6K, QuantKind::Q5K, QuantKind::Q4K] {
            assert_eq!(kind.name().parse::<QuantKind>().unwrap(), kind);
        }
        assert!("q8_0".parse::<QuantKind>().is_err());
    }
}
