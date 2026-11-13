//! マスク付き scaled dot-product attention（CPU 経路）。
//!
//! 上流 `attention.py` の `masked_sdpa` / `prefix_key_mask_attention` の SDPA 経路に相当する。
//! テンソルは `(B, S, H, D)` レイアウト。キーマスクは `(B, S_kv)`（1=有効）。

use candle_core::{DType, Tensor, D};

use crate::{IrodoriError, Result};

/// 有効キーが 1 つも無い行は位置 0 を有効にして有限値を保つ（上流の `safe_mask` と同じ）。
/// 戻り値は softmax 前に足す加算バイアス `(B, 1, 1, S_kv)`（有効=0, 無効=-1e30）。
fn key_bias(mask: &Tensor, batch: usize, kv_len: usize, dtype: DType) -> Result<Tensor> {
    if mask.dims() != [batch, kv_len] {
        return Err(IrodoriError::InvalidInput(format!(
            "key mask must have shape ({batch}, {kv_len}), got {:?}",
            mask.dims()
        )));
    }
    let m = mask.to_dtype(DType::F32)?.gt(0.0f32)?.to_dtype(DType::F32)?;
    let empty = m.sum_keepdim(1)?.eq(0.0f32)?.to_dtype(DType::F32)?; // (B, 1)
    let mut first = vec![0f32; kv_len];
    first[0] = 1.0;
    let first = Tensor::from_vec(first, (1, kv_len), m.device())?;
    let safe = (m + empty.broadcast_mul(&first)?)?;
    Ok(safe.affine(1e30, -1e30)?.reshape((batch, 1, 1, kv_len))?.to_dtype(dtype)?)
}

/// `q: (B, S_q, H, D)`, `k, v: (B, S_kv, H, D)` → `(B, S_q, H, D)`。
pub fn masked_sdpa(q: &Tensor, k: &Tensor, v: &Tensor, key_mask: Option<&Tensor>) -> Result<Tensor> {
    let (b, _sq, _h, d) = q.dims4()?;
    let kv_len = k.dim(1)?;
    let bias = key_mask.map(|m| key_bias(m, b, kv_len, q.dtype())).transpose()?;
    let q = q.transpose(1, 2)?.contiguous()?; // (B, H, S_q, D)
    let kt = k.transpose(1, 2)?.transpose(2, 3)?.contiguous()?; // (B, H, D, S_kv)
    let v = v.transpose(1, 2)?.contiguous()?; // (B, H, S_kv, D)
    let mut scores = (q.matmul(&kt)? * (1.0 / (d as f64).sqrt()))?;
    if let Some(bias) = bias {
        scores = scores.broadcast_add(&bias)?;
    }
    let probs = candle_nn::ops::softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(v.dtype())?;
    Ok(probs.matmul(&v)?.transpose(1, 2)?.contiguous()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::golden::assert_close;
    use crate::nn::test_util::t;
    use candle_core::{Device, Tensor};

    fn mask(v: &[u8], shape: &[usize]) -> Tensor {
        Tensor::from_slice(v, shape, &Device::Cpu).unwrap()
    }

    #[test]
    fn matches_manual_softmax_for_one_head() {
        // (B=1, S=2, H=1, D=2)。query 0 は key0 と一致、key1 と直交。
        let q = t(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 1, 2]);
        let k = t(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 1, 2]);
        let v = t(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 1, 2]);
        let y = masked_sdpa(&q, &k, &v, None).unwrap();
        let s = 1.0 / 2f32.sqrt();
        let w0 = s.exp() / (s.exp() + 1.0);
        let w1 = 1.0 - w0;
        let row0 = [w0 * 1.0 + w1 * 3.0, w0 * 2.0 + w1 * 4.0];
        let got = y.narrow(1, 0, 1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((got[0] - row0[0]).abs() < 1e-6 && (got[1] - row0[1]).abs() < 1e-6, "{got:?}");
    }

    #[test]
    fn masked_keys_do_not_influence_output() {
        let q = t(&[0.3, -0.2, 0.5, 0.1], &[1, 2, 1, 2]);
        let k1 = t(&[1.0, 0.0, 9.0, 9.0], &[1, 2, 1, 2]);
        let k2 = t(&[1.0, 0.0, -5.0, 3.0], &[1, 2, 1, 2]);
        let v1 = t(&[1.0, 2.0, 100.0, 100.0], &[1, 2, 1, 2]);
        let v2 = t(&[1.0, 2.0, -7.0, 8.0], &[1, 2, 1, 2]);
        let m = mask(&[1, 0], &[1, 2]);
        let a = masked_sdpa(&q, &k1, &v1, Some(&m)).unwrap();
        let b = masked_sdpa(&q, &k2, &v2, Some(&m)).unwrap();
        assert_close(&a, &b, 1e-6, 0.0).unwrap();
        // 有効キーが 1 つなので出力は v[0] そのもの。
        assert_close(&a.narrow(1, 0, 1).unwrap().flatten_all().unwrap(), &t(&[1.0, 2.0], &[2]), 1e-6, 0.0).unwrap();
    }

    #[test]
    fn fully_masked_row_stays_finite() {
        let q = t(&[0.3, -0.2], &[1, 1, 1, 2]);
        let k = t(&[1.0, 0.0], &[1, 1, 1, 2]);
        let v = t(&[1.0, 2.0], &[1, 1, 1, 2]);
        let y = masked_sdpa(&q, &k, &v, Some(&mask(&[0], &[1, 1]))).unwrap();
        assert!(y.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().all(|x| x.is_finite()));
    }

    #[test]
    fn mask_shape_mismatch_is_an_error() {
        let q = t(&[0.0; 4], &[1, 2, 1, 2]);
        assert!(masked_sdpa(&q, &q, &q, Some(&mask(&[1, 1, 1], &[1, 3]))).is_err());
    }
}
