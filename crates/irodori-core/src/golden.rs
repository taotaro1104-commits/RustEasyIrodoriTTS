//! Python 版が出力したゴールデンテンソルとの比較ユーティリティ。
//!
//! ゴールデンは `$IRODORI_GOLDEN_DIR/{name}.safetensors`。環境変数が無ければ
//! [`dir`] が `None` を返し、呼び出し側のテストは skip する。

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};

use crate::{IrodoriError, Result};

/// ゴールデンテンソルのディレクトリ（`IRODORI_GOLDEN_DIR`）。未設定なら `None`。
pub fn dir() -> Option<PathBuf> {
    std::env::var_os("IRODORI_GOLDEN_DIR")
        .map(PathBuf::from)
        .filter(|p| p.is_dir())
}

/// `{dir}/{name}.safetensors` を CPU に読み込む。
pub fn load(name: &str) -> Result<HashMap<String, Tensor>> {
    let path = dir()
        .ok_or_else(|| IrodoriError::GoldenMissing("IRODORI_GOLDEN_DIR".into()))?
        .join(format!("{name}.safetensors"));
    if !path.is_file() {
        return Err(IrodoriError::GoldenMissing(path.display().to_string()));
    }
    Ok(candle_core::safetensors::load(&path, &Device::Cpu)?)
}

/// `max|a-b| <= atol + rtol*|b|` を全要素で検査する。shape 不一致は Err。
pub fn assert_close(a: &Tensor, b: &Tensor, atol: f32, rtol: f32) -> Result<()> {
    if a.dims() != b.dims() {
        return Err(IrodoriError::GoldenMismatch(format!(
            "shape {:?} != {:?}",
            a.dims(),
            b.dims()
        )));
    }
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut max_diff = 0f32;
    let mut worst = (0usize, 0f32, 0f32);
    let mut ok = true;
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        let d = (x - y).abs();
        if d.is_nan() || d > atol + rtol * y.abs() {
            ok = false;
        }
        if d.is_nan() || d > max_diff {
            max_diff = if d.is_nan() { f32::NAN } else { d };
            worst = (i, *x, *y);
        }
    }
    if ok {
        Ok(())
    } else {
        Err(IrodoriError::GoldenMismatch(format!(
            "max abs diff {max_diff:e} at index {} (got {}, want {}), atol={atol:e} rtol={rtol:e}",
            worst.0, worst.1, worst.2
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn t(v: &[f32]) -> Tensor {
        Tensor::from_slice(v, v.len(), &Device::Cpu).unwrap()
    }

    #[test]
    fn assert_close_passes_within_tolerance() {
        assert_close(&t(&[1.0, 2.0]), &t(&[1.0, 2.0001]), 1e-3, 0.0).unwrap();
    }

    #[test]
    fn assert_close_fails_on_mismatch() {
        let err = assert_close(&t(&[1.0, 2.0]), &t(&[1.0, 2.1]), 1e-3, 0.0).unwrap_err();
        assert!(err.to_string().contains("max abs diff"), "{err}");
    }

    #[test]
    fn assert_close_fails_on_shape_mismatch() {
        assert!(assert_close(&t(&[1.0]), &t(&[1.0, 2.0]), 1e-3, 0.0).is_err());
    }
}
