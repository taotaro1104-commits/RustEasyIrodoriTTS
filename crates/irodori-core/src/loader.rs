//! safetensors の遅延ロード（mmap）とメタデータ読み出し。
//!
//! ヘッダ（先頭 8 バイトの長さ＋JSON）だけを読んで設定とテンソル名を得る。
//! 重み本体は [`Weights::var_builder`] が mmap し、`get` 時に必要な dtype へ変換する。

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::config::{ModelConfig, TextEncoderConfig};
use crate::{IrodoriError, Result};

/// ヘッダ長の上限（壊れたファイルで巨大確保をしない）。
const MAX_HEADER_BYTES: u64 = 100 * 1024 * 1024;
const METADATA_KEY: &str = "__metadata__";

pub struct Weights {
    path: PathBuf,
    device: Device,
    metadata: HashMap<String, String>,
    names: Vec<String>,
}

fn read_header(path: &Path) -> Result<(HashMap<String, String>, Vec<String>)> {
    let bad = |msg: String| IrodoriError::InvalidFile(format!("{}: {msg}", path.display()));
    let mut file = std::fs::File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .map_err(|_| bad("file shorter than 8 bytes".into()))?;
    let n = u64::from_le_bytes(len_buf);
    if n > MAX_HEADER_BYTES || 8 + n > file_len {
        return Err(bad(format!("header length {n} is invalid (file size {file_len})")));
    }
    let mut buf = vec![0u8; n as usize];
    file.read_exact(&mut buf)?;
    let root: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&buf).map_err(|e| bad(format!("header is not JSON: {e}")))?;
    let metadata = root
        .get(METADATA_KEY)
        .and_then(|v| v.as_object())
        .map(|o| {
            o.iter()
                .map(|(k, v)| (k.clone(), v.as_str().map_or_else(|| v.to_string(), String::from)))
                .collect()
        })
        .unwrap_or_default();
    let mut names: Vec<String> = root.keys().filter(|k| *k != METADATA_KEY).cloned().collect();
    names.sort();
    Ok((metadata, names))
}

impl Weights {
    pub fn open(path: &Path, device: &Device) -> Result<Self> {
        let (metadata, names) = read_header(path)?;
        Ok(Self { path: path.to_path_buf(), device: device.clone(), metadata, names })
    }

    fn metadata_entry(&self, key: &str) -> Result<&str> {
        self.metadata
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| IrodoriError::MissingMetadata(key.to_string()))
    }

    /// メタデータから DiT 設定とテキストエンコーダ設定を読む。
    pub fn config(&self) -> Result<(ModelConfig, TextEncoderConfig)> {
        let model = ModelConfig::from_json(self.metadata_entry("config_json")?)?;
        let text = TextEncoderConfig::from_json(self.metadata_entry("text_encoder_config_json")?)?;
        text.validate()?;
        Ok((model, text))
    }

    /// 要求 dtype へ読み込み時に変換する VarBuilder（mmap）。
    pub fn var_builder(&self, dtype: DType) -> Result<VarBuilder<'static>> {
        // SAFETY: 読み取り専用 mmap。実行中にファイルを書き換えない前提（モデルファイルは不変）。
        Ok(unsafe { VarBuilder::from_mmaped_safetensors(&[&self.path], dtype, &self.device)? })
    }

    pub fn tensor_names(&self) -> Vec<String> {
        self.names.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use std::collections::HashMap;

    const META: &str = include_str!("../tests/fixtures/v4_1_small_meta.json");

    fn write_file(dir: &std::path::Path, meta: Option<HashMap<String, String>>) -> std::path::PathBuf {
        let path = dir.join("m.safetensors");
        let w = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu).unwrap();
        safetensors::serialize_to_file([("block.w", &w)], meta, &path).unwrap();
        path
    }

    fn real_meta() -> HashMap<String, String> {
        serde_json::from_str(META).unwrap()
    }

    #[test]
    fn config_reads_both_metadata_entries() {
        let dir = tempfile::tempdir().unwrap();
        let w = Weights::open(&write_file(dir.path(), Some(real_meta())), &Device::Cpu).unwrap();
        let (m, t) = w.config().unwrap();
        assert_eq!((m.model_dim, m.num_layers), (1280, 12));
        assert_eq!(t.hidden_size, 768);
    }

    #[test]
    fn missing_metadata_names_the_missing_key() {
        let dir = tempfile::tempdir().unwrap();
        let w = Weights::open(&write_file(dir.path(), None), &Device::Cpu).unwrap();
        let err = w.config().unwrap_err();
        assert!(matches!(err, crate::IrodoriError::MissingMetadata(ref k) if k == "config_json"), "{err}");
    }

    #[test]
    fn missing_tensor_error_contains_the_key_name() {
        let dir = tempfile::tempdir().unwrap();
        let w = Weights::open(&write_file(dir.path(), None), &Device::Cpu).unwrap();
        let vb = w.var_builder(DType::F32).unwrap();
        let err = vb.pp("block").get((2, 2), "nope").unwrap_err().to_string();
        assert!(err.contains("block.nope"), "{err}");
    }

    #[test]
    fn var_builder_casts_dtype_on_load() {
        let dir = tempfile::tempdir().unwrap();
        let w = Weights::open(&write_file(dir.path(), None), &Device::Cpu).unwrap();
        let t = w.var_builder(DType::BF16).unwrap().pp("block").get((2, 2), "w").unwrap();
        assert_eq!(t.dtype(), DType::BF16);
        assert_eq!(t.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap(), vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[test]
    fn tensor_names_lists_header_keys_without_loading() {
        let dir = tempfile::tempdir().unwrap();
        let w = Weights::open(&write_file(dir.path(), Some(real_meta())), &Device::Cpu).unwrap();
        assert_eq!(w.tensor_names(), vec!["block.w".to_string()]);
    }

    #[test]
    fn corrupt_file_is_an_error_not_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("bad.safetensors");
        std::fs::write(&p, b"\xff\xff\xff\xff\xff\xff\xff\x7fgarbage").unwrap();
        assert!(Weights::open(&p, &Device::Cpu).is_err());
        std::fs::write(&p, b"abc").unwrap();
        assert!(Weights::open(&p, &Device::Cpu).is_err());
    }
}
