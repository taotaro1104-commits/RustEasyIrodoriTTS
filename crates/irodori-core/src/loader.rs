//! safetensors の遅延ロードとメタデータ読み出し。
//!
//! ヘッダ（先頭 8 バイトの長さ＋JSON）だけを読んで設定とテンソル名を得る。重み本体は
//! [`Weights::var_builder`] が **mmap を使わず**、要求されたテンソルのバイト列だけをファイルから読む。
//! mmap だとページがそのままプロセスのワーキングセットに乗り、変換後のコピーと二重に計上されて
//! ピークメモリが約 2 倍になるため。読み込み後はバッファを即座に手放す。

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Shape};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::VarBuilder;

use crate::config::{ModelConfig, TextEncoderConfig};
use crate::{IrodoriError, Result};

/// ヘッダ長の上限（壊れたファイルで巨大確保をしない）。
const MAX_HEADER_BYTES: u64 = 100 * 1024 * 1024;
const METADATA_KEY: &str = "__metadata__";

/// ヘッダに書かれた 1 テンソルの位置情報。`start`/`end` はデータ領域先頭からのバイトオフセット。
#[derive(Clone, Debug)]
struct TensorMeta {
    dtype: DType,
    shape: Vec<usize>,
    start: u64,
    end: u64,
}

struct Header {
    metadata: HashMap<String, String>,
    tensors: HashMap<String, TensorMeta>,
    data_start: u64,
}

pub struct Weights {
    path: PathBuf,
    device: Device,
    header: Header,
}

fn parse_dtype(s: &str) -> Option<DType> {
    Some(match s {
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        "F64" => DType::F64,
        "U8" => DType::U8,
        "U32" => DType::U32,
        "I64" => DType::I64,
        _ => return None,
    })
}

fn read_header(path: &Path) -> Result<Header> {
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
    let data_start = 8 + n;
    let mut tensors = HashMap::new();
    for (name, v) in root.iter().filter(|(k, _)| *k != METADATA_KEY) {
        let field = |key: &str| v.get(key).ok_or_else(|| bad(format!("tensor {name} has no {key}")));
        let dtype_str = field("dtype")?.as_str().unwrap_or_default().to_string();
        let dtype = parse_dtype(&dtype_str).ok_or_else(|| bad(format!("tensor {name} has unsupported dtype {dtype_str}")))?;
        let shape: Vec<usize> = field("shape")?
            .as_array()
            .and_then(|a| a.iter().map(|x| x.as_u64().and_then(|x| usize::try_from(x).ok())).collect())
            .ok_or_else(|| bad(format!("tensor {name} has an invalid shape")))?;
        let off = field("data_offsets")?.as_array().cloned().unwrap_or_default();
        let (start, end) = match (off.first().and_then(|x| x.as_u64()), off.get(1).and_then(|x| x.as_u64())) {
            (Some(s), Some(e)) if s <= e => (s, e),
            _ => return Err(bad(format!("tensor {name} has invalid data_offsets"))),
        };
        if data_start.checked_add(end).map_or(true, |e| e > file_len) {
            return Err(bad(format!("tensor {name} extends past the end of the file (truncated?)")));
        }
        let want = shape
            .iter()
            .try_fold(dtype.size_in_bytes() as u64, |acc, &d| acc.checked_mul(d as u64))
            .ok_or_else(|| bad(format!("tensor {name} has an overflowing shape")))?;
        if end - start != want {
            return Err(bad(format!("tensor {name}: {} bytes in header range, expected {want}", end - start)));
        }
        tensors.insert(name.clone(), TensorMeta { dtype, shape, start, end });
    }
    Ok(Header { metadata, tensors, data_start })
}

/// ファイルから要求されたテンソルだけを読む VarBuilder 用バックエンド。
struct FileBackend {
    file: Mutex<std::fs::File>,
    data_start: u64,
    tensors: Arc<HashMap<String, TensorMeta>>,
}

impl FileBackend {
    fn read_tensor(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<candle_core::Tensor> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| candle_core::Error::CannotFindTensor { path: name.to_string() }.bt())?;
        let mut file = self.file.lock().unwrap_or_else(|e| e.into_inner());
        file.seek(SeekFrom::Start(self.data_start + meta.start)).map_err(candle_core::Error::wrap)?;
        let len = (meta.end - meta.start) as usize;
        let tensor = if meta.dtype == DType::F32 {
            // 型付きバッファへ直接読み、バイト列との二重確保を避ける（モデルの重みは F32）。
            let mut v = vec![0f32; len / 4];
            file.read_exact(bytemuck::cast_slice_mut(&mut v)).map_err(candle_core::Error::wrap)?;
            drop(file);
            candle_core::Tensor::from_vec(v, meta.shape.clone(), dev)?
        } else {
            let mut bytes = vec![0u8; len];
            file.read_exact(&mut bytes).map_err(candle_core::Error::wrap)?;
            drop(file);
            candle_core::Tensor::from_raw_buffer(&bytes, meta.dtype, &meta.shape, dev)?
        };
        tensor.to_dtype(dtype)
    }
}

impl SimpleBackend for FileBackend {
    fn get(&self, s: Shape, name: &str, _h: candle_nn::Init, dtype: DType, dev: &Device) -> candle_core::Result<candle_core::Tensor> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| candle_core::Error::CannotFindTensor { path: name.to_string() }.bt())?;
        if meta.shape != s.dims() {
            return Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: Shape::from(meta.shape.clone()),
            }
            .bt());
        }
        self.read_tensor(name, dtype, dev)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<candle_core::Tensor> {
        self.read_tensor(name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

impl Weights {
    pub fn open(path: &Path, device: &Device) -> Result<Self> {
        Ok(Self { path: path.to_path_buf(), device: device.clone(), header: read_header(path)? })
    }

    fn metadata_entry(&self, key: &str) -> Result<&str> {
        self.header
            .metadata
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

    /// 要求 dtype へ読み込み時に変換する VarBuilder。テンソルは `get` のたびにファイルから読む
    /// （キャッシュしない。使い終えた部品を落とせばメモリはすぐ戻る）。
    pub fn var_builder(&self, dtype: DType) -> Result<VarBuilder<'static>> {
        let backend = FileBackend {
            file: Mutex::new(std::fs::File::open(&self.path)?),
            data_start: self.header.data_start,
            tensors: Arc::new(self.header.tensors.clone()),
        };
        Ok(VarBuilder::from_backend(Box::new(backend), dtype, self.device.clone()))
    }

    pub fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.header.tensors.keys().cloned().collect();
        names.sort();
        names
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

    #[test]
    fn truncated_data_is_an_error_not_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        let p = write_file(dir.path(), None);
        let bytes = std::fs::read(&p).unwrap();
        std::fs::write(&p, &bytes[..bytes.len() - 4]).unwrap();
        let err = Weights::open(&p, &Device::Cpu).err().expect("must fail at open").to_string();
        assert!(err.contains("truncated") || err.contains("past the end"), "{err}");
    }

    #[test]
    fn requesting_the_wrong_shape_names_the_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let w = Weights::open(&write_file(dir.path(), None), &Device::Cpu).unwrap();
        let err = w.var_builder(DType::F32).unwrap().pp("block").get((3, 3), "w").unwrap_err().to_string();
        assert!(err.contains("block.w"), "{err}");
    }

    #[test]
    fn values_roundtrip_exactly_for_f32() {
        let dir = tempfile::tempdir().unwrap();
        let w = Weights::open(&write_file(dir.path(), None), &Device::Cpu).unwrap();
        let t = w.var_builder(DType::F32).unwrap().pp("block").get((2, 2), "w").unwrap();
        assert_eq!(t.to_vec2::<f32>().unwrap(), vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    fn file_with_raw_header(dir: &std::path::Path, header: &str, data: &[u8]) -> std::path::PathBuf {
        let path = dir.join("raw.safetensors");
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn overflowing_shape_product_is_rejected_not_wrapped() {
        // 4611686018427387905 * 4 は u64 で 4 に巻き戻る。巻き戻りを通すと巨大確保で落ちる。
        let dir = tempfile::tempdir().unwrap();
        let h = r#"{"w":{"dtype":"F32","shape":[4611686018427387905],"data_offsets":[0,4]}}"#;
        let p = file_with_raw_header(dir.path(), h, &[0, 0, 0, 0]);
        assert!(Weights::open(&p, &Device::Cpu).is_err());
    }

    #[test]
    fn overflowing_data_offsets_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let h = r#"{"w":{"dtype":"U8","shape":[1],"data_offsets":[0,18446744073709551615]}}"#;
        let p = file_with_raw_header(dir.path(), h, &[0]);
        assert!(Weights::open(&p, &Device::Cpu).is_err());
    }

    #[test]
    fn non_integer_shape_entries_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let h = r#"{"w":{"dtype":"U8","shape":["a"],"data_offsets":[0,1]}}"#;
        let p = file_with_raw_header(dir.path(), h, &[0]);
        let err = Weights::open(&p, &Device::Cpu).err().expect("must fail").to_string();
        assert!(err.contains("shape"), "{err}");
    }
}
