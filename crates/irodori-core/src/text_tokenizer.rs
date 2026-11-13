//! HF tokenizer のラッパ（BOS 手動付与・右パディング）。
//!
//! 上流 `PretrainedTextTokenizer` と同じ挙動: `add_special_tokens=false`、BOS は手動で先頭に付与、
//! バッチは右パディングで本文を `max_len - 1` に切る。

use std::path::Path;

use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

use crate::normalize::normalize_text;
use crate::{IrodoriError, Result};

pub struct TextTokenizer {
    inner: Tokenizer,
    bos_id: u32,
    pad_id: u32,
}

impl TextTokenizer {
    /// `tokenizer.json` を読む。ファイル内の truncation/padding 設定は無効化する。
    pub fn from_file(path: &Path, bos_id: u32, pad_id: u32) -> Result<Self> {
        let mut inner =
            Tokenizer::from_file(path).map_err(|e| IrodoriError::Tokenizer(e.to_string()))?;
        inner
            .with_truncation(None)
            .map_err(|e| IrodoriError::Tokenizer(e.to_string()))?;
        inner.with_padding(None);
        Ok(Self { inner, bos_id, pad_id })
    }

    fn body_ids(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| IrodoriError::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// BOS を先頭に付けたトークン列。
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = vec![self.bos_id];
        ids.extend(self.body_ids(text)?);
        Ok(ids)
    }

    /// 右パディング済みバッチ。`ids: [B, max_len] u32`, `mask: [B, max_len] u8`。
    pub fn batch_encode(&self, texts: &[&str], max_len: usize) -> Result<(Tensor, Tensor)> {
        if texts.is_empty() {
            return Err(IrodoriError::InvalidInput("texts must not be empty".into()));
        }
        if max_len == 0 {
            return Err(IrodoriError::InvalidInput("max_len must be > 0".into()));
        }
        let mut ids = Vec::with_capacity(texts.len() * max_len);
        let mut mask = Vec::with_capacity(texts.len() * max_len);
        for text in texts {
            let body = if max_len == 1 { Vec::new() } else { self.body_ids(text)? };
            let keep = body.len().min(max_len - 1);
            ids.push(self.bos_id);
            mask.push(1u8);
            ids.extend_from_slice(&body[..keep]);
            mask.extend(std::iter::repeat(1u8).take(keep));
            let pad = max_len - 1 - keep;
            ids.extend(std::iter::repeat(self.pad_id).take(pad));
            mask.extend(std::iter::repeat(0u8).take(pad));
        }
        let shape = (texts.len(), max_len);
        Ok((
            Tensor::from_vec(ids, shape, &Device::Cpu)?,
            Tensor::from_vec(mask, shape, &Device::Cpu)?,
        ))
    }

    /// 生テキストを正規化し、空・長すぎを明示エラーにして BOS 付きトークン列を返す。
    pub fn prepare(&self, raw: &str, max_len: usize) -> Result<Vec<u32>> {
        let text = normalize_text(raw);
        if text.trim().is_empty() {
            return Err(IrodoriError::EmptyText);
        }
        let ids = self.encode(&text)?;
        if ids.len() > max_len {
            return Err(IrodoriError::TextTooLong { tokens: ids.len(), max: max_len });
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// `<unk>=0 <s>=1 </s>=2 <pad>=3 a=4 b=5` の最小 WordLevel tokenizer.json。
    const MINI: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"<s>":1,"</s>":2,"<pad>":3,"a":4,"b":5},"unk_token":"<unk>"}}"#;

    fn mini() -> (tempfile::NamedTempFile, TextTokenizer) {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(MINI.as_bytes()).unwrap();
        let tok = TextTokenizer::from_file(f.path(), 1, 3).unwrap();
        (f, tok)
    }

    #[test]
    fn encode_prepends_bos_without_special_tokens() {
        let (_f, tok) = mini();
        assert_eq!(tok.encode("a b").unwrap(), vec![1, 4, 5]);
    }

    #[test]
    fn batch_encode_right_pads_and_truncates_body() {
        let (_f, tok) = mini();
        let (ids, mask) = tok.batch_encode(&["a", "b b b b"], 3).unwrap();
        assert_eq!(ids.to_vec2::<u32>().unwrap(), vec![vec![1, 4, 3], vec![1, 5, 5]]);
        assert_eq!(mask.to_vec2::<u8>().unwrap(), vec![vec![1, 1, 0], vec![1, 1, 1]]);
    }

    #[test]
    fn batch_encode_max_len_one_is_bos_only() {
        let (_f, tok) = mini();
        let (ids, mask) = tok.batch_encode(&["a", "b"], 1).unwrap();
        assert_eq!(ids.to_vec2::<u32>().unwrap(), vec![vec![1], vec![1]]);
        assert_eq!(mask.to_vec2::<u8>().unwrap(), vec![vec![1], vec![1]]);
    }

    #[test]
    fn prepare_rejects_text_that_normalizes_to_empty() {
        let (_f, tok) = mini();
        for raw in ["", "\u{3000}", "「」"] {
            assert!(matches!(tok.prepare(raw, 256), Err(crate::IrodoriError::EmptyText)), "{raw:?}");
        }
    }

    #[test]
    fn prepare_rejects_too_long_text_instead_of_truncating() {
        let (_f, tok) = mini();
        let text = vec!["a"; 300].join(" ");
        match tok.prepare(&text, 256) {
            Err(crate::IrodoriError::TextTooLong { tokens, max }) => {
                assert_eq!((tokens, max), (301, 256)); // BOS を含む
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn real_tokenizer_matches_python_golden() {
        let (Some(model_dir), Some(_)) = (std::env::var_os("IRODORI_MODEL_DIR"), crate::golden::dir()) else {
            eprintln!("skip: IRODORI_MODEL_DIR / IRODORI_GOLDEN_DIR not set");
            return;
        };
        let tok = TextTokenizer::from_file(
            &std::path::Path::new(&model_dir).join("tokenizer").join("tokenizer.json"),
            1,
            3,
        )
        .unwrap();
        let g = crate::golden::load("tokens").unwrap();
        let (ids, mask) = tok.batch_encode(&["こんにちは、世界。", "短い"], 32).unwrap();
        assert_eq!(ids.to_vec2::<u32>().unwrap(), g["ids"].to_vec2::<u32>().unwrap());
        assert_eq!(mask.to_vec2::<u8>().unwrap(), g["mask"].to_vec2::<u8>().unwrap());
    }
}
