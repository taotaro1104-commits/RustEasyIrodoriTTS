//! DACVAE codec（Semantic-DACVAE-Japanese-32dim）。透かし無効の decode と、決定的な encode（mean のみ）。
//!
//! decode: `quantizer.out_proj → Conv(k7) → DecoderBlock×4 → Snake → Conv(→1ch) → tanh`。
//! Irodori は `decoder.alpha=0` で透かしを無効にし、`wm_model.encoder_block.forward_no_conv`
//! （= `pre.0` Snake、`pre.1` Conv(→1ch)、Tanh）へ差し替えるため、透かし用 LSTM・メッセージ処理・
//! `DecoderBlock` 内の透かし系の層は一切読まない。encode: `encoder(reflect_pad(wav)) → in_proj` の前半（mean）。

pub mod layers;

use std::path::Path;

use candle_core::pickle::PthTensors;
use candle_core::{DType, Device, Tensor};

use crate::{IrodoriError, Result};
use layers::{reflect_pad_right, snake, Conv1d, ConvT1d, Fetch, ResidualUnit};

/// コーデックの寸法。レート列はアーキテクチャ定数、チャンネル数はチェックポイントの形状から決める。
#[derive(Debug, Clone)]
pub struct CodecArch {
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    pub decoder_rates: Vec<usize>,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub codebook_dim: usize,
    pub sample_rate: u32,
}

const DILATIONS: [usize; 3] = [1, 3, 9];

/// 実運用の窓幅（フレーム）。decode の受容野は約 10 フレーム、encode は約 8000 サンプル（≈4 フレーム）なので
/// 重なりには十分な余裕を持たせている。
pub const DECODE_CHUNK_FRAMES: usize = 200;
pub const DECODE_OVERLAP_FRAMES: usize = 16;
pub const ENCODE_CHUNK_FRAMES: usize = 250;
pub const ENCODE_OVERLAP_FRAMES: usize = 25;

struct EncoderBlock {
    res: Vec<ResidualUnit>,
    alpha: Tensor,
    down: Conv1d,
}

struct DecoderBlock {
    alpha: Tensor,
    up: ConvT1d,
    res: Vec<ResidualUnit>,
}

pub struct Codec {
    first: Conv1d,
    enc_blocks: Vec<EncoderBlock>,
    enc_alpha: Tensor,
    enc_last: Conv1d,
    in_proj: Conv1d,
    out_proj: Conv1d,
    dec_first: Conv1d,
    dec_blocks: Vec<DecoderBlock>,
    out_alpha: Tensor,
    out_conv: Conv1d,
    hop: usize,
    sample_rate: u32,
    codebook_dim: usize,
    /// 重みの dtype。入出力は常に F32 で、層の内側だけこの dtype で計算する。
    dtype: DType,
}

impl Codec {
    pub fn from_fetch(fetch: Fetch, arch: CodecArch) -> Result<Self> {
        Self::from_fetch_dtype(fetch, arch, DType::F32)
    }

    pub fn from_fetch_dtype(fetch: Fetch, a: CodecArch, dt: DType) -> Result<Self> {
        if a.decoder_rates.iter().chain(&a.encoder_rates).any(|s| s % 2 != 0) {
            return Err(IrodoriError::InvalidConfig("only even strides are supported".into()));
        }
        let alpha = |name: String| -> Result<Tensor> { Ok(fetch(&name)?.to_dtype(dt)?) };
        let first = Conv1d::load_wn(fetch, "encoder.block.0", 3, 1, 1, dt)?;
        let enc_blocks = a
            .encoder_rates
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let p = format!("encoder.block.{}", i + 1);
                Ok(EncoderBlock {
                    res: DILATIONS.iter().enumerate().map(|(j, &d)| ResidualUnit::load(fetch, &format!("{p}.block.{j}"), d, dt)).collect::<Result<_>>()?,
                    alpha: alpha(format!("{p}.block.3.alpha"))?,
                    down: Conv1d::load_wn(fetch, &format!("{p}.block.4"), s / 2, s, 1, dt)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let n = a.encoder_rates.len() + 1;
        let dec_blocks = a
            .decoder_rates
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let p = format!("decoder.model.{}", i + 1);
                Ok(DecoderBlock {
                    alpha: alpha(format!("{p}.block.0.alpha"))?,
                    up: ConvT1d::load_wn(fetch, &format!("{p}.block.1"), (s + 1) / 2, s, dt)?,
                    res: [4, 5, 8].iter().zip(DILATIONS).map(|(&j, d)| ResidualUnit::load(fetch, &format!("{p}.block.{j}"), d, dt)).collect::<Result<_>>()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            first,
            enc_blocks,
            enc_alpha: alpha(format!("encoder.block.{n}.alpha"))?,
            enc_last: Conv1d::load_wn(fetch, &format!("encoder.block.{}", n + 1), 1, 1, 1, dt)?,
            in_proj: Conv1d::load_wn(fetch, "quantizer.in_proj", 0, 1, 1, dt)?,
            out_proj: Conv1d::load_wn(fetch, "quantizer.out_proj", 0, 1, 1, dt)?,
            dec_first: Conv1d::load_wn(fetch, "decoder.model.0", 3, 1, 1, dt)?,
            dec_blocks,
            out_alpha: alpha("decoder.wm_model.encoder_block.pre.0.alpha".into())?,
            out_conv: Conv1d::load_wn(fetch, "decoder.wm_model.encoder_block.pre.1", 3, 1, 1, dt)?,
            hop: a.encoder_rates.iter().product(),
            sample_rate: a.sample_rate,
            codebook_dim: a.codebook_dim,
            dtype: dt,
        })
    }

    /// `weights.pth`（`state_dict`）を読む。チャンネル数は形状から決め、レート列は DACVAE の定数。
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        Self::load_with_dtype(path, device, DType::F32)
    }

    pub fn load_with_dtype(path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let pth = PthTensors::new(path, Some("state_dict"))?;
        let fetch = |name: &str| -> Result<Tensor> {
            pth.get(name)?.ok_or_else(|| IrodoriError::MissingKey(name.to_string()))?.to_device(device).map_err(Into::into)
        };
        let dim0 = |name: &str, axis: usize| -> Result<usize> { Ok(fetch(name)?.dim(axis)?) };
        let arch = CodecArch {
            encoder_dim: dim0("encoder.block.0.weight_v", 0)?,
            encoder_rates: vec![2, 8, 10, 12],
            decoder_rates: vec![12, 10, 8, 2],
            latent_dim: dim0("quantizer.out_proj.weight_v", 0)?,
            decoder_dim: dim0("decoder.model.0.weight_v", 0)?,
            codebook_dim: dim0("quantizer.out_proj.weight_v", 1)?,
            sample_rate: 48000,
        };
        Self::from_fetch_dtype(&fetch, arch, dtype)
    }

    pub fn hop(&self) -> usize {
        self.hop
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// `wav: (B, 1, S)`（codec のサンプルレート）→ 潜在 `(B, T, codebook_dim)`。端数は reflect で hop の倍数に揃える。
    pub fn encode(&self, wav: &Tensor) -> Result<Tensor> {
        let (_, c, len) = wav.dims3()?;
        if c != 1 {
            return Err(IrodoriError::InvalidInput(format!("encode expects mono (B, 1, S), got {} channels", c)));
        }
        let x = reflect_pad_right(&wav.to_dtype(self.dtype)?, (self.hop - len % self.hop) % self.hop)?;
        let mut x = self.first.forward(&x)?;
        for b in &self.enc_blocks {
            for r in &b.res {
                x = r.forward(&x)?;
            }
            x = b.down.forward(&snake(&x, &b.alpha)?)?;
        }
        let z = self.enc_last.forward(&snake(&x, &self.enc_alpha)?)?;
        let mean = self.in_proj.forward(&z)?.narrow(1, 0, self.codebook_dim)?;
        Ok(mean.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?)
    }

    /// `decode` を `chunk_frames` フレームずつ、前後 `overlap_frames` の文脈つきで実行して連結する。
    /// 中間活性（30 秒で数百 MB〜GB）を窓 1 つ分に抑えるためのもの。`overlap_frames` が受容野以上なら
    /// `decode` と数値的に一致する（実 codec は約 10 フレームで足りる）。
    pub fn decode_chunked(&self, latent: &Tensor, chunk_frames: usize, overlap_frames: usize) -> Result<Tensor> {
        if chunk_frames == 0 {
            return Err(IrodoriError::InvalidInput("chunk_frames must be > 0".into()));
        }
        let t = latent.dim(1)?;
        let mut parts: Vec<Tensor> = Vec::new();
        for f0 in (0..t).step_by(chunk_frames) {
            let f1 = (f0 + chunk_frames).min(t);
            let (s, e) = (f0.saturating_sub(overlap_frames), (f1 + overlap_frames).min(t));
            let y = self.decode(&latent.narrow(1, s, e - s)?)?;
            parts.push(y.narrow(2, (f0 - s) * self.hop, (f1 - f0) * self.hop)?);
        }
        Ok(Tensor::cat(&parts.iter().collect::<Vec<_>>(), 2)?)
    }

    /// `encode` のチャンク版。全体を先に hop の倍数へ reflect パディングし、窓ごとに encode して
    /// 中央部分だけを連結する（窓は hop の倍数なので窓内のパディングは起きない）。
    pub fn encode_chunked(&self, wav: &Tensor, chunk_frames: usize, overlap_frames: usize) -> Result<Tensor> {
        if chunk_frames == 0 {
            return Err(IrodoriError::InvalidInput("chunk_frames must be > 0".into()));
        }
        let (_, c, len) = wav.dims3()?;
        if c != 1 {
            return Err(IrodoriError::InvalidInput(format!("encode expects mono (B, 1, S), got {c} channels")));
        }
        let wav = reflect_pad_right(wav, (self.hop - len % self.hop) % self.hop)?;
        let t = wav.dim(2)? / self.hop;
        let mut parts: Vec<Tensor> = Vec::new();
        for f0 in (0..t).step_by(chunk_frames) {
            let f1 = (f0 + chunk_frames).min(t);
            let (s, e) = (f0.saturating_sub(overlap_frames), (f1 + overlap_frames).min(t));
            let z = self.encode(&wav.narrow(2, s * self.hop, (e - s) * self.hop)?)?;
            parts.push(z.narrow(1, f0 - s, f1 - f0)?);
        }
        Ok(Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1)?)
    }

    /// 潜在 `(B, T, codebook_dim)` → 波形 `(B, 1, T * hop)`。
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let z = latent.transpose(1, 2)?.contiguous()?.to_dtype(self.out_alpha.dtype())?;
        let mut x = self.dec_first.forward(&self.out_proj.forward(&z)?)?;
        for b in &self.dec_blocks {
            x = b.up.forward(&snake(&x, &b.alpha)?)?;
            for r in &b.res {
                x = r.forward(&x)?;
            }
        }
        Ok(self.out_conv.forward(&snake(&x, &self.out_alpha)?)?.tanh()?.to_dtype(DType::F32)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::test_util::fake_tensor;
    use candle_core::{Device, Tensor};
    use std::collections::HashMap;

    /// 小さな寸法の合成アーキテクチャ（hop = 16）。
    fn arch() -> CodecArch {
        CodecArch { encoder_dim: 2, encoder_rates: vec![2, 2, 2, 2], decoder_rates: vec![2, 2, 2, 2], latent_dim: 32, decoder_dim: 32, codebook_dim: 4, sample_rate: 48000 }
    }

    fn wn(m: &mut HashMap<String, Tensor>, p: &str, cin: usize, cout: usize, k: usize) {
        m.insert(format!("{p}.weight_v"), fake_tensor(&format!("{p}v"), &[cout, cin, k]));
        m.insert(format!("{p}.weight_g"), fake_tensor(&format!("{p}g"), &[cout, 1, 1]).abs().unwrap().affine(1.0, 0.5).unwrap());
        m.insert(format!("{p}.bias"), fake_tensor(&format!("{p}b"), &[cout]));
    }

    fn wnt(m: &mut HashMap<String, Tensor>, p: &str, cin: usize, cout: usize, k: usize) {
        m.insert(format!("{p}.weight_v"), fake_tensor(&format!("{p}v"), &[cin, cout, k]));
        m.insert(format!("{p}.weight_g"), fake_tensor(&format!("{p}g"), &[cin, 1, 1]).abs().unwrap().affine(1.0, 0.5).unwrap());
        m.insert(format!("{p}.bias"), fake_tensor(&format!("{p}b"), &[cout]));
    }

    fn snake(m: &mut HashMap<String, Tensor>, p: &str, c: usize) {
        m.insert(format!("{p}.alpha"), fake_tensor(&format!("{p}a"), &[1, c, 1]).abs().unwrap().affine(1.0, 0.5).unwrap());
    }

    fn residual(m: &mut HashMap<String, Tensor>, p: &str, dim: usize) {
        snake(m, &format!("{p}.block.0"), dim);
        wn(m, &format!("{p}.block.1"), dim, dim, 7);
        snake(m, &format!("{p}.block.2"), dim);
        wn(m, &format!("{p}.block.3"), dim, dim, 1);
    }

    fn weights(a: &CodecArch) -> HashMap<String, Tensor> {
        let mut m = HashMap::new();
        wn(&mut m, "encoder.block.0", 1, a.encoder_dim, 7);
        let mut d = a.encoder_dim;
        for (i, _s) in a.encoder_rates.iter().enumerate() {
            d *= 2;
            let p = format!("encoder.block.{}", i + 1);
            for j in 0..3 {
                residual(&mut m, &format!("{p}.block.{j}"), d / 2);
            }
            snake(&mut m, &format!("{p}.block.3"), d / 2);
            wn(&mut m, &format!("{p}.block.4"), d / 2, d, 2 * a.encoder_rates[i]);
        }
        let n = a.encoder_rates.len() + 1;
        snake(&mut m, &format!("encoder.block.{n}"), d);
        wn(&mut m, &format!("encoder.block.{}", n + 1), d, a.latent_dim, 3);
        wn(&mut m, "quantizer.in_proj", a.latent_dim, a.codebook_dim * 2, 1);
        wn(&mut m, "quantizer.out_proj", a.codebook_dim, a.latent_dim, 1);
        wn(&mut m, "decoder.model.0", a.latent_dim, a.decoder_dim, 7);
        for (i, s) in a.decoder_rates.iter().enumerate() {
            let (cin, cout) = (a.decoder_dim >> i, a.decoder_dim >> (i + 1));
            let p = format!("decoder.model.{}", i + 1);
            snake(&mut m, &format!("{p}.block.0"), cin);
            wnt(&mut m, &format!("{p}.block.1"), cin, cout, 2 * s);
            for j in [4, 5, 8] {
                residual(&mut m, &format!("{p}.block.{j}"), cout);
            }
        }
        let last = a.decoder_dim >> a.decoder_rates.len();
        snake(&mut m, "decoder.wm_model.encoder_block.pre.0", last);
        wn(&mut m, "decoder.wm_model.encoder_block.pre.1", last, 1, 7);
        m
    }

    fn codec() -> Codec {
        let a = arch();
        let w = weights(&a);
        Codec::from_fetch(&|n| w.get(n).cloned().ok_or_else(|| crate::IrodoriError::MissingKey(n.to_string())), a).unwrap()
    }

    #[test]
    fn bf16_codec_takes_and_returns_f32_and_stays_close() {
        use candle_core::DType;
        let a = arch();
        let w = weights(&a);
        let fetch = |n: &str| w.get(n).cloned().ok_or_else(|| crate::IrodoriError::MissingKey(n.to_string()));
        let c16 = Codec::from_fetch_dtype(&fetch, a, DType::BF16).unwrap();
        let c32 = codec();
        let z = fake_tensor("zb", &[1, 4, 4]);
        let (y16, y32) = (c16.decode(&z).unwrap(), c32.decode(&z).unwrap());
        assert_eq!(y16.dtype(), DType::F32);
        crate::golden::assert_close(&y16, &y32, 8e-2, 8e-2).unwrap();
        let wav = fake_tensor("wb", &[1, 1, 64]);
        let (z16, z32) = (c16.encode(&wav).unwrap(), c32.encode(&wav).unwrap());
        assert_eq!(z16.dtype(), DType::F32);
        crate::golden::assert_close(&z16, &z32, 8e-2, 8e-2).unwrap();
    }

    #[test]
    fn hop_is_the_product_of_encoder_rates() {
        assert_eq!(codec().hop(), 16);
        assert_eq!(codec().sample_rate(), 48000);
    }

    #[test]
    fn decode_length_is_frames_times_hop_and_bounded_by_tanh() {
        let z = fake_tensor("z", &[1, 3, 4]);
        let y = codec().decode(&z).unwrap();
        assert_eq!(y.dims(), &[1, 1, 48]);
        let v = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| x.is_finite() && x.abs() <= 1.0));
    }

    #[test]
    fn encode_pads_to_a_whole_number_of_frames() {
        let c = codec();
        assert_eq!(c.encode(&fake_tensor("w1", &[1, 1, 40])).unwrap().dims(), &[1, 3, 4]);
        assert_eq!(c.encode(&fake_tensor("w2", &[1, 1, 32])).unwrap().dims(), &[1, 2, 4]);
    }

    #[test]
    fn chunked_decode_matches_full_decode_when_overlap_covers_the_receptive_field() {
        // 合成アーキテクチャ（レート 2）は受容野が広いので重なりを大きく取る。
        let c = codec();
        let z = fake_tensor("z-long", &[1, 60, 4]);
        let full = c.decode(&z).unwrap();
        let chunked = c.decode_chunked(&z, 20, 48).unwrap();
        assert_eq!(chunked.dims(), full.dims());
        crate::golden::assert_close(&chunked, &full, 1e-5, 0.0).unwrap();
    }

    #[test]
    fn chunked_decode_handles_a_ragged_last_chunk() {
        let c = codec();
        let z = fake_tensor("z-ragged", &[1, 47, 4]);
        let chunked = c.decode_chunked(&z, 20, 48).unwrap();
        assert_eq!(chunked.dims(), &[1, 1, 47 * 16]);
        crate::golden::assert_close(&chunked, &c.decode(&z).unwrap(), 1e-5, 0.0).unwrap();
    }

    #[test]
    fn chunked_encode_matches_full_encode_when_overlap_covers_the_receptive_field() {
        let c = codec();
        let wav = fake_tensor("w-long", &[1, 1, 16 * 100 + 5]); // hop の倍数ではない
        let full = c.encode(&wav).unwrap();
        let chunked = c.encode_chunked(&wav, 20, 80).unwrap();
        assert_eq!(chunked.dims(), full.dims());
        crate::golden::assert_close(&chunked, &full, 1e-5, 0.0).unwrap();
    }

    #[test]
    fn zero_chunk_size_is_rejected() {
        let c = codec();
        assert!(c.decode_chunked(&fake_tensor("z", &[1, 4, 4]), 0, 2).is_err());
        assert!(c.encode_chunked(&fake_tensor("w", &[1, 1, 64]), 0, 2).is_err());
    }

    #[test]
    fn encode_rejects_non_mono_input() {
        assert!(codec().encode(&fake_tensor("w", &[1, 2, 32])).is_err());
    }

    #[test]
    fn missing_tensor_error_names_the_key() {
        let a = arch();
        let mut w = weights(&a);
        w.remove("quantizer.out_proj.weight_v");
        let err = Codec::from_fetch(&|n| w.get(n).cloned().ok_or_else(|| crate::IrodoriError::MissingKey(n.to_string())), a).err().unwrap().to_string();
        assert!(err.contains("quantizer.out_proj.weight_v"), "{err}");
    }

    #[test]
    fn real_model_matches_python_golden() {
        let (Some(dir), Some(_)) = (std::env::var_os("IRODORI_CODEC_DIR"), crate::golden::dir()) else {
            eprintln!("skip: IRODORI_CODEC_DIR / IRODORI_GOLDEN_DIR not set");
            return;
        };
        let codec = Codec::load(&std::path::Path::new(&dir).join("weights.pth"), &Device::Cpu).unwrap();
        let g = crate::golden::load("codec").unwrap();
        let z = codec.encode(&g["wav"]).unwrap();
        crate::golden::assert_close(&z, &g["latent"], 2e-3, 2e-3).unwrap_or_else(|e| panic!("encode: {e}"));
        let y = codec.decode(&g["latent"]).unwrap();
        crate::golden::assert_close(&y, &g["decoded"], 2e-3, 2e-3).unwrap_or_else(|e| panic!("decode: {e}"));
        // チャンク版（実運用の窓幅）も Python の結果に一致する。
        let yc = codec.decode_chunked(&g["latent"], 20, DECODE_OVERLAP_FRAMES).unwrap();
        crate::golden::assert_close(&yc, &g["decoded"], 2e-3, 2e-3).unwrap_or_else(|e| panic!("chunked decode: {e}"));
        let zc = codec.encode_chunked(&g["wav"], 20, ENCODE_OVERLAP_FRAMES).unwrap();
        crate::golden::assert_close(&zc, &g["latent"], 2e-3, 2e-3).unwrap_or_else(|e| panic!("chunked encode: {e}"));
    }
}
