//! 参照音声の読み込み・リサンプル・正規化と WAV 書き出し。

pub mod loudness;

use std::fs::File;
use std::path::Path;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::{IrodoriError, Result};

/// これより短い参照音声は受け付けない（秒）。
const MIN_REFERENCE_SECONDS: f32 = 0.1;
/// ピークがこれ未満なら無音とみなす。
const SILENCE_PEAK: f32 = 1e-4;

fn audio_err(e: impl std::fmt::Display) -> IrodoriError {
    IrodoriError::Audio(e.to_string())
}

/// 音声ファイルをデコードしてモノラル（チャンネル平均）にする。`max_samples` に達したら打ち切る。
fn decode_file(path: &Path, max_samples: usize) -> Result<(Vec<f32>, u32)> {
    decode_until(path, |_| max_samples)
}

/// 音声ファイルをモノラル（チャンネル平均）にデコードする。そのファイルの実レートで `max_seconds` 分に達したら打ち切る。
/// 戻り値は（`max_seconds` 以内のサンプル, 元のレート）。
pub fn decode_mono(path: &Path, max_seconds: f32) -> Result<(Vec<f32>, u32)> {
    decode_until(path, |rate| (max_seconds * rate as f32) as usize)
}

/// `limit(元のレート)` サンプルに達するまでデコードする。
fn decode_until(path: &Path, limit: impl Fn(u32) -> usize) -> Result<(Vec<f32>, u32)> {
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(audio_err)?;
    let mut format = probed.format;
    let track = format.default_track().ok_or_else(|| audio_err("no audio track"))?;
    let (track_id, rate) = (track.id, track.codec_params.sample_rate.ok_or_else(|| audio_err("unknown sample rate"))?);
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default()).map_err(audio_err)?;

    let max_samples = limit(rate);
    let mut mono: Vec<f32> = Vec::new();
    while mono.len() < max_samples {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(audio_err(e)),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(buf) => {
                let spec = *buf.spec();
                let mut sb = SampleBuffer::<f32>::new(buf.capacity() as u64, spec);
                sb.copy_interleaved_ref(buf);
                let ch = spec.channels.count().max(1);
                mono.extend(sb.samples().chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32));
            }
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(audio_err(e)),
        }
    }
    mono.truncate(max_samples);
    Ok((mono, rate))
}

/// `torchaudio.functional.resample`（`sinc_interp_hann`, lowpass_filter_width=6, rolloff=0.99）と同じ
/// 窓付き sinc リサンプル。モノラル波形、出力長は `ceil(n * to / from)`。
///
/// 周波数比を gcd で約分し、位相ごとのカーネル表（`new` 本 × `2*width + orig` タップ）を作って、
/// 入力を `orig` サンプルずつ進めながら `new` 個ずつ出力する（上流の conv1d(stride=orig) と同じ計算）。
pub fn resample_mono(samples: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    if from == 0 || to == 0 {
        return Err(IrodoriError::Audio(format!("invalid sample rate: {from} -> {to}")));
    }
    if from == to || samples.is_empty() {
        return Ok(samples.to_vec());
    }
    const LOWPASS_WIDTH: f64 = 6.0;
    const ROLLOFF: f64 = 0.99;
    let g = gcd(from, to);
    let (orig, new) = ((from / g) as usize, (to / g) as usize);
    let base = orig.min(new) as f64 * ROLLOFF;
    let width = (LOWPASS_WIDTH * orig as f64 / base).ceil() as usize;
    let klen = 2 * width + orig;
    let scale = base / orig as f64;

    let mut kernels = vec![0f32; new * klen];
    for p in 0..new {
        for i in 0..klen {
            let idx = (i as f64 - width as f64) / orig as f64;
            let t = ((-(p as f64) / new as f64 + idx) * base).clamp(-LOWPASS_WIDTH, LOWPASS_WIDTH);
            let window = (t * std::f64::consts::PI / LOWPASS_WIDTH / 2.0).cos().powi(2);
            let x = t * std::f64::consts::PI;
            let sinc = if x == 0.0 { 1.0 } else { x.sin() / x };
            kernels[p * klen + i] = (sinc * window * scale) as f32;
        }
    }

    let n = samples.len();
    let mut padded = vec![0f32; width + n + width + orig];
    padded[width..width + n].copy_from_slice(samples);
    let target = (new * n).div_ceil(orig);
    let mut out = Vec::with_capacity(target);
    for j in 0..target {
        let (frame, p) = (j / new, j % new);
        let window = &padded[frame * orig..frame * orig + klen];
        let kernel = &kernels[p * klen..(p + 1) * klen];
        out.push(window.iter().zip(kernel).map(|(x, k)| x * k).sum::<f32>());
    }
    Ok(out)
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// 参照音声を読み、モノラル化して `target_sr` にリサンプルする。`max_seconds` を超える分は捨てる。
/// 無音は [`IrodoriError::SilentReference`]、0.1 秒未満は [`IrodoriError::ReferenceTooShort`]。
pub fn load_reference(path: &Path, target_sr: u32, max_seconds: f32) -> Result<Vec<f32>> {
    // 長い入力をまるごとデコードしないよう、入力側のレートで上限を決める（レート不明のため安全側に 192kHz 換算）。
    let (samples, rate) = decode_file(path, (max_seconds * 192_000.0) as usize)?;
    let max_in = (max_seconds * rate as f32) as usize;
    let samples = &samples[..samples.len().min(max_in)];
    let seconds = samples.len() as f32 / rate as f32;
    if seconds < MIN_REFERENCE_SECONDS {
        return Err(IrodoriError::ReferenceTooShort { seconds });
    }
    if samples.iter().fold(0f32, |m, v| m.max(v.abs())) < SILENCE_PEAK {
        return Err(IrodoriError::SilentReference);
    }
    resample_mono(samples, rate, target_sr)
}

/// モノラル 24bit PCM の WAV を書く（Python 版の保存形式と同じ）。
pub fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec { channels: 1, sample_rate, bits_per_sample: 24, sample_format: hound::SampleFormat::Int };
    let mut w = hound::WavWriter::create(path, spec).map_err(audio_err)?;
    for s in samples {
        w.write_sample((s.clamp(-1.0, 1.0) * 8_388_607.0) as i32).map_err(audio_err)?;
    }
    w.finalize().map_err(audio_err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IrodoriError;

    fn write_test_wav(path: &std::path::Path, channels: u16, rate: u32, samples: &[f32]) {
        let spec = hound::WavSpec { channels, sample_rate: rate, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        for s in samples {
            w.write_sample((s * 30000.0) as i16).unwrap();
        }
        w.finalize().unwrap();
    }

    fn tone(rate: u32, secs: f32, amp: f32) -> Vec<f32> {
        (0..(rate as f32 * secs) as usize).map(|i| amp * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / rate as f32).sin()).collect()
    }

    #[test]
    fn stereo_44k_becomes_mono_48k_with_expected_length() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.wav");
        let mono = tone(44100, 1.0, 0.3);
        let stereo: Vec<f32> = mono.iter().flat_map(|v| [*v, *v]).collect();
        write_test_wav(&p, 2, 44100, &stereo);
        let out = load_reference(&p, 48000, 120.0).unwrap();
        let want = 48000i64;
        assert!((out.len() as i64 - want).abs() <= 2, "{}", out.len());
    }

    /// 理想的にリサンプルした正弦波との最大誤差（両端の過渡部分は除く）。
    fn resample_error(from: u32, to: u32, freq: f32) -> f32 {
        let secs = 1.0f32;
        let src: Vec<f32> = (0..(from as f32 * secs) as usize).map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / from as f32).sin()).collect();
        let out = resample_mono(&src, from, to).unwrap();
        assert_eq!(out.len(), (to as f32 * secs) as usize);
        let margin = 400;
        (margin..out.len() - margin)
            .map(|i| (out[i] - (2.0 * std::f32::consts::PI * freq * i as f32 / to as f32).sin()).abs())
            .fold(0f32, f32::max)
    }

    #[test]
    fn upsampling_keeps_the_waveform_time_aligned() {
        // 位相が遅れ/進みすると正弦波との誤差が大きくなる（過去の実装は遅延補正を二重にかけて 100 サンプル級にずれた）。
        let err = resample_error(24000, 48000, 440.0);
        assert!(err < 0.02, "max error vs ideal sine {err}");
    }

    #[test]
    fn downsampling_keeps_the_waveform_time_aligned() {
        let err = resample_error(48000, 16000, 440.0);
        assert!(err < 0.02, "max error vs ideal sine {err}");
    }

    #[test]
    fn invalid_sample_rates_are_rejected() {
        assert!(resample_mono(&[0.1, 0.2], 0, 48000).is_err());
        assert!(resample_mono(&[0.1, 0.2], 48000, 0).is_err());
        assert_eq!(resample_mono(&[0.1, 0.2], 48000, 48000).unwrap(), vec![0.1, 0.2]);
    }

    #[test]
    fn audio_longer_than_the_limit_is_cut() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("long.wav");
        write_test_wav(&p, 1, 8000, &tone(8000, 8.0, 0.3));
        let out = load_reference(&p, 8000, 5.0).unwrap();
        assert_eq!(out.len(), 5 * 8000);
    }

    #[test]
    fn silent_reference_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("s.wav");
        write_test_wav(&p, 1, 8000, &vec![0.0; 16000]);
        assert!(matches!(load_reference(&p, 8000, 120.0), Err(IrodoriError::SilentReference)));
    }

    #[test]
    fn too_short_reference_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.wav");
        write_test_wav(&p, 1, 8000, &tone(8000, 0.05, 0.3));
        assert!(matches!(load_reference(&p, 8000, 120.0), Err(IrodoriError::ReferenceTooShort { .. })));
    }

    #[test]
    fn unreadable_file_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.wav");
        std::fs::write(&p, b"not audio").unwrap();
        assert!(load_reference(&p, 48000, 120.0).is_err());
        assert!(load_reference(&dir.path().join("missing.wav"), 48000, 120.0).is_err());
    }

    #[test]
    fn write_wav_roundtrips_at_24_bit() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("o.wav");
        let x = tone(8000, 0.5, 0.5);
        write_wav(&p, &x, 8000).unwrap();
        let r = hound::WavReader::open(&p).unwrap();
        assert_eq!((r.spec().bits_per_sample, r.spec().sample_rate, r.spec().channels), (24, 8000, 1));
        let back = load_reference(&p, 8000, 120.0).unwrap();
        let max_diff = x.iter().zip(&back).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(max_diff < 1e-3, "{max_diff}");
    }
}
