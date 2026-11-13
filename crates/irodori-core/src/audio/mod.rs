//! 参照音声の読み込み・リサンプル・正規化と WAV 書き出し。

pub mod loudness;

use std::fs::File;
use std::path::Path;

use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};
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

/// モノラル波形のリサンプル（窓付き sinc）。出力長は `round(n * to / from)`。
pub fn resample_mono(samples: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    if from == to || samples.is_empty() {
        return Ok(samples.to_vec());
    }
    let ratio = to as f64 / from as f64;
    let params = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: WindowFunction::BlackmanHarris2,
    };
    let chunk = 2048;
    let mut r = SincFixedIn::<f32>::new(ratio, 1.0, params, chunk, 1).map_err(audio_err)?;
    let delay = r.output_delay();
    let mut out: Vec<f32> = Vec::with_capacity((samples.len() as f64 * ratio) as usize + chunk);
    for piece in samples.chunks(chunk) {
        let res = if piece.len() == chunk {
            r.process(&[piece], None)
        } else {
            r.process_partial(Some(&[piece]), None)
        }
        .map_err(audio_err)?;
        out.extend_from_slice(&res[0]);
    }
    // 内部に残った遅延分を押し出す。
    out.extend_from_slice(&r.process_partial(None::<&[Vec<f32>]>, None).map_err(audio_err)?[0]);
    let want = (samples.len() as f64 * ratio).round() as usize;
    let out: Vec<f32> = out.into_iter().skip(delay).take(want).collect();
    Ok(out)
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
