//! 生成した音声の後処理（v2.0 の `easy_audio.py`）。ここでは終了無音の追加。話速の変更は M2c で実装する。

use std::path::{Path, PathBuf};

use crate::model::uid;
use crate::{Result, StoreError};

/// 24bit PCM の符号付き整数の範囲。
const MAX_24: i32 = 8_388_607;
const MIN_24: i32 = -8_388_608;

fn unreadable() -> StoreError {
    StoreError::Invalid("音声ファイルを読み込めません。".into())
}

/// どの形式の WAV も、24bit の整数（`MIN_24..=MAX_24`）に揃えて読む。
fn read_as_24bit(
    reader: &mut hound::WavReader<std::io::BufReader<std::fs::File>>,
) -> Result<Vec<i32>> {
    let spec = reader.spec();
    match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| {
                s.map(|v| ((v as f64 * 8_388_608.0).round() as i32).clamp(MIN_24, MAX_24))
                    .map_err(|_| unreadable())
            })
            .collect(),
        hound::SampleFormat::Int => {
            let bits = i32::from(spec.bits_per_sample);
            reader
                .samples::<i32>()
                .map(|s| {
                    s.map(|v| {
                        if bits < 24 {
                            v << (24 - bits)
                        } else {
                            v >> (bits - 24)
                        }
                    })
                    .map_err(|_| unreadable())
                })
                .collect()
        }
    }
}

/// 末尾に `seconds` 秒（0〜10）の無音を足した新しい WAV（24bit）を、同じフォルダに `<名前>_tail_<8桁>.wav` として作る。
/// 0 秒なら何もせず `source` を返す。元のサンプルはそのまま、チャンネル数も維持する。
pub fn add_silence(source: &Path, seconds: f64) -> Result<PathBuf> {
    if !seconds.is_finite() || !(0.0..=10.0).contains(&seconds) {
        return Err(StoreError::Invalid(
            "終了無音は0～10秒で指定してください。".into(),
        ));
    }
    if seconds == 0.0 {
        return Ok(source.to_path_buf());
    }
    let mut reader = hound::WavReader::open(source).map_err(|_| unreadable())?;
    let spec = reader.spec();
    let mut samples = read_as_24bit(&mut reader)?;
    let frames = (seconds * f64::from(spec.sample_rate)).round() as usize;
    samples.resize(samples.len() + frames * usize::from(spec.channels), 0);

    let stem = source
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "audio".into());
    let target = source.with_file_name(format!("{stem}_tail_{}.wav", &uid()[..8]));
    let partial = target.with_extension("partial.wav");
    let out_spec = hound::WavSpec {
        channels: spec.channels,
        sample_rate: spec.sample_rate,
        bits_per_sample: 24,
        sample_format: hound::SampleFormat::Int,
    };
    let written = (|| -> std::result::Result<(), hound::Error> {
        let mut writer = hound::WavWriter::create(&partial, out_spec)?;
        for s in &samples {
            writer.write_sample(*s)?;
        }
        writer.finalize()
    })();
    let moved = written
        .map_err(std::io::Error::other)
        .and_then(|()| std::fs::rename(&partial, &target));
    if let Err(e) = moved {
        let _ = std::fs::remove_file(&partial);
        return Err(StoreError::from(e));
    }
    Ok(target)
}

fn unreadable_format() -> StoreError {
    StoreError::Invalid("音声を読み込めません。形式を確認してください。".into())
}

/// WAV の長さ（秒）。読めなければ「形式を確認してください」。
pub fn wav_seconds(path: &Path) -> Result<f64> {
    let reader = hound::WavReader::open(path).map_err(|_| unreadable_format())?;
    let spec = reader.spec();
    if spec.sample_rate == 0 {
        return Err(unreadable_format());
    }
    Ok(f64::from(reader.duration()) / f64::from(spec.sample_rate))
}

/// WAV を、同じレート・チャンネル数の 24bit PCM にして `dst` へ書く（v2.0 の `sf.write(..., subtype='PCM_24')`）。
/// 戻り値は秒数。失敗したら `dst` は残さない。
pub fn copy_as_pcm24(src: &Path, dst: &Path) -> Result<f64> {
    let mut reader = hound::WavReader::open(src).map_err(|_| unreadable_format())?;
    let spec = reader.spec();
    let samples = read_as_24bit(&mut reader).map_err(|_| unreadable_format())?;
    let out_spec = hound::WavSpec {
        channels: spec.channels,
        sample_rate: spec.sample_rate,
        bits_per_sample: 24,
        sample_format: hound::SampleFormat::Int,
    };
    let written = (|| -> std::result::Result<(), hound::Error> {
        let mut writer = hound::WavWriter::create(dst, out_spec)?;
        for s in &samples {
            writer.write_sample(*s)?;
        }
        writer.finalize()
    })();
    if let Err(e) = written {
        let _ = std::fs::remove_file(dst);
        return Err(StoreError::from(std::io::Error::other(e)));
    }
    let frames = samples.len() / usize::from(spec.channels.max(1));
    Ok(frames as f64 / f64::from(spec.sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::temp_root;

    /// 24bit の WAV を書く（値は 24bit の整数のまま）。
    fn write_wav24(path: &std::path::Path, rate: u32, channels: u16, samples: &[i32]) {
        let spec = hound::WavSpec {
            channels,
            sample_rate: rate,
            bits_per_sample: 24,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        for s in samples {
            w.write_sample(*s).unwrap();
        }
        w.finalize().unwrap();
    }

    fn read_wav(path: &std::path::Path) -> (hound::WavSpec, Vec<i32>) {
        let mut r = hound::WavReader::open(path).unwrap();
        let spec = r.spec();
        (spec, r.samples::<i32>().map(|s| s.unwrap()).collect())
    }

    #[test]
    fn zero_seconds_returns_the_same_path() {
        let dir = temp_root();
        let src = dir.path().join("a.wav");
        write_wav24(&src, 24000, 1, &[1, 2, 3]);
        assert_eq!(add_silence(&src, 0.0).unwrap(), src);
        assert_eq!(
            std::fs::read_dir(dir.path())
                .unwrap()
                .filter(|e| e
                    .as_ref()
                    .unwrap()
                    .path()
                    .extension()
                    .is_some_and(|x| x == "wav"))
                .count(),
            1,
            "no new file"
        );
    }

    #[test]
    fn appends_exact_silence_frames() {
        let dir = temp_root();
        let src = dir.path().join("a.wav");
        write_wav24(&src, 24000, 1, &vec![1000; 2400]);
        let out = add_silence(&src, 1.5).unwrap();
        assert_ne!(out, src);
        let (spec, samples) = read_wav(&out);
        assert_eq!(
            samples.len(),
            2400 + 36000,
            "24000 Hz x 1.5 s of added silence"
        );
        assert!(samples[2400..].iter().all(|s| *s == 0));
        assert_eq!(
            (spec.bits_per_sample, spec.sample_rate, spec.channels),
            (24, 24000, 1)
        );
        let name = out.file_name().unwrap().to_string_lossy().into_owned();
        assert!(
            name.starts_with("a_tail_") && name.ends_with(".wav"),
            "{name}"
        );
        assert_eq!(out.parent(), src.parent());
    }

    #[test]
    fn keeps_original_samples_and_channels() {
        let dir = temp_root();
        let src = dir.path().join("stereo.wav");
        let original: Vec<i32> = (0..200).map(|i| (i - 100) * 40_000).collect(); // 100 フレームのステレオ、24bit の範囲内
        write_wav24(&src, 48000, 2, &original);
        let out = add_silence(&src, 0.5).unwrap();
        let (spec, samples) = read_wav(&out);
        assert_eq!(spec.channels, 2);
        assert_eq!(
            &samples[..200],
            &original[..],
            "original samples are bit-exact"
        );
        assert_eq!(samples.len(), 200 + 24000 * 2, "0.5 s of stereo frames");
    }

    #[test]
    fn converts_a_16_bit_source_to_24_bit() {
        let dir = temp_root();
        let src = dir.path().join("p16.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&src, spec).unwrap();
        for s in [1000i16, -2000, 3000] {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();
        let (spec, samples) = read_wav(&add_silence(&src, 0.001).unwrap());
        assert_eq!(spec.bits_per_sample, 24);
        assert_eq!(&samples[..3], &[1000 << 8, -2000 << 8, 3000 << 8]);
        assert_eq!(samples.len(), 3 + 16);
    }

    #[test]
    fn rejects_out_of_range_and_non_finite() {
        let dir = temp_root();
        let src = dir.path().join("a.wav");
        write_wav24(&src, 24000, 1, &[1]);
        for bad in [-1.0, 10.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = add_silence(&src, bad).unwrap_err();
            assert_eq!(
                (err.status(), err.message()),
                (400, "終了無音は0～10秒で指定してください。"),
                "{bad}"
            );
        }
        assert!(add_silence(&src, 10.0).is_ok(), "10 s is allowed");
    }

    #[test]
    fn unreadable_source_is_a_clear_error_and_leaves_no_partial_file() {
        let dir = temp_root();
        let src = dir.path().join("broken.wav");
        std::fs::write(&src, b"not a wav").unwrap();
        let err = add_silence(&src, 1.0).unwrap_err();
        assert_eq!(err.status(), 400);
        assert!(std::fs::read_dir(dir.path()).unwrap().all(|e| !e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains("_tail_")));
        assert!(add_silence(&dir.path().join("missing.wav"), 1.0).is_err());
    }
}
