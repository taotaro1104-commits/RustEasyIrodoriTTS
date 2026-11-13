//! `audio::decode_mono`（アップロード音声の取り込みに使う公開デコード）のテスト。

use std::path::{Path, PathBuf};

use irodori_core::audio::decode_mono;
use irodori_core::IrodoriError;

fn write_wav(path: &Path, channels: u16, rate: u32, frames: &[Vec<f32>]) {
    let spec = hound::WavSpec {
        channels,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for frame in frames {
        for s in frame {
            w.write_sample((s * 30000.0) as i16).unwrap();
        }
    }
    w.finalize().unwrap();
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn decodes_a_stereo_wav_to_mono_by_averaging() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("a.wav");
    write_wav(&p, 2, 8000, &vec![vec![0.5, -0.5]; 8000]);
    let (samples, rate) = decode_mono(&p, 10.0).unwrap();
    assert_eq!((samples.len(), rate), (8000, 8000));
    assert!(samples.iter().all(|s| s.abs() < 1e-3), "L=0.5,R=-0.5 averages to 0");
}

#[test]
fn averages_unequal_channels() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("b.wav");
    write_wav(&p, 2, 8000, &vec![vec![0.4, 0.0]; 800]);
    let (samples, _) = decode_mono(&p, 10.0).unwrap();
    // 0.4 は 16bit で 12000（/32768 で読み戻す）なので、左右の平均はその半分。
    let want = 12000.0 / 32768.0 / 2.0;
    assert!((samples[100] - want).abs() < 1e-3, "{}", samples[100]);
}

#[test]
fn stops_after_max_seconds() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("long.wav");
    write_wav(&p, 1, 8000, &vec![vec![0.3]; 8000 * 5]);
    let (samples, rate) = decode_mono(&p, 2.0).unwrap();
    assert_eq!(rate, 8000);
    assert!(samples.len() <= 2 * 8000, "{}", samples.len());
    assert!(samples.len() >= (1.9 * 8000.0) as usize, "{}", samples.len());
}

#[test]
fn reports_the_original_rate() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("c.wav");
    write_wav(&p, 1, 44100, &vec![vec![0.3]; 4410]);
    let (_, rate) = decode_mono(&p, 10.0).unwrap();
    assert_eq!(rate, 44100);
}

#[test]
fn garbage_bytes_are_an_audio_error() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("g.wav");
    let noise: Vec<u8> = (0..4096u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
    std::fs::write(&p, noise).unwrap();
    assert!(matches!(decode_mono(&p, 10.0), Err(IrodoriError::Audio(_))));
}

#[test]
fn empty_file_is_an_audio_error() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("e.wav");
    std::fs::write(&p, b"").unwrap();
    assert!(matches!(decode_mono(&p, 10.0), Err(IrodoriError::Audio(_))));
}

fn assert_one_second(name: &str) {
    let (samples, rate) = decode_mono(&fixture(name), 10.0).unwrap();
    let seconds = samples.len() as f32 / rate as f32;
    assert!((seconds - 1.0).abs() < 0.1, "{name}: {seconds}s at {rate}Hz");
}

#[test]
fn decodes_tiny_flac() {
    assert_one_second("tiny.flac");
}

#[test]
fn decodes_tiny_ogg() {
    assert_one_second("tiny.ogg");
}

#[test]
fn decodes_tiny_mp3() {
    assert_one_second("tiny.mp3");
}
