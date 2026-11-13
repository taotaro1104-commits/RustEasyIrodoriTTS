//! 話者まわりの HTTP 経路（取り込み・登録・名前変更・削除・試作）。

mod common;

use axum::http::StatusCode;
use common::*;
use serde_json::{json, Value};

fn uploads_dir(f: &Fixture) -> std::path::PathBuf {
    f.dir.path().join("temp/uploads")
}

fn leftovers(f: &Fixture) -> Vec<String> {
    std::fs::read_dir(uploads_dir(f))
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default()
}

async fn upload(f: &Fixture, filename: &str, bytes: &[u8]) -> Reply {
    post_file(&f.app, "/api/upload", "file", filename, bytes, true).await
}

// ---- /api/upload ----

#[tokio::test]
async fn upload_returns_an_asset_and_seconds() {
    let f = fixture();
    let r = upload(&f, "声.wav", &wav_bytes(48000, 2, 2.0)).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.error());
    let body = r.json();
    assert!(
        (body["seconds"].as_f64().unwrap() - 2.0).abs() < 0.05,
        "{body}"
    );
    let key = body["asset"].as_str().unwrap();
    let audio = get(&f.app, &format!("/audio/{key}")).await;
    assert_eq!(audio.status, StatusCode::OK);
    let reader = hound::WavReader::new(std::io::Cursor::new(audio.body)).unwrap();
    let spec = reader.spec();
    assert_eq!(
        (spec.sample_rate, spec.channels, spec.bits_per_sample),
        (24000, 1, 24)
    );
    assert!((reader.duration() as f64 / 24000.0 - 2.0).abs() < 0.05);
    assert_eq!(
        leftovers(&f).len(),
        1,
        "only the normalized file stays: {:?}",
        leftovers(&f)
    );
}

#[tokio::test]
async fn upload_without_the_file_field_is_400() {
    let f = fixture();
    let r = post_file(
        &f.app,
        "/api/upload",
        "other",
        "a.wav",
        &wav_bytes(24000, 1, 2.0),
        true,
    )
    .await;
    assert_eq!(
        (r.status, r.error()),
        (
            StatusCode::BAD_REQUEST,
            "音声ファイルを選んでください。".into()
        )
    );
    assert!(leftovers(&f).is_empty());
}

#[tokio::test]
async fn upload_rejects_unknown_extension() {
    let f = fixture();
    for name in ["a.txt", "a", "a.exe", ".wav.txt"] {
        let r = upload(&f, name, &wav_bytes(24000, 1, 2.0)).await;
        assert_eq!(
            (r.status, r.error()),
            (
                StatusCode::BAD_REQUEST,
                "音声ファイルを選んでください。".into()
            ),
            "{name}"
        );
    }
    assert!(leftovers(&f).is_empty());
}

#[tokio::test]
async fn upload_of_webm_is_501_with_the_message() {
    let f = fixture();
    let r = upload(&f, "clip.WEBM", b"\x1a\x45\xdf\xa3").await;
    assert_eq!(r.status, StatusCode::NOT_IMPLEMENTED);
    assert!(r.error().contains("webm"), "{}", r.error());
    assert!(leftovers(&f).is_empty());
}

#[tokio::test]
async fn upload_rejects_garbage_with_a_wav_extension() {
    let f = fixture();
    let noise: Vec<u8> = (0..20000u32)
        .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8)
        .collect();
    let r = upload(&f, "a.wav", &noise).await;
    assert_eq!(
        (r.status, r.error()),
        (
            StatusCode::BAD_REQUEST,
            "音声を読み込めません。形式を確認してください。".into()
        )
    );
    assert!(leftovers(&f).is_empty(), "{:?}", leftovers(&f));
}

#[tokio::test]
async fn upload_rejects_empty_file() {
    let f = fixture();
    let r = upload(&f, "a.wav", b"").await;
    assert_eq!(
        (r.status, r.error()),
        (
            StatusCode::BAD_REQUEST,
            "音声を読み込めません。形式を確認してください。".into()
        )
    );
    assert!(leftovers(&f).is_empty());
}

#[tokio::test]
async fn upload_rejects_half_second_and_121_second_audio() {
    let f = fixture();
    for seconds in [0.5, 121.0, 200.0] {
        let r = upload(&f, "a.wav", &wav_bytes(8000, 1, seconds)).await;
        assert_eq!(
            (r.status, r.error()),
            (
                StatusCode::BAD_REQUEST,
                "音声は1～120秒にしてください。".into()
            ),
            "{seconds}"
        );
    }
    assert!(leftovers(&f).is_empty(), "{:?}", leftovers(&f));
    let ok = upload(&f, "a.wav", &wav_bytes(8000, 1, 120.0)).await;
    assert_eq!(ok.status, StatusCode::OK, "{}", ok.error());
}

#[tokio::test]
async fn upload_requires_the_token() {
    let f = fixture();
    let r = post_file(
        &f.app,
        "/api/upload",
        "file",
        "a.wav",
        &wav_bytes(24000, 1, 2.0),
        false,
    )
    .await;
    assert_eq!(r.status, StatusCode::FORBIDDEN);
    assert!(leftovers(&f).is_empty());
}

#[tokio::test]
async fn upload_does_not_use_the_client_file_name_for_the_path() {
    let f = fixture();
    let r = upload(&f, "..\\..\\evil.wav", &wav_bytes(24000, 1, 2.0)).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.error());
    assert!(!f.dir.path().join("evil.wav").exists());
    assert!(!f.dir.path().parent().unwrap().join("evil.wav").exists());
    let names = leftovers(&f);
    assert!(names.iter().all(|n| !n.contains("evil")), "{names:?}");
}

#[allow(dead_code)]
fn _unused(_: Value) {
    let _ = json!(null);
}
