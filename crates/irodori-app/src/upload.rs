//! 音声の取り込み（`/api/upload`）。受け取った音声を 24 kHz・モノラル・24bit の WAV にそろえ、登録簿に載せて鍵を返す。
//!
//! 保存名にクライアントのファイル名は使わない（拡張子の判定にだけ使う）。失敗したときは、元ファイルも途中の出力も残さない。

use std::io::Write;
use std::path::{Path, PathBuf};

use axum::body::Bytes;
use axum::extract::multipart::{Field, Multipart, MultipartError};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use irodori_core::audio::{decode_mono, resample_mono, write_wav};
use irodori_store::model::uid;
use irodori_store::StoreError;
use serde_json::json;

use crate::errors::ApiError;
use crate::AppState;

/// 受け取る音声の大きさの上限。
pub const MAX_UPLOAD_BYTES: u64 = 100 * 1024 * 1024;
/// 取り込める音声の長さ（秒）。
const MIN_SECONDS: f64 = 1.0;
const MAX_SECONDS: f64 = 120.0;
/// 保存する WAV のレート。
const TARGET_RATE: u32 = 24_000;
const EXTENSIONS: [&str; 6] = ["wav", "mp3", "flac", "ogg", "m4a", "mp4"];

const PICK_MESSAGE: &str = "音声ファイルを選んでください。";
const UNREADABLE_MESSAGE: &str = "音声を読み込めません。形式を確認してください。";
const LENGTH_MESSAGE: &str = "音声は1～120秒にしてください。";
const SIZE_MESSAGE: &str = "音声は100MB以内にしてください。";
const WEBM_MESSAGE: &str =
    "未対応です（.webm は読み込めません。wav / mp3 / flac / ogg / m4a / mp4 を使ってください）。";

fn invalid(message: &str) -> StoreError {
    StoreError::Invalid(message.into())
}

/// 取り込みの途中で失敗したときに、置いたファイルを消す。`keep` したものだけ残る。
struct Cleanup(Vec<PathBuf>);

impl Cleanup {
    fn keep(&mut self, path: &Path) {
        self.0.retain(|p| p != path);
    }
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        for path in &self.0 {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// 本文を 1 かたまりずつ取り出せるもの（本物は multipart の項目、テストは配列）。
pub trait ChunkSource {
    #[allow(async_fn_in_trait)]
    async fn next_chunk(&mut self) -> Result<Option<Bytes>, StoreError>;
}

fn multipart_error(e: &MultipartError) -> StoreError {
    if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
        invalid(SIZE_MESSAGE)
    } else {
        invalid(UNREADABLE_MESSAGE)
    }
}

impl ChunkSource for Field<'_> {
    async fn next_chunk(&mut self) -> Result<Option<Bytes>, StoreError> {
        self.chunk().await.map_err(|e| multipart_error(&e))
    }
}

/// 本文を `dest` へ書く。合計が `max` を超えたら中断する（書きかけは呼び出し側が消す）。戻り値は書いた大きさ。
pub async fn receive<S: ChunkSource>(
    source: &mut S,
    dest: &Path,
    max: u64,
) -> Result<u64, StoreError> {
    let mut file = std::fs::File::create(dest)?;
    let mut total = 0u64;
    while let Some(chunk) = source.next_chunk().await? {
        total += chunk.len() as u64;
        if total > max {
            return Err(invalid(SIZE_MESSAGE));
        }
        file.write_all(&chunk)?;
    }
    file.sync_all()?;
    Ok(total)
}

/// 取り込んだ音声を 24 kHz・モノラル・24bit の WAV にして `out` へ書く。戻り値は秒数。
/// 1 秒未満・120 秒超は拒否する（121 秒までしか読まないので、長い音声を丸ごとは読まない）。
pub fn normalize(original: &Path, out: &Path) -> Result<f64, StoreError> {
    let (samples, rate) = decode_mono(original, (MAX_SECONDS + 1.0) as f32).map_err(|e| {
        tracing::info!("upload could not be decoded: {e}");
        invalid(UNREADABLE_MESSAGE)
    })?;
    let seconds = samples.len() as f64 / f64::from(rate);
    if !(MIN_SECONDS..=MAX_SECONDS).contains(&seconds) {
        return Err(invalid(LENGTH_MESSAGE));
    }
    let resampled = resample_mono(&samples, rate, TARGET_RATE).map_err(|e| {
        tracing::info!("upload could not be resampled: {e}");
        invalid(UNREADABLE_MESSAGE)
    })?;
    write_wav(out, &resampled, TARGET_RATE).map_err(|e| StoreError::Internal(e.to_string()))?;
    Ok(seconds)
}

/// ファイル名の拡張子（小文字、ドット無し）。区切り文字を含むものは拡張子と認めない。
fn extension_of(filename: &str) -> Option<String> {
    let (_, ext) = filename.rsplit_once('.')?;
    let ext = ext.to_lowercase();
    (!ext.is_empty() && ext.bytes().all(|b| b.is_ascii_alphanumeric())).then_some(ext)
}

fn not_implemented(message: &str) -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({ "error": message })),
    )
        .into_response()
}

pub async fn upload(
    State(app): State<AppState>,
    mut multipart: Multipart,
) -> Result<Response, ApiError> {
    loop {
        let Some(mut field) = multipart
            .next_field()
            .await
            .map_err(|e| ApiError(multipart_error(&e)))?
        else {
            return Err(invalid(PICK_MESSAGE).into());
        };
        if field.name() != Some("file") {
            continue;
        }
        let extension = extension_of(field.file_name().unwrap_or_default());
        match extension.as_deref() {
            Some("webm") => return Ok(not_implemented(WEBM_MESSAGE)),
            Some(ext) if EXTENSIONS.contains(&ext) => {}
            _ => return Err(invalid(PICK_MESSAGE).into()),
        }
        let folder = app.store.root().join("temp").join("uploads");
        std::fs::create_dir_all(&folder).map_err(StoreError::from)?;
        let id = uid();
        let original = folder.join(format!("{id}.{}", extension.unwrap_or_default()));
        let normalized = folder.join(format!("{id}.normalized.wav"));
        let mut cleanup = Cleanup(vec![original.clone(), normalized.clone()]);
        receive(&mut field, &original, MAX_UPLOAD_BYTES).await?;
        let (source, target) = (original.clone(), normalized.clone());
        let seconds = tokio::task::spawn_blocking(move || normalize(&source, &target))
            .await
            .map_err(|e| StoreError::Internal(format!("upload task failed: {e}")))??;
        let key = app
            .assets
            .register(&normalized)
            .ok_or_else(|| StoreError::Internal("could not register the upload".into()))?;
        cleanup.keep(&normalized);
        return Ok(Json(json!({ "asset": key, "seconds": seconds })).into_response());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_tone(path: &Path, rate: u32, seconds: f64) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        for i in 0..(f64::from(rate) * seconds) as usize {
            let v = (2.0 * std::f64::consts::PI * 440.0 * i as f64 / f64::from(rate)).sin() * 0.3;
            w.write_sample((v * 32767.0) as i16).unwrap();
        }
        w.finalize().unwrap();
    }

    struct Chunks(Vec<Bytes>);

    impl ChunkSource for Chunks {
        async fn next_chunk(&mut self) -> Result<Option<Bytes>, StoreError> {
            Ok(if self.0.is_empty() {
                None
            } else {
                Some(self.0.remove(0))
            })
        }
    }

    #[test]
    fn normalize_resamples_44100_to_24000() {
        let dir = tempfile::tempdir().unwrap();
        let (src, out) = (dir.path().join("a.wav"), dir.path().join("o.wav"));
        write_tone(&src, 44100, 2.0);
        let seconds = normalize(&src, &out).unwrap();
        assert!((seconds - 2.0).abs() < 0.01, "{seconds}");
        let r = hound::WavReader::open(&out).unwrap();
        assert_eq!(
            (
                r.spec().sample_rate,
                r.spec().channels,
                r.spec().bits_per_sample
            ),
            (24000, 1, 24)
        );
        assert!(
            (r.duration() as f64 - 48000.0).abs() <= 2.0,
            "{}",
            r.duration()
        );
    }

    #[test]
    fn normalize_rejects_short_long_and_unreadable_audio() {
        let dir = tempfile::tempdir().unwrap();
        let (src, out) = (dir.path().join("a.wav"), dir.path().join("o.wav"));
        for seconds in [0.5, 121.0] {
            write_tone(&src, 8000, seconds);
            let e = normalize(&src, &out).unwrap_err();
            assert_eq!(e.message(), LENGTH_MESSAGE, "{seconds}");
        }
        std::fs::write(&src, b"junk").unwrap();
        assert_eq!(
            normalize(&src, &out).unwrap_err().message(),
            UNREADABLE_MESSAGE
        );
        assert!(!out.exists());
    }

    #[test]
    fn extensions_are_taken_from_the_last_dot_only() {
        assert_eq!(extension_of("a.WAV").as_deref(), Some("wav"));
        assert_eq!(extension_of("a.b.mp3").as_deref(), Some("mp3"));
        assert_eq!(extension_of("noext"), None);
        assert_eq!(extension_of("trailing."), None);
        assert_eq!(extension_of("a.w/av"), None);
    }

    #[tokio::test]
    async fn receive_writes_all_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("x.bin");
        let mut src = Chunks(vec![Bytes::from_static(b"abc"), Bytes::from_static(b"de")]);
        assert_eq!(receive(&mut src, &dest, 10).await.unwrap(), 5);
        assert_eq!(std::fs::read(&dest).unwrap(), b"abcde");
    }

    #[tokio::test]
    async fn upload_over_the_limit_is_rejected_without_writing_past_it() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("x.bin");
        let mut src = Chunks(vec![
            Bytes::from_static(b"abcd"),
            Bytes::from_static(b"efgh"),
        ]);
        let e = receive(&mut src, &dest, 6).await.unwrap_err();
        assert_eq!(e.message(), SIZE_MESSAGE);
        assert!(std::fs::metadata(&dest).unwrap().len() <= 6);
    }

    #[test]
    fn cleanup_removes_everything_except_what_was_kept() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = (dir.path().join("a"), dir.path().join("b"));
        std::fs::write(&a, b"1").unwrap();
        std::fs::write(&b, b"2").unwrap();
        {
            let mut guard = Cleanup(vec![a.clone(), b.clone()]);
            guard.keep(&b);
        }
        assert!(!a.exists() && b.exists());
    }
}
