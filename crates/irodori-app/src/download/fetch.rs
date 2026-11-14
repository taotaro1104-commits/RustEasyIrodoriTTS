//! 1 ファイルの取得。Range で再開し、検証してから rename する。リダイレクトは許可ホストだけへ、手で追う。

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use sha2::{Digest, Sha256};
use url::Url;

use super::hub::{host_allowed, map_ureq, redact_url};
use super::{DownloadError, Hub, PlannedFile};

const CHUNK: usize = 256 * 1024;
const MAX_REDIRECTS: usize = 5;

/// 取得の取消と進捗。`progress` は「新しく進んだバイト数」で、再開時は残っていた書きかけの大きさを 1 回通知する。
pub struct Transfer<'a> {
    pub cancel: &'a AtomicBool,
    pub progress: &'a dyn Fn(u64),
}

fn protocol(detail: impl Into<String>) -> DownloadError {
    DownloadError::Protocol(detail.into())
}

/// 書きかけのファイル（`<dest>.incomplete`）。
pub fn partial_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().unwrap_or_default().to_os_string();
    name.push(".incomplete");
    dest.with_file_name(name)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// 既にあるファイルの SHA-256 を、続きの計算のために状態へ取り込む。
fn hash_existing(
    path: &Path,
    hasher: &mut Sha256,
    cancel: &AtomicBool,
) -> Result<(), DownloadError> {
    let mut file = File::open(path)?;
    let mut buf = vec![0u8; CHUNK];
    loop {
        if cancel.load(Ordering::SeqCst) {
            return Err(DownloadError::Cancelled);
        }
        let n = file.read(&mut buf)?;
        if n == 0 {
            return Ok(());
        }
        hasher.update(&buf[..n]);
    }
}

fn discard(path: &Path) {
    let _ = std::fs::remove_file(path);
}

/// `Content-Range: bytes <start>-<end>/<total>` の開始位置。
fn range_start(value: &str) -> Option<u64> {
    value
        .strip_prefix("bytes ")?
        .split('-')
        .next()?
        .trim()
        .parse()
        .ok()
}

impl Hub {
    /// リダイレクトを（許可ホストへだけ）最大 5 回追って、本文の応答を得る。
    fn open(&self, url: &str, range_from: u64) -> Result<ureq::Response, DownloadError> {
        let mut current = url.to_string();
        for _ in 0..=MAX_REDIRECTS {
            if !host_allowed(&current, self.policy) {
                return Err(protocol(format!(
                    "host not allowed: {}",
                    redact_url(&current)
                )));
            }
            let mut request = self.agent.get(&current);
            if range_from > 0 {
                request = request.set("Range", &format!("bytes={range_from}-"));
            }
            let response = request.call().map_err(map_ureq)?;
            if matches!(response.status(), 301 | 302 | 303 | 307 | 308) {
                let location = response
                    .header("Location")
                    .ok_or_else(|| protocol("redirect without Location"))?;
                let base = Url::parse(&current).map_err(|e| protocol(e.to_string()))?;
                current = base
                    .join(location)
                    .map_err(|e| protocol(format!("bad redirect target: {e}")))?
                    .to_string();
                continue;
            }
            return Ok(response);
        }
        Err(protocol("too many redirects"))
    }

    /// `repo` の固定したコミット `sha` から `file` を取得し、`dest` へ置く。書きかけは `<dest>.incomplete`。
    pub fn fetch(
        &self,
        repo: &str,
        sha: &str,
        file: &PlannedFile,
        dest: &Path,
        transfer: &Transfer,
    ) -> Result<(), DownloadError> {
        if transfer.cancel.load(Ordering::SeqCst) {
            return Err(DownloadError::Cancelled);
        }
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let partial = partial_path(dest);
        let existing = usable_partial_len(&partial, file.size);
        let mut hasher = Sha256::new();

        if existing == file.size && existing > 0 {
            // 書きかけがすでに全部ある。通信せずに検証する。
            hash_existing(&partial, &mut hasher, transfer.cancel)?;
            (transfer.progress)(existing);
            return finish(&partial, dest, file, hasher);
        }

        let url = self.url(&format!("/{repo}/resolve/{sha}/{}", file.path));
        let (response, existing) = self.open_resuming(&url, existing, &partial)?;
        let resumed = is_resume(&response, existing);
        if response.status() != 200 && !resumed {
            return Err(DownloadError::Http(response.status()));
        }
        let expected_len = file.size - if resumed { existing } else { 0 };
        check_content_length(&response, expected_len, &file.path)?;

        let (mut out, written) = if resumed {
            hash_existing(&partial, &mut hasher, transfer.cancel)?;
            (transfer.progress)(existing);
            (OpenOptions::new().append(true).open(&partial)?, existing)
        } else {
            // 最初から（Range を無視された場合を含む）。
            (File::create(&partial)?, 0)
        };
        let copied = copy_body(
            response.into_reader(),
            &mut out,
            &mut hasher,
            written,
            file.size,
            transfer,
        );
        settle(copied, out, &partial, dest, file, hasher)
    }

    /// 書きかけの続き（`existing` バイト目から）を頼む。416（書きかけが取得元の想定と合わない）なら、捨てて最初から。
    fn open_resuming(
        &self,
        url: &str,
        existing: u64,
        partial: &Path,
    ) -> Result<(ureq::Response, u64), DownloadError> {
        match self.open(url, existing) {
            Err(DownloadError::Http(416)) if existing > 0 => {
                discard(partial);
                Ok((self.open(url, 0)?, 0))
            }
            other => Ok((other?, existing)),
        }
    }
}

/// 書き込みの結果を受けて、検証・rename・後始末をする。
fn settle(
    copied: Result<u64, DownloadError>,
    mut out: File,
    partial: &Path,
    dest: &Path,
    file: &PlannedFile,
    hasher: Sha256,
) -> Result<(), DownloadError> {
    match copied {
        Ok(total) => {
            out.flush()?;
            out.sync_all()?;
            drop(out);
            if total != file.size {
                // 最後まで来たのに大きさが違う。取得元の応答がおかしいので、書きかけは捨てる。
                discard(partial);
                return Err(DownloadError::Verification);
            }
            finish(partial, dest, file, hasher)
        }
        Err(DownloadError::Verification) => {
            // 計画より長い本文。開いたままでは消せないので閉じてから捨てる。
            drop(out);
            discard(partial);
            Err(DownloadError::Verification)
        }
        // 切断・取消は、再開のために書きかけを残す。
        Err(other) => Err(other),
    }
}

/// 使える書きかけの大きさ。通常のファイルで計画より小さいものだけ（それ以外は捨てて 0）。
fn usable_partial_len(partial: &Path, planned: u64) -> u64 {
    match std::fs::symlink_metadata(partial) {
        Ok(meta) if meta.is_file() && meta.len() <= planned => meta.len(),
        Ok(_) => {
            discard(partial);
            0
        }
        Err(_) => 0,
    }
}

/// 続きから（206 で、`Content-Range` の開始位置が書きかけの大きさと一致）の応答か。
fn is_resume(response: &ureq::Response, existing: u64) -> bool {
    existing > 0
        && response.status() == 206
        && response
            .header("Content-Range")
            .and_then(range_start)
            .is_some_and(|start| start == existing)
}

fn check_content_length(
    response: &ureq::Response,
    expected_len: u64,
    path: &str,
) -> Result<(), DownloadError> {
    match response
        .header("Content-Length")
        .and_then(|v| v.parse::<u64>().ok())
    {
        Some(len) if len != expected_len => Err(protocol(format!(
            "content length {len} differs from the planned {expected_len} for {path}"
        ))),
        _ => Ok(()),
    }
}

/// 本文を書きかけへ写す。計画の大きさ（`limit`）を 1 バイトでも超えたら、書かずに `Verification` を返す。
/// 戻り値は、書きかけの大きさ。切断・取消のときは、再開のために書きかけを残す。
fn copy_body(
    mut reader: impl Read,
    out: &mut File,
    hasher: &mut Sha256,
    mut written: u64,
    limit: u64,
    transfer: &Transfer,
) -> Result<u64, DownloadError> {
    let mut buf = vec![0u8; CHUNK];
    loop {
        if transfer.cancel.load(Ordering::SeqCst) {
            out.flush()?;
            return Err(DownloadError::Cancelled);
        }
        let n = match reader.read(&mut buf) {
            Ok(0) => return Ok(written),
            Ok(n) => n,
            Err(e) => {
                // 接続が切れた。書きかけは再開のために残す。取消の最中に切れたなら取消として扱う。
                let _ = out.flush();
                tracing::warn!("download interrupted: {e}");
                return Err(if transfer.cancel.load(Ordering::SeqCst) {
                    DownloadError::Cancelled
                } else {
                    DownloadError::Offline
                });
            }
        };
        if written + n as u64 > limit {
            return Err(DownloadError::Verification);
        }
        out.write_all(&buf[..n])?;
        hasher.update(&buf[..n]);
        written += n as u64;
        (transfer.progress)(n as u64);
    }
}

/// 大きさと（あれば）SHA-256 を確かめ、合っていれば rename する。合わなければ書きかけを消す。
fn finish(
    partial: &Path,
    dest: &Path,
    file: &PlannedFile,
    hasher: Sha256,
) -> Result<(), DownloadError> {
    let size = std::fs::metadata(partial)?.len();
    let hash_ok = match &file.sha256 {
        Some(expected) => hex(&hasher.finalize()) == *expected,
        None => true,
    };
    if size != file.size || !hash_ok {
        discard(partial);
        return Err(DownloadError::Verification);
    }
    std::fs::rename(partial, dest)?;
    Ok(())
}
