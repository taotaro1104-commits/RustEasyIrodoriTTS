//! モデルと codec の導入。HF キャッシュ互換のフォルダへ置き、各リポジトリの最後に `refs/main` を書く。

use std::fs::{File, TryLockError};
use std::path::{Path, PathBuf};

use irodori_store::storage::atomic_write;

use super::{DownloadError, Hub, RepoPlan, Transfer};
use crate::models::{self, ALLOW_LIST, CODEC_ID};

/// 画面で確認するための概算（実際の大きさは取得計画で分かる）。
pub const APPROX_MODEL_BYTES: u64 = 3_100_000_000;
pub const APPROX_CODEC_BYTES: u64 = 430_000_000;
/// 取得するファイル（固定。サーバが返す一覧は、ここにあるものだけを使う）。
pub const MODEL_FILES: [&str; 3] = models::MODEL_REQUIRED_FILES;
pub const CODEC_FILES: [&str; 1] = ["weights.pth"];
/// 必要な空き容量の余裕（残りのバイト数に掛ける）。
const SPACE_MARGIN: f64 = 1.2;

/// このモデルを使えるようにするのに、これから取得するバイト数の概算。取得済みなら 0、許可外も 0。
pub fn approximate_bytes(hub_dir: &Path, model_id: &str) -> u64 {
    if !ALLOW_LIST.contains(&model_id) {
        return 0;
    }
    let model = if models::model_present(hub_dir, model_id) {
        0
    } else {
        APPROX_MODEL_BYTES
    };
    let codec = if models::codec_present(hub_dir) {
        0
    } else {
        APPROX_CODEC_BYTES
    };
    model + codec
}

/// 実際の空き容量（`fs4`）で導入する。空き容量を調べられなければ（ネットワークドライブ等）、警告を残して進める。
pub fn install_model(
    hub: &Hub,
    hub_dir: &Path,
    model_id: &str,
    transfer: &Transfer,
    on_total: &dyn Fn(u64),
) -> Result<(), DownloadError> {
    install_with_space(hub, hub_dir, model_id, transfer, on_total, &|path| {
        fs4::available_space(path).unwrap_or_else(|e| {
            tracing::warn!("cannot read the free disk space, continuing: {e}");
            u64::MAX
        })
    })
}

/// 空き容量の調べ方を差し込める版（テスト用）。`available` は「そのフォルダのあるドライブの空き」を返す。
pub fn install_with_space(
    hub: &Hub,
    hub_dir: &Path,
    model_id: &str,
    transfer: &Transfer,
    on_total: &dyn Fn(u64),
    available: &dyn Fn(&Path) -> u64,
) -> Result<(), DownloadError> {
    if !ALLOW_LIST.contains(&model_id) {
        return Err(DownloadError::Protocol(format!(
            "model not allowed: {model_id:?}"
        )));
    }
    // 同じ置き場への取得は 1 つだけ（二重起動で書きかけが混ざらないように）。終わるまで持つ。
    let _lock = acquire_lock(hub_dir)?;
    let mut wanted: Vec<(&str, &[&str])> = Vec::new();
    if !models::model_present(hub_dir, model_id) {
        wanted.push((model_id, &MODEL_FILES));
    }
    if !models::codec_present(hub_dir) {
        wanted.push((CODEC_ID, &CODEC_FILES));
    }
    if wanted.is_empty() {
        on_total(0);
        return Ok(());
    }
    let mut plans: Vec<RepoPlan> = Vec::new();
    for (repo, files) in wanted {
        plans.push(hub.plan(repo, files)?);
    }

    for plan in &plans {
        remove_stale_partials(hub_dir, plan);
    }
    let (missing, kept) = measure(hub_dir, &plans);
    on_total(missing);
    let needed = ((missing - kept) as f64 * SPACE_MARGIN).ceil() as u64;
    let free = available(&nearest_existing_dir(hub_dir));
    if needed > free {
        return Err(DownloadError::DiskSpace {
            needed,
            available: free,
        });
    }

    for plan in &plans {
        fetch_repo(hub, hub_dir, plan, transfer)?;
    }
    Ok(())
}

/// まだ完成していないファイルの合計と、そのうち書きかけとして残っている分。
fn measure(hub_dir: &Path, plans: &[RepoPlan]) -> (u64, u64) {
    let (mut missing, mut kept) = (0u64, 0u64);
    for plan in plans {
        for file in &plan.files {
            let dest = snapshot_dir(hub_dir, plan).join(&file.path);
            if is_complete(&dest, file.size) {
                continue;
            }
            missing += file.size;
            kept += std::fs::metadata(super::partial_path(&dest))
                .map(|m| m.len().min(file.size))
                .unwrap_or(0);
        }
    }
    (missing, kept)
}

/// 1 つのリポジトリの未完成のファイルを取得し、全部が検証済みになってから最後に `refs/main` を書く。
fn fetch_repo(
    hub: &Hub,
    hub_dir: &Path,
    plan: &RepoPlan,
    transfer: &Transfer,
) -> Result<(), DownloadError> {
    let snapshot = snapshot_dir(hub_dir, plan);
    for file in &plan.files {
        let dest = snapshot.join(&file.path);
        if !is_complete(&dest, file.size) {
            hub.fetch(&plan.repo, &plan.sha, file, &dest, transfer)?;
        }
    }
    let refs = models::repo_dir(hub_dir, &plan.repo)
        .join("refs")
        .join("main");
    atomic_write(
        &refs,
        &format!(
            "{}
",
            plan.sha
        ),
    )
    .map_err(|e| DownloadError::Io(e.message().to_string()))
}

fn snapshot_dir(hub_dir: &Path, plan: &RepoPlan) -> PathBuf {
    models::repo_dir(hub_dir, &plan.repo)
        .join("snapshots")
        .join(&plan.sha)
}

/// 通常のファイルで、大きさが合うか（シンボリックリンクは完成とみなさない）。
fn is_complete(dest: &Path, size: u64) -> bool {
    std::fs::symlink_metadata(dest).is_ok_and(|m| m.is_file() && m.len() == size)
}

/// `<hub>/.locks/` は HF のキャッシュも使う置き場。ここにアプリ専用のロックを置く。
fn acquire_lock(hub_dir: &Path) -> Result<File, DownloadError> {
    let dir = hub_dir.join(".locks");
    std::fs::create_dir_all(&dir)?;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(dir.join("irodori-app-download.lock"))?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(TryLockError::WouldBlock) => Err(DownloadError::Locked),
        Err(TryLockError::Error(e)) => Err(e.into()),
    }
}

/// 別のコミットの書きかけ（`*.incomplete`）は、再開に使えず空き容量を圧迫するので消す。
fn remove_stale_partials(hub_dir: &Path, plan: &RepoPlan) {
    let snapshots = models::repo_dir(hub_dir, &plan.repo).join("snapshots");
    let Ok(entries) = std::fs::read_dir(&snapshots) else {
        return;
    };
    for entry in entries.flatten() {
        if entry.file_name() == plan.sha.as_str() {
            continue;
        }
        remove_partials_under(&entry.path());
    }
}

fn remove_partials_under(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_dir() {
            remove_partials_under(&path);
        } else if path.extension().is_some_and(|x| x == "incomplete") {
            if let Err(e) = std::fs::remove_file(&path) {
                tracing::warn!("cannot remove a stale partial file: {e}");
            }
        }
    }
}

/// まだ作っていないフォルダでも、書き込み先のドライブを調べられるように、実在する最も近い親を返す。
fn nearest_existing_dir(path: &Path) -> PathBuf {
    let mut current = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    while !current.is_dir() {
        match current.parent() {
            Some(parent) => current = parent.to_path_buf(),
            None => break,
        }
    }
    current
}
