//! 音声ファイルの登録簿。画面にはパスではなく、解決済みパスの SHA-256 を鍵として渡す。

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use sha2::{Digest, Sha256};

use crate::{Result, StoreError};

#[derive(Default)]
pub struct Assets {
    paths: Mutex<HashMap<String, PathBuf>>,
}

/// Windows の `\\?\` 接頭辞を外す（Python の `Path.resolve()` と同じ見た目にする）。
fn plain(path: PathBuf) -> PathBuf {
    let text = path.to_string_lossy();
    match text.strip_prefix(r"\\?\") {
        Some(rest) if !rest.starts_with("UNC\\") => PathBuf::from(rest),
        _ => path,
    }
}

/// 鍵の元になる、解決済みの絶対パス。`canonicalize` できない場所（一部のネットワーク共有など）でも、
/// ファイルが実在すれば絶対パスで代用する。
pub fn resolve(path: &Path) -> Option<PathBuf> {
    resolve_using(path, |p| p.canonicalize())
}

fn resolve_using(
    path: &Path,
    canonicalize: impl Fn(&Path) -> std::io::Result<PathBuf>,
) -> Option<PathBuf> {
    match canonicalize(path) {
        Ok(resolved) => Some(plain(resolved)),
        Err(_) if path.is_file() => std::path::absolute(path).ok(),
        Err(_) => None,
    }
}

pub fn key_for(resolved: &Path) -> String {
    let digest = Sha256::digest(resolved.to_string_lossy().as_bytes());
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

impl Assets {
    pub fn new() -> Assets {
        Assets::default()
    }

    /// 存在するファイルだけ登録して鍵を返す。
    pub fn register(&self, path: &Path) -> Option<String> {
        if path.as_os_str().is_empty() || !path.is_file() {
            return None;
        }
        let resolved = resolve(path)?;
        let key = key_for(&resolved);
        self.paths
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key.clone(), resolved);
        Some(key)
    }

    /// 鍵から実ファイルを引く。登録後に消えていればエラー。
    pub fn source(&self, key: &str) -> Result<PathBuf> {
        let found = self
            .paths
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(key)
            .cloned();
        match found {
            Some(p) if p.is_file() => Ok(p),
            _ => Err(StoreError::Invalid(
                "音声が見つかりません。もう一度追加してください。".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::temp_root;

    #[test]
    fn resolve_falls_back_to_an_absolute_path_when_canonicalize_fails() {
        let dir = temp_root();
        let file = dir.path().join("a.wav");
        std::fs::write(&file, b"x").unwrap();
        let failing = |_: &Path| -> std::io::Result<PathBuf> {
            Err(std::io::Error::other("no canonicalize here"))
        };
        let got = resolve_using(&file, failing).unwrap();
        assert!(got.is_absolute() && got.is_file(), "{got:?}");
        assert_eq!(
            resolve_using(&dir.path().join("missing.wav"), failing),
            None,
            "a missing file is still not resolved"
        );
    }

    #[test]
    fn assets_register_only_existing_files() {
        let dir = temp_root();
        let assets = Assets::new();
        let file = dir.path().join("a.wav");
        std::fs::write(&file, b"x").unwrap();
        assert!(assets.register(&file).is_some());
        assert!(assets.register(&dir.path().join("missing.wav")).is_none());
        assert!(
            assets.register(dir.path()).is_none(),
            "directories are not audio files"
        );
        assert!(assets.register(Path::new("")).is_none());
    }

    #[test]
    fn asset_keys_are_stable_sha256_of_the_resolved_path() {
        let dir = temp_root();
        let assets = Assets::new();
        let file = dir.path().join("a.wav");
        std::fs::write(&file, b"x").unwrap();
        let key = assets.register(&file).unwrap();
        assert_eq!(key.len(), 64);
        assert_eq!(assets.register(&file), Some(key.clone()));
        let roundabout = dir.path().join("config").join("..").join("a.wav");
        assert_eq!(
            assets.register(&roundabout),
            Some(key.clone()),
            "the key follows the resolved path"
        );
        assert_eq!(key, key_for(&resolve(&file).unwrap()));
        assert_eq!(assets.source(&key).unwrap(), resolve(&file).unwrap());
    }

    #[test]
    fn unknown_asset_key_is_an_error_with_the_python_message() {
        let assets = Assets::new();
        let err = assets.source("nope").unwrap_err();
        assert_eq!(
            (err.status(), err.message()),
            (400, "音声が見つかりません。もう一度追加してください。")
        );
    }

    #[test]
    fn a_registered_file_that_disappears_is_reported() {
        let dir = temp_root();
        let assets = Assets::new();
        let file = dir.path().join("gone.wav");
        std::fs::write(&file, b"x").unwrap();
        let key = assets.register(&file).unwrap();
        std::fs::remove_file(&file).unwrap();
        assert!(assets.source(&key).is_err());
    }
}
