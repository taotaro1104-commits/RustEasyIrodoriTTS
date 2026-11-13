//! モデルの検出。許可リストのモデルだけを、HF キャッシュ互換のフォルダから探す（取得は別の段階）。
//!
//! 配置: `<hub>/models--<org>--<name>/snapshots/<sha>/{model.safetensors, tokenizer/tokenizer.json}`、
//! codec は `models--Aratako--Semantic-DACVAE-Japanese-32dim/snapshots/<sha>/weights.pth`。
//! v2.0 が使っていた `models/huggingface/hub` もそのまま使える。

use std::path::{Path, PathBuf};

/// 利用できるモデル（話者 JSON の任意の ID は扱わない）。
pub const ALLOW_LIST: [&str; 3] = [
    "Aratako/Irodori-TTS-v4.1-Small",
    "phasefield-audio/Irodori-TTS-v4.1-Anime",
    "Aratako/Irodori-TTS-v4.1-Small-MF",
];
/// 全モデル共通の codec。
pub const CODEC_ID: &str = "Aratako/Semantic-DACVAE-Japanese-32dim";

#[derive(Debug, Clone, PartialEq)]
pub struct ModelFiles {
    pub model: PathBuf,
    pub codec: PathBuf,
}

fn repo_dir(hub: &Path, id: &str) -> PathBuf {
    hub.join(format!("models--{}", id.replace('/', "--")))
}

/// `refs/main` に書かれたコミットの印。16 進の文字だけ（パスとして使うため）。
fn main_ref(repo: &Path) -> Option<String> {
    let text = std::fs::read_to_string(repo.join("refs").join("main")).ok()?;
    let sha = text.trim();
    (!sha.is_empty() && sha.len() <= 64 && sha.bytes().all(|b| b.is_ascii_hexdigit()))
        .then(|| sha.to_string())
}

/// 必要なファイルがそろった snapshot のフォルダ。`refs/main` を優先し、無ければ名前順で最初のもの。
fn snapshot_with(hub: &Path, id: &str, required: &[&str]) -> Option<PathBuf> {
    let repo = repo_dir(hub, id);
    let snapshots = repo.join("snapshots");
    let complete = |dir: &Path| required.iter().all(|f| dir.join(f).is_file());
    if let Some(sha) = main_ref(&repo) {
        let dir = snapshots.join(sha);
        if complete(&dir) {
            return Some(dir);
        }
    }
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs.into_iter().find(|d| complete(d))
}

/// モデルと codec がそろっていれば、そのファイルの場所。許可リスト外の ID は常に `None`。
pub fn locate(hub: &Path, model_id: &str) -> Option<ModelFiles> {
    if !ALLOW_LIST.contains(&model_id) {
        return None;
    }
    let model = snapshot_with(
        hub,
        model_id,
        &["model.safetensors", "tokenizer/tokenizer.json"],
    )?
    .join("model.safetensors");
    let codec = snapshot_with(hub, CODEC_ID, &["weights.pth"])?.join("weights.pth");
    Some(ModelFiles { model, codec })
}

/// すぐ使えるモデルの ID（許可リストの順）。
pub fn installed_ids(hub: &Path) -> Vec<&'static str> {
    ALLOW_LIST
        .iter()
        .copied()
        .filter(|id| locate(hub, id).is_some())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    const SMALL: &str = "Aratako/Irodori-TTS-v4.1-Small";
    const MF: &str = "Aratako/Irodori-TTS-v4.1-Small-MF";

    fn touch(path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"x").unwrap();
    }

    fn snapshot(hub: &Path, id: &str, sha: &str, files: &[&str]) {
        let dir = hub
            .join(format!("models--{}", id.replace('/', "--")))
            .join("snapshots")
            .join(sha);
        for f in files {
            touch(&dir.join(f));
        }
    }

    const MODEL_FILES: [&str; 2] = ["model.safetensors", "tokenizer/tokenizer.json"];

    fn complete_hub() -> tempfile::TempDir {
        let hub = tempfile::tempdir().unwrap();
        snapshot(hub.path(), SMALL, "aaaa1111", &MODEL_FILES);
        snapshot(hub.path(), CODEC_ID, "cccc3333", &["weights.pth"]);
        hub
    }

    #[test]
    fn locate_finds_a_complete_snapshot() {
        let hub = complete_hub();
        let files = locate(hub.path(), SMALL).expect("installed");
        assert!(
            files
                .model
                .ends_with("snapshots/aaaa1111/model.safetensors")
                || files
                    .model
                    .ends_with("snapshots\\aaaa1111\\model.safetensors")
        );
        assert!(files.codec.file_name().unwrap() == "weights.pth");
        assert!(files.model.exists() && files.codec.exists());
    }

    #[test]
    fn refs_main_wins_over_other_snapshots() {
        let hub = complete_hub();
        snapshot(hub.path(), SMALL, "bbbb2222", &MODEL_FILES);
        let refs = hub
            .path()
            .join(format!("models--{}", SMALL.replace('/', "--")))
            .join("refs");
        std::fs::create_dir_all(&refs).unwrap();
        std::fs::write(refs.join("main"), "bbbb2222\n").unwrap();
        assert!(locate(hub.path(), SMALL)
            .unwrap()
            .model
            .to_string_lossy()
            .contains("bbbb2222"));
    }

    #[test]
    fn a_refs_main_that_points_outside_or_nowhere_is_ignored() {
        let hub = complete_hub();
        let refs = hub
            .path()
            .join(format!("models--{}", SMALL.replace('/', "--")))
            .join("refs");
        std::fs::create_dir_all(&refs).unwrap();
        for bad in ["../../elsewhere", "ffff9999", "", "a/b"] {
            std::fs::write(refs.join("main"), bad).unwrap();
            assert!(
                locate(hub.path(), SMALL)
                    .unwrap()
                    .model
                    .to_string_lossy()
                    .contains("aaaa1111"),
                "ref {bad:?}"
            );
        }
    }

    #[test]
    fn missing_tokenizer_or_codec_is_not_installed() {
        let hub = tempfile::tempdir().unwrap();
        snapshot(hub.path(), SMALL, "aaaa1111", &["model.safetensors"]);
        snapshot(hub.path(), CODEC_ID, "cccc3333", &["weights.pth"]);
        assert!(locate(hub.path(), SMALL).is_none(), "tokenizer missing");
        let hub = tempfile::tempdir().unwrap();
        snapshot(hub.path(), SMALL, "aaaa1111", &MODEL_FILES);
        assert!(locate(hub.path(), SMALL).is_none(), "codec missing");
        assert!(locate(&hub.path().join("no-such-dir"), SMALL).is_none());
    }

    #[test]
    fn ids_outside_the_allow_list_are_never_located() {
        let hub = complete_hub();
        snapshot(hub.path(), "someone/else", "dddd4444", &MODEL_FILES);
        for id in [
            "someone/else",
            "aratako/irodori-tts-v4.1-small",
            "../Aratako/Irodori-TTS-v4.1-Small",
            "",
            CODEC_ID,
        ] {
            assert!(locate(hub.path(), id).is_none(), "{id:?}");
        }
    }

    #[test]
    fn installed_ids_lists_only_complete_models() {
        let hub = complete_hub();
        snapshot(hub.path(), MF, "eeee5555", &["model.safetensors"]);
        assert_eq!(installed_ids(hub.path()), [SMALL]);
        snapshot(hub.path(), MF, "eeee5555", &MODEL_FILES);
        assert_eq!(installed_ids(hub.path()), [SMALL, MF]);
    }
}
