//! テスト用の一時データフォルダ。

use tempfile::TempDir;

/// v2.0 と同じサブフォルダを持つ一時ルート。
pub fn temp_root() -> TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    for sub in ["projects", "voices", "config", "temp", "outputs"] {
        std::fs::create_dir_all(dir.path().join(sub)).expect("mkdir");
    }
    dir
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_root_has_the_v2_layout() {
        let d = temp_root();
        for sub in ["projects", "voices", "config", "temp", "outputs"] {
            assert!(d.path().join(sub).is_dir(), "{sub}");
        }
    }
}
