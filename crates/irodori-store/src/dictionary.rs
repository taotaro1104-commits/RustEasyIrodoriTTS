//! 読み辞書（`config/reading_dictionary.json`）。表記を読みに置き換えるだけで、元の文章は変えない。

use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use serde_json::Value;

use crate::py_json::dumps_pretty;
use crate::storage::atomic_write;
use crate::unicode_tables::{in_table, PY_SPACE};
use crate::{Result, StoreError};

/// 表記 → 読み。v2.0 のファイル順を保つ。
pub type Dictionary = IndexMap<String, String>;

const UNREADABLE: &str = "読み辞書を読み込めません。config/reading_dictionary.jsonを確認してください。元のファイルは変更していません。";
const BAD_FORMAT: &str = "読み辞書の形式が不正です。";

fn file(root: &Path) -> PathBuf {
    root.join("config/reading_dictionary.json")
}

/// Python の `str.strip()` と同じ空白判定で前後を削る。
fn py_strip(s: &str) -> &str {
    s.trim_matches(|c| in_table(PY_SPACE, c))
}

fn is_blank(s: &str) -> bool {
    py_strip(s).is_empty()
}

fn read_entries(root: &Path, message: &str, strict: bool) -> Result<Dictionary> {
    let path = file(root);
    if !path.exists() {
        return Ok(Dictionary::new());
    }
    let bad = || StoreError::Invalid(message.to_string());
    let bytes = std::fs::read(&path).map_err(|_| bad())?;
    let body = bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(&bytes);
    let raw: IndexMap<String, Value> = serde_json::from_slice(body).map_err(|_| bad())?;
    let mut out = Dictionary::new();
    for (k, v) in raw {
        let Value::String(v) = v else {
            return Err(bad());
        };
        let invalid = if strict {
            is_blank(&k) || is_blank(&v)
        } else {
            k.is_empty()
        };
        if invalid {
            return Err(bad());
        }
        out.insert(k, v);
    }
    Ok(out)
}

/// 厳密な読み込み（保存・置換用）。壊れていれば元のファイルに触れずエラー。
pub fn load(root: &Path) -> Result<Dictionary> {
    read_entries(root, UNREADABLE, true)
}

/// 画面の状態用。空白だけの表記なども通し、形式の誤りだけを知らせる。
pub fn load_lenient(root: &Path) -> Result<Dictionary> {
    read_entries(root, BAD_FORMAT, false)
}

fn write(root: &Path, entries: &Dictionary) -> Result<()> {
    let mut parts = Vec::with_capacity(entries.len());
    for (k, v) in entries {
        let key = dumps_pretty(&Value::String(k.clone()))?;
        let val = dumps_pretty(&Value::String(v.clone()))?;
        parts.push(format!("  {key}: {val}"));
    }
    let body = if parts.is_empty() {
        "{}".to_string()
    } else {
        format!("{{\n{}\n}}", parts.join(",\n"))
    };
    atomic_write(&file(root), &format!("{body}\n"))
}

pub fn save(root: &Path, word: &str, reading: &str) -> Result<&'static str> {
    let (word, reading) = (py_strip(word), py_strip(reading));
    if word.is_empty() || reading.is_empty() {
        return Err(StoreError::Invalid(
            "「表記」と「読み方」の両方を入力してください。".into(),
        ));
    }
    if word.chars().count() > 100
        || reading.chars().count() > 200
        || word.contains('\n')
        || reading.contains('\n')
    {
        return Err(StoreError::Invalid(
            "1行で入力してください。表記は100文字、読み方は200文字までです。".into(),
        ));
    }
    let mut entries = load(root)?;
    let updated = entries.contains_key(word);
    entries.insert(word.to_string(), reading.to_string());
    write(root, &entries)?;
    Ok(if updated {
        "読み方を更新しました。"
    } else {
        "読み方を登録しました。"
    })
}

pub fn delete(root: &Path, word: &str) -> Result<&'static str> {
    let mut entries = load(root)?;
    if entries.shift_remove(word).is_none() {
        return Err(StoreError::Invalid(
            "削除する登録を一覧から選んでください。".into(),
        ));
    }
    write(root, &entries)?;
    Ok("選んだ登録を削除しました。")
}

/// 長い表記から順（同じ長さは登録順）。長さは Python と同じくコードポイント数。
pub fn keys_longest_first(entries: &Dictionary) -> Vec<&str> {
    let mut keys: Vec<&str> = entries
        .keys()
        .map(String::as_str)
        .filter(|k| !k.is_empty())
        .collect();
    keys.sort_by_key(|k| std::cmp::Reverse(k.chars().count()));
    keys
}

/// 最長一致で置き換える。置き換えた結果は再処理しない。
pub fn apply(text: &str, entries: &Dictionary) -> String {
    let keys = keys_longest_first(entries);
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        match keys.iter().find(|k| rest.starts_with(**k)) {
            Some(k) => {
                out.push_str(&entries[*k]);
                rest = &rest[k.len()..];
            }
            None => {
                out.push(c);
                rest = &rest[c.len_utf8()..];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::temp_root;
    use serde_json::Value;

    const CASES: &str = include_str!("../tests/fixtures/dictionary_cases.json");
    const UNREADABLE: &str = "読み辞書を読み込めません。config/reading_dictionary.jsonを確認してください。元のファイルは変更していません。";

    fn path(dir: &tempfile::TempDir) -> std::path::PathBuf {
        dir.path().join("config/reading_dictionary.json")
    }

    #[test]
    fn apply_matches_python_for_every_case() {
        let cases: Vec<Value> = serde_json::from_str(CASES).unwrap();
        assert!(cases.len() >= 10);
        for c in cases {
            let entries: Dictionary = serde_json::from_value(c["entries"].clone()).unwrap();
            let text = c["text"].as_str().unwrap();
            assert_eq!(
                apply(text, &entries),
                c["output"].as_str().unwrap(),
                "text {text:?}"
            );
        }
    }

    #[test]
    fn save_update_delete_messages_follow_python() {
        let dir = temp_root();
        assert_eq!(
            save(dir.path(), "Irodori", "いろどり").unwrap(),
            "読み方を登録しました。"
        );
        assert_eq!(
            save(dir.path(), "Irodori", "イロドリ").unwrap(),
            "読み方を更新しました。"
        );
        assert_eq!(load(dir.path()).unwrap()["Irodori"], "イロドリ");
        assert_eq!(
            delete(dir.path(), "Irodori").unwrap(),
            "選んだ登録を削除しました。"
        );
        assert_eq!(
            delete(dir.path(), "Irodori").unwrap_err().message(),
            "削除する登録を一覧から選んでください。"
        );
        assert!(load(dir.path()).unwrap().is_empty());
    }

    #[test]
    fn save_trims_and_validates_input() {
        let dir = temp_root();
        save(dir.path(), "  語  ", "  よみ ").unwrap();
        assert_eq!(
            load(dir.path()).unwrap().get("語").map(String::as_str),
            Some("よみ")
        );
        let both = "「表記」と「読み方」の両方を入力してください。";
        assert_eq!(save(dir.path(), "", "x").unwrap_err().message(), both);
        assert_eq!(save(dir.path(), "x", "   ").unwrap_err().message(), both);
        let one_line = "1行で入力してください。表記は100文字、読み方は200文字までです。";
        assert_eq!(
            save(dir.path(), &"あ".repeat(101), "x")
                .unwrap_err()
                .message(),
            one_line
        );
        assert_eq!(
            save(dir.path(), "x", &"あ".repeat(201))
                .unwrap_err()
                .message(),
            one_line
        );
        assert_eq!(
            save(dir.path(), "a\nb", "x").unwrap_err().message(),
            one_line
        );
        assert_eq!(
            save(dir.path(), "x", "a\nb").unwrap_err().message(),
            one_line
        );
        assert!(save(dir.path(), &"あ".repeat(100), &"あ".repeat(200)).is_ok());
    }

    #[test]
    fn entries_keep_insertion_order_and_the_file_ends_with_a_newline() {
        let dir = temp_root();
        save(dir.path(), "b", "ビー").unwrap();
        save(dir.path(), "a", "エー").unwrap();
        let text = std::fs::read_to_string(path(&dir)).unwrap();
        assert!(text.ends_with("}\n"), "{text:?}");
        assert!(text.find("\"b\"").unwrap() < text.find("\"a\"").unwrap());
        let keys: Vec<String> = load(dir.path()).unwrap().keys().cloned().collect();
        assert_eq!(keys, ["b", "a"]);
    }

    #[test]
    fn corrupt_dictionary_is_not_overwritten() {
        let dir = temp_root();
        std::fs::write(path(&dir), b"{broken").unwrap();
        assert_eq!(load(dir.path()).unwrap_err().message(), UNREADABLE);
        assert_eq!(
            save(dir.path(), "x", "y").unwrap_err().message(),
            UNREADABLE
        );
        assert_eq!(std::fs::read(path(&dir)).unwrap(), b"{broken");
    }

    #[test]
    fn invalid_entries_are_refused_by_the_strict_loader() {
        for bad in [
            r#"["a"]"#,
            r#"{"a": 1}"#,
            r#"{"  ": "x"}"#,
            r#"{"a": "  "}"#,
        ] {
            let dir = temp_root();
            std::fs::write(path(&dir), bad).unwrap();
            assert_eq!(load(dir.path()).unwrap_err().message(), UNREADABLE, "{bad}");
        }
    }

    #[test]
    fn missing_dictionary_is_empty_and_bom_is_accepted() {
        let dir = temp_root();
        assert!(load(dir.path()).unwrap().is_empty());
        std::fs::write(path(&dir), "\u{feff}{\"a\": \"b\"}").unwrap();
        assert_eq!(load(dir.path()).unwrap()["a"], "b");
    }

    #[test]
    fn the_state_loader_uses_the_lenient_rules_and_message() {
        let dir = temp_root();
        std::fs::write(path(&dir), r#"{"  ": "x", "a": ""}"#).unwrap();
        assert_eq!(
            load_lenient(dir.path()).unwrap().len(),
            2,
            "state accepts what the editor tab would reject"
        );
        std::fs::write(path(&dir), r#"{"a": 1}"#).unwrap();
        assert_eq!(
            load_lenient(dir.path()).unwrap_err().message(),
            "読み辞書の形式が不正です。"
        );
        std::fs::write(path(&dir), r#"{"": "x"}"#).unwrap();
        assert_eq!(
            load_lenient(dir.path()).unwrap_err().message(),
            "読み辞書の形式が不正です。"
        );
    }
}
