//! v2.0（Python）が作った実データを Rust が読み書きして壊さないこと、Rust が書いたものを Python が読めることの確認。
//!
//! Python 環境が要るので `#[ignore]`。手順は `tools/compat/README.md`:
//!
//! ```text
//! IRODORI_COMPAT_DIR=<gen_fixtures.py --make-data-dir で作ったフォルダ>
//! IRODORI_UPSTREAM_DIR=<上流のリポジトリ（uv run の実行場所）>
//! cargo test -p irodori-store --test compat_roundtrip -- --ignored
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

use irodori_store::assets::Assets;
use irodori_store::dictionary::{self, Dictionary};
use irodori_store::signature::{result_matches, signature};
use irodori_store::state::build_state;
use irodori_store::{ops, Store};
use serde_json::{json, Value};

const BUSY_AFTER_LOAD: &str = "中断（再生成できます）";

fn env_dir(name: &str) -> PathBuf {
    let value = std::env::var_os(name)
        .unwrap_or_else(|| panic!("環境変数 {name} が必要です（tools/compat/README.md を参照）"));
    PathBuf::from(value)
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// Python が作ったフォルダを一時フォルダへ写す（元は変えない）。
struct Compat {
    dir: tempfile::TempDir,
    expected: Value,
}

fn compat() -> Compat {
    let source = env_dir("IRODORI_COMPAT_DIR");
    let dir = tempfile::tempdir().unwrap();
    copy_dir(&source, dir.path());
    let expected = read_json(&dir.path().join("compat_expected.json"));
    Compat { dir, expected }
}

impl Compat {
    fn root(&self) -> &Path {
        self.dir.path()
    }
    fn pid(&self) -> String {
        self.expected["project_id"].as_str().unwrap().to_string()
    }
    fn project_file(&self) -> PathBuf {
        self.root().join(format!("projects/{}.json", self.pid()))
    }
}

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn keys(v: &Value) -> Vec<String> {
    v.as_object()
        .map(|m| m.keys().cloned().collect())
        .unwrap_or_default()
}

#[test]
#[ignore = "needs IRODORI_COMPAT_DIR (Python-made data)"]
fn python_made_data_loads_edits_and_saves_without_losing_fields() {
    let c = compat();
    let pid = c.pid();
    let before = read_json(&c.project_file());
    let settings_before = read_json(&c.root().join("config/settings.json"));
    let voices_before = read_json(&c.root().join("voices/index.json"));

    let store = Store::open(c.root()).unwrap();
    assert_eq!(store.list_projects().len(), 2);
    let doc = store.load(Some(&pid)).unwrap();
    let first = doc.cards[0].id.clone();
    ops::edit(
        &store,
        &pid,
        doc.revision,
        &json!({"cards": [{"id": first, "text": "編集後の文章"}]}),
    )
    .unwrap();

    // 期待: 文章・revision・中断扱いの状態だけが変わり、未知の項目も含め他は一切そのまま。
    let mut expected = before.clone();
    expected["revision"] = json!(before["revision"].as_i64().unwrap() + 1);
    expected["cards"][0]["text"] = json!("編集後の文章");
    for card in expected["cards"].as_array_mut().unwrap() {
        if card["status"] == "生成中" {
            card["status"] = json!(BUSY_AFTER_LOAD);
        }
    }
    let after = read_json(&c.project_file());
    assert_eq!(after, expected);
    assert_eq!(
        after["future_project_field"],
        before["future_project_field"]
    );
    assert_eq!(after["cards"][0]["future_card_field"], "keep me");
    assert_eq!(keys(&after), keys(&before), "project key order");
    for (a, b) in after["cards"]
        .as_array()
        .unwrap()
        .iter()
        .zip(before["cards"].as_array().unwrap())
    {
        assert_eq!(keys(a), keys(b), "card key order");
    }
    assert_eq!(
        keys(&after["cards"][0]["result"]),
        keys(&before["cards"][0]["result"]),
        "result key order"
    );
    assert_eq!(
        keys(&after["generation"]),
        keys(&before["generation"]),
        "generation key order"
    );

    assert_eq!(
        read_json(&c.root().join("config/settings.json")),
        settings_before,
        "settings (with unknown keys) survive"
    );
    assert_eq!(
        read_json(&c.root().join("voices/index.json")),
        voices_before,
        "voices are not rewritten"
    );
    assert!(
        !std::fs::read_dir(c.root().join("projects"))
            .unwrap()
            .any(|e| e.unwrap().file_name().to_string_lossy().ends_with(".tmp")),
        "no temp files left"
    );
}

#[test]
#[ignore = "needs IRODORI_COMPAT_DIR (Python-made data)"]
fn signatures_of_python_results_still_match() {
    let c = compat();
    let store = Store::open(c.root()).unwrap();
    let doc = store.load(Some(&c.pid())).unwrap();
    let dict = dictionary::load_lenient(c.root()).unwrap();
    let voices = store.voices();
    let results = c.expected["results"].as_array().unwrap();
    assert!(results.len() >= 2);
    for r in results {
        let id = r["card_id"].as_str().unwrap();
        let card = doc.cards.iter().find(|x| x.id == id).unwrap();
        let voice = voices.iter().find(|v| v.id == card.voice_id);
        assert_eq!(
            signature(&doc, card, &dict, voice),
            r["signature"].as_str().unwrap(),
            "Python signature of {id}"
        );
        assert!(
            result_matches(&doc, card, &dict, voice),
            "result of {id} must still be accepted"
        );
    }
}

#[test]
#[ignore = "needs IRODORI_COMPAT_DIR (Python-made data)"]
fn state_for_python_data_has_no_regeneration_needed_for_unchanged_cards() {
    let c = compat();
    let store = Store::open(c.root()).unwrap();
    // 結果ファイルの相対パスは、データのルートを基準に解決される。
    let state = build_state(&store, &Assets::new(), Some(&c.pid()), &[], "cpu").unwrap();
    let cards = state["project"]["cards"].as_array().unwrap();
    let status = |id: &str| {
        cards.iter().find(|x| x["id"] == id).unwrap()["status"]
            .as_str()
            .unwrap()
            .to_string()
    };
    for r in c.expected["results"].as_array().unwrap() {
        assert_eq!(
            status(r["card_id"].as_str().unwrap()),
            "完了",
            "unchanged card keeps its result"
        );
    }
    assert_eq!(
        status(c.expected["busy_card_id"].as_str().unwrap()),
        BUSY_AFTER_LOAD
    );
    assert_eq!(
        status(c.expected["missing_voice_card_id"].as_str().unwrap()),
        "話者を選び直してください"
    );
    assert!(cards.iter().all(|x| x["status"] != "再生成が必要"));
    assert_eq!(state["dictionary"]["Irodori"], "いろどり");
    assert_eq!(state["settings"]["theme"], "dark");
    assert_eq!(state["voices"].as_array().unwrap().len(), 2);
    assert_eq!(state["playback_stamp"].as_str().unwrap().len(), 64);
}

#[test]
#[ignore = "needs IRODORI_COMPAT_DIR and IRODORI_UPSTREAM_DIR (Python environment)"]
fn rust_written_folder_is_readable_by_python() {
    let c = compat();
    let pid = c.pid();
    let store = Store::open(c.root()).unwrap();
    let doc = store.load(Some(&pid)).unwrap();
    let first = doc.cards[0].id.clone();
    let edited = ops::edit(&store, &pid, doc.revision, &json!({"name": "Rust で名前を変えた台本", "cards": [{"id": first, "text": "Rust が書いた文章 😀"}]})).unwrap();
    dictionary::save(c.root(), "新語", "しんご").unwrap();
    dictionary::save(c.root(), "日本橋", "にっぽんばし").unwrap(); // 既存の更新（並びは変わらない。結果つきのカードには出てこない語）
    let dict: Dictionary = dictionary::load(c.root()).unwrap();
    let unchanged: Vec<&str> = c.expected["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["card_id"].as_str().unwrap())
        .filter(|id| *id != first)
        .collect();
    let expect = json!({
        "pid": pid, "revision": edited.revision, "edited_card_id": first, "edited_text": "Rust が書いた文章 😀",
        "name": "Rust で名前を変えた台本", "busy_card_id": c.expected["busy_card_id"],
        "unchanged_result_cards": unchanged,
        "dictionary": dict.iter().map(|(k, v)| json!([k, v])).collect::<Vec<_>>(),
    });
    std::fs::write(c.root().join("compat_rust_expect.json"), expect.to_string()).unwrap();

    let script =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/compat/verify_python_reads.py");
    let output = Command::new("uv")
        .args(["run", "--python", "3.11", "--with", "regex", "python"])
        .arg(&script)
        .arg(c.root())
        .current_dir(env_dir("IRODORI_UPSTREAM_DIR"))
        .output()
        .expect("uv を実行できません");
    let (out, err) = (
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    assert!(
        output.status.success() && out.trim().ends_with("OK"),
        "Python could not read the Rust-written folder\nstdout: {out}\nstderr: {err}"
    );
}
