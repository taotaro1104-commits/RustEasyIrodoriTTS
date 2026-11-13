//! `/api/state`：画面が読むすべての情報を、v2.0（`Service.state`）と同じ形で組み立てる。

use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

use crate::assets::Assets;
use crate::dictionary::{load_lenient, Dictionary};
use crate::model::{Card, Project, Voice};
use crate::naming::filename;
use crate::signature::{playback_stamp, result_matches};
use crate::splitter::estimate;
use crate::{Result, Store, StoreError};

/// `easy_models.MODELS`。`installed` は取得済みかどうか（M2a では常に未取得）。
fn models() -> Value {
    json!([
        {"id": "Aratako/Irodori-TTS-v4.1-Small", "name": "標準", "steps": 40, "mf": false, "installed": false},
        {"id": "phasefield-audio/Irodori-TTS-v4.1-Anime", "name": "Anime", "steps": 40, "mf": false, "installed": false},
        {"id": "Aratako/Irodori-TTS-v4.1-Small-MF", "name": "MF（高速）", "steps": 4, "mf": true, "installed": false},
    ])
}

fn under(root: &Path, p: &str) -> PathBuf {
    let path = Path::new(p);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn base_name(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or(path).to_string()
}

/// 保存名の候補。書き出し済みならその名前、依頼の記録があれば規則どおりの名前、なければ「音声.wav」。
fn download_name(store: &Store, result: &Value) -> String {
    const DEFAULT: &str = "音声.wav";
    let request = result
        .get("request")
        .filter(|r| r.as_object().is_some_and(|o| !o.is_empty()));
    let suggested = request
        .and_then(|r| {
            let text = r.pointer("/card/text")?.as_str()?;
            let name = r.pointer("/voice/name")?.as_str()?;
            let ordinal = r.get("ordinal")?.as_i64()?;
            let date = r.get("date")?.as_str()?;
            Some(format!(
                "{}.wav",
                filename(&store.settings(), text, name, ordinal, date)
            ))
        })
        .unwrap_or_else(|| DEFAULT.to_string());
    let exported = result
        .pointer("/export/wav")
        .and_then(Value::as_str)
        .map(str::to_string);
    base_name(&exported.unwrap_or(suggested))
}

fn card_value(
    store: &Store,
    assets: &Assets,
    doc: &Project,
    card: &Card,
    dict: &Dictionary,
    voices: &[Voice],
) -> Value {
    let mut v = serde_json::to_value(card).unwrap_or(Value::Null);
    let Some(map) = v.as_object_mut() else {
        return v;
    };
    let rules = if doc.dictionary_enabled {
        dict.clone()
    } else {
        Dictionary::new()
    };
    let scale = doc
        .generation
        .get("duration_scale")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    map.insert(
        "estimate".into(),
        json!(estimate(&card.text, &rules, scale)),
    );
    let result = card.result.as_ref().filter(|r| !r.is_null());
    let audio = result
        .and_then(|r| r.get("path"))
        .and_then(Value::as_str)
        .and_then(|p| assets.register(&under(store.root(), p)));
    map.insert("audio".into(), json!(audio));
    map.insert(
        "download_name".into(),
        json!(result.map_or("音声.wav".to_string(), |r| download_name(store, r))),
    );
    let voice = voices.iter().find(|x| x.id == card.voice_id);
    let mut status = card.status.clone();
    if result.is_some() && !matches!(status.as_str(), "生成中" | "待機中") {
        if !result_matches(doc, card, dict, voice) {
            status = "再生成が必要".into();
        } else if audio.is_none() {
            status = "音声が見つかりません".into();
        }
    }
    let missing_name = card
        .extra
        .get("missing_voice_name")
        .is_some_and(|n| !n.is_null() && n != "");
    if (!card.voice_id.is_empty() && voice.is_none()) || (card.voice_id.is_empty() && missing_name)
    {
        status = "話者を選び直してください".into();
    }
    map.insert("status".into(), json!(status));
    v
}

pub fn build_state(
    store: &Store,
    assets: &Assets,
    pid: Option<&str>,
    jobs: &[Value],
    backend: &str,
) -> Result<Value> {
    let doc = store.load(pid)?;
    let dict = load_lenient(store.root())?;
    let voices = store.voices();
    let mut project =
        serde_json::to_value(&doc).map_err(|e| StoreError::Internal(e.to_string()))?;
    let cards: Vec<Value> = doc
        .cards
        .iter()
        .map(|c| card_value(store, assets, &doc, c, &dict, &voices))
        .collect();
    if let Some(m) = project.as_object_mut() {
        m.insert("cards".into(), Value::Array(cards));
    }
    let voice_values: Vec<Value> = voices
        .iter()
        .map(|vo| {
            let mut v = serde_json::to_value(vo).unwrap_or(Value::Null);
            if let Some(m) = v.as_object_mut() {
                m.insert(
                    "asset".into(),
                    json!(assets.register(&under(store.root(), &vo.audio))),
                );
            }
            v
        })
        .collect();
    let projects: Vec<Value> = store
        .list_projects()
        .iter()
        .map(|p| json!({"id": p.id, "name": p.name}))
        .collect();
    let (undo, redo) = store.history(&doc.id);
    let exists = |p: &str| under(store.root(), p).is_file();
    let mut out = Map::new();
    out.insert("project".into(), project);
    out.insert("projects".into(), Value::Array(projects));
    out.insert("voices".into(), Value::Array(voice_values));
    out.insert(
        "settings".into(),
        serde_json::to_value(store.settings()).map_err(|e| StoreError::Internal(e.to_string()))?,
    );
    out.insert("models".into(), models());
    out.insert(
        "dictionary".into(),
        serde_json::to_value(&dict).map_err(|e| StoreError::Internal(e.to_string()))?,
    );
    out.insert("jobs".into(), Value::Array(jobs.to_vec()));
    out.insert("backend".into(), json!(backend));
    out.insert("history".into(), json!({"undo": undo, "redo": redo}));
    out.insert(
        "playback_stamp".into(),
        json!(playback_stamp(&doc, &dict, &voices, &exists)),
    );
    Ok(Value::Object(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary;
    use crate::model::{Project, Voice};
    use crate::signature::signature;
    use crate::testutil::temp_root;
    use serde_json::json;

    fn voice(id: &str) -> Voice {
        serde_json::from_value(json!({"id": id, "revision": 1, "audio": format!("voices/{id}/a.wav"), "name": "話者A", "model": "m", "caption": ""})).unwrap()
    }

    /// 話者 1 人・カード 1 枚の台本。音声ファイルを作っておく。
    fn setup() -> (tempfile::TempDir, Store, Assets, String) {
        let dir = temp_root();
        let store = Store::open(dir.path()).unwrap();
        let v = voice(&"b".repeat(32));
        std::fs::create_dir_all(dir.path().join(format!("voices/{}", v.id))).unwrap();
        std::fs::write(dir.path().join(&v.audio), b"RIFF").unwrap();
        store.lock().voices.push(v.clone());
        let pid = store.load(None).unwrap().id;
        store
            .internal_update(&pid, &mut |d: &mut Project| {
                d.cards[0].text = "こんにちは".into();
                d.cards[0].voice_id = v.id.clone();
            })
            .unwrap();
        (dir, store, Assets::new(), pid)
    }

    /// 現行入力に一致する結果を持たせ、音声ファイルも置く。
    fn give_result(dir: &tempfile::TempDir, store: &Store, pid: &str, extra: Value) {
        let doc = store.load(Some(pid)).unwrap();
        let voices = store.voices();
        let sig = signature(
            &doc,
            &doc.cards[0],
            &dictionary::Dictionary::new(),
            voices.first(),
        );
        std::fs::create_dir_all(dir.path().join("outputs/internal/r")).unwrap();
        std::fs::write(dir.path().join("outputs/internal/r/01.wav"), b"RIFF").unwrap();
        let mut result = json!({"path": "outputs/internal/r/01.wav", "signature": sig});
        if let (Some(o), Some(e)) = (result.as_object_mut(), extra.as_object()) {
            o.extend(e.clone());
        }
        store
            .internal_update(pid, &mut |d: &mut Project| {
                d.cards[0].result = Some(result.clone());
                d.cards[0].status = "完了".into();
            })
            .unwrap();
    }

    fn card_of(state: &Value) -> &Value {
        &state["project"]["cards"][0]
    }

    #[test]
    fn state_has_every_field_the_editor_reads() {
        let (dir, store, assets, pid) = setup();
        give_result(&dir, &store, &pid, json!({}));
        let s = build_state(&store, &assets, Some(&pid), &[], "cpu").unwrap();
        for k in [
            "project",
            "projects",
            "voices",
            "settings",
            "models",
            "dictionary",
            "jobs",
            "backend",
            "history",
            "playback_stamp",
        ] {
            assert!(s.get(k).is_some(), "missing {k}");
        }
        for k in ["estimate", "audio", "download_name", "status"] {
            assert!(card_of(&s).get(k).is_some(), "card missing {k}");
        }
        assert!(s["voices"][0]["asset"].is_string());
        assert_eq!(s["backend"], "cpu");
        assert_eq!(s["history"], json!({"undo": false, "redo": false}));
        assert_eq!(s["models"].as_array().unwrap().len(), 3);
        assert_eq!(
            s["models"][2],
            json!({"id": "Aratako/Irodori-TTS-v4.1-Small-MF", "name": "MF（高速）", "steps": 4, "mf": true, "installed": false})
        );
        assert_eq!(s["projects"][0]["id"], pid.as_str());
        assert_eq!(s["playback_stamp"].as_str().unwrap().len(), 64);
        assert_eq!(card_of(&s)["status"], "完了");
        assert!(card_of(&s)["audio"].is_string());
    }

    #[test]
    fn jobs_are_passed_through() {
        let (_d, store, assets, pid) = setup();
        let jobs = vec![json!({"id": "j1", "stopping": false})];
        assert_eq!(
            build_state(&store, &assets, Some(&pid), &jobs, "cpu").unwrap()["jobs"],
            json!(jobs)
        );
    }

    #[test]
    fn card_without_result_has_defaults() {
        let (_d, store, assets, pid) = setup();
        let s = build_state(&store, &assets, Some(&pid), &[], "cpu").unwrap();
        let c = card_of(&s);
        assert_eq!(
            (
                c["audio"].clone(),
                c["download_name"].as_str(),
                c["status"].as_str()
            ),
            (Value::Null, Some("音声.wav"), Some("未生成"))
        );
        assert_eq!(c["estimate"], 0.8, "5 kana at 0.16 s each");
    }

    #[test]
    fn card_status_is_derived_like_python() {
        // 音声ファイルが無い
        let (dir, store, assets, pid) = setup();
        give_result(&dir, &store, &pid, json!({}));
        std::fs::remove_file(dir.path().join("outputs/internal/r/01.wav")).unwrap();
        assert_eq!(
            card_of(&build_state(&store, &assets, Some(&pid), &[], "cpu").unwrap())["status"],
            "音声が見つかりません"
        );
        // 話者が削除された
        let (dir, store, assets, pid) = setup();
        give_result(&dir, &store, &pid, json!({}));
        store.lock().voices.clear();
        assert_eq!(
            card_of(&build_state(&store, &assets, Some(&pid), &[], "cpu").unwrap())["status"],
            "話者を選び直してください"
        );
        // 話者なし＋消えた話者の名前
        let (_d, store, assets, pid) = setup();
        store
            .internal_update(&pid, &mut |d: &mut Project| {
                d.cards[0].voice_id.clear();
                d.cards[0]
                    .extra
                    .insert("missing_voice_name".into(), json!("消えた話者"));
            })
            .unwrap();
        assert_eq!(
            card_of(&build_state(&store, &assets, Some(&pid), &[], "cpu").unwrap())["status"],
            "話者を選び直してください"
        );
    }

    #[test]
    fn stale_result_shows_regeneration_needed() {
        let (dir, store, assets, pid) = setup();
        give_result(&dir, &store, &pid, json!({}));
        store
            .internal_update(&pid, &mut |d: &mut Project| {
                d.cards[0].text = "変えた文章".into()
            })
            .unwrap();
        assert_eq!(
            card_of(&build_state(&store, &assets, Some(&pid), &[], "cpu").unwrap())["status"],
            "再生成が必要"
        );
    }

    #[test]
    fn running_cards_keep_their_status_even_with_a_stale_result() {
        let (dir, store, assets, pid) = setup();
        give_result(&dir, &store, &pid, json!({}));
        store
            .internal_update(&pid, &mut |d: &mut Project| {
                d.cards[0].text = "変えた".into();
                d.cards[0].status = "生成中".into();
            })
            .unwrap();
        assert_eq!(
            card_of(&build_state(&store, &assets, Some(&pid), &[], "cpu").unwrap())["status"],
            "生成中"
        );
    }

    #[test]
    fn download_name_prefers_the_export_then_the_request_then_a_default() {
        let (dir, store, assets, pid) = setup();
        give_result(
            &dir,
            &store,
            &pid,
            json!({"export": {"wav": "outputs/exports/001_x.wav"}}),
        );
        assert_eq!(
            card_of(&build_state(&store, &assets, Some(&pid), &[], "cpu").unwrap())
                ["download_name"],
            "001_x.wav"
        );
        let (dir, store, assets, pid) = setup();
        let req = json!({"request": {"card": {"text": "こんにちは"}, "voice": {"name": "話者A"}, "ordinal": 3, "date": "20261113"}});
        give_result(&dir, &store, &pid, req);
        assert_eq!(
            card_of(&build_state(&store, &assets, Some(&pid), &[], "cpu").unwrap())
                ["download_name"],
            "003_話者A_20261113.wav"
        );
        let (dir, store, assets, pid) = setup();
        give_result(&dir, &store, &pid, json!({}));
        assert_eq!(
            card_of(&build_state(&store, &assets, Some(&pid), &[], "cpu").unwrap())
                ["download_name"],
            "音声.wav"
        );
    }

    #[test]
    fn dictionary_only_affects_estimates_when_enabled() {
        let (dir, store, assets, pid) = setup();
        dictionary::save(dir.path(), "こんにちは", &"あ".repeat(50)).unwrap();
        let with = build_state(&store, &assets, Some(&pid), &[], "cpu").unwrap();
        assert_eq!(with["dictionary"]["こんにちは"], "あ".repeat(50).as_str());
        assert_eq!(card_of(&with)["estimate"], 8.0);
        store
            .internal_update(&pid, &mut |d: &mut Project| d.dictionary_enabled = false)
            .unwrap();
        let without = build_state(&store, &assets, Some(&pid), &[], "cpu").unwrap();
        assert_ne!(card_of(&without)["estimate"], 8.0);
    }

    #[test]
    fn a_broken_dictionary_is_reported_as_an_error() {
        let (dir, store, assets, pid) = setup();
        std::fs::write(
            dir.path().join("config/reading_dictionary.json"),
            r#"{"a": 1}"#,
        )
        .unwrap();
        assert_eq!(
            build_state(&store, &assets, Some(&pid), &[], "cpu")
                .unwrap_err()
                .message(),
            "読み辞書の形式が不正です。"
        );
    }
}
