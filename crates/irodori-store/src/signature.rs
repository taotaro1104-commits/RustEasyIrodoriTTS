//! 生成結果の署名・一致判定・再生スタンプ（`easy_service.py` の `signature` など）。
//!
//! 署名が 1 文字でも違うと、v2.0 で作った音声が「古い」と判定されて作り直しになる。

use std::fmt::Write;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::dictionary::{apply, Dictionary};
use crate::model::{Card, Project, Voice};
use crate::py_json::dumps_sorted;

fn sha256_hex(text: &str) -> String {
    let digest = Sha256::digest(text.as_bytes());
    digest.iter().fold(String::with_capacity(64), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// 辞書を使う設定のときだけ読みに置き換えた文章。
pub fn reading_text(doc: &Project, card: &Card, dict: &Dictionary) -> String {
    if doc.dictionary_enabled {
        apply(&card.text, dict)
    } else {
        card.text.clone()
    }
}

/// `reading` は文字列。旧形式の署名だけ、当時の辞書全体（オブジェクト）が入る。
pub fn input_signature(
    doc: &Project,
    card: &Card,
    reading: &Value,
    voice: Option<&Voice>,
) -> String {
    let sound = voice.map_or(Value::Null, |v| json!(v.sound_revision_or_revision()));
    let value = json!([
        card.text,
        card.voice_id,
        sound,
        card.speed,
        card.silence,
        doc.generation,
        reading
    ]);
    sha256_hex(&dumps_sorted(&value))
}

pub fn signature(doc: &Project, card: &Card, dict: &Dictionary, voice: Option<&Voice>) -> String {
    input_signature(
        doc,
        card,
        &Value::String(reading_text(doc, card, dict)),
        voice,
    )
}

/// 保存された結果が、現在の入力内容に対応しているか。
pub fn result_matches(
    doc: &Project,
    card: &Card,
    dict: &Dictionary,
    voice: Option<&Voice>,
) -> bool {
    let Some(result) = card.result.as_ref().filter(|r| !is_falsy(r)) else {
        return false;
    };
    let Some(stored) = result.get("signature") else {
        return false;
    };
    if stored == &Value::String(signature(doc, card, dict, voice)) {
        return true;
    }
    // 以前の録音は辞書全体を署名に含めていた。
    let request = result.get("request");
    let Some(old_rules) = request
        .and_then(|r| r.get("dictionary"))
        .filter(|d| d.is_object())
    else {
        return false;
    };
    stored == &Value::String(input_signature(doc, card, old_rules, voice))
        && request.and_then(|r| r.get("text"))
            == Some(&Value::String(reading_text(doc, card, dict)))
}

pub(crate) fn is_falsy(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::Bool(b) => !b,
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
        Value::Number(n) => n.as_f64() == Some(0.0),
    }
}

/// Python の `json.dumps`（既定の `ensure_ascii=True`）と同じ形にする。
fn ascii_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            let mut units = [0u16; 2];
            for u in c.encode_utf16(&mut units) {
                let _ = write!(out, "\\u{u:04x}");
            }
        }
    }
    out
}

/// 画面が音声を取り直すべきかを判断するための印。カードごとの署名・結果の場所・状態・ファイルの有無から作る。
pub fn playback_stamp(
    doc: &Project,
    dict: &Dictionary,
    voices: &[Voice],
    file_exists: &dyn Fn(&str) -> bool,
) -> String {
    let values: Vec<Value> = doc
        .cards
        .iter()
        .map(|c| {
            let voice = voices.iter().find(|v| v.id == c.voice_id);
            let result = c.result.as_ref().filter(|r| !is_falsy(r));
            let path = result
                .and_then(|r| r.get("path"))
                .cloned()
                .unwrap_or(Value::Null);
            let busy = matches!(c.status.as_str(), "生成中" | "待機中" | "失敗");
            let exists = result
                .and_then(|r| r.get("path"))
                .and_then(Value::as_str)
                .is_some_and(file_exists);
            json!([
                c.id,
                signature(doc, c, dict, voice),
                path,
                if busy { c.status.as_str() } else { "" },
                exists
            ])
        })
        .collect();
    sha256_hex(&ascii_escape(&dumps_sorted(&Value::Array(values))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const CASES: &str = include_str!("../tests/fixtures/signature_cases.json");

    fn fixture() -> Value {
        serde_json::from_str(CASES).unwrap()
    }

    fn voices(f: &Value) -> Vec<Voice> {
        serde_json::from_value(f["voices"].clone()).unwrap()
    }

    fn with_result(card: &Card, result: Value) -> Card {
        Card {
            result: Some(result),
            ..card.clone()
        }
    }

    #[test]
    fn signature_matches_python_for_every_case() {
        let f = fixture();
        let voices = voices(&f);
        let cases = f["cases"].as_array().unwrap();
        assert!(cases.len() >= 60, "{}", cases.len());
        for c in cases {
            let doc: Project = serde_json::from_value(c["project"].clone()).unwrap();
            let dict: Dictionary = serde_json::from_value(c["dictionary"].clone()).unwrap();
            let card = &doc.cards[0];
            let voice = voices.iter().find(|v| v.id == card.voice_id);
            assert_eq!(
                reading_text(&doc, card, &dict),
                c["reading"].as_str().unwrap()
            );
            assert_eq!(
                signature(&doc, card, &dict, voice),
                c["signature"].as_str().unwrap(),
                "{:?}",
                card.text
            );
        }
    }

    #[test]
    fn result_matching_matches_python_including_legacy_forms() {
        let f = fixture();
        let voices = voices(&f);
        for c in f["cases"].as_array().unwrap() {
            let doc: Project = serde_json::from_value(c["project"].clone()).unwrap();
            let dict: Dictionary = serde_json::from_value(c["dictionary"].clone()).unwrap();
            let card = &doc.cards[0];
            let voice = voices.iter().find(|v| v.id == card.voice_id);
            let check = |name: &str| {
                let mut result = json!({"signature": c[name]["signature"]});
                if let Some(req) = c[name].get("request") {
                    result["request"] = req.clone();
                }
                result_matches(&doc, &with_result(card, result), &dict, voice)
            };
            let current = result_matches(
                &doc,
                &with_result(card, json!({"signature": c["signature"]})),
                &dict,
                voice,
            );
            assert_eq!(current, c["matches_current"].as_bool().unwrap());
            assert_eq!(
                check("legacy"),
                c["legacy"]["matches"].as_bool().unwrap(),
                "legacy {:?}",
                card.text
            );
            assert_eq!(
                check("legacy_wrong_text"),
                c["legacy_wrong_text"]["matches"].as_bool().unwrap()
            );
            assert_eq!(check("stale"), c["stale"]["matches"].as_bool().unwrap());
        }
    }

    #[test]
    fn legacy_result_signature_is_still_accepted() {
        let f = fixture();
        let voices = voices(&f);
        let accepted = f["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["legacy"]["matches"].as_bool() == Some(true))
            .count();
        assert!(accepted > 0, "fixture must contain accepted legacy results");
        let c = f["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["legacy"]["matches"].as_bool() == Some(true))
            .unwrap();
        let doc: Project = serde_json::from_value(c["project"].clone()).unwrap();
        let dict: Dictionary = serde_json::from_value(c["dictionary"].clone()).unwrap();
        let card = with_result(
            &doc.cards[0],
            json!({"signature": c["legacy"]["signature"], "request": c["legacy"]["request"]}),
        );
        let voice = voices.iter().find(|v| v.id == card.voice_id);
        assert!(result_matches(&doc, &card, &dict, voice));
    }

    #[test]
    fn missing_or_empty_result_never_matches() {
        let f = fixture();
        let doc: Project = serde_json::from_value(f["cases"][0]["project"].clone()).unwrap();
        let card = &doc.cards[0];
        assert!(!result_matches(&doc, card, &Dictionary::new(), None));
        for r in [json!(null), json!({}), json!({"path": "x.wav"})] {
            assert!(!result_matches(
                &doc,
                &with_result(card, r),
                &Dictionary::new(),
                None
            ));
        }
    }

    #[test]
    fn signature_changes_when_any_input_changes() {
        let f = fixture();
        let doc: Project = serde_json::from_value(f["cases"][0]["project"].clone()).unwrap();
        let voices = voices(&f);
        let dict: Dictionary = [("あ".to_string(), "い".to_string())].into_iter().collect();
        let base_card = Card {
            text: "あいう".into(),
            voice_id: voices[0].id.clone(),
            ..doc.cards[0].clone()
        };
        let base = signature(&doc, &base_card, &dict, Some(&voices[0]));
        let mut seen = vec![base.clone()];
        let mut check = |s: String| {
            assert!(!seen.contains(&s), "unchanged signature");
            seen.push(s);
        };
        check(signature(
            &doc,
            &Card {
                text: "あいうえ".into(),
                ..base_card.clone()
            },
            &dict,
            Some(&voices[0]),
        ));
        check(signature(
            &doc,
            &Card {
                voice_id: voices[1].id.clone(),
                ..base_card.clone()
            },
            &dict,
            Some(&voices[1]),
        ));
        check(signature(
            &doc,
            &Card {
                speed: 1.25,
                ..base_card.clone()
            },
            &dict,
            Some(&voices[0]),
        ));
        check(signature(
            &doc,
            &Card {
                silence: 2.0,
                ..base_card.clone()
            },
            &dict,
            Some(&voices[0]),
        ));
        let mut g = doc.clone();
        g.generation.insert("seed".into(), json!(9));
        check(signature(&g, &base_card, &dict, Some(&voices[0])));
        check(signature(
            &doc,
            &base_card,
            &Dictionary::new(),
            Some(&voices[0]),
        ));
        let newer = Voice {
            sound_revision: Some(99),
            ..voices[0].clone()
        };
        check(signature(&doc, &base_card, &dict, Some(&newer)));
        check(signature(&doc, &base_card, &dict, None));
        let off = Project {
            dictionary_enabled: false,
            ..doc.clone()
        };
        // 辞書を無効にすると、辞書が空のときと同じ読み（＝同じ署名）になる。
        assert_eq!(
            signature(&off, &base_card, &dict, Some(&voices[0])),
            signature(&doc, &base_card, &Dictionary::new(), Some(&voices[0]))
        );
    }

    #[test]
    fn playback_stamp_matches_python() {
        let f = fixture();
        let stamps = f["stamps"].as_array().unwrap();
        assert!(stamps.len() >= 3);
        for s in stamps {
            let doc: Project = serde_json::from_value(s["project"].clone()).unwrap();
            let dict: Dictionary = serde_json::from_value(s["dictionary"].clone()).unwrap();
            let voices: Vec<Voice> = serde_json::from_value(s["voices"].clone()).unwrap();
            let existing: Vec<String> = serde_json::from_value(s["existing"].clone()).unwrap();
            let exists = |p: &str| existing.iter().any(|e| e == p);
            assert_eq!(
                playback_stamp(&doc, &dict, &voices, &exists),
                s["stamp"].as_str().unwrap()
            );
        }
    }

    #[test]
    fn playback_stamp_changes_with_file_presence() {
        let f = fixture();
        let s = &f["stamps"][0];
        let doc: Project = serde_json::from_value(s["project"].clone()).unwrap();
        let voices: Vec<Voice> = serde_json::from_value(s["voices"].clone()).unwrap();
        let none = |_: &str| false;
        let all = |_: &str| true;
        assert_ne!(
            playback_stamp(&doc, &Dictionary::new(), &voices, &none),
            playback_stamp(&doc, &Dictionary::new(), &voices, &all)
        );
    }

    #[test]
    fn ascii_escape_uses_surrogate_pairs_like_python() {
        assert_eq!(ascii_escape("a日😀"), "a\\u65e5\\ud83d\\ude00");
    }
}
