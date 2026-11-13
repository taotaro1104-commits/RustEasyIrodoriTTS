//! 台本の編集・操作（`easy_service.edit/action/split_doc`、`easy_web.project/recover_previous`）。

use std::collections::{HashMap, HashSet};

use serde_json::Value;

use crate::dictionary::{py_strip, Dictionary};
use crate::generation::{number, validate_generation};
use crate::model::{new_card, new_project, Card, Project};
use crate::py_json::float_repr;
use crate::signature::is_falsy;
use crate::splitter::split_text;
use crate::{Result, Store, StoreError};

pub const MAX_CARDS: usize = 1000;
pub const MAX_TEXT_CHARS: usize = 100_000;
const TOO_MANY_CARDS: &str =
    "セリフは1000枚までです。不要なセリフを削除するか、台本を分けてください。";

#[derive(Debug, Clone, PartialEq)]
pub struct ActionOutcome {
    pub message: String,
}

fn invalid(msg: &str) -> StoreError {
    StoreError::Invalid(msg.into())
}

/// Python の `str(value)` に近い文字列化（画面は文字列しか送らないが、型が違っても落とさない）。
fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) => n
            .as_f64()
            .filter(|_| n.is_f64())
            .map_or_else(|| n.to_string(), float_repr),
        other => other.to_string(),
    }
}

fn cut(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn apply_edit(doc: &mut Project, values: &Value) -> Result<()> {
    if let Some(name) = values.get("name") {
        let name = cut(py_strip(&py_str(name)), 100);
        doc.name = if name.is_empty() {
            "無題の台本".into()
        } else {
            name
        };
    }
    if let Some(sel) = values.get("selected").and_then(Value::as_str) {
        if doc.cards.iter().any(|c| c.id == sel) {
            doc.selected = sel.to_string();
        }
    }
    if let Some(v) = values.get("dictionary_enabled") {
        doc.dictionary_enabled = !is_falsy(v);
    }
    if let Some(g) = values.get("generation") {
        doc.generation = validate_generation(g)?;
    }
    let patches = values
        .get("cards")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    for patch in patches {
        let id = patch
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("編集対象のカードを指定してください。"))?;
        let Some(card) = doc.cards.iter_mut().find(|c| c.id == id) else {
            return Err(StoreError::Conflict(
                "編集対象が削除されています。台本を読み直してください。".into(),
            ));
        };
        patch_card(card, patch)?;
    }
    Ok(())
}

fn patch_card(card: &mut Card, patch: &Value) -> Result<()> {
    if let Some(v) = patch.get("text") {
        card.text = py_str(v);
    }
    if let Some(v) = patch.get("voice_id") {
        card.voice_id = py_str(v);
    }
    if let Some(v) = patch.get("filename") {
        card.filename = py_str(v);
    }
    if card.text.chars().count() > MAX_TEXT_CHARS {
        return Err(invalid("1回の入力は10万文字以内にしてください。"));
    }
    if let Some(v) = patch.get("speed") {
        card.speed = number(v, 0.75, 1.5, "speed", "0.75", "1.5")?;
    }
    if let Some(v) = patch.get("silence") {
        card.silence = number(v, 0.0, 10.0, "silence", "0", "10")?;
    }
    Ok(())
}

pub fn edit(store: &Store, pid: &str, revision: i64, values: &Value) -> Result<Project> {
    store.update(pid, revision, true, &mut |doc| apply_edit(doc, values))
}

/// 長い本文を分けて、続きのカードを直後に足す。足した枚数を返す。
fn split_doc(doc: &mut Project, dict: &Dictionary, cid: Option<&str>) -> Result<usize> {
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
    let (mut count, mut cards) = (0, Vec::with_capacity(doc.cards.len()));
    for card in &doc.cards {
        let mut card = card.clone();
        let target = cid.is_none() || cid == Some(card.id.as_str());
        let pieces = if target {
            split_text(&card.text, &rules, scale)?
        } else {
            vec![card.text.clone()]
        };
        if pieces.len() > 1 {
            card.text = pieces[0].clone();
            let tail: Vec<Card> = pieces[1..]
                .iter()
                .map(|p| Card {
                    text: p.clone(),
                    ..new_card(Some(&card))
                })
                .collect();
            cards.push(card);
            count += tail.len();
            cards.extend(tail);
        } else {
            cards.push(card);
        }
    }
    if cards.len() > MAX_CARDS {
        return Err(invalid("台本は1000枚以内に分けて保存してください。"));
    }
    doc.cards = cards;
    Ok(count)
}

fn reorder(doc: &mut Project, order: Option<&[String]>) -> Result<()> {
    let bad = || invalid("並べ替え対象が変わりました。画面を読み直してください。");
    let order = order.ok_or_else(bad)?;
    let ids: HashSet<&str> = doc.cards.iter().map(|c| c.id.as_str()).collect();
    let wanted: HashSet<&str> = order.iter().map(String::as_str).collect();
    if order.len() != doc.cards.len() || ids != wanted {
        return Err(bad());
    }
    let mut by_id: HashMap<String, Card> = doc.cards.drain(..).map(|c| (c.id.clone(), c)).collect();
    doc.cards = order.iter().filter_map(|id| by_id.remove(id)).collect();
    Ok(())
}

fn apply_action(
    doc: &mut Project,
    inherit: bool,
    dict: &Dictionary,
    action: &str,
    cid: Option<&str>,
    order: Option<&[String]>,
) -> Result<String> {
    let index = cid.and_then(|id| doc.cards.iter().position(|c| c.id == id));
    if matches!(action, "add" | "duplicate") && doc.cards.len() >= MAX_CARDS {
        return Err(invalid(TOO_MANY_CARDS));
    }
    match action {
        "reorder" => reorder(doc, order)?,
        "add" => {
            let source = doc.cards.iter().find(|c| c.id == doc.selected).cloned();
            let card = new_card(if inherit { source.as_ref() } else { None });
            doc.selected = card.id.clone();
            doc.cards.push(card);
        }
        "reset" => {
            let card = new_card(None);
            doc.selected = card.id.clone();
            doc.cards = vec![card];
        }
        "split" => {
            let n = split_doc(doc, dict, cid)?;
            return Ok(if n > 0 {
                format!("続きのカードを{n}枚追加しました。")
            } else {
                "分割は不要です。".into()
            });
        }
        _ => {
            let Some(i) = index else {
                return Err(invalid("カードを選んでください。"));
            };
            match action {
                "delete" => {
                    doc.cards.remove(i);
                    if doc.cards.is_empty() {
                        doc.cards.push(new_card(None));
                    }
                    doc.selected = doc.cards[i.min(doc.cards.len() - 1)].id.clone();
                }
                "duplicate" => {
                    let copy = Card {
                        text: doc.cards[i].text.clone(),
                        ..new_card(Some(&doc.cards[i]))
                    };
                    doc.selected = copy.id.clone();
                    doc.cards.insert(i + 1, copy);
                }
                "up" | "down" => {
                    let target = if action == "up" {
                        i.checked_sub(1)
                    } else {
                        Some(i + 1)
                    };
                    if let Some(t) = target.filter(|t| *t < doc.cards.len()) {
                        doc.cards.swap(i, t);
                    }
                }
                _ => return Err(invalid("操作を認識できません。")),
            }
        }
    }
    Ok(String::new())
}

/// `active` は、まだ動いている生成依頼に残っているカード ID（元に戻した後に「停止」へ変えない対象）。
#[allow(clippy::too_many_arguments)]
pub fn action(
    store: &Store,
    dict: &Dictionary,
    active: &HashSet<String>,
    pid: &str,
    revision: i64,
    action: &str,
    cid: Option<&str>,
    order: Option<&[String]>,
) -> Result<ActionOutcome> {
    if matches!(action, "undo" | "redo") {
        let redo = action == "redo";
        store.travel(pid, revision, redo)?;
        store.internal_update(pid, &mut |d: &mut Project| {
            for c in d.cards.iter_mut().filter(|c| {
                matches!(c.status.as_str(), "待機中" | "生成中") && !active.contains(&c.id)
            }) {
                c.status = "停止".into();
            }
        })?;
        let message = if redo {
            "やり直しました。"
        } else {
            "元に戻しました。"
        };
        return Ok(ActionOutcome {
            message: message.into(),
        });
    }
    let inherit = store.settings().inherit;
    let mut message = String::new();
    store.update(pid, revision, true, &mut |doc| {
        message = apply_action(doc, inherit, dict, action, cid, order)?;
        Ok(())
    })?;
    Ok(ActionOutcome { message })
}

/// 台本を開く（`load`）か、新しく作る（`name`、`copy` なら `pid` の内容を引き継ぐ）。
pub fn new_or_load_project(store: &Store, req: &Value) -> Result<Project> {
    let pid = req
        .get("pid")
        .and_then(Value::as_str)
        .filter(|p| !p.is_empty());
    let truthy = |k: &str| req.get(k).is_some_and(|v| !is_falsy(v));
    if truthy("load") {
        let doc = store.load(Some(pid.ok_or_else(|| invalid("台本を選んでください。"))?))?;
        store.set_last_project(&doc.id)?;
        return Ok(doc);
    }
    let name = req
        .get("name")
        .map_or_else(|| "新しい台本".to_string(), |n| cut(&py_str(n), 100));
    let mut doc = new_project(&name);
    if truthy("copy") {
        let old = store.load(Some(pid.ok_or_else(|| invalid("台本を選んでください。"))?))?;
        doc.cards = old.cards;
        doc.selected = old.selected;
        doc.generation = old.generation;
        doc.dictionary_enabled = old.dictionary_enabled;
    }
    store.persist(&doc)?;
    Ok(doc)
}

/// 前回起動時の入力（`settings.recovery_project`）を、いまの台本へ戻す。生成中は拒否する。
pub fn recover_previous(store: &Store, pid: &str, revision: i64, busy: bool) -> Result<Project> {
    let previous = store
        .settings()
        .extra
        .get("recovery_project")
        .and_then(Value::as_str)
        .filter(|p| !p.is_empty())
        .map(str::to_string);
    let Some(previous) = previous else {
        return Err(invalid("復元できる前回の入力がありません。"));
    };
    if busy {
        return Err(invalid("生成が終わってから復元してください。"));
    }
    let old = store.load(Some(&previous))?;
    store.update(pid, revision, true, &mut |d| {
        d.cards = old.cards.clone();
        d.generation = old.generation.clone();
        d.dictionary_enabled = old.dictionary_enabled;
        d.selected = old.selected.clone();
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{new_card, Card};
    use crate::testutil::temp_root;
    use serde_json::json;

    fn setup() -> (tempfile::TempDir, Store, String) {
        let dir = temp_root();
        let store = Store::open(dir.path()).unwrap();
        let pid = store.load(None).unwrap().id;
        (dir, store, pid)
    }

    fn doc(store: &Store, pid: &str) -> Project {
        store.load(Some(pid)).unwrap()
    }

    /// 本文だけ違うカードを並べた台本にする（履歴には積まない）。
    fn with_cards(store: &Store, pid: &str, texts: &[&str]) -> Vec<String> {
        let cards: Vec<Card> = texts
            .iter()
            .map(|t| Card {
                text: (*t).into(),
                ..new_card(None)
            })
            .collect();
        let ids: Vec<String> = cards.iter().map(|c| c.id.clone()).collect();
        store
            .internal_update(pid, &mut |d: &mut Project| {
                d.cards = cards.clone();
                d.selected = cards[0].id.clone();
            })
            .unwrap();
        ids
    }

    fn run(store: &Store, pid: &str, name: &str, cid: Option<&str>) -> Result<ActionOutcome> {
        let rev = doc(store, pid).revision;
        action(
            store,
            &Dictionary::new(),
            &HashSet::new(),
            pid,
            rev,
            name,
            cid,
            None,
        )
    }

    fn texts(store: &Store, pid: &str) -> Vec<String> {
        doc(store, pid)
            .cards
            .iter()
            .map(|c| c.text.clone())
            .collect()
    }

    #[test]
    fn edit_updates_cards_and_bumps_revision() {
        let (_d, store, pid) = setup();
        let before = doc(&store, &pid);
        let id = before.cards[0].id.clone();
        let values = json!({
            "name": "  新しい名前  ", "dictionary_enabled": false, "selected": id,
            "generation": {"steps": 4},
            "cards": [{"id": id, "text": "あいう", "voice_id": "v", "filename": "f", "speed": "1.25", "silence": 2}],
        });
        let after = edit(&store, &pid, before.revision, &values).unwrap();
        assert_eq!(after.revision, before.revision + 1);
        assert_eq!(
            (after.name.as_str(), after.dictionary_enabled),
            ("新しい名前", false)
        );
        assert_eq!(after.generation["steps"], json!(4));
        let c = &after.cards[0];
        assert_eq!(
            (
                c.text.as_str(),
                c.voice_id.as_str(),
                c.filename.as_str(),
                c.speed,
                c.silence
            ),
            ("あいう", "v", "f", 1.25, 2.0)
        );
        assert_eq!(doc(&store, &pid), after, "persisted");
        assert!(store.history(&pid).0, "an edit can be undone");
    }

    #[test]
    fn edit_name_is_trimmed_cut_and_defaulted() {
        let (_d, store, pid) = setup();
        let p = edit(&store, &pid, 0, &json!({"name": "   "})).unwrap();
        assert_eq!(p.name, "無題の台本");
        let p = edit(&store, &pid, p.revision, &json!({"name": "あ".repeat(150)})).unwrap();
        assert_eq!(p.name.chars().count(), 100);
        let p = edit(
            &store,
            &pid,
            p.revision,
            &json!({"selected": "no-such-card"}),
        )
        .unwrap();
        assert_eq!(p.selected, p.cards[0].id, "unknown selection is ignored");
    }

    #[test]
    fn edit_with_stale_revision_conflicts_and_changes_nothing() {
        let (_d, store, pid) = setup();
        let id = doc(&store, &pid).cards[0].id.clone();
        edit(
            &store,
            &pid,
            0,
            &json!({"cards": [{"id": id, "text": "A"}]}),
        )
        .unwrap();
        let before = doc(&store, &pid);
        let err = edit(
            &store,
            &pid,
            0,
            &json!({"cards": [{"id": id, "text": "B"}]}),
        )
        .unwrap_err();
        assert_eq!(err.status(), 409);
        assert_eq!(doc(&store, &pid), before);
    }

    #[test]
    fn edit_of_a_deleted_card_conflicts() {
        let (_d, store, pid) = setup();
        let err = edit(
            &store,
            &pid,
            0,
            &json!({"cards": [{"id": "gone", "text": "x"}]}),
        )
        .unwrap_err();
        assert_eq!(
            (err.status(), err.message()),
            (
                409,
                "編集対象が削除されています。台本を読み直してください。"
            )
        );
    }

    #[test]
    fn text_over_100000_chars_is_rejected_and_state_is_unchanged() {
        let (_d, store, pid) = setup();
        let before = doc(&store, &pid);
        let id = before.cards[0].id.clone();
        let err = edit(
            &store,
            &pid,
            0,
            &json!({"cards": [{"id": id, "text": "あ".repeat(100_001)}]}),
        )
        .unwrap_err();
        assert_eq!(
            (err.status(), err.message()),
            (400, "1回の入力は10万文字以内にしてください。")
        );
        assert_eq!(doc(&store, &pid), before);
        edit(
            &store,
            &pid,
            0,
            &json!({"cards": [{"id": id, "text": "あ".repeat(100_000)}]}),
        )
        .unwrap();
    }

    #[test]
    fn speed_and_silence_out_of_range_are_rejected() {
        let (_d, store, pid) = setup();
        let id = doc(&store, &pid).cards[0].id.clone();
        let msg = |card: Value| {
            edit(&store, &pid, 0, &json!({"cards": [card]}))
                .unwrap_err()
                .message()
                .to_string()
        };
        assert_eq!(
            msg(json!({"id": id, "speed": 1.6})),
            "speedは0.75～1.5で指定してください。"
        );
        assert_eq!(
            msg(json!({"id": id, "speed": 0.7})),
            "speedは0.75～1.5で指定してください。"
        );
        assert_eq!(
            msg(json!({"id": id, "silence": 11})),
            "silenceは0～10で指定してください。"
        );
        assert_eq!(
            msg(json!({"id": id, "silence": -1})),
            "silenceは0～10で指定してください。"
        );
        assert_eq!(
            msg(json!({"id": id, "speed": "abc"})),
            "speedは0.75～1.5で指定してください。"
        );
        assert!(edit(
            &store,
            &pid,
            0,
            &json!({"cards": [{"id": id, "speed": 0.75, "silence": 10}]})
        )
        .is_ok());
    }

    #[test]
    fn invalid_generation_is_rejected_without_saving() {
        let (_d, store, pid) = setup();
        let before = doc(&store, &pid);
        assert!(edit(
            &store,
            &pid,
            0,
            &json!({"name": "x", "generation": {"steps": 999}})
        )
        .is_err());
        assert_eq!(doc(&store, &pid), before);
    }

    #[test]
    fn add_inherits_voice_speed_silence_when_enabled() {
        let (_d, store, pid) = setup();
        store
            .internal_update(&pid, &mut |d: &mut Project| {
                d.cards[0].voice_id = "v1".into();
                d.cards[0].speed = 1.25;
                d.cards[0].silence = 2.0;
                d.cards[0].text = "本文は引き継がない".into();
            })
            .unwrap();
        run(&store, &pid, "add", None).unwrap();
        let p = doc(&store, &pid);
        assert_eq!(p.cards.len(), 2);
        let c = &p.cards[1];
        assert_eq!(
            (c.voice_id.as_str(), c.speed, c.silence, c.text.as_str()),
            ("v1", 1.25, 2.0, "")
        );
        assert_eq!(p.selected, c.id);
        store.lock().settings.inherit = false;
        run(&store, &pid, "add", None).unwrap();
        let c = &doc(&store, &pid).cards[2];
        assert_eq!((c.voice_id.as_str(), c.speed, c.silence), ("", 1.0, 0.0));
    }

    #[test]
    fn delete_last_card_leaves_one_empty_card() {
        let (_d, store, pid) = setup();
        let first = doc(&store, &pid).cards[0].clone();
        run(&store, &pid, "delete", Some(&first.id)).unwrap();
        let p = doc(&store, &pid);
        assert_eq!(p.cards.len(), 1);
        assert_ne!(p.cards[0].id, first.id);
        assert_eq!(
            (p.cards[0].text.as_str(), p.selected.as_str()),
            ("", p.cards[0].id.as_str())
        );
    }

    #[test]
    fn delete_selects_the_card_now_at_that_position() {
        let (_d, store, pid) = setup();
        let ids = with_cards(&store, &pid, &["a", "b", "c"]);
        run(&store, &pid, "delete", Some(&ids[1])).unwrap();
        assert_eq!(
            (texts(&store, &pid), doc(&store, &pid).selected),
            (vec!["a".to_string(), "c".to_string()], ids[2].clone())
        );
        run(&store, &pid, "delete", Some(&ids[2])).unwrap();
        assert_eq!(
            doc(&store, &pid).selected,
            ids[0],
            "deleting the last card selects the new last"
        );
    }

    #[test]
    fn duplicate_up_down_move_cards_as_python() {
        let (_d, store, pid) = setup();
        let ids = with_cards(&store, &pid, &["a", "b", "c"]);
        store
            .internal_update(&pid, &mut |d: &mut Project| {
                d.cards[1].status = "完了".into()
            })
            .unwrap();
        run(&store, &pid, "duplicate", Some(&ids[1])).unwrap();
        let p = doc(&store, &pid);
        assert_eq!(texts(&store, &pid), ["a", "b", "b", "c"]);
        assert_eq!(p.selected, p.cards[2].id);
        assert_ne!(p.cards[2].id, ids[1]);
        assert_eq!(
            p.cards[2].status, "未生成",
            "a duplicate starts without a result"
        );
        run(&store, &pid, "up", Some(&p.cards[2].id)).unwrap();
        assert_eq!(doc(&store, &pid).cards[1].id, p.cards[2].id);
        let rev = doc(&store, &pid).revision;
        run(&store, &pid, "up", Some(&ids[0])).unwrap();
        assert_eq!(
            doc(&store, &pid).revision,
            rev,
            "moving the first card up changes nothing"
        );
        run(&store, &pid, "down", Some(&ids[0])).unwrap();
        assert_eq!(doc(&store, &pid).cards[1].id, ids[0]);
        let last = doc(&store, &pid).cards.last().unwrap().id.clone();
        let rev = doc(&store, &pid).revision;
        run(&store, &pid, "down", Some(&last)).unwrap();
        assert_eq!(doc(&store, &pid).revision, rev);
    }

    #[test]
    fn reorder_requires_exactly_the_same_ids() {
        let (_d, store, pid) = setup();
        let ids = with_cards(&store, &pid, &["a", "b", "c"]);
        let go = |order: Vec<String>| {
            let rev = doc(&store, &pid).revision;
            action(
                &store,
                &Dictionary::new(),
                &HashSet::new(),
                &pid,
                rev,
                "reorder",
                None,
                Some(&order),
            )
        };
        go(vec![ids[2].clone(), ids[0].clone(), ids[1].clone()]).unwrap();
        assert_eq!(texts(&store, &pid), ["c", "a", "b"]);
        let msg = "並べ替え対象が変わりました。画面を読み直してください。";
        for bad in [
            vec![ids[0].clone()],
            vec![ids[0].clone(), ids[1].clone(), "x".into()],
            vec![ids[0].clone(), ids[0].clone(), ids[1].clone()],
        ] {
            assert_eq!(go(bad).unwrap_err().message(), msg);
        }
        let rev = doc(&store, &pid).revision;
        let err = action(
            &store,
            &Dictionary::new(),
            &HashSet::new(),
            &pid,
            rev,
            "reorder",
            None,
            None,
        )
        .unwrap_err();
        assert_eq!(err.message(), msg);
    }

    #[test]
    fn more_than_1000_cards_is_rejected() {
        let (_d, store, pid) = setup();
        let many: Vec<Card> = (0..1000).map(|_| new_card(None)).collect();
        store
            .internal_update(&pid, &mut |d: &mut Project| {
                d.cards = many.clone();
                d.selected = many[0].id.clone();
            })
            .unwrap();
        let msg = "セリフは1000枚までです。不要なセリフを削除するか、台本を分けてください。";
        assert_eq!(run(&store, &pid, "add", None).unwrap_err().message(), msg);
        let first = doc(&store, &pid).cards[0].id.clone();
        assert_eq!(
            run(&store, &pid, "duplicate", Some(&first))
                .unwrap_err()
                .message(),
            msg
        );
        assert_eq!(doc(&store, &pid).cards.len(), 1000);
        // 分割で増える場合
        let mut cards: Vec<Card> = (0..999).map(|_| new_card(None)).collect();
        cards.push(Card {
            text: "あ。".repeat(200),
            ..new_card(None)
        });
        store
            .internal_update(&pid, &mut |d: &mut Project| d.cards = cards.clone())
            .unwrap();
        let err = run(&store, &pid, "split", None).unwrap_err();
        assert_eq!(err.message(), "台本は1000枚以内に分けて保存してください。");
        assert_eq!(doc(&store, &pid).cards.len(), 1000, "nothing was split");
    }

    #[test]
    fn split_creates_continuation_cards_and_reports_the_count() {
        let (_d, store, pid) = setup();
        let ids = with_cards(&store, &pid, &["短い", &"あ。".repeat(100)]);
        store
            .internal_update(&pid, &mut |d: &mut Project| {
                d.cards[1].voice_id = "v9".into();
            })
            .unwrap();
        let out = run(&store, &pid, "split", Some(&ids[1])).unwrap();
        let p = doc(&store, &pid);
        let added = p.cards.len() - 2;
        assert!(added >= 1);
        assert_eq!(
            out.message,
            format!("続きのカードを{added}枚追加しました。")
        );
        assert_eq!(p.cards[0].text, "短い");
        assert_eq!(
            p.cards[1..]
                .iter()
                .map(|c| c.text.as_str())
                .collect::<String>(),
            "あ。".repeat(100)
        );
        assert!(
            p.cards[1..].iter().all(|c| c.voice_id == "v9"),
            "continuations inherit the voice"
        );
        let out = run(&store, &pid, "split", Some(&ids[0])).unwrap();
        assert_eq!(out.message, "分割は不要です。");
    }

    #[test]
    fn split_without_a_card_id_splits_every_card() {
        let (_d, store, pid) = setup();
        with_cards(&store, &pid, &[&"い。".repeat(100), &"う。".repeat(100)]);
        let out = run(&store, &pid, "split", None).unwrap();
        let n: usize = out
            .message
            .trim_start_matches("続きのカードを")
            .trim_end_matches("枚追加しました。")
            .parse()
            .unwrap();
        assert_eq!(doc(&store, &pid).cards.len(), 2 + n);
        assert!(n >= 2);
    }

    #[test]
    fn reset_replaces_everything_with_one_empty_card() {
        let (_d, store, pid) = setup();
        with_cards(&store, &pid, &["a", "b"]);
        run(&store, &pid, "reset", None).unwrap();
        let p = doc(&store, &pid);
        assert_eq!(
            (p.cards.len(), p.cards[0].text.as_str(), p.selected.as_str()),
            (1, "", p.cards[0].id.as_str())
        );
    }

    #[test]
    fn unknown_actions_and_missing_cards_have_python_messages() {
        let (_d, store, pid) = setup();
        let id = doc(&store, &pid).cards[0].id.clone();
        assert_eq!(
            run(&store, &pid, "bogus", Some(&id)).unwrap_err().message(),
            "操作を認識できません。"
        );
        assert_eq!(
            run(&store, &pid, "bogus", None).unwrap_err().message(),
            "カードを選んでください。"
        );
        assert_eq!(
            run(&store, &pid, "delete", Some("gone"))
                .unwrap_err()
                .message(),
            "カードを選んでください。"
        );
    }

    #[test]
    fn undo_redo_actions_return_messages() {
        let (_d, store, pid) = setup();
        let id = doc(&store, &pid).cards[0].id.clone();
        edit(
            &store,
            &pid,
            0,
            &json!({"cards": [{"id": id, "text": "変更"}]}),
        )
        .unwrap();
        assert_eq!(
            run(&store, &pid, "undo", None).unwrap().message,
            "元に戻しました。"
        );
        assert_eq!(doc(&store, &pid).cards[0].text, "");
        assert_eq!(
            run(&store, &pid, "redo", None).unwrap().message,
            "やり直しました。"
        );
        assert_eq!(doc(&store, &pid).cards[0].text, "変更");
        assert_eq!(
            run(&store, &pid, "redo", None).unwrap_err().message(),
            "やり直せる操作がありません。"
        );
    }

    #[test]
    fn undo_stops_cards_that_look_busy_but_have_no_active_request() {
        let (_d, store, pid) = setup();
        let ids = with_cards(&store, &pid, &["a", "b"]);
        let edited = edit(
            &store,
            &pid,
            0,
            &json!({"cards": [{"id": ids[0], "text": "x"}]}),
        )
        .unwrap();
        store
            .internal_update(&pid, &mut |d: &mut Project| {
                d.cards[0].status = "生成中".into();
                d.cards[1].status = "待機中".into();
            })
            .unwrap();
        let active: HashSet<String> = [ids[1].clone()].into_iter().collect();
        action(
            &store,
            &Dictionary::new(),
            &active,
            &pid,
            edited.revision,
            "undo",
            None,
            None,
        )
        .unwrap();
        let p = doc(&store, &pid);
        assert_eq!(p.cards[0].status, "停止");
        assert_eq!(p.cards[1].status, "待機中", "still queued in a live job");
    }

    #[test]
    fn project_load_new_and_copy() {
        let (_d, store, pid) = setup();
        let id = doc(&store, &pid).cards[0].id.clone();
        edit(&store, &pid, 0, &json!({"cards": [{"id": id, "text": "元の文章"}], "dictionary_enabled": false, "generation": {"steps": 4}})).unwrap();
        let fresh = new_or_load_project(&store, &json!({"name": "あ".repeat(150)})).unwrap();
        assert_eq!(fresh.name.chars().count(), 100);
        assert_eq!(store.settings().last_project, fresh.id);
        assert_eq!(fresh.cards[0].text, "");
        let unnamed = new_or_load_project(&store, &json!({})).unwrap();
        assert_eq!(unnamed.name, "新しい台本");
        let copy =
            new_or_load_project(&store, &json!({"name": "コピー", "copy": true, "pid": pid}))
                .unwrap();
        assert_eq!(
            (
                copy.cards[0].text.as_str(),
                copy.dictionary_enabled,
                copy.generation["steps"].clone()
            ),
            ("元の文章", false, json!(4))
        );
        assert_ne!(copy.id, pid);
        let loaded = new_or_load_project(&store, &json!({"load": true, "pid": pid})).unwrap();
        assert_eq!(loaded.id, pid);
        assert_eq!(store.settings().last_project, pid);
        assert!(new_or_load_project(&store, &json!({"load": true})).is_err());
        assert!(new_or_load_project(&store, &json!({"load": true, "pid": "../x"})).is_err());
    }

    #[test]
    fn recover_previous_restores_cards_and_refuses_while_busy() {
        let (_d, store, old_pid) = setup();
        let id = doc(&store, &old_pid).cards[0].id.clone();
        edit(
            &store,
            &old_pid,
            0,
            &json!({"cards": [{"id": id, "text": "前回の入力"}], "generation": {"steps": 8}}),
        )
        .unwrap();
        let current = new_or_load_project(&store, &json!({"name": "今回"})).unwrap();
        assert_eq!(
            recover_previous(&store, &current.id, 0, false)
                .unwrap_err()
                .message(),
            "復元できる前回の入力がありません。"
        );
        store
            .lock()
            .settings
            .extra
            .insert("recovery_project".into(), json!(old_pid));
        assert_eq!(
            recover_previous(&store, &current.id, 0, true)
                .unwrap_err()
                .message(),
            "生成が終わってから復元してください。"
        );
        let restored = recover_previous(&store, &current.id, 0, false).unwrap();
        assert_eq!(
            (
                restored.cards[0].text.as_str(),
                restored.generation["steps"].clone()
            ),
            ("前回の入力", json!(8))
        );
        assert_eq!(restored.id, current.id);
        assert_eq!(restored.revision, 1);
        assert!(store.history(&current.id).0, "recovery can be undone");
    }
}
