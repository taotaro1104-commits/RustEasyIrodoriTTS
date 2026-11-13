//! 元に戻す／やり直しの履歴（台本ごとにメモリ上、最大 100 段）。
//!
//! 履歴に積むのは「入力内容」（生成設定・辞書の有効/無効・各カードから `result/status/error` を除いたもの）だけ。
//! 選択の変更や生成結果は編集操作ではない。生成結果は `Outcome` として別に覚えておき、戻した後に復元する。

use std::collections::HashMap;

use serde_json::Value;

use crate::generation::Generation;
use crate::model::{Card, Project};
use crate::storage::Store;
use crate::{Result, StoreError};

pub const HISTORY_LIMIT: usize = 100;

#[derive(Debug, Clone, PartialEq)]
pub struct InputState {
    pub generation: Generation,
    pub dictionary_enabled: bool,
    /// 各カードを JSON にし、`result` `status` `error` を除いたもの。
    pub cards: Vec<Value>,
}

pub fn input_state(p: &Project) -> InputState {
    let cards = p
        .cards
        .iter()
        .map(|c| {
            let mut v = serde_json::to_value(c).unwrap_or(Value::Null);
            if let Value::Object(m) = &mut v {
                for k in ["result", "status", "error"] {
                    m.remove(k);
                }
            }
            v
        })
        .collect();
    InputState { generation: p.generation.clone(), dictionary_enabled: p.dictionary_enabled, cards }
}

/// カードの生成結果・状態。
#[derive(Debug, Clone)]
pub struct Outcome {
    pub result: Option<Value>,
    pub status: String,
    pub error: String,
}

#[derive(Default)]
pub struct History {
    pub undo: HashMap<String, Vec<InputState>>,
    pub redo: HashMap<String, Vec<InputState>>,
    pub outcomes: HashMap<String, HashMap<String, Outcome>>,
}

impl History {
    /// 編集前の入力内容を undo に積む（100 段で切り詰め、redo は消す）。
    pub fn push(&mut self, pid: &str, old: InputState) {
        let stack = self.undo.entry(pid.to_string()).or_default();
        stack.push(old);
        if stack.len() > HISTORY_LIMIT {
            let excess = stack.len() - HISTORY_LIMIT;
            stack.drain(..excess);
        }
        self.redo.insert(pid.to_string(), Vec::new());
    }

    /// 現在のカードごとの結果を覚える（戻した後に復元するため）。
    pub fn capture(&mut self, pid: &str, doc: &Project) {
        let cache = self.outcomes.entry(pid.to_string()).or_default();
        for c in &doc.cards {
            cache.insert(c.id.clone(), Outcome { result: c.result.clone(), status: c.status.clone(), error: c.error.clone() });
        }
    }
}

impl Store {
    /// `(元に戻せる, やり直せる)`。
    pub fn history(&self, pid: &str) -> (bool, bool) {
        let inner = self.lock();
        let has = |m: &HashMap<String, Vec<InputState>>| m.get(pid).is_some_and(|v| !v.is_empty());
        (has(&inner.history.undo), has(&inner.history.redo))
    }

    /// 履歴をたどる（`redo=false` は元に戻す）。入力内容は履歴から、カードごとの結果は直近の記憶から復元する。
    pub fn travel(&self, pid: &str, revision: i64, redo: bool) -> Result<Project> {
        let mut inner = self.lock();
        let old = inner.load(Some(pid))?;
        if old.revision != revision {
            return Err(StoreError::Conflict("入力内容が更新されています。画面を読み直してください。".into()));
        }
        let stack = if redo { &inner.history.redo } else { &inner.history.undo };
        let Some(target) = stack.get(pid).and_then(|v| v.last()).cloned() else {
            let msg = if redo { "やり直せる操作がありません。" } else { "元に戻せる操作がありません。" };
            return Err(StoreError::Invalid(msg.into()));
        };
        inner.history.capture(pid, &old);

        let mut doc = old.clone();
        doc.generation = target.generation.clone();
        doc.dictionary_enabled = target.dictionary_enabled;
        doc.cards = target
            .cards
            .iter()
            .map(|v| serde_json::from_value::<Card>(v.clone()).map_err(|e| StoreError::Internal(e.to_string())))
            .collect::<Result<Vec<_>>>()?;
        let outcomes = inner.history.outcomes.get(pid);
        for c in &mut doc.cards {
            match outcomes.and_then(|m| m.get(&c.id)) {
                Some(o) => {
                    c.result = o.result.clone();
                    c.status = o.status.clone();
                    c.error = o.error.clone();
                }
                None => {
                    c.result = None;
                    c.status = "未生成".into();
                    c.error.clear();
                }
            }
        }
        if !doc.cards.iter().any(|c| c.id == doc.selected) {
            doc.selected = doc.cards[0].id.clone();
        }
        doc.revision += 1;
        inner.persist(&doc)?;
        inner.projects.insert(pid.to_string(), doc.clone());

        let h = &mut inner.history;
        let (source, destination) = if redo { (&mut h.redo, &mut h.undo) } else { (&mut h.undo, &mut h.redo) };
        if let Some(v) = source.get_mut(pid) {
            v.pop();
        }
        let dest = destination.entry(pid.to_string()).or_default();
        dest.push(input_state(&old));
        if dest.len() > HISTORY_LIMIT {
            let excess = dest.len() - HISTORY_LIMIT;
            dest.drain(..excess);
        }
        Ok(doc)
    }
}

#[cfg(test)]
mod tests {
    use crate::model::{new_card, Project};
    use crate::testutil::temp_root;
    use crate::Store;
    use serde_json::json;

    fn setup() -> (tempfile::TempDir, Store, String) {
        let dir = temp_root();
        let store = Store::open(dir.path()).unwrap();
        let pid = store.load(None).unwrap().id;
        (dir, store, pid)
    }

    fn set_text(store: &Store, pid: &str, text: &str) -> Project {
        let rev = store.load(Some(pid)).unwrap().revision;
        store.update(pid, rev, true, &mut |d: &mut Project| { d.cards[0].text = text.into(); Ok(()) }).unwrap()
    }

    #[test]
    fn undo_then_redo_restores_text() {
        let (_d, store, pid) = setup();
        set_text(&store, &pid, "A");
        let b = set_text(&store, &pid, "B");
        let undone = store.travel(&pid, b.revision, false).unwrap();
        assert_eq!((undone.cards[0].text.as_str(), undone.revision), ("A", b.revision + 1));
        let redone = store.travel(&pid, undone.revision, true).unwrap();
        assert_eq!(redone.cards[0].text, "B");
        assert_eq!(store.history(&pid), (true, false));
    }

    #[test]
    fn history_is_capped_at_100() {
        let (_d, store, pid) = setup();
        for i in 0..105 {
            set_text(&store, &pid, &format!("t{i}"));
        }
        let mut undone = 0;
        loop {
            let rev = store.load(Some(&pid)).unwrap().revision;
            match store.travel(&pid, rev, false) {
                Ok(_) => undone += 1,
                Err(e) => {
                    assert_eq!(e.message(), "元に戻せる操作がありません。");
                    break;
                }
            }
        }
        assert_eq!(undone, 100);
    }

    #[test]
    fn new_edit_clears_redo() {
        let (_d, store, pid) = setup();
        set_text(&store, &pid, "A");
        let b = set_text(&store, &pid, "B");
        store.travel(&pid, b.revision, false).unwrap();
        assert!(store.history(&pid).1);
        set_text(&store, &pid, "C");
        assert_eq!(store.history(&pid).1, false);
    }

    #[test]
    fn selection_and_results_are_not_edit_operations() {
        let (_d, store, pid) = setup();
        let rev = store.update(&pid, 0, true, &mut |d: &mut Project| { d.cards.push(new_card(None)); Ok(()) }).unwrap().revision;
        assert_eq!(store.history(&pid), (true, false));
        let _ = store.travel(&pid, rev, false).unwrap(); // 履歴を空にする
        let p = store.load(Some(&pid)).unwrap();
        let other = new_card(None);
        let other_id = other.id.clone();
        let p = store.update(&pid, p.revision, false, &mut |d: &mut Project| { d.cards.push(other.clone()); Ok(()) }).unwrap();
        // 選択の変更は履歴に積まない
        store.update(&pid, p.revision, true, &mut |d: &mut Project| { d.selected = other_id.clone(); Ok(()) }).unwrap();
        // 生成結果の更新も積まない
        store.internal_update(&pid, &mut |d: &mut Project| d.cards[0].status = "完了".into()).unwrap();
        assert_eq!(store.history(&pid).0, false);
    }

    #[test]
    fn undo_restores_cached_results_for_cards() {
        let (_d, store, pid) = setup();
        store.internal_update(&pid, &mut |d: &mut Project| { d.cards[0].status = "完了".into(); d.cards[0].result = Some(json!({"path": "x.wav"})); }).unwrap();
        let edited = set_text(&store, &pid, "変更");
        let undone = store.travel(&pid, edited.revision, false).unwrap();
        assert_eq!(undone.cards[0].status, "完了");
        assert_eq!(undone.cards[0].result, Some(json!({"path": "x.wav"})));
    }

    #[test]
    fn undo_removes_cards_added_later_and_fixes_the_selection() {
        let (_d, store, pid) = setup();
        let added = new_card(None);
        let added_id = added.id.clone();
        let p = store.update(&pid, 0, true, &mut |d: &mut Project| { d.cards.push(added.clone()); d.selected = added.id.clone(); Ok(()) }).unwrap();
        assert_eq!(p.cards.len(), 2);
        let undone = store.travel(&pid, p.revision, false).unwrap();
        assert_eq!(undone.cards.len(), 1);
        assert!(undone.cards.iter().all(|c| c.id != added_id));
        assert_eq!(undone.selected, undone.cards[0].id);
    }

    #[test]
    fn empty_history_errors_have_python_messages() {
        let (_d, store, pid) = setup();
        assert_eq!(store.travel(&pid, 0, false).err().unwrap().message(), "元に戻せる操作がありません。");
        assert_eq!(store.travel(&pid, 0, true).err().unwrap().message(), "やり直せる操作がありません。");
    }

    #[test]
    fn travel_with_stale_revision_conflicts() {
        let (_d, store, pid) = setup();
        set_text(&store, &pid, "A");
        let err = store.travel(&pid, 0, false).unwrap_err();
        assert_eq!((err.status(), err.message()), (409, "入力内容が更新されています。画面を読み直してください。"));
    }

    #[test]
    fn history_is_persisted_text_but_not_across_restarts() {
        let (dir, store, pid) = setup();
        set_text(&store, &pid, "A");
        let reopened = Store::open(dir.path()).unwrap();
        assert_eq!(reopened.history(&pid), (false, false), "履歴はメモリ上のみ（v2.0 と同じ）");
        assert_eq!(reopened.load(Some(&pid)).unwrap().cards[0].text, "A");
    }
}
