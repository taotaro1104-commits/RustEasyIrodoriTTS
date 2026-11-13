//! 元に戻す／やり直しの履歴（台本ごとにメモリ上、最大 100 段）。
//!
//! 履歴に積むのは「入力内容」（生成設定・辞書の有効/無効・各カードから `result/status/error` を除いたもの）だけ。
//! 選択の変更や生成結果は編集操作ではない。生成結果は `Outcome` として別に覚えておき、戻した後に復元する。

use std::collections::HashMap;

use serde_json::Value;

use crate::generation::Generation;
use crate::model::Project;

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
