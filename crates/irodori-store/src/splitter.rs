//! 台本の分割と推定時間（`easy_splitter.py` の移植）。

use unicode_segmentation::UnicodeSegmentation;

use crate::dictionary::{keys_longest_first, Dictionary};
use crate::unicode_tables::{in_table, LETTER_OR_NUMBER, PY_DIGIT, PY_SPACE, SYMBOL_OTHER};
use crate::{Result, StoreError};

pub const TARGET_SECONDS: f64 = 25.0;
const MAX_UNITS: usize = 180;
const TOO_LONG: &str = "読み辞書の1語が長すぎます。登録した読み方を短くしてください。";

/// （元の文字列, 読み）。辞書の一致は分割しない。
type Span = (String, String);

fn units(text: &str, dict: &Dictionary) -> Vec<Span> {
    let keys = keys_longest_first(dict);
    let mut out = Vec::new();
    let mut plain_start = 0;
    let mut pos = 0;
    let flush = |out: &mut Vec<Span>, from: usize, to: usize| {
        for g in text[from..to].graphemes(true) {
            out.push((g.to_string(), g.to_string()));
        }
    };
    while pos < text.len() {
        let rest = &text[pos..];
        match keys.iter().find(|k| rest.starts_with(**k)) {
            Some(k) => {
                flush(&mut out, plain_start, pos);
                out.push((k.to_string(), dict[*k].clone()));
                pos += k.len();
                plain_start = pos;
            }
            None => pos += rest.chars().next().map_or(1, char::len_utf8),
        }
    }
    flush(&mut out, plain_start, text.len());
    out
}

fn char_seconds(ch: char) -> f64 {
    if "。！？!?".contains(ch) {
        0.38
    } else if "、，,;；".contains(ch) {
        0.18
    } else if in_table(PY_SPACE, ch) {
        0.0
    } else if ('\u{4e00}'..='\u{9fff}').contains(&ch) {
        0.38
    } else if in_table(PY_DIGIT, ch) {
        0.25
    } else if ch.is_ascii_alphabetic() {
        0.13
    } else if in_table(LETTER_OR_NUMBER, ch) {
        0.16
    } else if in_table(SYMBOL_OTHER, ch) {
        0.2
    } else {
        0.0
    }
}

/// 保守的な読み上げ時間の目安（秒）。音響モデルによる予測ではない。
pub fn seconds(text: &str) -> f64 {
    text.chars().fold(0.0, |total, ch| total + char_seconds(ch))
}

/// Python の `round(x, 1)` と同じく、2 進数の正確な値を丸める。
fn round1(x: f64) -> f64 {
    format!("{x:.1}").parse().unwrap_or(x)
}

pub fn estimate(text: &str, dict: &Dictionary, scale: f64) -> f64 {
    let total = units(text, dict)
        .iter()
        .fold(0.0, |t, (_, reading)| t + seconds(reading));
    round1(total * scale.max(1.0))
}

fn is_latin_unit(s: &str) -> bool {
    s.len() == 1 && s.as_bytes()[0].is_ascii_alphanumeric()
}

fn ends_with_any(raw: &str, marks: &str) -> bool {
    raw.chars().next_back().is_some_and(|c| marks.contains(c))
}

pub fn split_text(text: &str, dict: &Dictionary, scale: f64) -> Result<Vec<String>> {
    split_with_target(text, dict, scale, TARGET_SECONDS)
}

pub fn split_with_target(
    text: &str,
    dict: &Dictionary,
    scale: f64,
    target: f64,
) -> Result<Vec<String>> {
    let spans = units(text, dict);
    if is_blank_text(text) {
        return Ok(vec![text.to_string()]);
    }
    let limit = target / scale.max(1.0);
    let mut pieces: Vec<String> = Vec::new();
    let mut start = 0;
    while start < spans.len() {
        let (mut weight, mut end) = (0.0, start);
        while end < spans.len() {
            let amount = seconds(&spans[end].1);
            if amount > limit {
                return Err(StoreError::Invalid(TOO_LONG.into()));
            }
            if end > start && (weight + amount > limit || end - start >= MAX_UNITS) {
                break;
            }
            weight += amount;
            end += 1;
        }
        if end < spans.len() {
            end = prefer_boundary(&spans, start, end);
        }
        let piece: String = spans[start..end]
            .iter()
            .map(|(raw, _)| raw.as_str())
            .collect();
        match pieces.last_mut() {
            Some(last) if is_blank_text(&piece) => last.push_str(&piece),
            _ => pieces.push(piece),
        }
        start = end;
    }
    if pieces.len() > 1 && is_blank_text(&pieces[0]) {
        let first = pieces.remove(0);
        pieces[0].insert_str(0, &first);
    }
    debug_assert_eq!(pieces.concat(), text);
    Ok(pieces)
}

fn is_blank_text(s: &str) -> bool {
    s.chars().all(|c| in_table(PY_SPACE, c))
}

/// 区間の後半にある自然な切れ目、なければ短い英数字の語の頭まで戻す。
fn prefer_boundary(spans: &[Span], start: usize, mut end: usize) -> usize {
    let lower = start + ((end - start) / 2).max(1);
    for marks in ["\n。！？!?", "、，,;；", " \t"] {
        if let Some(i) = (lower - 1..end)
            .rev()
            .find(|&i| ends_with_any(&spans[i].0, marks))
        {
            end = i + 1;
            break;
        }
    }
    if end < spans.len() && is_latin_unit(&spans[end].0) {
        let mut boundary = end;
        while boundary > start && is_latin_unit(&spans[boundary - 1].0) {
            boundary -= 1;
        }
        if boundary > start && end - boundary < 30 {
            end = boundary;
        }
    }
    end
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary::Dictionary;
    use serde_json::Value;

    const CASES: &str = include_str!("../tests/fixtures/splitter_cases.json");

    fn fixture() -> Value {
        serde_json::from_str(CASES).unwrap()
    }

    #[test]
    fn seconds_matches_python_for_sample_characters() {
        let f = fixture();
        let map = f["seconds"].as_object().unwrap();
        assert!(map.len() >= 15);
        for (ch, want) in map {
            let got = seconds(ch);
            assert!(
                (got - want.as_f64().unwrap()).abs() < 1e-12,
                "{ch:?}: {got}"
            );
        }
    }

    #[test]
    fn estimate_matches_python_for_every_case() {
        let f = fixture();
        let cases = f["cases"].as_array().unwrap();
        assert!(cases.len() > 300);
        for c in cases.iter().filter(|c| c.get("estimate").is_some()) {
            let dict: Dictionary = serde_json::from_value(c["dictionary"].clone()).unwrap();
            let got = estimate(
                c["text"].as_str().unwrap(),
                &dict,
                c["scale"].as_f64().unwrap(),
            );
            assert_eq!(got, c["estimate"].as_f64().unwrap(), "text {}", c["text"]);
        }
    }

    #[test]
    fn split_matches_python_for_every_case() {
        let f = fixture();
        for c in f["cases"].as_array().unwrap() {
            let dict: Dictionary = serde_json::from_value(c["dictionary"].clone()).unwrap();
            let got = split_text(
                c["text"].as_str().unwrap(),
                &dict,
                c["scale"].as_f64().unwrap(),
            );
            match c.get("error") {
                Some(e) => assert_eq!(
                    got.err().map(|x| x.message().to_string()).as_deref(),
                    e.as_str(),
                    "text {}",
                    c["text"]
                ),
                None => {
                    let want: Vec<String> = c["split"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_str().unwrap().to_string())
                        .collect();
                    assert_eq!(
                        got.unwrap(),
                        want,
                        "text {} scale {}",
                        c["text"],
                        c["scale"]
                    );
                }
            }
        }
    }

    #[test]
    fn split_is_lossless() {
        let f = fixture();
        for c in f["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c.get("split").is_some())
        {
            let pieces: Vec<&str> = c["split"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect();
            assert_eq!(pieces.concat(), c["text"].as_str().unwrap());
        }
    }

    #[test]
    fn long_dictionary_reading_is_reported() {
        let mut dict = Dictionary::new();
        dict.insert("長い".into(), "あ".repeat(200));
        assert_eq!(
            split_text("長い", &dict, 1.0).unwrap_err().message(),
            "読み辞書の1語が長すぎます。登録した読み方を短くしてください。"
        );
    }

    #[test]
    fn blank_text_comes_back_as_one_unchanged_piece() {
        assert_eq!(
            split_text("   ", &Dictionary::new(), 1.0).unwrap(),
            vec!["   ".to_string()]
        );
        assert_eq!(
            split_text("", &Dictionary::new(), 1.0).unwrap(),
            vec!["".to_string()]
        );
    }
}
