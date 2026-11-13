//! 日本語テキスト正規化（上流 `text_normalization.py` の移植）。
//!
//! 処理順は上流と同一: 単純置換 → 正規表現置換 → 外側括弧の除去 → NFKC → `...`/`..` → `…`。

use std::sync::OnceLock;

use regex::Regex;
use unicode_normalization::UnicodeNormalization;

/// 上流 `SIMPLE_REPLACE_MAP`（辞書順を保持）。
const SIMPLE_REPLACE: &[(&str, &str)] = &[
    ("\t", ""),
    ("[n]", ""),
    (r"\[n\]", ""),
    ("\u{3000}", ""),
    ("？", "?"),
    ("！", "!"),
    ("♥", "♡"),
    ("●", "○"),
    ("◯", "○"),
    ("〇", "○"),
];

/// 上流 `REGEX_REPLACE_MAP`（パターン, 置換）。
const REGEX_REPLACE: &[(&str, &str)] = &[
    ("[;▼♀♂《》≪≫①②③④⑤⑥]", ""),
    (
        r"[\x{02d7}\x{2010}-\x{2015}\x{2043}\x{2212}\x{23af}\x{23e4}\x{2500}\x{2501}\x{2e3a}\x{2e3b}]",
        "",
    ),
    (r"[\x{ff5e}\x{301c}]", "ー"),
    ("…{3,}", "……"),
];

const BRACKET_PAIRS: &[(char, char)] =
    &[('「', '」'), ('『', '』'), ('（', '）'), ('【', '】'), ('(', ')')];

fn compiled_regexes() -> &'static Vec<(Regex, &'static str)> {
    static CELL: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    CELL.get_or_init(|| {
        REGEX_REPLACE
            .iter()
            .map(|(p, r)| (Regex::new(p).expect("static regex"), *r))
            .collect()
    })
}

/// 全体を囲む括弧を繰り返し外す。`「a」「b」` のように途中で閉じる場合は外さない。
fn strip_outer_brackets(text: &str) -> String {
    let mut chars: Vec<char> = text.chars().collect();
    loop {
        if chars.len() < 2 {
            break;
        }
        let (start, end) = (chars[0], chars[chars.len() - 1]);
        if !BRACKET_PAIRS.iter().any(|&(s, e)| s == start && e == end) {
            break;
        }
        let mut depth = 0i32;
        let mut enclosing_all = true;
        for (i, &c) in chars.iter().enumerate() {
            if c == start {
                depth += 1;
            } else if c == end {
                depth -= 1;
            }
            if depth == 0 && i < chars.len() - 1 {
                enclosing_all = false;
                break;
            }
        }
        if enclosing_all && depth == 0 {
            chars = chars[1..chars.len() - 1].to_vec();
            continue;
        }
        break;
    }
    chars.into_iter().collect()
}

/// 上流 `normalize_text` と同じ正規化を行う。結果が空でも panic しない。
pub fn normalize_text(text: &str) -> String {
    let mut out = text.to_string();
    for (old, new) in SIMPLE_REPLACE {
        out = out.replace(old, new);
    }
    for (re, rep) in compiled_regexes() {
        out = re.replace_all(&out, *rep).into_owned();
    }
    out = strip_outer_brackets(&out);
    out = out.nfkc().collect::<String>();
    out.replace("...", "…").replace("..", "…")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_enclosing_brackets() {
        assert_eq!(normalize_text("「こんにちは」"), "こんにちは");
        assert_eq!(normalize_text("(a)"), "a");
        assert_eq!(normalize_text("『「a」』"), "a");
    }

    #[test]
    fn keeps_brackets_that_do_not_enclose_everything() {
        assert_eq!(normalize_text("「a」「b」"), "「a」「b」");
    }

    #[test]
    fn replaces_fullwidth_punctuation() {
        assert_eq!(normalize_text("え？そう！"), "え?そう!");
    }

    #[test]
    fn applies_nfkc() {
        assert_eq!(normalize_text("ｱｲｳ"), "アイウ");
    }

    #[test]
    fn collapses_dots_and_ellipsis() {
        assert_eq!(normalize_text("a...b"), "a…b");
        assert_eq!(normalize_text("a..b"), "a…b");
        assert_eq!(normalize_text("a…………b"), "a……b");
    }

    #[test]
    fn maps_wave_dash_to_long_vowel() {
        assert_eq!(normalize_text("あ\u{301C}い\u{ff5e}"), "あーいー");
    }

    #[test]
    fn removes_whitespace_markers_and_symbols() {
        assert_eq!(normalize_text("a\tb\u{3000}c[n]d▼e"), "abcde");
    }

    #[test]
    fn unifies_circles_and_hearts() {
        assert_eq!(normalize_text("●◯〇♥"), "○○○♡");
    }

    #[test]
    fn empty_and_bracket_only_inputs_do_not_panic() {
        assert_eq!(normalize_text(""), "");
        assert_eq!(normalize_text("「」"), "");
        assert_eq!(normalize_text("\u{3000}"), "");
    }
}
