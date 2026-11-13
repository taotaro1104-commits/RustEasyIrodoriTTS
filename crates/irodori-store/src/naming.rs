//! 書き出しファイル名の規則（`easy_export.safe_name` / `filename`）。

use std::sync::OnceLock;

use regex::Regex;

use crate::model::Settings;
use crate::unicode_tables::{in_table, PY_SPACE};

const DEFAULT_PATTERN: &str = "$連番$_$話者名$_$日付$";

fn re(cell: &'static OnceLock<Regex>, pattern: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("static regex"))
}

fn trim_py(s: &str) -> &str {
    s.trim_matches(|c| in_table(PY_SPACE, c))
}

fn rstrip_dots(s: &str) -> &str {
    s.trim_end_matches(|c| c == '.' || c == ' ')
}

/// Windows で使えない文字を `_` にし、予約名を避ける。
pub fn safe_name(value: &str) -> String {
    static FORBIDDEN: OnceLock<Regex> = OnceLock::new();
    static RESERVED: OnceLock<Regex> = OnceLock::new();
    let replaced = re(&FORBIDDEN, r#"[<>:"/\\|?*\x00-\x1f]"#).replace_all(value, "_");
    let stripped = rstrip_dots(trim_py(&replaced));
    let cut: String = stripped.chars().take(100).collect();
    let mut name = rstrip_dots(&cut).to_string();
    if name.is_empty() {
        name = "音声".into();
    }
    if re(
        &RESERVED,
        r"(?i)^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)",
    )
    .is_match(&name)
    {
        name.insert(0, '_');
    }
    name
}

/// `filename(settings, card, voice, ordinal, date)`（拡張子なし）。
pub fn filename(
    settings: &Settings,
    text: &str,
    voice_name: &str,
    ordinal: i64,
    date: &str,
) -> String {
    static TOKEN: OnceLock<Regex> = OnceLock::new();
    let pattern = if settings.custom_pattern {
        settings.pattern.as_str()
    } else {
        DEFAULT_PATTERN
    };
    let out = re(&TOKEN, r"\$(?:連番|話者名|テキスト|日付)\$").replace_all(
        pattern,
        |caps: &regex::Captures| match &caps[0] {
            "$連番$" => format!("{ordinal:03}"),
            "$話者名$" => voice_name.to_string(),
            "$テキスト$" => text.chars().take(40).collect(),
            _ => date.to_string(),
        },
    );
    safe_name(&out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_name_replaces_forbidden_characters_and_trims() {
        assert_eq!(safe_name("a<b>c:d\"e/f\\g|h?i*j"), "a_b_c_d_e_f_g_h_i_j");
        assert_eq!(safe_name("  名前. . "), "名前");
        assert_eq!(safe_name(""), "音声");
        assert_eq!(safe_name("...  "), "音声");
        assert_eq!(safe_name("line\nbreak"), "line_break");
    }

    #[test]
    fn safe_name_cuts_at_100_characters_and_avoids_reserved_names() {
        assert_eq!(safe_name(&"あ".repeat(150)).chars().count(), 100);
        assert_eq!(safe_name("con"), "_con");
        assert_eq!(safe_name("NUL.txt"), "_NUL.txt");
        assert_eq!(safe_name("COM1"), "_COM1");
        assert_eq!(safe_name("console"), "console");
        assert_eq!(safe_name("COM10"), "COM10");
    }

    #[test]
    fn filename_uses_the_default_or_the_custom_pattern() {
        let mut s = Settings::defaults("cpu");
        assert_eq!(
            filename(&s, "テキスト", "話者A", 7, "20261113"),
            "007_話者A_20261113"
        );
        s.custom_pattern = true;
        s.pattern = "$テキスト$-$連番$".into();
        assert_eq!(
            filename(&s, &"あ".repeat(60), "x", 12, "d"),
            format!("{}-012", "あ".repeat(40))
        );
        s.pattern = "$日付$/$話者名$".into();
        assert_eq!(filename(&s, "t", "a:b", 1, "d"), "d_a_b");
    }
}
