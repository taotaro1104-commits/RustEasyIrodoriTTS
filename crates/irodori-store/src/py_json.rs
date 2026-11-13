//! Python の `json.dumps` と同じ書式（署名とファイル書式の互換の土台）。
//!
//! - [`dumps_sorted`] = `json.dumps(v, ensure_ascii=False, sort_keys=True)`（区切りは `", "` と `": "`）
//! - [`dumps_pretty`] = `json.dumps(v, ensure_ascii=False, indent=2)`
//! - [`float_repr`] = `repr(float)`（最短表現。指数表記は 10 進位置が `<= -4` または `> 16` のとき）

use serde_json::Value;

use crate::{Result, StoreError};

/// ryu の最短表現を `(有効数字, 小数点位置)` に分解する。値 = 0.DIGITS × 10^decpt。
fn shortest_digits(x: f64) -> (String, i32) {
    let mut buf = ryu::Buffer::new();
    let s = buf.format_finite(x);
    let (mant, exp) = match s.split_once('e') {
        Some((m, e)) => (m, e.parse::<i32>().unwrap_or(0)),
        None => (s, 0),
    };
    let (int_part, frac_part) = mant.split_once('.').unwrap_or((mant, ""));
    let mut digits = format!("{int_part}{frac_part}");
    let mut point = int_part.len() as i32 + exp;
    let leading = digits.len() - digits.trim_start_matches('0').len();
    digits.drain(..leading);
    point -= leading as i32;
    let digits = digits.trim_end_matches('0').to_string();
    (digits, point)
}

/// Python の `repr(float)`。NaN / Infinity は Python の `json` と同じ綴り（`NaN` `Infinity` `-Infinity`）。
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    let sign = if x < 0.0 { "-" } else { "" };
    let (digits, decpt) = shortest_digits(x.abs());
    let body = if decpt <= -4 || decpt > 16 {
        let exp = decpt - 1;
        let mant = if digits.len() > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits.clone()
        };
        format!("{mant}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    } else if decpt <= 0 {
        format!("0.{}{digits}", "0".repeat((-decpt) as usize))
    } else if decpt as usize >= digits.len() {
        format!("{digits}{}.0", "0".repeat(decpt as usize - digits.len()))
    } else {
        format!(
            "{}.{}",
            &digits[..decpt as usize],
            &digits[decpt as usize..]
        )
    };
    format!("{sign}{body}")
}

/// 有限でない数は `allow_nan=False` 相当でエラーにする。
pub fn float_repr_checked(x: f64) -> Result<String> {
    if x.is_finite() {
        Ok(float_repr(x))
    } else {
        Err(StoreError::Invalid(
            "保存できない数値（NaN または無限大）が含まれています。".into(),
        ))
    }
}

fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_number(out: &mut String, n: &serde_json::Number) {
    if let Some(i) = n.as_i64() {
        out.push_str(&i.to_string());
    } else if let Some(u) = n.as_u64() {
        out.push_str(&u.to_string());
    } else {
        out.push_str(&float_repr(n.as_f64().unwrap_or(0.0)));
    }
}

fn write_value(out: &mut String, v: &Value, indent: Option<usize>, level: usize, sort: bool) {
    let nl = |out: &mut String, lvl: usize| {
        if let Some(step) = indent {
            out.push('\n');
            out.push_str(&" ".repeat(step * lvl));
        }
    };
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => write_number(out, n),
        Value::String(s) => write_string(out, s),
        Value::Array(a) if a.is_empty() => out.push_str("[]"),
        Value::Array(a) => {
            out.push('[');
            for (i, item) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(if indent.is_some() { "," } else { ", " });
                }
                nl(out, level + 1);
                write_value(out, item, indent, level + 1, sort);
            }
            nl(out, level);
            out.push(']');
        }
        Value::Object(m) if m.is_empty() => out.push_str("{}"),
        Value::Object(m) => {
            let mut entries: Vec<(&String, &Value)> = m.iter().collect();
            if sort {
                entries.sort_by(|a, b| a.0.cmp(b.0));
            }
            out.push('{');
            for (i, (k, val)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(if indent.is_some() { "," } else { ", " });
                }
                nl(out, level + 1);
                write_string(out, k);
                out.push_str(": ");
                write_value(out, val, indent, level + 1, sort);
            }
            nl(out, level);
            out.push('}');
        }
    }
}

/// `json.dumps(v, ensure_ascii=False, sort_keys=True)`。
pub fn dumps_sorted(v: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, v, None, 0, true);
    out
}

/// `json.dumps(v, ensure_ascii=False, indent=2)`（キー順は保持）。
pub fn dumps_pretty(v: &Value) -> Result<String> {
    let mut out = String::new();
    write_value(&mut out, v, Some(2), 0, false);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const CASES: &str = include_str!("../tests/fixtures/py_json_cases.json");

    fn cases() -> Vec<(String, String, String)> {
        let v: Vec<Value> = serde_json::from_str(CASES).unwrap();
        v.iter()
            .map(|c| {
                let g = |k: &str| c[k].as_str().unwrap().to_string();
                (g("input"), g("sorted"), g("pretty"))
            })
            .collect()
    }

    #[test]
    fn matches_python_for_every_fixture_case() {
        let all = cases();
        assert!(all.len() > 30);
        for (input, sorted, pretty) in all {
            let v: Value = serde_json::from_str(&input).unwrap();
            assert_eq!(dumps_sorted(&v), sorted, "sorted: {input}");
            assert_eq!(dumps_pretty(&v).unwrap(), pretty, "pretty: {input}");
        }
    }

    #[test]
    fn float_repr_switches_to_exponent_like_python() {
        let cases: [(f64, &str); 14] = [
            (1.0, "1.0"),
            (0.75, "0.75"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (1.5e-7, "1.5e-07"),
            (1e22, "1e+22"),
            (123456789.123, "123456789.123"),
            (-0.0, "-0.0"),
            (0.1, "0.1"),
            (123456789012345678.0, "1.2345678901234568e+17"),
            (1e100, "1e+100"),
            (5e-324, "5e-324"),
        ];
        for (x, want) in cases {
            assert_eq!(float_repr(x), want, "{x:e}");
        }
    }

    #[test]
    fn non_finite_numbers_are_rejected_by_dumps_pretty() {
        // serde_json の Value は NaN/Inf を保持できないので、直接の変換関数で確認する。
        assert!(float_repr_checked(f64::NAN).is_err());
        assert!(float_repr_checked(f64::INFINITY).is_err());
        assert_eq!(float_repr_checked(1.5).unwrap(), "1.5");
    }

    #[test]
    fn keys_sort_by_code_point() {
        let v: Value = serde_json::from_str(r#"{"b":1,"B":2,"あ":3,"a":4}"#).unwrap();
        assert_eq!(dumps_sorted(&v), r#"{"B": 2, "a": 4, "b": 1, "あ": 3}"#);
    }

    #[test]
    fn seventeen_digit_floats_survive_parse_and_dump_exactly() {
        // serde_json の既定の浮動小数パーサは 1 ULP ずれることがある（`float_roundtrip` が要る）。
        // ずれると署名が変わり、v2.0 の結果が「再生成が必要」になる。
        for text in [
            "1.1500000000000001",
            "0.30000000000000004",
            "2.2250738585072014e-308",
            "1.7976931348623157e+308",
            "0.1",
            "123456789.12345679",
        ] {
            let expected: f64 = text.parse().unwrap();
            let parsed: Value = serde_json::from_str(text).unwrap();
            assert_eq!(parsed.as_f64(), Some(expected), "parse {text}");
            assert_eq!(dumps_sorted(&parsed), float_repr(expected), "dump {text}");
        }
        let v: Value = serde_json::from_str("[1.1500000000000001]").unwrap();
        assert_eq!(dumps_sorted(&v), "[1.1500000000000001]");
    }
}
