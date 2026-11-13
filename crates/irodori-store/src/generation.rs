//! 生成設定の検証（`easy_storage.validate_generation` / `easy_models.DEFAULT_GENERATION`）。
//!
//! 結果の型は Python と同一（`duration_scale` などは常に浮動小数、`steps`/`seed`/`max_*_len` は整数または null）で、
//! キー順は既定値の順。範囲外メッセージの min/max 表記も Python の `str()` と同じ。

use serde_json::{json, Map, Number, Value};

use crate::{Result, StoreError};

pub type Generation = Map<String, Value>;

fn invalid(msg: impl Into<String>) -> StoreError {
    StoreError::Invalid(msg.into())
}

/// `DEFAULT_GENERATION`（`easy_models.py`）。
pub fn default_generation() -> Generation {
    let v = json!({
        "steps": null, "duration_scale": 1.0, "seed": null,
        "cfg_scale_text": 3.0, "cfg_scale_caption": 4.0, "cfg_scale_speaker": 5.0,
        "cfg_guidance_mode": "independent", "cfg_min_t": 0.5, "cfg_max_t": 1.0,
        "t_schedule_mode": "linear", "sway_coeff": -1.0, "context_kv_cache": true,
        "seconds": null, "cfg_scale": null, "speaker_kv_scale": null, "max_text_len": null,
        "max_caption_len": null, "truncation_factor": null, "rescale_k": null, "rescale_sigma": null,
        "lora_adapter": null, "model_precision": "fp32", "codec_precision": "fp32",
    });
    match v {
        Value::Object(m) => m,
        _ => unreachable!("json! object literal"),
    }
}

/// 範囲つき数値。範囲外・非数・非有限は `"{label}は{min}～{max}で指定してください。"`。
fn number(v: &Value, min: f64, max: f64, label: &str, min_s: &str, max_s: &str) -> Result<f64> {
    let x = match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    };
    match x {
        Some(x) if x.is_finite() && min <= x && x <= max => Ok(x),
        _ => Err(invalid(format!(
            "{label}は{min_s}～{max_s}で指定してください。"
        ))),
    }
}

fn float(x: f64) -> Value {
    Value::Number(Number::from_f64(x).expect("finite"))
}

fn is_blank(v: &Value) -> bool {
    matches!(v, Value::Null) || v.as_str() == Some("")
}

/// `(キー, min, max, min表記, max表記)`。
type Range = (&'static str, f64, f64, &'static str, &'static str);

const FLOAT_RANGES: [Range; 7] = [
    ("duration_scale", 0.5, 1.5, "0.5", "1.5"),
    ("cfg_scale_text", 0.0, 10.0, "0", "10"),
    ("cfg_scale_caption", 0.0, 10.0, "0", "10"),
    ("cfg_scale_speaker", 0.0, 10.0, "0", "10"),
    ("cfg_min_t", 0.0, 1.0, "0", "1"),
    ("cfg_max_t", 0.0, 1.0, "0", "1"),
    ("sway_coeff", -1.0, 1.5, "-1", "1.5"),
];

const OPTIONAL_RANGES: [Range; 8] = [
    ("seconds", 0.5, 30.0, "0.5", "30"),
    ("cfg_scale", 0.0, 10.0, "0", "10"),
    ("speaker_kv_scale", 0.01, 10.0, "0.01", "10"),
    ("max_text_len", 1.0, 256.0, "1", "256"),
    ("max_caption_len", 1.0, 512.0, "1", "512"),
    ("truncation_factor", 0.01, 10.0, "0.01", "10"),
    ("rescale_k", 0.01, 100.0, "0.01", "100"),
    ("rescale_sigma", 0.01, 100.0, "0.01", "100"),
];

/// 利用者の入力から正規化した生成設定を作る。検証順は Python と同じ（先に当たったエラーが返る）。
pub fn validate_generation(values: &Value) -> Result<Generation> {
    let obj = values
        .as_object()
        .ok_or_else(|| invalid("生成設定が不正です。"))?;
    let mut r = default_generation();
    for (k, v) in obj {
        if r.contains_key(k) {
            r.insert(k.clone(), v.clone());
        }
    }
    for (key, min, max, min_s, max_s) in FLOAT_RANGES {
        let x = number(&r[key], min, max, key, min_s, max_s)?;
        r.insert(key.into(), float(x));
    }
    if r["cfg_min_t"].as_f64() > r["cfg_max_t"].as_f64() {
        return Err(invalid("条件反映の開始位置は終了位置以下にしてください。"));
    }
    let steps = if is_blank(&r["steps"]) {
        Value::Null
    } else {
        json!(number(&r["steps"], 1.0, 120.0, "計算回数", "1", "120")?.trunc() as i64)
    };
    r.insert("steps".into(), steps);
    let seed = if is_blank(&r["seed"]) {
        Value::Null
    } else {
        json!(number(&r["seed"], 0.0, 4294967295.0, "シード", "0", "4294967295")?.trunc() as i64)
    };
    r.insert("seed".into(), seed);
    let mode_ok = matches!(r["t_schedule_mode"].as_str(), Some("linear" | "sway"))
        && matches!(
            r["cfg_guidance_mode"].as_str(),
            Some("independent" | "joint" | "alternating")
        );
    if !mode_ok {
        return Err(invalid("生成設定が不正です。"));
    }
    for (key, min, max, min_s, max_s) in OPTIONAL_RANGES {
        let v = if is_blank(&r[key]) {
            Value::Null
        } else {
            float(number(&r[key], min, max, key, min_s, max_s)?)
        };
        r.insert(key.into(), v);
    }
    for key in ["max_text_len", "max_caption_len"] {
        if let Some(x) = r[key].as_f64() {
            r.insert(key.into(), json!(x.trunc() as i64));
        }
    }
    if r["rescale_k"].is_null() != r["rescale_sigma"].is_null() {
        return Err(invalid("再スケーリングの2項目は両方指定してください。"));
    }
    let lora = match &r["lora_adapter"] {
        Value::Null | Value::Bool(false) => Value::Null,
        Value::String(s) if s.is_empty() => Value::Null,
        Value::Number(n) if n.as_f64() == Some(0.0) => Value::Null,
        Value::String(s) => Value::String(s.trim().to_string()),
        other => Value::String(other.to_string()),
    };
    r.insert("lora_adapter".into(), lora);
    for key in ["model_precision", "codec_precision"] {
        if !matches!(r[key].as_str(), Some("fp32" | "fp16" | "bf16")) {
            return Err(invalid("計算精度が不正です。"));
        }
    }
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const CASES: &str = include_str!("../tests/fixtures/generation_cases.json");

    #[test]
    fn validate_generation_matches_python_cases() {
        let cases: Vec<Value> = serde_json::from_str(CASES).unwrap();
        assert!(cases.len() >= 50);
        for case in cases {
            let input = &case["input"];
            let got = validate_generation(input);
            match case.get("error") {
                Some(e) => assert_eq!(
                    got.err().map(|x| x.message().to_string()).as_deref(),
                    e.as_str(),
                    "input {input}"
                ),
                None => assert_eq!(
                    Value::Object(got.unwrap_or_else(|e| panic!("input {input}: {}", e.message()))),
                    case["output"],
                    "input {input}"
                ),
            }
        }
    }

    #[test]
    fn defaults_have_python_key_order() {
        let keys: Vec<String> = default_generation().keys().cloned().collect();
        assert_eq!(
            &keys[..4],
            ["steps", "duration_scale", "seed", "cfg_scale_text"]
        );
        assert_eq!(keys.last().unwrap(), "codec_precision");
        assert_eq!(keys.len(), 23);
    }

    #[test]
    fn non_object_input_is_rejected() {
        assert_eq!(
            validate_generation(&Value::Null).err().unwrap().message(),
            "生成設定が不正です。"
        );
        assert!(validate_generation(&serde_json::json!([1])).is_err());
    }

    #[test]
    fn non_numeric_strings_are_rejected_with_the_range_message() {
        let e = validate_generation(&serde_json::json!({"duration_scale": "abc"}))
            .err()
            .unwrap();
        assert_eq!(e.message(), "duration_scaleは0.5～1.5で指定してください。");
    }
}
