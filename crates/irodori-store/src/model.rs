//! v2.0 の台本・カード・話者・設定の型。未知のフィールドは `extra` に保持して書き戻す。
//!
//! フィールドの並びは v2.0 が JSON に書く順（`new_card` / `new_project` / `Store` の既定値）に合わせる。

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::generation::{default_generation, Generation};

/// 型に出てこない未知のフィールド（Python 版が追加した項目など）。
pub type Extra = Map<String, Value>;

/// 32 桁の小文字 hex（`uuid.uuid4().hex`）。
pub fn uid() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

fn one() -> f64 {
    1.0
}

fn untouched() -> String {
    "未生成".into()
}

fn default_name() -> String {
    "無題の台本".into()
}

fn yes() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Card {
    pub id: String,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub voice_id: String,
    #[serde(default = "one")]
    pub speed: f64,
    #[serde(default)]
    pub silence: f64,
    #[serde(default)]
    pub filename: String,
    #[serde(default)]
    pub result: Option<Value>,
    #[serde(default)]
    pub error: String,
    #[serde(default = "untouched")]
    pub status: String,
    #[serde(flatten)]
    pub extra: Extra,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Project {
    pub schema: i64,
    pub id: String,
    #[serde(default)]
    pub revision: i64,
    #[serde(default = "default_name")]
    pub name: String,
    #[serde(default)]
    pub selected: String,
    #[serde(default = "yes")]
    pub dictionary_enabled: bool,
    #[serde(default = "default_generation")]
    pub generation: Generation,
    pub cards: Vec<Card>,
    #[serde(flatten)]
    pub extra: Extra,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Voice {
    pub id: String,
    #[serde(default)]
    pub revision: i64,
    #[serde(default)]
    pub audio: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub caption: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sound_revision: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub creation: Option<Value>,
    #[serde(flatten)]
    pub extra: Extra,
}

impl Voice {
    /// 音の版（`sound_revision` が無ければ `revision`）。署名に使う。
    pub fn sound_revision_or_revision(&self) -> i64 {
        self.sound_revision.unwrap_or(self.revision)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub schema: i64,
    pub backend: String,
    pub theme: String,
    pub auto_browser: bool,
    pub restore_input: bool,
    pub inherit: bool,
    pub auto_export: bool,
    pub custom_output: bool,
    pub output_dir: String,
    pub custom_pattern: bool,
    pub pattern: String,
    pub txt: bool,
    pub last_project: String,
    #[serde(flatten)]
    pub extra: Extra,
}

impl Settings {
    /// `Store.__init__` の既定値。`backend` は `cpu` / `cu128` 以外なら `cpu`。
    pub fn defaults(backend: &str) -> Settings {
        Settings {
            schema: 2,
            backend: if matches!(backend, "cpu" | "cu128") { backend } else { "cpu" }.into(),
            theme: "light".into(),
            auto_browser: true,
            restore_input: false,
            inherit: true,
            auto_export: true,
            custom_output: false,
            output_dir: String::new(),
            custom_pattern: false,
            pattern: "$連番$_$話者名$_$日付$".into(),
            txt: false,
            last_project: String::new(),
            extra: Extra::new(),
        }
    }
}

impl Default for Settings {
    fn default() -> Self {
        Settings::defaults("cpu")
    }
}

/// 新しい空のカード。`source` があれば voice_id / speed / silence を引き継ぐ。
pub fn new_card(source: Option<&Card>) -> Card {
    let mut c = Card {
        id: uid(),
        text: String::new(),
        voice_id: String::new(),
        speed: 1.0,
        silence: 0.0,
        filename: String::new(),
        result: None,
        error: String::new(),
        status: untouched(),
        extra: Extra::new(),
    };
    if let Some(s) = source {
        c.voice_id = s.voice_id.clone();
        c.speed = s.speed;
        c.silence = s.silence;
    }
    c
}

pub fn new_project(name: &str) -> Project {
    let card = new_card(None);
    Project {
        schema: 2,
        id: uid(),
        revision: 0,
        name: name.into(),
        selected: card.id.clone(),
        dictionary_enabled: true,
        generation: default_generation(),
        cards: vec![card],
        extra: Extra::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const PROJECT: &str = include_str!("../tests/fixtures/project_sample.json");
    const VOICES: &str = include_str!("../tests/fixtures/voices_sample.json");
    const SETTINGS: &str = include_str!("../tests/fixtures/settings_sample.json");

    fn roundtrip<T: serde::Serialize + serde::de::DeserializeOwned>(text: &str) -> (Value, Value) {
        let original: Value = serde_json::from_str(text).unwrap();
        let typed: T = serde_json::from_str(text).unwrap();
        (original, serde_json::to_value(&typed).unwrap())
    }

    #[test]
    fn project_round_trip_keeps_unknown_fields_and_number_types() {
        let (a, b) = roundtrip::<Project>(PROJECT);
        assert_eq!(a, b);
        assert_eq!(b["future_project_field"]["nested"][1], serde_json::json!(2.5));
        assert_eq!(b["cards"][0]["future_card_field"], "keep me");
        assert_eq!(b["cards"][1]["missing_voice_name"], "消えた話者");
    }

    #[test]
    fn voices_round_trip_keeps_unknown_fields_and_absent_sound_revision() {
        let (a, b) = roundtrip::<Vec<Voice>>(VOICES);
        assert_eq!(a, b);
        assert!(b[1].get("sound_revision").is_none(), "absent stays absent");
        assert!(b[1].get("creation").is_none());
        assert_eq!(b[0]["future_voice_field"], serde_json::json!([1, 2]));
    }

    #[test]
    fn settings_round_trip_keeps_unknown_fields() {
        let v: Value = serde_json::from_str(SETTINGS).unwrap();
        let saved = v["saved"].to_string();
        let (a, b) = roundtrip::<Settings>(&saved);
        assert_eq!(a, b);
        assert_eq!(b["future_setting"], "x");
    }

    #[test]
    fn settings_defaults_match_python_store_defaults() {
        let v: Value = serde_json::from_str(SETTINGS).unwrap();
        let want = &v["defaults"];
        let got = serde_json::to_value(Settings::defaults("cpu")).unwrap();
        assert_eq!(&got, want);
    }

    #[test]
    fn partial_settings_fall_back_to_defaults() {
        let s: Settings = serde_json::from_str(r#"{"theme": "dark"}"#).unwrap();
        assert_eq!(s.theme, "dark");
        assert!(s.auto_browser && s.inherit && s.auto_export);
        assert_eq!(s.pattern, "$連番$_$話者名$_$日付$");
    }

    #[test]
    fn new_project_has_schema_2_and_one_empty_card() {
        let p = new_project("台本");
        assert_eq!((p.schema, p.revision, p.name.as_str()), (2, 0, "台本"));
        assert_eq!(p.cards.len(), 1);
        assert_eq!(p.selected, p.cards[0].id);
        assert!(p.dictionary_enabled);
        assert_eq!(p.cards[0].status, "未生成");
        assert_eq!(p.generation, crate::generation::default_generation());
    }

    #[test]
    fn uid_is_32_lowercase_hex_and_unique() {
        let (a, b) = (uid(), uid());
        assert_eq!(a.len(), 32);
        assert!(a.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')));
        assert_ne!(a, b);
    }

    #[test]
    fn integer_speed_from_the_ui_is_stored_as_float() {
        let c: Card = serde_json::from_str(r#"{"id":"x","speed":1,"silence":2}"#).unwrap();
        assert_eq!((c.speed, c.silence), (1.0, 2.0));
        let v = serde_json::to_value(&c).unwrap();
        assert_eq!(v["speed"], serde_json::json!(1.0));
        assert!(v["speed"].is_f64());
    }

    #[test]
    fn new_card_inherits_voice_speed_and_silence_only() {
        let mut src = new_card(None);
        src.voice_id = "v".into();
        src.speed = 1.25;
        src.silence = 0.5;
        src.text = "元".into();
        let c = new_card(Some(&src));
        assert_eq!((c.voice_id.as_str(), c.speed, c.silence), ("v", 1.25, 0.5));
        assert_eq!(c.text, "");
        assert_ne!(c.id, src.id);
    }
}
