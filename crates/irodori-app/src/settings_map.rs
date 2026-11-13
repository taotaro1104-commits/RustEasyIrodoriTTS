//! 生成設定（v2.0 の `validate_generation` の結果）を、コアのサンプラ設定に変換する。
//!
//! 対応しない設定（LoRA、fp16）は、黙って無視せず日本語のエラーにする。

use irodori_core::sampler::{GuidanceMode, SamplerParams, TSchedule};
use irodori_store::generation::Generation;
use serde_json::Value;

use crate::synth::SynthError;

/// 許可リストのモデルと、既定のステップ数（RF は 40、MeanFlow は 4）。
const MODEL_STEPS: [(&str, usize); 3] = [
    ("Aratako/Irodori-TTS-v4.1-Small", 40),
    ("phasefield-audio/Irodori-TTS-v4.1-Anime", 40),
    ("Aratako/Irodori-TTS-v4.1-Small-MF", 4),
];
const DEFAULT_STEPS: usize = 40;

#[derive(Debug, Clone)]
pub struct GenerationOptions {
    /// `None` ならモデルの既定。
    pub steps: Option<usize>,
    pub seed: Option<u64>,
    pub seconds: Option<f32>,
    pub duration_scale: f32,
    pub sampler: SamplerParams,
    pub max_text_len: Option<usize>,
    pub max_caption_len: Option<usize>,
    /// モデル本体を bf16 で持つか（既定は fp32）。
    pub model_bf16: bool,
    pub codec_bf16: bool,
}

pub fn model_steps(model: &str) -> Option<usize> {
    MODEL_STEPS
        .iter()
        .find(|(id, _)| *id == model)
        .map(|(_, n)| *n)
}

/// 実際に回すステップ数（設定があればそれ、無ければモデルの既定）。
pub fn effective_steps(options: &GenerationOptions, model: &str) -> usize {
    options
        .steps
        .or_else(|| model_steps(model))
        .unwrap_or(DEFAULT_STEPS)
}

fn number(g: &Generation, key: &str) -> Option<f64> {
    g.get(key).and_then(Value::as_f64)
}

fn whole(g: &Generation, key: &str) -> Option<u64> {
    let v = g.get(key)?;
    v.as_u64().or_else(|| {
        v.as_f64()
            .filter(|f| *f >= 0.0 && f.fract() == 0.0)
            .map(|f| f as u64)
    })
}

fn text<'a>(g: &'a Generation, key: &str) -> Option<&'a str> {
    g.get(key).and_then(Value::as_str)
}

/// `fp32` / `bf16` だけを受け付ける。
fn precision_is_bf16(g: &Generation, key: &str) -> Result<bool, SynthError> {
    match text(g, key).unwrap_or("fp32") {
        "fp32" => Ok(false),
        "bf16" => Ok(true),
        _ => Err(SynthError::Invalid(
            "この版では fp32 か bf16 を選んでください。".into(),
        )),
    }
}

pub fn options_from(g: &Generation) -> Result<GenerationOptions, SynthError> {
    if text(g, "lora_adapter").is_some_and(|s| !s.trim().is_empty()) {
        return Err(SynthError::Invalid(
            "LoRAはこの版ではまだ使えません。".into(),
        ));
    }
    let (model_bf16, codec_bf16) = (
        precision_is_bf16(g, "model_precision")?,
        precision_is_bf16(g, "codec_precision")?,
    );

    let mut sampler = SamplerParams::default();
    let f = |key: &str, default: f32| number(g, key).map_or(default, |v| v as f32);
    sampler.cfg_text = f("cfg_scale_text", sampler.cfg_text);
    sampler.cfg_caption = f("cfg_scale_caption", sampler.cfg_caption);
    sampler.cfg_speaker = f("cfg_scale_speaker", sampler.cfg_speaker);
    // 上流と同じく、`cfg_scale` があれば 3 つとも上書きする。
    if let Some(all) = number(g, "cfg_scale") {
        (sampler.cfg_text, sampler.cfg_caption, sampler.cfg_speaker) =
            (all as f32, all as f32, all as f32);
    }
    sampler.cfg_min_t = f("cfg_min_t", sampler.cfg_min_t);
    sampler.cfg_max_t = f("cfg_max_t", sampler.cfg_max_t);
    sampler.sway_coeff = f("sway_coeff", sampler.sway_coeff);
    sampler.guidance = match text(g, "cfg_guidance_mode") {
        Some("joint") => GuidanceMode::Joint,
        Some("alternating") => GuidanceMode::Alternating,
        _ => GuidanceMode::Independent,
    };
    sampler.schedule = if text(g, "t_schedule_mode") == Some("sway") {
        TSchedule::Sway
    } else {
        TSchedule::Linear
    };
    sampler.speaker_kv_scale = number(g, "speaker_kv_scale").map(|v| v as f32);
    sampler.truncation_factor = number(g, "truncation_factor").map(|v| v as f32);
    sampler.rescale = match (number(g, "rescale_k"), number(g, "rescale_sigma")) {
        (Some(k), Some(sigma)) => Some((k as f32, sigma as f32)),
        _ => None,
    };

    Ok(GenerationOptions {
        steps: whole(g, "steps").map(|n| n as usize),
        seed: whole(g, "seed"),
        seconds: number(g, "seconds").map(|v| v as f32),
        duration_scale: f("duration_scale", 1.0),
        sampler,
        max_text_len: whole(g, "max_text_len").map(|n| n as usize),
        max_caption_len: whole(g, "max_caption_len").map(|n| n as usize),
        model_bf16,
        codec_bf16,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use irodori_core::sampler::{GuidanceMode, TSchedule};
    use irodori_store::generation::{default_generation, validate_generation};
    use serde_json::json;

    fn gen(values: serde_json::Value) -> irodori_store::generation::Generation {
        validate_generation(&values).unwrap()
    }

    #[test]
    fn defaults_map_to_the_upstream_defaults() {
        let o = options_from(&default_generation()).unwrap();
        assert_eq!((o.steps, o.seed, o.seconds), (None, None, None));
        assert_eq!(o.duration_scale, 1.0);
        let s = &o.sampler;
        assert_eq!((s.cfg_text, s.cfg_caption, s.cfg_speaker), (3.0, 4.0, 5.0));
        assert_eq!((s.cfg_min_t, s.cfg_max_t, s.sway_coeff), (0.5, 1.0, -1.0));
        assert_eq!(
            (s.guidance, s.schedule),
            (GuidanceMode::Independent, TSchedule::Linear)
        );
        assert_eq!(
            (s.truncation_factor, s.rescale, s.speaker_kv_scale),
            (None, None, None)
        );
        assert_eq!((o.max_text_len, o.max_caption_len), (None, None));
        assert!(!o.model_bf16 && !o.codec_bf16);
    }

    #[test]
    fn explicit_values_map_through() {
        let o = options_from(&gen(json!({
            "steps": 12, "seed": 77, "seconds": 3.5, "duration_scale": 1.25,
            "cfg_scale_text": 2.0, "cfg_scale_caption": 6.0, "cfg_scale_speaker": 1.5,
            "cfg_min_t": 0.25, "cfg_max_t": 0.75, "sway_coeff": -0.5,
            "speaker_kv_scale": 1.5, "truncation_factor": 0.8,
            "rescale_k": 1.2, "rescale_sigma": 0.5, "max_text_len": 100, "max_caption_len": 50,
        })))
        .unwrap();
        assert_eq!(
            (o.steps, o.seed, o.seconds, o.duration_scale),
            (Some(12), Some(77), Some(3.5), 1.25)
        );
        let s = &o.sampler;
        assert_eq!((s.cfg_text, s.cfg_caption, s.cfg_speaker), (2.0, 6.0, 1.5));
        assert_eq!((s.cfg_min_t, s.cfg_max_t, s.sway_coeff), (0.25, 0.75, -0.5));
        assert_eq!(
            (s.speaker_kv_scale, s.truncation_factor, s.rescale),
            (Some(1.5), Some(0.8), Some((1.2, 0.5)))
        );
        assert_eq!((o.max_text_len, o.max_caption_len), (Some(100), Some(50)));
    }

    #[test]
    fn a_single_cfg_scale_overrides_all_three_like_upstream() {
        let o = options_from(&gen(json!({"cfg_scale": 2.5}))).unwrap();
        let s = &o.sampler;
        assert_eq!((s.cfg_text, s.cfg_caption, s.cfg_speaker), (2.5, 2.5, 2.5));
    }

    #[test]
    fn guidance_and_schedule_names() {
        for (name, want) in [
            ("independent", GuidanceMode::Independent),
            ("alternating", GuidanceMode::Alternating),
        ] {
            assert_eq!(
                options_from(&gen(json!({"cfg_guidance_mode": name})))
                    .unwrap()
                    .sampler
                    .guidance,
                want,
                "{name}"
            );
        }
        let joint = options_from(&gen(
            json!({"cfg_guidance_mode": "joint", "cfg_scale": 3.0}),
        ))
        .unwrap();
        assert_eq!(joint.sampler.guidance, GuidanceMode::Joint);
        let sway = options_from(&gen(json!({"t_schedule_mode": "sway"}))).unwrap();
        assert_eq!(sway.sampler.schedule, TSchedule::Sway);
    }

    #[test]
    fn bf16_is_accepted_and_fp16_is_not() {
        let o = options_from(&gen(
            json!({"model_precision": "bf16", "codec_precision": "bf16"}),
        ))
        .unwrap();
        assert!(o.model_bf16 && o.codec_bf16);
        let err = options_from(&gen(json!({"model_precision": "fp16"})))
            .err()
            .unwrap();
        assert_eq!(
            err,
            SynthError::Invalid("この版では fp32 か bf16 を選んでください。".into())
        );
        let err = options_from(&gen(json!({"codec_precision": "fp16"})))
            .err()
            .unwrap();
        assert_eq!(
            err,
            SynthError::Invalid("この版では fp32 か bf16 を選んでください。".into())
        );
    }

    #[test]
    fn lora_adapter_is_refused() {
        let err = options_from(&gen(json!({"lora_adapter": "my/adapter"})))
            .err()
            .unwrap();
        assert_eq!(
            err,
            SynthError::Invalid("LoRAはこの版ではまだ使えません。".into())
        );
        assert!(options_from(&gen(json!({"lora_adapter": null}))).is_ok());
    }

    #[test]
    fn rescale_needs_both_values() {
        // 検証（validate_generation）は片方だけを弾くが、手で書き換えた設定が来ても片方だけでは有効にしない。
        let mut g = default_generation();
        g.insert("rescale_k".into(), json!(1.2));
        assert_eq!(
            options_from(&g).unwrap().sampler.rescale,
            None,
            "a lone value never enables rescaling"
        );
        g.insert("rescale_sigma".into(), json!(0.5));
        assert_eq!(options_from(&g).unwrap().sampler.rescale, Some((1.2, 0.5)));
    }

    #[test]
    fn model_steps_for_the_allow_list_only() {
        assert_eq!(model_steps("Aratako/Irodori-TTS-v4.1-Small"), Some(40));
        assert_eq!(
            model_steps("phasefield-audio/Irodori-TTS-v4.1-Anime"),
            Some(40)
        );
        assert_eq!(model_steps("Aratako/Irodori-TTS-v4.1-Small-MF"), Some(4));
        assert_eq!(model_steps("someone/else"), None);
        assert_eq!(model_steps(""), None);
    }

    #[test]
    fn steps_for_a_job_follow_the_setting_or_the_model() {
        assert_eq!(
            effective_steps(
                &options_from(&default_generation()).unwrap(),
                "Aratako/Irodori-TTS-v4.1-Small-MF"
            ),
            4
        );
        assert_eq!(
            effective_steps(
                &options_from(&default_generation()).unwrap(),
                "Aratako/Irodori-TTS-v4.1-Small"
            ),
            40
        );
        assert_eq!(
            effective_steps(
                &options_from(&gen(json!({"steps": 9}))).unwrap(),
                "Aratako/Irodori-TTS-v4.1-Small"
            ),
            9
        );
    }
}
