//! モデル設定（safetensors メタデータの `config_json` / `text_encoder_config_json`）。
//!
//! 既定値は上流 `config.py` の `ModelConfig` に合わせる（欠落フィールド用）。未知キーは無視する。

use candle_transformers::models::modernbert::Config as ModernBertConfig;
use serde::Deserialize;

use crate::{IrodoriError, Result};

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    /// `rf_velocity`（既定）または `meanflow`（`delta_t` 条件つき）。
    pub flow_parameterization: String,
    pub latent_dim: usize,
    pub latent_patch_size: usize,
    pub model_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub mlp_ratio: f64,
    pub text_mlp_ratio: Option<f64>,
    pub speaker_mlp_ratio: Option<f64>,
    pub text_vocab_size: usize,
    pub text_tokenizer_repo: String,
    pub text_add_bos: bool,
    pub text_encoder_type: String,
    pub pretrained_projector_type: String,
    pub pretrained_projector_hidden_ratio: f64,
    pub text_dim: usize,
    pub text_layers: usize,
    pub text_heads: usize,
    pub use_caption_condition: bool,
    pub use_speaker_condition: Option<bool>,
    pub caption_vocab_size: Option<usize>,
    pub caption_tokenizer_repo: Option<String>,
    pub caption_add_bos: Option<bool>,
    pub caption_dim: Option<usize>,
    pub caption_layers: Option<usize>,
    pub caption_heads: Option<usize>,
    pub caption_mlp_ratio: Option<f64>,
    pub speaker_dim: usize,
    pub speaker_layers: usize,
    pub speaker_heads: usize,
    pub speaker_patch_size: usize,
    pub timestep_embed_dim: usize,
    pub adaln_rank: usize,
    pub norm_eps: f64,
    pub use_duration_predictor: bool,
    pub duration_aux_dim: usize,
    pub duration_hidden_dim: usize,
    pub duration_layers: usize,
    pub duration_attention_heads: usize,
    pub duration_architecture: String,
    pub duration_token_init_frames: f64,
    pub duration_speaker_fusion: String,
    pub duration_caption_fusion: String,
    pub duration_caption_pooling: String,
    pub max_text_len: usize,
    pub max_caption_len: Option<usize>,
    pub ref_max_seconds: f64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            flow_parameterization: "rf_velocity".into(),
            latent_dim: 128,
            latent_patch_size: 1,
            model_dim: 2048,
            num_layers: 24,
            num_heads: 16,
            mlp_ratio: 2.875,
            text_mlp_ratio: Some(2.6),
            speaker_mlp_ratio: Some(2.6),
            text_vocab_size: 102400,
            text_tokenizer_repo: "sbintuitions/sarashina2.2-0.5b".into(),
            text_add_bos: true,
            text_encoder_type: "scratch".into(),
            pretrained_projector_type: "linear".into(),
            pretrained_projector_hidden_ratio: 2.0,
            text_dim: 1280,
            text_layers: 14,
            text_heads: 10,
            use_caption_condition: false,
            use_speaker_condition: None,
            caption_vocab_size: None,
            caption_tokenizer_repo: None,
            caption_add_bos: None,
            caption_dim: None,
            caption_layers: None,
            caption_heads: None,
            caption_mlp_ratio: None,
            speaker_dim: 1280,
            speaker_layers: 14,
            speaker_heads: 10,
            speaker_patch_size: 1,
            timestep_embed_dim: 512,
            adaln_rank: 256,
            norm_eps: 1e-5,
            use_duration_predictor: false,
            duration_aux_dim: 14,
            duration_hidden_dim: 1024,
            duration_layers: 3,
            duration_attention_heads: 8,
            duration_architecture: "token_sum_adarn_zero_no_aux".into(),
            duration_token_init_frames: 9.0,
            duration_speaker_fusion: "adarn_zero".into(),
            duration_caption_fusion: "adarn_zero".into(),
            duration_caption_pooling: "masked_mean".into(),
            max_text_len: 256,
            max_caption_len: None,
            ref_max_seconds: 120.0,
        }
    }
}

impl ModelConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    pub fn patched_latent_dim(&self) -> usize {
        self.latent_dim * self.latent_patch_size
    }

    pub fn speaker_patched_latent_dim(&self) -> usize {
        self.patched_latent_dim() * self.speaker_patch_size
    }

    /// 旧 caption 設定は speaker 分岐なしを意味する（上流互換）。
    pub fn use_speaker_condition_resolved(&self) -> bool {
        self.use_speaker_condition.unwrap_or(!self.use_caption_condition)
    }

    pub fn use_pretrained_text_encoder(&self) -> bool {
        self.text_encoder_type.trim().eq_ignore_ascii_case("pretrained")
    }

    pub fn text_mlp_ratio_resolved(&self) -> f64 {
        self.text_mlp_ratio.unwrap_or(self.mlp_ratio)
    }

    pub fn caption_vocab_size_resolved(&self) -> usize {
        self.caption_vocab_size.unwrap_or(self.text_vocab_size)
    }

    pub fn caption_add_bos_resolved(&self) -> bool {
        self.caption_add_bos.unwrap_or(self.text_add_bos)
    }

    pub fn caption_dim_resolved(&self) -> usize {
        self.caption_dim.unwrap_or(self.text_dim)
    }

    pub fn caption_layers_resolved(&self) -> usize {
        self.caption_layers.unwrap_or(self.text_layers)
    }

    pub fn caption_heads_resolved(&self) -> usize {
        self.caption_heads.unwrap_or(self.text_heads)
    }

    pub fn caption_mlp_ratio_resolved(&self) -> f64 {
        self.caption_mlp_ratio.unwrap_or_else(|| self.text_mlp_ratio_resolved())
    }

    pub fn speaker_mlp_ratio_resolved(&self) -> f64 {
        self.speaker_mlp_ratio.unwrap_or(self.mlp_ratio)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParams {
    pub rope_theta: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub sliding_attention: RopeParams,
    pub full_attention: RopeParams,
}

/// HF `ModernBertConfig`（バックボーン）の必要部分。
#[derive(Debug, Clone, Deserialize)]
pub struct TextEncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub norm_eps: f64,
    pub pad_token_id: u32,
    pub bos_token_id: u32,
    pub local_attention: usize,
    pub global_attn_every_n_layers: usize,
    pub rope_parameters: RopeParameters,
    #[serde(default)]
    pub layer_types: Vec<String>,
}

impl TextEncoderConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    /// `layer_types` が「`i % global_attn_every_n_layers == 0` の層だけ full」と整合するか確認する。
    pub fn validate(&self) -> Result<()> {
        let n = self.global_attn_every_n_layers.max(1);
        if !self.layer_types.is_empty() && self.layer_types.len() != self.num_hidden_layers {
            return Err(IrodoriError::InvalidConfig(format!(
                "layer_types has {} entries, expected {}",
                self.layer_types.len(),
                self.num_hidden_layers
            )));
        }
        for (i, t) in self.layer_types.iter().enumerate() {
            let want_full = i % n == 0;
            if (t == "full_attention") != want_full {
                return Err(IrodoriError::InvalidConfig(format!(
                    "layer {i} is {t}, but global_attn_every_n_layers={n} implies {}",
                    if want_full { "full_attention" } else { "sliding_attention" }
                )));
            }
        }
        Ok(())
    }

    pub fn into_candle(&self) -> ModernBertConfig {
        ModernBertConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            intermediate_size: self.intermediate_size,
            max_position_embeddings: self.max_position_embeddings,
            layer_norm_eps: self.norm_eps,
            pad_token_id: self.pad_token_id,
            global_attn_every_n_layers: self.global_attn_every_n_layers,
            global_rope_theta: self.rope_parameters.full_attention.rope_theta,
            local_attention: self.local_attention,
            local_rope_theta: self.rope_parameters.sliding_attention.rope_theta,
            classifier_config: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const META: &str = include_str!("../tests/fixtures/v4_1_small_meta.json");

    fn meta_field(name: &str) -> String {
        let v: serde_json::Value = serde_json::from_str(META).unwrap();
        v[name].as_str().unwrap().to_string()
    }

    #[test]
    fn parses_real_v4_1_small_model_config() {
        let cfg = ModelConfig::from_json(&meta_field("config_json")).unwrap();
        assert_eq!((cfg.model_dim, cfg.num_layers, cfg.num_heads), (1280, 12, 20));
        assert_eq!((cfg.latent_dim, cfg.latent_patch_size), (32, 1));
        assert_eq!((cfg.speaker_dim, cfg.speaker_layers, cfg.speaker_patch_size), (768, 8, 4));
        assert_eq!(cfg.adaln_rank, 192);
        assert_eq!(cfg.max_text_len, 256);
        assert_eq!(cfg.max_caption_len, Some(512));
        assert_eq!(cfg.caption_dim_resolved(), 512);
        assert!(cfg.use_speaker_condition_resolved());
        assert_eq!(cfg.patched_latent_dim(), 32);
        assert_eq!(cfg.speaker_patched_latent_dim(), 128);
        assert!(cfg.use_pretrained_text_encoder());
    }

    #[test]
    fn missing_fields_fall_back_to_python_defaults() {
        let cfg = ModelConfig::from_json("{}").unwrap();
        assert_eq!((cfg.model_dim, cfg.num_layers, cfg.text_dim), (2048, 24, 1280));
        assert!((cfg.text_mlp_ratio_resolved() - 2.6).abs() < 1e-9);
    }

    #[test]
    fn legacy_caption_config_implies_no_speaker_branch() {
        let cfg = ModelConfig::from_json(r#"{"use_caption_condition": true}"#).unwrap();
        assert!(!cfg.use_speaker_condition_resolved());
    }

    #[test]
    fn invalid_json_is_an_error() {
        assert!(ModelConfig::from_json("{not json").is_err());
    }

    #[test]
    fn text_encoder_config_converts_to_candle() {
        let te = TextEncoderConfig::from_json(&meta_field("text_encoder_config_json")).unwrap();
        te.validate().unwrap();
        let c = te.into_candle();
        assert_eq!((c.hidden_size, c.num_hidden_layers, c.num_attention_heads), (768, 25, 12));
        assert_eq!(c.intermediate_size, 3072);
        assert_eq!(c.global_attn_every_n_layers, 3);
        assert_eq!(c.local_attention, 128);
        assert_eq!(c.global_rope_theta, 160000.0);
        assert_eq!(c.local_rope_theta, 10000.0);
        assert_eq!(c.pad_token_id, 3);
    }

    #[test]
    fn layer_types_inconsistent_with_global_every_n_is_rejected() {
        let mut te = TextEncoderConfig::from_json(&meta_field("text_encoder_config_json")).unwrap();
        te.layer_types[3] = "sliding_attention".into();
        assert!(te.validate().is_err());
    }
}
