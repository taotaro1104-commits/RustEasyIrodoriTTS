//! コマンドライン引数。

use std::path::PathBuf;

use candle_core::DType;
use clap::{Parser, ValueEnum};
use irodori_core::pipeline::{MemoryMode, PipelineOptions, SynthRequest};
use irodori_core::nn::qlinear::QuantKind;
use irodori_core::sampler::{GuidanceMode, SamplerParams, TSchedule};

/// 長さの上限の既定値（秒）。上流の既定と同じ。
const DEFAULT_MAX_SECONDS: f32 = 30.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum DtypeArg {
    F32,
    Bf16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum MemoryArg {
    /// 全部品を常駐させる（連続生成向け）。
    Resident,
    /// 段階ごとに必要な部品だけ読み、使い終えたら解放する（ピークメモリ最小）。
    Staged,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum GuidanceArg {
    /// 条件ごとに無条件の分岐を持つ（上流の既定。窓の中で最大 4 回の forward）。
    Independent,
    /// 全条件を落とした 1 分岐だけ（3 つのスケールが同じ値のときだけ使える。窓の中で 2 回）。
    Joint,
    /// 1 ステップに 1 条件だけ無条件分岐を評価する（窓の中で 2 回）。
    Alternating,
}

/// Irodori-TTS（Rust/Candle）でテキストから WAV を生成する。
#[derive(Parser, Debug)]
#[command(name = "irodori-cli", version)]
pub struct Args {
    /// 読み上げるテキスト。
    pub text: String,

    /// `model.safetensors` と `tokenizer/` があるフォルダ。
    #[arg(long, env = "IRODORI_MODEL_DIR")]
    pub model_dir: PathBuf,

    /// codec の `weights.pth` があるフォルダ。
    #[arg(long, env = "IRODORI_CODEC_DIR")]
    pub codec_dir: PathBuf,

    /// 声の参照音声（WAV/MP3/FLAC/OGG）。省略すると参照なし。
    #[arg(long)]
    pub voice: Option<PathBuf>,

    /// 話し方・声質の説明文（caption）。
    #[arg(long)]
    pub caption: Option<String>,

    /// サンプリングのステップ数（既定: RF は 40、MeanFlow は 4）。
    #[arg(long)]
    pub steps: Option<usize>,

    /// 乱数シード（省略すると毎回ランダム）。
    #[arg(long)]
    pub seed: Option<u64>,

    /// 長さ（秒）を固定する。省略すると duration 予測。
    #[arg(long)]
    pub seconds: Option<f32>,

    /// 長さの上限（秒、既定 30）。長いテキストはこれを上げないと途中で詰まる／切れる。
    #[arg(long)]
    pub max_seconds: Option<f32>,

    /// CFG の分岐の取り方（RF のみ。MeanFlow には効かない）。
    #[arg(long, value_enum)]
    pub guidance: Option<GuidanceArg>,

    /// テキストの CFG スケール（既定 3）。
    #[arg(long)]
    pub cfg_text: Option<f32>,

    /// caption の CFG スケール（既定 4）。
    #[arg(long)]
    pub cfg_caption: Option<f32>,

    /// 声の参照の CFG スケール（既定 5）。
    #[arg(long)]
    pub cfg_speaker: Option<f32>,

    /// CFG を掛け始める時刻 t の下限（既定 0.5。大きくすると無条件分岐を評価するステップが減る）。
    #[arg(long)]
    pub cfg_min_t: Option<f32>,

    /// Sway Sampling の係数（指定すると時刻の刻みを Sway にする。上流の例は -1）。
    #[arg(long, allow_negative_numbers = true)]
    pub sway_coeff: Option<f32>,

    /// DiT ブロックの重みを K-quants で量子化する（q6k / q5k / q4k。f32 のときだけ。起動時に量子化するので読み込みは遅くなる）。
    #[arg(long)]
    pub dit_quant: Option<QuantKind>,

    /// 重みの精度。bf16 はメモリ半分（潜在の相対誤差は約 4%）。
    #[arg(long, value_enum, default_value = "f32")]
    pub dtype: DtypeArg,

    #[arg(long, value_enum, default_value = "staged")]
    pub memory: MemoryArg,

    /// 出力 WAV（モノラル 24bit）。
    #[arg(short, long, default_value = "out.wav")]
    pub output: PathBuf,

    /// 処理時間・RTF・ピークメモリを表示する。
    #[arg(long)]
    pub report: bool,
}

impl Args {
    pub fn pipeline_options(&self) -> PipelineOptions {
        PipelineOptions {
            dtype: match self.dtype {
                DtypeArg::F32 => DType::F32,
                DtypeArg::Bf16 => DType::BF16,
            },
            codec_dtype: DType::F32,
            dit_quant: self.dit_quant,
            memory: match self.memory {
                MemoryArg::Resident => MemoryMode::Resident,
                MemoryArg::Staged => MemoryMode::Staged,
            },
        }
    }

    /// 既定値（`SamplerParams::default()`）に、指定されたオプションだけを重ねる。
    fn sampler(&self) -> SamplerParams {
        let mut p = SamplerParams::default();
        if let Some(g) = self.guidance {
            p.guidance = match g {
                GuidanceArg::Independent => GuidanceMode::Independent,
                GuidanceArg::Joint => GuidanceMode::Joint,
                GuidanceArg::Alternating => GuidanceMode::Alternating,
            };
        }
        p.cfg_text = self.cfg_text.unwrap_or(p.cfg_text);
        p.cfg_caption = self.cfg_caption.unwrap_or(p.cfg_caption);
        p.cfg_speaker = self.cfg_speaker.unwrap_or(p.cfg_speaker);
        p.cfg_min_t = self.cfg_min_t.unwrap_or(p.cfg_min_t);
        if let Some(c) = self.sway_coeff {
            p.schedule = TSchedule::Sway;
            p.sway_coeff = c;
        }
        p
    }

    pub fn request(&self) -> SynthRequest {
        SynthRequest {
            text: self.text.clone(),
            caption: self.caption.clone(),
            ref_wav: self.voice.clone(),
            seed: self.seed,
            seconds: self.seconds,
            max_seconds: self.max_seconds.unwrap_or(DEFAULT_MAX_SECONDS),
            num_steps: self.steps,
            sampler: self.sampler(),
            ..SynthRequest::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use irodori_core::pipeline::MemoryMode;

    fn parse(args: &[&str]) -> Result<Args, clap::Error> {
        let mut v = vec!["irodori-cli", "--model-dir", "m", "--codec-dir", "c"];
        v.extend_from_slice(args);
        Args::try_parse_from(v)
    }

    #[test]
    fn minimal_invocation_uses_the_documented_defaults() {
        let a = parse(&["こんにちは"]).unwrap();
        assert_eq!(a.text, "こんにちは");
        assert_eq!(a.dtype, DtypeArg::F32);
        assert_eq!(a.memory, MemoryArg::Staged);
        assert_eq!(a.output.to_str().unwrap(), "out.wav");
        assert!(a.voice.is_none() && a.caption.is_none() && a.steps.is_none() && a.seed.is_none() && a.seconds.is_none());
        assert!(!a.report);
    }

    #[test]
    fn all_options_are_parsed() {
        let a = parse(&["x", "--voice", "ref.wav", "--caption", "明るい声", "--steps", "4", "--seed", "7", "--seconds", "2.5", "--dtype", "bf16", "--memory", "resident", "-o", "a.wav", "--report"]).unwrap();
        assert_eq!((a.steps, a.seed, a.seconds), (Some(4), Some(7), Some(2.5)));
        assert_eq!(a.dtype, DtypeArg::Bf16);
        assert_eq!(a.memory, MemoryArg::Resident);
        assert_eq!(a.output.to_str().unwrap(), "a.wav");
        assert!(a.report);
    }

    #[test]
    fn unknown_dtype_and_missing_text_are_rejected() {
        assert!(parse(&["x", "--dtype", "int3"]).is_err());
        assert!(parse(&[]).is_err());
        assert!(Args::try_parse_from(["irodori-cli", "x"]).is_err(), "model/codec dirs are required");
    }

    #[test]
    fn options_convert_to_pipeline_options() {
        let a = parse(&["x", "--dtype", "bf16"]).unwrap();
        let o = a.pipeline_options();
        assert_eq!(o.dtype, candle_core::DType::BF16);
        assert_eq!(o.codec_dtype, candle_core::DType::F32, "codec stays f32");
        assert_eq!(o.memory, MemoryMode::Staged);
    }

    #[test]
    fn max_seconds_option_raises_the_length_limit() {
        assert_eq!(parse(&["x"]).unwrap().request().max_seconds, 30.0, "default limit");
        assert_eq!(parse(&["x", "--max-seconds", "60"]).unwrap().request().max_seconds, 60.0);
    }

    #[test]
    fn sampler_options_default_to_the_core_defaults() {
        let r = parse(&["x"]).unwrap().request();
        let d = SamplerParams::default();
        assert_eq!(r.sampler.guidance, d.guidance);
        assert_eq!((r.sampler.cfg_text, r.sampler.cfg_caption, r.sampler.cfg_speaker), (d.cfg_text, d.cfg_caption, d.cfg_speaker));
        assert_eq!((r.sampler.cfg_min_t, r.sampler.cfg_max_t), (d.cfg_min_t, d.cfg_max_t));
        assert_eq!(r.sampler.schedule, TSchedule::Linear);
    }

    #[test]
    fn sampler_options_are_carried_into_the_request() {
        let a = parse(&["x", "--guidance", "joint", "--cfg-text", "4", "--cfg-caption", "4", "--cfg-speaker", "4", "--cfg-min-t", "0.7", "--sway-coeff", "-1"]).unwrap();
        let s = a.request().sampler;
        assert_eq!(s.guidance, GuidanceMode::Joint);
        assert_eq!((s.cfg_text, s.cfg_caption, s.cfg_speaker, s.cfg_min_t), (4.0, 4.0, 4.0, 0.7));
        assert_eq!((s.schedule, s.sway_coeff), (TSchedule::Sway, -1.0));
        let alt = parse(&["x", "--guidance", "alternating"]).unwrap().request().sampler;
        assert_eq!(alt.guidance, GuidanceMode::Alternating);
        assert!(parse(&["x", "--guidance", "sideways"]).is_err());
    }

    #[test]
    fn dit_quantization_is_off_by_default_and_parsed_when_given() {
        assert_eq!(parse(&["x"]).unwrap().pipeline_options().dit_quant, None);
        for (arg, want) in [("q6k", QuantKind::Q6K), ("q5k", QuantKind::Q5K), ("q4k", QuantKind::Q4K)] {
            let a = parse(&["x", "--dit-quant", arg]).unwrap();
            assert_eq!(a.pipeline_options().dit_quant, Some(want));
        }
        assert!(parse(&["x", "--dit-quant", "q8_0"]).is_err(), "q8_0 is slow in candle and not offered");
    }

    #[test]
    fn request_carries_the_user_options() {
        let a = parse(&["こんにちは", "--steps", "6", "--seed", "3", "--seconds", "1.5", "--caption", "c"]).unwrap();
        let r = a.request();
        assert_eq!((r.text.as_str(), r.num_steps, r.seed, r.seconds), ("こんにちは", Some(6), Some(3), Some(1.5)));
        assert_eq!(r.caption.as_deref(), Some("c"));
    }
}
