//! コマンドライン引数。

use std::path::PathBuf;

use candle_core::DType;
use clap::{Parser, ValueEnum};
use irodori_core::pipeline::{MemoryMode, PipelineOptions, SynthRequest};

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
            memory: match self.memory {
                MemoryArg::Resident => MemoryMode::Resident,
                MemoryArg::Staged => MemoryMode::Staged,
            },
        }
    }

    pub fn request(&self) -> SynthRequest {
        SynthRequest {
            text: self.text.clone(),
            caption: self.caption.clone(),
            ref_wav: self.voice.clone(),
            seed: self.seed,
            seconds: self.seconds,
            num_steps: self.steps,
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
    fn request_carries_the_user_options() {
        let a = parse(&["こんにちは", "--steps", "6", "--seed", "3", "--seconds", "1.5", "--caption", "c"]).unwrap();
        let r = a.request();
        assert_eq!((r.text.as_str(), r.num_steps, r.seed, r.seconds), ("こんにちは", Some(6), Some(3), Some(1.5)));
        assert_eq!(r.caption.as_deref(), Some("c"));
    }
}
