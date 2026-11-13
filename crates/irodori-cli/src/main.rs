mod args;
mod mem;

use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::Device;
use clap::Parser;
use irodori_core::audio::write_wav;
use irodori_core::pipeline::Pipeline;

use args::Args;

fn main() -> Result<()> {
    let args = Args::parse();
    let model = args.model_dir.join("model.safetensors");
    let codec = args.codec_dir.join("weights.pth");

    let t0 = Instant::now();
    let pipeline = Pipeline::load_with(&model, &codec, &Device::Cpu, args.pipeline_options())
        .with_context(|| format!("failed to load model from {}", args.model_dir.display()))?;
    let load_secs = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let out = pipeline.synthesize(&args.request()).context("synthesis failed")?;
    let synth_secs = t1.elapsed().as_secs_f64();

    for w in &out.warnings {
        eprintln!("warning: {w}");
    }
    write_wav(&args.output, &out.samples, out.sample_rate).with_context(|| format!("failed to write {}", args.output.display()))?;
    let audio_secs = out.samples.len() as f64 / out.sample_rate as f64;
    println!("wrote {} ({audio_secs:.2}s, seed {})", args.output.display(), out.seed);

    if args.report {
        println!("load: {load_secs:.2}s, synthesis: {synth_secs:.2}s, RTF: {:.2}", synth_secs / audio_secs.max(1e-9));
        match mem::peak_memory_mb() {
            Some(mb) => println!("peak memory: {mb:.0} MB"),
            None => println!("peak memory: unavailable"),
        }
    }
    Ok(())
}
