"""Python 版 Irodori-TTS の中間テンソルを safetensors に書き出す。

上流リポジトリ（commit 89f9d8fb…）の環境で実行すること。README 参照。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import torch
from safetensors.torch import save_file

DUMPERS: dict[str, Callable[[], dict[str, torch.Tensor]]] = {}


def dumper(name: str):
    """`name.safetensors` を作る関数を登録するデコレータ。"""

    def deco(fn: Callable[[], dict[str, torch.Tensor]]):
        DUMPERS[name] = fn
        return fn

    return deco


@dumper("tokens")
def dump_tokens() -> dict[str, torch.Tensor]:
    """Task 2: PretrainedTextTokenizer.batch_encode の出力。環境変数 IRODORI_MODEL_DIR が必要。"""
    import os

    from irodori_tts.tokenizer import PretrainedTextTokenizer
    from transformers import AutoTokenizer

    model_dir = Path(os.environ["IRODORI_MODEL_DIR"])
    hf_tok = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"), use_fast=True)
    tok = PretrainedTextTokenizer(hf_tok, add_bos=True)
    ids, mask = tok.batch_encode(["こんにちは、世界。", "短い"], max_length=32)
    return {"ids": ids.to(torch.int64), "mask": mask.to(torch.uint8)}


@dumper("nn_basics")
def dump_nn_basics() -> dict[str, torch.Tensor]:
    """Task 4: rope / half-rope / RMSNorm / LowRankAdaLN / SwiGLU / timestep embedding。"""
    from irodori_tts.model import (
        JointAttention,
        LowRankAdaLN,
        RMSNorm,
        SwiGLU,
        apply_rotary_emb,
        get_timestep_embedding,
        precompute_freqs_cis,
    )

    torch.manual_seed(0)
    out: dict[str, torch.Tensor] = {}

    freqs = precompute_freqs_cis(64, 50, 10000.0)
    x = torch.randn(2, 50, 4, 64)
    out["rope.x"] = x
    out["rope.y"] = apply_rotary_emb(x, freqs)
    out["rope.half_y"] = JointAttention._apply_rotary_half(None, x, freqs)

    norm = RMSNorm((4, 64), eps=1e-5)
    norm.weight.data.normal_(1.0, 0.1)
    out["rms.weight"] = norm.weight.data
    out["rms.y"] = norm(x)

    ada = LowRankAdaLN(32, 8, 1e-5)
    for p in ada.parameters():
        p.data.normal_(0.0, 0.2)
    for name, p in ada.named_parameters():
        out[f"ada.{name}"] = p.data
    ax = torch.randn(2, 5, 32)
    cond = torch.randn(2, 5, 96)
    y, gate = ada(ax, cond)
    out.update({"ada.x": ax, "ada.cond": cond, "ada.y": y, "ada.gate": gate})

    swi = SwiGLU(16, 40)
    for name, p in swi.named_parameters():
        p.data.normal_(0.0, 0.3)
        out[f"swi.{name}"] = p.data
    sx = torch.randn(3, 7, 16)
    out.update({"swi.x": sx, "swi.y": swi(sx)})

    t = torch.tensor([0.0, 0.25, 0.9])
    out.update({"ts.t": t, "ts.y": get_timestep_embedding(t, 512)})
    return out


def _load_checkpoint(prefixes: tuple[str, ...]) -> tuple[dict, dict[str, torch.Tensor]]:
    """IRODORI_MODEL_DIR/model.safetensors からメタデータと指定接頭辞のテンソルを読む。"""
    import os

    from safetensors import safe_open

    path = Path(os.environ["IRODORI_MODEL_DIR"]) / "model.safetensors"
    with safe_open(str(path), "pt") as f:
        meta = f.metadata()
        tensors = {k: f.get_tensor(k) for k in f.keys() if k.startswith(prefixes)}
    return meta, tensors


def _strip(tensors: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}


@dumper("text_enc")
def dump_text_enc() -> dict[str, torch.Tensor]:
    """Task 5: ModernBERT バックボーン → projector → norm（text/caption）。"""
    import json
    import os

    from irodori_tts.model import PretrainedConditionProjector, PretrainedTextBackbone, RMSNorm
    from irodori_tts.tokenizer import PretrainedTextTokenizer
    from transformers import AutoTokenizer

    meta, w = _load_checkpoint(
        ("pretrained_text_backbone.", "text_encoder.", "caption_encoder.", "text_norm.", "caption_norm.")
    )
    cfg = json.loads(meta["config_json"])
    backbone = PretrainedTextBackbone(
        "unused", config_dict=json.loads(meta["text_encoder_config_json"]), load_pretrained_weights=False
    )
    backbone.load_state_dict(_strip(w, "pretrained_text_backbone."))
    backbone.eval()

    def head(prefix: str, norm_name: str, dim: int):
        proj = PretrainedConditionProjector(
            backbone.hidden_size, dim, projector_type=cfg["pretrained_projector_type"],
            hidden_ratio=cfg["pretrained_projector_hidden_ratio"], norm_eps=cfg["norm_eps"],
        )
        proj.load_state_dict(_strip(w, prefix))
        norm = RMSNorm(dim, eps=cfg["norm_eps"])
        norm.load_state_dict(_strip(w, norm_name))
        return proj.eval(), norm.eval()

    model_dir = Path(os.environ["IRODORI_MODEL_DIR"])
    tok = PretrainedTextTokenizer(AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"), use_fast=True))
    ids, mask = tok.batch_encode(["こんにちは、世界。", "短い"], max_length=32)

    out: dict[str, torch.Tensor] = {"ids": ids.to(torch.int64), "mask": mask.to(torch.uint8)}
    with torch.no_grad():
        for key, prefix, norm_name, dim in (
            ("text_state", "text_encoder.", "text_norm.", cfg["text_dim"]),
            ("caption_state", "caption_encoder.", "caption_norm.", cfg.get("caption_dim") or cfg["text_dim"]),
        ):
            proj, norm = head(prefix, norm_name, dim)
            out[key] = norm(proj(backbone, ids, mask))
    return out


def _model_config(meta: dict):
    import dataclasses
    import json

    from irodori_tts.config import ModelConfig

    raw = json.loads(meta["config_json"])
    names = {f.name for f in dataclasses.fields(ModelConfig)}
    return ModelConfig(**{k: v for k, v in raw.items() if k in names})


@dumper("speaker_enc")
def dump_speaker_enc() -> dict[str, torch.Tensor]:
    """Task 6: patch_sequence_with_mask と ReferenceLatentEncoder（実重み）。"""
    from irodori_tts.model import ReferenceLatentEncoder, patch_sequence_with_mask

    meta, w = _load_checkpoint(("speaker_encoder.",))
    cfg = _model_config(meta)
    enc = ReferenceLatentEncoder(cfg)
    enc.load_state_dict(_strip(w, "speaker_encoder."))
    enc.eval()
    torch.manual_seed(1)
    latent = torch.randn(1, 103, cfg.latent_dim)
    mask = torch.zeros(1, 103, dtype=torch.bool)
    mask[:, :90] = True
    patched, pmask = patch_sequence_with_mask(latent, mask, cfg.speaker_patch_size)
    with torch.no_grad():
        state = enc(patched, pmask)
    return {"latent": latent, "mask": mask.to(torch.uint8), "patched": patched, "state": state}


@dumper("dit_step")
def dump_dit_step() -> dict[str, torch.Tensor]:
    """Task 7: 条件エンコード全体と DiT 1 ステップ（各ブロック出力つき）。"""
    import json
    import os

    from irodori_tts.model import TextToLatentRFDiT
    from irodori_tts.tokenizer import PretrainedTextTokenizer
    from transformers import AutoTokenizer

    model_dir = Path(os.environ["IRODORI_MODEL_DIR"])
    meta, w = _load_checkpoint(("",))
    cfg = _model_config(meta)
    model = TextToLatentRFDiT(
        cfg,
        pretrained_backbone_config=json.loads(meta["text_encoder_config_json"]),
        load_pretrained_backbone_weights=False,
    )
    model.load_state_dict(w, strict=True)
    model.eval()

    tok = PretrainedTextTokenizer(AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"), use_fast=True))
    text_ids, text_mask = tok.batch_encode(["こんにちは、世界。"], max_length=32)
    cap_ids, cap_mask = tok.batch_encode(["落ち着いた女性の声"], max_length=32)
    torch.manual_seed(2)
    ref_latent = torch.randn(1, 103, cfg.latent_dim)
    ref_mask = torch.zeros(1, 103, dtype=torch.bool)
    ref_mask[:, :90] = True
    x_t = torch.randn(1, 50, cfg.latent_dim)
    t = torch.tensor([0.5])

    out: dict[str, torch.Tensor] = {}
    hooks = [
        blk.register_forward_hook(lambda _m, _i, o, k=f"blk{i}": out.__setitem__(k, o))
        for i, blk in enumerate(model.blocks)
    ]
    with torch.no_grad():
        ts, tm, ss, sm, cs, cm = model.encode_conditions(
            text_input_ids=text_ids, text_mask=text_mask, ref_latent=ref_latent, ref_mask=ref_mask,
            caption_input_ids=cap_ids, caption_mask=cap_mask,
        )
        v = model.forward_with_encoded_conditions(x_t, t, ts, tm, ss, sm, cs, cm)
        duration = model.predict_duration_log_frames(
            text_state=ts, text_mask=tm, speaker_state=ss, speaker_mask=sm,
            duration_features=torch.zeros(1, cfg.duration_aux_dim),
            has_speaker=torch.tensor([True]), caption_state=cs, caption_mask=cm,
            has_caption=torch.tensor([True]),
        )
    for h in hooks:
        h.remove()
    out.update({
        "text_ids": text_ids.to(torch.int64), "text_mask": text_mask.to(torch.uint8),
        "caption_ids": cap_ids.to(torch.int64), "caption_mask": cap_mask.to(torch.uint8),
        "ref_latent": ref_latent, "ref_mask": ref_mask.to(torch.uint8),
        "text_state": ts, "speaker_state": ss, "speaker_mask": sm.to(torch.uint8),
        "caption_state": cs, "x_t": x_t, "t": t, "velocity": v, "duration": duration,
    })
    return out


@dumper("codec")
def dump_codec() -> dict[str, torch.Tensor]:
    """Task 9: DACVAE の決定的 encode と、透かし無効の decode（実重み）。IRODORI_CODEC_DIR が必要。"""
    import os

    from irodori_tts.codec import DACVAECodec

    codec = DACVAECodec.load(
        repo_id=str(Path(os.environ["IRODORI_CODEC_DIR"]) / "weights.pth"),
        device="cpu", dtype=None, normalize_db=None,
    )
    torch.manual_seed(3)
    n = 95000  # hop(1920) の倍数ではない → reflect パディング経路も検証する
    t = torch.arange(n) / 48000.0
    wav = (0.4 * torch.sin(2 * torch.pi * 220 * t) + 0.2 * torch.sin(2 * torch.pi * 1330 * t)
           + 0.05 * torch.randn(n)).reshape(1, 1, n)
    latent = codec.encode_waveform(wav, 48000, normalize_db=None)
    decoded = codec.decode_latent(latent)
    return {"wav": wav, "latent": latent, "decoded": decoded}


@dumper("loudness")
def dump_loudness() -> dict[str, torch.Tensor]:
    """Task 10: audiotools の loudness / normalize(-16) / ensure_max（ゲーティングが効く信号）。"""
    from audiotools import AudioSignal

    sr = 48000
    t = torch.arange(sr // 2) / sr
    loud = 0.5 * torch.sin(2 * torch.pi * 300 * t)
    quiet = 0.01 * torch.sin(2 * torch.pi * 700 * t)
    mid = 0.3 * torch.sin(2 * torch.pi * 1200 * t) + 0.1 * torch.sin(2 * torch.pi * 90 * t)
    wav = torch.cat([loud, quiet, torch.zeros(sr // 2), mid]).reshape(1, 1, -1)
    sig = AudioSignal(wav.clone(), sr)
    loudness = sig.loudness().clone()
    sig.normalize(-16.0)
    sig.ensure_max_of_audio()
    return {"wav": wav, "loudness": loudness.reshape(1), "normalized": sig.audio_data.clone()}


@dumper("e2e")
def dump_e2e() -> dict[str, torch.Tensor]:
    """Task 11: 参照なし・1 秒・8 ステップの RF 合成（同一ノイズ）。潜在と decode 後の波形を保存。"""
    import os

    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey
    from irodori_tts.rf import _make_rng, sample_euler_rf_cfg
    from irodori_tts.text_normalization import normalize_text

    model_dir = Path(os.environ["IRODORI_MODEL_DIR"])
    rt = InferenceRuntime.from_key(RuntimeKey(
        checkpoint=str(model_dir / "model.safetensors"), model_device="cpu",
        codec_repo=str(Path(os.environ["IRODORI_CODEC_DIR"]) / "weights.pth"),
        model_precision="fp32", codec_device="cpu", codec_precision="fp32",
    ))
    cfg = rt.model_cfg
    text_ids, text_mask = rt.tokenizer.batch_encode([normalize_text("こんにちは、世界。").strip()], max_length=rt.default_text_max_len)
    cap_ids, cap_mask = rt.caption_tokenizer.batch_encode(["落ち着いた女性の声"], max_length=rt.default_caption_max_len)
    ref_len = max(1, int(cfg.speaker_patch_size))
    ref_latent = torch.zeros(1, ref_len, cfg.latent_dim * cfg.latent_patch_size)
    ref_mask = torch.zeros(1, ref_len, dtype=torch.bool)
    steps_len, seed = 25, 0
    rng, _ = _make_rng(seed, rt.model_device)
    noise = torch.randn((1, steps_len, cfg.patched_latent_dim), generator=rng)
    z = sample_euler_rf_cfg(
        model=rt.model, text_input_ids=text_ids, text_mask=text_mask, ref_latent=ref_latent,
        ref_mask=ref_mask, sequence_length=steps_len, caption_input_ids=cap_ids, caption_mask=cap_mask,
        num_steps=8, cfg_scale_text=3.0, cfg_scale_caption=4.0, cfg_scale_speaker=5.0, seed=seed,
    )
    audio = rt.codec.decode_latent(z)
    return {"noise": noise, "latent": z, "audio": audio}


@dumper("e2e_ref")
def dump_e2e_ref() -> dict[str, torch.Tensor]:
    """声の参照つき合成（12 秒・24kHz の参照 WAV → ラウドネス正規化 → encode → speaker CFG）。
    参照 WAV は `$IRODORI_GOLDEN_DIR/ref_24k.wav` に書き出す（Rust 側のテストが同じファイルを読む）。"""
    import os

    import numpy as np
    import soundfile as sf
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
    from irodori_tts.rf import _make_rng, sample_euler_rf_cfg
    from irodori_tts.text_normalization import normalize_text

    rate, seconds = 24000, 12
    n = rate * seconds
    t = np.arange(n) / rate
    noise_rng = np.random.default_rng(5)
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))
    wav = 0.3 * envelope * (np.sin(2 * np.pi * 140 * t) + 0.5 * np.sin(2 * np.pi * 280 * t)
                             + 0.25 * np.sin(2 * np.pi * 420 * t)) + 0.02 * noise_rng.standard_normal(n)
    ref_path = Path(os.environ["IRODORI_GOLDEN_DIR"]) / "ref_24k.wav"
    sf.write(str(ref_path), wav.astype("float32"), rate, subtype="PCM_16")

    model_dir = Path(os.environ["IRODORI_MODEL_DIR"])
    rt = InferenceRuntime.from_key(RuntimeKey(
        checkpoint=str(model_dir / "model.safetensors"), model_device="cpu",
        codec_repo=str(Path(os.environ["IRODORI_CODEC_DIR"]) / "weights.pth"),
        model_precision="fp32", codec_device="cpu", codec_precision="fp32",
    ))
    cfg = rt.model_cfg
    ref_latent, ref_mask = rt._load_reference_latent(
        req=SamplingRequest(text="x", ref_wav=str(ref_path)), batch_size=1, messages=[]
    )
    text_ids, text_mask = rt.tokenizer.batch_encode([normalize_text("こんにちは、世界。").strip()], max_length=rt.default_text_max_len)
    cap_ids, cap_mask = rt.caption_tokenizer.batch_encode(["落ち着いた女性の声"], max_length=rt.default_caption_max_len)
    steps_len, seed = 25, 0
    rng, _ = _make_rng(seed, rt.model_device)
    noise = torch.randn((1, steps_len, cfg.patched_latent_dim), generator=rng)
    z = sample_euler_rf_cfg(
        model=rt.model, text_input_ids=text_ids, text_mask=text_mask, ref_latent=ref_latent, ref_mask=ref_mask,
        sequence_length=steps_len, caption_input_ids=cap_ids, caption_mask=cap_mask,
        num_steps=8, cfg_scale_text=3.0, cfg_scale_caption=4.0, cfg_scale_speaker=5.0, seed=seed,
    )
    audio = rt.codec.decode_latent(z)

    # 切り分け用: 参照音声の中間段（リサンプル後 / ラウドネス正規化後）
    import torchaudio
    from irodori_tts.inference_runtime import _load_audio

    loaded, loaded_sr = _load_audio(str(ref_path))
    wav48 = torchaudio.functional.resample(loaded.mean(dim=0, keepdim=True), loaded_sr, rt.codec.sample_rate)
    wav_norm = rt.codec._normalize_loudness(wav48[0], rt.codec.sample_rate, -16.0)
    return {"noise": noise, "latent": z, "audio": audio, "ref_latent": ref_latent,
            "wav48": wav48.reshape(-1), "wav_norm": wav_norm.reshape(-1)}


@dumper("e2e_mf")
def dump_e2e_mf() -> dict[str, torch.Tensor]:
    """Task 12: MeanFlow（Small-MF）の 1 秒・4 ステップ合成（同一ノイズ）。IRODORI_MF_DIR が必要。"""
    import os

    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey
    from irodori_tts.meanflow import sample_euler_meanflow
    from irodori_tts.rf import _make_rng
    from irodori_tts.text_normalization import normalize_text

    rt = InferenceRuntime.from_key(RuntimeKey(
        checkpoint=str(Path(os.environ["IRODORI_MF_DIR"]) / "model.safetensors"), model_device="cpu",
        codec_repo=str(Path(os.environ["IRODORI_CODEC_DIR"]) / "weights.pth"),
        model_precision="fp32", codec_device="cpu", codec_precision="fp32",
    ))
    cfg = rt.model_cfg
    text_ids, text_mask = rt.tokenizer.batch_encode([normalize_text("こんにちは、世界。").strip()], max_length=rt.default_text_max_len)
    cap_ids, cap_mask = rt.caption_tokenizer.batch_encode(["落ち着いた女性の声"], max_length=rt.default_caption_max_len)
    ref_len = max(1, int(cfg.speaker_patch_size))
    ref_latent = torch.zeros(1, ref_len, cfg.latent_dim * cfg.latent_patch_size)
    ref_mask = torch.zeros(1, ref_len, dtype=torch.bool)
    steps_len, seed = 25, 0
    rng, _ = _make_rng(seed, rt.model_device)
    noise = torch.randn((1, steps_len, cfg.patched_latent_dim), generator=rng)
    z = sample_euler_meanflow(
        model=rt.model, text_input_ids=text_ids, text_mask=text_mask, ref_latent=ref_latent, ref_mask=ref_mask,
        sequence_length=steps_len, caption_input_ids=cap_ids, caption_mask=cap_mask, num_steps=4, seed=seed,
    )
    audio = rt.codec.decode_latent(z)
    return {"noise": noise, "latent": z, "audio": audio}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="*", help="指定した名前だけ生成")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = args.only or sorted(DUMPERS)
    unknown = [n for n in names if n not in DUMPERS]
    if unknown:
        raise SystemExit(f"unknown dumpers: {unknown} (available: {sorted(DUMPERS)})")
    for name in names:
        tensors = {k: v.detach().cpu().contiguous().clone() for k, v in DUMPERS[name]().items()}
        save_file(tensors, str(args.out / f"{name}.safetensors"))
        print(f"wrote {name}: {len(tensors)} tensors")


if __name__ == "__main__":
    main()
