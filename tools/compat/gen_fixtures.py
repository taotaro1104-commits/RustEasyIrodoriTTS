"""v2.0（Python）の実コードを使って、Rust 側の互換テスト用 fixture を生成する。

実行（上流の venv で。v2.0 のコードが `regex` などを import するため）:
    cd F:\\claude\\upstream-irodori
    uv run python ..\\RustEasyIrodoriTTS\\tools\\compat\\gen_fixtures.py --out ..\\RustEasyIrodoriTTS\\crates\\irodori-store\\tests\\fixtures
各タスクが `@fixture("名前")` で関数を追加する。関数は JSON にできる値を返す。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[2]
V2_DIR = REPO / "EasyIrodoriTTS_v2.0" / "EasyIrodoriTTS_v2.0"
sys.path.insert(0, str(V2_DIR))

FIXTURES: dict[str, Callable[[], object]] = {}


def fixture(name: str):
    """`<name>.json` を作る関数を登録するデコレータ。"""

    def deco(fn: Callable[[], object]):
        FIXTURES[name] = fn
        return fn

    return deco


@fixture("py_json_cases")
def py_json_cases() -> list[dict]:
    """Task 2: `json.dumps` の出力（sort_keys 版と indent=2 版）。入力は JSON 文字列。"""
    inputs = [
        "1", "1.0", "0.75", "-0.0", "0.0", "1e-05", "1e+16", "1e16", "1e15", "123456789.123",
        "0.0001", "0.00001", "1.5e-07", "123456789012345678.0", "1e22", "3.0", "-1.0", "2.5", "100", "0.1",
        "9007199254740993", "-42", "true", "false", "null",
        '"日本語"', '"abc"', '"\\u0007"', '"\\u007f"', '"\\ud834\\udd1e"', '"line\\nbreak\\ttab"',
        '"quote\\" and back\\\\slash"', '"\\u001f"', '"😀 emoji"', '""',
        "[]", "{}", "[1, 2.0, 3.5]", '{"b": 1, "a": 2}', '{"b": {"z": 1, "y": [1, {"k": null}]}, "a": []}',
        '{"日本語": "値", "a": 1.0, "B": 2}', '{"": 1, " ": 2}',
        '[[], {}, [[]], {"a": {}}]', '{"speed": 1.0, "silence": 0.0, "steps": null, "seed": null, "flag": true}',
    ]
    cases = []
    for text in inputs:
        value = json.loads(text)
        cases.append(dict(
            input=text,
            sorted=json.dumps(value, ensure_ascii=False, sort_keys=True),
            pretty=json.dumps(value, ensure_ascii=False, indent=2),
        ))
    return cases


@fixture("generation_cases")
def generation_cases() -> list[dict]:
    """Task 3: `easy_storage.validate_generation` の入力と結果（または ValueError の文）。"""
    from easy_storage import validate_generation

    inputs = [
        {}, {"steps": "40"}, {"steps": 40}, {"steps": 40.9}, {"steps": 0}, {"steps": 121}, {"steps": ""}, {"steps": None},
        {"seed": 5}, {"seed": ""}, {"seed": 2**32}, {"seed": -1}, {"seed": "7"},
        {"duration_scale": 2}, {"duration_scale": 0.4}, {"duration_scale": 1}, {"duration_scale": "1.25"},
        {"cfg_scale_text": 11}, {"cfg_scale_caption": -1}, {"cfg_scale_speaker": 0}, {"cfg_scale_speaker": 10},
        {"cfg_min_t": 0.9, "cfg_max_t": 0.5}, {"cfg_min_t": 0, "cfg_max_t": 1},
        {"sway_coeff": -2}, {"sway_coeff": 1.5}, {"sway_coeff": 0},
        {"t_schedule_mode": "bad"}, {"t_schedule_mode": "sway"}, {"cfg_guidance_mode": "joint"}, {"cfg_guidance_mode": "x"},
        {"seconds": 1}, {"seconds": 31}, {"seconds": ""}, {"cfg_scale": 3}, {"speaker_kv_scale": 0.001}, {"speaker_kv_scale": 2},
        {"max_text_len": 100}, {"max_text_len": 257}, {"max_caption_len": 512}, {"max_caption_len": 513}, {"max_text_len": 12.7},
        {"truncation_factor": 0.5}, {"truncation_factor": 11},
        {"rescale_k": 1.5}, {"rescale_sigma": 2}, {"rescale_k": 1.5, "rescale_sigma": 2}, {"rescale_k": 0, "rescale_sigma": 1},
        {"lora_adapter": " x "}, {"lora_adapter": ""}, {"lora_adapter": None},
        {"model_precision": "bf16", "codec_precision": "fp16"}, {"model_precision": "fp64"}, {"codec_precision": 3},
        {"context_kv_cache": False}, {"context_kv_cache": 0}, {"unknown_key": 1, "steps": 4},
        {"cfg_scale_text": 3, "cfg_scale_caption": 4, "cfg_scale_speaker": 5},
    ]
    cases = []
    for values in inputs:
        try:
            cases.append(dict(input=values, output=validate_generation(values)))
        except ValueError as exc:
            cases.append(dict(input=values, error=str(exc)))
    return cases


@fixture("project_sample")
def project_sample() -> dict:
    """Task 3: 実データ形式の台本（生成結果つきカードと、将来の未知フィールドを含む）。"""
    from easy_storage import new_card, new_project

    project = new_project("サンプル台本")
    project["id"] = "0123456789abcdef0123456789abcdef"
    project["revision"] = 7
    project["future_project_field"] = {"nested": [1, 2.5, None]}
    first = project["cards"][0]
    first.update(id="a" * 32, text="こんにちは。今日は晴れです。", voice_id="b" * 32, speed=0.75, silence=1.5,
                 filename="01", status="完了", error="",
                 result=dict(path="outputs/internal/r1/01.wav", signature="deadbeef", seed=123, record="r1",
                             request=dict(text="こんにちは。", ordinal=1, date="20261113", dictionary={"今日": "きょう"},
                                          card=dict(id="a" * 32, text="こんにちは。今日は晴れです。")),
                             export=dict(wav="outputs/exports/001_x_20261113.wav")))
    first["future_card_field"] = "keep me"
    second = new_card(first)
    second.update(id="c" * 32, text="二枚目", status="失敗", error="boom")
    second["missing_voice_name"] = "消えた話者"
    project["cards"].append(second)
    project["selected"] = second["id"]
    return project


@fixture("voices_sample")
def voices_sample() -> list[dict]:
    """Task 3: 話者一覧（sound_revision 無し・creation あり・将来の未知フィールドを含む）。"""
    return [
        dict(id="b" * 32, revision=3, audio="voices/" + "b" * 32 + "/" + "d" * 32 + ".wav", name="話者A",
             model="Aratako/Irodori-TTS-v4.1-Small", caption="落ち着いた女性の声", sound_revision=2,
             creation=dict(mode="design", model="Aratako/Irodori-TTS-v4.1-Small", candidates=2), future_voice_field=[1, 2]),
        dict(id="e" * 32, revision=1, audio="voices/" + "e" * 32 + "/" + "f" * 32 + ".wav", name="話者B",
             model="Aratako/Irodori-TTS-v4.1-Small-MF", caption=""),
    ]


@fixture("settings_sample")
def settings_sample() -> dict:
    """Task 3: 既定値（v2.0 の `Store`）＋保存された値＋将来の未知フィールド。"""
    import tempfile
    from easy_storage import Store

    with tempfile.TemporaryDirectory() as tmp:
        store = Store(tmp)
        defaults = dict(store.settings)
    saved = dict(defaults, theme="dark", txt=True, last_project="0123456789abcdef0123456789abcdef", future_setting="x")
    return dict(defaults=defaults, saved=saved)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = args.only or sorted(FIXTURES)
    unknown = [n for n in names if n not in FIXTURES]
    if unknown:
        raise SystemExit(f"unknown fixtures: {unknown} (available: {sorted(FIXTURES)})")
    for name in names:
        value = FIXTURES[name]()
        (args.out / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {name}.json")


if __name__ == "__main__":
    main()
