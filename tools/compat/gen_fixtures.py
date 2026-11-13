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


@fixture("splitter_cases")
def splitter_cases() -> list[dict]:
    """Task 6: `easy_splitter.estimate` と `split_text`（結果または ValueError の文）。"""
    from easy_splitter import estimate, seconds, split_text

    sentence = "今日はとても良い天気ですので、公園まで散歩に出かけることにしました。"
    dicts = [
        {},
        {"Irodori": "いろどり", "TTS": "てぃーてぃーえす", "日本橋": "にほんばし"},
        {"長い": "あ" * 200},
        {"AB": "えーびー", "ABC": "えーびーしー"},
    ]
    texts = [
        "", "   ", "\n\n", "こんにちは。", "こんにちは", "Hello world. This is a test!", "a b c",
        "Irodori TTS version 2.0 は 2026年11月13日に公開されました。日本橋で会いましょう。",
        "ABC and AB and ABCD", "😀😀😀", "👨‍👩‍👧‍👦 家族", "🇯🇵 日本", "é café", "ｱｲｳ１２３ａｂｃ", "²³①⑩⅓٣５",
        "行1\n行2\n\n行3", "\t先頭タブ", "終わり。\n", sentence, sentence * 3, sentence * 12, "あ" * 400,
        "アルファベットのlongwordwithoutanyspaces" + "x" * 40 + "と続く文章です。", "、" * 300, "ABC" * 100, "長い",
        "Irodori" * 50, "あ。" * 120, "한국어 문장입니다. 안녕하세요.", "ひらがな カタカナ 漢字 Latin 123 !?",
    ]
    cases = []
    for scale in (1.0, 1.5, 0.5):
        for dictionary in dicts:
            for text in texts:
                case = dict(text=text, dictionary=dictionary, scale=scale)
                try:
                    case["estimate"] = estimate(text, dictionary, scale)
                    case["split"] = split_text(text, dictionary, scale)
                except ValueError as exc:
                    case["error"] = str(exc)
                cases.append(case)
    chars = ["。", "！", "、", ",", " ", "漢", "5", "a", "é", "あ", "ー", "①", "²", "٣", "😀", "☃", "\n", "　", "-"]
    return dict(cases=cases, seconds={c: seconds(c) for c in chars})


@fixture("dictionary_cases")
def dictionary_cases() -> list[dict]:
    """Task 6: 読み辞書の置換（最長一致、置換後は再置換しない）。`easy_dictionary.apply` と同じ式。"""
    import re

    def apply(text, entries):
        if not entries:
            return text
        pattern = "|".join(re.escape(k) for k in sorted(entries, key=len, reverse=True))
        return re.sub(pattern, lambda m: entries[m.group()], text)

    entries = {"Irodori": "いろどり", "Iro": "いろ", "TTS": "てぃーてぃーえす", "日本橋": "にほんばし", "日本": "にほん", "a.b": "えーどっとびー", "(x)": "かっこx"}
    texts = ["Irodori TTS", "Iroha", "日本橋と日本", "a.b a-b", "(x) と x", "", "Irodori" * 3, "いろどり", "TTSTTS"]
    cases = [dict(text=t, entries=entries, output=apply(t, entries)) for t in texts]
    cases.append(dict(text="abc", entries={}, output="abc"))
    cases.append(dict(text="ab ba", entries={"a": "b", "b": "a"}, output=apply("ab ba", {"a": "b", "b": "a"})))
    return cases


@fixture("signature_cases")
def signature_cases() -> dict:
    """Task 7: `Service.signature / result_matches / playback_stamp`（v2.0 の実コード）。"""
    import copy
    import os
    import tempfile
    from easy_service import Service
    from easy_storage import Store, new_card, new_project

    service = Service.__new__(Service)  # エンジンは不要。署名は純粋関数＋話者一覧だけを使う
    tmp = tempfile.mkdtemp()
    service.store = Store(tmp)
    voices = voices_sample()
    service.store.voices = copy.deepcopy(voices)
    vid_a, vid_b = voices[0]["id"], voices[1]["id"]
    dicts = [{}, {"Irodori": "いろどり", "TTS": "てぃーてぃーえす"}, {"日本": "にほん", "日本橋": "にほんばし"}]
    generations = [
        {},
        {"steps": 4, "seed": 123, "duration_scale": 1.25},
        {"lora_adapter": "x/ y", "context_kv_cache": False, "cfg_min_t": 0.25},
        {"model_precision": "bf16", "truncation_factor": 0.9, "rescale_k": 1.5, "rescale_sigma": 0.5},
    ]
    texts = ["", "こんにちは。", "Irodori TTS で 日本橋 へ", "😀 emoji 👨‍👩‍👧‍👦", 'quote " and \\ back\nslash\t']
    cases = []
    for text in texts:
        for voice_id in ("", vid_a, vid_b, "0" * 32):
            for speed, silence in ((1.0, 0.0), (0.75, 1.5), (1.5, 10.0)):
                for g, extra in enumerate(generations):
                    dictionary = dicts[(len(cases)) % 3]
                    enabled = len(cases) % 4 != 3
                    project = new_project("x")
                    project["dictionary_enabled"] = enabled
                    project["generation"] = service_generation(project["generation"], extra)
                    card = project["cards"][0]
                    card.update(text=text, voice_id=voice_id, speed=speed, silence=silence)
                    reading = service.reading_text(project, card, dictionary)
                    sig = service.signature(project, card, dictionary)
                    old_rules = dicts[1]
                    legacy_sig = service.input_signature(project, card, old_rules)
                    cases.append(dict(
                        project=project, dictionary=dictionary, reading=reading, signature=sig,
                        matches_current=_matches(service, project, card, dictionary, dict(signature=sig)),
                        legacy=dict(
                            signature=legacy_sig,
                            request=dict(dictionary=old_rules, text=reading),
                            matches=_matches(service, project, card, dictionary, dict(signature=legacy_sig, request=dict(dictionary=old_rules, text=reading))),
                        ),
                        legacy_wrong_text=dict(
                            signature=legacy_sig,
                            request=dict(dictionary=old_rules, text=reading + "x"),
                            matches=_matches(service, project, card, dictionary, dict(signature=legacy_sig, request=dict(dictionary=old_rules, text=reading + "x"))),
                        ),
                        stale=dict(signature="0" * 64, matches=_matches(service, project, card, dictionary, dict(signature="0" * 64))),
                    ))
    cases = cases[::3]  # 約 80 件に間引く（組み合わせを偏らせない）
    # 再生スタンプ: 日本語のパス・状態・存在するファイル／しないファイル
    stamps = []
    cwd = os.getcwd()
    workdir = tempfile.mkdtemp()
    os.chdir(workdir)
    try:
        for dictionary in dicts:
            project = new_project("スタンプ")
            base = project["cards"][0]
            base.update(text="一枚目", voice_id=vid_a)
            c2, c3, c4 = new_card(base), new_card(base), new_card(base)
            c2.update(text="二枚目", status="生成中")
            c3.update(text="三枚目", status="失敗", result=dict(path="outputs/internal/日本語/03.wav", signature="s"))
            c4.update(text="Irodori", status="完了", result=dict(path="outputs/internal/r/04.wav", signature="s"))
            base.update(status="完了", result=dict(path="outputs/internal/r/01.wav", signature="s"))
            project["cards"] += [c2, c3, c4]
            existing = ["outputs/internal/r/01.wav"]
            for rel in existing:
                os.makedirs(os.path.dirname(rel), exist_ok=True)
                open(rel, "wb").close()
            stamps.append(dict(project=project, dictionary=dictionary, voices=voices, existing=existing,
                               stamp=service.playback_stamp(project, dictionary)))
    finally:
        os.chdir(cwd)
    return dict(voices=voices, cases=cases, stamps=stamps)


def service_generation(base: dict, overrides: dict) -> dict:
    out = dict(base)
    out.update(overrides)
    return out


def _matches(service, project, card, dictionary, result) -> bool:
    card = dict(card, result=result)
    return bool(service.result_matches(project, card, dictionary))


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
