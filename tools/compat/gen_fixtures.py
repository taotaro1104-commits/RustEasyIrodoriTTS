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
