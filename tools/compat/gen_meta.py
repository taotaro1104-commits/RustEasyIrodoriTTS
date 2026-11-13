"""`/api/meta` 用のデータを v2.0 と上流の実物から生成する。

    cd F:\\claude\\upstream-irodori
    uv run --python 3.11 --with regex python ..\\RustEasyIrodoriTTS\\tools\\compat\\gen_meta.py --out ..\\RustEasyIrodoriTTS\\crates\\irodori-app\\data

- `emoji_palette.json`: 上流の `EMOJI_PALETTE_ITEMS`（そのままの並び）と、v2.0 の `grouped_emojis` によるグループ。
- `credits.md`: v2.0 の `easy_ui.footer_text()`（制作者表記・ライセンス・禁止事項。UI から消さないこと）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V2_DIR = REPO / "EasyIrodoriTTS_v2.0" / "EasyIrodoriTTS_v2.0"
sys.path.insert(0, str(V2_DIR))
sys.path.insert(0, str(Path.cwd()))  # 上流のリポジトリ（irodori_tts）で実行する


def item(e) -> dict:
    return dict(emoji=e.emoji, label=e.label, description=e.description)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from irodori_tts.gradio_emoji_palette import EMOJI_PALETTE_ITEMS
    from easy_text_tools import grouped_emojis
    from easy_ui import footer_text

    palette = dict(
        emojis=[item(e) for e in EMOJI_PALETTE_ITEMS],
        emoji_groups=[dict(title=title, items=[item(e) for e in items]) for title, items in grouped_emojis(EMOJI_PALETTE_ITEMS)],
    )
    (args.out / "emoji_palette.json").write_text(json.dumps(palette, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / "credits.md").write_text(footer_text(), encoding="utf-8", newline="\n")
    print(f"wrote {len(palette['emojis'])} emojis in {len(palette['emoji_groups'])} groups, credits.md")


if __name__ == "__main__":
    main()
