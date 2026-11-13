"""Rust が書いたデータフォルダを、v2.0（Python）の実コードで読み直して検証する。

    cd F:\\claude\\upstream-irodori
    uv run --python 3.11 --with regex python ..\\RustEasyIrodoriTTS\\tools\\compat\\verify_python_reads.py <データフォルダ>

フォルダには、Rust のテストが書いた `compat_rust_expect.json`（期待する内容）が要る。成功なら `OK` を出して終了コード 0。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "EasyIrodoriTTS_v2.0" / "EasyIrodoriTTS_v2.0"))


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def main() -> None:
    from easy_service import Service
    from easy_storage import Store

    root = Path(sys.argv[1])
    expect = json.loads((root / "compat_rust_expect.json").read_text(encoding="utf-8"))
    store = Store(root)  # 設定・話者一覧の形式検証を通る

    doc = store.load(expect["pid"])
    check(doc["revision"] == expect["revision"], f"revision {doc['revision']} != {expect['revision']}")
    card = next(c for c in doc["cards"] if c["id"] == expect["edited_card_id"])
    check(card["text"] == expect["edited_text"], "edited text")
    check(doc["name"] == expect["name"], "project name")
    check(doc.get("future_project_field") == {"nested": [1, 2.5, None]}, "unknown project field kept")
    check(doc["cards"][0].get("future_card_field") == "keep me", "unknown card field kept")
    busy = next(c for c in doc["cards"] if c["id"] == expect["busy_card_id"])
    check(busy["status"] == "中断（再生成できます）", f"interrupted status was {busy['status']}")
    check(store.settings.get("future_setting") == "x" and store.settings["theme"] == "dark", "settings kept")
    check(store.settings["last_project"] == expect["pid"], "last_project")
    check(store.voices[0].get("future_voice_field") == [1, 2], "unknown voice field kept")

    entries = json.loads((root / "config" / "reading_dictionary.json").read_text(encoding="utf-8-sig"))
    check(isinstance(entries, dict) and all(isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip() for k, v in entries.items()), "dictionary entries valid")
    check(list(entries.items()) == [tuple(p) for p in expect["dictionary"]], f"dictionary order/content {list(entries.items())}")
    check((root / "config" / "reading_dictionary.json").read_text(encoding="utf-8").endswith("}\n"), "dictionary ends with newline")

    service = Service.__new__(Service)
    service.store = store
    for card_id in expect["unchanged_result_cards"]:
        c = next(c for c in doc["cards"] if c["id"] == card_id)
        check(service.result_matches(doc, c, entries), f"result of {card_id} no longer matches")
    check(len(store.list_projects()) >= 2, "projects listed")
    print("OK")


if __name__ == "__main__":
    main()
