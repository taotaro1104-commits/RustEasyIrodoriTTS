"""Rust 版が書いたデータフォルダを、v2.0（Python）の実コードで読めるか確認する（手動）。

確認すること:
  1. `Store(root)` が voices/index.json と全台本を読める。
  2. 各話者の WAV（`audio`）が存在し、読める（soundfile）。
  3. カードの生成結果の署名が、v2.0 の `Service.result_matches` で「現行の結果」と判定される。

実行（Rust 版を止めてから。上流の venv で）:
    cd F:\\claude\\upstream-irodori
    uv run --python 3.11 --with regex python ..\\RustEasyIrodoriTTS\\tools\\compat\\check_voices.py <データフォルダ>
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "EasyIrodoriTTS_v2.0" / "EasyIrodoriTTS_v2.0"))


def main() -> int:
    import soundfile as sf
    from easy_service import Service
    from easy_storage import Store

    root = Path(sys.argv[1])
    store = Store(root)
    service = Service.__new__(Service)  # エンジンは不要
    service.store = store
    dictionary = service.dictionary() if hasattr(service, "dictionary") else {}
    problems = 0

    print(f"voices: {len(store.voices)}")
    for voice in store.voices:
        path = root / voice["audio"]
        try:
            info = sf.info(path)
            print(f"  {voice['name']!r} rev={voice['revision']} sound_revision={voice.get('sound_revision')} "
                  f"{info.samplerate}Hz ch={info.channels} {info.subtype} {info.duration:.1f}s creation={'yes' if voice.get('creation') else 'no'}")
        except Exception as exc:  # noqa: BLE001
            problems += 1
            print(f"  NG {voice['name']!r}: {exc}")

    for project_file in sorted((root / "projects").glob("*.json")):
        doc = store.load(project_file.stem)
        for index, card in enumerate(doc["cards"], 1):
            result = card.get("result")
            if not result:
                continue
            matches = service.result_matches(doc, card, dictionary)
            exists = Path(result["path"]).is_file()
            print(f"  project {doc['name']!r} card {index}: status={card['status']} signature_matches={matches} wav_exists={exists}")
            if card["status"] == "完了" and not (matches and exists):
                problems += 1
    print("OK" if problems == 0 else f"PROBLEMS: {problems}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
