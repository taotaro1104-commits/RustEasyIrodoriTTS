"""Python の文字判定を Rust に写す表を生成する（`easy_splitter.seconds` の互換用）。

v2.0 は Python 3.11（Unicode 14）で動く。`str.isspace` / `str.isdigit` / `unicodedata.category` は Python の版で結果が変わるので、
Rust 側は判定そのものを書かず、同じ判定の結果を範囲表として埋め込む。

    uv run --python 3.11 python tools/compat/gen_unicode_tables.py > crates/irodori-store/src/unicode_tables.rs
"""
from __future__ import annotations

import sys
import unicodedata


def ranges(predicate) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = None
    for cp in range(0x110000):
        ok = predicate(chr(cp))
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, 0x10FFFF))
    return out


TABLES = {
    "PY_SPACE": lambda c: c.isspace(),
    "PY_DIGIT": lambda c: c.isdigit(),
    "LETTER_OR_NUMBER": lambda c: unicodedata.category(c)[0] in ("L", "N"),
    "SYMBOL_OTHER": lambda c: unicodedata.category(c) == "So",
}


def main() -> None:
    out = [
        f"//! 自動生成: tools/compat/gen_unicode_tables.py（Python {sys.version.split()[0]}, Unicode {unicodedata.unidata_version}）。",
        "//! 手で編集しない。Python の文字判定と同じ結果を返す範囲表。",
        "",
    ]
    for name, predicate in TABLES.items():
        table = ranges(predicate)
        out.append(f"pub static {name}: &[(u32, u32)] = &[")
        for i in range(0, len(table), 6):
            out.append("    " + " ".join(f"(0x{a:X}, 0x{b:X})," for a, b in table[i : i + 6]))
        out.append("];")
        out.append("")
    out.append("/// `table`（昇順・非重複の閉区間）に `c` が含まれるか。")
    out.append("pub fn in_table(table: &[(u32, u32)], c: char) -> bool {")
    out.append("    let cp = c as u32;")
    out.append("    table.binary_search_by(|&(lo, hi)| if cp < lo { std::cmp::Ordering::Greater } else if cp > hi { std::cmp::Ordering::Less } else { std::cmp::Ordering::Equal }).is_ok()")
    out.append("}")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
