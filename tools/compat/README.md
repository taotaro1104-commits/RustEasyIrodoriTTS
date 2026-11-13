# v2.0 互換 fixture（Python が作る正解データ）

Rust の保存層（`irodori-store`）が v2.0 と**同じ結果**を出すことを、v2.0 の実コードが作る fixture で固定する。

## 生成
```powershell
cd F:\claude\upstream-irodori
$env:UV_CACHE_DIR = "F:\claude\upstream-irodori-cache\uv"
uv run python ..\RustEasyIrodoriTTS\tools\compat\gen_fixtures.py --out ..\RustEasyIrodoriTTS\crates\irodori-store\tests\fixtures
```
`splitter_cases` は v2.0 と同じ Python 3.11（Unicode 14）で、`regex` 付きで作る: `uv run --python 3.11 --with regex python ...`。
`unicode_tables.rs`（`isspace`/`isdigit`/カテゴリの範囲表）は `tools/compat/gen_unicode_tables.py` が同じ Python で生成する
（Rust 側は判定を書かず表を引くので、Python の版による差が出ない）。
`--only py_json_cases signature_cases` のように名前を絞れる。生成した JSON はコミットする（Rust のテストは Python なしで走る）。

## 内容（タスクごとに追加）
| fixture | 使うタスク | 内容 |
|---|---|---|
| `py_json_cases` | 2 | `json.dumps` の 2 種類の出力（`sort_keys` と `indent=2`） |
| `generation_cases`, `project_sample`, `voices_sample`, `settings_sample` | 3 | 生成設定の検証結果、実データ形式 |
| `splitter_cases`, `dictionary_cases` | 6 | 分割・推定時間・辞書置換 |
| `signature_cases` | 7 | 生成結果の署名（**互換の最重要項目**） |

## 画面のメタ情報（Task 10）
`gen_meta.py` が `crates/irodori-app/data/{emoji_palette.json,credits.md}` を作る（上流の絵文字一覧・v2.0 のグループ・クレジット文）。
上流のリポジトリで実行する:
```powershell
cd F:\claude\upstream-irodori
uv run --python 3.11 --with regex python ..\RustEasyIrodoriTTS\tools\compat\gen_meta.py --out ..\RustEasyIrodoriTTS\crates\irodori-app\data
```
クレジット（制作者表記・ライセンス・禁止事項）は画面から消さない。

## 往復テスト（Task 13）
v2.0 の `Store` が作った実データを Rust が読み書きして壊さないこと、Rust が書いたフォルダを v2.0 が読めることを確かめる。
Python 環境が要るので、テストは `#[ignore]`（普段の `cargo test` では動かない）。

```powershell
# 1. Python（v2.0 の実コード）でデータフォルダを作る（空のフォルダを指定）
cd F:\claude\upstream-irodori
$env:UV_CACHE_DIR = "F:\claude\upstream-irodori-cache\uv"
uv run --python 3.11 --with regex python ..\RustEasyIrodoriTTS\tools\compat\gen_fixtures.py --make-data-dir F:\claude\compat-data

# 2. Rust のテストを動かす（データは一時フォルダに写して使うので、元は変わらない）
cd F:\claude\RustEasyIrodoriTTS
$env:IRODORI_COMPAT_DIR = "F:\claude\compat-data"
$env:IRODORI_UPSTREAM_DIR = "F:\claude\upstream-irodori"
cargo test -p irodori-store --test compat_roundtrip -- --ignored
```

| テスト | 確かめること |
|---|---|
| `python_made_data_loads_edits_and_saves_without_losing_fields` | 編集して保存しても、未知の項目・キーの並び・設定・話者一覧が一切変わらない（変わるのは文章・revision・中断扱いの状態だけ） |
| `signatures_of_python_results_still_match` | Python が計算した署名と Rust の署名が一致し、結果が引き続き有効と判定される |
| `state_for_python_data_has_no_regeneration_needed_for_unchanged_cards` | 変更のないカードが「再生成が必要」にならない。中断・話者なしの状態も v2.0 と同じ |
| `rust_written_folder_is_readable_by_python` | `verify_python_reads.py` が、Rust の書いたフォルダを v2.0 の `Store` で読み直して検証する |

最後に確認した結果: 4 件すべて成功（2026-11-14、Python 3.11.3）。`verify_python_reads.py` は単独でも動く（フォルダに `compat_rust_expect.json` が必要）。
