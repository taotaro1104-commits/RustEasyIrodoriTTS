# v2.0 互換 fixture（Python が作る正解データ）

Rust の保存層（`irodori-store`）が v2.0 と**同じ結果**を出すことを、v2.0 の実コードが作る fixture で固定する。

## 生成
```powershell
cd F:\claude\upstream-irodori
$env:UV_CACHE_DIR = "F:\claude\upstream-irodori-cache\uv"
uv run python ..\RustEasyIrodoriTTS\tools\compat\gen_fixtures.py --out ..\RustEasyIrodoriTTS\crates\irodori-store\tests\fixtures
```
`--only py_json_cases signature_cases` のように名前を絞れる。生成した JSON はコミットする（Rust のテストは Python なしで走る）。

## 内容（タスクごとに追加）
| fixture | 使うタスク | 内容 |
|---|---|---|
| `py_json_cases` | 2 | `json.dumps` の 2 種類の出力（`sort_keys` と `indent=2`） |
| `generation_cases`, `project_sample`, `voices_sample`, `settings_sample` | 3 | 生成設定の検証結果、実データ形式 |
| `splitter_cases`, `dictionary_cases` | 6 | 分割・推定時間・辞書置換 |
| `signature_cases` | 7 | 生成結果の署名（**互換の最重要項目**） |

## 往復テスト（Task 13）
`--make-data-dir <dir>` で v2.0 の `Store` が実データを作り、Rust が読み書きした後、`verify_python_reads.py` で Python が読み直す。
