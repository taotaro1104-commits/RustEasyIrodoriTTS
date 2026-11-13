# ゴールデンテンソルの生成（Python 版との数値一致検証用）

Rust 側のテストは `IRODORI_GOLDEN_DIR` が未設定なら skip する。実モデルとの一致検証をするときだけ以下を行う。

```powershell
# 1. 上流を固定コミットで取得（このリポジトリの外に置く）
git clone https://github.com/Aratako/Irodori-TTS ..\upstream-irodori
git -C ..\upstream-irodori checkout 89f9d8fbd4d51ea019867ee1197725ede1df13c5
cd ..\upstream-irodori
uv sync --frozen --no-dev --extra cpu --python 3.11

# 2. ゴールデンを書き出す（約3.5GB の重みを初回ダウンロード）
uv run python ..\RustEasyIrodoriTTS\tools\golden\dump_golden.py --out ..\RustEasyIrodoriTTS\golden

# 3. Rust 側で検証
$env:IRODORI_GOLDEN_DIR = "F:\claude\RustEasyIrodoriTTS\golden"
$env:IRODORI_MODEL_DIR  = "<HF cache の Irodori-TTS-v4.1-Small スナップショット>"
cargo test --workspace
```

`dump_golden.py` は各タスクが `@dumper("名前")` で関数を追加していく。
各関数は固定 seed の入力と Python 版の出力を `{name}.safetensors` に保存する。
`golden/` は git 管理外。
