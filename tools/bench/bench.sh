#!/usr/bin/env bash
# Rust 版 irodori-cli のピークメモリ / RTF を構成ごとに別プロセスで測る。
#   使い方: IRODORI_MODEL_DIR=<標準> IRODORI_MF_DIR=<MF> IRODORI_CODEC_DIR=<codec> tools/bench/bench.sh [秒数]
set -euo pipefail
SECONDS_LEN="${1:-2}"
BIN="${IRODORI_CLI:-target/release/irodori-cli.exe}"
TEXT="こんにちは、今日はいい天気ですね。"
OUT="$(mktemp -d)"

run() { # 名前 モデルdir 追加引数...
  local name="$1" dir="$2"; shift 2
  echo "=== $name"
  "$BIN" "$TEXT" --model-dir "$dir" --codec-dir "$IRODORI_CODEC_DIR" --seconds "$SECONDS_LEN" --seed 1 \
    --caption "落ち着いた女性の声" --report -o "$OUT/$name.wav" "$@" | grep -E "load:|peak"
}

run std-f32-resident  "$IRODORI_MODEL_DIR" --dtype f32  --memory resident
run std-f32-staged    "$IRODORI_MODEL_DIR" --dtype f32  --memory staged
run std-bf16-staged   "$IRODORI_MODEL_DIR" --dtype bf16 --memory staged
if [ -n "${IRODORI_MF_DIR:-}" ]; then
  run mf-f32-staged   "$IRODORI_MF_DIR" --dtype f32  --memory staged
  run mf-bf16-staged  "$IRODORI_MF_DIR" --dtype bf16 --memory staged
fi
