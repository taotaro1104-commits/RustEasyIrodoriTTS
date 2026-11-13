# Easy-Irodori-TTS（Rust 版）

日本語の音声合成エディタ「Easy-Irodori-TTS v2.0」を、省メモリを目的に Rust（Candle）で作り直したものです。
音声合成の本体は [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)（コード: MIT。モデルの重みには別の倫理上の条件があります）。

> **利用上の注意**: 本人の同意なしに声を複製して公開したり、なりすましに使ったり、人をだます目的・誤情報の拡散に使うことは禁止です。
> 詳しくは画面の「クレジット・注意事項」と、各モデルの利用条件を確認してください。

## 構成
| クレート | 役割 | 状態 |
|---|---|---|
| `irodori-core` | 推論コア（Candle の CPU 実装。正規化・トークナイザ・ModernBERT・DiT・DACVAE・サンプラ・音声処理） | M1 完了。Python 版と数値一致（`docs/bench.md`） |
| `irodori-cli` | テキスト→WAV のコマンド | M1 完了 |
| `irodori-store` | v2.0 互換の保存層（台本・話者・設定・辞書・分割・署名・編集操作・状態） | M2a 完了 |
| `irodori-app` | ローカルサーバ＋画面（v2.0 の editor を埋め込み） | M2a: 編集・辞書・設定・クレジットまで。声の登録・生成・書き出しは M2b 以降 |

## 起動（M2a）
```powershell
cargo run --release -p irodori-app
# 例: データフォルダとポートを指定し、ブラウザを自動で開かない
cargo run --release -p irodori-app -- --data-dir D:\easy-irodori --port 7860 --no-browser
```
- 既定のデータフォルダは実行ファイルと同じ場所。v2.0 と同じ形式（`projects/` `voices/` `config/` `outputs/`）で読み書きします。
- ポートは既定 7860。使用中なら次の空きを使います。起動時に表示される URL を開いてください（自動で開く設定も可）。
- 同じデータフォルダでの二重起動はできません。ログは `logs/editor.log`。
- 接続できるのは同じ PC（`127.0.0.1` / `localhost`）だけで、書き込み系の操作は画面に埋め込まれたトークンが必要です。

## 確認・テスト
```powershell
cargo test -p irodori-store -p irodori-app                  # 普段のテスト
cargo test -p irodori-core --release                        # 推論コア（実モデルのテストは環境変数が要る）
```
- 画面の通し確認の結果: `docs/m2a-verification.md`
- v2.0（Python）との互換確認: `tools/compat/README.md`（Python 環境が要るため手動）
- 設計・計画: `docs/superpowers/specs/` と `docs/superpowers/plans/`
