# M2b-3 検証（モデルの自動ダウンロード）

実ネットワーク（Hugging Face）・実ブラウザ（Playwright）・release ビルドで確認。

## 条件
- 起動: `irodori-app --data-dir F:\claude\m2b3-verify-data --models-dir F:\claude\m2b3-verify-data\models --port 7871`（モデルフォルダは空から開始）
- 操作: 声の登録 → ボイスデザイン「候補の音声を作成する」（標準モデル、候補数 1、CPU）
- 取得対象: `Aratako/Irodori-TTS-v4.1-Small`（model.safetensors 約 3.06GB、tokenizer 2 ファイル）と codec `Aratako/Semantic-DACVAE-Japanese-32dim`（weights.pth 約 430MB）。合計 約 3.3GB（画面の表示は「約3.5GB」の概算）

## 結果
| 項目 | 結果 |
|---|---|
| 未取得で操作 → 確認ダイアログ（約3.5GB） | OK |
| 確認を拒否 | OK。何も起きず、モデルフォルダは 0 バイトのまま（通信なし）。`01-declined.png` |
| 承認 → 進捗表示と「取得を中止」 | OK。「モデルを取得中：235MB／3.5GB（6%）」。`02-downloading.png` |
| 取得中に中止 | OK。ジョブは停止、`model.safetensors.incomplete`（269MB）が残り、`refs/main` なし（取得済みに見えない） |
| 再承認で再開 | OK。269MB から続きを取得 |
| 取得中にプロセスを強制終了 → 再起動 → 再承認 | OK。425MB 残 → そこから再開して完了 |
| 完了後に元の操作（候補作成）を自動でやり直す | OK。生成が走り、音声が付いた。`03-generated-after-download.png` |
| 取得ファイルの SHA-256 | OK。4 ファイルすべて既存キャッシュ（`F:\claude\upstream-irodori-cache\hf\hub`）と一致 |
| 完了後の状態 | `.incomplete` 無し、`refs/main` は 2 リポジトリとも存在 |

## 時間
- 取得（再開 2 回を含む）: 約 5.5 分（02:50:46 の再開から 02:56:14 の完了。合計ではそれ以前の約 1 分を含め 約 6.5 分）。約 9〜10MB/s。
- 続く生成（CPU、候補 1）: 約 336 秒（sampling 274 秒、decode 54 秒）。

## 自動テスト
`cargo test -p irodori-core -p irodori-store -p irodori-app`: 567 件成功、失敗 0（テストは実ネットワークに出ない）。

## 未検証・未解決
- **ネットワークを切った状態**: 機内モード相当にできないため実機では確認していない。偽サーバのテスト（`plan_maps_http_errors_and_offline` ほか）で担保。
- プロキシ環境（`HTTPS_PROXY`）、並列取得、取得済みモデルの削除・更新は対象外（計画の持ち越し）。
- 検証用データは `F:\claude\m2b3-verify-data`（約 3.3GB）。不要なら削除してよい。
