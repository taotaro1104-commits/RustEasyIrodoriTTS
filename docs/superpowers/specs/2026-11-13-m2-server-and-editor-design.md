# M2 設計仕様: ローカルサーバ＋既存エディタ（v2.0 全機能の Rust 移植）

日付: 2026-11-13 / 前提: M1（推論コア＋CLI）完了、`main` に取り込み済み。仕様の根拠は
`EasyIrodoriTTS_v2.0/` の Python ソース（`easy_web.py` ほか）と `docs/bench.md`。

## 1. 目的と成功基準
- v2.0 の**全機能**（5 タブ・約 28 の API）を、Python/ffmpeg 不要の単一 exe で提供する。
- **v2.0 のデータ（台本・話者・設定・辞書・生成済み音声）をそのまま読み書き**でき、Python 版と行き来しても壊さない。
- 既存のブラウザ画面（HTML/JS/CSS）を最小限の改修で流用する。
- 成功基準: ① v2.0 の実データを開いて編集・保存しても差分が出ない（往復テスト）② 音声生成・停止・書き出しが画面から一通り動く
  ③ 未取得モデルは確認のうえ自動ダウンロードされる ④ 全 HTTP 経路を擬似エンジンでテストできる。

## 2. ユーザーが決めたこと（2026-11-13）
1. モデルは**初回に自動ダウンロード**（確認つき）。2. データは **v2.0 と同じフォルダ形式を読み書き**。3. 範囲は**全機能を M2 に含める**。
4. 実装は M2a〜M2d の 4 段階（§11）。M3 は配布・性能改善・GPU 等（§11 末尾）。

## 3. 構成
```
crates/
  irodori-core   推論（M1。取消・進捗フックを追加: §6）
  irodori-store  v2.0 互換の保存層・後処理・書き出し（新規）
  irodori-app    axum サーバ、起動処理、モデル管理、生成ジョブ、画面資産の埋め込み（新規）
```
- 画面資産（`editor/`）は v2.0 のものを `crates/irodori-app/editor/` にコピーして埋め込む（rust-embed）。改修は §9 の最小限のみ。
- 起動: `irodori-app [--data-dir DIR] [--port N]`。既定ポート 7860（使用中なら空きを探す）。`auto_browser` 設定でブラウザを開く。
  データフォルダはロックファイル（`temp/app.lock`、PID）で多重起動を防ぐ。

## 4. データ互換（`irodori-store`）
- ファイル配置は v2.0 のまま: `projects/<32hex>.json`（schema=2）、`voices/index.json`、`voices/<id>/<uid>.wav`、`config/settings.json`、
  `config/reading_dictionary.json`、`outputs/internal/<run>/NN.wav`＋`generation.json`＋`card.json`、`outputs/exports/`、`temp/`。
- **型付き構造体＋`#[serde(flatten)] extra`**: 使うフィールドは型にし、未知のフィールドは `extra` に保持して書き戻す
  （Python 版が追加した項目を消さない）。ステータス文字列（「未生成」「待機中」「生成中」「完了」「失敗」「停止」「生成完了・書き出し失敗」
  「中断（再生成できます）」「再生成が必要」「音声が見つかりません」「話者を選び直してください」）は v2.0 の値をそのまま使う。
- 書き込みは原子的（一時ファイル＋fsync＋rename）。台本は `revision` による楽観ロック（不一致は 409 `Conflict`）。
- 元に戻す／やり直しは台本ごとにメモリ上で最大 100 段（入力内容のみ。`result/status/error` は対象外）。
- 生成結果の署名（`signature`）は v2.0 と同じ入力（text, voice_id, voice の sound_revision, speed, silence, generation, 読み置換後テキスト）の
  SHA-256（`json.dumps(sort_keys, ensure_ascii=False)` 相当の正規化）。**Python 版が付けた署名と一致すること**を互換テストで確認する。
- 読み込み時、`待機中/生成中` のカードは「中断（再生成できます）」へ戻す。

## 5. API（v2.0 と同名・同形）
| 段階 | エンドポイント |
|---|---|
| M2a | `GET /`, `/editor/{app.js,style.css}`, `/api/state`, `/api/meta`, `/audio/{key}`, `POST /api/edit`, `/api/action`(add/delete/duplicate/up/down/reorder/reset/split/undo/redo), `/api/project`, `/api/settings`, `/api/dictionary`, `/api/recover-previous` |
| M2b | `POST /api/upload`, `/api/voice`, `/api/voice/delete`, `/api/voice/rename`, `/api/design`, `/api/design-registration`, `/api/generate`, `/api/stop`, `/api/unload`、**新規** `/api/models/download`, `/api/models/cancel` |
| M2c | `POST /api/export`, `/api/audio-list`, `/api/bundle`, `GET /bundle/{key}`, `POST /api/save-audio`, `/api/cleanup-preview`, `/api/cleanup`, `/api/input-data/save`, `/api/input-data/load`, `/api/choose-folder`, `/api/open-output` |
- エラーは `{error: <日本語メッセージ>}`。`ValueError` 相当は 400、`Conflict` は 409、I/O は 400、想定外は 500（詳細はログへ）。
- `/api/state` は v2.0 と同じ形（project, projects, voices, settings, models, dictionary, jobs, backend, history, playback_stamp）。
  `models[]` に `installed`、`jobs[]` に `kind`（`generate`/`design`/`download`）と進捗（`step`/`steps`、`bytes_done`/`bytes_total`）を**追加**する（既存画面は無視できる）。
- 未実装・非対応の操作は 501 と「未対応です（理由）」を返す（例: `backend=cu128`、`lora_adapter`、`.webm`）。

## 6. 生成（`irodori-app`）
- `trait Synthesizer { fn synthesize(&self, job: SynthJob) -> Result<SynthRecord> }`。本物は `Runtime<Pipeline>`（M1）をモデル ID ごとに保持し、
  別モデルに切り替えるときは解放して読み込む（段階ロードで速い）。テストは擬似実装に差し替える。
- 生成設定 → `SynthRequest`: steps, seed, duration_scale, seconds, cfg_scale_*, cfg_guidance_mode, cfg_min_t/max_t, t_schedule_mode,
  sway_coeff, speaker_kv_scale, truncation_factor, rescale_k/sigma, max_text_len/max_caption_len, model_precision（bf16 可）。
  `lora_adapter` と `cu128` は非対応エラー。MeanFlow モデルは CFG 設定を無視（上流と同じ）。
- **`irodori-core` の追加**: `SynthRequest` に取消フラグ（`Arc<AtomicBool>`）と進捗（`Arc<AtomicUsize>` の現在ステップ）を足し、
  サンプラの各ステップで確認する（取消は `IrodoriError::Cancelled`）。停止の応答を「1 件の完了待ち」から「1 ステップ」に縮める。
- ジョブは直列（専用ワーカースレッド 1 本）。要求は提出時にスナップショット（カード・話者・生成設定・辞書・設定）。カードごとに状態を保存。
  25 秒超（推定）または 180 文字超のカードは `allow_long` の確認を要求（v2.0 と同じ）。
- 結果は `outputs/internal/<run_id>/NN.wav`（24bit）＋`generation.json`。候補（最大 8）はシードを変えて順に生成する（候補数倍の時間）。
- 参照音声は話者の `voices/<id>/<uid>.wav`。`design` の 4 モード（design / clone / both / auto）は v2.0 と同じ意味
  （auto = 説明も参照もなし）。試作文は 25 秒（推定）以内。

## 7. モデル管理と自動ダウンロード
- **許可リスト**: `Aratako/Irodori-TTS-v4.1-Small`、`phasefield-audio/Irodori-TTS-v4.1-Anime`、`Aratako/Irodori-TTS-v4.1-Small-MF`、
  共通 codec `Aratako/Semantic-DACVAE-Japanese-32dim`。話者 JSON の任意 ID は取得しない。
- 保存先: `models/huggingface/hub/models--<org>--<name>/{refs/main, snapshots/<sha>/…}`（HF キャッシュ互換。v2.0 のキャッシュも検出して流用）。
- 取得ファイルは必要最小限（`model.safetensors`、`tokenizer/tokenizer.json`、`tokenizer/tokenizer_config.json`、codec の `weights.pth`）。
  HF API（`/api/models/<id>/tree/main?recursive=1`）でサイズと LFS の SHA-256 を得て検証する。HTTPS のみ、リダイレクト先は `huggingface.co` /
  `hf.co` 系に限定。`.incomplete` に書いて Range で再開、完了後に rename。空き容量（必要量×1.2）を事前確認。
- **ユーザーの確認なしに取得しない**。未取得モデルでの生成要求は「ダウンロードしますか（約 3GB）」を返し、承認後に `download` ジョブを開始。
  進捗は `jobs[]`、キャンセル可能。ネットワーク不通は明確なエラー。

## 8. 後処理・書き出し・Windows 連携（ffmpeg / PowerShell 非依存）
- 話速変更: WSOLA による音程保持の時間伸縮（0.75〜1.5）。無音追加（0〜10 秒）。PCM 24bit。
- アップロード: symphonia で wav/mp3/flac/ogg/m4a/mp4 → 24kHz モノラル WAV、1〜120 秒、100MB 以内。`.webm`（Opus）は未対応メッセージ。
- 書き出し: パターン `$連番$ $話者名$ $テキスト$ $日付$`、`safe_name`（予約名・禁止文字・100 文字）、排他的作成での衝突回避、TXT は BOM 付き UTF-8、再試行。
- まとめ保存: ZIP（無圧縮）／連結 WAV（24bit、4GB 上限、レート不一致は M1 のリサンプラで揃える）。
- 整理: 対象は `outputs/internal` と `temp/uploads` の WAV のみ（話者参照は保護、シンボリックリンク・結合点は除外、一覧の stamp 一致を確認）。ごみ箱は `trash`。
- フォルダ選択・名前を付けて保存は `rfd`、フォルダを開くのは `open`。
- 台本分割・推定時間（`unicode-segmentation` の書記素）、読み辞書（最長一致）、入力済データ（8MB 以内）は v2.0 の実装を**読んで忠実に移植**する
  （`easy_splitter.py` 等。出力が Python 版と一致することをテストで固定）。
- `/api/meta`: 上流の絵文字パレットとクレジットを、生成スクリプトで JSON にして埋め込む。クレジット・ライセンス表記は必ず維持する。

## 9. 画面の改修（最小限）
- ダウンロード確認と進捗の表示、「未対応です」の表示、GPU 選択の無効化。それ以外は v2.0 のまま。

## 10. 安全対策（v2.0 を踏襲）
- `Host` が `127.0.0.1` / `localhost` 以外は 403。書き込み系は起動ごとのトークン（`x-easy-token`、`index.html` の `__TOKEN__` に埋め込み）必須。
- 音声はサーバが登録したキー（パスの SHA-256）経由でのみ配信。パスの直接指定は不可。アップロード 100MB、入力済データ 8MB。
- ファイル名は `safe_name` で無害化。ZIP/書き出しのパス結合はディレクトリトラバーサルを検証する。

## 11. 段階分け
| 段階 | 内容 | 完了の目安 |
|---|---|---|
| M2a | 保存層、状態 API、台本・カード編集（元に戻す/やり直し/並べ替え/分割）、辞書、設定、メタ、起動・安全対策、画面配信 | ブラウザで台本を編集・保存できる |
| M2b | 話者（登録・試作・アップロード）、生成ジョブ（進捗・停止）、モデル管理・自動 DL、core の取消・進捗 | 画面から音声を生成できる |
| M2c | 話速・無音、書き出し、まとめ保存、整理、入力済データ、フォルダ選択 | v2.0 の全機能がそろう |
| M2d | 互換テスト（Python 実データ）、ブラウザ通しテスト、仕上げ・文書 | v2.0 のデータで動作確認 |
- M3（別仕様）: 配布（zip/インストーラ・署名）、性能改善（`docs/bench.md` の残課題）、GPU、LoRA、ログ/診断。

## 12. テスト方針
- 単体: 保存層（往復・楽観ロック・undo/redo・署名一致）、分割/推定/ファイル名/辞書（Python 出力との一致、`tools/golden` と同様に生成して固定）、WSOLA、ZIP/連結、モデル管理（偽サーバでのダウンロード・再開・検証失敗）。
- HTTP: `tower::ServiceExt` で全エンドポイント（擬似 `Synthesizer`）。安全対策（Host/トークン/パス）の否定テスト。
- 互換: Python（v2.0 の `Store`）で実データを生成 → Rust で読み込み・編集・保存 → Python で読み直して同一。
- 実ブラウザ: Playwright で編集・生成・書き出しの通しを確認（実モデルは MF 1 本）。
- 実モデルの生成は M1 のゴールデンで担保済みなので、M2 では配線（設定のマッピング・取消・進捗）を確認する。

## 13. リスクと未確定事項
| リスク/未確定 | 影響 | 対処 |
|---|---|---|
| `signature` の正規化が Python の `json.dumps(sort_keys, ensure_ascii=False)` と一致しない | 既存の生成結果が「再生成が必要」になる | 浮動小数の表記（`1.0`）と並び順を互換テストで固定。M2a で最初に検証 |
| `easy_input_data.py` / `easy_dictionary.py` / `easy_text_tools.py` / `easy_launcher.py` の詳細は未読 | 仕様の取りこぼし | 各段階の着手時に該当ファイルを読み、差分を実装計画に反映 |
| `app.js` の依存（`/api/state` の全フィールド、ダウンロード用 UI） | 画面が壊れる | M2a で既存 `app.js` を読み、state の必須フィールドを一覧化して型にする |
| WSOLA の品質（atempo と同等か） | 話速変更の音質 | 主観確認と客観指標（長さ比・基本周波数）で検証。不足なら M3 で改良 |
| HF のダウンロード仕様変更（Xet/CDN） | 取得失敗 | リダイレクト追従と検証失敗時の明確なエラー。手動配置でも動く（§7） |
| 生成が遅い（標準 RTF 25〜30） | 体験 | MeanFlow を既定にし、進捗と停止を重視（§6）。性能改善は M3 |
