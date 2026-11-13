# M2b-2 実ブラウザ・実モデルでの通し確認（取り込み・話者の登録・試作）

## 条件
- 日付: 2026-11-14。ビルド: `cargo build --release -p irodori-app`、CPU のみ（Windows 11）。ブラウザは Playwright（Chromium）。
- モデル: MeanFlow（`Aratako/Irodori-TTS-v4.1-Small-MF`、4 ステップ）。HF 構造のキャッシュ（`F:\claude\upstream-irodori-cache\hf\hub`）を `--models-dir` で指定。
- データ: **話者 0 人の空のフォルダ**から開始。音声は、M2b-1 の確認で作った CLI の合成音声（3.3 秒）を `soundfile` で wav / mp3 / ogg / flac に書き出したもの、0.5 秒の wav、テキスト、`.webm` を装った 68 バイト、壊れた wav。

## 結果

| # | 確認項目 | 結果 | 補足 |
|---|---|---|---|
| 1 | wav を取り込む → 長さ表示 → 話者名を入れて登録 | OK | 「voice.wav ／ 3.3秒」。登録後、`voices/<id>/<uid>.wav`（24kHz・モノラル・**24bit**）と `voices/index.json`。`/api/state` の `voices[].asset` で試聴できる。`01-uploaded.png` |
| 2 | mp3 / ogg の取り込み | OK | どちらも「3.3秒」。flac はテスト（`tests/decode.rs`）で確認 |
| 3 | 0.5 秒の音声 | OK（拒否） | 「音声は1～120秒にしてください。」登録ボタンは無効のまま。`02-short-error.png` |
| 4 | `.txt` / `.webm` / 壊れた wav | OK（拒否） | 「音声ファイルを選んでください。」／「未対応です（.webm は読み込めません。wav / mp3 / flac / ogg / m4a / mp4 を使ってください）。」（501）／「音声を読み込めません。形式を確認してください。」 |
| 5 | 失敗した取り込みの後片付け | OK | `temp/uploads` に残るのは成功した 3 件の `.normalized.wav` だけ。元ファイルと失敗分は残らない |
| 6 | 名前の変更 | OK | `revision` 1→2、**`sound_revision` は 1 のまま**（音は同じ）。話者の並びも保たれる |
| 7 | 実モデルでの声の試作（`design-registration`、説明つき、候補 2） | OK | `jobs[]` に `kind: design`、`step` 0→4、`stage` が `conditions`→`sampling`→`decode`。約 2 分 35 秒（定型文 74 字の合成音声が約 11 秒あり、1 候補あたり 77 秒。デコードが 53 秒）。候補 2 つが試聴でき、2 つ目を選んで登録 → `creation` にジョブの記録（文章・説明・シード・候補のパス）が入る。`03-design-candidates.png` |
| 8 | 試作中の停止 | OK | 12 秒後に停止 → 2 秒以内に「停止」（`stage: sampling`）、候補 0 件 |
| 9 | 登録した話者（取り込み・試作の 2 種）でカードを生成 | OK | どちらも「完了」（22 秒・43 秒） |
| 10 | 話者の削除 | OK | 一覧から消え、使っていたカードは「話者を選び直してください」。WAV ファイルは残る（生成待ちの要求が参照するため） |
| 11 | コンソールの未処理エラー | OK | JavaScript の例外は 0 件。出るのは、意図して起こした 400 ×3・501 ×1 のブラウザ自身の記録だけ |
| 12 | **v2.0 の Python で読めるか**（`tools/compat/check_voices.py`） | OK | Rust が書いた `voices/index.json`・台本を v2.0 の `Store` が読み込み、話者の WAV は `24000Hz ch=1 PCM_24`、カードの結果は v2.0 の `Service.result_matches` で**署名が一致**（`signature_matches=True`） |

## 気づいたこと（未解決ではないもの）
- 取り込んだ音声の `temp/uploads/*.normalized.wav` は、登録に使わなくても残る（v2.0 と同じ）。整理は M2c の「整理」で対象になる。
- `design` の説明（caption）は MF モデルでも通る（エラーなし）。声に効くかどうかの聴き比べまではしていない。
- 試作のシードは 64bit の大きな整数。話者の `creation` には正確に保存されるが、ブラウザの JSON（JS の数値）を経由して読むと下位桁が丸まる。画面からこの値を送り返す経路は無い。

## 自動テストの状況（このコミット時点）
- `irodori-core`: 178（＋実モデル 1 ignored）＋デコード 9。`irodori-store`: 132（話者 19 件は v2.0 の実コードが作った fixture `voice_ops_cases` と一致）。`irodori-app`: 172（単体 96・API 17・生成 11・堅牢化 4・セキュリティ 10・話者 31・起動 3）。

## 再現手順
```powershell
cargo build --release -p irodori-app
.\target\release\irodori-app.exe --data-dir <空のフォルダ> --models-dir F:\claude\upstream-irodori-cache\hf\hub --port 17890 --no-browser
# ブラウザで http://127.0.0.1:17890/ →「声を登録する」
# v2.0 での読み直し（Rust 版を止めてから）:
#   cd F:\claude\upstream-irodori
#   uv run --python 3.11 --with regex python ..\RustEasyIrodoriTTS\tools\compat\check_voices.py <データフォルダ>
```
