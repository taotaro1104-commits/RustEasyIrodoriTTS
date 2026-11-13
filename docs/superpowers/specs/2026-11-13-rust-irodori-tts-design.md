# RustEasyIrodoriTTS 設計仕様 (Draft 1)

日付: 2026-11-13 / 対象: EasyIrodoriTTS v2.0 の Rust 移植（Candle、CPU中心、省メモリ）

## 1. 目的と成功基準
- Python/PyTorch/uv/CUDA(約20GB)を不要にし、**単一exe＋モデルファイル**で動く。
- 常駐メモリ（ピークRSS）を Python版(fp32, 3GB超の重み常駐＋torch)より大幅に下げる。目標値は M1 の実測後に確定（暫定: 標準モデルで RSS 2.5GB 以下、MFモデル+int8で 1.5GB 以下）。
- 既存 editor（HTML/JS）と REST API を互換維持し、ユーザー操作は v2.0 と同じ。
- 音質は上流 Python 版と同等（同一seedで波形ではなくスペクトル・主観で許容、内部テンソルは許容誤差で一致）。

## 2. 確定した上流事実（固定コミット 89f9d8fb…、HF実測）
| 項目 | 内容 |
|---|---|
| DiTモデル | `TextToLatentRFDiT`（model.py 2233行）。RF（rectified flow）velocity。LowRankAdaLN、half-RoPE、SwiGLU、JointAttention、ReferenceLatentEncoder、DurationPredictor |
| 重み | `Aratako/Irodori-TTS-v4.1-Small/model.safetensors` = **3,064,295,596 B (F32)**。`tokenizer/tokenizer.json`(6.7MB)同梱 |
| テキスト系 | `text_encoder_type="pretrained"` 時は `PretrainedTextBackbone`（ModernBERT系、HF transformers）が DiT 重みに内包。`PretrainedConditionProjector` で DiT 次元へ射影。tokenizer: BOS を手動で先頭付与、右パディング、`add_special_tokens=False` |
| caption | text と別 tokenizer 設定可（config 由来）、caption CFG あり |
| 正規化 | `text_normalization.py`（74行）。置換表＋正規表現＋外側括弧除去＋NFKC。**そのままRustへ移植可能** |
| Codec | 外部パッケージ `dacvae`（上流リポジトリ外）。重み `Aratako/Semantic-DACVAE-Japanese-32dim/weights.pth` = **429,620,065 B**（torch pickle）。48kHz、潜在32次元。encode は deterministic（encoder→`_pad`→quantizer.in_proj の mean のみ）。decode は watermark 無効化（`alpha=0`、passthrough） |
| 参照音声前処理 | mono化 → codec SRへリサンプル → ラウドネス正規化 `-16 dB`（audiotools AudioSignal.normalize + ensure_max）→ encode |
| サンプラ | rf.py（584行）Euler + CFG（text/caption/speaker）、sway/linear スケジュール。meanflow.py（563行）MF 4step |
| 設定 | `config.py` の `ModelConfig` 既定値は旧世代用。**実寸法はチェックポイントのメタデータ/configから読む** |
| ライセンス | コードMIT、重みMIT＋倫理制限（UIのクレジット表記を維持） |

## 2b. 追加調査で確定（2026-11-13、safetensors ヘッダ＋dacvae参照実装）
**v4.1-Small の実寸法**（safetensors メタデータ `config_json` 実測、総 766,052,385 params）
- DiT: `model_dim=1280, num_layers=12, num_heads=20, mlp_ratio=2.875, latent_dim=32, latent_patch_size=1, adaln_rank=192, timestep_embed_dim=512`。1ブロック 29.87M ×12 = 358M。
- テキスト: ModernBERT-ja-310m（`sbintuitions/modernbert-ja-310m`）が重みに**内包**（314.6M）。25層, hidden 768, ffn 3072, 12 heads, full/sliding attention を 3層周期で交互、rope_theta 10000(sliding)/160000(full)。`residual_mlp` projector（768→512）。**caption は同一バックボーンを共有**（caption_encoder は 1.7M の小ヘッドのみ）。max_text_len=256, max_caption_len=512, BOS 手動付与。
- speaker: `speaker_dim=768, layers=8, heads=12, speaker_patch_size=4`（60.5M）、参照最大120秒。
- duration predictor: 21.8M（`token_sum_dual_adarn_zero_no_aux`）。
- 重み内訳: ModernBERT 315M / DiT 358M / speaker 61M / duration 22M / cond_module 7M。

**DACVAE**（参照: `facebookresearch/dacvae` と単一ファイル実装 `fakerybakery/DACVAE-selfcontained` 912行、いずれも Apache-2.0）
- Irodori は `decoder.alpha=0` にして透かしを無効化し、`watermark` を `encoder_block.forward_no_conv`（Snake→Conv(→1ch)→Tanh）に差し替える。**よって decode は `quantizer.out_proj → decoder.model(Conv7 + DecoderBlock×4) → Snake→Conv→Tanh` のみ。透かしの LSTM / MsgProcessor / decoder_block は不要で、ロード時にスキップできる（メモリ削減）。**
- encode は `encoder(_pad(wav)) → quantizer.in_proj → chunk(2)` の前半（mean）のみ（決定的、サンプリング無し）。
- 既定 `encoder_rates=[2,8,10,12]` → hop=1920 → 48kHz で潜在 **25 Hz**、`codebook_dim=32`（`weights.pth` の `metadata.kwargs` で要確認）。
- 必要演算: weight-norm 付き Conv1d / ConvTranspose1d、Snake、reflect pad、ELU(不要)。Candle で実装可能。`weights.pth` は torch pickle → `candle_core::pickle` で読める見込み（要検証）。

## 3. 未確定事項と解消方法（M1-0 タスク）
1. ~~実寸法~~ → 2b で確定。
2. `PretrainedTextBackbone`(model.py 673–790) の差分確認（ModernBERT の最終層出力の取り方、attention mask、Candle `modernbert` 実装との整合）。
3. `weights.pth` の `metadata.kwargs`（encoder_rates, decoder_dim, codebook_dim）とキー名の列挙、weight-norm の parametrization 形式（`weight_g/v` か `parametrizations`）。
4. duration predictor の出力単位と `inference_runtime.py`(1574行) の全体フロー（候補生成、trim_tail、max_seconds、参照音声の結合方法）。
5. Anime / MF モデルの差分（MF は delta 条件を追加、Anime は別リポジトリ）。
6. 決定性: Rust と Python の乱数(seed→初期ノイズ)は一致しない。比較は「同一ノイズをファイル経由で与える」ゴールデンテストで行う。
7. 参照音声のラウドネス正規化（audiotools `normalize(-16)`、pyloudnorm 系 BS.1770）の再現。

## 4. 構成（Cargo workspace）
- `irodori-core`: `normalize`, `tokenizer`(tokenizers crate), `encoder`(ModernBERT), `model`(DiT), `rf`/`meanflow`(サンプラ), `codec`(DACVAE), `loader`(mmap/型変換/量子化), `runtime`(段階ロード/アンロード)
- `irodori-store`: projects/voices/settings/辞書/splitter/export/bundle/cleanup（Python版 schema=2 互換）
- `irodori-app`: axum サーバ、既存 `editor/` を埋込み、トークン付き API（127.0.0.1のみ）
- `irodori-cli`: 検証・ベンチ用

## 5. 省メモリ設計
- 重みサイズ目安: F32 3.06GB → bf16/f16 約1.5GB → int8 約0.8GB（DiT）。codec 430MB(F32) → f16 約0.2GB。
- safetensors を mmap し、必要テンソルのみ変換。CPUの bf16 演算が遅い場合に備え、weight-only int8（candle quantized/自前dequant matmul）を M1 後半で評価。
- 生成時のピークを下げるため、`tokenize/encode条件 → DiT sampling → codec decode` を順に実行し、不要になった重みは解放（mmap の drop）。連続生成ではモデル常駐を既定にしつつ **アイドル自動アンロード**（既定5分、設定可）。
- CFG の3分岐はバッチ化せず順次評価の選択肢も用意（活性メモリ vs 速度）。`context_kv_cache` は再現。
- 生成は直列1ワーカー、出力は逐次ディスク書込み、連結WAVはストリーミング。

## 6. マイルストーン
- **M1 推論コア**: M1-0 精読 → ゴールデンテンソル作成（Python版環境を一度だけ用意）→ normalize/tokenizer → encoder → DiT → rf → codec(dec→enc) → CLI。完了条件: 同一ノイズ入力で DiT/codec 出力が許容誤差内、CLI で日本語文がWAV化できる。
- **M2 サーバ/UI**: axum+既存 editor、声登録、カード生成/停止、状態ポーリング、プロジェクト保存。
- **M3 周辺**: 辞書、分割、速度変更(ffmpeg非依存の time-stretch を検討、当面は任意でffmpeg検出)、書出し/ZIP/連結、入力データ、クリーンアップ、設定、配布。

## 7. リスク
| リスク | 影響 | 緩和 |
|---|---|---|
| DACVAE 移植（構造は参照実装で判明、リスク低下） | weight-norm/ConvTranspose の数値差 | 参照実装(Apache-2.0)を基にし、層ごとにテンソル比較。最悪 codec のみ ONNX/ort の逃げ道 |
| CPUで 0.8B×40step が遅い | 実用性 | MF(4step) を既定、int8、スレッド調整、ベンチで可否判断 |
| ModernBERT-ja の細部差 | 数値不一致 | candle 既存実装を基に層ごとにテンソル比較 |
| 重みの再配布/ライセンス | 配布 | 重みは同梱せず初回DL、倫理条項をUI表示 |
| 参照音声のラウドネス正規化(audiotools) | 音質差 | BS.1770 準拠実装を Rust で再現しテスト |

## 8. テスト方針
- 各モジュールをテスト先行。推論はゴールデンテンソル（Python版で保存）、store/splitter/export は Python 版入出力フィクスチャで互換テスト。カバレッジ 80%+（推論重み依存テストは `#[ignore]`＋CIでスキップ可）。

## 9. 次のステップ
この仕様をレビュー → 承認後に writing-plans で M1 の実装計画（M1-0 精読タスク含む）を作成。
