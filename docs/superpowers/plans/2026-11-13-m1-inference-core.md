# M1 推論コア Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Candle のみで Irodori-TTS v4.1 を実行し、`irodori-cli "日本語文" --voice ref.wav -o out.wav` で WAV を生成する（Python/torch 不要）。

**Architecture:** Cargo workspace の `irodori-core`（推論ライブラリ）と `irodori-cli`。重みは safetensors を mmap、設定は safetensors メタデータの `config_json` / `text_encoder_config_json` から読む。各モジュールは Python 版が出力したゴールデンテンソルと許容誤差で一致させてから次へ進む。

**Tech Stack:** Rust stable, candle-core/candle-nn/candle-transformers (CPU), safetensors, tokenizers, unicode-normalization, regex, hound, symphonia, rubato, clap, serde_json, anyhow, thiserror。Python（ゴールデン生成のみ、`uv`＋上流 commit 89f9d8fb）。

**Spec:** `docs/superpowers/specs/2026-11-13-rust-irodori-tts-design.md`

## Global Constraints
- 上流固定: `Aratako/Irodori-TTS` commit `89f9d8fbd4d51ea019867ee1197725ede1df13c5`。重み `Aratako/Irodori-TTS-v4.1-Small`、codec `Aratako/Semantic-DACVAE-Japanese-32dim`。
- v4.1-Small 寸法: `model_dim=1280, num_layers=12, num_heads=20, latent_dim=32, adaln_rank=192, speaker_dim=768, speaker_layers=8, speaker_patch_size=4, text_dim=512, max_text_len=256, max_caption_len=512`。ModernBERT: 25層, hidden 768, rope_theta 10000(sliding)/160000(full)。コードは寸法をハードコードせず config から読む。
- codec: 48kHz、hop 1920（25Hz）、潜在32次元。decode は透かし無効（`forward_no_conv` 経路のみ）。
- 計算は CPU、既定 dtype F32。bf16/int8 は Task 12 で追加。
- 1ファイル 400 行以内（最大800）、関数 50 行以内、入力は境界で検証、エラーを握りつぶさない（`thiserror`）。
- 重み依存テストは `#[ignore]` ではなく、環境変数 `IRODORI_GOLDEN_DIR` が無ければ skip メッセージを出して成功扱い（`golden::dir() -> Option<PathBuf>`）。純ロジックのテストは常時実行。
- コミットは Conventional Commits、各タスク末尾で1コミット。

## Review Focus
1. 空文字・全角空白のみ・括弧のみ（`「」`）の入力 → 正規化後が空でも panic せず、明確なエラーを返す（Task 2/11）。
2. 256 トークン超の長文 → 黙って切り捨てず `TextTooLong{tokens,max}` を返す（Python 版も token 数を検査する）（Task 2）。
3. 参照音声が無音・1秒未満・120秒超・ステレオ・44.1kHz → 無音は `SilentReference`、120秒超は先頭120秒に切る、mono化＋リサンプルが通る（Task 10）。
4. `weights.pth` が破損/キー不足、safetensors メタデータ欠落 → 欠けたキー名を含むエラー（Task 3/9）。
5. 生成長が極端に短い/長い（duration 予測が 0 や > max_seconds）→ 最小1フレーム、最大 `max_seconds=30` にクランプ（Task 11）。

---

### Task 1: ワークスペース雛形とゴールデン基盤
**Files:**
- Create: `Cargo.toml`（workspace）, `crates/irodori-core/{Cargo.toml,src/lib.rs,src/golden.rs,src/error.rs}`, `crates/irodori-cli/{Cargo.toml,src/main.rs}`, `tools/golden/dump_golden.py`, `tools/golden/README.md`, `.gitignore`（`target/`, `golden/`, `models/`）
- Test: `crates/irodori-core/src/golden.rs`（inline）

**Interfaces:**
- Produces: `golden::dir() -> Option<PathBuf>`、`golden::load(name:&str) -> Result<HashMap<String,Tensor>>`（`$IRODORI_GOLDEN_DIR/{name}.safetensors`）、`golden::assert_close(a:&Tensor,b:&Tensor,atol:f32,rtol:f32) -> Result<()>`、`error::IrodoriError`（thiserror enum、各タスクが variant を追加）。

- [ ] **Step 1:** テスト `assert_close_passes_within_tolerance` と `assert_close_fails_on_mismatch`（`[1.0,2.0]` vs `[1.0,2.0001]` は atol=1e-3 で OK、`[1.0,2.1]` は Err）を書く。
- [ ] **Step 2:** `cargo test -p irodori-core golden` が失敗（未定義）を確認。
- [ ] **Step 3:** `assert_close` を実装（`max|a-b| <= atol + rtol*|b|`、shape 不一致は Err、エラー文に最大誤差を含める）。workspace/依存を追加。
- [ ] **Step 4:** `cargo test --workspace` が PASS。
- [ ] **Step 5:** `tools/golden/README.md` に Python 環境手順を記載（`git clone` 上流を commit 固定、`uv sync --extra cpu`、`python tools/golden/dump_golden.py --out golden/`）。コミット `chore: scaffold workspace and golden test harness`。

### Task 2: テキスト正規化とトークナイザ
**Files:**
- Create: `crates/irodori-core/src/normalize.rs`, `src/text_tokenizer.rs`
- Test: 各ファイル inline

**Interfaces:**
- Produces: `normalize::normalize_text(&str) -> String`（上流 text_normalization.py と同一順序: 単純置換→正規表現置換→`strip_outer_brackets`→NFKC→`...`/`..`→`…`）、`TextTokenizer::from_file(path, bos_id:u32, pad_id:u32)`、`TextTokenizer::encode(&self,&str)->Result<Vec<u32>>`（BOS 先頭付与、special tokens なし）、`TextTokenizer::batch_encode(&self,&[&str],max_len:usize)->Result<(Tensor /*[B,L] u32*/, Tensor /*[B,L] u8 mask*/)>`（右パディング、本文は `max_len-1` で切る）、`IrodoriError::TextTooLong{tokens,max}`、`IrodoriError::EmptyText`。

- [ ] **Step 1:** normalize のテスト: `「こんにちは」→こんにちは`、`「a」「b」` は括弧を外さない、`！？→!?`、`ｱ→ア`（NFKC）、`...→…`、`…………→……`、`〜→ー`、全角空白と `\t` を除去、`(a)→a`。
- [ ] **Step 2:** 失敗を確認 → **Step 3:** 上流 `text_normalization.py` を忠実に移植（深さカウントのブラケット判定を含む）→ **Step 4:** PASS。
- [ ] **Step 5:** tokenizer テスト（`tokenizer.json` を `IRODORI_MODEL_DIR` から。無ければ skip）: `encode("こんにちは")[0]==1`、`batch_encode(["a","bbbb"],3)` の mask 行が `[1,1,0]`/`[1,1,1]`、`max_len==1` のとき全行 BOS のみ。ゴールデン `tokens.safetensors`（Python の `PretrainedTextTokenizer.batch_encode` 出力）と完全一致。空文字（正規化後）は `EmptyText`、トークン数 > 256 は `TextTooLong`（Review Focus 1,2）。
- [ ] **Step 6:** 実装して PASS、コミット `feat: text normalization and tokenizer`。

### Task 3: モデルローダ（mmap・メタデータ・dtype）
**Files:**
- Create: `crates/irodori-core/src/loader.rs`, `src/config.rs`
- Test: inline ＋ `tests/loader_header.rs`

**Interfaces:**
- Produces: `config::ModelConfig`（serde、`config_json` の全フィールド、未知キーは無視、欠落は Python 既定値）、`config::TextEncoderConfig`（`text_encoder_config_json`、ModernBERT 用 `candle_transformers::models::modernbert::Config` へ `into_candle()`）、`loader::Weights::open(path:&Path, device:&Device) -> Result<Weights>`（mmap）、`Weights::config(&self)->Result<(ModelConfig,TextEncoderConfig)>`、`Weights::var_builder(&self, dtype:DType)->VarBuilder<'_>`、`Weights::tensor_names()->Vec<String>`。

- [ ] **Step 1:** テスト: 小さな合成 safetensors（メタデータ付き）を `tempfile` に作り、`config()` が `model_dim` 等を返す／`config_json` 欠落で `MissingMetadata("config_json")`／存在しないキーを `var_builder` で取ると**キー名を含む**エラー（Review Focus 4）。実モデルがあれば `model_dim==1280 && num_layers==12`。
- [ ] **Step 2–4:** 失敗確認→実装（`candle_core::safetensors::MmapedSafetensors` 利用）→PASS。
- [ ] **Step 5:** コミット `feat: safetensors loader with metadata config`。

### Task 4: DiT 基本部品
**Files:**
- Create: `crates/irodori-core/src/nn/{mod.rs,rope.rs,norm.rs,adaln.rs,swiglu.rs,timestep.rs}`
- Test: inline（ゴールデン）

**Interfaces:**
- Produces: `rope::precompute_freqs_cis(dim,end,theta,&Device)->Result<Tensor>`、`rope::apply_rotary_emb(x:&Tensor,freqs:&Tensor)->Result<Tensor>`、`rope::apply_rotary_half`（JointAttention 用 half-RoPE）、`norm::RmsNorm::new(dim,eps,vb)`、`adaln::LowRankAdaLn::new(model_dim,rank,eps,vb)`（forward は `(x, cond)->(x_norm, gate)` ）、`swiglu::SwiGlu::new(dim,hidden,vb)`、`timestep::timestep_embedding(t:&Tensor,dim:usize)->Result<Tensor>`。形は `model.py` の同名関数（37–120行）と同一。
- [ ] **Step 1:** `dump_golden.py` に各関数の入出力ダンプ（`nn_basics.safetensors`）を追加。各部品のテスト `*_matches_golden`（atol 1e-5）。
- [ ] **Step 2–4:** 失敗→`model.py` 37–120, 428–437 を移植→PASS。
- [ ] **Step 5:** コミット `feat: DiT primitives (rope, rmsnorm, adaln, swiglu, timestep)`。

### Task 5: テキスト条件エンコーダ（ModernBERT＋projector）
**Files:** Create `crates/irodori-core/src/text_encoder.rs`

**Interfaces:**
- Consumes: Task 3 の `Weights`, `TextEncoderConfig`。
- Produces: `TextEncoder::load(vb, &ModelConfig, &TextEncoderConfig)`、`TextEncoder::encode_text(&self, ids:&Tensor, mask:&Tensor)->Result<Tensor /*[B,L,text_dim]*/>`、`encode_caption(...)`（同一バックボーン＋caption 用ヘッド）。`model.py` の `PretrainedTextBackbone`(673–790)・`PretrainedConditionProjector`(806–880)・`encoder_conditions` 該当部に準拠。

- [ ] **Step 1:** ゴールデン `text_enc.safetensors`（入力 ids/mask、バックボーン最終 hidden、projector 後）で `backbone_hidden_matches`（atol 1e-3）と `projected_matches`（atol 1e-3）。
- [ ] **Step 2:** 失敗→**Step 3:** candle-transformers の ModernBERT を利用。キー名が合わなければ `VarBuilder::rename_f` で `pretrained_text_backbone.` 接頭辞を剥がす。パディング位置が出力に影響しないことを確認するテスト `padding_does_not_change_valid_tokens` を追加。
- [ ] **Step 4:** PASS、コミット `feat: ModernBERT text/caption encoder`。

### Task 6: 参照潜在エンコーダ
**Files:** Create `crates/irodori-core/src/speaker_encoder.rs`（`ReferenceLatentEncoder`, model.py 883–925）

**Interfaces:** `ReferenceLatentEncoder::load(vb,&ModelConfig)`、`forward(latent:&Tensor /*[B,T,32]*/, mask:&Tensor)->Result<(Tensor,Tensor)>`（patch=4 でパッチ化、`patch_sequence_with_mask` 121–157 を `patch_sequence_with_mask(...)` として同ファイルに）。
- [ ] **Step 1:** ゴールデン一致テスト（atol 1e-4）＋ T が 4 の倍数でないとき末尾を捨てるテスト（`patchify_latent` 準拠）。
- [ ] **Step 2–4:** 失敗→実装→PASS。**Step 5:** コミット `feat: reference latent encoder`。

### Task 7: JointAttention と DiffusionBlock、DiT フォワード
**Files:** Create `crates/irodori-core/src/attention.rs`, `src/dit.rs`

**Interfaces:**
- Produces: `JointAttention`（`project_context_kv`/`build_context_masks`/`forward`、model.py 202–426）、`DiffusionBlock`（927–988）、`Dit::load(vb,&ModelConfig)`、`Dit::encode_conditions(&self, ConditionInputs)->Result<EncodedConditions>`、`Dit::forward_with_encoded_conditions(&self, x_t:&Tensor, t:&Tensor, &EncodedConditions, kv_cache:Option<&ContextKv>)->Result<Tensor /*velocity [B,T,32]*/>`、`Dit::build_context_kv_cache(&EncodedConditions)->Result<ContextKv>`。`model.py` 1455–2150 を対応。マスク付き SDPA は candle の softmax で実装（FlashAttention 不要）。
- [ ] **Step 1:** ゴールデン `dit_step.safetensors`（固定 x_t, t, 条件入力 → velocity）で `velocity_matches`（atol 1e-3, rtol 1e-3）。キャッシュあり/なしで結果が一致するテスト `kv_cache_equals_uncached`。
- [ ] **Step 2–4:** 失敗→実装（block 単位で中間出力もゴールデン化し、ずれた層を二分探索できるようにする）→PASS。
- [ ] **Step 5:** コミット `feat: joint-attention DiT forward`。

### Task 8: Duration predictor
**Files:** Create `crates/irodori-core/src/duration.rs`（model.py 990–1453, `predict_duration_log_frames` 2176）

**Interfaces:** `DurationPredictor::load(vb,&ModelConfig)`、`Dit::predict_duration_log_frames(&EncodedConditions, ...)->Result<Tensor>`。単位は `log(frames)`（要 `inference_runtime.py` で確認し doc コメント化）。
- [ ] **Step 1:** ゴールデン一致テスト（atol 1e-3）。**Step 2–4:** 失敗→実装→PASS。**Step 5:** コミット `feat: duration predictor`。

### Task 9: DACVAE codec（decode → encode）
**Files:** Create `crates/irodori-core/src/codec/{mod.rs,conv.rs,snake.rs,blocks.rs,pth.rs}`

**Interfaces:**
- Produces: `Codec::load(weights_pth:&Path, dtype)->Result<Codec>`（`candle_core::pickle` で読込、`metadata.kwargs` から `encoder_rates/decoder_dim/codebook_dim` を取得、**透かし系キー(`decoder.wm_model.*` の LSTM/msg/decoder_block)は読まない**）、`Codec::decode(&self, latent:&Tensor /*[B,T,32]*/)->Result<Tensor /*[B,1,samples]*/>`、`Codec::encode(&self, wav:&Tensor /*[B,1,S] @48k*/)->Result<Tensor /*[B,T,32]*/>`（`_pad` reflect→encoder→`in_proj` の mean のみ）、`Codec::sample_rate()->u32`、`Codec::hop()->usize`。参照: `dacvae_sc.py`（Snake=`x+1/(a+1e-9)*sin²(ax)`、NormConv1d/ConvTranspose1d の weight-norm 解決は `pth.rs` でロード時に `w = g*v/||v||` を事前計算）。
- [ ] **Step 1:** ゴールデン `codec.safetensors`（Python で encode→mean、decode 出力）と `decode_matches`（atol 1e-3）、`encode_matches`、`decode_length_is_frames_times_hop`（T=25 → 48000 サンプル）。キー欠落は `MissingKey(name)`（Review Focus 4）。
- [ ] **Step 2–4:** 失敗→実装→PASS。層ごとの中間出力ゴールデンで不一致箇所を特定する。
- [ ] **Step 5:** コミット `feat: DACVAE codec (decode/encode, watermark-free)`。

### Task 10: 参照音声の読込・前処理
**Files:** Create `crates/irodori-core/src/audio.rs`

**Interfaces:** `audio::load_reference(path:&Path, target_sr:u32, max_seconds:f32)->Result<Vec<f32> /*mono*/>`（symphonia でデコード、mono 平均、rubato でリサンプル、120秒で切る）、`audio::normalize_loudness(&mut [f32], sr:u32, target_db:f32)`（ITU-R BS.1770 の ungated 近似に audiotools 同様の `normalize` + ピーク1.0制限）、`audio::write_wav(path,&[f32],sr)`（hound, PCM24）。
- [ ] **Step 1:** テスト: 無音 → `SilentReference`、1.0s 未満でも通る/0.1s 未満は `ReferenceTooShort`、130s → 120s に切る、ステレオ44.1k → mono 48k で長さ比が `48000/44100`±1サンプル、`normalize_loudness` 後の測定ラウドネスが -16±0.5 dB、ピーク ≤ 1.0。ゴールデン `loudness.safetensors`（audiotools 出力）に対し波形 atol 1e-2。
- [ ] **Step 2–4:** 失敗→実装→PASS。**Step 5:** コミット `feat: reference audio loading and loudness normalization`。

### Task 11: Euler+CFG サンプラ、MeanFlow、パイプライン統合
**Files:** Create `crates/irodori-core/src/sampler.rs`, `src/meanflow.rs`, `src/pipeline.rs`

**Interfaces:**
- Consumes: Task 2–10。
- Produces: `sampler::sample_euler_cfg(&Dit,&EncodedConditions,&SamplerParams,noise:&Tensor)->Result<Tensor>`（rf.py 準拠: `t_schedule_mode` linear/sway、`cfg_scale_text/caption/speaker`＝3/4/5、`cfg_min_t/max_t`、`rescale_k/sigma`、`truncation_factor`）、`meanflow::sample_meanflow(...)`（4 step）、`Pipeline::load(model_dir:&Path)->Result<Pipeline>`、`Pipeline::synthesize(&self, SynthRequest)->Result<SynthOutput{samples:Vec<f32>,sample_rate:u32,seed:u64}>`。`SynthRequest{text,caption:Option<String>,ref_wav:Option<PathBuf>,num_steps,seed,max_seconds:f32(=30),noise:Option<Tensor>}`（`noise` はテスト用の注入口）。生成フレーム数は duration 予測（min 1、max `max_seconds*25`、Review Focus 5）、`trim_tail` を再現。
- [ ] **Step 1:** テスト: ゴールデン `sampler.safetensors`（固定ノイズ・条件から最終潜在）で `latent_matches`（atol 5e-3）、`sway_schedule_is_monotonic_and_ends_at_1`、`duration_clamped_to_max_seconds`、`empty_text_returns_error`。MF: `meanflow_uses_4_steps_by_default`。
- [ ] **Step 2–4:** 失敗→`rf.py`/`meanflow.rs`/`inference_runtime.py` を精読して実装→PASS。
- [ ] **Step 5:** E2E ゴールデン `e2e.safetensors`: 同一ノイズで Python 版 vs Rust 版の最終波形のメルスペクトル距離が閾値以下（閾値は初回計測して仕様書に記録）。コミット `feat: sampler, meanflow and synthesis pipeline`。

### Task 12: CLI と省メモリ（bf16・段階アンロード・計測）
**Files:** Create/Modify `crates/irodori-cli/src/main.rs`, `crates/irodori-core/src/runtime.rs`, `docs/bench.md`

**Interfaces:** CLI: `irodori-cli <text> [--voice ref.wav] [--caption s] [--model standard|anime|mf] [--steps N] [--seed N] [--dtype f32|bf16] [-o out.wav]`。`runtime::Runtime{ load(model,dtype), unload(), idle_unload_after: Duration }`、`Pipeline` は decode 前に DiT を、encode 後に codec encoder を `drop` できる構造（`Option<Dit>` 等）。
- [ ] **Step 1:** テスト: `bf16_output_close_to_f32`（latent の相対誤差 < 5e-2）、`unload_releases_weights`（unload 後 `is_loaded()==false`、再 synthesize で再ロードされる）、CLI 引数テスト（`clap` の `try_parse_from`）。
- [ ] **Step 2–4:** 失敗→実装→PASS。
- [ ] **Step 5:** 実機計測（`Get-Process` でピークWorkingSet、RTF、起動時間）を F32/bf16 × 標準/MF で `docs/bench.md` に記録し、仕様書 §1 の暫定メモリ目標を実測値で更新。int8 は bf16 が目標未達の場合のみ別タスク化。コミット `feat: CLI, bf16 and staged unload; bench results`。

---

## Self-Review メモ
- 仕様 §6 M1（M1-0 精読、ゴールデン、normalize→…→CLI）→ Task 1–12 に対応。§5 省メモリ → Task 3(mmap)/9(透かしスキップ)/12。§7 リスクのラウドネス → Task 10。
- M2/M3、CUDA、int8 は本計画の対象外（Task 12 の結果次第で追加計画）。
- 型整合: `EncodedConditions`/`ContextKv`/`SynthRequest` は Task 7/11 で定義、Task 8/12 が参照。
