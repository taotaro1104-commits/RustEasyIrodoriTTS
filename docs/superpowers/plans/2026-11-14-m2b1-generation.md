# M2b-1 生成（ジョブ・停止・進捗・モデルの検出） Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** v2.0 のデータにある話者を使って、画面から音声を生成・停止できる（`POST /api/generate` `/api/stop` `/api/unload`）。生成は 1 ステップ単位で取り消せ、進捗が `jobs[]` に出る。

**Architecture:** `irodori-core` に取消・進捗フック（`Control`）を足す。`irodori-app` に `Synthesizer` トレイト（本物は `Runtime<Pipeline>`、テストは擬似）と、専用ワーカースレッド 1 本の `JobManager`（v2.0 の `Service.submit/run/stop` の移植）を置く。モデルは**既にあるものを検出するだけ**（取得は M2b-3）。

**Tech Stack:** Rust, candle（M1）、axum、std スレッド、`hound`（24bit WAV）。

**Spec:** `docs/superpowers/specs/2026-11-13-m2-server-and-editor-design.md`（§5 M2b の生成系、§6、§7 の検出部分）。振る舞いの正解は `EasyIrodoriTTS_v2.0/EasyIrodoriTTS_v2.0/` の `easy_service.py`（`submit` `run` `stop`）、`easy_generation.py`、`easy_web.py`。

## M2b の分割（この計画は 1/3）
- **M2b-1（この計画）**: 生成ジョブ・停止・進捗・アンロード、モデルの検出（`installed`）。話者は v2.0 のデータにあるものを使う。
- M2b-2: 話者の登録・名前変更・削除、アップロード（symphonia で 24kHz モノラル化）、声の試作（`/api/design*`）。
- M2b-3: モデルの自動ダウンロード（確認つき・再開・SHA-256 検証・キャンセル）。

## Global Constraints
- ユーザー向け文言は v2.0 と同じ日本語（下の各タスクに明記）。想定外の失敗の詳細はログだけに出す。
- 生成結果は `outputs/internal/<run_id>/NN.wav`（**24bit モノラル**）＋`generation.json`＋`card.json`。`result.path` は **絶対パス**（v2.0 と同じ）。
- 署名・`result` の形・カードの状態文字列は M2a のものと v2.0 のものをそのまま使う（`irodori_store::signature` 他）。
- ネットワークに出ない（取得は M2b-3）。`cu128`（GPU）・`lora_adapter` は未対応エラー。計算精度は `fp32` / `bf16` のみ（`fp16` は未対応）。
- 話速（`speed` ≠ 1.0）は M2c で実装する。それまでは**生成時に拒否**して黙って無視しない。無音の追加（`silence`）は本計画で実装する。
- 1 ファイル 400 行以内（最大 800）。`cargo test -p irodori-store -p irodori-app` は常に緑。実モデルのテストは `--release` で、環境変数があるときだけ動く（無ければ skip）。
- `cargo test --workspace`（debug）を、`IRODORI_MODEL_DIR` などを設定したまま走らせない。

## Review Focus
1. 生成中の「停止」→ 1 ステップ以内に止まり、カードは「停止」（失敗ではない）、途中の WAV を結果として参照しない（Task 1, 5, 6）。
2. 生成エンジンが panic／モデルファイルが欠損・破損 → カードは「失敗」＋日本語メッセージで、ジョブが「生成中」のまま固まらず、次のカードへ進める／ワーカーが生き残る（Task 5, 6）。
3. 生成中にそのカードの文章を編集 → 保存済みの結果は署名が合わず「再生成が必要」になり、新しい入力を上書きしない（Task 6）。
4. 生成中の二重 submit・`unload`・`recover-previous` → 400 と v2.0 のメッセージで拒否され、状態は変わらない（Task 6, 7）。
5. 100,000 文字級の本文・25 秒超／180 文字超のカード → 確認（`needs_long_confirmation`）を要求し、確認後はモデルの上限で明確なエラー（Task 3, 6）。
6. 保存先に書けない（ディスクフル・権限）→ そのカードだけ「失敗」、他のカードは続行（Task 6）。

---

### Task 1: core の取消・進捗フック
**Files:** Create `crates/irodori-core/src/control.rs`; Modify `crates/irodori-core/src/{lib.rs,error.rs,sampler.rs,pipeline.rs}`; Test inline

**Interfaces:**
- Produces: `irodori_core::control::Control { pub cancel: Option<Arc<AtomicBool>>, pub progress: Option<Arc<AtomicUsize>> }`（`Clone + Default + Debug`）、`Control::new() -> Control`（両方を新しく作る）、`fn cancelled(&self) -> bool`、`fn check(&self) -> Result<()>`（取消なら `IrodoriError::Cancelled`）、`fn set_progress(&self, done: usize)`。`IrodoriError::Cancelled`（表示は `"cancelled"`）。`SynthRequest` に `pub control: Control`、`pub max_text_len: Option<usize>`、`pub max_caption_len: Option<usize>`（既定 `Control::default()` / `None`）。`sampler::sample_euler_cfg_ctl(dit, cond, p, noise, ctl: &Control)`、`sample_meanflow_ctl(dit, cond, num_steps, noise, ctl: &Control)`。既存の `sample_euler_cfg` / `sample_meanflow` は `Control::default()` で委譲し、**シグネチャを変えない**（ゴールデンのテストを触らない）。

- [ ] **Step 1: テスト（`control.rs` 内）** `check_is_ok_without_flags`、`check_returns_cancelled_once_the_flag_is_set`（別スレッドからの `store(true)` を含む）、`set_progress_is_visible_to_other_holders`、`default_control_is_a_no_op`。`IrodoriError::Cancelled.to_string() == "cancelled"`。失敗を確認。
- [ ] **Step 2: 実装** `Control` と `Cancelled`。サンプラは各ステップの**先頭で `check()`、終わりで `set_progress(i + 1)`**。`Pipeline::synthesize` は、参照 encode の後・条件エンコードの後・decode の前にも `check()`。長さ上限は `req.max_text_len.map_or(cfg.max_text_len, |l| l.min(cfg.max_text_len))`（caption も同様）に置き換える。
- [ ] **Step 3: 実モデルのテスト（`pipeline.rs` の既存方式、環境変数が無ければ skip）** MeanFlow モデルで ① 通常実行後 `progress == 4` ② 進捗 ≥ 2 になったら別スレッドで取消 → `Err(Cancelled)` かつ `progress < 4` ③ 事前に取消済み → ステップを 1 つも進めず `Cancelled`。`cargo test -p irodori-core --release` で確認（環境変数は既存の `tools/golden/README.md` の通り）。
- [ ] **Step 4:** 既存のゴールデンテストが変わらず通ること（`--release`）。コミット `feat(core): cancel and progress hooks`。

### Task 2: 擬似・本物に共通の `Synthesizer`
**Files:** Create `crates/irodori-app/src/synth.rs`, `crates/irodori-app/src/fake.rs`; Modify `crates/irodori-app/src/lib.rs`; Test inline

**Interfaces:**
- Produces: `synth::{SynthJob, SynthRecord, SynthError, Synthesizer}`。
  - `SynthJob { text: String, model: String, caption: String, references: Vec<PathBuf>, generation: irodori_store::generation::Generation, candidates: usize, cancel: Arc<AtomicBool>, progress: Arc<AtomicUsize> }`
  - `SynthRecord { id: String, model: String, text: String, caption: String, references: Vec<String>, generation: Value, seed: u64, paths: Vec<PathBuf>, seconds: f64, messages: Vec<String> }`（`paths` は絶対パスの 24bit WAV。`id` は `outputs/internal/<id>/` のフォルダ名）。
  - `enum SynthError { Cancelled, Invalid(String), Failed(String) }`（`Invalid` は利用者向け日本語、`Failed` は内部詳細）。
  - `trait Synthesizer: Send + Sync { fn synthesize(&self, job: &SynthJob) -> Result<SynthRecord, SynthError>; fn unload(&self); fn is_loaded(&self) -> bool; fn model_available(&self, model: &str) -> bool; }`
  - `fake::FakeSynth`（本体コードに置く。テストと `tests/` から使う）: `FakeSynth::new(root: &Path) -> FakeSynth`、`.with_steps(n)`・`.with_step_delay(Duration)`・`.fail_on(text: &str, SynthError)`・`.panic_on(text: &str)`・`.missing_models(&[&str])`。`synthesize` は `steps` 回（`step_delay` ずつ待つ）進捗を進め、各ステップで取消を確認し、`root/outputs/internal/<id>/NN.wav` に 0.2 秒の 24bit サイン波を書く。呼ばれた `text` の履歴を `calls()` で返す。

- [ ] **Step 1: テスト** `fake_synth_writes_24bit_wavs_and_reports_progress`、`fake_synth_stops_between_steps_when_cancelled`（取消後 `progress < steps`、`Err(Cancelled)`、WAV を書かない）、`fake_synth_candidates_produce_one_file_each`、`fake_synth_can_fail_or_panic_on_a_text`。失敗確認。
- [ ] **Step 2: 実装** 上のインターフェースどおり。`outputs/internal/<id>/` の作成・WAV 書き込みは `irodori_core::audio::write_wav`。コミット `feat(app): synthesizer trait and fake`。

### Task 3: 生成設定 → 合成要求の変換
**Files:** Create `crates/irodori-app/src/settings_map.rs`; Test inline

**Interfaces:**
- Consumes: `irodori_store::generation::Generation`（検証済みの Map）。Produces: `settings_map::{GenerationOptions, options_from(generation: &Generation) -> Result<GenerationOptions, SynthError>, model_steps(model: &str) -> Option<usize>}`。
  - `GenerationOptions { steps: Option<usize>, seed: Option<u64>, seconds: Option<f32>, duration_scale: f32, sampler: irodori_core::sampler::SamplerParams, max_text_len: Option<usize>, max_caption_len: Option<usize>, dtype: DType, codec_dtype: DType }`。
  - `model_steps` は許可リストの 3 モデル（標準 40・Anime 40・MF 4）。それ以外は `None`。

- [ ] **Step 1: テスト（Python の `DEFAULT_GENERATION` を基準）** `defaults_map_to_the_upstream_defaults`、`explicit_values_map_through`（steps/seed/cfg_scale_*/cfg_min_t/cfg_max_t/sway_coeff/speaker_kv_scale/truncation_factor/rescale_k+sigma/seconds/max_*_len）、`guidance_and_schedule_names`（`independent`/`joint`/`alternating`、`linear`/`sway`）、`bf16_is_accepted_and_fp16_is_not`（`fp16` → `Invalid("この版では fp32 か bf16 を選んでください。")`）、`lora_adapter_is_refused`（`Invalid("LoRAはこの版ではまだ使えません。")`）、`rescale_needs_both_values`（片方だけなら無効＝`None`）、`model_steps_for_the_allow_list_only`。失敗確認。
- [ ] **Step 2: 実装** `options_from`。`steps=None` は `model_steps` に任せる（MeanFlow は 4）。コミット `feat(app): map generation settings to sampler options`。

### Task 4: モデルの検出と本物の `Synthesizer`
**Files:** Create `crates/irodori-app/src/models.rs`, `crates/irodori-app/src/real_synth.rs`; Modify `crates/irodori-app/src/launch.rs`（`--models-dir`）; Test inline ＋ 実モデルのテスト

**Interfaces:**
- Produces: `models::{ALLOW_LIST, CODEC_ID, ModelFiles{ model: PathBuf, codec: PathBuf }, locate(hub: &Path, model_id: &str) -> Option<ModelFiles>, installed_ids(hub: &Path) -> Vec<String>}`。HF キャッシュ互換: `<hub>/models--<org>--<name>/snapshots/<sha>/{model.safetensors, tokenizer/tokenizer.json}`、codec は `models--Aratako--Semantic-DACVAE-Japanese-32dim/snapshots/<sha>/weights.pth`。`refs/main` があればその sha を優先、無ければ必要なファイルが全部そろった最初の snapshot。許可リスト外の ID は常に `None`。
- `real_synth::RealSynth::new(root: PathBuf, hub: PathBuf) -> RealSynth` が `Synthesizer` を実装。モデル ID と精度の組ごとに `Runtime<Pipeline>`（`MemoryMode::Staged`）を 1 つ保持し、別の組に切り替えるときは前のを解放してから読む。`Args.models_dir: Option<PathBuf>`（既定 `<data-dir>/models/huggingface/hub`）。

- [ ] **Step 1: テスト（`models.rs`、一時フォルダに偽の HF 構造を作る）** `locate_finds_a_complete_snapshot`、`refs_main_wins_over_other_snapshots`、`missing_tokenizer_or_codec_is_not_installed`、`ids_outside_the_allow_list_are_never_located`（`../`・大文字小文字違いを含む）、`installed_ids_lists_only_complete_models`。失敗確認。
- [ ] **Step 2: `models.rs` を実装。** 
- [ ] **Step 3: `RealSynth`**: `SynthJob` → `SynthRequest`（Task 3 の `options_from`、`ref_wav` は `references` の先頭、2 つ以上なら `Invalid("お手本の音声は1つだけ使えます。")`、候補は seed を `seed + i`（無指定なら毎回ランダム）で順に生成）、`Control { cancel, progress }`。`IrodoriError` → `SynthError`: `Cancelled`→`Cancelled`、`EmptyText`→`Invalid("モデルと読み上げ文章を確認してください。")`、`TextTooLong`→`Invalid("文章がモデルの処理上限を超えています。カードをさらに短くしてください。")`、`SilentReference`/`ReferenceTooShort`→`Invalid("参照音声を確認してください。")`、モデル未検出→`Invalid("モデルがまだ取得されていません。")`、その他→`Failed(詳細)`。WAV は `write_wav`（24bit）。`generation.json` を `atomic_json` で保存。
- [ ] **Step 4: 実モデルのテスト（環境変数があるときだけ）** `IRODORI_HUB_DIR`（HF 構造のキャッシュ。`F:\claude\upstream-irodori-cache\hf\hub`）で MF モデルを 1 文生成し、24bit WAV・`generation.json`・進捗 4 を確認、取消も 1 回確認。`--release`。
- [ ] **Step 5: コミット** `feat(app): model discovery and real synthesizer`。

### Task 5: 無音の追加（後処理）
**Files:** Create `crates/irodori-store/src/postprocess.rs`; Modify `crates/irodori-store/{Cargo.toml,src/lib.rs}`; Test inline

**Interfaces:** Produces: `postprocess::add_silence(source: &Path, seconds: f64) -> Result<PathBuf>`（`irodori_store::Result`）。`seconds` は 0〜10（外れる・非有限は `Invalid("終了無音は0～10秒で指定してください。")`）。0 なら `source` をそのまま返す。それ以外は 24bit で `<stem>_tail_<8hex>.wav` を同じフォルダに作り、元のサンプルの後ろに `round(seconds × sample_rate)` サンプルの無音を足す（チャンネル数は維持）。

- [ ] **Step 1:** テスト `zero_seconds_returns_the_same_path`、`appends_exact_silence_frames`（24kHz・1.5 秒 → 36000 サンプル増）、`keeps_original_samples_and_channels`（ステレオ入力も）、`rejects_out_of_range_and_non_finite`（-1, 10.1, NaN, inf）、`output_is_24_bit`。失敗確認。
- [ ] **Step 2:** `hound` で実装（`hound` を store の依存に追加）。コミット `feat(store): append trailing silence`。

### Task 6: `JobManager`（v2.0 の `submit / run / stop` の移植）
**Files:** Create `crates/irodori-app/src/jobs.rs`（大きくなるなら `jobs/{mod,submit,run}.rs`）; Modify `crates/irodori-app/src/lib.rs`; Test inline（`FakeSynth`）

**Interfaces:**
- Consumes: Task 2（`Synthesizer`, `FakeSynth`）、Task 3（`model_steps`）、Task 5（`add_silence`）、M2a（`Store`, `signature`, `reading_text`, `splitter::estimate`, `dictionary::load_lenient`）。
- Produces: `jobs::JobManager::new(store: Arc<Store>, synth: Arc<dyn Synthesizer>) -> JobManager`（ワーカースレッドを 1 本起動）。
  - `submit(&self, pid: &str, revision: i64, cid: Option<&str>, changed: bool, allow_long: bool) -> Result<SubmitOutcome>`、`enum SubmitOutcome { Started, NeedsLongConfirmation { long_cards: Vec<usize> } }`。
  - `stop(&self)`、`busy(&self) -> bool`（待機中/生成中のジョブがある）、`active_card_ids(&self, pid: &str) -> HashSet<String>`（これから処理される（未完了の）依頼のカード ID）、`jobs_json(&self) -> Vec<Value>`、`unload(&self) -> Result<()>`（ジョブ中は `Invalid("生成完了後に実行してください。")`、なければ `synth.unload()`）。
  - ジョブの JSON: `{id, project, status, done, total, error, stopping, kind: "generate", step, steps}`（`cancel`/`requests` は含めない）。`status` は `待機中`/`生成中`/`完了`/`完了（一部失敗）`/`停止`/`失敗`。`step` は実行中の候補の進捗（0〜`steps`）、`steps` は `model_steps`。

**仕様（`easy_service.submit/run/stop` の通り）:**
- submit: すでに待機中/生成中のジョブがあれば `Invalid("現在の生成が終わるか、停止してから実行してください。")`。revision 不一致は `Conflict("台本が更新されています。読み直してください。")`。対象は `cid` 指定ならそのカード、無ければ全カード（空白だけの文章は除く）。`changed` なら、失敗でなく結果が現行入力に一致し WAV もあるカードは除く。対象カードの話者が無ければ `Invalid("{n}番のカードの話者を選んでください。")`。話者のモデルが `model_available` でなければ `Invalid("モデルがまだ取得されていません。")`。対象が空なら `Invalid("生成対象がありません。文章と生成状態を確認してください。")`。`speed != 1.0` のカードがあれば `Invalid("話速の変更はこの版ではまだ使えません。話速を1.0倍にしてください。")`。推定が 25 秒超か 180 文字超のカードがあり `allow_long` でなければ `NeedsLongConfirmation`（カードの番号 1 始まり）。受け付けたら対象カードを `待機中`（error 空）にする。
- 依頼のスナップショット（提出時点のカード・話者・生成設定・設定・辞書・日付 `%Y%m%d`・読み置換後テキスト・署名）を保持する。`result.request` には v2.0 と同じ形（`card{id,text,voice_id,speed,silence,filename}`, `voice`, `signature`, `text`, `generation`, `ordinal`, `settings`, `date`, `dictionary`）を保存する。
- run（ワーカー）: 依頼ごとに、取消済みなら中断。カードを `生成中` にして `synth.synthesize`（参照は `<root>/<voice.audio>`、キャプションは `voice.caption`）。成功なら `add_silence(paths[0], card.silence)` を実 WAV に適用し、`result = {path: <絶対パス>, signature, seed, request, record: <run id>, export: {}}` を `card.json`（`atomic_json`、結果の WAV と同じフォルダ）にも保存し、カードを `完了` にする（自動書き出しは M2c。`export` は空）。`Cancelled` → カードは `停止`（結果は付けない）。`Invalid(msg)` → カード `失敗`＋ msg。`Failed`/panic（`catch_unwind`）→ カード `失敗`＋ `"処理に失敗しました。logs/editor.logを確認してください。"`（詳細はログ）。ジョブの `error` に最後の失敗を保持する。1 件ごとに `done += 1`。全体が終わったら `停止`（取消）／`完了（一部失敗）`／`完了`。**どの経路でも最後に、待機中/生成中のまま残ったカードを `停止` にする。**
- 保存は `Store::internal_update`（revision は上げない）。結果の反映は**カード ID で探す**ので、生成中に編集されても入力は上書きしない（署名が変わるので画面は「再生成が必要」になる）。

- [ ] **Step 1: テスト** `submit_runs_cards_in_order_and_saves_results`（結果の署名が `signature::signature` と一致、`path` が絶対パスで存在、`card.json` あり）、`unchanged_cards_are_skipped_when_changed_only`、`missing_voice_and_missing_model_are_reported_with_the_card_number`、`speed_other_than_one_is_refused_for_now`、`long_cards_need_confirmation_then_run`、`second_submit_while_running_is_refused`、`stop_cancels_within_a_step_and_marks_cards_stopped`（FakeSynth の遅延で実行中に stop）、`failure_marks_only_that_card_and_the_job_continues`、`panic_in_the_synthesizer_does_not_wedge_the_job`（次のジョブが動く）、`editing_a_card_while_it_generates_keeps_the_new_text`、`silence_is_appended_to_the_saved_wav`、`unload_is_refused_while_busy`、`jobs_json_has_progress_and_stopping`、`active_card_ids_lists_only_pending_requests`。失敗確認。
- [ ] **Step 2: 実装。** ワーカーは `mpsc` で依頼を受ける。ジョブ表は `Mutex<HashMap<String, Job>>`（ロックを保持したまま `synth` や `store` を呼ばない）。
- [ ] **Step 3: 全テスト緑、コミット** `feat(app): generation jobs`。

### Task 7: ルート（generate / stop / unload）と状態への接続
**Files:** Modify `crates/irodori-app/src/{lib.rs,routes.rs,launch.rs}`, `crates/irodori-app/tests/{common/mod.rs,http_security.rs,http_api.rs}`; Create `crates/irodori-app/tests/http_generate.rs`

**Interfaces:**
- `AppState` に `pub jobs: Arc<JobManager>`、`pub synth: Arc<dyn Synthesizer>` を追加。`POST /api/generate` は v2.0 の本文（`pid, revision, cid?, changed?, allow_long?`）を受け、`Started` なら `{message: "生成を開始しました。", ...state}`、`NeedsLongConfirmation` なら `{needs_long_confirmation: true, long_cards: [..]}`。`POST /api/stop` は `{message: "停止を受け付けました。処理中の1件が終わるまでお待ちください。"}`。`POST /api/unload` は `{message: "モデルを解放しました。"}`。
- `/api/state` の `jobs` は `JobManager::jobs_json()`、`models[].installed` は `synth.model_available(id)` で上書きする。`/api/action`（undo/redo）の `active` は `jobs.active_card_ids(pid)`、`/api/recover-previous` の `busy` は `jobs.busy()`。`UNIMPLEMENTED_POST` から `generate` `stop` `unload` を外す。

- [ ] **Step 1: テスト（擬似 `FakeSynth`、`tower::ServiceExt`）** `generate_then_poll_state_until_the_card_is_done`（`jobs[]` の進捗、完了後 `audio` キーで `/audio/{key}` が 200 の 24bit WAV）、`generate_requires_the_token`、`long_text_returns_the_confirmation_payload`、`stop_over_http_marks_the_card_stopped`、`generate_with_missing_model_is_400_with_the_japanese_message`、`unload_while_generating_is_400`、`recover_previous_is_refused_while_generating`、`undo_does_not_stop_cards_that_are_still_queued`（`active_card_ids` の効果）、`installed_flag_follows_the_synthesizer`。既存の `unimplemented_routes_return_501_json` から 3 本を外す。`common/mod.rs` と `http_security.rs` の `AppState` 構築に新しい項目を足す。失敗確認。
- [ ] **Step 2: 実装。** `launch::run` は `RealSynth` と `JobManager` を作って `AppState` に入れる。`run` の起動順を「`bind_listener` → `fresh_start`」に直し、致命的な起動エラーを `logs/editor.log` にも書く（M2a の持ち越し）。
- [ ] **Step 3: 全テスト緑、コミット** `feat(app): generate, stop and unload routes`。

### Task 8: 実ブラウザでの通し確認と文書
**Files:** Create `docs/m2b1-verification.md`; Modify `README.md`

- [ ] **Step 1:** v2.0 の実データの話者（`tools/compat/gen_fixtures.py --make-data-dir` で作ったフォルダ、または既存の v2.0 データ）と、HF 構造のキャッシュ（`F:\claude\upstream-irodori-cache\hf\hub`、`--models-dir`）で `irodori-app` を起動し、Playwright で確認して結果とスクリーンショットを記録: 「すべて生成」→ 進捗表示 → カードが完了・再生できる／生成中の「停止」／「未生成・変更分を生成」／25 秒超のカードの確認ダイアログ／モデル未取得時のエラー表示／コンソールに未処理エラーが無い（MF モデル 1 本で十分。標準モデルは 1 文だけ）。
- [ ] **Step 2:** 問題があれば該当タスクのテストを先に足して直す（RED→GREEN）。`README.md` に `--models-dir` と、M2b-1 でできること（話者の登録は M2b-2）を追記。コミット `docs: m2b-1 browser verification`。

---

## Self-Review メモ
- 仕様 §5（M2b の生成系 `generate`/`stop`/`unload`）→ Task 6, 7。§6（`Synthesizer`・要求への変換・取消/進捗・直列ジョブ・`allow_long`・結果の保存）→ Task 1〜6。§7 のうち検出（`installed`）→ Task 4（取得は M2b-3）。
- 型整合: `SynthJob/SynthRecord/SynthError/Synthesizer` は Task 2 で定義し、Task 4, 6, 7 が使う。`Control` は Task 1 で定義し Task 2（擬似）・4 が使う。`add_silence` は Task 5 → 6。`JobManager` は Task 6 → 7。`options_from` は Task 3 → 4。
- 仕様との差（要レビュー）: ① 話速（M2c）は生成時に拒否 ② 自動書き出し（`export`）は M2c ③ 参照音声は 1 つだけ（コアが 1 つしか使えない）。いずれも本計画で明記。
- M2a の持ち越しの軽微な指摘のうち「起動順（bind→fresh_start）と致命エラーのログ」は Task 7 に入れた。残り（io エラーのログ、2MB 超の JSON 413、panic 捕捉層、canonicalize 代替）は M2b-2 の冒頭に回す。
