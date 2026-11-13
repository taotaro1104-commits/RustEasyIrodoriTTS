# M2a 保存層・状態 API・エディタ配信 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** v2.0 と同じ保存形式を読み書きし、既存のブラウザ画面で台本（カード）を編集・保存できるローカルサーバ `irodori-app` を作る（音声生成はまだ無い）。

**Architecture:** 新規クレート `irodori-store`（v2.0 互換の保存層・台本操作・辞書・分割・署名・状態ビルダ。HTTP 非依存）と `irodori-app`（axum。安全対策・ルート・画面資産の埋め込み・起動）。保存データは型付き構造体＋`#[serde(flatten)] extra` で未知フィールドを保持する。Python 版との互換は、Python（v2.0 の実コード）が作る fixture で固定する。

**Tech Stack:** Rust, axum 0.8, tokio, tower, serde/serde_json(preserve_order), sha2, uuid, regex, unicode-segmentation, rust-embed, pulldown-cmark, thiserror, tempfile。fixture 生成のみ Python（`F:\claude\upstream-irodori` の venv。v2.0 のコードを import）。

**Spec:** `docs/superpowers/specs/2026-11-13-m2-server-and-editor-design.md`（§3〜§5, §8, §10, §13）。v2.0 のソース: `EasyIrodoriTTS_v2.0/EasyIrodoriTTS_v2.0/`（`easy_storage.py` `easy_service.py` `easy_web.py` `easy_splitter.py` `easy_dictionary.py` `easy_launcher.py` `editor/`）。

## Global Constraints
- 保存形式は v2.0 と同一: `projects/<32桁hex>.json`（`schema=2`）、`voices/index.json`、`config/settings.json`、`config/reading_dictionary.json`、`temp/`、`outputs/`。ファイルは UTF-8（読み込みは BOM 付き `utf-8-sig` も可）。書き込みは「一時ファイル→fsync→rename」。
- ステータス文字列は v2.0 の値のみ: 未生成 / 待機中 / 生成中 / 完了 / 失敗 / 停止 / 生成完了・書き出し失敗 / 中断（再生成できます） / 再生成が必要 / 音声が見つかりません / 話者を選び直してください。エラーメッセージの日本語も v2.0 と同一。
- 台本: カード上限 1000 枚、1 カード 10 万文字以内、`speed` 0.75〜1.5、`silence` 0〜10、`name` 100 文字。undo/redo は台本ごとに最大 100 段。
- HTTP: `Host` が `127.0.0.1`/`localhost`/`testserver` 以外は 403。`GET`/`HEAD` 以外は `x-easy-token` 必須（不一致は 403）。`/editor/` は `app.js` と `style.css` のみ。音声は登録キー経由のみ。
- ID（`pid`）は 32 桁の小文字 hex のみ。パス結合に使う値は必ず検証する。
- エラー応答は `{"error": "<日本語>"}`: 不正入力 400、`Conflict` 409、I/O 400、想定外 500（詳細はログのみ）。
- 1 ファイル 400 行以内（最大 800）、関数 50 行以内。エラーを握りつぶさない。コミットは Conventional Commits、各タスク末尾で 1 コミット。
- 画面（`editor/` の HTML/JS/CSS）は v2.0 のものをそのままコピーして使う（M2a では改修しない）。

## Review Focus
1. `pid` / ファイル名に `../` や 31・33 桁・大文字 hex を渡す → 保存先の外を読み書きせず 400（Task 4, 10, 11）。
2. 古い `revision` での編集・操作 → 409 で、保存済みデータは一切変わらない（Task 4, 9, 11）。
3. 壊れた JSON（台本・話者一覧・設定・辞書）→ 日本語エラーを返し、**元のファイルを上書きしない**（Task 4, 6）。
4. 10 万文字超のテキスト／1000 枚超のカード → 拒否され、状態が変わらない（Task 9）。
5. Python 版が書いた未知フィールド・BOM 付きファイル → 読み書きの往復で保持される（Task 3, 4, 13）。
6. `Host` 偽装、トークンなしの書き込み、未登録の音声キー → 拒否（Task 10, 11）。

---

### Task 1: クレートの雛形と Python fixture 生成基盤
**Files:**
- Create: `crates/irodori-store/{Cargo.toml,src/lib.rs,src/error.rs}`, `crates/irodori-app/{Cargo.toml,src/lib.rs,src/main.rs}`, `crates/irodori-store/src/testutil.rs`（`#[cfg(test)]`）, `tools/compat/gen_fixtures.py`, `tools/compat/README.md`
- Modify: `Cargo.toml`（workspace members と共通依存）

**Interfaces:**
- Produces: `irodori_store::{StoreError, Result}`。`enum StoreError { Invalid(String), Conflict(String), Io(String), Internal(String) }`、`fn message(&self) -> &str`、`fn status(&self) -> u16`（400/409/400/500）。`testutil::temp_root() -> tempfile::TempDir`（`projects/ voices/ config/ temp/ outputs/` を作成）。`gen_fixtures.py --out <dir>` は fixture を `crates/irodori-store/tests/fixtures/` に書く（以降のタスクが関数を追加）。

- [ ] **Step 1:** テスト `status_mapping_follows_the_spec`（Invalid→400, Conflict→409, Io→400, Internal→500）と `message_returns_the_original_text` を書く。
- [ ] **Step 2:** `cargo test -p irodori-store` が失敗（未定義）を確認 → **Step 3:** 実装 → **Step 4:** PASS。
- [ ] **Step 5:** `gen_fixtures.py` の骨組み（`sys.path` に v2.0 フォルダを追加し `easy_storage` 等を import、`--out`）を作り、README に実行手順（`uv run python tools/compat/gen_fixtures.py` を `F:\claude\upstream-irodori` で、`PYTHONPATH` 指定）を書く。コミット `chore: scaffold store/app crates and compat fixture generator`。

### Task 2: Python 互換の JSON 書式（署名の土台）
**Files:** Create `crates/irodori-store/src/py_json.rs`; Test inline ＋ `tests/fixtures/py_json_cases.json`; Modify `tools/compat/gen_fixtures.py`

**Interfaces:**
- Produces: `py_json::float_repr(x: f64) -> String`（Python の `repr(float)`）、`py_json::dumps_sorted(&Value) -> String`（`json.dumps(v, ensure_ascii=False, sort_keys=True)` と同一。区切りは `", "` と `": "`）、`py_json::dumps_pretty(&Value) -> String`（`json.dumps(v, ensure_ascii=False, indent=2)` と同一、`allow_nan=False` 相当で NaN/Inf は Err）。
- 仕様: 浮動小数は `1.0` `0.75` `1e-05` `1e+16` `-0.0` の書式。整数と浮動小数を区別（`1` と `1.0`）。文字列は `"` `\` と制御文字（`\n` `\t` `\u001f` 等）のみエスケープし、日本語・絵文字・U+007F はそのまま。

- [ ] **Step 1:** Python で `py_json_cases.json`（入力 JSON 文字列と、`json.dumps` の 2 種類の出力）を生成する。ケース: `1`, `1.0`, `0.75`, `1e-05`, `1e+16`, `123456789.123`, `-0.0`, `"日本語"`, `"\u0007"`, `"\u007f"`, `"𝄞"`, 空配列/空オブジェクト、ネスト、キー順の入れ替え、`null/true/false`。
- [ ] **Step 2:** Rust テスト `matches_python_for_every_fixture_case`（両関数）、`float_repr_switches_to_exponent_like_python`（`1e-05`, `1e+16`, `0.0001`, `123456789012345.0`, `1e+22`）、`non_finite_numbers_are_rejected_by_dumps_pretty` を書き、失敗を確認。
- [ ] **Step 3:** 実装（`serde_json::Value` を再帰的に走査。浮動小数は ryu の最短表現から Python 規則（指数が -4 未満または 16 以上で指数表記、指数は 2 桁以上と符号付き）に整形）。**Step 4:** PASS、コミット `feat: python-compatible json formatting`。

### Task 3: 型付きモデル（extra 保持）と生成設定の検証
**Files:** Create `crates/irodori-store/src/{model.rs,generation.rs}`; Test inline ＋ `tests/fixtures/{project_sample.json,voices_sample.json,settings_sample.json,generation_cases.json}`

**Interfaces:**
- Produces: `model::{Card, Voice, Project, Settings}`（serde。`Card{id,text,voice_id,speed:f64,silence:f64,filename,result:Option<Value>,error,status, #[serde(flatten)] extra: Map}`、`Voice{id,name,model,caption,audio,revision:i64,sound_revision:Option<i64>,creation:Option<Value>, extra}`、`Project{schema:i64,id,revision:i64,name,selected,dictionary_enabled:bool,generation:Generation,cards:Vec<Card>, extra}`、`Settings{schema,backend,theme,auto_browser,restore_input,inherit,auto_export,custom_output,output_dir,custom_pattern,pattern,txt,last_project, extra}`）、`model::{new_card(source:Option<&Card>)->Card, new_project(name:&str)->Project, uid()->String}`（`uid` = uuid4 の 32 桁 hex）。`generation::{Generation = serde_json::Map<String,Value>, default_generation()->Generation, validate_generation(&Value)->Result<Generation>}`。
- 仕様（`easy_storage.validate_generation`・`easy_models.DEFAULT_GENERATION` を忠実に）: 既定キー集合のみ残す。範囲外・非有限値は `"{label}は{min}～{max}で指定してください。"`（min/max の表記は Python の `str(float)`）。`cfg_min_t > cfg_max_t` は専用メッセージ。`steps`/`seed` は整数または null、`max_*_len` は整数、`rescale_k/sigma` は両方指定か両方 null、`model_precision`/`codec_precision` は fp32/fp16/bf16。型は Python と同じ（`duration_scale` は常に浮動小数）。

- [ ] **Step 1:** Python で `generation_cases.json`（入力・期待出力 or 期待エラー文）と実データ形式の `project_sample.json`（未知フィールド `"future_field": {"a":1}` をカードとプロジェクトに追加）、`voices_sample.json`、`settings_sample.json` を生成。
- [ ] **Step 2:** テスト `validate_generation_matches_python_cases`、`unknown_fields_survive_a_serde_round_trip`（project/voice/settings 各々。再シリアライズで `future_field` が残る）、`new_project_has_schema_2_and_one_empty_card`、`uid_is_32_lowercase_hex`、`integer_speed_from_the_ui_is_stored_as_float`（`1` → `1.0`）を書き失敗確認。
- [ ] **Step 3–4:** 実装して PASS、コミット `feat: typed v2.0 models and generation validation`。

### Task 4: ストレージ（原子的書き込み・読み込み・楽観ロック）
**Files:** Create `crates/irodori-store/src/storage.rs`; Test inline

**Interfaces:**
- Consumes: Task 2/3。Produces: `storage::{atomic_json(path:&Path, v:&Value)->Result<()>, read_json(path:&Path)->Result<Option<Value>>}`、`Store`:
  `Store::open(root:&Path)->Result<Store>`、`settings()->Settings`、`voices()->Vec<Voice>`、`root()->&Path`、`load(pid:Option<&str>)->Result<Project>`（`None` は `last_project`、無ければ新規作成して保存）、`persist(&Project)->Result<()>`（`last_project` も更新）、`update(pid:&str, revision:i64, remember:bool, f:&mut dyn FnMut(&mut Project)->Result<()>)->Result<Project>`、`internal_update(pid:&str, f:&mut dyn FnMut(&mut Project))->Result<Project>`、`list_projects()->Vec<ProjectSummary{id,name}>`、`save_settings(&Value)->Result<()>`。`project_path(pid)` は 32 桁小文字 hex 以外を `Invalid("台本IDが不正です。")`。
- 仕様: `update` は `revision` 不一致で `Conflict("別の操作で台本が更新されています。保存済みの内容を読み直してください。")`。変更なしなら revision を上げず保存もしない。変更があれば revision+1 を保存。`load` は `待機中/生成中` を「中断（再生成できます）」にして返す。`schema != 2`・JSON 破損・カード重複は `Invalid("台本を読み込めません。形式または保存場所を確認してください。")`（`read_json` 失敗は `保存データを読み込めません。元のファイルは変更していません: {path}`）。`save_settings` の検証は `easy_storage.save_settings` と同じ（bool 項目、backend は cpu/cu128、theme は light/dark/system、`custom_output` なら `output_dir` は絶対パス、`custom_pattern` ならパターン非空）。

- [ ] **Step 1:** テスト: `atomic_json_leaves_no_temp_file_and_replaces_atomically`、`read_json_accepts_a_utf8_bom`、`corrupt_project_is_reported_and_the_file_is_untouched`、`corrupt_settings_and_voices_do_not_get_overwritten`、`pid_with_traversal_or_wrong_length_is_rejected`（`../x`、31 桁、33 桁、大文字）、`update_bumps_revision_and_persists`、`stale_revision_is_a_conflict_and_changes_nothing`、`no_op_update_keeps_the_revision`、`interrupted_cards_load_as_resumable`、`load_none_creates_and_remembers_a_project`、`save_settings_validates_like_python`（各エラー文）を書き、失敗確認。
- [ ] **Step 2–4:** 実装して PASS（`Store` は `Mutex` で内部状態を保護。クロージャ内から `Store` を呼ばない設計）。コミット `feat: v2.0-compatible store with optimistic locking`。

### Task 5: 元に戻す／やり直し
**Files:** Modify `storage.rs`（または `history.rs` に分割）; Test inline

**Interfaces:**
- Produces: `Store::travel(pid:&str, revision:i64, redo:bool)->Result<Project>`、`Store::history(pid:&str)->(bool /*undo*/, bool /*redo*/)`。`update(..., remember=true)` が入力内容（`generation`, `dictionary_enabled`, 各カードから `result/status/error` を除いたもの）の変化を履歴へ積む（100 段で切り詰め、redo を消去）。`travel` は履歴の入力内容を戻しつつ、カードごとの `result/status/error` は直近の成果キャッシュから復元（無ければ `result=None, status=未生成`）。選択カードが消えたら先頭へ。
- メッセージ: 空のとき「元に戻せる操作がありません。」／「やり直せる操作がありません。」、revision 不一致は `Conflict("入力内容が更新されています。画面を読み直してください。")`。

- [ ] **Step 1:** テスト: `undo_then_redo_restores_text`、`history_is_capped_at_100`、`new_edit_clears_redo`、`selection_and_results_are_not_edit_operations`（選択変更は履歴に積まない）、`undo_restores_cached_results_for_cards`、`empty_history_errors_have_python_messages`、`travel_with_stale_revision_conflicts`。失敗確認 → **Step 2–3:** 実装・PASS、コミット `feat: undo/redo history`。

### Task 6: 読み辞書と台本分割・推定時間
**Files:** Create `crates/irodori-store/src/{dictionary.rs,splitter.rs}`; Test inline ＋ `tests/fixtures/{splitter_cases.json,dictionary_cases.json}`; Modify `gen_fixtures.py`

**Interfaces:**
- Produces: `dictionary::{load(root:&Path)->Result<BTreeMap<String,String>>, save(root:&Path, word:&str, reading:&str)->Result<String /*message*/>, delete(root:&Path, word:&str)->Result<String>, apply(text:&str, entries:&BTreeMap<String,String>)->String}`（`config/reading_dictionary.json`。メッセージは `easy_dictionary.py` と同一。最長一致・置換後は再置換しない）。`splitter::{seconds(text:&str)->f64, estimate(text:&str, dict:&BTreeMap<String,String>, scale:f64)->f64, split_text(text:&str, dict:&BTreeMap<String,String>, scale:f64)->Result<Vec<String>>}`（書記素は `unicode-segmentation`。ロジックは `easy_splitter.py` を忠実に: 25 秒目標、180 書記素上限、句読点優先、英数字の語境界、辞書語は不可分）。

- [ ] **Step 1:** Python で `splitter_cases.json`（text, dict, scale → estimate と split 結果。日本語長文・英数字・辞書語・絵文字・結合文字・改行のみ・空文字・辞書語が長すぎるエラー）と `dictionary_cases.json`（apply の入力/出力）を生成。
- [ ] **Step 2:** テスト `estimate_matches_python`、`split_matches_python_for_every_case`、`split_is_lossless`（結合すると元の文字列）、`dictionary_apply_matches_python`、`dictionary_save_update_delete_messages`（登録/更新/削除/不在削除/空入力/長すぎ/改行）、`corrupt_dictionary_is_not_overwritten`（読み込み失敗時 save しても元ファイル不変でエラー）を書き、失敗確認。
- [ ] **Step 3–4:** 実装・PASS、コミット `feat: reading dictionary, splitter and duration estimate`。

### Task 7: 生成結果の署名・一致判定・再生スタンプ
**Files:** Create `crates/irodori-store/src/signature.rs`; Test inline ＋ `tests/fixtures/signature_cases.json`; Modify `gen_fixtures.py`

**Interfaces:**
- Consumes: Task 2, 3, 6。Produces: `signature::{reading_text(doc:&Project, card:&Card, dict:&BTreeMap<String,String>)->String, input_signature(doc:&Project, card:&Card, reading:&str, voice:Option<&Voice>)->String, signature(doc, card, dict, voice)->String, result_matches(doc, card, dict, voice)->bool, playback_stamp(doc, dict, voices:&[Voice], file_exists:&dyn Fn(&str)->bool)->String}`。
- 仕様（`easy_service.py` 55〜85 行）: 署名 = `sha256(dumps_sorted([text, voice_id, voice の sound_revision（無ければ revision。話者なしは null）, speed, silence, generation, reading]))` の hex。`result_matches` は署名一致、または旧形式（`result.request.dictionary` が辞書で、旧署名と一致し、`request.text` が現在の読み置換後テキストと一致）。`reading_text` は `dictionary_enabled` が偽なら置換しない。

- [ ] **Step 1:** Python（`Service.signature` 等を v2.0 のコードで実行）で `signature_cases.json` を生成（複数カード・話者あり/なし・辞書あり/なし・speed 1/1.0/0.75・generation に None/int/float/bool/str・日本語/絵文字・旧形式 result）。
- [ ] **Step 2:** テスト `signature_matches_python_for_every_case`（**最重要。1 文字でも違えば失敗**）、`signature_changes_when_any_input_changes`（text/voice/speed/silence/generation/辞書）、`legacy_result_signature_is_still_accepted`、`playback_stamp_matches_python` を書き、失敗確認。
- [ ] **Step 3–4:** 実装・PASS、コミット `feat: python-compatible generation signatures`。

### Task 8: 音声アセット登録と `/api/state` ビルダ
**Files:** Create `crates/irodori-store/src/{assets.rs,state.rs}`; Test inline

**Interfaces:**
- Consumes: Task 4〜7。Produces: `assets::Assets`（`new()`、`register(path:&Path)->Option<String>`（存在するファイルのみ。キー = 解決済みパスの SHA-256 hex）、`source(key:&str)->Result<PathBuf>`（無ければ `Invalid("音声が見つかりません。もう一度追加してください。")`））。`state::build_state(store:&Store, assets:&Assets, pid:Option<&str>, jobs:&[Value], backend:&str)->Result<Value>`。
- 仕様（`easy_service.state` を忠実に）: `project`（各カードに `estimate`, `audio`, `download_name`、状態の導出: 結果が現行入力と不一致→「再生成が必要」、音声ファイル無し→「音声が見つかりません」、話者が無い→「話者を選び直してください」）、`projects`、`voices`（各 `asset`）、`settings`、`models`（`easy_models.MODELS` の 3 件。`installed` は M2a では常に false）、`dictionary`、`jobs`、`backend`、`history{undo,redo}`、`playback_stamp`。

- [ ] **Step 1:** テスト: `state_has_every_field_the_editor_reads`（`project settings voices models dictionary jobs backend history playback_stamp`、カードの `estimate audio download_name status`、voice の `asset`）、`card_status_is_derived_like_python`（上記 3 ケース）、`stale_result_shows_regeneration_needed`、`assets_register_only_existing_files`、`asset_keys_are_stable_sha256_of_the_resolved_path`、`unknown_asset_key_is_an_error_with_the_python_message`。失敗確認 → **Step 2–3:** 実装・PASS、コミット `feat: asset registry and state builder`。

### Task 9: 台本の編集・操作（edit / action / project / recover）
**Files:** Create `crates/irodori-store/src/ops.rs`; Test inline

**Interfaces:**
- Consumes: Task 4〜8。Produces: `ops::{edit(store:&Store, pid:&str, revision:i64, values:&Value)->Result<Project>, action(store:&Store, root_dict:&BTreeMap<String,String>, pid:&str, revision:i64, action:&str, cid:Option<&str>, order:Option<&[String]>)->Result<ActionOutcome{message:String}>, new_or_load_project(store:&Store, req:&Value)->Result<Project>, recover_previous(store:&Store, pid:&str, revision:i64, busy:bool)->Result<Project>}`。
- 仕様（`easy_service.edit/action/split_doc`、`easy_web.project/recover_previous`）: `edit` は name(100 文字)、selected、dictionary_enabled、generation(検証)、カード patch（text/voice_id/filename、10 万文字超は `1回の入力は10万文字以内にしてください。`、speed/silence の範囲）。`action` は add（`inherit` 設定で選択カードの voice/speed/silence を継承）/delete（最後の 1 枚なら空カードを作る）/duplicate/up/down/reorder（集合不一致は「並べ替え対象が変わりました。画面を読み直してください。」）/reset/split（`続きのカードを{n}枚追加しました。` / `分割は不要です。`）/undo/redo。カード 1000 枚上限は「セリフは1000枚までです。不要なセリフを削除するか、台本を分けてください。」、分割後 1000 超は「台本は1000枚以内に分けて保存してください。」。`project`: `load` で `last_project` 更新、`name` で新規（100 文字）、`copy` で内容を引き継ぐ。

- [ ] **Step 1:** テスト: `edit_updates_cards_and_bumps_revision`、`edit_with_stale_revision_conflicts_and_changes_nothing`、`text_over_100000_chars_is_rejected_and_state_is_unchanged`、`speed_and_silence_out_of_range_are_rejected`、`add_inherits_voice_speed_silence_when_enabled`、`delete_last_card_leaves_one_empty_card`、`duplicate_up_down_move_cards_as_python`、`reorder_requires_exactly_the_same_ids`、`more_than_1000_cards_is_rejected`（add/duplicate/split）、`split_creates_continuation_cards_and_reports_the_count`、`undo_redo_actions_return_messages`（「元に戻しました。」「やり直しました。」）、`project_load_new_and_copy`、`recover_previous_restores_cards_and_refuses_while_busy`。失敗確認。
- [ ] **Step 2–3:** 実装・PASS、コミット `feat: script editing operations`。

### Task 10: axum の骨組み（安全対策・エラー変換・画面配信・メタ）
**Files:** Create `crates/irodori-app/src/{lib.rs,security.rs,errors.rs,assets_embed.rs,meta.rs}`, `crates/irodori-app/editor/{index.html,app.js,style.css}`（v2.0 からそのままコピー）, `crates/irodori-app/data/{emoji_palette.json,credits.md}`, `tools/compat/gen_meta.py`; Test `crates/irodori-app/tests/http_security.rs`

**Interfaces:**
- Produces: `irodori_app::{AppState{store:Arc<Store>, assets:Arc<Assets>, token:String, root:PathBuf, backend:String}, build_router(state:AppState)->axum::Router}`。`GET /` は `index.html` の `__TOKEN__` を実トークンに置換し `Cache-Control: no-store`。`GET /editor/{name}` は `app.js`/`style.css` のみ（他は 404）。`GET /api/meta` は `{emojis:[{emoji,label,description}], emoji_groups:[{title,items:[…]}], credits:<HTML>}`。ミドルウェア `security::local_only`（`Host` 検証、非 GET のトークン検証は `subtle` で定数時間比較）。`errors::ApiError`（`StoreError`→JSON 応答）。
- `gen_meta.py` は上流の `EMOJI_PALETTE_ITEMS`（`irodori_tts.gradio_emoji_palette`）と v2.0 の `grouped_emojis`、`easy_ui.footer_text()` から `emoji_palette.json`（グループ済み）と `credits.md` を生成する。クレジットは `pulldown-cmark` で HTML 化（HTML 直接埋め込みは無効）。

- [ ] **Step 1:** テスト（`tower::ServiceExt::oneshot`、`Host` を指定）: `foreign_host_is_forbidden`、`localhost_and_127_0_0_1_are_allowed`、`post_without_token_is_forbidden`、`post_with_wrong_token_is_forbidden`、`get_without_token_is_allowed`、`index_injects_the_token_and_disables_caching`、`editor_serves_only_app_js_and_style_css`（`../Cargo.toml`・`index.html`・`secret` は 404）、`meta_contains_emoji_groups_and_rendered_credits`（クレジットにライセンス節と禁止事項が含まれる）、`store_errors_map_to_status_and_japanese_json`（400/409/500）。失敗確認。
- [ ] **Step 2–3:** 実装・PASS、コミット `feat: http shell with local-only security and embedded editor`。

### Task 11: M2a のルート
**Files:** Create `crates/irodori-app/src/routes.rs`; Test `crates/irodori-app/tests/http_api.rs`

**Interfaces:**
- Consumes: Task 8〜10。Produces: `GET /api/state?pid=`、`GET /audio/{key}?download=&name=`（`audio/wav`、`download` 時は `safe_name` した `filename`。M2a では `safe_name` を `irodori_store::filename::safe_name` として最小実装: 禁止文字 `<>:"/\|?*` と制御文字を `_`、末尾の空白/ドット除去、100 文字、Windows 予約名の先頭に `_`）、`POST /api/{edit,action,project,settings,dictionary,recover-previous}`（本文は v2.0 と同じ JSON。`/api/state` と同形の応答を返す箇所は `build_state`）。`/api/dictionary` は `{message}`。`/api/settings` は「設定を保存しました。生成モード・自動起動は次回起動時に反映されます。」。M2b/M2c のルートは 501「未対応です（この版ではまだ使えません）。」を返す（画面が落ちないように）。

- [ ] **Step 1:** テスト: `state_returns_a_new_project_on_first_run`、`edit_then_state_roundtrip`、`stale_revision_returns_409_with_python_message`、`action_add_delete_reorder_split_undo_redo_over_http`、`settings_validation_errors_are_400`、`dictionary_register_update_delete_over_http`、`audio_requires_a_registered_key`、`audio_download_uses_a_safe_filename`、`unimplemented_routes_return_501_json`、`invalid_pid_is_400_for_state`。失敗確認 → **Step 2–3:** 実装・PASS、コミット `feat: m2a api routes`。

### Task 12: 起動処理（irodori-app バイナリ）
**Files:** Modify `crates/irodori-app/src/main.rs`; Create `crates/irodori-app/src/launch.rs`; Test inline ＋ `tests/launch.rs`

**Interfaces:**
- Produces: `launch::{Args{data_dir:PathBuf, port:u16, no_browser:bool}（clap。既定 data_dir は exe と同じ場所、port 7860）, pick_port(preferred:u16)->Result<u16>（使用中なら次の空きを探す）, acquire_lock(root:&Path)->Result<LockGuard>（temp/app.lock。既に別プロセスが保持なら「アプリはすでに起動しています。」）, open_browser_when_ready(url:String)}`。`fresh_start`（v2.0: `restore_input` が偽なら起動ごとに新しい台本を開き、前回に入力があれば `recovery_project` に記録）。ロギングは `logs/editor.log`、`GET /api/state` の成功は記録しない。

- [ ] **Step 1:** テスト: `pick_port_skips_a_busy_port`、`second_lock_is_refused_with_a_clear_message`、`lock_is_released_on_drop`、`fresh_start_records_a_recovery_project_only_when_text_exists`、`fresh_start_is_skipped_when_restore_input_is_true`、`args_defaults`。失敗確認 → **Step 2–3:** 実装・PASS、`cargo run -p irodori-app -- --data-dir <tmp> --no-browser` で起動し `curl http://127.0.0.1:<port>/api/state` が 200 を返すことを確認。コミット `feat: app launcher`。

### Task 13: v2.0 互換の往復テスト
**Files:** Create `crates/irodori-store/tests/compat_roundtrip.rs`; Modify `tools/compat/gen_fixtures.py`（`--make-data-dir <dir>`: v2.0 の `Store` で実データ（複数台本・話者・辞書・設定・生成結果つきカード）を作る）、`tools/compat/verify_python_reads.py`（Rust が書いたフォルダを v2.0 の `Store` で読み、期待内容を検証）

**Interfaces:** 実行は手動＋`#[ignore]`（Python 環境が要るため）。`IRODORI_COMPAT_DIR` に Python が作ったデータフォルダを渡す。

- [ ] **Step 1:** テスト（`#[ignore]`）: `python_made_data_loads_edits_and_saves_without_losing_fields`（読み込み→カード編集→保存→全フィールド（未知項目含む）が保持）、`signatures_of_python_results_still_match`（Python が作った生成結果つきカードが `result_matches` で true）、`state_for_python_data_has_no_regeneration_needed_for_unchanged_cards`、`rust_written_folder_is_readable_by_python`（`verify_python_reads.py` を実行）。
- [ ] **Step 2:** Python でデータ作成 → 実行し失敗箇所を修正（期待は全 PASS）。結果と手順を `tools/compat/README.md` に記録。コミット `test: v2.0 compatibility round-trip`。

### Task 14: 実ブラウザでの通し確認と文書
**Files:** Create `docs/m2a-verification.md`; Modify `README.md`（起動方法）

- [ ] **Step 1:** `irodori-app` を起動し、Playwright（Chrome）で次を確認して結果とスクリーンショットを `docs/m2a-verification.md` に記録: 画面が表示される／カードの追加・編集・削除・並べ替え・複製・分割／元に戻す・やり直し／台本の新規・切替・複製／辞書の登録・更新・削除／設定の保存／クレジットタブ／`生成`ボタンで「未対応です」が表示され画面が壊れない／ブラウザ再読み込み後に編集内容が残る／ブラウザのコンソールに未処理エラーが無い。
- [ ] **Step 2:** 問題があれば該当タスクのテストを先に足して修正（RED→GREEN）。コミット `docs: m2a browser verification`。

---

## Self-Review メモ
- 仕様 §3（構成）→ Task 1,10,12。§4（互換）→ Task 2,3,4,5,7,13。§5（M2a の API）→ Task 8,9,11。§8（辞書・分割・meta）→ Task 6,10。§10（安全対策）→ Task 10,11。§13 の最大リスク（署名の一致）→ Task 2,7 を最優先に配置。
- 型整合: `Store`/`Project`/`Card`/`Voice`/`Settings`/`Generation` は Task 3,4 で定義し、Task 5〜11 が参照。`Assets` は Task 8、`AppState` は Task 10。`ops::action` の辞書引数は Task 6 の型。
- M2b（生成・話者・DL）と M2c（書き出し等）のルートは Task 11 で 501 を返し、後続の計画で実装する。
