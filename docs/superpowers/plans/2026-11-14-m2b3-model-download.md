# M2b-3 モデルの自動ダウンロード Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 未取得のモデルで生成・試作を頼むと、画面が「取得しますか（約 3.3GB）」と確認し、承認後に Hugging Face から取得して（進捗表示・キャンセル・再開つき）、そのまま使えるようになる。**ユーザーの確認なしにはネットワークへ出ない。**

**Architecture:** `irodori-app` に `download/` を足す。`Hub`（ureq の同期クライアント。HF の API でファイル一覧・サイズ・SHA-256 を得て、許可ホストだけへリダイレクトを手で追う）→ `fetch`（Range で再開し、検証してから rename）→ `install_model`（モデルと、無ければ codec を HF キャッシュ互換のフォルダへ置き、最後に `refs/main` を書く）。`JobManager` に `kind="download"` のジョブを足し（既存のワーカー・取消・進捗の仕組みを使う）、`ModelInstaller` トレイトで本物（`RealInstaller`）と擬似（`FakeInstaller`）を差し替える（テストはネットに出ない）。

**Tech Stack:** Rust、`ureq` 2（rustls）、`sha2`、`fs4`（空き容量）、axum（テスト用の偽 HF サーバ）。

**Spec:** `docs/superpowers/specs/2026-11-13-m2-server-and-editor-design.md`（§5 の `/api/models/download` `/api/models/cancel`、§7、§9、§10）。画面の挙動の正解は v2.0 の `editor/app.js`（`startGeneration` `generateRegistration`）。

## M2b の分割（この計画は 3/3）
M2b-1（生成）・M2b-2（話者）は main 済み。この計画で M2b が完了する。

## Global Constraints
- **取得してよいのは許可リストだけ**: `Aratako/Irodori-TTS-v4.1-Small`、`phasefield-audio/Irodori-TTS-v4.1-Anime`、`Aratako/Irodori-TTS-v4.1-Small-MF`（`models::ALLOW_LIST`）と、共通 codec `Aratako/Semantic-DACVAE-Japanese-32dim`（`models::CODEC_ID`）。取得するファイルは固定の一覧だけ: モデルは `model.safetensors` `tokenizer/tokenizer.json` `tokenizer/tokenizer_config.json`、codec は `weights.pth`。サーバが返すパスは保存先に使わない（一覧に無ければ無視）。
- 通信は **HTTPS のみ**。接続先・リダイレクト先のホストは `huggingface.co` / `hf.co` かそのサブドメインだけ。IP リテラル・ユーザー情報つき URL・`http` へのリダイレクトは拒否。リダイレクトは最大 5 回、手で追い、許可ホスト以外へは**要求を出さない**。接続 15 秒・読み取り 30 秒のタイムアウト。自動の再試行はしない（利用者が再実行し、再開する）。
- 取得は**同じコミット**で行う: `GET /api/models/<repo>` の `sha`（40 桁の小文字 hex。フォルダ名に使うので厳密に検査）→ `GET /api/models/<repo>/tree/<sha>?recursive=1` → `GET /<repo>/resolve/<sha>/<path>`。LFS のファイルは `lfs.oid`（SHA-256）で検証、それ以外はサイズだけで検証。
- 置き場所は HF キャッシュ互換: `<hub>/models--<org>--<name>/snapshots/<sha>/<path>`。書きかけは `<path>.incomplete`、検証後に rename。**`refs/main` はそのリポジトリの全ファイルが検証済みになってから最後に書く**（途中の状態が「取得済み」に見えないように）。既存の `models::locate` を変えない。
- 空き容量: 残りのバイト数 × 1.2 を、書き込み先のドライブの空きと比べ、足りなければ取得を始めない。
- 文言（日本語）: オフライン「ネットワークに接続できません。接続を確認してもう一度お試しください。」／HTTP エラー「モデルの取得元から取得できませんでした（HTTP {code}）。」／検証失敗「ダウンロードしたファイルの検証に失敗しました。もう一度お試しください。」／容量「空き容量が足りません。あと約{n}GB 必要です。」／取得済み「このモデルはすでに取得済みです。」／実行中「現在の処理が終わってから実行してください。」／確認なし「取得の確認が必要です。」
- 想定外の失敗の詳細はログだけ。1 ファイル 400 行以内（最大 800）。`cargo test -p irodori-store -p irodori-app` は常に緑。**テストは実ネットワークに出ない**（偽サーバ・偽インストーラ）。実ネットでの確認は Task 7 だけ（約 3.3GB。実行前に利用者へ確認する）。

## Review Focus
1. 途中で切れた・アプリを落とした → 次の取得が Range で再開し、書きかけは決して「取得済み」に見えない（`refs/main` が最後）。Range を無視する（200 を返す）サーバでは最初からやり直す（Task 2, 3）。
2. 壊れた／すり替えられた中身（SHA-256 不一致・サイズ不足・エラーページを 200 で返す）→ 拒否して書きかけを消し、明確な文言（Task 2）。
3. 許可外ホスト・`http`・IP・ユーザー情報つきへのリダイレクト、`..` を含むパスを返す API → 要求を出さない／無視する（Task 1, 2）。
4. 空き容量不足・書き込み中のディスクフル → 始める前に断る／最終ファイルを残さない（Task 3）。
5. **確認なしでは取得しない**: 状態の取得・起動・生成の要求だけでは通信しない。`/api/models/download` は `confirm: true` が無ければ 400（Task 4, 5）。
6. 取得中の再要求・生成要求・キャンセル（検証中を含む）→ 400／「停止」で、書きかけは再開用に残る（Task 4, 5）。

---

### Task 1: HF のクライアントと取得計画（`download/hub.rs`）
**Files:** Create `crates/irodori-app/src/download/{mod.rs,error.rs,hub.rs}`、`crates/irodori-app/tests/support/fake_hub.rs`（偽 HF サーバ）、`crates/irodori-app/tests/hub_plan.rs`; Modify `crates/irodori-app/{Cargo.toml,src/lib.rs}`（`pub mod download;`、`ureq = { version = "2", default-features = false, features = ["tls"] }`、`sha2.workspace = true`、`fs4`、dev: `tokio`/`axum` は既存）

**Interfaces:**
- Produces:
  - `pub enum DownloadError { Offline, Http(u16), Verification, DiskSpace { needed: u64, available: u64 }, Cancelled, Protocol(String), Io(String) }` と `impl DownloadError { pub fn message(&self) -> String }`（Global Constraints の文言。`Protocol`/`Io` は一般文言「処理に失敗しました。logs/editor.logを確認してください。」で、詳細は `Display` でログへ）。
  - `pub enum HostPolicy { HuggingFace, Local }`（`Local` はテスト専用: `http` と `127.0.0.1` を許す）、`pub struct Hub`、`Hub::huggingface() -> Hub`、`Hub::with_base(base: &str, policy: HostPolicy) -> Hub`。
  - `pub struct PlannedFile { pub path: String, pub size: u64, pub sha256: Option<String> }`、`pub struct RepoPlan { pub repo: String, pub sha: String, pub files: Vec<PlannedFile> }`。
  - `Hub::plan(&self, repo: &str, wanted: &[&str]) -> Result<RepoPlan, DownloadError>`。`wanted` の全ファイルが API の一覧に無ければ `Protocol`。`wanted` 以外は捨てる。`sha` は `^[0-9a-f]{40}$`、`sha256` は `^[0-9a-f]{64}$` でなければ `Protocol`。
  - 偽サーバ（テスト）: `FakeHub::start(files: Vec<FakeFile>) -> FakeHub`（`base_url()`、要求の記録 `requests()`、挙動の切り替え: `redirect_to(host)`・`ignore_range()`・`truncate_after(bytes)`・`serve_html_with_200()`・`status(path, code)`）。axum を別スレッドの tokio で動かす。`/api/models/<org>/<name>` と `/api/models/<org>/<name>/tree/<sha>`、`/<org>/<name>/resolve/<sha>/<path>`（Range 対応、302 の切り替え可）。

- [ ] **Step 1: 失敗するテストを書く**（`hub_plan.rs`）: `plan_reads_sha_sizes_and_lfs_hashes`（LFS の 2 ファイルと非 LFS の 1 ファイル → `sha256` は前者だけ `Some`）、`plan_ignores_files_that_were_not_asked_for`（`../evil` や `README.md` を返しても `files` に入らない）、`plan_fails_when_a_wanted_file_is_missing`、`plan_rejects_a_malformed_commit_sha_or_hash`（`../../x`、39 桁、大文字、空）、`plan_maps_http_errors_and_offline`（404 → `Http(404)`、閉じたポート → `Offline`）、`non_https_base_is_refused_for_the_huggingface_policy`。
- [ ] **Step 2: 実行して失敗を確かめる**: `cargo test -p irodori-app --test hub_plan` → Expected: コンパイル失敗（`download` が無い）。
- [ ] **Step 3: 実装**。`Hub` は `ureq::Agent`（リダイレクト 0 回・上記タイムアウト）。ホスト検査は小さな純関数 `host_allowed(url: &str, policy) -> bool` にして単体テスト（`huggingface.co`、`cdn-lfs.hf.co`、`cas-bridge.xethub.hf.co`、`evilhuggingface.co`、`huggingface.co.evil.com`、`user@huggingface.co`、`127.0.0.1`、`http://huggingface.co`）。
- [ ] **Step 4: 通す**: `cargo test -p irodori-app --test hub_plan` と `cargo test -p irodori-app --lib download::` → Expected: 通る。
- [ ] **Step 5: Commit** `feat(app): hugging face client and download planning`

### Task 2: 1 ファイルの取得（再開・検証・取消）（`download/fetch.rs`）
**Files:** Create `crates/irodori-app/src/download/fetch.rs`、`crates/irodori-app/tests/hub_fetch.rs`; Modify `download/mod.rs`

**Interfaces:**
- Consumes: Task 1 の `Hub`・`PlannedFile`・`DownloadError`、偽サーバ。
- Produces: `pub struct Transfer<'a> { pub cancel: &'a AtomicBool, pub progress: &'a dyn Fn(u64) }`（`progress` は「新しく進んだバイト数」。再開時は、最初に既存の書きかけの大きさを 1 回通知する）、`Hub::fetch(&self, repo: &str, sha: &str, file: &PlannedFile, dest: &Path, transfer: &Transfer) -> Result<(), DownloadError>`。
- 動作: 書きかけは `<dest>.incomplete`。あれば `Range: bytes=<長さ>-` で再開（206 以外・`Content-Range` の開始がずれていれば最初から）。リダイレクト（301/302/303/307/308）は最大 5 回、各 `Location` を `host_allowed` で検査してから要求（相対 URL は元のホストに対して解決）。本文は 256 KiB ずつ書き、SHA-256 を流しながら計算（再開時は既存分を先に読み込んで計算）、`cancel` を毎回確認（立っていれば `Cancelled`、書きかけは残す）。完了したらサイズ（と `sha256` があれば SHA-256）を検証し、不一致なら書きかけを消して `Verification`。`Content-Length` が計画と食い違えば `Protocol`。検証後に rename、`dest` の親フォルダは作る。書き込み失敗は `Io`（書きかけは残す／消してよい）。

- [ ] **Step 1: 失敗するテストを書く**（`hub_fetch.rs`、偽サーバ）: `fetch_downloads_verifies_and_renames`（`.incomplete` が残らない・`progress` の合計がサイズ）、`fetch_resumes_a_partial_file_with_a_range_request`（途中で切れる → 次の呼び出しが `Range` つきで残りだけ取る。要求の記録で確認）、`fetch_restarts_when_the_server_ignores_range`、`fetch_follows_a_redirect_to_an_allowed_host`、`fetch_refuses_a_redirect_to_a_forbidden_host_without_requesting_it`（別ホストへの要求が 0 件）、`fetch_rejects_a_hash_mismatch_and_removes_the_partial_file`、`fetch_rejects_a_short_body`、`fetch_rejects_an_html_error_page_served_with_200`（サイズ・ハッシュ不一致で拒否）、`fetch_rejects_too_many_redirects`、`fetch_can_be_cancelled_and_keeps_the_partial_file`（`cancel` を途中で立てる）、`fetch_maps_http_404_and_a_dropped_connection`。
- [ ] **Step 2: 実行して失敗を確かめる**: `cargo test -p irodori-app --test hub_fetch` → Expected: コンパイル失敗。
- [ ] **Step 3: 実装**（上の動作）。`fetch.rs` は 400 行以内。
- [ ] **Step 4: 通す**: `cargo test -p irodori-app --test hub_fetch` → Expected: 通る。
- [ ] **Step 5: Commit** `feat(app): resumable verified file fetch`

### Task 3: モデルの導入（`download/install.rs`）
**Files:** Create `crates/irodori-app/src/download/install.rs`、`crates/irodori-app/tests/hub_install.rs`; Modify `download/mod.rs`

**Interfaces:**
- Consumes: Task 1・2、`models::{ALLOW_LIST, CODEC_ID, locate}`。
- Produces:
  - `pub const APPROX_MODEL_BYTES: u64 = 3_100_000_000; pub const APPROX_CODEC_BYTES: u64 = 430_000_000;`（画面の確認用の概算）、`pub fn approximate_bytes(hub_dir: &Path, model_id: &str) -> u64`（codec が取得済みなら codec 分を引く。許可外は 0）。
  - `pub fn install_model(hub: &Hub, hub_dir: &Path, model_id: &str, transfer: &Transfer, on_total: &dyn Fn(u64)) -> Result<(), DownloadError>` — 許可リスト外は `Protocol`。モデルと、`locate` で codec が無いときだけ codec を計画 → 合計（既に完成しているファイルは除く）を `on_total` へ → 空き容量（`fs4::available_space` を、`hub_dir` の最も近い既存の親で。残り×1.2 に満たなければ `DiskSpace`）→ 各ファイルを `snapshots/<sha>/<path>` へ（同じサイズで既にあるファイルは取り直さず、サイズを `progress` へ）→ そのリポジトリの全ファイルが済んだら `refs/main` を書く（原子的に）。冪等（取得済みなら通信せず `Ok`）。
- [ ] **Step 1: 失敗するテストを書く**（`hub_install.rs`）: `install_places_a_model_and_codec_where_locate_finds_them`（導入後に `locate(hub, id)` が `Some`）、`install_skips_the_codec_when_it_is_already_installed`（codec への要求が 0 件）、`refs_main_is_written_last`（途中で失敗させると `refs/main` が無く `locate` は `None`）、`install_resumes_after_a_failure`、`install_is_idempotent_when_everything_is_present`（要求 0 件）、`install_refuses_when_there_is_not_enough_free_space`（`fs4` の戻りを注入できる内部関数 `install_with_space(…, available: u64)` で）、`models_outside_the_allow_list_are_refused`、`on_total_counts_only_what_is_missing`、`approximate_bytes_drops_the_codec_when_installed`、`install_never_writes_outside_the_hub_dir`（API が `../..` を返しても `wanted` だけが置かれる）。
- [ ] **Step 2: 実行して失敗を確かめる**: `cargo test -p irodori-app --test hub_install` → Expected: コンパイル失敗。
- [ ] **Step 3: 実装**。
- [ ] **Step 4: 通す**: `cargo test -p irodori-app` → Expected: 通る。
- [ ] **Step 5: Commit** `feat(app): install models into the hf-compatible cache`

### Task 4: 取得ジョブ（`kind="download"`）と `ModelInstaller`
**Files:** Create `crates/irodori-app/src/download/real.rs`（`RealInstaller`）、`crates/irodori-app/src/jobs/download.rs`、`crates/irodori-app/src/jobs/download_tests.rs`; Modify `jobs/{mod.rs,run.rs,submit.rs,design.rs}`、`fake.rs`（`FakeInstaller`、`FakeSynth` の未取得を後から切り替えられるように: `set_installed(&self, model: &str, installed: bool)`）、`lib.rs`（`AppState::new(..., synth, installer)`）、`launch.rs`（`RealInstaller` を渡す）、既存のテストの `JobManager::new`／`AppState::new` 呼び出し（`jobs/tests.rs`、`jobs/design_tests.rs`、`tests/common/mod.rs`、`tests/launch.rs`）

**Interfaces:**
- Produces:
  - `pub struct DownloadControl { pub cancel: Arc<AtomicBool>, pub done: Arc<AtomicU64>, pub total: Arc<AtomicU64> }`、`pub trait ModelInstaller: Send + Sync { fn install(&self, model: &str, control: &DownloadControl) -> Result<(), DownloadError>; fn approximate_bytes(&self, model: &str) -> u64; }`。`RealInstaller::new(hub_dir: PathBuf) -> RealInstaller`（`Hub::huggingface()` と `install_model` を使い、`done`/`total` を更新）。`FakeInstaller`（`new(on_install: impl Fn(&str) + Send + Sync)`、`with_steps`・`with_delay`・`fail_with(DownloadError)`・`installs()`）。
  - `JobManager::new(store, synth, assets, installer: Arc<dyn ModelInstaller>)`。`Kind::Download`。`Job` に `model: String`、`bytes_done`/`bytes_total`（`Arc<AtomicU64>`）。`jobs_json` に `model`、`bytes_done`、`bytes_total`（生成・試作では 0）。
  - `JobManager::submit_download(&self, model: &str, confirmed: bool) -> Result<String /* job id */>` — 検証順: 許可外 → 「使用するモデルを選んでください。」／`confirmed` が偽 → 「取得の確認が必要です。」／他のジョブが動いている → 「現在の処理が終わってから実行してください。」／すでに取得済み（`synth.model_available`）→ 「このモデルはすでに取得済みです。」。
  - `JobManager::cancel_downloads(&self)` — 動いている download ジョブにだけ取消を立てる（`stop()` も download を止める）。
  - `SubmitOutcome::NeedsModelDownload { model: String }`（`submit` が、対象カードの話者のモデルが未取得のとき、これまでの「モデルがまだ取得されていません。」エラーの代わりに返す）。`submit_design` は `Result<DesignSubmit>`（`enum DesignSubmit { Started(String), NeedsModelDownload(String) }`）を返す。
  - ワーカー: `Work::Download { job_id, model }`。結果は 完了／失敗（`DownloadError::message()`、`Protocol`/`Io` は一般文言でログに詳細）／停止（`Cancelled`）。panic でもワーカーは生きる。`status` は既存の `BUSY` と同じ語（待機中・生成中）を使う。

- [ ] **Step 1: 失敗するテストを書く**（`download_tests.rs`、`FakeInstaller`）: `download_job_runs_and_reports_bytes`（`bytes_done` が増え、`bytes_total` が入り、完了後にそのモデルが `model_available`）、`unconfirmed_download_is_refused_and_creates_no_job`、`unknown_model_is_refused`、`already_installed_is_refused`、`a_busy_manager_refuses_a_download_and_a_download_blocks_generation_and_design`、`cancel_downloads_stops_only_downloads`（生成ジョブは止めない）、`stop_also_cancels_a_download`、`failed_install_marks_the_job_failed_with_the_message`（オフライン・容量・検証の 3 種の文言）、`install_panic_does_not_kill_the_worker`、`generate_returns_needs_model_download_for_a_missing_model`（ジョブは作られず、通信も起きない＝`FakeInstaller::installs()` が空）、`design_returns_needs_model_download_for_a_missing_model`、`jobs_json_has_the_download_shape`（`kind`・`model`・`bytes_done`・`bytes_total`・`project: null`）。
- [ ] **Step 2: 実行して失敗を確かめる**: `cargo test -p irodori-app --lib jobs::download` → Expected: コンパイル失敗。
- [ ] **Step 3: 実装**。`jobs/mod.rs` は 400 行以内に（超えるなら `Job`/`Watch` を `job.rs` へ）。既存呼び出しは機械的に直す。
- [ ] **Step 4: 通す**: `cargo test -p irodori-store -p irodori-app` → Expected: 既存すべて＋新規が通る（M2b-1・M2b-2 のテストが無傷）。
- [ ] **Step 5: Commit** `feat(app): model download jobs behind a ModelInstaller`

### Task 5: ルートと状態（`/api/models/download` `/api/models/cancel`、確認の応答）
**Files:** Modify `routes.rs`、`voices.rs`（`/api/design*` の応答）、`crates/irodori-store/src/state.rs`（`models[]` の `download_bytes` は app 側で重ねる。store は触らない）、`tests/http_models.rs`（新規）、`tests/http_api.rs`（501 の一覧は変わらない）

**Interfaces:**
- Produces:
  - `POST /api/models/download` `{model, confirm}` → 200 `{"job": <id>}`。`confirm` は **真（`true`）のときだけ**有効。エラーは Task 4 の文言で 400。
  - `POST /api/models/cancel` → 200 `{"message": "取得を中止します。"}`（download が無くても 200）。
  - `POST /api/generate` が未取得のモデルを必要とするとき → 200 `{"needs_model_download": true, "model": <id>, "name": <表示名>, "download_bytes": <概算>}`（`needs_long_confirmation` と同じ流儀。ジョブ・カードの状態は変えない）。`POST /api/design`・`/api/design-registration` も同じ形（`job` の代わりに）。
  - `/api/state` の `models[]` に `download_bytes`（`installer.approximate_bytes`。取得済みなら 0）。`jobs[]` は Task 4 のとおり。
- [ ] **Step 1: 失敗するテストを書く**（`http_models.rs`、`FakeInstaller`）: `download_requires_confirm_true`（`confirm` 無し・`false`・`"true"`・`1` は 400「取得の確認が必要です。」、ジョブ無し）、`download_returns_a_job_and_the_model_becomes_installed`（状態の `installed` が true、`download_bytes` が 0）、`state_shows_download_bytes_for_missing_models`、`generate_asks_before_downloading`（`needs_model_download`・`download_bytes`、通信 0）、`design_asks_before_downloading`、`generating_after_the_download_works`（取得 → 生成が完了）、`second_download_while_running_is_400`、`cancel_stops_a_running_download`、`cancel_without_a_download_is_ok`、`download_routes_require_the_token`、`state_polling_never_downloads`（`state` を 20 回呼んでも `installs()` が空）、`unknown_model_is_400`。
- [ ] **Step 2: 実行して失敗を確かめる**: `cargo test -p irodori-app --test http_models` → Expected: 404/失敗。
- [ ] **Step 3: 実装**。
- [ ] **Step 4: 通す**: `cargo test -p irodori-store -p irodori-app` → Expected: 通る。
- [ ] **Step 5: Commit** `feat(app): model download routes and confirmation responses`

### Task 6: 画面の最小の改修（確認・進捗・キャンセル）と起動の配線
**Files:** Modify `crates/irodori-app/editor/app.js`（および必要なら `style.css`、`index.html` の最小限）、`launch.rs`（確認）、`README.md`、`crates/irodori-app/tests/`（画面の文字列を直接検査する簡単なテストだけ）

- 画面の挙動: `generate`・`design`・`design-registration` の応答が `needs_model_download` のとき、**既存の `confirm()` ダイアログ**で「『{name}』をダウンロードします（約{n}GB）。インターネットに接続し、Hugging Face から取得します。よろしいですか？」と聞く。承認したら `POST /api/models/download {model, confirm: true}` → 状態のポーリングに出る download ジョブを、既存の状態表示欄に「モデルを取得中… {done}/{total}（{percent}%）」と出し、完了したら**元の操作（生成・試作）を自動でやり直す**。拒否したら何もしない（「取得しませんでした。」を通知）。取得中は生成・試作のボタンを無効にし、「取得を中止」ボタン（`/api/models/cancel`）を出す。失敗は `job.error` をそのまま表示。`app.js` を読んで（`startGeneration`、`generateRegistration`、状態のポーリングと `activeJobs`）、**既存の流れに最小の差分**で足す（書き換えない）。
- [ ] **Step 1: 失敗するテストを書く**: `editor_script_handles_needs_model_download`（`/editor/app.js` を取得し、`needs_model_download`・`models/download`・`models/cancel`・`confirm: true` を含む。`assets_embed` 経由の配信）。
- [ ] **Step 2: 実行して失敗を確かめる**: `cargo test -p irodori-app --test http_api editor_script` → Expected: 失敗。
- [ ] **Step 3: 実装**（上の挙動）。`launch.rs` が `RealInstaller::new(args.hub_dir())` を渡していることを確認する。README に「初回は確認のうえ自動取得（約 3.3GB）、`--models-dir` の置き場所、中断しても再開できる、確認なしには通信しない」を書く。
- [ ] **Step 4: 通す**: `cargo test -p irodori-store -p irodori-app` → Expected: 通る。
- [ ] **Step 5: Commit** `feat(app): download confirmation and progress in the editor`

### Task 7: 実ネットワーク・実ブラウザでの確認と文書
**Files:** Create `docs/m2b3-verification.md`、`docs/m2b3-screenshots/`; Modify `README.md`（M2b 完了の状態）

- [ ] **Step 1: 事前確認**: `taskkill //F //IM irodori-app.exe`、`cargo build --release -p irodori-app`、`cargo test -p irodori-core -p irodori-store -p irodori-app` → Expected: 緑。**実ネットワークで約 3.3GB を取得するので、実行前に利用者へ確認する**（空き容量 F: は約 15GB。確認用フォルダ `F:\claude\m2b3-verify-data` と空の `models`）。
- [ ] **Step 2: 画面で通し**: 空のモデルフォルダで起動 → 話者を登録（M2b-2 の手順、または既存の話者データ）→ 「生成」→ 確認ダイアログ（約 3.3GB）→ 拒否すると何も起きない（通信なし）→ 再度「生成」→ 承認 → 進捗（バイト数・%）→ 完了後に自動で生成され、音声が付く。`models[].installed` が true。スクリーンショットを保存。
- [ ] **Step 3: 再開・取消・検証**: 取得中に「取得を中止」→ 「停止」・`.incomplete` が残る → 再度承認 → 残りだけ取得して完了（取得済みのキャッシュ `F:\claude\upstream-irodori-cache\hf\hub` のファイルと SHA-256 が一致することを PowerShell の `Get-FileHash` で確認）。取得中にアプリを強制終了 → 再起動して再開。ネットワークを切って（`--models-dir` を空にし、`HTTPS_PROXY` 等ではなく、機内モード相当にできなければこの項目は偽サーバのテストで担保したと記録）。
- [ ] **Step 4: 文書**: `docs/m2b3-verification.md`（条件・結果の表・所要時間と速度・未解決）、README（M2b 完了）。`cargo test` の件数を記録。コミット `docs: m2b-3 verification`。

## 持ち越し（この計画の外）
- プロキシ環境（`HTTPS_PROXY`）の対応、並列ダウンロード、帯域制限、取得済みモデルの削除・更新の確認は扱わない（必要になったら別の計画）。
- 話速・書き出し・整理・入力済データは M2c。
