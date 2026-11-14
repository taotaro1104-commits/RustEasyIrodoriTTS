//! `JobManager` のテスト（擬似エンジン `FakeSynth` で、モデル無しに全経路を確かめる）。

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use irodori_store::dictionary;
use irodori_store::model::{Card, Project};
use irodori_store::signature::{result_matches, signature};
use irodori_store::{ops, Store, StoreError};
use serde_json::{json, Value};

use super::{JobManager, SubmitOutcome};
use crate::fake::{FakeInstaller, FakeSynth};
use crate::synth::{SynthError, Synthesizer};

pub(super) const MF: &str = "Aratako/Irodori-TTS-v4.1-Small-MF";
pub(super) const VOICE: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
pub(super) const GENERIC: &str = "処理に失敗しました。logs/editor.logを確認してください。";

pub(super) struct Env {
    pub(super) dir: tempfile::TempDir,
    pub(super) store: Arc<Store>,
    pub(super) synth: Arc<FakeSynth>,
    pub(super) installer: Arc<FakeInstaller>,
    pub(super) jobs: JobManager,
    pub(super) pid: String,
}

/// 話者を 1 人持つデータフォルダに、`texts` のカードを並べた台本を作る。
pub(super) fn env(texts: &[&str], build: impl FnOnce(FakeSynth) -> FakeSynth) -> Env {
    let dir = tempfile::tempdir().unwrap();
    for sub in ["projects", "voices", "config", "temp", "outputs"] {
        std::fs::create_dir_all(dir.path().join(sub)).unwrap();
    }
    let audio = format!("voices/{VOICE}/ref.wav");
    std::fs::create_dir_all(dir.path().join(&audio).parent().unwrap()).unwrap();
    std::fs::write(dir.path().join(&audio), b"RIFF").unwrap();
    let voices = json!([{"id": VOICE, "revision": 1, "audio": audio, "name": "話者A", "model": MF, "caption": "落ち着いた声"}]);
    std::fs::write(dir.path().join("voices/index.json"), voices.to_string()).unwrap();
    let store = Arc::new(Store::open(dir.path()).unwrap());
    let pid = store.load(None).unwrap().id;
    let cards: Vec<Card> = texts
        .iter()
        .map(|t| Card {
            text: (*t).into(),
            voice_id: VOICE.into(),
            ..irodori_store::model::new_card(None)
        })
        .collect();
    store
        .internal_update(&pid, &mut |d: &mut Project| {
            d.selected = cards[0].id.clone();
            d.cards = cards.clone();
        })
        .unwrap();
    let synth = Arc::new(build(FakeSynth::new(dir.path())));
    let installer = Arc::new(FakeInstaller::for_synth(Arc::clone(&synth)));
    let jobs = JobManager::new(
        Arc::clone(&store),
        synth.clone() as Arc<dyn Synthesizer>,
        Arc::new(irodori_store::assets::Assets::new()),
        installer.clone() as Arc<dyn crate::download::ModelInstaller>,
    );
    Env {
        dir,
        store,
        synth,
        installer,
        jobs,
        pid,
    }
}

impl Env {
    pub(super) fn doc(&self) -> Project {
        self.store.load(Some(&self.pid)).unwrap()
    }
    pub(super) fn submit(
        &self,
        cid: Option<&str>,
        changed: bool,
        allow_long: bool,
    ) -> Result<SubmitOutcome, StoreError> {
        self.jobs
            .submit(&self.pid, self.doc().revision, cid, changed, allow_long)
    }
    pub(super) fn run_all(&self) {
        assert_eq!(
            self.submit(None, false, false).unwrap(),
            SubmitOutcome::Started
        );
        self.wait_idle();
    }
    pub(super) fn wait_idle(&self) {
        let deadline = Instant::now() + Duration::from_secs(20);
        while self.jobs.busy() {
            assert!(Instant::now() < deadline, "the job did not finish");
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    fn wait_for(&self, what: &str, mut ok: impl FnMut(&Value) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            if self.jobs.jobs_json().last().is_some_and(&mut ok) {
                return;
            }
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    pub(super) fn statuses(&self) -> Vec<String> {
        self.doc().cards.iter().map(|c| c.status.clone()).collect()
    }
}

fn wav_len(path: &Path) -> u32 {
    hound::WavReader::open(path).unwrap().len()
}

#[test]
fn submit_runs_cards_in_order_and_saves_results() {
    let e = env(&["一つ目です。", "二つ目です。"], |s| s);
    e.run_all();
    assert_eq!(e.statuses(), ["完了", "完了"]);
    assert_eq!(e.synth.calls(), ["一つ目です。", "二つ目です。"]);
    let doc = e.doc();
    let dict = dictionary::load_lenient(e.dir.path()).unwrap();
    let voices = e.store.voices();
    for card in &doc.cards {
        let result = card.result.as_ref().expect("result");
        let path = Path::new(result["path"].as_str().unwrap());
        assert!(path.is_absolute() && path.exists(), "{path:?}");
        assert_eq!(
            result["signature"].as_str().unwrap(),
            signature(&doc, card, &dict, voices.first())
        );
        assert!(
            result_matches(&doc, card, &dict, voices.first()),
            "a fresh result is current"
        );
        assert_eq!(result["export"], json!({}));
        assert_eq!(
            result["record"].as_str(),
            path.parent().unwrap().file_name().unwrap().to_str()
        );
        let saved: Value = serde_json::from_slice(
            &std::fs::read(path.parent().unwrap().join("card.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(&saved, result, "card.json mirrors the result");
        let request = &result["request"];
        assert_eq!(request["card"]["id"], card.id.as_str());
        assert_eq!(request["voice"]["id"], VOICE);
        assert_eq!(
            (request["ordinal"].as_i64(), request["text"].as_str()),
            (
                Some(1 + doc.cards.iter().position(|c| c.id == card.id).unwrap() as i64),
                Some(card.text.as_str())
            )
        );
        assert_eq!(request["date"].as_str().unwrap().len(), 8);
    }
    let job = e.jobs.jobs_json().pop().unwrap();
    assert_eq!(
        (
            job["status"].as_str(),
            job["done"].as_i64(),
            job["total"].as_i64(),
            job["kind"].as_str()
        ),
        (Some("完了"), Some(2), Some(2), Some("generate"))
    );
}

#[test]
fn the_dictionary_is_applied_to_the_text_that_is_synthesized() {
    let e = env(&["Irodori です"], |s| s);
    dictionary::save(e.dir.path(), "Irodori", "いろどり").unwrap();
    e.run_all();
    assert_eq!(e.synth.calls(), ["いろどり です"]);
    let card = &e.doc().cards[0];
    assert_eq!(
        card.result.as_ref().unwrap()["request"]["dictionary"],
        json!({"Irodori": "いろどり"})
    );
}

#[test]
fn unchanged_cards_are_skipped_when_changed_only() {
    let e = env(&["一つ目です。", "二つ目です。"], |s| s);
    e.run_all();
    let err = e.submit(None, true, false).unwrap_err();
    assert_eq!(
        err.message(),
        "生成対象がありません。文章と生成状態を確認してください。"
    );
    let first = e.doc().cards[0].id.clone();
    ops::edit(
        &e.store,
        &e.pid,
        e.doc().revision,
        &json!({"cards": [{"id": first, "text": "書き換えた。"}]}),
    )
    .unwrap();
    assert_eq!(e.submit(None, true, false).unwrap(), SubmitOutcome::Started);
    e.wait_idle();
    assert_eq!(
        e.synth.calls(),
        ["一つ目です。", "二つ目です。", "書き換えた。"],
        "only the edited card runs again"
    );
}

#[test]
fn missing_voice_and_missing_model_are_reported_with_the_card_number() {
    let e = env(&["一つ目です。", "二つ目です。"], |s| s);
    let second = e.doc().cards[1].id.clone();
    ops::edit(
        &e.store,
        &e.pid,
        e.doc().revision,
        &json!({"cards": [{"id": second, "voice_id": ""}]}),
    )
    .unwrap();
    assert_eq!(
        e.submit(None, false, false).unwrap_err().message(),
        "2番のカードの話者を選んでください。"
    );
    assert_eq!(e.statuses(), ["未生成", "未生成"], "nothing was queued");

    // 未取得のモデルは誤りではなく、取得の確認を求める（ジョブは作らず、カードも変えない）。
    let e = env(&["一つ目です。"], |s| s.missing_models(&[MF]));
    assert_eq!(
        e.submit(None, false, false).unwrap(),
        SubmitOutcome::NeedsModelDownload { model: MF.into() }
    );
    assert_eq!(e.statuses(), ["未生成"]);
    assert!(e.jobs.jobs_json().is_empty());
}

#[test]
fn blank_cards_are_ignored_and_an_all_blank_script_has_nothing_to_do() {
    let e = env(&["  ", ""], |s| s);
    assert_eq!(
        e.submit(None, false, false).unwrap_err().message(),
        "生成対象がありません。文章と生成状態を確認してください。"
    );
}

#[test]
fn speed_other_than_one_is_refused_for_now() {
    let e = env(&["一つ目です。"], |s| s);
    let id = e.doc().cards[0].id.clone();
    ops::edit(
        &e.store,
        &e.pid,
        e.doc().revision,
        &json!({"cards": [{"id": id, "speed": 1.25}]}),
    )
    .unwrap();
    assert_eq!(
        e.submit(None, false, false).unwrap_err().message(),
        "話速の変更はこの版ではまだ使えません。話速を1.0倍にしてください。"
    );
    assert_eq!(e.statuses(), ["未生成"]);
}

#[test]
fn unsupported_generation_settings_are_refused_at_submit() {
    let e = env(&["一つ目です。"], |s| s);
    ops::edit(
        &e.store,
        &e.pid,
        e.doc().revision,
        &json!({"generation": {"model_precision": "fp16"}}),
    )
    .unwrap();
    assert_eq!(
        e.submit(None, false, false).unwrap_err().message(),
        "この版では fp32 か bf16 を選んでください。"
    );
    assert_eq!(e.statuses(), ["未生成"]);
}

#[test]
fn stale_revision_conflicts_and_changes_nothing() {
    let e = env(&["一つ目です。"], |s| s);
    let err = e.jobs.submit(&e.pid, 999, None, false, false).unwrap_err();
    assert_eq!(
        (err.status(), err.message()),
        (409, "台本が更新されています。読み直してください。")
    );
    assert_eq!(e.statuses(), ["未生成"]);
}

#[test]
fn long_cards_need_confirmation_then_run() {
    let long = "あ。".repeat(100); // 推定 25 秒を超え、180 文字も超える
    let e = env(&["短い文です。", &long], |s| s);
    let outcome = e.submit(None, false, false).unwrap();
    assert_eq!(
        outcome,
        SubmitOutcome::NeedsLongConfirmation {
            long_cards: vec![2]
        }
    );
    assert_eq!(
        e.statuses(),
        ["未生成", "未生成"],
        "nothing is queued before the user confirms"
    );
    assert!(!e.jobs.busy());
    assert_eq!(e.submit(None, false, true).unwrap(), SubmitOutcome::Started);
    e.wait_idle();
    assert_eq!(e.statuses(), ["完了", "完了"]);
}

#[test]
fn second_submit_while_running_is_refused() {
    let e = env(&["一つ目です。"], |s| {
        s.with_steps(100).with_step_delay(Duration::from_millis(20))
    });
    assert_eq!(
        e.submit(None, false, false).unwrap(),
        SubmitOutcome::Started
    );
    let err = e.submit(None, false, false).unwrap_err();
    assert_eq!(
        (err.status(), err.message()),
        (400, "現在の生成が終わるか、停止してから実行してください。")
    );
    e.jobs.stop();
    e.wait_idle();
}

#[test]
fn stop_cancels_within_a_step_and_marks_cards_stopped() {
    let e = env(
        &["一つ目です。", "二つ目です。", "三つ目です。"],
        |s| s.with_steps(500).with_step_delay(Duration::from_millis(10)),
    );
    assert_eq!(
        e.submit(None, false, false).unwrap(),
        SubmitOutcome::Started
    );
    e.wait_for("the first step", |j| j["step"].as_u64().unwrap_or(0) >= 2);
    let stopped_at = Instant::now();
    e.jobs.stop();
    e.wait_idle();
    assert!(
        stopped_at.elapsed() < Duration::from_secs(2),
        "stopped in about one step, not after the whole 5 s card"
    );
    assert_eq!(e.statuses(), ["停止", "停止", "停止"]);
    assert!(
        e.doc().cards.iter().all(|c| c.result.is_none()),
        "a cancelled card keeps no result"
    );
    assert!(
        !e.dir.path().join("outputs/internal").exists(),
        "no partial WAV was written"
    );
    let job = e.jobs.jobs_json().pop().unwrap();
    assert_eq!(job["status"], "停止");
    assert_eq!(
        e.synth.calls(),
        ["一つ目です。"],
        "later cards never started"
    );
}

#[test]
fn stop_with_nothing_running_is_harmless() {
    let e = env(&["一つ目です。"], |s| s);
    e.jobs.stop();
    e.run_all();
    assert_eq!(e.statuses(), ["完了"]);
}

#[test]
fn failure_marks_only_that_card_and_the_job_continues() {
    let e = env(
        &["一つ目です。", "失敗する", "三つ目です。"],
        |s| {
            s.fail_on(
                "失敗する",
                SynthError::Invalid("文章を確認してください。".into()),
            )
        },
    );
    e.run_all();
    assert_eq!(e.statuses(), ["完了", "失敗", "完了"]);
    let doc = e.doc();
    assert_eq!(doc.cards[1].error, "文章を確認してください。");
    assert!(doc.cards[1].result.is_none());
    let job = e.jobs.jobs_json().pop().unwrap();
    assert_eq!(
        (
            job["status"].as_str(),
            job["error"].as_str(),
            job["done"].as_i64()
        ),
        (
            Some("完了（一部失敗）"),
            Some("文章を確認してください。"),
            Some(3)
        )
    );
}

#[test]
fn an_unexpected_failure_shows_a_generic_message_and_keeps_the_detail_out_of_the_card() {
    let e = env(
        &["一つ目です。", "壊れる", "三つ目です。"],
        |s| {
            s.fail_on(
                "壊れる",
                SynthError::Failed("disk full at C:\\secret".into()),
            )
        },
    );
    e.run_all();
    assert_eq!(
        e.statuses(),
        ["完了", "失敗", "完了"],
        "the other cards still run"
    );
    assert_eq!(e.doc().cards[1].error, GENERIC);
    assert!(!e.doc().cards[1].error.contains("secret"));
}

#[test]
fn panic_in_the_synthesizer_does_not_wedge_the_job() {
    let e = env(
        &["一つ目です。", "爆発", "三つ目です。"],
        |s| s.panic_on("爆発"),
    );
    e.run_all();
    assert_eq!(e.statuses(), ["完了", "失敗", "完了"]);
    assert_eq!(e.doc().cards[1].error, GENERIC);
    // ワーカーは生きていて、次のジョブも動く
    let id = e.doc().cards[1].id.clone();
    ops::edit(
        &e.store,
        &e.pid,
        e.doc().revision,
        &json!({"cards": [{"id": id, "text": "直した文章"}]}),
    )
    .unwrap();
    assert_eq!(
        e.submit(Some(&id), false, false).unwrap(),
        SubmitOutcome::Started
    );
    e.wait_idle();
    assert_eq!(e.doc().cards[1].status, "完了");
}

#[test]
fn editing_a_card_while_it_generates_keeps_the_new_text() {
    let e = env(&["元の文章です。"], |s| {
        s.with_steps(20).with_step_delay(Duration::from_millis(15))
    });
    assert_eq!(
        e.submit(None, false, false).unwrap(),
        SubmitOutcome::Started
    );
    e.wait_for("generation to start", |j| {
        j["step"].as_u64().unwrap_or(0) >= 1
    });
    let id = e.doc().cards[0].id.clone();
    ops::edit(
        &e.store,
        &e.pid,
        e.doc().revision,
        &json!({"cards": [{"id": id, "text": "編集後の文章です。"}]}),
    )
    .unwrap();
    e.wait_idle();
    let doc = e.doc();
    assert_eq!(
        doc.cards[0].text, "編集後の文章です。",
        "the new input is never overwritten"
    );
    assert_eq!(doc.cards[0].status, "完了");
    let dict = dictionary::load_lenient(e.dir.path()).unwrap();
    assert!(
        !result_matches(&doc, &doc.cards[0], &dict, e.store.voices().first()),
        "the old result no longer matches the edited text"
    );
}

#[test]
fn silence_is_appended_to_the_saved_wav() {
    let e = env(&["一つ目です。"], |s| s);
    let id = e.doc().cards[0].id.clone();
    ops::edit(
        &e.store,
        &e.pid,
        e.doc().revision,
        &json!({"cards": [{"id": id, "silence": 0.5}]}),
    )
    .unwrap();
    e.run_all();
    let card = &e.doc().cards[0];
    let path = Path::new(card.result.as_ref().unwrap()["path"].as_str().unwrap());
    assert!(
        path.file_name()
            .unwrap()
            .to_string_lossy()
            .contains("_tail_"),
        "{path:?}"
    );
    assert_eq!(
        wav_len(path),
        4800 + 12000,
        "0.2 s of audio plus 0.5 s of silence at 24 kHz"
    );
    assert!(path.parent().unwrap().join("card.json").exists());
}

#[test]
fn unload_is_refused_while_busy_and_works_when_idle() {
    let e = env(&["一つ目です。"], |s| {
        s.with_steps(100).with_step_delay(Duration::from_millis(20))
    });
    assert_eq!(
        e.submit(None, false, false).unwrap(),
        SubmitOutcome::Started
    );
    e.wait_for("the synthesizer to run", |j| {
        j["step"].as_u64().unwrap_or(0) >= 1
    });
    assert_eq!(
        e.jobs.unload().unwrap_err().message(),
        "生成完了後に実行してください。"
    );
    assert!(e.synth.is_loaded());
    e.jobs.stop();
    e.wait_idle();
    e.jobs.unload().unwrap();
    assert!(!e.synth.is_loaded());
}

#[test]
fn jobs_json_has_progress_and_stopping() {
    let e = env(&["一つ目です。"], |s| {
        s.with_steps(100).with_step_delay(Duration::from_millis(20))
    });
    e.submit(None, false, false).unwrap();
    e.wait_for("progress", |j| j["step"].as_u64().unwrap_or(0) >= 1);
    let job = e.jobs.jobs_json().pop().unwrap();
    assert_eq!(
        (
            job["project"].as_str(),
            job["kind"].as_str(),
            job["total"].as_i64()
        ),
        (Some(e.pid.as_str()), Some("generate"), Some(1))
    );
    assert_eq!(
        (
            job["steps"].as_i64(),
            job["stopping"].as_bool(),
            job["status"].as_str()
        ),
        (Some(4), Some(false), Some("生成中"))
    );
    assert!(job.get("cancel").is_none() && job.get("requests").is_none());
    e.jobs.stop();
    e.wait_for("stopping", |j| {
        j["stopping"] == true || j["status"] == "停止"
    });
    e.wait_idle();
}

#[test]
fn active_card_ids_lists_only_pending_requests() {
    let e = env(
        &["一つ目です。", "二つ目です。", "三つ目です。"],
        |s| s.with_steps(30).with_step_delay(Duration::from_millis(10)),
    );
    let ids: Vec<String> = e.doc().cards.iter().map(|c| c.id.clone()).collect();
    assert!(e.jobs.active_card_ids(&e.pid).is_empty());
    e.submit(None, false, false).unwrap();
    e.wait_for("the first card to finish", |j| {
        j["done"].as_i64().unwrap_or(0) >= 1
    });
    let active = e.jobs.active_card_ids(&e.pid);
    assert!(
        !active.contains(&ids[0]),
        "a finished card is no longer active"
    );
    assert!(
        active.contains(&ids[2]),
        "a queued card is active: {active:?}"
    );
    assert!(e.jobs.active_card_ids("another-project").is_empty());
    e.jobs.stop();
    e.wait_idle();
    assert_eq!(e.jobs.active_card_ids(&e.pid), HashSet::new());
}

#[test]
fn old_finished_jobs_are_not_kept_forever() {
    let e = env(&["一つ目です。"], |s| s.with_steps(1));
    for _ in 0..60 {
        e.run_all();
    }
    assert!(
        e.jobs.jobs_json().len() <= 50,
        "{}",
        e.jobs.jobs_json().len()
    );
}

/// プロジェクトのファイルを読み取り専用にして、保存を失敗させる（テストの後始末で戻す）。
fn set_projects_readonly(root: &Path, readonly: bool) {
    for entry in std::fs::read_dir(root.join("projects")).unwrap().flatten() {
        // 保存の途中の一時ファイルは、列挙のあとで rename されて消えることがある（その場合は読み飛ばす）。
        let Ok(meta) = entry.metadata() else { continue };
        let mut perms = meta.permissions();
        perms.set_readonly(readonly);
        let _ = std::fs::set_permissions(entry.path(), perms);
    }
}

#[test]
fn a_failed_result_save_is_reported_as_a_job_error_not_as_success() {
    let e = env(&["一つ目です。"], |s| {
        s.on_success(|root| set_projects_readonly(root, true))
    });
    assert_eq!(
        e.submit(None, false, false).unwrap(),
        SubmitOutcome::Started
    );
    e.wait_idle();
    set_projects_readonly(e.dir.path(), false);
    let job = e.jobs.jobs_json().pop().unwrap();
    assert_eq!(
        job["status"], "完了（一部失敗）",
        "the job must not claim full success: {job}"
    );
    assert_eq!(
        job["error"],
        "結果を保存できませんでした。保存先の空き容量とアクセス権を確認してください。"
    );
    assert_eq!(e.jobs.jobs_json().len(), 1);
}

#[test]
fn a_job_that_stops_making_progress_is_reported_as_stalled_once() {
    let e = env(&["一つ目です。"], |s| {
        s.with_steps(100).with_step_delay(Duration::from_secs(3))
    });
    assert_eq!(
        e.submit(None, false, false).unwrap(),
        SubmitOutcome::Started
    );
    e.wait_for("the first step", |j| j["step"].as_u64().unwrap_or(0) >= 1);
    let t0 = Instant::now();
    let threshold = Duration::from_secs(120);
    assert!(
        e.jobs.check_stalls(t0, threshold).is_empty(),
        "the first look only records the baseline"
    );
    assert!(
        e.jobs
            .check_stalls(t0 + Duration::from_secs(60), threshold)
            .is_empty(),
        "not stalled long enough yet"
    );
    let found = e
        .jobs
        .check_stalls(t0 + Duration::from_secs(200), threshold);
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(
        found[0].contains("sampling") && found[0].contains("step 1/4"),
        "the message names the stage and step: {}",
        found[0]
    );
    assert!(
        e.jobs
            .check_stalls(t0 + Duration::from_secs(400), threshold)
            .is_empty(),
        "reported once per stall"
    );
    e.jobs.stop();
    e.wait_idle();
    assert!(
        e.jobs
            .check_stalls(t0 + Duration::from_secs(600), threshold)
            .is_empty(),
        "finished jobs are never reported"
    );
}

#[test]
fn jobs_json_names_the_current_stage() {
    let e = env(&["一つ目です。"], |s| {
        s.with_steps(100).with_step_delay(Duration::from_millis(20))
    });
    e.submit(None, false, false).unwrap();
    e.wait_for("sampling", |j| j["stage"] == "sampling");
    e.jobs.stop();
    e.wait_idle();
}
