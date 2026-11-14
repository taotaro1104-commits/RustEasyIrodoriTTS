//! 声の試作ジョブのテスト（擬似エンジン `FakeSynth`）。

use std::sync::Arc;
use std::time::{Duration, Instant};

use irodori_store::assets::Assets;
use irodori_store::dictionary::Dictionary;
use irodori_store::splitter::split_text;
use irodori_store::Store;
use serde_json::{json, Map, Value};

use super::{DesignSubmit, JobManager, REGISTRATION_SAMPLE_TEXT};
use crate::fake::{FakeInstaller, FakeSynth};
use crate::synth::{SynthError, Synthesizer};

const MF: &str = "Aratako/Irodori-TTS-v4.1-Small-MF";
const GENERIC: &str = "処理に失敗しました。logs/editor.logを確認してください。";

struct Env {
    dir: tempfile::TempDir,
    store: Arc<Store>,
    assets: Arc<Assets>,
    synth: Arc<FakeSynth>,
    jobs: JobManager,
}

fn env(build: impl FnOnce(FakeSynth) -> FakeSynth) -> Env {
    let dir = tempfile::tempdir().unwrap();
    for sub in ["projects", "voices", "config", "temp", "outputs"] {
        std::fs::create_dir_all(dir.path().join(sub)).unwrap();
    }
    let store = Arc::new(Store::open(dir.path()).unwrap());
    let assets = Arc::new(Assets::new());
    let synth = Arc::new(build(FakeSynth::new(dir.path())));
    let jobs = JobManager::new(
        Arc::clone(&store),
        synth.clone() as Arc<dyn Synthesizer>,
        Arc::clone(&assets),
        Arc::new(FakeInstaller::new(|_| {})) as Arc<dyn crate::download::ModelInstaller>,
    );
    Env {
        dir,
        store,
        assets,
        synth,
        jobs,
    }
}

fn values(v: Value) -> Map<String, Value> {
    match v {
        Value::Object(m) => m,
        _ => panic!("object expected"),
    }
}

fn basic() -> Map<String, Value> {
    values(
        json!({"model": MF, "mode": "design", "caption": "明るい声", "text": "こんにちは。", "candidates": 1}),
    )
}

fn message(e: irodori_store::StoreError) -> String {
    e.message().to_string()
}

fn job_json(env: &Env, id: &str) -> Value {
    env.jobs
        .jobs_json()
        .into_iter()
        .find(|j| j["id"] == id)
        .expect("job listed")
}

fn wait_done(env: &Env, id: &str) -> Value {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let job = job_json(env, id);
        if !["待機中", "生成中"].contains(&job["status"].as_str().unwrap()) {
            return job;
        }
        assert!(
            Instant::now() < deadline,
            "design job did not finish: {job}"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// 参照用の音声を登録簿に載せて、鍵を返す。
fn reference_key(env: &Env, name: &str) -> String {
    let path = env.dir.path().join("temp").join(name);
    std::fs::write(&path, b"RIFF").unwrap();
    env.assets.register(&path).unwrap()
}

#[test]
fn design_job_runs_and_registers_candidate_assets() {
    let env = env(|s| s);
    let mut v = basic();
    v.insert("candidates".into(), json!(3));
    let id = env.jobs.submit_design(&v, false).unwrap().id();
    let job = wait_done(&env, &id);
    assert_eq!(job["status"], "完了", "{job}");
    assert_eq!(
        (
            job["kind"].as_str(),
            job["done"].as_u64(),
            job["total"].as_u64()
        ),
        (Some("design"), Some(1), Some(1))
    );
    assert!(job["project"].is_null());
    let keys = job["assets"].as_array().unwrap();
    assert_eq!(keys.len(), 3);
    for key in keys {
        assert!(env.assets.source(key.as_str().unwrap()).unwrap().is_file());
    }
}

#[test]
fn registration_uses_the_fixed_sample_text() {
    let env = env(|s| s);
    let mut v = basic();
    v.remove("text");
    let id = env.jobs.submit_design(&v, true).unwrap().id();
    assert_eq!(wait_done(&env, &id)["status"], "完了");
    assert_eq!(
        env.synth.calls(),
        vec![REGISTRATION_SAMPLE_TEXT.to_string()]
    );
    let (_, record) = env.jobs.design_result(&id).unwrap();
    assert_eq!(record["text"], REGISTRATION_SAMPLE_TEXT);
}

#[test]
fn modes_decide_what_the_engine_receives() {
    let env = env(|s| s);
    let key = reference_key(&env, "ref.wav");
    for (mode, caption, references) in [
        ("design", "明るい声", 0),
        ("clone", "", 1),
        ("both", "明るい声", 1),
        ("auto", "", 0),
    ] {
        let mut v = basic();
        v.insert("mode".into(), json!(mode));
        v.insert("references".into(), json!([key]));
        let id = env.jobs.submit_design(&v, false).unwrap().id();
        assert_eq!(wait_done(&env, &id)["status"], "完了", "{mode}");
        let seen = env.synth.last_job().unwrap();
        assert_eq!(
            (seen.caption.as_str(), seen.references.len()),
            (caption, references),
            "{mode}"
        );
    }
}

#[test]
fn clone_without_references_is_rejected() {
    let env = env(|s| s);
    for mode in ["clone", "both"] {
        let mut v = basic();
        v.insert("mode".into(), json!(mode));
        assert_eq!(
            message(env.jobs.submit_design(&v, false).unwrap_err()),
            "お手本の音声を追加してください。",
            "{mode}"
        );
        v.insert("references".into(), json!([]));
        assert!(env.jobs.submit_design(&v, false).is_err());
    }
    assert!(
        env.jobs.jobs_json().is_empty(),
        "rejected requests create no job"
    );
}

#[test]
fn unknown_reference_key_is_rejected() {
    let env = env(|s| s);
    let mut v = basic();
    v.insert("mode".into(), json!("clone"));
    v.insert("references".into(), json!(["0".repeat(64)]));
    assert_eq!(
        message(env.jobs.submit_design(&v, false).unwrap_err()),
        "音声が見つかりません。もう一度追加してください。"
    );
    v.insert("references".into(), json!([5]));
    assert!(env.jobs.submit_design(&v, false).is_err());
}

#[test]
fn two_references_are_rejected() {
    let env = env(|s| s);
    let (a, b) = (reference_key(&env, "a.wav"), reference_key(&env, "b.wav"));
    let mut v = basic();
    v.insert("mode".into(), json!("clone"));
    v.insert("references".into(), json!([a, b]));
    assert_eq!(
        message(env.jobs.submit_design(&v, false).unwrap_err()),
        "お手本の音声は1つだけ使えます。"
    );
}

#[test]
fn bad_mode_is_rejected() {
    let env = env(|s| s);
    for mode in [json!("remix"), json!(3), json!(null)] {
        let mut v = basic();
        v.insert("mode".into(), mode);
        assert_eq!(
            message(env.jobs.submit_design(&v, false).unwrap_err()),
            "声の作り方が不正です。"
        );
    }
}

#[test]
fn long_text_is_rejected_with_the_25_second_message() {
    let env = env(|s| s);
    let text = "これは試作には長すぎる文章です。".repeat(40);
    assert!(
        split_text(&text, &Dictionary::new(), 1.0).unwrap().len() > 1,
        "the test text must really be long"
    );
    let mut v = basic();
    v.insert("text".into(), json!(text));
    assert_eq!(
        message(env.jobs.submit_design(&v, false).unwrap_err()),
        "声の試作には推定25秒以内の短い文章を入力してください。"
    );
}

#[test]
fn empty_or_missing_text_is_rejected() {
    let env = env(|s| s);
    for text in [json!(""), json!("  \n"), json!(null), json!(5)] {
        let mut v = basic();
        v.insert("text".into(), text);
        assert_eq!(
            message(env.jobs.submit_design(&v, false).unwrap_err()),
            "声の試作には推定25秒以内の短い文章を入力してください。"
        );
    }
    let mut v = basic();
    v.remove("text");
    assert!(env.jobs.submit_design(&v, false).is_err());
}

#[test]
fn bad_generation_settings_are_rejected() {
    let env = env(|s| s);
    let mut v = basic();
    v.insert("generation".into(), json!({"steps": 0}));
    assert!(message(env.jobs.submit_design(&v, false).unwrap_err()).contains("計算回数"));
    v.insert("generation".into(), json!("nope"));
    assert_eq!(
        message(env.jobs.submit_design(&v, false).unwrap_err()),
        "生成設定が不正です。"
    );
    // 画面は空の steps / seed を送る（空は「既定」）。
    v.insert("generation".into(), json!({"steps": "", "seed": null}));
    assert!(env.jobs.submit_design(&v, false).is_ok());
}

#[test]
fn candidates_must_be_one_to_eight() {
    let env = env(|s| s);
    for bad in [json!(0), json!(9), json!("x"), json!(null), json!(-1)] {
        let mut v = basic();
        v.insert("candidates".into(), bad.clone());
        assert_eq!(
            message(env.jobs.submit_design(&v, false).unwrap_err()),
            "候補数は1～8で指定してください。",
            "{bad}"
        );
    }
    let mut v = basic();
    v.insert("candidates".into(), json!("2"));
    let id = env.jobs.submit_design(&v, false).unwrap().id();
    assert_eq!(wait_done(&env, &id)["assets"].as_array().unwrap().len(), 2);
    v.insert("candidates".into(), json!(2.7));
    let id = env.jobs.submit_design(&v, false).unwrap().id();
    assert_eq!(
        wait_done(&env, &id)["assets"].as_array().unwrap().len(),
        2,
        "fractions are cut"
    );
}

#[test]
fn unknown_or_missing_models_are_rejected() {
    let env = env(|s| s.missing_models(&[MF]));
    // 未取得のモデルは誤りではなく、取得の確認を求める（ジョブは作らない）。
    assert_eq!(
        env.jobs.submit_design(&basic(), false).unwrap(),
        DesignSubmit::NeedsModelDownload(MF.into())
    );
    let mut v = basic();
    v.insert("model".into(), json!("evil/model"));
    assert_eq!(
        message(env.jobs.submit_design(&v, false).unwrap_err()),
        "使用するモデルを選んでください。"
    );
    assert!(env.jobs.jobs_json().is_empty());
}

fn slow() -> impl FnOnce(FakeSynth) -> FakeSynth {
    |s| s.with_steps(40).with_step_delay(Duration::from_millis(25))
}

#[test]
fn a_busy_manager_rejects_a_second_design_and_a_generation() {
    let env = env(slow());
    let first = env.jobs.submit_design(&basic(), false).unwrap().id();
    assert_eq!(
        message(env.jobs.submit_design(&basic(), false).unwrap_err()),
        "現在の生成が終わってから実行してください。"
    );
    let doc = env.store.load(None).unwrap();
    let e = env
        .jobs
        .submit(&doc.id, doc.revision, None, false, false)
        .unwrap_err();
    assert_eq!(
        e.message(),
        "現在の生成が終わるか、停止してから実行してください。"
    );
    assert_eq!(
        env.jobs.unload().unwrap_err().message(),
        "生成完了後に実行してください。"
    );
    env.jobs.stop();
    wait_done(&env, &first);
}

#[test]
fn stop_marks_a_design_job_stopped_with_no_assets() {
    let env = env(slow());
    let id = env.jobs.submit_design(&basic(), false).unwrap().id();
    std::thread::sleep(Duration::from_millis(60));
    env.jobs.stop();
    let job = wait_done(&env, &id);
    assert_eq!(job["status"], "停止", "{job}");
    assert!(job["assets"].as_array().unwrap().is_empty());
    assert!(env.jobs.design_result(&id).is_none());
}

#[test]
fn engine_failure_marks_the_job_failed_with_a_message() {
    let env = env(|s| {
        s.fail_on(
            "失敗する文。",
            SynthError::Invalid("モデルの上限を超えています。".into()),
        )
        .fail_on("想定外。", SynthError::Failed("secret detail".into()))
    });
    let mut v = basic();
    v.insert("text".into(), json!("失敗する文。"));
    let id = env.jobs.submit_design(&v, false).unwrap().id();
    let job = wait_done(&env, &id);
    assert_eq!(
        (job["status"].as_str(), job["error"].as_str()),
        (Some("失敗"), Some("モデルの上限を超えています。"))
    );
    v.insert("text".into(), json!("想定外。"));
    let id = env.jobs.submit_design(&v, false).unwrap().id();
    let job = wait_done(&env, &id);
    assert_eq!(
        (job["status"].as_str(), job["error"].as_str()),
        (Some("失敗"), Some(GENERIC))
    );
    assert!(!job.to_string().contains("secret"));
}

#[test]
fn engine_panic_does_not_kill_the_worker() {
    let env = env(|s| s.panic_on("パニック。"));
    let mut v = basic();
    v.insert("text".into(), json!("パニック。"));
    let id = env.jobs.submit_design(&v, false).unwrap().id();
    let job = wait_done(&env, &id);
    assert_eq!(
        (job["status"].as_str(), job["error"].as_str()),
        (Some("失敗"), Some(GENERIC))
    );
    let id = env.jobs.submit_design(&basic(), false).unwrap().id();
    assert_eq!(
        wait_done(&env, &id)["status"],
        "完了",
        "the worker survived"
    );
}

#[test]
fn a_design_job_does_not_touch_cards_or_hold_card_ids() {
    let env = env(slow());
    let doc = env.store.load(None).unwrap();
    let before = serde_json::to_value(&doc).unwrap();
    let id = env.jobs.submit_design(&basic(), false).unwrap().id();
    assert!(env.jobs.active_card_ids(&doc.id).is_empty());
    env.jobs.stop();
    wait_done(&env, &id);
    assert_eq!(
        serde_json::to_value(env.store.load(Some(&doc.id)).unwrap()).unwrap(),
        before
    );
}

#[test]
fn design_result_is_none_until_done_and_carries_the_record() {
    let env = env(slow());
    let id = env.jobs.submit_design(&basic(), false).unwrap().id();
    assert!(env.jobs.design_result(&id).is_none(), "not finished yet");
    assert!(env.jobs.design_result("nope").is_none());
    env.jobs.stop();
    wait_done(&env, &id);
    let env2 = self::env(|s| s);
    let id = env2.jobs.submit_design(&basic(), false).unwrap().id();
    wait_done(&env2, &id);
    let (assets, record) = env2.jobs.design_result(&id).unwrap();
    assert_eq!(assets.len(), 1);
    for key in [
        "id",
        "model",
        "text",
        "caption",
        "references",
        "generation",
        "seed",
        "paths",
        "seconds",
        "messages",
    ] {
        assert!(record.get(key).is_some(), "record lacks {key}: {record}");
    }
    assert_eq!(
        (record["text"].as_str(), record["model"].as_str()),
        (Some("こんにちは。"), Some(MF))
    );
}

#[test]
fn a_generation_job_id_is_not_a_design_result() {
    let env = env(|s| s);
    let doc = env.store.load(None).unwrap();
    env.store
        .internal_update(&doc.id, &mut |d| {
            d.cards[0].text = "こんにちは。".into();
        })
        .unwrap();
    // 話者が無いので生成は断られる。ここでは、試作の ID でないものは None になることだけ見る。
    assert!(env.jobs.design_result("anything").is_none());
}

#[test]
fn jobs_json_has_the_design_shape() {
    let env = env(|s| s);
    let id = env.jobs.submit_design(&basic(), false).unwrap().id();
    let job = wait_done(&env, &id);
    for key in [
        "id", "project", "status", "done", "total", "error", "stopping", "kind", "step", "steps",
        "stage", "assets",
    ] {
        assert!(job.get(key).is_some(), "jobs[] lacks {key}: {job}");
    }
    assert_eq!(job["kind"], "design");
    assert!(job["steps"].as_u64().unwrap() >= 1);
}
