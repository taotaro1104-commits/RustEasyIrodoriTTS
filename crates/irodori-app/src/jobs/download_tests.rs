//! モデルの取得ジョブのテスト（擬似の導入 `FakeInstaller` と擬似エンジン `FakeSynth`）。通信は起きない。

use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};

use super::tests::{env, Env, GENERIC, MF};
use super::{DesignSubmit, SubmitOutcome};
use crate::download::DownloadError;
use crate::synth::Synthesizer;

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
            "download job did not finish: {job}"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn message(e: irodori_store::StoreError) -> String {
    e.message().to_string()
}

/// モデルが未取得の環境。
fn missing() -> Env {
    env(&["一つ目です。"], |s| s.missing_models(&[MF]))
}

fn slow(e: &Env) {
    e.installer.set_steps(200);
    e.installer.set_delay(Duration::from_millis(10));
}

#[test]
fn a_download_job_runs_reports_bytes_and_makes_the_model_available() {
    let e = missing();
    assert!(!e.synth.model_available(MF));
    let id = e.jobs.submit_download(MF, true).unwrap();
    let job = wait_done(&e, &id);
    assert_eq!(job["status"], "完了", "{job}");
    assert_eq!(
        (job["bytes_done"].as_u64(), job["bytes_total"].as_u64()),
        (Some(4000), Some(4000))
    );
    assert_eq!(
        (job["kind"].as_str(), job["model"].as_str()),
        (Some("download"), Some(MF))
    );
    assert!(job["project"].is_null());
    assert!(
        e.synth.model_available(MF),
        "the installer finished and the model is usable"
    );
    assert_eq!(e.installer.installs(), vec![MF.to_string()]);
}

#[test]
fn bytes_advance_while_the_download_runs() {
    let e = missing();
    slow(&e);
    let id = e.jobs.submit_download(MF, true).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let job = job_json(&e, &id);
        let (done, total) = (
            job["bytes_done"].as_u64().unwrap(),
            job["bytes_total"].as_u64().unwrap(),
        );
        if done > 0 && done < total {
            assert_eq!(job["status"], "生成中");
            break;
        }
        assert!(Instant::now() < deadline, "no in-between progress: {job}");
        std::thread::sleep(Duration::from_millis(5));
    }
    e.jobs.cancel_downloads();
    wait_done(&e, &id);
}

#[test]
fn an_unconfirmed_download_is_refused_and_creates_no_job() {
    let e = missing();
    assert_eq!(
        message(e.jobs.submit_download(MF, false).unwrap_err()),
        "取得の確認が必要です。"
    );
    assert!(e.jobs.jobs_json().is_empty());
    assert!(e.installer.installs().is_empty(), "nothing was downloaded");
}

#[test]
fn an_unknown_model_is_refused_even_when_confirmed() {
    let e = missing();
    for model in [
        "evil/model",
        "",
        "../x",
        "Aratako/Semantic-DACVAE-Japanese-32dim",
    ] {
        assert_eq!(
            message(e.jobs.submit_download(model, true).unwrap_err()),
            "使用するモデルを選んでください。",
            "{model:?}"
        );
    }
    assert!(e.installer.installs().is_empty());
}

#[test]
fn an_installed_model_is_not_downloaded_again() {
    let e = env(&["一つ目です。"], |s| s);
    assert_eq!(
        message(e.jobs.submit_download(MF, true).unwrap_err()),
        "このモデルはすでに取得済みです。"
    );
    assert!(e.installer.installs().is_empty());
}

#[test]
fn a_running_download_blocks_another_download_a_generation_and_a_design() {
    let e = missing();
    slow(&e);
    let id = e.jobs.submit_download(MF, true).unwrap();
    assert_eq!(
        message(e.jobs.submit_download(MF, true).unwrap_err()),
        "現在の処理が終わってから実行してください。"
    );
    assert_eq!(
        message(e.submit(None, false, false).unwrap_err()),
        "現在の生成が終わるか、停止してから実行してください。"
    );
    let design: Map<String, Value> = match json!({"model": MF, "mode": "design", "caption": "x", "text": "こんにちは。"})
    {
        Value::Object(m) => m,
        _ => unreachable!(),
    };
    assert_eq!(
        message(e.jobs.submit_design(&design, false).unwrap_err()),
        "現在の生成が終わってから実行してください。"
    );
    assert_eq!(
        message(e.jobs.unload().unwrap_err()),
        "生成完了後に実行してください。"
    );
    e.jobs.cancel_downloads();
    wait_done(&e, &id);
}

#[test]
fn cancel_downloads_stops_a_download_and_keeps_the_job_history() {
    let e = missing();
    slow(&e);
    let id = e.jobs.submit_download(MF, true).unwrap();
    std::thread::sleep(Duration::from_millis(40));
    e.jobs.cancel_downloads();
    let job = wait_done(&e, &id);
    assert_eq!(job["status"], "停止", "{job}");
    assert!(!e.synth.model_available(MF));
    // 取り消したあとは、もう一度取得できる。
    e.installer.set_steps(2);
    e.installer.set_delay(Duration::ZERO);
    let again = e.jobs.submit_download(MF, true).unwrap();
    assert_eq!(wait_done(&e, &again)["status"], "完了");
}

#[test]
fn cancel_downloads_does_not_touch_a_generation() {
    let e = env(&["一つ目です。"], |s| {
        s.with_steps(40).with_step_delay(Duration::from_millis(10))
    });
    assert_eq!(
        e.submit(None, false, false).unwrap(),
        SubmitOutcome::Started
    );
    e.jobs.cancel_downloads();
    e.wait_idle();
    assert_eq!(e.statuses(), ["完了"], "the generation finished normally");
}

#[test]
fn stop_also_cancels_a_download() {
    let e = missing();
    slow(&e);
    let id = e.jobs.submit_download(MF, true).unwrap();
    std::thread::sleep(Duration::from_millis(40));
    e.jobs.stop();
    assert_eq!(wait_done(&e, &id)["status"], "停止");
}

#[test]
fn a_failed_install_marks_the_job_failed_with_the_japanese_message() {
    let e = missing();
    for error in [
        DownloadError::Offline,
        DownloadError::DiskSpace {
            needed: 3_300_000_000,
            available: 1_000_000_000,
        },
        DownloadError::Verification,
        DownloadError::Http(503),
    ] {
        e.installer.fail_with(Some(error.clone()));
        let id = e.jobs.submit_download(MF, true).unwrap();
        let job = wait_done(&e, &id);
        assert_eq!(
            (job["status"].as_str(), job["error"].as_str()),
            (Some("失敗"), Some(error.message().as_str())),
            "{error:?}"
        );
        assert!(
            !e.synth.model_available(MF),
            "a failed install does not make the model usable"
        );
    }
    e.installer
        .fail_with(Some(DownloadError::Protocol("secret detail".into())));
    let id = e.jobs.submit_download(MF, true).unwrap();
    let job = wait_done(&e, &id);
    assert_eq!(job["error"], GENERIC);
    assert!(!job.to_string().contains("secret"));
}

#[test]
fn an_installer_panic_does_not_kill_the_worker() {
    let e = missing();
    e.installer.panic_next();
    let id = e.jobs.submit_download(MF, true).unwrap();
    let job = wait_done(&e, &id);
    assert_eq!(
        (job["status"].as_str(), job["error"].as_str()),
        (Some("失敗"), Some(GENERIC))
    );
    let again = e.jobs.submit_download(MF, true).unwrap();
    assert_eq!(
        wait_done(&e, &again)["status"],
        "完了",
        "the worker survived"
    );
}

#[test]
fn generate_asks_for_a_download_when_the_model_is_missing() {
    let e = missing();
    assert_eq!(
        e.submit(None, false, false).unwrap(),
        SubmitOutcome::NeedsModelDownload { model: MF.into() }
    );
    assert!(e.jobs.jobs_json().is_empty(), "no job is created");
    assert!(
        e.installer.installs().is_empty(),
        "and nothing is downloaded"
    );
    assert_eq!(e.statuses(), ["未生成"], "the card is untouched");
}

#[test]
fn design_asks_for_a_download_when_the_model_is_missing() {
    let e = missing();
    let values: Map<String, Value> = match json!({"model": MF, "mode": "design", "caption": "x", "text": "こんにちは。"})
    {
        Value::Object(m) => m,
        _ => unreachable!(),
    };
    assert_eq!(
        e.jobs.submit_design(&values, false).unwrap(),
        DesignSubmit::NeedsModelDownload(MF.into())
    );
    assert!(e.jobs.jobs_json().is_empty());
    assert!(e.installer.installs().is_empty());
}

#[test]
fn after_a_download_the_generation_runs() {
    let e = missing();
    let id = e.jobs.submit_download(MF, true).unwrap();
    assert_eq!(wait_done(&e, &id)["status"], "完了");
    e.run_all();
    assert_eq!(e.statuses(), ["完了"]);
}

#[test]
fn jobs_json_has_the_download_shape() {
    let e = missing();
    let id = e.jobs.submit_download(MF, true).unwrap();
    let job = wait_done(&e, &id);
    for key in [
        "id",
        "project",
        "status",
        "done",
        "total",
        "error",
        "stopping",
        "kind",
        "step",
        "steps",
        "stage",
        "assets",
        "model",
        "bytes_done",
        "bytes_total",
    ] {
        assert!(job.get(key).is_some(), "jobs[] lacks {key}: {job}");
    }
    // 生成・試作のジョブでも、同じ項目が（0 や空で）ある。
    let e = env(&["一つ目です。"], |s| s);
    e.run_all();
    let generated = e.jobs.jobs_json().pop().unwrap();
    assert_eq!(
        (
            generated["bytes_done"].as_u64(),
            generated["model"].as_str()
        ),
        (Some(0), Some(""))
    );
}

#[test]
fn a_running_download_is_not_reported_as_a_stall() {
    let e = missing();
    slow(&e);
    let id = e.jobs.submit_download(MF, true).unwrap();
    std::thread::sleep(Duration::from_millis(40));
    // ステップも段階も動かない種類のジョブなので、止まりの検出の対象にしない。
    let found = e
        .jobs
        .check_stalls(Instant::now() + Duration::from_secs(3600), Duration::ZERO);
    assert!(found.is_empty(), "{found:?}");
    e.jobs.cancel_downloads();
    wait_done(&e, &id);
}
