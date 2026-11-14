//! モデルの取得の HTTP 経路（確認・ジョブ・キャンセル・状態）。擬似の導入を使い、通信は起きない。

mod common;

use std::time::Duration;

use axum::http::StatusCode;
use common::*;
use irodori_app::download::DownloadError;
use serde_json::{json, Value};

async fn state(f: &Fixture) -> Value {
    get(&f.app, "/api/state").await.json()
}

fn model_entry(s: &Value, id: &str) -> Value {
    s["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == id)
        .cloned()
        .unwrap_or_else(|| panic!("model {id} not listed"))
}

async fn wait_idle(f: &Fixture) {
    for _ in 0..3000 {
        let s = state(f).await;
        let busy = s["jobs"]
            .as_array()
            .unwrap()
            .iter()
            .any(|j| ["待機中", "生成中"].contains(&j["status"].as_str().unwrap()));
        if !busy {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("jobs did not finish");
}

fn missing() -> Fixture {
    fixture_with_voice(|s| s.missing_models(&[MF]))
}

fn slow(f: &Fixture) {
    f.installer.set_steps(300);
    f.installer.set_delay(Duration::from_millis(10));
}

async fn start_download(f: &Fixture) -> String {
    let r = post(
        &f.app,
        "/api/models/download",
        json!({"model": MF, "confirm": true}),
    )
    .await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.error());
    let body = r.json();
    assert_eq!(
        body.as_object().unwrap().len(),
        1,
        "only the job id: {body}"
    );
    body["job"].as_str().unwrap().to_string()
}

#[tokio::test]
async fn download_requires_confirm_true() {
    let f = missing();
    for body in [
        json!({"model": MF}),
        json!({"model": MF, "confirm": false}),
        json!({"model": MF, "confirm": "true"}),
        json!({"model": MF, "confirm": 1}),
        json!({"model": MF, "confirm": null}),
    ] {
        let r = post(&f.app, "/api/models/download", body.clone()).await;
        assert_eq!(
            (r.status, r.error()),
            (StatusCode::BAD_REQUEST, "取得の確認が必要です。".into()),
            "{body}"
        );
    }
    assert!(f.installer.installs().is_empty(), "nothing was downloaded");
    assert!(state(&f).await["jobs"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn a_confirmed_download_returns_a_job_and_the_model_becomes_installed() {
    let f = missing();
    assert_eq!(model_entry(&state(&f).await, MF)["installed"], false);
    let id = start_download(&f).await;
    wait_idle(&f).await;
    let s = state(&f).await;
    let job = s["jobs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|j| j["id"] == id)
        .unwrap()
        .clone();
    assert_eq!(
        (
            job["kind"].as_str(),
            job["status"].as_str(),
            job["model"].as_str()
        ),
        (Some("download"), Some("完了"), Some(MF))
    );
    assert_eq!(job["bytes_done"], job["bytes_total"]);
    let entry = model_entry(&s, MF);
    assert_eq!(
        (
            entry["installed"].as_bool(),
            entry["download_bytes"].as_u64()
        ),
        (Some(true), Some(0))
    );
}

#[tokio::test]
async fn state_shows_download_bytes_for_missing_models_only() {
    let f = missing();
    f.installer.set_approximate_bytes(3_300_000_000);
    let s = state(&f).await;
    let mf = model_entry(&s, MF);
    assert_eq!(
        (mf["installed"].as_bool(), mf["download_bytes"].as_u64()),
        (Some(false), Some(3_300_000_000))
    );
    let small = model_entry(&s, "Aratako/Irodori-TTS-v4.1-Small");
    assert_eq!(
        (
            small["installed"].as_bool(),
            small["download_bytes"].as_u64()
        ),
        (Some(true), Some(0))
    );
}

#[tokio::test]
async fn generate_asks_before_downloading() {
    let f = missing();
    f.installer.set_approximate_bytes(3_300_000_000);
    let s = state(&f).await;
    let (pid, rev) = (
        s["project"]["id"].as_str().unwrap().to_string(),
        s["project"]["revision"].as_i64().unwrap(),
    );
    let cid = s["project"]["cards"][0]["id"].as_str().unwrap().to_string();
    let e = post(&f.app, "/api/edit", json!({"pid": pid, "revision": rev, "values": {"cards": [{"id": cid, "text": "こんにちは。", "voice_id": VOICE}]}})).await;
    let rev = e.json()["project"]["revision"].as_i64().unwrap();
    let r = post(
        &f.app,
        "/api/generate",
        json!({"pid": pid, "revision": rev}),
    )
    .await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.error());
    let body = r.json();
    assert_eq!(body["needs_model_download"], true);
    assert_eq!(
        (
            body["model"].as_str(),
            body["name"].as_str(),
            body["download_bytes"].as_u64()
        ),
        (Some(MF), Some("MF（高速）"), Some(3_300_000_000))
    );
    assert!(
        f.installer.installs().is_empty(),
        "asking does not download"
    );
    let after = state(&f).await;
    assert!(after["jobs"].as_array().unwrap().is_empty());
    assert_eq!(
        after["project"]["cards"][0]["status"], "未生成",
        "the card is untouched"
    );
}

#[tokio::test]
async fn design_asks_before_downloading() {
    let f = missing();
    f.installer.set_approximate_bytes(123);
    let values = json!({"model": MF, "mode": "design", "caption": "明るい声", "text": "こんにちは。", "candidates": 1});
    for uri in ["/api/design", "/api/design-registration"] {
        let r = post(&f.app, uri, values.clone()).await;
        assert_eq!(r.status, StatusCode::OK, "{uri}: {}", r.error());
        let body = r.json();
        assert_eq!(
            (
                body["needs_model_download"].as_bool(),
                body["download_bytes"].as_u64()
            ),
            (Some(true), Some(123)),
            "{uri}"
        );
        assert!(body.get("job").is_none());
    }
    assert!(f.installer.installs().is_empty());
    assert!(state(&f).await["jobs"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn generating_after_the_download_works() {
    let f = missing();
    start_download(&f).await;
    wait_idle(&f).await;
    let s = state(&f).await;
    let (pid, rev) = (
        s["project"]["id"].as_str().unwrap().to_string(),
        s["project"]["revision"].as_i64().unwrap(),
    );
    let cid = s["project"]["cards"][0]["id"].as_str().unwrap().to_string();
    let e = post(&f.app, "/api/edit", json!({"pid": pid, "revision": rev, "values": {"cards": [{"id": cid, "text": "こんにちは。", "voice_id": VOICE}]}})).await;
    let rev = e.json()["project"]["revision"].as_i64().unwrap();
    let g = post(
        &f.app,
        "/api/generate",
        json!({"pid": pid, "revision": rev}),
    )
    .await;
    assert_eq!(g.json()["message"], "生成を開始しました。", "{}", g.json());
    wait_idle(&f).await;
    assert_eq!(state(&f).await["project"]["cards"][0]["status"], "完了");
}

#[tokio::test]
async fn a_second_download_while_one_runs_is_400() {
    let f = missing();
    slow(&f);
    start_download(&f).await;
    let r = post(
        &f.app,
        "/api/models/download",
        json!({"model": MF, "confirm": true}),
    )
    .await;
    assert_eq!(
        (r.status, r.error()),
        (
            StatusCode::BAD_REQUEST,
            "現在の処理が終わってから実行してください。".into()
        )
    );
    post(&f.app, "/api/models/cancel", json!({})).await;
    wait_idle(&f).await;
}

#[tokio::test]
async fn cancel_stops_a_running_download() {
    let f = missing();
    slow(&f);
    let id = start_download(&f).await;
    tokio::time::sleep(Duration::from_millis(60)).await;
    let r = post(&f.app, "/api/models/cancel", json!({})).await;
    assert_eq!(r.status, StatusCode::OK);
    assert_eq!(r.json()["message"], "取得を中止します。");
    wait_idle(&f).await;
    let s = state(&f).await;
    let job = s["jobs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|j| j["id"] == id)
        .unwrap()
        .clone();
    assert_eq!(job["status"], "停止");
    assert_eq!(model_entry(&s, MF)["installed"], false);
}

#[tokio::test]
async fn cancel_without_a_download_is_ok_and_leaves_a_generation_alone() {
    let f = fixture_with_voice(|s| s.with_steps(40).with_step_delay(Duration::from_millis(10)));
    assert_eq!(
        post(&f.app, "/api/models/cancel", json!({})).await.status,
        StatusCode::OK
    );
    let s = state(&f).await;
    let (pid, rev) = (
        s["project"]["id"].as_str().unwrap().to_string(),
        s["project"]["revision"].as_i64().unwrap(),
    );
    let cid = s["project"]["cards"][0]["id"].as_str().unwrap().to_string();
    let e = post(&f.app, "/api/edit", json!({"pid": pid, "revision": rev, "values": {"cards": [{"id": cid, "text": "こんにちは。", "voice_id": VOICE}]}})).await;
    let rev = e.json()["project"]["revision"].as_i64().unwrap();
    post(
        &f.app,
        "/api/generate",
        json!({"pid": pid, "revision": rev}),
    )
    .await;
    post(&f.app, "/api/models/cancel", json!({})).await;
    wait_idle(&f).await;
    assert_eq!(
        state(&f).await["project"]["cards"][0]["status"],
        "完了",
        "the generation finished normally"
    );
}

#[tokio::test]
async fn download_routes_require_the_token() {
    let f = missing();
    for uri in ["/api/models/download", "/api/models/cancel"] {
        assert_eq!(
            post_raw_no_token(&f.app, uri).await.status,
            StatusCode::FORBIDDEN,
            "{uri}"
        );
    }
    assert!(f.installer.installs().is_empty());
}

#[tokio::test]
async fn state_polling_and_other_reads_never_download() {
    let f = missing();
    for _ in 0..20 {
        assert_eq!(get(&f.app, "/api/state").await.status, StatusCode::OK);
        get(&f.app, "/api/meta").await;
    }
    assert!(f.installer.installs().is_empty());
}

#[tokio::test]
async fn unknown_or_installed_models_are_400() {
    let f = missing();
    for body in [
        json!({"model": "evil/model", "confirm": true}),
        json!({"confirm": true}),
        json!({"model": 5, "confirm": true}),
    ] {
        let r = post(&f.app, "/api/models/download", body.clone()).await;
        assert_eq!(
            (r.status, r.error()),
            (
                StatusCode::BAD_REQUEST,
                "使用するモデルを選んでください。".into()
            ),
            "{body}"
        );
    }
    let r = post(
        &f.app,
        "/api/models/download",
        json!({"model": "Aratako/Irodori-TTS-v4.1-Small", "confirm": true}),
    )
    .await;
    assert_eq!(
        (r.status, r.error()),
        (
            StatusCode::BAD_REQUEST,
            "このモデルはすでに取得済みです。".into()
        )
    );
    let r = post_raw(&f.app, "/api/models/download", "not json").await;
    assert_eq!(r.status, StatusCode::BAD_REQUEST);
    assert!(f.installer.installs().is_empty());
}

#[tokio::test]
async fn a_failed_download_shows_the_japanese_reason() {
    let f = missing();
    f.installer.fail_with(Some(DownloadError::Offline));
    let id = start_download(&f).await;
    wait_idle(&f).await;
    let s = state(&f).await;
    let job = s["jobs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|j| j["id"] == id)
        .unwrap()
        .clone();
    assert_eq!(job["status"], "失敗");
    assert_eq!(
        job["error"],
        "ネットワークに接続できません。接続を確認してもう一度お試しください。"
    );
    assert_eq!(model_entry(&s, MF)["installed"], false);
}
