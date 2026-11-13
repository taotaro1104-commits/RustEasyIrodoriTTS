//! 生成・停止・アンロードの HTTP 経路（擬似エンジンで、モデル無しに全体を通す）。

mod common;

use std::time::{Duration, Instant};

use axum::http::{header, Method, Request, StatusCode};
use common::*;
use serde_json::{json, Value};
use tower::ServiceExt;

async fn state(f: &Fixture) -> Value {
    get(&f.app, "/api/state").await.json()
}

/// 1 枚目のカードに文章と話者を入れて、(pid, revision, card id) を返す。
async fn prepare(f: &Fixture, text: &str) -> (String, i64, String) {
    let s = state(f).await;
    let (pid, rev) = (
        s["project"]["id"].as_str().unwrap().to_string(),
        s["project"]["revision"].as_i64().unwrap(),
    );
    let cid = s["project"]["cards"][0]["id"].as_str().unwrap().to_string();
    let r = post(&f.app, "/api/edit", json!({"pid": pid, "revision": rev, "values": {"cards": [{"id": cid, "text": text, "voice_id": VOICE}]}})).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.error());
    let rev = r.json()["project"]["revision"].as_i64().unwrap();
    (pid, rev, cid)
}

async fn wait_state(f: &Fixture, what: &str, ok: impl Fn(&Value) -> bool) -> Value {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let s = state(f).await;
        if ok(&s) {
            return s;
        }
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn last_job(s: &Value) -> Option<&Value> {
    s["jobs"].as_array().and_then(|j| j.last())
}

#[tokio::test]
async fn generate_then_poll_state_until_the_card_is_done() {
    let f = fixture_with_voice(|s| s);
    let (pid, rev, _) = prepare(&f, "こんにちは。").await;
    let r = post(
        &f.app,
        "/api/generate",
        json!({"pid": pid, "revision": rev}),
    )
    .await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.error());
    let started = r.json();
    assert_eq!(started["message"], "生成を開始しました。");
    assert!(started["project"]["cards"][0]["status"].is_string());
    assert_eq!(started["jobs"].as_array().unwrap().len(), 1, "{started}");

    let done = wait_state(&f, "the job to finish", |s| {
        last_job(s).is_some_and(|j| j["status"] == "完了")
    })
    .await;
    let job = last_job(&done).unwrap();
    assert_eq!(
        (
            job["kind"].as_str(),
            job["done"].as_i64(),
            job["total"].as_i64(),
            job["steps"].as_i64()
        ),
        (Some("generate"), Some(1), Some(1), Some(4))
    );
    let card = &done["project"]["cards"][0];
    assert_eq!(card["status"], "完了");
    let key = card["audio"]
        .as_str()
        .expect("the finished card has an audio key");
    let audio = get(&f.app, &format!("/audio/{key}")).await;
    assert_eq!(
        (
            audio.status,
            audio.headers[header::CONTENT_TYPE].to_str().unwrap()
        ),
        (StatusCode::OK, "audio/wav")
    );
    let reader = hound::WavReader::new(std::io::Cursor::new(audio.body)).unwrap();
    assert_eq!(
        (reader.spec().bits_per_sample, reader.spec().channels),
        (24, 1)
    );
    let name = card["download_name"].as_str().unwrap();
    assert!(
        name.starts_with("001_話者A_") && name.ends_with(".wav"),
        "the suggested save name comes from the request (v2.0): {name}"
    );
}

#[tokio::test]
async fn generate_requires_the_token() {
    let f = fixture_with_voice(|s| s);
    let (pid, rev, _) = prepare(&f, "こんにちは。").await;
    for token in [None, Some("wrong")] {
        let mut req = Request::builder()
            .method(Method::POST)
            .uri("/api/generate")
            .header(header::HOST, "127.0.0.1:7860")
            .header(header::CONTENT_TYPE, "application/json");
        if let Some(t) = token {
            req = req.header("x-easy-token", t);
        }
        let body = json!({"pid": pid, "revision": rev}).to_string();
        let res = f
            .app
            .clone()
            .oneshot(req.body(axum::body::Body::from(body)).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::FORBIDDEN, "{token:?}");
    }
    assert!(!f.jobs.busy());
    assert!(f.synth.calls().is_empty(), "nothing was generated");
}

#[tokio::test]
async fn bad_generate_requests_are_400_or_409() {
    let f = fixture_with_voice(|s| s);
    let (pid, rev, _) = prepare(&f, "こんにちは。").await;
    assert_eq!(
        post(&f.app, "/api/generate", json!({})).await.status,
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        post(
            &f.app,
            "/api/generate",
            json!({"pid": "../x", "revision": 0})
        )
        .await
        .status,
        StatusCode::BAD_REQUEST
    );
    let stale = post(
        &f.app,
        "/api/generate",
        json!({"pid": pid, "revision": rev - 1}),
    )
    .await;
    assert_eq!(
        (stale.status, stale.error().as_str()),
        (
            StatusCode::CONFLICT,
            "台本が更新されています。読み直してください。"
        )
    );
    assert!(f.synth.calls().is_empty());
}

#[tokio::test]
async fn long_text_returns_the_confirmation_payload() {
    let f = fixture_with_voice(|s| s);
    let (pid, rev, _) = prepare(&f, &"あ。".repeat(100)).await;
    let r = post(
        &f.app,
        "/api/generate",
        json!({"pid": pid, "revision": rev}),
    )
    .await;
    assert_eq!(r.status, StatusCode::OK);
    assert_eq!(
        r.json(),
        json!({"needs_long_confirmation": true, "long_cards": [1]})
    );
    assert!(
        f.synth.calls().is_empty() && !f.jobs.busy(),
        "nothing starts before the user confirms"
    );
    let ok = post(
        &f.app,
        "/api/generate",
        json!({"pid": pid, "revision": rev, "allow_long": true}),
    )
    .await;
    assert_eq!(ok.json()["message"], "生成を開始しました。");
    wait_state(&f, "the long card to finish", |s| {
        last_job(s).is_some_and(|j| j["status"] == "完了")
    })
    .await;
}

#[tokio::test]
async fn stop_over_http_marks_the_card_stopped() {
    let f = fixture_with_voice(|s| s.with_steps(500).with_step_delay(Duration::from_millis(10)));
    let (pid, rev, _) = prepare(&f, "止めます。").await;
    post(
        &f.app,
        "/api/generate",
        json!({"pid": pid, "revision": rev}),
    )
    .await;
    wait_state(&f, "progress", |s| {
        last_job(s).is_some_and(|j| j["step"].as_u64().unwrap_or(0) >= 1)
    })
    .await;
    let stop = post(&f.app, "/api/stop", json!({})).await;
    assert_eq!(
        (stop.status, stop.json()["message"].clone()),
        (
            StatusCode::OK,
            json!("停止を受け付けました。処理中の1件が終わるまでお待ちください。")
        )
    );
    let done = wait_state(&f, "the job to stop", |s| {
        last_job(s).is_some_and(|j| j["status"] == "停止")
    })
    .await;
    assert_eq!(done["project"]["cards"][0]["status"], "停止");
    assert!(done["project"]["cards"][0]["audio"].is_null());
}

#[tokio::test]
async fn stop_with_nothing_running_is_a_quiet_ok() {
    let f = fixture_with_voice(|s| s);
    assert_eq!(
        post(&f.app, "/api/stop", json!({})).await.status,
        StatusCode::OK
    );
}

#[tokio::test]
async fn generate_with_a_missing_model_is_400_with_the_japanese_message() {
    let f = fixture_with_voice(|s| s.missing_models(&[MF]));
    let (pid, rev, _) = prepare(&f, "こんにちは。").await;
    let r = post(
        &f.app,
        "/api/generate",
        json!({"pid": pid, "revision": rev}),
    )
    .await;
    assert_eq!(
        (r.status, r.error().as_str()),
        (StatusCode::BAD_REQUEST, "モデルがまだ取得されていません。")
    );
    let s = state(&f).await;
    let installed: Vec<(String, bool)> = s["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            (
                m["id"].as_str().unwrap().to_string(),
                m["installed"].as_bool().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        installed,
        [
            ("Aratako/Irodori-TTS-v4.1-Small".to_string(), true),
            ("phasefield-audio/Irodori-TTS-v4.1-Anime".to_string(), true),
            (MF.to_string(), false)
        ]
    );
    assert_eq!(s["project"]["cards"][0]["status"], "未生成");
}

#[tokio::test]
async fn unload_works_when_idle_and_is_refused_while_generating() {
    let f = fixture_with_voice(|s| s.with_steps(500).with_step_delay(Duration::from_millis(10)));
    let idle = post(&f.app, "/api/unload", json!({})).await;
    assert_eq!(
        (idle.status, idle.json()["message"].clone()),
        (StatusCode::OK, json!("モデルを解放しました。"))
    );
    let (pid, rev, _) = prepare(&f, "こんにちは。").await;
    post(
        &f.app,
        "/api/generate",
        json!({"pid": pid, "revision": rev}),
    )
    .await;
    wait_state(&f, "progress", |s| {
        last_job(s).is_some_and(|j| j["step"].as_u64().unwrap_or(0) >= 1)
    })
    .await;
    let busy = post(&f.app, "/api/unload", json!({})).await;
    assert_eq!(
        (busy.status, busy.error().as_str()),
        (StatusCode::BAD_REQUEST, "生成完了後に実行してください。")
    );
    post(&f.app, "/api/stop", json!({})).await;
    wait_state(&f, "the job to stop", |s| {
        last_job(s).is_some_and(|j| j["status"] == "停止")
    })
    .await;
}

#[tokio::test]
async fn second_generate_while_running_is_400() {
    let f = fixture_with_voice(|s| s.with_steps(500).with_step_delay(Duration::from_millis(10)));
    let (pid, rev, _) = prepare(&f, "こんにちは。").await;
    post(
        &f.app,
        "/api/generate",
        json!({"pid": pid, "revision": rev}),
    )
    .await;
    let again = post(
        &f.app,
        "/api/generate",
        json!({"pid": pid, "revision": rev}),
    )
    .await;
    assert_eq!(
        (again.status, again.error().as_str()),
        (
            StatusCode::BAD_REQUEST,
            "現在の生成が終わるか、停止してから実行してください。"
        )
    );
    post(&f.app, "/api/stop", json!({})).await;
    wait_state(&f, "the job to stop", |s| {
        last_job(s).is_some_and(|j| j["status"] == "停止")
    })
    .await;
}

#[tokio::test]
async fn recover_previous_is_refused_while_generating() {
    let f = fixture_with_voice(|s| s.with_steps(500).with_step_delay(Duration::from_millis(10)));
    let (old_pid, _, _) = prepare(&f, "前回の入力").await;
    let new = post(&f.app, "/api/project", json!({"name": "今回"}))
        .await
        .json();
    let new_pid = new["project"]["id"].as_str().unwrap().to_string();
    let cid = new["project"]["cards"][0]["id"]
        .as_str()
        .unwrap()
        .to_string();
    let edited = post(&f.app, "/api/edit", json!({"pid": new_pid, "revision": 0, "values": {"cards": [{"id": cid, "text": "生成する", "voice_id": VOICE}]}})).await.json();
    let rev = edited["project"]["revision"].as_i64().unwrap();
    f.store.set_recovery_project(Some(&old_pid)).unwrap();
    post(
        &f.app,
        "/api/generate",
        json!({"pid": new_pid, "revision": rev}),
    )
    .await;
    wait_state(&f, "progress", |s| {
        last_job(s).is_some_and(|j| j["step"].as_u64().unwrap_or(0) >= 1)
    })
    .await;
    let r = post(
        &f.app,
        "/api/recover-previous",
        json!({"pid": new_pid, "revision": rev}),
    )
    .await;
    assert_eq!(
        (r.status, r.error().as_str()),
        (
            StatusCode::BAD_REQUEST,
            "生成が終わってから復元してください。"
        )
    );
    post(&f.app, "/api/stop", json!({})).await;
    wait_state(&f, "the job to stop", |s| {
        last_job(s).is_some_and(|j| j["status"] == "停止")
    })
    .await;
}

#[tokio::test]
async fn undo_does_not_stop_cards_that_are_still_queued() {
    let f = fixture_with_voice(|s| s.with_steps(500).with_step_delay(Duration::from_millis(10)));
    let (pid, rev, first) = prepare(&f, "一つ目。").await;
    let added = post(
        &f.app,
        "/api/action",
        json!({"pid": pid, "revision": rev, "action": "add"}),
    )
    .await
    .json();
    let rev = added["project"]["revision"].as_i64().unwrap();
    let second = added["project"]["cards"][1]["id"]
        .as_str()
        .unwrap()
        .to_string();
    let edited = post(&f.app, "/api/edit", json!({"pid": pid, "revision": rev, "values": {"cards": [{"id": second, "text": "二つ目。", "voice_id": VOICE}]}})).await.json();
    let rev = edited["project"]["revision"].as_i64().unwrap();
    post(
        &f.app,
        "/api/generate",
        json!({"pid": pid, "revision": rev}),
    )
    .await;
    wait_state(&f, "progress", |s| {
        last_job(s).is_some_and(|j| j["step"].as_u64().unwrap_or(0) >= 1)
    })
    .await;
    let undone = post(
        &f.app,
        "/api/action",
        json!({"pid": pid, "revision": rev, "action": "undo"}),
    )
    .await;
    assert_eq!(undone.status, StatusCode::OK, "{}", undone.error());
    let statuses: Vec<String> = undone.json()["project"]["cards"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["status"].as_str().unwrap().to_string())
        .collect();
    assert!(
        statuses.iter().all(|s| s != "停止"),
        "cards that are still part of the running job keep their status: {statuses:?}"
    );
    let _ = first;
    post(&f.app, "/api/stop", json!({})).await;
    wait_state(&f, "the job to stop", |s| {
        last_job(s).is_some_and(|j| j["status"] == "停止")
    })
    .await;
}
