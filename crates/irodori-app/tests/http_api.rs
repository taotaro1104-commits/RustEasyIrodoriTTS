//! M2a の API ルート（状態・編集・操作・台本・設定・辞書・復元・音声）。

mod common;

use axum::http::{header, StatusCode};
use common::*;
use serde_json::{json, Value};

async fn state_of(f: &Fixture) -> Value {
    let r = get(&f.app, "/api/state").await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.error());
    r.json()
}

fn pid_rev(state: &Value) -> (String, i64) {
    (
        state["project"]["id"].as_str().unwrap().to_string(),
        state["project"]["revision"].as_i64().unwrap(),
    )
}

#[tokio::test]
async fn state_returns_a_new_project_on_first_run() {
    let f = fixture();
    let s = state_of(&f).await;
    assert_eq!(s["project"]["cards"].as_array().unwrap().len(), 1);
    assert_eq!(s["projects"].as_array().unwrap().len(), 1);
    assert_eq!(
        (s["backend"].as_str(), s["jobs"].clone()),
        (Some("cpu"), json!([]))
    );
    assert_eq!(s["settings"]["theme"], "light");
    assert_eq!(s["models"].as_array().unwrap().len(), 3);
    assert_eq!(s["history"], json!({"undo": false, "redo": false}));
}

#[tokio::test]
async fn edit_then_state_roundtrip() {
    let f = fixture();
    let s = state_of(&f).await;
    let (pid, rev) = pid_rev(&s);
    let cid = s["project"]["cards"][0]["id"].as_str().unwrap();
    let r = post(&f.app, "/api/edit", json!({"pid": pid, "revision": rev, "values": {"cards": [{"id": cid, "text": "こんにちは。"}]}})).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.error());
    let edited = r.json();
    assert_eq!(edited["project"]["revision"], rev + 1);
    assert_eq!(edited["project"]["cards"][0]["text"], "こんにちは。");
    assert!(edited["project"]["cards"][0]["estimate"].as_f64().unwrap() > 0.0);
    assert_eq!(edited["history"]["undo"], true);
    let again = get(&f.app, &format!("/api/state?pid={pid}")).await.json();
    assert_eq!(again["project"]["cards"][0]["text"], "こんにちは。");
}

#[tokio::test]
async fn stale_revision_returns_409_with_python_message() {
    let f = fixture();
    let (pid, rev) = pid_rev(&state_of(&f).await);
    let ok = post(
        &f.app,
        "/api/edit",
        json!({"pid": pid, "revision": rev, "values": {"name": "一回目"}}),
    )
    .await;
    assert_eq!(ok.status, StatusCode::OK);
    let stale = post(
        &f.app,
        "/api/edit",
        json!({"pid": pid, "revision": rev, "values": {"name": "二回目"}}),
    )
    .await;
    assert_eq!(stale.status, StatusCode::CONFLICT);
    assert_eq!(
        stale.error(),
        "別の操作で台本が更新されています。保存済みの内容を読み直してください。"
    );
    assert_eq!(state_of(&f).await["project"]["name"], "一回目");
}

#[tokio::test]
async fn bad_request_bodies_are_400_not_crashes() {
    let f = fixture();
    assert_eq!(
        post_raw(&f.app, "/api/edit", "{not json").await.status,
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        post(&f.app, "/api/edit", json!({})).await.status,
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        post(
            &f.app,
            "/api/edit",
            json!({"pid": "x", "revision": "no", "values": {}})
        )
        .await
        .status,
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        post(&f.app, "/api/action", json!({"pid": "x"}))
            .await
            .status,
        StatusCode::BAD_REQUEST
    );
}

#[tokio::test]
async fn action_add_delete_reorder_split_undo_redo_over_http() {
    let f = fixture();
    let (pid, mut rev) = pid_rev(&state_of(&f).await);
    let act = |rev: i64, extra: Value| {
        let mut body = json!({"pid": pid, "revision": rev});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        post(&f.app, "/api/action", body)
    };
    let added = act(rev, json!({"action": "add"})).await;
    assert_eq!(added.status, StatusCode::OK, "{}", added.error());
    let s = added.json();
    assert_eq!(
        (
            s["message"].as_str(),
            s["project"]["cards"].as_array().unwrap().len()
        ),
        (Some(""), 2)
    );
    rev = s["project"]["revision"].as_i64().unwrap();
    let ids: Vec<String> = s["project"]["cards"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["id"].as_str().unwrap().to_string())
        .collect();
    // 並べ替え
    let s = act(rev, json!({"action": "reorder", "order": [ids[1], ids[0]]}))
        .await
        .json();
    assert_eq!(s["project"]["cards"][0]["id"], ids[1].as_str());
    rev = s["project"]["revision"].as_i64().unwrap();
    // 分割
    let long = "あ。".repeat(100);
    let s = post(
        &f.app,
        "/api/edit",
        json!({"pid": pid, "revision": rev, "values": {"cards": [{"id": ids[1], "text": long}]}}),
    )
    .await
    .json();
    rev = s["project"]["revision"].as_i64().unwrap();
    let s = act(rev, json!({"action": "split", "cid": ids[1]}))
        .await
        .json();
    assert!(
        s["message"].as_str().unwrap().starts_with("続きのカードを"),
        "{}",
        s["message"]
    );
    rev = s["project"]["revision"].as_i64().unwrap();
    assert!(s["project"]["cards"].as_array().unwrap().len() > 2);
    // 元に戻す・やり直す
    let s = act(rev, json!({"action": "undo"})).await.json();
    assert_eq!(s["message"], "元に戻しました。");
    assert_eq!(s["project"]["cards"].as_array().unwrap().len(), 2);
    rev = s["project"]["revision"].as_i64().unwrap();
    let s = act(rev, json!({"action": "redo"})).await.json();
    assert_eq!(s["message"], "やり直しました。");
    rev = s["project"]["revision"].as_i64().unwrap();
    // 削除と不正な操作
    let s = act(rev, json!({"action": "delete", "cid": ids[1]}))
        .await
        .json();
    assert!(s["project"]["cards"]
        .as_array()
        .unwrap()
        .iter()
        .all(|c| c["id"] != ids[1].as_str()));
    let bad = act(
        s["project"]["revision"].as_i64().unwrap(),
        json!({"action": "bogus", "cid": ids[0]}),
    )
    .await;
    assert_eq!(
        (bad.status, bad.error().as_str()),
        (StatusCode::BAD_REQUEST, "操作を認識できません。")
    );
}

#[tokio::test]
async fn a_broken_dictionary_is_reported_after_the_change_is_saved() {
    let f = fixture();
    let (pid, rev) = pid_rev(&state_of(&f).await);
    std::fs::write(
        f.dir.path().join("config/reading_dictionary.json"),
        r#"{"a": 1}"#,
    )
    .unwrap();
    // v2.0 と同じく、保存したあとに状態を組み立てるところで誤りを知らせる。
    let r = post(
        &f.app,
        "/api/edit",
        json!({"pid": pid, "revision": rev, "values": {"dictionary_enabled": false}}),
    )
    .await;
    assert_eq!(
        (r.status, r.error().as_str()),
        (StatusCode::BAD_REQUEST, "読み辞書の形式が不正です。")
    );
    let saved = f.store.load(Some(&pid)).unwrap();
    assert!(!saved.dictionary_enabled, "the edit itself was saved");
    assert_eq!(
        get(&f.app, "/api/state").await.status,
        StatusCode::BAD_REQUEST
    );
    // 辞書を使わない台本の分割は、辞書ファイルを読まずに実行される。
    let r = post(
        &f.app,
        "/api/action",
        json!({"pid": pid, "revision": saved.revision, "action": "split"}),
    )
    .await;
    assert_eq!(
        r.error(),
        "読み辞書の形式が不正です。",
        "only the final state build complains"
    );
}

#[tokio::test]
async fn settings_validation_errors_are_400() {
    let f = fixture();
    let bad = post(&f.app, "/api/settings", json!({"theme": "neon"})).await;
    assert_eq!(bad.status, StatusCode::BAD_REQUEST);
    let bad = post(&f.app, "/api/settings", json!({"backend": "gpu"})).await;
    assert_eq!(
        (bad.status, bad.error().as_str()),
        (
            StatusCode::BAD_REQUEST,
            "CPUまたはNVIDIA GPUを選んでください。"
        )
    );
    let bad = post(&f.app, "/api/settings", json!({"inherit": "yes"})).await;
    assert_eq!(bad.error(), "設定値が不正です。");
    let ok = post(
        &f.app,
        "/api/settings",
        json!({"theme": "dark", "inherit": false}),
    )
    .await;
    assert_eq!(ok.status, StatusCode::OK);
    assert_eq!(
        ok.json()["message"],
        "設定を保存しました。生成モード・自動起動は次回起動時に反映されます。"
    );
    let s = state_of(&f).await;
    assert_eq!(
        (
            s["settings"]["theme"].as_str(),
            s["settings"]["inherit"].as_bool()
        ),
        (Some("dark"), Some(false))
    );
}

#[tokio::test]
async fn dictionary_register_update_delete_over_http() {
    let f = fixture();
    let reg = post(
        &f.app,
        "/api/dictionary",
        json!({"word": "Irodori", "reading": "いろどり"}),
    )
    .await;
    assert_eq!(
        (reg.status, reg.json()["message"].clone()),
        (StatusCode::OK, json!("読み方を登録しました。"))
    );
    assert_eq!(state_of(&f).await["dictionary"]["Irodori"], "いろどり");
    let upd = post(
        &f.app,
        "/api/dictionary",
        json!({"word": "Irodori", "reading": "イロドリ"}),
    )
    .await;
    assert_eq!(upd.json()["message"], "読み方を更新しました。");
    let empty = post(
        &f.app,
        "/api/dictionary",
        json!({"word": "", "reading": "x"}),
    )
    .await;
    assert_eq!(
        (empty.status, empty.error().as_str()),
        (
            StatusCode::BAD_REQUEST,
            "「表記」と「読み方」の両方を入力してください。"
        )
    );
    let del = post(
        &f.app,
        "/api/dictionary",
        json!({"delete": true, "word": "Irodori"}),
    )
    .await;
    assert_eq!(del.json()["message"], "選んだ登録を削除しました。");
    let again = post(
        &f.app,
        "/api/dictionary",
        json!({"delete": true, "word": "Irodori"}),
    )
    .await;
    assert_eq!(
        (again.status, again.error().as_str()),
        (
            StatusCode::BAD_REQUEST,
            "削除する登録を一覧から選んでください。"
        )
    );
    assert_eq!(state_of(&f).await["dictionary"], json!({}));
}

#[tokio::test]
async fn project_new_copy_and_load_over_http() {
    let f = fixture();
    let (pid, rev) = pid_rev(&state_of(&f).await);
    let cid = state_of(&f).await["project"]["cards"][0]["id"]
        .as_str()
        .unwrap()
        .to_string();
    post(&f.app, "/api/edit", json!({"pid": pid, "revision": rev, "values": {"cards": [{"id": cid, "text": "元の台本"}]}})).await;
    let copy = post(
        &f.app,
        "/api/project",
        json!({"name": "コピー", "copy": true, "pid": pid}),
    )
    .await
    .json();
    assert_ne!(copy["project"]["id"], pid.as_str());
    assert_eq!(
        (
            copy["project"]["name"].as_str(),
            copy["project"]["cards"][0]["text"].as_str()
        ),
        (Some("コピー"), Some("元の台本"))
    );
    assert_eq!(copy["projects"].as_array().unwrap().len(), 2);
    let loaded = post(&f.app, "/api/project", json!({"load": true, "pid": pid}))
        .await
        .json();
    assert_eq!(loaded["project"]["id"], pid.as_str());
    assert_eq!(loaded["settings"]["last_project"], pid.as_str());
}

#[tokio::test]
async fn recover_previous_without_a_saved_input_is_400() {
    let f = fixture();
    let (pid, rev) = pid_rev(&state_of(&f).await);
    let r = post(
        &f.app,
        "/api/recover-previous",
        json!({"pid": pid, "revision": rev}),
    )
    .await;
    assert_eq!(
        (r.status, r.error().as_str()),
        (
            StatusCode::BAD_REQUEST,
            "復元できる前回の入力がありません。"
        )
    );
}

#[tokio::test]
async fn recover_previous_restores_the_saved_input() {
    let f = fixture();
    let (old_pid, rev) = pid_rev(&state_of(&f).await);
    let cid = state_of(&f).await["project"]["cards"][0]["id"]
        .as_str()
        .unwrap()
        .to_string();
    post(&f.app, "/api/edit", json!({"pid": old_pid, "revision": rev, "values": {"cards": [{"id": cid, "text": "前回の入力"}]}})).await;
    let new = post(&f.app, "/api/project", json!({"name": "今回"}))
        .await
        .json();
    let new_pid = new["project"]["id"].as_str().unwrap();
    // 起動時の処理（Task 12）が行う設定の書き込みを再現する
    f.store.set_recovery_project(Some(&old_pid)).unwrap();
    let r = post(
        &f.app,
        "/api/recover-previous",
        json!({"pid": new_pid, "revision": 0}),
    )
    .await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.error());
    assert_eq!(r.json()["project"]["cards"][0]["text"], "前回の入力");
}

#[tokio::test]
async fn invalid_pid_is_400_for_state() {
    let f = fixture();
    for pid in ["../x", "ZZZ", "0123456789abcdef0123456789abcdeg"] {
        let r = get(&f.app, &format!("/api/state?pid={pid}")).await;
        assert_eq!(r.status, StatusCode::BAD_REQUEST, "pid {pid}");
        assert!(!r.error().is_empty());
    }
}

fn wav_asset(f: &Fixture) -> String {
    let path = f.dir.path().join("outputs/sample.wav");
    std::fs::write(&path, (0..=255u8).collect::<Vec<_>>()).unwrap();
    f.assets.register(&path).unwrap()
}

#[tokio::test]
async fn audio_requires_a_registered_key() {
    let f = fixture();
    let r = get(&f.app, "/audio/not-registered").await;
    assert_eq!(
        (r.status, r.error().as_str()),
        (
            StatusCode::BAD_REQUEST,
            "音声が見つかりません。もう一度追加してください。"
        )
    );
    let key = wav_asset(&f);
    let ok = get(&f.app, &format!("/audio/{key}")).await;
    assert_eq!(ok.status, StatusCode::OK);
    assert_eq!(ok.headers[header::CONTENT_TYPE], "audio/wav");
    assert_eq!(ok.body, (0..=255u8).collect::<Vec<_>>());
    assert!(ok.headers.get(header::CONTENT_DISPOSITION).is_none());
}

#[tokio::test]
async fn audio_supports_range_requests_for_seeking() {
    let f = fixture();
    let key = wav_asset(&f);
    let r = get_with(
        &f.app,
        &format!("/audio/{key}"),
        &[("range", "bytes=10-19")],
    )
    .await;
    assert_eq!(r.status, StatusCode::PARTIAL_CONTENT);
    assert_eq!(r.body, (10..20u8).collect::<Vec<_>>());
    assert_eq!(r.headers[header::CONTENT_TYPE], "audio/wav");
}

async fn disposition(f: &Fixture, key: &str, name: &str) -> Option<String> {
    let uri = format!("/audio/{key}?download=true&name={name}");
    get(&f.app, &uri)
        .await
        .headers
        .get(header::CONTENT_DISPOSITION)
        .map(|v| v.to_str().unwrap().to_string())
}

#[tokio::test]
async fn audio_download_uses_a_safe_filename() {
    let f = fixture();
    let key = wav_asset(&f);
    assert_eq!(
        disposition(&f, &key, "a%3Ab%3F.wav").await.as_deref(),
        Some(r#"attachment; filename="a_b_.wav""#)
    );
    assert_eq!(
        disposition(&f, &key, "..%2F..%2Fevil.wav").await.as_deref(),
        Some(r#"attachment; filename=".._.._evil.wav""#)
    );
    assert_eq!(
        disposition(&f, &key, "%E6%97%A5%E6%9C%AC%E8%AA%9E.wav")
            .await
            .as_deref(),
        Some("attachment; filename*=utf-8''%E6%97%A5%E6%9C%AC%E8%AA%9E.wav")
    );
    assert_eq!(
        disposition(&f, &key, "con").await.as_deref(),
        Some(r#"attachment; filename="_con""#)
    );
    let default = get(&f.app, &format!("/audio/{key}?download=1")).await;
    assert!(default.headers[header::CONTENT_DISPOSITION]
        .to_str()
        .unwrap()
        .contains("filename*=utf-8''%E9%9F%B3%E5%A3%B0.wav"));
}

#[tokio::test]
async fn unimplemented_routes_return_501_json() {
    let f = fixture();
    for uri in [
        "/api/generate",
        "/api/stop",
        "/api/voice",
        "/api/voice/delete",
        "/api/voice/rename",
        "/api/upload",
        "/api/export",
        "/api/bundle",
        "/api/save-audio",
        "/api/audio-list",
        "/api/cleanup",
        "/api/cleanup-preview",
        "/api/choose-folder",
        "/api/open-output",
        "/api/unload",
        "/api/design",
        "/api/design-registration",
        "/api/input-data/save",
        "/api/input-data/load",
    ] {
        let r = post(&f.app, uri, json!({})).await;
        assert_eq!(r.status, StatusCode::NOT_IMPLEMENTED, "{uri}");
        assert_eq!(r.error(), "未対応です（この版ではまだ使えません）。");
    }
    let r = get(&f.app, "/bundle/abc").await;
    assert_eq!(r.status, StatusCode::NOT_IMPLEMENTED);
}

#[tokio::test]
async fn favicon_request_is_answered_quietly() {
    let f = fixture();
    let r = get(&f.app, "/favicon.ico").await;
    assert_eq!(
        r.status,
        StatusCode::NO_CONTENT,
        "browsers ask for it automatically; a 404 shows up as a console error"
    );
    assert!(r.body.is_empty());
}
