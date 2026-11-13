//! M2a からの持ち越し: 本文の大きさ、ハンドラの panic、アップロードだけが大きな本文を受けること。

mod common;

use axum::http::StatusCode;
use axum::routing::{get as get_route, post as post_route};
use axum::Router;
use common::*;
use irodori_app::build_router_with;
use serde_json::json;

#[tokio::test]
async fn oversized_json_body_is_413_with_a_json_message() {
    let f = fixture();
    let big = format!("{{\"pid\": \"{}\"}}", "a".repeat(2 * 1024 * 1024 + 1));
    let r = post_raw(&f.app, "/api/edit", &big).await;
    assert_eq!(r.status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(r.error(), "リクエストが大きすぎます。");
    // 上限ぎりぎり以下は通常どおり処理される（形式の誤りとして 400）。
    let ok = post_raw(
        &f.app,
        "/api/edit",
        &format!("{{\"pid\": \"{}\"}}", "a".repeat(1024 * 1024)),
    )
    .await;
    assert_eq!(ok.status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn upload_keeps_its_own_larger_limit() {
    let f = fixture();
    let wav = wav_bytes(24000, 1, 90.0);
    assert!(wav.len() > 2 * 1024 * 1024, "{}", wav.len());
    let r = post_file(&f.app, "/api/upload", "file", "long.wav", &wav, true).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.error());
}

#[tokio::test]
async fn a_panicking_handler_becomes_a_json_500_and_the_server_survives() {
    let f = fixture();
    async fn boom() -> &'static str {
        panic!("boom with secret detail")
    }
    let extra = Router::new().route("/api/boom", get_route(boom));
    let app = build_router_with(f.state.clone(), extra);
    let r = get(&app, "/api/boom").await;
    assert_eq!(r.status, StatusCode::INTERNAL_SERVER_ERROR);
    let text = r.error();
    assert!(text.contains("logs/editor.log"), "{text}");
    assert!(!text.contains("secret"), "{text}");
    // 続けて普通の要求に答える。
    let s = get(&app, "/api/state").await;
    assert_eq!(s.status, StatusCode::OK);
    let edit = post(
        &app,
        "/api/dictionary",
        json!({"word": "犬", "reading": "いぬ"}),
    )
    .await;
    assert_eq!(edit.status, StatusCode::OK, "{}", edit.error());
}

#[tokio::test]
async fn extra_routes_are_still_behind_the_token_check() {
    let f = fixture();
    let extra = Router::new().route("/api/boom", post_route(|| async { "hi" }));
    let app = build_router_with(f.state.clone(), extra);
    let r = post_raw_no_token(&app, "/api/boom").await;
    assert_eq!(r.status, StatusCode::FORBIDDEN);
}
