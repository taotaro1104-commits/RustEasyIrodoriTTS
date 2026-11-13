//! HTTP の安全対策と、埋め込み画面・メタ情報の配信。

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{header, Method, Request, StatusCode};
use axum::Router;
use irodori_app::{build_router, AppState};
use irodori_store::Store;
use serde_json::Value;
use tower::ServiceExt;

const TOKEN: &str = "test-token-123";

struct Fixture {
    _dir: tempfile::TempDir,
    app: Router,
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    for sub in ["projects", "voices", "config", "temp", "outputs"] {
        std::fs::create_dir_all(dir.path().join(sub)).unwrap();
    }
    let store = Arc::new(Store::open(dir.path()).unwrap());
    let synth = Arc::new(irodori_app::fake::FakeSynth::new(dir.path()));
    let state = AppState::new(
        store,
        PathBuf::from(dir.path()),
        TOKEN.to_string(),
        "cpu".to_string(),
        synth,
    );
    Fixture {
        app: build_router(state),
        _dir: dir,
    }
}

struct Reply {
    status: StatusCode,
    headers: axum::http::HeaderMap,
    body: Vec<u8>,
}

impl Reply {
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap()
    }
}

async fn send(
    app: &Router,
    method: Method,
    uri: &str,
    host: Option<&str>,
    token: Option<&str>,
) -> Reply {
    let mut req = Request::builder().method(method).uri(uri);
    if let Some(h) = host {
        req = req.header(header::HOST, h);
    }
    if let Some(t) = token {
        req = req.header("x-easy-token", t);
    }
    let res = app
        .clone()
        .oneshot(req.body(Body::empty()).unwrap())
        .await
        .unwrap();
    let (parts, body) = res.into_parts();
    Reply {
        status: parts.status,
        headers: parts.headers,
        body: to_bytes(body, 8 * 1024 * 1024).await.unwrap().to_vec(),
    }
}

async fn get(app: &Router, uri: &str) -> Reply {
    send(app, Method::GET, uri, Some("127.0.0.1:7860"), None).await
}

#[tokio::test]
async fn foreign_host_is_forbidden() {
    let f = fixture();
    for host in [
        Some("evil.example"),
        Some("evil.example:7860"),
        Some("localhost.evil.example"),
        Some("127.0.0.1.evil.example"),
        Some(""),
        None,
    ] {
        let r = send(&f.app, Method::GET, "/", host, None).await;
        assert_eq!(r.status, StatusCode::FORBIDDEN, "host {host:?}");
        assert_eq!(r.json()["error"], "ローカル接続のみ利用できます。");
    }
}

#[tokio::test]
async fn forbidden_host_also_blocks_unknown_paths_and_static_files() {
    let f = fixture();
    for uri in ["/api/meta", "/editor/app.js", "/nothing-here"] {
        let r = send(&f.app, Method::GET, uri, Some("evil.example"), None).await;
        assert_eq!(r.status, StatusCode::FORBIDDEN, "{uri}");
    }
}

#[tokio::test]
async fn localhost_and_127_0_0_1_are_allowed() {
    let f = fixture();
    for host in ["127.0.0.1", "127.0.0.1:7860", "localhost", "localhost:1234"] {
        let r = send(&f.app, Method::GET, "/", Some(host), None).await;
        assert_eq!(r.status, StatusCode::OK, "host {host}");
    }
}

#[tokio::test]
async fn post_without_token_is_forbidden() {
    let f = fixture();
    let r = send(
        &f.app,
        Method::POST,
        "/api/edit",
        Some("127.0.0.1:7860"),
        None,
    )
    .await;
    assert_eq!(r.status, StatusCode::FORBIDDEN);
    assert_eq!(r.json()["error"], "画面を再読み込みしてください。");
}

#[tokio::test]
async fn post_with_wrong_token_is_forbidden() {
    let f = fixture();
    for token in ["wrong", "", "test-token-12", "test-token-1234"] {
        let r = send(
            &f.app,
            Method::POST,
            "/api/edit",
            Some("127.0.0.1:7860"),
            Some(token),
        )
        .await;
        assert_eq!(r.status, StatusCode::FORBIDDEN, "token {token:?}");
    }
    for method in [Method::PUT, Method::DELETE, Method::PATCH] {
        let r = send(&f.app, method.clone(), "/api/edit", Some("127.0.0.1"), None).await;
        assert_eq!(r.status, StatusCode::FORBIDDEN, "{method}");
    }
}

#[tokio::test]
async fn post_with_the_right_token_passes_the_security_check() {
    let f = fixture();
    let r = send(
        &f.app,
        Method::POST,
        "/api/no-such-route",
        Some("127.0.0.1:7860"),
        Some(TOKEN),
    )
    .await;
    assert_ne!(r.status, StatusCode::FORBIDDEN, "reaches routing instead");
}

#[tokio::test]
async fn get_without_token_is_allowed() {
    let f = fixture();
    assert_eq!(get(&f.app, "/api/meta").await.status, StatusCode::OK);
    let r = send(&f.app, Method::HEAD, "/", Some("localhost"), None).await;
    assert_eq!(r.status, StatusCode::OK);
}

#[tokio::test]
async fn index_injects_the_token_and_disables_caching() {
    let f = fixture();
    let r = get(&f.app, "/").await;
    assert_eq!(r.status, StatusCode::OK);
    let html = r.text();
    assert!(
        html.contains(&format!(r#"<meta name="easy-token" content="{TOKEN}">"#)),
        "token injected"
    );
    assert!(!html.contains("__TOKEN__"));
    assert_eq!(r.headers[header::CACHE_CONTROL], "no-store");
    assert!(r.headers[header::CONTENT_TYPE]
        .to_str()
        .unwrap()
        .starts_with("text/html"));
}

#[tokio::test]
async fn editor_serves_only_app_js_and_style_css() {
    let f = fixture();
    for (name, kind) in [("app.js", "javascript"), ("style.css", "text/css")] {
        let r = get(&f.app, &format!("/editor/{name}")).await;
        assert_eq!(r.status, StatusCode::OK, "{name}");
        assert!(
            r.headers[header::CONTENT_TYPE]
                .to_str()
                .unwrap()
                .contains(kind),
            "{name}"
        );
        assert_eq!(r.headers[header::CACHE_CONTROL], "no-store");
        assert!(!r.body.is_empty());
    }
    for uri in [
        "/editor/index.html",
        "/editor/secret",
        "/editor/../Cargo.toml",
        "/editor/..%2FCargo.toml",
        "/editor/%2e%2e%2fCargo.toml",
        "/editor/",
        "/editor",
        "/editor/app.js/extra",
    ] {
        assert_eq!(
            get(&f.app, uri).await.status,
            StatusCode::NOT_FOUND,
            "{uri}"
        );
    }
}

#[tokio::test]
async fn meta_contains_emoji_groups_and_rendered_credits() {
    let f = fixture();
    let r = get(&f.app, "/api/meta").await;
    assert_eq!(r.status, StatusCode::OK);
    let v = r.json();
    let emojis = v["emojis"].as_array().unwrap();
    assert!(emojis.len() >= 40);
    for e in emojis {
        for k in ["emoji", "label", "description"] {
            assert!(e[k].is_string(), "emoji missing {k}");
        }
    }
    let groups = v["emoji_groups"].as_array().unwrap();
    assert!(!groups.is_empty());
    let grouped: usize = groups
        .iter()
        .map(|g| g["items"].as_array().unwrap().len())
        .sum();
    assert_eq!(grouped, emojis.len(), "every emoji is in exactly one group");
    assert!(groups.iter().all(|g| g["title"].is_string()));
    let credits = v["credits"].as_str().unwrap();
    for needle in [
        "ライセンス",
        "禁止事項・免責事項",
        "本人の同意なしに声を複製",
        "ゆうぷろ",
        "<a href=\"https://github.com/Aratako/Irodori-TTS\">",
    ] {
        assert!(credits.contains(needle), "credits lack {needle}");
    }
    assert!(!credits.contains("###"), "markdown was rendered");
}
