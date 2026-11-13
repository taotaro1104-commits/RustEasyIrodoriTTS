//! HTTP テスト共通の道具。

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{header, HeaderMap, Method, Request, StatusCode};
use axum::Router;
use irodori_app::fake::FakeSynth;
use irodori_app::jobs::JobManager;
use irodori_app::{build_router, AppState};
use irodori_store::{assets::Assets, Store};
use serde_json::Value;
use tower::ServiceExt;

pub const TOKEN: &str = "test-token-123";

pub const MF: &str = "Aratako/Irodori-TTS-v4.1-Small-MF";
pub const VOICE: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

pub struct Fixture {
    pub dir: tempfile::TempDir,
    pub app: Router,
    pub assets: Arc<Assets>,
    pub store: Arc<Store>,
    pub jobs: Arc<JobManager>,
    pub synth: Arc<FakeSynth>,
}

pub fn fixture() -> Fixture {
    build(false, |s| s)
}

/// 話者を 1 人（`VOICE`）持つデータフォルダ。擬似エンジンは `tweak` で調整する。
pub fn fixture_with_voice(tweak: impl FnOnce(FakeSynth) -> FakeSynth) -> Fixture {
    build(true, tweak)
}

fn build(with_voice: bool, tweak: impl FnOnce(FakeSynth) -> FakeSynth) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    for sub in ["projects", "voices", "config", "temp", "outputs"] {
        std::fs::create_dir_all(dir.path().join(sub)).unwrap();
    }
    if with_voice {
        let audio = format!("voices/{VOICE}/ref.wav");
        std::fs::create_dir_all(dir.path().join(&audio).parent().unwrap()).unwrap();
        std::fs::write(dir.path().join(&audio), b"RIFF").unwrap();
        let voices = serde_json::json!([{"id": VOICE, "revision": 1, "audio": audio, "name": "話者A", "model": MF, "caption": "落ち着いた声"}]);
        std::fs::write(dir.path().join("voices/index.json"), voices.to_string()).unwrap();
    }
    let store = Arc::new(Store::open(dir.path()).unwrap());
    let synth = Arc::new(tweak(FakeSynth::new(dir.path())));
    let state = AppState::new(
        store.clone(),
        PathBuf::from(dir.path()),
        TOKEN.to_string(),
        "cpu".to_string(),
        synth.clone(),
    );
    Fixture {
        assets: state.assets.clone(),
        jobs: state.jobs.clone(),
        app: build_router(state),
        dir,
        store,
        synth,
    }
}

pub struct Reply {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
}

impl Reply {
    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body)
            .unwrap_or_else(|_| panic!("not json: {}", String::from_utf8_lossy(&self.body)))
    }
    pub fn error(&self) -> String {
        self.json()["error"]
            .as_str()
            .unwrap_or_default()
            .to_string()
    }
}

async fn run(app: &Router, req: Request<Body>) -> Reply {
    let res = app.clone().oneshot(req).await.unwrap();
    let (parts, body) = res.into_parts();
    Reply {
        status: parts.status,
        headers: parts.headers,
        body: to_bytes(body, 64 * 1024 * 1024).await.unwrap().to_vec(),
    }
}

pub async fn get(app: &Router, uri: &str) -> Reply {
    get_with(app, uri, &[]).await
}

pub async fn get_with(app: &Router, uri: &str, headers: &[(&str, &str)]) -> Reply {
    let mut req = Request::builder()
        .method(Method::GET)
        .uri(uri)
        .header(header::HOST, "127.0.0.1:7860");
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    run(app, req.body(Body::empty()).unwrap()).await
}

pub async fn post(app: &Router, uri: &str, body: Value) -> Reply {
    let req = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(header::HOST, "127.0.0.1:7860")
        .header("x-easy-token", TOKEN)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    run(app, req).await
}

pub async fn post_raw(app: &Router, uri: &str, body: &str) -> Reply {
    let req = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(header::HOST, "127.0.0.1:7860")
        .header("x-easy-token", TOKEN)
        .body(Body::from(body.to_string()))
        .unwrap();
    run(app, req).await
}
