//! ローカルだけで動く Easy-Irodori-TTS のサーバ。画面（editor）を配り、v2.0 と同じ API を提供する。

pub mod assets_embed;
pub mod errors;
pub mod fake;
pub mod jobs;
pub mod launch;
pub mod logging;
pub mod meta;
pub mod models;
pub mod real_synth;
pub mod routes;
pub mod security;
pub mod settings_map;
pub mod synth;
pub mod upload;
pub mod voices;

use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::middleware;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use irodori_store::{assets::Assets, Store};

use crate::jobs::JobManager;
use crate::synth::Synthesizer;

#[derive(Clone)]
pub struct AppState {
    pub store: Arc<Store>,
    pub assets: Arc<Assets>,
    /// 画面に埋め込む使い捨てトークン。書き込み系の API はこれが無いと拒否する。
    pub token: String,
    pub root: PathBuf,
    pub backend: String,
    /// 生成ジョブ（専用スレッドで直列に実行）。
    pub jobs: Arc<JobManager>,
    /// 合成エンジン（本物か、テスト用の擬似）。
    pub synth: Arc<dyn Synthesizer>,
}

impl AppState {
    /// 資産の登録簿と生成ジョブを新しく作って組み立てる。
    pub fn new(
        store: Arc<Store>,
        root: PathBuf,
        token: String,
        backend: String,
        synth: Arc<dyn Synthesizer>,
    ) -> AppState {
        let assets = Arc::new(Assets::new());
        let jobs = Arc::new(JobManager::new(
            Arc::clone(&store),
            Arc::clone(&synth),
            Arc::clone(&assets),
        ));
        AppState {
            store,
            assets,
            token,
            root,
            backend,
            jobs,
            synth,
        }
    }
}

fn no_store(mut response: Response) -> Response {
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

async fn index(State(state): State<AppState>) -> Response {
    let html = assets_embed::index_html().replace("__TOKEN__", &state.token);
    no_store(([(header::CONTENT_TYPE, "text/html; charset=utf-8")], html).into_response())
}

/// `app.js` と `style.css` だけを配る。それ以外の名前（パス移動を含む）は 404。
async fn editor_file(Path(name): Path<String>) -> Response {
    match assets_embed::editor_file(&name) {
        Some((bytes, mime)) => no_store(([(header::CONTENT_TYPE, mime)], bytes).into_response()),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

pub fn build_router(state: AppState) -> Router {
    build_router_with(state, Router::new())
}

/// `extra` のルートも、同じ安全対策（ローカル限定・トークン・panic の捕捉・ログ）の内側に置く（テスト用の差し込み口）。
pub fn build_router_with(state: AppState, extra: Router<AppState>) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/editor/{name}", get(editor_file))
        .route("/api/meta", get(meta::meta))
        // ブラウザが自動で取りに来る。無いことを 404 にすると、画面のコンソールにエラーとして残る。
        .route("/favicon.ico", get(|| async { StatusCode::NO_CONTENT }))
        .merge(routes::routes())
        .merge(extra)
        .layer(middleware::from_fn(errors::json_payload_too_large))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            security::local_only,
        ))
        .layer(tower_http::catch_panic::CatchPanicLayer::custom(
            errors::panic_response,
        ))
        .layer(middleware::from_fn(logging::access_log))
        .with_state(state)
}
