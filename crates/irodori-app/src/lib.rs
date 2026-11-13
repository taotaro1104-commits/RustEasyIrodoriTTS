//! ローカルだけで動く Easy-Irodori-TTS のサーバ。画面（editor）を配り、v2.0 と同じ API を提供する。

pub mod assets_embed;
pub mod errors;
pub mod fake;
pub mod launch;
pub mod logging;
pub mod meta;
pub mod routes;
pub mod security;
pub mod settings_map;
pub mod synth;

use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::middleware;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use irodori_store::{assets::Assets, Store};

#[derive(Clone)]
pub struct AppState {
    pub store: Arc<Store>,
    pub assets: Arc<Assets>,
    /// 画面に埋め込む使い捨てトークン。書き込み系の API はこれが無いと拒否する。
    pub token: String,
    pub root: PathBuf,
    pub backend: String,
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
    Router::new()
        .route("/", get(index))
        .route("/editor/{name}", get(editor_file))
        .route("/api/meta", get(meta::meta))
        // ブラウザが自動で取りに来る。無いことを 404 にすると、画面のコンソールにエラーとして残る。
        .route("/favicon.ico", get(|| async { StatusCode::NO_CONTENT }))
        .merge(routes::routes())
        .layer(middleware::from_fn_with_state(
            state.clone(),
            security::local_only,
        ))
        .layer(middleware::from_fn(logging::access_log))
        .with_state(state)
}
