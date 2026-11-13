//! ローカル接続だけを受け付ける。書き込み系は画面に埋め込んだトークンも要求する。

use axum::extract::{Request, State};
use axum::http::{header, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use subtle::ConstantTimeEq;

use crate::AppState;

const ALLOWED_HOSTS: [&str; 2] = ["127.0.0.1", "localhost"];

fn forbidden(message: &str) -> Response {
    (StatusCode::FORBIDDEN, Json(json!({ "error": message }))).into_response()
}

fn host_allowed(request: &Request) -> bool {
    let host = request
        .headers()
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");
    let name = host.split(':').next().unwrap_or("");
    ALLOWED_HOSTS.contains(&name)
}

fn token_matches(request: &Request, token: &str) -> bool {
    let given = request
        .headers()
        .get("x-easy-token")
        .map_or(&[][..], |v| v.as_bytes());
    bool::from(given.ct_eq(token.as_bytes()))
}

/// DNS リバインディング対策（`Host` の検証）と、GET/HEAD 以外のトークン検証（定数時間比較）。
pub async fn local_only(State(state): State<AppState>, request: Request, next: Next) -> Response {
    if !host_allowed(&request) {
        return forbidden("ローカル接続のみ利用できます。");
    }
    let reads = matches!(*request.method(), Method::GET | Method::HEAD);
    if !reads && !token_matches(&request, &state.token) {
        return forbidden("画面を再読み込みしてください。");
    }
    next.run(request).await
}
