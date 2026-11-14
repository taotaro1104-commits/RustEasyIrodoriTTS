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

/// `Host` ヘッダ（小文字にしたもの）。`名前` か `名前:ポート`（ポートは数字）だけを受け付ける。
fn checked_host(request: &Request) -> Option<String> {
    let host = request.headers().get(header::HOST)?.to_str().ok()?;
    let host = host.to_ascii_lowercase();
    let mut parts = host.split(':');
    let name = parts.next()?;
    let port_ok = match parts.next() {
        None => true,
        Some(port) => {
            !port.is_empty()
                && port.bytes().all(|b| b.is_ascii_digit())
                && port.parse::<u16>().is_ok()
        }
    };
    (port_ok && parts.next().is_none() && ALLOWED_HOSTS.contains(&name)).then_some(host)
}

/// 書き込み系の `Origin` は、あるなら画面と同じオリジン（`http://<Host>`）だけ。
fn origin_matches(request: &Request, host: &str) -> bool {
    match request.headers().get(header::ORIGIN) {
        None => true,
        Some(origin) => origin
            .to_str()
            .is_ok_and(|o| o.eq_ignore_ascii_case(&format!("http://{host}"))),
    }
}

fn token_matches(request: &Request, token: &str) -> bool {
    let given = request
        .headers()
        .get("x-easy-token")
        .map_or(&[][..], |v| v.as_bytes());
    bool::from(given.ct_eq(token.as_bytes()))
}

/// DNS リバインディング対策（`Host` の検証）と、GET/HEAD 以外の `Origin` の検証・トークン検証（定数時間比較）。
pub async fn local_only(State(state): State<AppState>, request: Request, next: Next) -> Response {
    let Some(host) = checked_host(&request) else {
        return forbidden("ローカル接続のみ利用できます。");
    };
    let reads = matches!(*request.method(), Method::GET | Method::HEAD);
    if !reads && !origin_matches(&request, &host) {
        return forbidden("ローカル接続のみ利用できます。");
    }
    if !reads && !token_matches(&request, &state.token) {
        return forbidden("画面を再読み込みしてください。");
    }
    next.run(request).await
}
