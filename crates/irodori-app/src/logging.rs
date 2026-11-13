//! ログ：`logs/editor.log` にアクセスと失敗を残す。成功した状態取得（画面が 2 秒ごとに行う）は記録しない。

use std::fs::OpenOptions;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{anyhow, Context, Result};
use axum::extract::Request;
use axum::http::{Method, StatusCode};
use axum::middleware::Next;
use axum::response::Response;

/// 成功した `GET /api/state` 以外は記録する。
pub fn should_log(method: &Method, path: &str, status: StatusCode) -> bool {
    !(*method == Method::GET && path == "/api/state" && status == StatusCode::OK)
}

pub async fn access_log(request: Request, next: Next) -> Response {
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let response = next.run(request).await;
    if should_log(&method, &path, response.status()) {
        tracing::info!("{method} {path} -> {}", response.status().as_u16());
    }
    response
}

/// `logs/editor.log` へ追記する形でログを始める。
pub fn init_file_logging(root: &Path) -> Result<()> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.join("logs/editor.log"))
        .context("logs/editor.log を開けません。")?;
    tracing_subscriber::fmt()
        .with_writer(Mutex::new(file))
        .with_ansi(false)
        .with_max_level(tracing::Level::INFO)
        .try_init()
        .map_err(|e| anyhow!("ログを始められません: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{Method, StatusCode};

    #[test]
    fn successful_state_polling_is_not_logged() {
        assert!(!should_log(&Method::GET, "/api/state", StatusCode::OK));
    }

    #[test]
    fn failures_and_other_requests_are_logged() {
        assert!(should_log(
            &Method::GET,
            "/api/state",
            StatusCode::BAD_REQUEST
        ));
        assert!(should_log(
            &Method::GET,
            "/api/state",
            StatusCode::INTERNAL_SERVER_ERROR
        ));
        assert!(should_log(&Method::POST, "/api/state", StatusCode::OK));
        assert!(should_log(&Method::GET, "/api/meta", StatusCode::OK));
        assert!(should_log(&Method::POST, "/api/edit", StatusCode::OK));
    }
}
