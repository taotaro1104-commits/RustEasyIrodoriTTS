//! `StoreError` を、v2.0 と同じ JSON（`{"error": "…"}`）と状態コードに変える。

use std::any::Any;

use axum::extract::Request;
use axum::http::{header, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use irodori_store::StoreError;
use serde_json::json;

const INTERNAL_MESSAGE: &str =
    "処理に失敗しました。入力内容を保持したまま、logs/editor.logを確認してください。";

const TOO_LARGE_MESSAGE: &str = "リクエストが大きすぎます。";

/// 本文が上限（2 MB。アップロードのルートだけは別）を超えたとき、axum は文字だけの 413 を返す。画面が読める JSON にそろえる。
pub async fn json_payload_too_large(request: Request, next: Next) -> Response {
    let response = next.run(request).await;
    let is_json = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("application/json"));
    if response.status() == StatusCode::PAYLOAD_TOO_LARGE && !is_json {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(json!({ "error": TOO_LARGE_MESSAGE })),
        )
            .into_response();
    }
    response
}

/// ハンドラが panic したときの応答。詳細はログにだけ残し、画面には決まった文を返す。
pub fn panic_response(panic: Box<dyn Any + Send + 'static>) -> Response {
    let detail = panic
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".into());
    tracing::error!("Editor handler panicked: {detail}");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({ "error": INTERNAL_MESSAGE })),
    )
        .into_response()
}

#[derive(Debug)]
pub struct ApiError(pub StoreError);

impl From<StoreError> for ApiError {
    fn from(e: StoreError) -> Self {
        ApiError(e)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status =
            StatusCode::from_u16(self.0.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let message = match &self.0 {
            // 想定外の失敗は詳細をログにだけ残し、画面には決まった文を返す。
            StoreError::Internal(detail) => {
                tracing::error!("Editor operation failed: {detail}");
                INTERNAL_MESSAGE.to_string()
            }
            other => other.message().to_string(),
        };
        (status, Json(json!({ "error": message }))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    async fn render(e: StoreError) -> (StatusCode, String) {
        let res = ApiError(e).into_response();
        let status = res.status();
        let body = to_bytes(res.into_body(), 1024 * 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        (status, v["error"].as_str().unwrap().to_string())
    }

    #[tokio::test]
    async fn store_errors_map_to_status_and_japanese_json() {
        assert_eq!(
            render(StoreError::Invalid("入力が不正です。".into())).await,
            (StatusCode::BAD_REQUEST, "入力が不正です。".into())
        );
        assert_eq!(
            render(StoreError::Conflict("更新されています。".into())).await,
            (StatusCode::CONFLICT, "更新されています。".into())
        );
        assert_eq!(
            render(StoreError::Io("読み書きできません。".into())).await,
            (StatusCode::BAD_REQUEST, "読み書きできません。".into())
        );
    }

    #[tokio::test]
    async fn internal_errors_hide_details_from_the_screen() {
        let (status, message) =
            render(StoreError::Internal("secret path C:\\x failed".into())).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(message, INTERNAL_MESSAGE);
        assert!(!message.contains("secret"));
    }
}
