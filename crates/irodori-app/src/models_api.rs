//! モデルの取得に関する API の応答（確認の要求）。

use axum::body::Bytes;
use axum::extract::State;
use axum::Json;
use serde_json::{json, Value};

use crate::models::display_name;
use crate::routes::{parse, ApiResult};
use crate::AppState;

/// 「このモデルを取得しますか」の確認を求める応答（`needs_long_confirmation` と同じ流儀）。
/// ジョブも状態も変えない。`download_bytes` は取得するバイト数の概算。
pub fn needs_download(app: &AppState, model: &str) -> Value {
    json!({
        "needs_model_download": true,
        "model": model,
        "name": display_name(model),
        "download_bytes": app.installer.approximate_bytes(model),
    })
}

/// `POST /api/models/download` `{model, confirm}`。`confirm` が真（`true`）のときだけ始める。
/// 画面で利用者が「取得する」と答えたことの印で、これが無ければ通信しない。
pub async fn download(State(app): State<AppState>, body: Bytes) -> ApiResult<Json<Value>> {
    let data = parse(&body)?;
    let model = data
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let confirmed = data.get("confirm") == Some(&Value::Bool(true));
    let id = app.jobs.submit_download(model, confirmed)?;
    Ok(Json(json!({ "job": id })))
}

/// `POST /api/models/cancel`。動いている取得だけを中止する（書きかけは次の取得で再開できる）。
pub async fn cancel(State(app): State<AppState>) -> Json<Value> {
    app.jobs.cancel_downloads();
    Json(json!({"message": "取得を中止します。"}))
}
