//! モデルの取得に関する API の応答（確認の要求）。

use serde_json::{json, Value};

use crate::models::display_name;
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
