//! 話者の登録・名前変更・削除のルート（`easy_web.py` の `voice` `voice_rename` `voice_delete`）。

use axum::body::Bytes;
use axum::extract::State;
use axum::Json;
use irodori_store::signature::is_falsy;
use irodori_store::StoreError;
use serde_json::{json, Map, Value};

use crate::errors::ApiError;
use crate::models::ALLOW_LIST;
use crate::routes::{parse, text, ApiResult};
use crate::AppState;

const PICK_AGAIN: &str = "生成候補を選び直してください。";
const AUDIO_MISSING: &str = "音声が見つかりません。もう一度追加してください。";

fn invalid(message: &str) -> ApiError {
    ApiError(StoreError::Invalid(message.into()))
}

fn voice_json(voice: &irodori_store::model::Voice) -> ApiResult<Value> {
    serde_json::to_value(voice).map_err(|e| ApiError(StoreError::Internal(e.to_string())))
}

/// `asset` が真なら、登録簿から実ファイルを引く。文字列でない値は「見つからない」。
fn source_of(app: &AppState, data: &Map<String, Value>) -> ApiResult<Option<std::path::PathBuf>> {
    match data.get("asset") {
        None => Ok(None),
        Some(v) if is_falsy(v) => Ok(None),
        Some(Value::String(key)) => Ok(Some(app.assets.source(key)?)),
        Some(_) => Err(invalid(AUDIO_MISSING)),
    }
}

pub async fn voice(State(app): State<AppState>, body: Bytes) -> ApiResult<Json<Value>> {
    let data = parse(&body)?;
    let source = source_of(&app, &data)?;
    // 試作ジョブの候補を採用する経路（`job`）。ジョブが無い・候補でなければ、選び直しを促す。
    if data.get("job").is_some_and(|j| !is_falsy(j)) {
        return Err(invalid(PICK_AGAIN));
    }
    let saved = app
        .store
        .save_voice(&data, source.as_deref(), &ALLOW_LIST)?;
    Ok(Json(
        json!({"message": "話者を登録しました。", "voice": voice_json(&saved)?}),
    ))
}

pub async fn rename(State(app): State<AppState>, body: Bytes) -> ApiResult<Json<Value>> {
    let data = parse(&body)?;
    let renamed = app.store.rename_voice(&data)?;
    Ok(Json(
        json!({"message": "名前を変更しました。", "voice": voice_json(&renamed)?}),
    ))
}

pub async fn delete(State(app): State<AppState>, body: Bytes) -> ApiResult<Json<Value>> {
    let data = parse(&body)?;
    app.store.delete_voice(text(&data, "id")?)?;
    Ok(Json(json!({"message": "登録を削除しました。"})))
}
