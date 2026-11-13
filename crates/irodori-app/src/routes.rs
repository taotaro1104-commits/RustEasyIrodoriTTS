//! M2a の API ルート。応答の形・メッセージ・状態コードは v2.0（`easy_web.py`）に合わせる。
//! 声の登録・生成・書き出しなど後続の段階（M2b/M2c）のルートは、画面が落ちないよう 501 を返す。

use crate::jobs::SubmitOutcome;
use axum::body::{Body, Bytes};
use axum::extract::{Path, Query, Request, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use irodori_store::dictionary::{self, Dictionary};
use irodori_store::naming::safe_name;
use irodori_store::ops;
use irodori_store::state::build_state;
use irodori_store::StoreError;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tower::ServiceExt;
use tower_http::services::ServeFile;

use crate::errors::ApiError;
use crate::AppState;

type ApiResult<T> = Result<T, ApiError>;

const NOT_YET: &str = "未対応です（この版ではまだ使えません）。";

/// M2b/M2c で実装するルート（POST）。
const UNIMPLEMENTED_POST: [&str; 16] = [
    "/api/upload",
    "/api/choose-folder",
    "/api/cleanup-preview",
    "/api/cleanup",
    "/api/input-data/save",
    "/api/input-data/load",
    "/api/voice",
    "/api/voice/delete",
    "/api/voice/rename",
    "/api/design-registration",
    "/api/design",
    "/api/audio-list",
    "/api/bundle",
    "/api/save-audio",
    "/api/export",
    "/api/open-output",
];

fn bad(message: &str) -> ApiError {
    ApiError(StoreError::Invalid(message.into()))
}

/// 本文を JSON のオブジェクトとして読む（Content-Type は問わない）。
fn parse(body: &Bytes) -> ApiResult<Map<String, Value>> {
    match serde_json::from_slice::<Value>(body) {
        Ok(Value::Object(m)) => Ok(m),
        _ => Err(bad(
            "リクエストの形式が不正です。画面を再読み込みしてください。",
        )),
    }
}

fn text<'a>(data: &'a Map<String, Value>, key: &str) -> ApiResult<&'a str> {
    data.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| bad("リクエストの内容が不足しています。画面を再読み込みしてください。"))
}

fn revision(data: &Map<String, Value>) -> ApiResult<i64> {
    data.get("revision")
        .and_then(|v| {
            v.as_i64()
                .or_else(|| v.as_f64().filter(|f| f.fract() == 0.0).map(|f| f as i64))
        })
        .ok_or_else(|| bad("リクエストの内容が不足しています。画面を再読み込みしてください。"))
}

/// `/api/state`：保存層の状態に、実行中のジョブと、モデルが使えるか（`installed`）を重ねる。
fn full_state(app: &AppState, pid: Option<&str>) -> ApiResult<Value> {
    let mut state = build_state(
        &app.store,
        &app.assets,
        pid,
        &app.jobs.jobs_json(),
        &app.backend,
    )?;
    if let Some(models) = state.get_mut("models").and_then(Value::as_array_mut) {
        for model in models {
            let available = model["id"]
                .as_str()
                .is_some_and(|id| app.synth.model_available(id));
            model["installed"] = json!(available);
        }
    }
    Ok(state)
}

fn state_of(app: &AppState, pid: &str) -> ApiResult<Value> {
    full_state(app, Some(pid))
}

#[derive(Deserialize)]
struct StateQuery {
    pid: Option<String>,
}

async fn state(State(app): State<AppState>, Query(q): Query<StateQuery>) -> ApiResult<Json<Value>> {
    let pid = q.pid.filter(|p| !p.is_empty());
    Ok(Json(full_state(&app, pid.as_deref())?))
}

async fn edit(State(app): State<AppState>, body: Bytes) -> ApiResult<Json<Value>> {
    let data = parse(&body)?;
    let pid = text(&data, "pid")?;
    let values = data
        .get("values")
        .ok_or_else(|| bad("リクエストの内容が不足しています。画面を再読み込みしてください。"))?;
    ops::edit(&app.store, pid, revision(&data)?, values)?;
    Ok(Json(state_of(&app, pid)?))
}

/// 分割だけが辞書を使う。辞書を使わない台本では、辞書ファイルの状態を問わない。
fn dictionary_for(app: &AppState, pid: &str, action: &str) -> ApiResult<Dictionary> {
    if action != "split" {
        return Ok(Dictionary::new());
    }
    let doc = app.store.load(Some(pid))?;
    if doc.dictionary_enabled {
        Ok(dictionary::load_lenient(app.store.root())?)
    } else {
        Ok(Dictionary::new())
    }
}

async fn action(State(app): State<AppState>, body: Bytes) -> ApiResult<Json<Value>> {
    let data = parse(&body)?;
    let (pid, act) = (text(&data, "pid")?, text(&data, "action")?);
    let rev = revision(&data)?;
    let cid = data.get("cid").and_then(Value::as_str);
    let order: Option<Vec<String>> = data
        .get("order")
        .and_then(Value::as_array)
        .and_then(|a| a.iter().map(|v| v.as_str().map(str::to_string)).collect());
    let dict = dictionary_for(&app, pid, act)?;
    // 動いている生成依頼は M2b から。ここでは無い。
    let active = app.jobs.active_card_ids(pid);
    let outcome = ops::action(
        &app.store,
        &dict,
        &active,
        pid,
        rev,
        act,
        cid,
        order.as_deref(),
    )?;
    let mut out = state_of(&app, pid)?;
    if let Some(m) = out.as_object_mut() {
        m.insert("message".into(), json!(outcome.message));
    }
    Ok(Json(out))
}

async fn project(State(app): State<AppState>, body: Bytes) -> ApiResult<Json<Value>> {
    let data = Value::Object(parse(&body)?);
    let doc = ops::new_or_load_project(&app.store, &data)?;
    Ok(Json(state_of(&app, &doc.id)?))
}

async fn recover_previous(State(app): State<AppState>, body: Bytes) -> ApiResult<Json<Value>> {
    let data = parse(&body)?;
    let pid = text(&data, "pid")?;
    ops::recover_previous(&app.store, pid, revision(&data)?, app.jobs.busy())?;
    Ok(Json(state_of(&app, pid)?))
}

async fn settings(State(app): State<AppState>, body: Bytes) -> ApiResult<Json<Value>> {
    let data = Value::Object(parse(&body)?);
    app.store.save_settings(&data)?;
    Ok(Json(
        json!({"message": "設定を保存しました。生成モード・自動起動は次回起動時に反映されます。"}),
    ))
}

async fn dictionary_route(State(app): State<AppState>, body: Bytes) -> ApiResult<Json<Value>> {
    let data = parse(&body)?;
    let word = data.get("word").and_then(Value::as_str).unwrap_or("");
    let deleting = data
        .get("delete")
        .is_some_and(|v| !irodori_store::signature::is_falsy(v));
    let message = if deleting {
        dictionary::delete(app.store.root(), word)?
    } else {
        let reading = data.get("reading").and_then(Value::as_str).unwrap_or("");
        dictionary::save(app.store.root(), word, reading)?
    };
    Ok(Json(json!({ "message": message })))
}

#[derive(Deserialize)]
struct AudioQuery {
    download: Option<String>,
    name: Option<String>,
}

/// Python の `urllib.parse.quote`（`/` はそのまま）。
fn quote(s: &str) -> String {
    s.bytes().fold(String::new(), |mut out, b| {
        if b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-' | b'~' | b'/') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
        out
    })
}

fn attachment(name: &str) -> String {
    let safe = safe_name(name);
    let quoted = quote(&safe);
    if quoted == safe {
        format!("attachment; filename=\"{safe}\"")
    } else {
        format!("attachment; filename*=utf-8''{quoted}")
    }
}

/// 登録済みの音声を配る（Range 対応。ブラウザのシークに必要）。
async fn audio(
    State(app): State<AppState>,
    Path(key): Path<String>,
    Query(q): Query<AudioQuery>,
    request: Request,
) -> ApiResult<Response> {
    let path = app.assets.source(&key)?;
    let served = ServeFile::new(path).oneshot(request).await;
    let mut response = match served {
        Ok(r) => r.map(Body::new),
        Err(never) => match never {},
    };
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static("audio/wav"));
    let download = q.download.as_deref().is_some_and(|d| {
        matches!(
            d.to_ascii_lowercase().as_str(),
            "true" | "1" | "yes" | "on" | "t" | "y"
        )
    });
    if download {
        let name = q.name.as_deref().unwrap_or("音声.wav");
        if let Ok(value) = HeaderValue::from_str(&attachment(name)) {
            response
                .headers_mut()
                .insert(header::CONTENT_DISPOSITION, value);
        }
    }
    Ok(response)
}

async fn generate(State(app): State<AppState>, body: Bytes) -> ApiResult<Json<Value>> {
    let data = parse(&body)?;
    let pid = text(&data, "pid")?;
    let flag = |key: &str| {
        data.get(key)
            .is_some_and(|v| !irodori_store::signature::is_falsy(v))
    };
    let cid = data
        .get("cid")
        .and_then(Value::as_str)
        .filter(|c| !c.is_empty());
    match app.jobs.submit(
        pid,
        revision(&data)?,
        cid,
        flag("changed"),
        flag("allow_long"),
    )? {
        SubmitOutcome::Started => {
            let mut out = state_of(&app, pid)?;
            if let Some(m) = out.as_object_mut() {
                m.insert("message".into(), json!("生成を開始しました。"));
            }
            Ok(Json(out))
        }
        SubmitOutcome::NeedsLongConfirmation { long_cards } => Ok(Json(
            json!({"needs_long_confirmation": true, "long_cards": long_cards}),
        )),
    }
}

async fn stop(State(app): State<AppState>) -> Json<Value> {
    app.jobs.stop();
    Json(json!({"message": "停止を受け付けました。処理中の1件が終わるまでお待ちください。"}))
}

async fn unload(State(app): State<AppState>) -> ApiResult<Json<Value>> {
    app.jobs.unload()?;
    Ok(Json(json!({"message": "モデルを解放しました。"})))
}

async fn not_yet() -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({ "error": NOT_YET })),
    )
        .into_response()
}

pub fn routes() -> Router<AppState> {
    let mut router = Router::new()
        .route("/api/state", get(state))
        .route("/api/edit", post(edit))
        .route("/api/action", post(action))
        .route("/api/project", post(project))
        .route("/api/settings", post(settings))
        .route("/api/dictionary", post(dictionary_route))
        .route("/api/recover-previous", post(recover_previous))
        .route("/api/generate", post(generate))
        .route("/api/stop", post(stop))
        .route("/api/unload", post(unload))
        .route("/audio/{key}", get(audio))
        .route("/bundle/{key}", get(not_yet));
    for path in UNIMPLEMENTED_POST {
        router = router.route(path, post(not_yet));
    }
    router
}
