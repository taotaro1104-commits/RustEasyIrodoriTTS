//! 話者まわりの HTTP 経路（取り込み・登録・名前変更・削除・試作）。

mod common;

use axum::http::StatusCode;
use common::*;
use serde_json::{json, Value};

fn uploads_dir(f: &Fixture) -> std::path::PathBuf {
    f.dir.path().join("temp/uploads")
}

fn leftovers(f: &Fixture) -> Vec<String> {
    std::fs::read_dir(uploads_dir(f))
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default()
}

async fn upload(f: &Fixture, filename: &str, bytes: &[u8]) -> Reply {
    post_file(&f.app, "/api/upload", "file", filename, bytes, true).await
}

// ---- /api/upload ----

#[tokio::test]
async fn upload_returns_an_asset_and_seconds() {
    let f = fixture();
    let r = upload(&f, "声.wav", &wav_bytes(48000, 2, 2.0)).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.error());
    let body = r.json();
    assert!(
        (body["seconds"].as_f64().unwrap() - 2.0).abs() < 0.05,
        "{body}"
    );
    let key = body["asset"].as_str().unwrap();
    let audio = get(&f.app, &format!("/audio/{key}")).await;
    assert_eq!(audio.status, StatusCode::OK);
    let reader = hound::WavReader::new(std::io::Cursor::new(audio.body)).unwrap();
    let spec = reader.spec();
    assert_eq!(
        (spec.sample_rate, spec.channels, spec.bits_per_sample),
        (24000, 1, 24)
    );
    assert!((reader.duration() as f64 / 24000.0 - 2.0).abs() < 0.05);
    assert_eq!(
        leftovers(&f).len(),
        1,
        "only the normalized file stays: {:?}",
        leftovers(&f)
    );
}

#[tokio::test]
async fn upload_without_the_file_field_is_400() {
    let f = fixture();
    let r = post_file(
        &f.app,
        "/api/upload",
        "other",
        "a.wav",
        &wav_bytes(24000, 1, 2.0),
        true,
    )
    .await;
    assert_eq!(
        (r.status, r.error()),
        (
            StatusCode::BAD_REQUEST,
            "音声ファイルを選んでください。".into()
        )
    );
    assert!(leftovers(&f).is_empty());
}

#[tokio::test]
async fn upload_rejects_unknown_extension() {
    let f = fixture();
    for name in ["a.txt", "a", "a.exe", ".wav.txt"] {
        let r = upload(&f, name, &wav_bytes(24000, 1, 2.0)).await;
        assert_eq!(
            (r.status, r.error()),
            (
                StatusCode::BAD_REQUEST,
                "音声ファイルを選んでください。".into()
            ),
            "{name}"
        );
    }
    assert!(leftovers(&f).is_empty());
}

#[tokio::test]
async fn upload_of_webm_is_501_with_the_message() {
    let f = fixture();
    let r = upload(&f, "clip.WEBM", b"\x1a\x45\xdf\xa3").await;
    assert_eq!(r.status, StatusCode::NOT_IMPLEMENTED);
    assert!(r.error().contains("webm"), "{}", r.error());
    assert!(leftovers(&f).is_empty());
}

#[tokio::test]
async fn upload_rejects_garbage_with_a_wav_extension() {
    let f = fixture();
    let noise: Vec<u8> = (0..20000u32)
        .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8)
        .collect();
    let r = upload(&f, "a.wav", &noise).await;
    assert_eq!(
        (r.status, r.error()),
        (
            StatusCode::BAD_REQUEST,
            "音声を読み込めません。形式を確認してください。".into()
        )
    );
    assert!(leftovers(&f).is_empty(), "{:?}", leftovers(&f));
}

#[tokio::test]
async fn upload_rejects_empty_file() {
    let f = fixture();
    let r = upload(&f, "a.wav", b"").await;
    assert_eq!(
        (r.status, r.error()),
        (
            StatusCode::BAD_REQUEST,
            "音声を読み込めません。形式を確認してください。".into()
        )
    );
    assert!(leftovers(&f).is_empty());
}

#[tokio::test]
async fn upload_rejects_half_second_and_121_second_audio() {
    let f = fixture();
    for seconds in [0.5, 121.0, 200.0] {
        let r = upload(&f, "a.wav", &wav_bytes(8000, 1, seconds)).await;
        assert_eq!(
            (r.status, r.error()),
            (
                StatusCode::BAD_REQUEST,
                "音声は1～120秒にしてください。".into()
            ),
            "{seconds}"
        );
    }
    assert!(leftovers(&f).is_empty(), "{:?}", leftovers(&f));
    let ok = upload(&f, "a.wav", &wav_bytes(8000, 1, 120.0)).await;
    assert_eq!(ok.status, StatusCode::OK, "{}", ok.error());
}

#[tokio::test]
async fn upload_requires_the_token() {
    let f = fixture();
    let r = post_file(
        &f.app,
        "/api/upload",
        "file",
        "a.wav",
        &wav_bytes(24000, 1, 2.0),
        false,
    )
    .await;
    assert_eq!(r.status, StatusCode::FORBIDDEN);
    assert!(leftovers(&f).is_empty());
}

#[tokio::test]
async fn upload_does_not_use_the_client_file_name_for_the_path() {
    let f = fixture();
    let r = upload(&f, "..\\..\\evil.wav", &wav_bytes(24000, 1, 2.0)).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.error());
    assert!(!f.dir.path().join("evil.wav").exists());
    assert!(!f.dir.path().parent().unwrap().join("evil.wav").exists());
    let names = leftovers(&f);
    assert!(names.iter().all(|n| !n.contains("evil")), "{names:?}");
}

// ---- /api/voice ----

const SMALL: &str = "Aratako/Irodori-TTS-v4.1-Small";
const REGISTRATION_SAMPLE: &str = "おはようございます。今日は空が晴れて気持ちのよい朝ですね。窓を開けると涼しい風が入ってきました。温かいお茶を飲んでゆっくり出かける準備をしましょう。";

async fn upload_asset(f: &Fixture, seconds: f64) -> String {
    let r = upload(f, "a.wav", &wav_bytes(24000, 1, seconds)).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.error());
    r.json()["asset"].as_str().unwrap().to_string()
}

async fn register(f: &Fixture, name: &str) -> Value {
    let asset = upload_asset(f, 2.0).await;
    let r = post(
        &f.app,
        "/api/voice",
        json!({"name": name, "model": MF, "caption": "説明", "asset": asset}),
    )
    .await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.error());
    r.json()["voice"].clone()
}

async fn state(f: &Fixture) -> Value {
    get(&f.app, "/api/state").await.json()
}

#[tokio::test]
async fn register_voice_from_an_uploaded_asset() {
    let f = fixture();
    let asset = upload_asset(&f, 2.0).await;
    let r = post(
        &f.app,
        "/api/voice",
        json!({"name": "  話者A ", "model": MF, "caption": "落ち着いた声", "asset": asset}),
    )
    .await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.error());
    let body = r.json();
    assert_eq!(body["message"], "話者を登録しました。");
    let voice = &body["voice"];
    assert_eq!(
        (
            voice["name"].as_str(),
            voice["revision"].as_i64(),
            voice["sound_revision"].as_i64()
        ),
        (Some("話者A"), Some(1), Some(1))
    );
    let id = voice["id"].as_str().unwrap();
    let audio = voice["audio"].as_str().unwrap();
    assert!(
        audio.starts_with(&format!("voices/{id}/")) && audio.ends_with(".wav"),
        "{audio}"
    );
    let reader = hound::WavReader::open(f.dir.path().join(audio)).unwrap();
    assert_eq!(reader.spec().bits_per_sample, 24);
    let index: Value =
        serde_json::from_slice(&std::fs::read(f.dir.path().join("voices/index.json")).unwrap())
            .unwrap();
    assert_eq!(index.as_array().unwrap().len(), 1);
    let s = state(&f).await;
    let listed = &s["voices"][0];
    assert_eq!(listed["id"], voice["id"]);
    let served = get(
        &f.app,
        &format!("/audio/{}", listed["asset"].as_str().unwrap()),
    )
    .await;
    assert_eq!(served.status, StatusCode::OK);
}

#[tokio::test]
async fn register_without_audio_is_400() {
    let f = fixture();
    let r = post(&f.app, "/api/voice", json!({"name": "A", "model": MF})).await;
    assert_eq!(
        (r.status, r.error()),
        (
            StatusCode::BAD_REQUEST,
            "参照音声を追加するか、生成した候補を選んでください。".into()
        )
    );
}

#[tokio::test]
async fn register_with_bad_name_or_model_is_400() {
    let f = fixture();
    let asset = upload_asset(&f, 2.0).await;
    for (values, want) in [
        (
            json!({"name": "", "model": MF, "asset": asset}),
            "話者名を1～80文字で入力してください。",
        ),
        (
            json!({"name": "あ".repeat(81), "model": MF, "asset": asset}),
            "話者名を1～80文字で入力してください。",
        ),
        (
            json!({"name": "A", "model": "evil/model", "asset": asset}),
            "使用するモデルを選んでください。",
        ),
        (
            json!({"name": "A", "asset": asset}),
            "使用するモデルを選んでください。",
        ),
        (
            json!({"name": 5, "model": MF, "asset": asset}),
            "話者名を1～80文字で入力してください。",
        ),
    ] {
        let r = post(&f.app, "/api/voice", values.clone()).await;
        assert_eq!(
            (r.status, r.error().as_str()),
            (StatusCode::BAD_REQUEST, want),
            "{values}"
        );
    }
    assert!(state(&f).await["voices"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn update_without_new_audio_keeps_the_wav_and_bumps_both_revisions() {
    let f = fixture();
    let v = register(&f, "A").await;
    let r = post(
        &f.app,
        "/api/voice",
        json!({"id": v["id"], "revision": v["revision"], "name": "A", "model": SMALL, "caption": "新しい説明"}),
    )
    .await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.error());
    let updated = r.json()["voice"].clone();
    assert_eq!(updated["audio"], v["audio"]);
    assert_eq!(
        (
            updated["revision"].as_i64(),
            updated["sound_revision"].as_i64()
        ),
        (Some(2), Some(2))
    );
    assert_eq!(
        (updated["model"].as_str(), updated["caption"].as_str()),
        (Some(SMALL), Some("新しい説明"))
    );
}

#[tokio::test]
async fn stale_revision_is_400() {
    let f = fixture();
    let v = register(&f, "A").await;
    let r = post(
        &f.app,
        "/api/voice",
        json!({"id": v["id"], "revision": 0, "name": "A", "model": MF}),
    )
    .await;
    assert_eq!(
        (r.status, r.error()),
        (
            StatusCode::BAD_REQUEST,
            "話者が更新されています。一覧から読み直してください。".into()
        )
    );
}

#[tokio::test]
async fn rename_changes_only_the_name_and_keeps_sound_revision() {
    let f = fixture();
    let v = register(&f, "A").await;
    let r = post(
        &f.app,
        "/api/voice/rename",
        json!({"id": v["id"], "revision": v["revision"], "name": "新しい名前"}),
    )
    .await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.error());
    let body = r.json();
    assert_eq!(body["message"], "名前を変更しました。");
    let renamed = &body["voice"];
    assert_eq!(
        (
            renamed["name"].as_str(),
            renamed["revision"].as_i64(),
            renamed["sound_revision"].as_i64()
        ),
        (Some("新しい名前"), Some(2), Some(1))
    );
    assert_eq!(renamed["audio"], v["audio"]);
    let stale = post(
        &f.app,
        "/api/voice/rename",
        json!({"id": v["id"], "revision": 1, "name": "また"}),
    )
    .await;
    assert_eq!(stale.status, StatusCode::BAD_REQUEST);
    let missing = post(
        &f.app,
        "/api/voice/rename",
        json!({"id": "f".repeat(32), "revision": 1, "name": "x"}),
    )
    .await;
    assert_eq!(
        (missing.status, missing.error()),
        (
            StatusCode::BAD_REQUEST,
            "話者が見つかりません。選び直してください。".into()
        )
    );
}

#[tokio::test]
async fn rename_to_the_same_name_changes_nothing() {
    let f = fixture();
    let v = register(&f, "A").await;
    let before = std::fs::read(f.dir.path().join("voices/index.json")).unwrap();
    let r = post(
        &f.app,
        "/api/voice/rename",
        json!({"id": v["id"], "revision": v["revision"], "name": " A "}),
    )
    .await;
    assert_eq!(r.status, StatusCode::OK);
    assert_eq!(r.json()["voice"]["revision"], v["revision"]);
    assert_eq!(
        std::fs::read(f.dir.path().join("voices/index.json")).unwrap(),
        before
    );
}

/// 1 枚目のカードに文章と話者を入れて、(pid, revision, card id) を返す。
async fn assign_voice(f: &Fixture, voice_id: &Value) -> (String, i64, String) {
    let s = state(f).await;
    let pid = s["project"]["id"].as_str().unwrap().to_string();
    let rev = s["project"]["revision"].as_i64().unwrap();
    let cid = s["project"]["cards"][0]["id"].as_str().unwrap().to_string();
    let e = post(
        &f.app,
        "/api/edit",
        json!({"pid": pid, "revision": rev, "values": {"cards": [{"id": cid, "text": "こんにちは。", "voice_id": voice_id}]}}),
    )
    .await;
    assert_eq!(e.status, StatusCode::OK, "{}", e.error());
    (pid, e.json()["project"]["revision"].as_i64().unwrap(), cid)
}

#[tokio::test]
async fn delete_removes_the_voice_and_cards_ask_for_a_new_one() {
    let f = fixture();
    let v = register(&f, "A").await;
    assign_voice(&f, &v["id"]).await;
    let wav = f.dir.path().join(v["audio"].as_str().unwrap());
    let r = post(&f.app, "/api/voice/delete", json!({"id": v["id"]})).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.error());
    assert_eq!(r.json()["message"], "登録を削除しました。");
    let after = state(&f).await;
    assert!(after["voices"].as_array().unwrap().is_empty());
    assert_eq!(
        after["project"]["cards"][0]["status"],
        "話者を選び直してください"
    );
    assert!(wav.is_file(), "the reference file is kept");
    let again = post(&f.app, "/api/voice/delete", json!({"id": v["id"]})).await;
    assert_eq!(again.status, StatusCode::OK, "deleting twice is fine");
}

async fn wait_idle(f: &Fixture) {
    for _ in 0..2000 {
        let s = state(f).await;
        let busy = s["jobs"]
            .as_array()
            .unwrap()
            .iter()
            .any(|j| ["待機中", "生成中"].contains(&j["status"].as_str().unwrap()));
        if !busy {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("jobs did not finish");
}

#[tokio::test]
async fn registering_again_keeps_the_old_wav_and_old_results_need_regeneration() {
    let f = fixture();
    let v = register(&f, "A").await;
    let (pid, rev, _) = assign_voice(&f, &v["id"]).await;
    let g = post(
        &f.app,
        "/api/generate",
        json!({"pid": pid, "revision": rev}),
    )
    .await;
    assert_eq!(g.status, StatusCode::OK, "{}", g.error());
    wait_idle(&f).await;
    assert_eq!(state(&f).await["project"]["cards"][0]["status"], "完了");
    let asset = upload_asset(&f, 3.0).await;
    let r = post(
        &f.app,
        "/api/voice",
        json!({"id": v["id"], "revision": v["revision"], "name": "A", "model": MF, "asset": asset}),
    )
    .await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.error());
    let new = r.json()["voice"].clone();
    assert_ne!(new["audio"], v["audio"]);
    assert!(
        f.dir.path().join(v["audio"].as_str().unwrap()).is_file(),
        "the old version stays for queued jobs"
    );
    assert_eq!(
        state(&f).await["project"]["cards"][0]["status"],
        "再生成が必要"
    );
}

#[tokio::test]
async fn voice_routes_require_the_token() {
    let f = fixture();
    for uri in ["/api/voice", "/api/voice/rename", "/api/voice/delete"] {
        let r = post_raw_no_token(&f.app, uri).await;
        assert_eq!(r.status, StatusCode::FORBIDDEN, "{uri}");
    }
}

#[tokio::test]
async fn unknown_asset_key_is_400() {
    let f = fixture();
    let r = post(
        &f.app,
        "/api/voice",
        json!({"name": "A", "model": MF, "asset": "0".repeat(64)}),
    )
    .await;
    assert_eq!(
        (r.status, r.error()),
        (
            StatusCode::BAD_REQUEST,
            "音声が見つかりません。もう一度追加してください。".into()
        )
    );
    let r = post(
        &f.app,
        "/api/voice",
        json!({"name": "A", "model": MF, "asset": 7}),
    )
    .await;
    assert_eq!(r.status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn adopting_a_job_candidate_is_refused_for_an_unknown_job() {
    let f = fixture();
    let asset = upload_asset(&f, 2.0).await;
    let r = post(
        &f.app,
        "/api/voice",
        json!({"name": "A", "model": MF, "asset": asset, "job": "nope"}),
    )
    .await;
    assert_eq!(
        (r.status, r.error()),
        (
            StatusCode::BAD_REQUEST,
            "生成候補を選び直してください。".into()
        )
    );
}

// ---- /api/design・/api/design-registration と、候補の採用 ----

use std::time::Duration;

fn design_values() -> Value {
    json!({"model": MF, "mode": "design", "caption": "明るい声", "text": "こんにちは。", "candidates": 2, "generation": {"steps": "", "seed": ""}})
}

async fn start_design(f: &Fixture, uri: &str, values: Value) -> String {
    let r = post(&f.app, uri, values).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.error());
    let body = r.json();
    assert_eq!(
        body.as_object().unwrap().len(),
        1,
        "v2.0 returns only the job id: {body}"
    );
    body["job"].as_str().unwrap().to_string()
}

fn find_job(s: &Value, id: &str) -> Value {
    s["jobs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|j| j["id"] == id)
        .cloned()
        .unwrap_or_else(|| panic!("job {id} not listed: {}", s["jobs"]))
}

#[tokio::test]
async fn design_returns_a_job_id_and_the_state_lists_the_design_job() {
    let f = fixture();
    let id = start_design(&f, "/api/design", design_values()).await;
    wait_idle(&f).await;
    let job = find_job(&state(&f).await, &id);
    assert_eq!(
        (job["kind"].as_str(), job["status"].as_str()),
        (Some("design"), Some("完了"))
    );
    assert_eq!(job["assets"].as_array().unwrap().len(), 2);
    assert!(job["project"].is_null());
    let first = job["assets"][0].as_str().unwrap();
    assert_eq!(
        get(&f.app, &format!("/audio/{first}")).await.status,
        StatusCode::OK
    );
}

#[tokio::test]
async fn a_design_with_an_uploaded_reference_passes_it_to_the_engine() {
    let f = fixture();
    let asset = upload_asset(&f, 2.0).await;
    let mut v = design_values();
    v["mode"] = json!("clone");
    v["references"] = json!([asset]);
    start_design(&f, "/api/design", v).await;
    wait_idle(&f).await;
    let seen = f.synth.last_job().unwrap();
    assert_eq!((seen.references.len(), seen.caption.as_str()), (1, ""));
}

#[tokio::test]
async fn a_registered_candidate_becomes_a_voice_with_its_creation_record() {
    let f = fixture_with_voice(|s| s.with_clip_seconds(2.0));
    let mut v = design_values();
    v.as_object_mut().unwrap().remove("text");
    let id = start_design(&f, "/api/design-registration", v).await;
    wait_idle(&f).await;
    let job = find_job(&state(&f).await, &id);
    let asset = job["assets"][1].as_str().unwrap();
    let r = post(
        &f.app,
        "/api/voice",
        json!({"name": "試作の声", "model": MF, "caption": "明るい声", "asset": asset, "job": id, "creation": {"forged": true}}),
    )
    .await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.error());
    let voice = r.json()["voice"].clone();
    let creation = &voice["creation"];
    assert_eq!(creation["text"], REGISTRATION_SAMPLE);
    assert!(
        creation.get("forged").is_none(),
        "the job's record replaces anything the client sent"
    );
    assert!(creation["paths"].as_array().unwrap().len() == 2 && creation["seed"].is_number());
    let reader =
        hound::WavReader::open(f.dir.path().join(voice["audio"].as_str().unwrap())).unwrap();
    assert_eq!(reader.spec().bits_per_sample, 24);
    let index: Value =
        serde_json::from_slice(&std::fs::read(f.dir.path().join("voices/index.json")).unwrap())
            .unwrap();
    let saved = index
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["id"] == voice["id"])
        .unwrap();
    assert_eq!(saved["creation"], *creation);
}

#[tokio::test]
async fn a_candidate_from_another_job_is_rejected() {
    let f = fixture();
    let first = start_design(&f, "/api/design", design_values()).await;
    wait_idle(&f).await;
    let second = start_design(&f, "/api/design", design_values()).await;
    wait_idle(&f).await;
    let s = state(&f).await;
    let asset = find_job(&s, &first)["assets"][0]
        .as_str()
        .unwrap()
        .to_string();
    let r = post(
        &f.app,
        "/api/voice",
        json!({"name": "A", "model": MF, "asset": asset, "job": second}),
    )
    .await;
    assert_eq!(
        (r.status, r.error()),
        (
            StatusCode::BAD_REQUEST,
            "生成候補を選び直してください。".into()
        )
    );
    assert!(state(&f).await["voices"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn an_asset_that_is_not_a_candidate_of_the_job_is_rejected() {
    let f = fixture();
    let id = start_design(&f, "/api/design", design_values()).await;
    wait_idle(&f).await;
    let uploaded = upload_asset(&f, 2.0).await;
    let r = post(
        &f.app,
        "/api/voice",
        json!({"name": "A", "model": MF, "asset": uploaded, "job": id}),
    )
    .await;
    assert_eq!(r.status, StatusCode::BAD_REQUEST);
    assert_eq!(r.error(), "生成候補を選び直してください。");
    // job だけで asset が無いのも同じ。
    let r = post(
        &f.app,
        "/api/voice",
        json!({"name": "A", "model": MF, "job": id}),
    )
    .await;
    assert_eq!(r.error(), "生成候補を選び直してください。");
}

fn slow_fixture() -> Fixture {
    fixture_with_voice(|s| s.with_steps(60).with_step_delay(Duration::from_millis(25)))
}

#[tokio::test]
async fn an_unfinished_job_cannot_be_adopted() {
    let f = slow_fixture();
    let id = start_design(&f, "/api/design", design_values()).await;
    let uploaded = upload_asset(&f, 2.0).await;
    let r = post(
        &f.app,
        "/api/voice",
        json!({"name": "A", "model": MF, "asset": uploaded, "job": id}),
    )
    .await;
    assert_eq!(r.error(), "生成候補を選び直してください。");
    post(&f.app, "/api/stop", json!({})).await;
    wait_idle(&f).await;
}

#[tokio::test]
async fn design_validation_errors_are_400() {
    let f = fixture();
    let mut long = design_values();
    long["text"] = json!("これは試作には長すぎる文章です。".repeat(40));
    let r = post(&f.app, "/api/design", long).await;
    assert_eq!(
        (r.status, r.error()),
        (
            StatusCode::BAD_REQUEST,
            "声の試作には推定25秒以内の短い文章を入力してください。".into()
        )
    );
    let mut clone = design_values();
    clone["mode"] = json!("clone");
    let r = post(&f.app, "/api/design", clone).await;
    assert_eq!(
        (r.status, r.error()),
        (
            StatusCode::BAD_REQUEST,
            "お手本の音声を追加してください。".into()
        )
    );
    let mut unknown = design_values();
    unknown["references"] = json!(["0".repeat(64)]);
    unknown["mode"] = json!("both");
    let r = post(&f.app, "/api/design", unknown).await;
    assert_eq!(
        r.error(),
        "音声が見つかりません。もう一度追加してください。"
    );
    let r = post(&f.app, "/api/design", json!([1, 2])).await;
    assert_eq!(r.status, StatusCode::BAD_REQUEST);
    assert!(state(&f).await["jobs"].as_array().unwrap().is_empty());
    let missing = fixture_with_voice(|s| s.missing_models(&[MF]));
    let r = post(&missing.app, "/api/design", design_values()).await;
    assert_eq!(
        (r.status, r.error()),
        (
            StatusCode::BAD_REQUEST,
            "モデルがまだ取得されていません。".into()
        )
    );
}

#[tokio::test]
async fn a_design_is_refused_while_a_generation_runs() {
    let f = slow_fixture();
    let s = state(&f).await;
    let (pid, rev) = (
        s["project"]["id"].as_str().unwrap().to_string(),
        s["project"]["revision"].as_i64().unwrap(),
    );
    let cid = s["project"]["cards"][0]["id"].as_str().unwrap().to_string();
    let e = post(&f.app, "/api/edit", json!({"pid": pid, "revision": rev, "values": {"cards": [{"id": cid, "text": "こんにちは。", "voice_id": VOICE}]}})).await;
    let rev = e.json()["project"]["revision"].as_i64().unwrap();
    let g = post(
        &f.app,
        "/api/generate",
        json!({"pid": pid, "revision": rev}),
    )
    .await;
    assert_eq!(g.status, StatusCode::OK, "{}", g.error());
    let r = post(&f.app, "/api/design", design_values()).await;
    assert_eq!(
        (r.status, r.error()),
        (
            StatusCode::BAD_REQUEST,
            "現在の生成が終わってから実行してください。".into()
        )
    );
    post(&f.app, "/api/stop", json!({})).await;
    wait_idle(&f).await;
}

#[tokio::test]
async fn stop_cancels_a_design() {
    let f = slow_fixture();
    let id = start_design(&f, "/api/design", design_values()).await;
    tokio::time::sleep(Duration::from_millis(80)).await;
    assert_eq!(
        post(&f.app, "/api/stop", json!({})).await.status,
        StatusCode::OK
    );
    wait_idle(&f).await;
    let job = find_job(&state(&f).await, &id);
    assert_eq!(job["status"], "停止");
    assert!(job["assets"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn design_routes_require_the_token() {
    let f = fixture();
    for uri in ["/api/design", "/api/design-registration"] {
        assert_eq!(
            post_raw_no_token(&f.app, uri).await.status,
            StatusCode::FORBIDDEN,
            "{uri}"
        );
    }
}
