//! `voices.rs` のテスト。v2.0 の実コードが作った `voice_ops_cases.json` を同じ順に再現する。

use std::collections::HashMap;
use std::path::Path;

use serde_json::{json, Map, Value};

use super::*;
use crate::storage::read_json;
use crate::testutil::temp_root;

const SMALL: &str = "Aratako/Irodori-TTS-v4.1-Small";
const MF: &str = "Aratako/Irodori-TTS-v4.1-Small-MF";
const ALLOWED: [&str; 3] = [SMALL, MF, "phasefield-audio/Irodori-TTS-v4.1-Anime"];

/// 24 kHz・モノラル・16bit の正弦波 WAV を書く。
fn write_tone(path: &Path, seconds: f64) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for i in 0..(24000.0 * seconds) as usize {
        let v = (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 24000.0).sin() * 0.3;
        w.write_sample((v * 32767.0) as i16).unwrap();
    }
    w.finalize().unwrap();
}

fn map(v: Value) -> Map<String, Value> {
    match v {
        Value::Object(m) => m,
        _ => panic!("object expected"),
    }
}

fn open_with(root: &Path, voices: Value) -> Store {
    std::fs::write(root.join("voices/index.json"), voices.to_string()).unwrap();
    Store::open(root).unwrap()
}

fn new_voice(store: &Store, root: &Path, name: &str) -> Voice {
    let wav = root.join("temp/in.wav");
    write_tone(&wav, 2.0);
    store
        .save_voice(
            &map(json!({"name": name, "model": MF, "caption": "c"})),
            Some(&wav),
            &ALLOWED,
        )
        .unwrap()
}

fn msg(e: StoreError) -> (u16, String) {
    (e.status(), e.message().to_string())
}

#[test]
fn python_voice_operations_are_reproduced() {
    let fixture: Value =
        serde_json::from_str(include_str!("../tests/fixtures/voice_ops_cases.json")).unwrap();
    let dir = temp_root();
    let root = dir.path();
    let store = open_with(root, fixture["initial"].clone());
    let mut ids: HashMap<String, String> = HashMap::new();
    ids.insert("legacy".into(), "1".repeat(32));
    ids.insert("ghost".into(), "f".repeat(32));
    let wavs = fixture["wavs"].as_object().unwrap();
    for (n, case) in fixture["steps"].as_array().unwrap().iter().enumerate() {
        let step = &case["step"];
        let alias = step["alias"].as_str().unwrap().to_string();
        let mut values = step.get("values").cloned().map(map).unwrap_or_default();
        if let Some(id) = ids.get(&alias) {
            values.insert("id".into(), json!(id));
        }
        let current = store
            .voices()
            .into_iter()
            .find(|v| ids.get(&alias).is_some_and(|i| *i == v.id));
        match (step["revision"].as_str(), &current) {
            (Some("current"), Some(v)) => values.insert("revision".into(), json!(v.revision)),
            (Some("stale"), Some(v)) => values.insert("revision".into(), json!(v.revision - 1)),
            (Some("stale"), None) => values.insert("revision".into(), json!(0)),
            (Some("current"), None) => values.insert("revision".into(), json!(1)),
            _ => None,
        };
        let outcome: Result<()> = match step["op"].as_str().unwrap() {
            "delete" => store.delete_voice(ids.get(&alias).map_or(alias.as_str(), String::as_str)),
            "rename" => store.rename_voice(&values).map(|_| ()),
            _ => {
                let wav = step["wav"].as_str().map(|name| {
                    let path = root.join("temp").join(format!("{name}.wav"));
                    write_tone(&path, wavs[name].as_f64().unwrap());
                    path
                });
                store
                    .save_voice(&values, wav.as_deref(), &ALLOWED)
                    .map(|v| {
                        ids.entry(alias.clone()).or_insert(v.id);
                    })
            }
        };
        match (&case["error"], outcome) {
            (Value::Null, Ok(())) => {}
            (Value::String(want), Err(e)) => {
                assert_eq!(e.message(), want, "step {n} message");
                assert_eq!(e.status(), 400, "step {n} status");
            }
            (want, got) => panic!("step {n}: wanted {want}, got {got:?}"),
        }
        let by_id: HashMap<&str, &str> =
            ids.iter().map(|(a, i)| (i.as_str(), a.as_str())).collect();
        let actual: Vec<Value> = store
            .voices()
            .iter()
            .map(|v| {
                let alias = by_id.get(v.id.as_str()).copied().unwrap_or(&v.id);
                let mut item = serde_json::to_value(v).unwrap();
                item["id"] = json!(alias);
                let audio = v.audio.as_str();
                item["audio"] = if audio.ends_with("old.wav") {
                    json!(format!("voices/{alias}/old.wav"))
                } else {
                    assert!(
                        audio.starts_with(&format!("voices/{}/", v.id))
                            && audio.ends_with(".wav")
                            && audio.len() == format!("voices/{}/", v.id).len() + 36,
                        "audio path shape: {audio}"
                    );
                    json!(format!("voices/{alias}/*.wav"))
                };
                item
            })
            .collect();
        assert_eq!(json!(actual), case["index"], "step {n}: {step}");
        let on_disk = read_json(&root.join("voices/index.json")).unwrap().unwrap();
        assert_eq!(
            on_disk,
            serde_json::to_value(store.voices()).unwrap(),
            "step {n} disk"
        );
    }
}

#[test]
fn name_boundaries_count_code_points() {
    let dir = temp_root();
    let store = Store::open(dir.path()).unwrap();
    let wav = dir.path().join("temp/a.wav");
    write_tone(&wav, 2.0);
    let save = |name: &str| {
        store.save_voice(
            &map(json!({"name": name, "model": MF})),
            Some(&wav),
            &ALLOWED,
        )
    };
    assert!(
        save(&"😀".repeat(80)).is_ok(),
        "80 emoji are 80 code points"
    );
    let e = save(&"あ".repeat(81)).unwrap_err();
    assert_eq!(
        msg(e),
        (400, "話者名を1～80文字で入力してください。".into())
    );
    assert!(save("　 ").is_err(), "ideographic space only");
    assert!(save("").is_err());
    assert!(save("\t\n").is_err());
}

#[test]
fn non_string_fields_are_rejected_not_coerced() {
    let dir = temp_root();
    let store = Store::open(dir.path()).unwrap();
    let wav = dir.path().join("temp/a.wav");
    write_tone(&wav, 2.0);
    for values in [
        json!({"name": 123, "model": MF}),
        json!({"name": "x", "model": 5}),
        json!({"model": MF}),
        json!({"name": null, "model": MF}),
    ] {
        assert!(store
            .save_voice(&map(values), Some(&wav), &ALLOWED)
            .is_err());
    }
    assert!(store.voices().is_empty());
}

#[test]
fn unknown_model_is_rejected() {
    let dir = temp_root();
    let store = Store::open(dir.path()).unwrap();
    let wav = dir.path().join("temp/a.wav");
    write_tone(&wav, 2.0);
    let e = store
        .save_voice(
            &map(json!({"name": "x", "model": "evil/model"})),
            Some(&wav),
            &ALLOWED,
        )
        .unwrap_err();
    assert_eq!(msg(e), (400, "使用するモデルを選んでください。".into()));
}

#[test]
fn audio_length_limits_are_one_to_one_hundred_twenty_seconds() {
    let dir = temp_root();
    let store = Store::open(dir.path()).unwrap();
    let save = |seconds: f64| {
        let wav = dir.path().join("temp/len.wav");
        write_tone(&wav, seconds);
        store.save_voice(
            &map(json!({"name": "x", "model": MF})),
            Some(&wav),
            &ALLOWED,
        )
    };
    let want = (400, "参照音声は1～120秒にしてください。".to_string());
    assert_eq!(msg(save(0.5).unwrap_err()), want);
    assert_eq!(msg(save(121.0).unwrap_err()), want);
    assert!(save(1.0).is_ok());
    assert!(save(120.0).is_ok());
    assert_eq!(
        std::fs::read_dir(dir.path().join("voices"))
            .unwrap()
            .count() as usize,
        store.voices().len() + 1, // index.json
        "a rejected save leaves no voice folder"
    );
}

#[test]
fn caption_is_cut_at_2000_chars() {
    let dir = temp_root();
    let store = Store::open(dir.path()).unwrap();
    let wav = dir.path().join("temp/a.wav");
    write_tone(&wav, 2.0);
    let v = store
        .save_voice(
            &map(json!({"name": "x", "model": MF, "caption": "あ".repeat(2100)})),
            Some(&wav),
            &ALLOWED,
        )
        .unwrap();
    assert_eq!(v.caption.chars().count(), 2000);
}

#[test]
fn editing_a_missing_voice_or_a_traversal_id_is_rejected() {
    let dir = temp_root();
    let store = Store::open(dir.path()).unwrap();
    let wav = dir.path().join("temp/a.wav");
    write_tone(&wav, 2.0);
    for id in ["f".repeat(32), "../escape".to_string()] {
        let e = store
            .save_voice(
                &map(json!({"id": id, "name": "x", "model": MF, "revision": 1})),
                Some(&wav),
                &ALLOWED,
            )
            .unwrap_err();
        assert_eq!(msg(e), (400, "編集対象の話者が見つかりません。".into()));
    }
    assert!(!dir.path().join("escape").exists());
}

#[test]
fn revision_may_arrive_as_a_whole_float() {
    let dir = temp_root();
    let store = Store::open(dir.path()).unwrap();
    let v = new_voice(&store, dir.path(), "a");
    let ok = store.save_voice(
        &map(json!({"id": v.id, "revision": 1.0, "name": "b", "model": MF})),
        None,
        &ALLOWED,
    );
    assert!(ok.is_ok(), "Python treats 1.0 == 1");
}

#[test]
fn new_voice_without_audio_is_rejected() {
    let dir = temp_root();
    let store = Store::open(dir.path()).unwrap();
    let e = store
        .save_voice(&map(json!({"name": "x", "model": MF})), None, &ALLOWED)
        .unwrap_err();
    assert_eq!(
        msg(e),
        (
            400,
            "参照音声を追加するか、生成した候補を選んでください。".into()
        )
    );
}

#[test]
fn unknown_fields_survive_a_save_and_a_rename() {
    let dir = temp_root();
    let id = "a".repeat(32);
    let store = open_with(
        dir.path(),
        json!([{"id": id, "revision": 2, "audio": "voices/x/old.wav", "name": "A", "model": MF, "caption": "", "future": {"k": [1]}}]),
    );
    store
        .save_voice(
            &map(json!({"id": id, "revision": 2, "name": "B", "model": MF})),
            None,
            &ALLOWED,
        )
        .unwrap();
    store
        .rename_voice(&map(json!({"id": id, "revision": 3, "name": "C"})))
        .unwrap();
    assert_eq!(store.voices()[0].extra["future"], json!({"k": [1]}));
    let disk = read_json(&dir.path().join("voices/index.json"))
        .unwrap()
        .unwrap();
    assert_eq!(disk[0]["future"], json!({"k": [1]}));
}

#[test]
fn failed_index_write_leaves_memory_and_files_unchanged() {
    let dir = temp_root();
    let store = Store::open(dir.path()).unwrap();
    let before = new_voice(&store, dir.path(), "first");
    // index.json の場所をフォルダにして、書けないようにする。
    let index = dir.path().join("voices/index.json");
    std::fs::remove_file(&index).unwrap();
    std::fs::create_dir(&index).unwrap();
    let wav = dir.path().join("temp/b.wav");
    write_tone(&wav, 2.0);
    let err = store
        .save_voice(
            &map(json!({"name": "second", "model": MF})),
            Some(&wav),
            &ALLOWED,
        )
        .unwrap_err();
    assert_eq!(err.status(), 400);
    assert_eq!(store.voices(), vec![before.clone()]);
    let folders: Vec<_> = std::fs::read_dir(dir.path().join("voices"))
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir() && e.file_name() != "index.json")
        .collect();
    assert_eq!(folders.len(), 1, "only the first voice's folder remains");
    // 名前変更の書き込み失敗でもメモリは変わらない。
    let e = store
        .rename_voice(&map(
            json!({"id": before.id, "revision": before.revision, "name": "zzz"}),
        ))
        .unwrap_err();
    assert_eq!(e.status(), 400);
    assert_eq!(store.voices(), vec![before]);
}

#[test]
fn delete_keeps_files_and_unknown_id_is_ok() {
    let dir = temp_root();
    let store = Store::open(dir.path()).unwrap();
    let v = new_voice(&store, dir.path(), "a");
    let wav = dir.path().join(&v.audio);
    store.delete_voice("nope").unwrap();
    assert_eq!(store.voices().len(), 1);
    store.delete_voice(&v.id).unwrap();
    assert!(store.voices().is_empty());
    assert!(wav.is_file(), "queued jobs may still read the reference");
    let disk = read_json(&dir.path().join("voices/index.json"))
        .unwrap()
        .unwrap();
    assert_eq!(disk, json!([]));
}

#[test]
fn old_audio_file_stays_after_replacement() {
    let dir = temp_root();
    let store = Store::open(dir.path()).unwrap();
    let v = new_voice(&store, dir.path(), "a");
    let wav = dir.path().join("temp/new.wav");
    write_tone(&wav, 3.0);
    let v2 = store
        .save_voice(
            &map(json!({"id": v.id, "revision": v.revision, "name": "a", "model": MF})),
            Some(&wav),
            &ALLOWED,
        )
        .unwrap();
    assert_ne!(v.audio, v2.audio);
    assert!(dir.path().join(&v.audio).is_file(), "old version stays");
    assert!(dir.path().join(&v2.audio).is_file());
    assert_eq!(v2.sound_revision, Some(v2.revision));
}

#[test]
fn saved_audio_is_24_bit_pcm_at_the_source_rate() {
    let dir = temp_root();
    let store = Store::open(dir.path()).unwrap();
    let v = new_voice(&store, dir.path(), "a");
    let r = hound::WavReader::open(dir.path().join(&v.audio)).unwrap();
    assert_eq!(
        (
            r.spec().bits_per_sample,
            r.spec().sample_rate,
            r.spec().channels
        ),
        (24, 24000, 1)
    );
    assert!(
        v.audio.starts_with(&format!("voices/{}/", v.id)),
        "{}",
        v.audio
    );
}

#[test]
fn copy_as_pcm24_converts_16bit_and_float() {
    let dir = temp_root();
    let (a16, f32p) = (dir.path().join("temp/a.wav"), dir.path().join("temp/f.wav"));
    write_tone(&a16, 1.0);
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 48000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(&f32p, spec).unwrap();
    for _ in 0..4800 {
        w.write_sample(0.5f32).unwrap();
        w.write_sample(-0.5f32).unwrap();
    }
    w.finalize().unwrap();
    for (src, secs) in [(&a16, 1.0), (&f32p, 0.1)] {
        let dst = dir.path().join("temp/out.wav");
        let got = postprocess::copy_as_pcm24(src, &dst).unwrap();
        assert!((got - secs).abs() < 1e-6, "{got}");
        let r = hound::WavReader::open(&dst).unwrap();
        assert_eq!(r.spec().bits_per_sample, 24);
    }
    let r = hound::WavReader::open(dir.path().join("temp/out.wav")).unwrap();
    assert_eq!((r.spec().sample_rate, r.spec().channels), (48000, 2));
}

#[test]
fn copy_as_pcm24_rejects_garbage_and_leaves_nothing() {
    let dir = temp_root();
    let (bad, dst) = (
        dir.path().join("temp/bad.wav"),
        dir.path().join("temp/out.wav"),
    );
    std::fs::write(&bad, b"not a wav at all").unwrap();
    let e = postprocess::copy_as_pcm24(&bad, &dst).unwrap_err();
    assert_eq!(
        msg(e),
        (400, "音声を読み込めません。形式を確認してください。".into())
    );
    assert!(!dst.exists());
    // 書き込み先のフォルダが無い。
    let good = dir.path().join("temp/good.wav");
    write_tone(&good, 1.0);
    assert!(postprocess::copy_as_pcm24(&good, &dir.path().join("nodir/out.wav")).is_err());
}
