//! `Hub::plan`（取得計画）のテスト。偽の HF サーバを使い、実ネットワークには出ない。

mod support;

use irodori_app::download::{DownloadError, HostPolicy, Hub};
use support::fake_hub::{FakeFile, FakeHub, SHA};

const REPO: &str = "Aratako/Irodori-TTS-v4.1-Small-MF";
const WANTED: [&str; 3] = [
    "model.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
];

fn files() -> Vec<FakeFile> {
    vec![
        FakeFile::new(REPO, "model.safetensors", &[7u8; 1000], true),
        FakeFile::new(REPO, "tokenizer/tokenizer.json", b"{\"a\":1}", false),
        FakeFile::new(REPO, "tokenizer/tokenizer_config.json", b"{}", false),
        FakeFile::new(REPO, "README.md", b"hello", false),
    ]
}

fn hub(server: &FakeHub) -> Hub {
    Hub::with_base(&server.base_url(), HostPolicy::Local)
}

#[test]
fn plan_reads_sha_sizes_and_lfs_hashes() {
    let server = FakeHub::start(files());
    let plan = hub(&server).plan(REPO, &WANTED).unwrap();
    assert_eq!(plan.repo, REPO);
    assert_eq!(plan.sha, SHA);
    let paths: Vec<&str> = plan.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(paths, WANTED, "files follow the order that was asked for");
    let model = &plan.files[0];
    assert_eq!(model.size, 1000);
    let expected = files()[0].sha256();
    assert_eq!(model.sha256.as_deref(), Some(expected.as_str()));
    assert_eq!(
        plan.files[1].sha256, None,
        "non-LFS files are checked by size only"
    );
    assert_eq!(plan.files[1].size, 7);
}

#[test]
fn plan_asks_for_the_tree_at_the_pinned_commit() {
    let server = FakeHub::start(files());
    hub(&server).plan(REPO, &WANTED).unwrap();
    let paths: Vec<String> = server.requests().into_iter().map(|r| r.path).collect();
    assert!(paths.contains(&format!("/api/models/{REPO}")), "{paths:?}");
    assert!(
        paths.contains(&format!("/api/models/{REPO}/tree/{SHA}")),
        "{paths:?}"
    );
}

#[test]
fn plan_ignores_files_that_were_not_asked_for() {
    let server = FakeHub::start(files());
    server.add_decoys();
    let plan = hub(&server).plan(REPO, &WANTED).unwrap();
    let paths: Vec<&str> = plan.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(
        paths, WANTED,
        "README.md and ../evil.bin never get in: {paths:?}"
    );
}

#[test]
fn plan_fails_when_a_wanted_file_is_missing() {
    let server = FakeHub::start(files()[..2].to_vec());
    let err = hub(&server).plan(REPO, &WANTED).unwrap_err();
    assert!(matches!(err, DownloadError::Protocol(_)), "{err:?}");
    assert!(
        err.message().contains("logs/editor.log"),
        "{}",
        err.message()
    );
}

#[test]
fn plan_rejects_a_malformed_commit_sha_or_hash() {
    for sha in [
        "../../x",
        "0123456789abcdef0123456789abcdef0123456",
        "0123456789ABCDEF0123456789ABCDEF01234567",
        "",
        "0123456789abcdef0123456789abcdef012345678",
    ] {
        let server = FakeHub::start(files());
        server.commit_sha(sha);
        let err = hub(&server).plan(REPO, &WANTED).unwrap_err();
        assert!(
            matches!(err, DownloadError::Protocol(_)),
            "sha {sha:?}: {err:?}"
        );
    }
    for oid in ["xyz", "ABCDEF", &"g".repeat(64), &"a".repeat(63)] {
        let server = FakeHub::start(files());
        server.lfs_oid(oid);
        let err = hub(&server).plan(REPO, &WANTED).unwrap_err();
        assert!(
            matches!(err, DownloadError::Protocol(_)),
            "oid {oid:?}: {err:?}"
        );
    }
}

#[test]
fn plan_maps_http_errors_and_offline() {
    let server = FakeHub::start(files());
    server.status(&format!("/api/models/{REPO}"), 404);
    assert_eq!(
        hub(&server).plan(REPO, &WANTED).unwrap_err(),
        DownloadError::Http(404)
    );
    server.clear_status();
    server.status(&format!("/tree/{SHA}"), 503);
    let err = hub(&server).plan(REPO, &WANTED).unwrap_err();
    assert_eq!(err, DownloadError::Http(503));
    assert_eq!(
        err.message(),
        "モデルの取得元から取得できませんでした（HTTP 503）。"
    );
    let offline = Hub::with_base("http://127.0.0.1:1", HostPolicy::Local);
    let err = offline.plan(REPO, &WANTED).unwrap_err();
    assert_eq!(err, DownloadError::Offline);
    assert_eq!(
        err.message(),
        "ネットワークに接続できません。接続を確認してもう一度お試しください。"
    );
}

#[test]
fn plan_refuses_a_plain_http_base_for_the_huggingface_policy() {
    let hub = Hub::with_base("http://huggingface.co", HostPolicy::HuggingFace);
    let err = hub.plan(REPO, &WANTED).unwrap_err();
    assert!(matches!(err, DownloadError::Protocol(_)), "{err:?}");
}

#[test]
fn disk_space_and_other_messages_are_japanese() {
    assert_eq!(
        DownloadError::DiskSpace {
            needed: 3_300_000_000,
            available: 1_000_000_000
        }
        .message(),
        "空き容量が足りません。あと約2.3GB 必要です。"
    );
    assert_eq!(
        DownloadError::Verification.message(),
        "ダウンロードしたファイルの検証に失敗しました。もう一度お試しください。"
    );
    let generic = "処理に失敗しました。logs/editor.logを確認してください。";
    assert_eq!(
        DownloadError::Io("secret path C:\\x".into()).message(),
        generic
    );
    assert_eq!(DownloadError::Protocol("secret".into()).message(), generic);
}
