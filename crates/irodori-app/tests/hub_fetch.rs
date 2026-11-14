//! `Hub::fetch`（1 ファイルの取得）のテスト。偽の HF サーバを使い、実ネットワークには出ない。

mod support;

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use irodori_app::download::{DownloadError, HostPolicy, Hub, PlannedFile, Transfer};
use support::fake_hub::{FakeFile, FakeHub, SHA};

const REPO: &str = "Aratako/Irodori-TTS-v4.1-Small-MF";
const PATH: &str = "model.safetensors";

fn content() -> Vec<u8> {
    (0..3 * 1024 * 1024u32)
        .map(|i| (i.wrapping_mul(31) >> 3) as u8)
        .collect()
}

fn server() -> FakeHub {
    FakeHub::start(vec![FakeFile::new(REPO, PATH, &content(), true)])
}

fn hub(server: &FakeHub) -> Hub {
    Hub::with_base(&server.base_url(), HostPolicy::Local)
}

fn planned(server: &FakeHub) -> PlannedFile {
    hub(server).plan(REPO, &[PATH]).unwrap().files.remove(0)
}

struct Run {
    cancel: AtomicBool,
    done: AtomicU64,
}

impl Run {
    fn new() -> Run {
        Run {
            cancel: AtomicBool::new(false),
            done: AtomicU64::new(0),
        }
    }

    fn fetch(&self, hub: &Hub, file: &PlannedFile, dest: &Path) -> Result<(), DownloadError> {
        let progress = |n: u64| {
            self.done.fetch_add(n, Ordering::SeqCst);
        };
        hub.fetch(
            REPO,
            SHA,
            file,
            dest,
            &Transfer {
                cancel: &self.cancel,
                progress: &progress,
            },
        )
    }

    fn done(&self) -> u64 {
        self.done.load(Ordering::SeqCst)
    }
}

fn partial(dest: &Path) -> std::path::PathBuf {
    let mut name = dest.file_name().unwrap().to_os_string();
    name.push(".incomplete");
    dest.with_file_name(name)
}

#[test]
fn fetch_downloads_verifies_and_renames() {
    let server = server();
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("snapshots").join(SHA).join(PATH);
    let run = Run::new();
    run.fetch(&hub(&server), &planned(&server), &dest).unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), content());
    assert!(!partial(&dest).exists(), "no .incomplete is left behind");
    assert_eq!(
        run.done(),
        content().len() as u64,
        "progress adds up to the size"
    );
}

#[test]
fn fetch_resumes_a_partial_file_with_a_range_request() {
    let server = server();
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join(PATH);
    let file = planned(&server);
    server.truncate_after(Some(1_000_000));
    let first = Run::new();
    let err = first.fetch(&hub(&server), &file, &dest).unwrap_err();
    assert_eq!(
        err,
        DownloadError::Offline,
        "a cut connection is a network error"
    );
    let kept = std::fs::metadata(partial(&dest)).unwrap().len();
    assert!(kept > 0 && kept < file.size, "{kept}");
    assert!(!dest.exists());

    server.clear_faults();
    let second = Run::new();
    second.fetch(&hub(&server), &file, &dest).unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), content());
    assert_eq!(second.done(), file.size, "the kept bytes count as done");
    let ranges: Vec<Option<String>> = server
        .requests_matching("/resolve/")
        .into_iter()
        .map(|r| r.range)
        .collect();
    assert_eq!(
        ranges.last().unwrap().as_deref(),
        Some(format!("bytes={kept}-").as_str()),
        "{ranges:?}"
    );
}

#[test]
fn fetch_restarts_when_the_server_ignores_range() {
    let server = server();
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join(PATH);
    let file = planned(&server);
    server.truncate_after(Some(500_000));
    let _ = Run::new().fetch(&hub(&server), &file, &dest);
    assert!(partial(&dest).exists());
    server.clear_faults();
    server.ignore_range();
    let run = Run::new();
    run.fetch(&hub(&server), &file, &dest).unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), content());
    assert_eq!(run.done(), file.size, "restarting does not double-count");
}

#[test]
fn a_complete_partial_file_is_verified_without_a_request() {
    let server = server();
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join(PATH);
    std::fs::write(partial(&dest), content()).unwrap();
    let file = planned(&server);
    let before = server.requests_matching("/resolve/").len();
    let run = Run::new();
    run.fetch(&hub(&server), &file, &dest).unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), content());
    assert_eq!(server.requests_matching("/resolve/").len(), before);
    assert_eq!(run.done(), file.size);
}

#[test]
fn a_partial_file_larger_than_the_plan_is_discarded() {
    let server = server();
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join(PATH);
    std::fs::write(partial(&dest), vec![1u8; content().len() + 10]).unwrap();
    Run::new()
        .fetch(&hub(&server), &planned(&server), &dest)
        .unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), content());
}

#[test]
fn a_corrupt_partial_file_fails_verification_and_is_removed() {
    let server = server();
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join(PATH);
    // 途中までの中身がすり替わっている（先頭の 1000 バイトが別物）。
    let mut bad = content()[..1_000_000].to_vec();
    bad[..1000].fill(0xAA);
    std::fs::write(partial(&dest), bad).unwrap();
    let err = Run::new()
        .fetch(&hub(&server), &planned(&server), &dest)
        .unwrap_err();
    assert_eq!(err, DownloadError::Verification);
    assert!(!dest.exists() && !partial(&dest).exists());
}

#[test]
fn fetch_follows_a_redirect_to_an_allowed_host() {
    let server = server();
    server.redirect_internally();
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join(PATH);
    Run::new()
        .fetch(&hub(&server), &planned(&server), &dest)
        .unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), content());
    assert!(
        !server.requests_matching("/cdn/").is_empty(),
        "the redirect target was fetched"
    );
}

#[test]
fn fetch_refuses_a_redirect_to_a_forbidden_host_without_requesting_it() {
    let server = server();
    let file = planned(&server);
    // `localhost` はこのサーバへ届くが、許可ホスト（127.0.0.1）ではない。要求が出ていれば記録に残る。
    server.redirect_to(&format!("localhost:{}", server.port()));
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join(PATH);
    let err = Run::new().fetch(&hub(&server), &file, &dest).unwrap_err();
    assert!(matches!(err, DownloadError::Protocol(_)), "{err:?}");
    assert!(
        server.requests_matching("/x/").is_empty(),
        "nothing was requested from the forbidden host"
    );
    assert!(!dest.exists() && !partial(&dest).exists());
}

#[test]
fn fetch_rejects_too_many_redirects() {
    let server = server();
    let file = planned(&server);
    server.redirect_forever();
    let dir = tempfile::tempdir().unwrap();
    let err = Run::new()
        .fetch(&hub(&server), &file, &dir.path().join(PATH))
        .unwrap_err();
    assert!(matches!(err, DownloadError::Protocol(_)), "{err:?}");
    assert!(
        server.requests_matching("/resolve/").len() <= 6,
        "{}",
        server.requests_matching("/resolve/").len()
    );
}

#[test]
fn fetch_rejects_a_hash_mismatch_and_removes_the_partial_file() {
    let server = server();
    let mut file = planned(&server);
    file.sha256 = Some("ab".repeat(32));
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join(PATH);
    let err = Run::new().fetch(&hub(&server), &file, &dest).unwrap_err();
    assert_eq!(err, DownloadError::Verification);
    assert!(!dest.exists() && !partial(&dest).exists());
}

#[test]
fn fetch_rejects_swapped_content_with_the_right_size() {
    let server = server();
    let file = planned(&server);
    let mut swapped = content();
    swapped[10] ^= 0xFF;
    server.replace_bytes(REPO, PATH, &swapped);
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join(PATH);
    let err = Run::new().fetch(&hub(&server), &file, &dest).unwrap_err();
    assert_eq!(err, DownloadError::Verification);
    assert!(!dest.exists() && !partial(&dest).exists());
}

#[test]
fn fetch_rejects_a_body_that_does_not_match_the_planned_size() {
    let server = server();
    let mut file = planned(&server);
    file.size += 10;
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join(PATH);
    let err = Run::new().fetch(&hub(&server), &file, &dest).unwrap_err();
    assert!(
        matches!(
            err,
            DownloadError::Protocol(_) | DownloadError::Verification
        ),
        "{err:?}"
    );
    assert!(!dest.exists() && !partial(&dest).exists());
}

#[test]
fn fetch_rejects_an_html_error_page_served_with_200() {
    let server = server();
    let file = planned(&server);
    server.serve_html_with_200();
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join(PATH);
    let err = Run::new().fetch(&hub(&server), &file, &dest).unwrap_err();
    assert!(
        matches!(
            err,
            DownloadError::Protocol(_) | DownloadError::Verification
        ),
        "{err:?}"
    );
    assert!(!dest.exists() && !partial(&dest).exists());
}

#[test]
fn fetch_can_be_cancelled_and_keeps_the_partial_file() {
    let server = server();
    let file = planned(&server);
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join(PATH);
    let cancel = AtomicBool::new(false);
    let seen = AtomicU64::new(0);
    let progress = |n: u64| {
        if seen.fetch_add(n, Ordering::SeqCst) + n >= 256 * 1024 {
            cancel.store(true, Ordering::SeqCst);
        }
    };
    let err = hub(&server)
        .fetch(
            REPO,
            SHA,
            &file,
            &dest,
            &Transfer {
                cancel: &cancel,
                progress: &progress,
            },
        )
        .unwrap_err();
    assert_eq!(err, DownloadError::Cancelled);
    let kept = std::fs::metadata(partial(&dest)).unwrap().len();
    assert!(kept >= 256 * 1024 && kept < file.size, "{kept}");
    assert!(!dest.exists());
    // 取り消してから再開できる。
    let run = Run::new();
    run.fetch(&hub(&server), &file, &dest).unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), content());
}

#[test]
fn a_cancel_flag_that_is_already_set_stops_before_any_request() {
    let server = server();
    let file = planned(&server);
    let dir = tempfile::tempdir().unwrap();
    let run = Run::new();
    run.cancel.store(true, Ordering::SeqCst);
    let before = server.requests_matching("/resolve/").len();
    let err = run
        .fetch(&hub(&server), &file, &dir.path().join(PATH))
        .unwrap_err();
    assert_eq!(err, DownloadError::Cancelled);
    assert_eq!(server.requests_matching("/resolve/").len(), before);
}

#[test]
fn fetch_maps_http_404_and_an_unreachable_server() {
    let server = server();
    let file = planned(&server);
    server.status(&format!("/resolve/{SHA}/{PATH}"), 404);
    let dir = tempfile::tempdir().unwrap();
    let err = Run::new()
        .fetch(&hub(&server), &file, &dir.path().join(PATH))
        .unwrap_err();
    assert_eq!(err, DownloadError::Http(404));
    let offline = Hub::with_base("http://127.0.0.1:1", HostPolicy::Local);
    let err = Run::new()
        .fetch(&offline, &file, &dir.path().join("b"))
        .unwrap_err();
    assert_eq!(err, DownloadError::Offline);
}

#[test]
fn fetch_never_follows_a_path_outside_the_destination_folder() {
    // 保存先は呼び出し側が決める。`file.path` は URL にだけ使い、保存先のパスには使わない。
    let server = server();
    let file = planned(&server);
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("a").join("b").join(PATH);
    Run::new().fetch(&hub(&server), &file, &dest).unwrap();
    assert!(dest.is_file(), "parent folders are created");
}

#[test]
fn fetch_rejects_a_body_longer_than_planned_and_discards_it() {
    let server = server();
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join(PATH);
    let file = planned(&server);
    server.oversize();
    let run = Run::new();
    let err = run.fetch(&hub(&server), &file, &dest).unwrap_err();
    assert_eq!(err, DownloadError::Verification);
    assert!(!dest.exists());
    assert!(!partial(&dest).exists(), "the oversized partial is removed");
    assert!(
        run.done() <= file.size,
        "never counts or writes past the planned size: {}",
        run.done()
    );
}

#[test]
fn fetch_starts_over_when_the_range_is_not_satisfiable() {
    let server = server();
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join(PATH);
    let file = planned(&server);
    server.truncate_after(Some(500_000));
    let _ = Run::new().fetch(&hub(&server), &file, &dest);
    assert!(partial(&dest).exists());
    server.clear_faults();
    server.reject_range_once();
    let run = Run::new();
    run.fetch(&hub(&server), &file, &dest).unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), content());
    assert_eq!(run.done(), file.size);
}
