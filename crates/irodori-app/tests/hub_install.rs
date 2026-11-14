//! `install_model`（モデルと codec の導入）のテスト。偽の HF サーバを使い、実ネットワークには出ない。

mod support;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use irodori_app::download::{
    approximate_bytes, install_model, install_with_space, DownloadControl, DownloadError,
    HostPolicy, Hub, ModelInstaller, RealInstaller, Transfer, APPROX_CODEC_BYTES,
    APPROX_MODEL_BYTES,
};
use irodori_app::models::{locate, CODEC_ID};
use support::fake_hub::{FakeFile, FakeHub, SHA};

const MODEL: &str = "Aratako/Irodori-TTS-v4.1-Small-MF";
/// 切断のテストで、書きかけが確実に残る大きさ（サーバの送信バッファより大きく切る）。
const MODEL_LEN: usize = 2_500_000;

fn model_bytes() -> Vec<u8> {
    (0..MODEL_LEN as u32).map(|i| (i % 251) as u8).collect()
}

fn files() -> Vec<FakeFile> {
    vec![
        FakeFile::new(MODEL, "model.safetensors", &model_bytes(), true),
        FakeFile::new(MODEL, "tokenizer/tokenizer.json", b"{\"tok\":1}", false),
        FakeFile::new(MODEL, "tokenizer/tokenizer_config.json", b"{}", false),
        FakeFile::new(CODEC_ID, "weights.pth", &[9u8; 5000], true),
    ]
}

fn hub(server: &FakeHub) -> Hub {
    Hub::with_base(&server.base_url(), HostPolicy::Local)
}

struct Run {
    cancel: AtomicBool,
    done: AtomicU64,
    total: AtomicU64,
}

impl Run {
    fn new() -> Run {
        Run {
            cancel: AtomicBool::new(false),
            done: AtomicU64::new(0),
            total: AtomicU64::new(0),
        }
    }

    fn install(&self, server: &FakeHub, hub_dir: &Path) -> Result<(), DownloadError> {
        let progress = |n: u64| {
            self.done.fetch_add(n, Ordering::SeqCst);
        };
        let on_total = |n: u64| self.total.store(n, Ordering::SeqCst);
        install_model(
            &hub(server),
            hub_dir,
            MODEL,
            &Transfer {
                cancel: &self.cancel,
                progress: &progress,
            },
            &on_total,
        )
    }
}

fn repo_dir(hub_dir: &Path, id: &str) -> PathBuf {
    hub_dir.join(format!("models--{}", id.replace('/', "--")))
}

fn place(hub_dir: &Path, id: &str, relative: &str, bytes: &[u8]) {
    let path = repo_dir(hub_dir, id)
        .join("snapshots")
        .join(SHA)
        .join(relative);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, bytes).unwrap();
}

fn all_files(root: &Path) -> Vec<String> {
    fn walk(dir: &Path, root: &Path, out: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).unwrap().flatten() {
            let path = entry.path();
            if entry.file_name() == ".locks" {
                continue; // アプリのロック置き場（HF キャッシュ互換の `.locks`）
            }
            if path.is_dir() {
                walk(&path, root, out);
            } else {
                out.push(
                    path.strip_prefix(root)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/"),
                );
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, &mut out);
    out.sort();
    out
}

#[test]
fn install_places_a_model_and_codec_where_locate_finds_them() {
    let server = FakeHub::start(files());
    let dir = tempfile::tempdir().unwrap();
    let hub_dir = dir.path().join("hub");
    let run = Run::new();
    run.install(&server, &hub_dir).unwrap();
    let found = locate(&hub_dir, MODEL).expect("installed and found");
    assert_eq!(std::fs::read(&found.model).unwrap(), model_bytes());
    assert_eq!(std::fs::read(&found.codec).unwrap(), vec![9u8; 5000]);
    for id in [MODEL, CODEC_ID] {
        let refs = std::fs::read_to_string(repo_dir(&hub_dir, id).join("refs/main")).unwrap();
        assert_eq!(refs.trim(), SHA);
    }
    assert!(
        all_files(&hub_dir)
            .iter()
            .all(|f| !f.ends_with(".incomplete")),
        "{:?}",
        all_files(&hub_dir)
    );
    assert_eq!(
        run.done.load(Ordering::SeqCst),
        run.total.load(Ordering::SeqCst)
    );
    assert_eq!(
        run.total.load(Ordering::SeqCst),
        MODEL_LEN as u64 + 9 + 2 + 5000
    );
}

#[test]
fn install_skips_the_codec_when_it_is_already_installed() {
    let server = FakeHub::start(files());
    let dir = tempfile::tempdir().unwrap();
    let hub_dir = dir.path().join("hub");
    place(&hub_dir, CODEC_ID, "weights.pth", &[9u8; 5000]);
    Run::new().install(&server, &hub_dir).unwrap();
    assert!(
        server.requests().iter().all(|r| !r.path.contains("DACVAE")),
        "{:?}",
        server.requests()
    );
    assert!(locate(&hub_dir, MODEL).is_some());
}

#[test]
fn refs_main_is_written_last_and_a_failed_install_is_not_installed() {
    let server = FakeHub::start(files());
    let dir = tempfile::tempdir().unwrap();
    let hub_dir = dir.path().join("hub");
    server.status("/tokenizer/tokenizer_config.json", 500);
    let err = Run::new().install(&server, &hub_dir).unwrap_err();
    assert_eq!(err, DownloadError::Http(500));
    assert!(
        !repo_dir(&hub_dir, MODEL).join("refs/main").exists(),
        "refs/main comes last"
    );
    assert!(locate(&hub_dir, MODEL).is_none());
}

#[test]
fn install_resumes_after_a_failure_without_refetching_finished_files() {
    let server = FakeHub::start(files());
    let dir = tempfile::tempdir().unwrap();
    let hub_dir = dir.path().join("hub");
    server.status("/tokenizer/tokenizer_config.json", 500);
    assert!(Run::new().install(&server, &hub_dir).is_err());
    let model_requests = server
        .requests_matching("/resolve/")
        .iter()
        .filter(|r| r.path.ends_with("model.safetensors"))
        .count();
    server.clear_faults();
    Run::new().install(&server, &hub_dir).unwrap();
    assert!(locate(&hub_dir, MODEL).is_some());
    let after = server
        .requests_matching("/resolve/")
        .iter()
        .filter(|r| r.path.ends_with("model.safetensors"))
        .count();
    assert_eq!(
        after, model_requests,
        "finished files are not downloaded again"
    );
}

#[test]
fn install_resumes_a_cut_download() {
    let server = FakeHub::start(files());
    let dir = tempfile::tempdir().unwrap();
    let hub_dir = dir.path().join("hub");
    server.truncate_after(Some(1_000_000));
    assert_eq!(
        Run::new().install(&server, &hub_dir).unwrap_err(),
        DownloadError::Offline
    );
    assert!(locate(&hub_dir, MODEL).is_none());
    server.clear_faults();
    Run::new().install(&server, &hub_dir).unwrap();
    assert!(locate(&hub_dir, MODEL).is_some());
    assert!(
        server
            .requests_matching("/resolve/")
            .iter()
            .any(|r| r.range.is_some()),
        "a Range request was used"
    );
}

#[test]
fn install_is_idempotent_when_everything_is_present() {
    let server = FakeHub::start(files());
    let dir = tempfile::tempdir().unwrap();
    let hub_dir = dir.path().join("hub");
    Run::new().install(&server, &hub_dir).unwrap();
    let before = server.requests().len();
    let run = Run::new();
    run.install(&server, &hub_dir).unwrap();
    assert_eq!(server.requests().len(), before, "no request at all");
    assert_eq!(run.total.load(Ordering::SeqCst), 0);
}

#[test]
fn install_refuses_when_there_is_not_enough_free_space() {
    let server = FakeHub::start(files());
    let dir = tempfile::tempdir().unwrap();
    let hub_dir = dir.path().join("hub");
    let run = Run::new();
    let progress = |_: u64| {};
    let on_total = |_: u64| {};
    let asked = std::sync::Mutex::new(Vec::new());
    let err = install_with_space(
        &hub(&server),
        &hub_dir,
        MODEL,
        &Transfer {
            cancel: &run.cancel,
            progress: &progress,
        },
        &on_total,
        &|p: &Path| {
            asked.lock().unwrap().push(p.to_path_buf());
            1000
        },
    )
    .unwrap_err();
    let needed = ((MODEL_LEN as u64 + 9 + 2 + 5000) as f64 * 1.2).ceil() as u64;
    assert_eq!(
        err,
        DownloadError::DiskSpace {
            needed,
            available: 1000
        }
    );
    assert!(
        server.requests_matching("/resolve/").is_empty(),
        "nothing was downloaded"
    );
    assert!(!hub_dir.exists() || all_files(&hub_dir).is_empty());
    let asked = asked.lock().unwrap();
    assert!(
        asked.iter().all(|p| p.is_dir()),
        "the nearest existing folder is asked: {asked:?}"
    );
}

#[test]
fn models_outside_the_allow_list_are_refused() {
    let server = FakeHub::start(files());
    let dir = tempfile::tempdir().unwrap();
    let run = Run::new();
    let progress = |_: u64| {};
    let on_total = |_: u64| {};
    for id in ["someone/else", CODEC_ID, "", "../x"] {
        let err = install_model(
            &hub(&server),
            dir.path(),
            id,
            &Transfer {
                cancel: &run.cancel,
                progress: &progress,
            },
            &on_total,
        )
        .unwrap_err();
        assert!(matches!(err, DownloadError::Protocol(_)), "{id:?}: {err:?}");
    }
    assert!(server.requests().is_empty());
}

#[test]
fn on_total_counts_only_what_is_missing() {
    let server = FakeHub::start(files());
    let dir = tempfile::tempdir().unwrap();
    let hub_dir = dir.path().join("hub");
    place(&hub_dir, MODEL, "model.safetensors", &model_bytes());
    let run = Run::new();
    run.install(&server, &hub_dir).unwrap();
    assert_eq!(run.total.load(Ordering::SeqCst), 9 + 2 + 5000);
    assert_eq!(run.done.load(Ordering::SeqCst), 9 + 2 + 5000);
    assert!(server
        .requests_matching("/resolve/")
        .iter()
        .all(|r| !r.path.ends_with("model.safetensors")));
}

#[test]
fn a_file_with_the_wrong_size_is_fetched_again() {
    let server = FakeHub::start(files());
    let dir = tempfile::tempdir().unwrap();
    let hub_dir = dir.path().join("hub");
    place(&hub_dir, MODEL, "model.safetensors", b"short");
    Run::new().install(&server, &hub_dir).unwrap();
    let found = locate(&hub_dir, MODEL).unwrap();
    assert_eq!(std::fs::read(found.model).unwrap(), model_bytes());
}

#[test]
fn approximate_bytes_drops_the_codec_when_installed() {
    let dir = tempfile::tempdir().unwrap();
    let hub_dir = dir.path().join("hub");
    assert_eq!(
        approximate_bytes(&hub_dir, MODEL),
        APPROX_MODEL_BYTES + APPROX_CODEC_BYTES
    );
    place(&hub_dir, CODEC_ID, "weights.pth", b"x");
    assert_eq!(approximate_bytes(&hub_dir, MODEL), APPROX_MODEL_BYTES);
    place(&hub_dir, MODEL, "model.safetensors", b"x");
    place(&hub_dir, MODEL, "tokenizer/tokenizer.json", b"x");
    place(&hub_dir, MODEL, "tokenizer/tokenizer_config.json", b"x");
    assert_eq!(
        approximate_bytes(&hub_dir, MODEL),
        0,
        "installed models need nothing"
    );
    assert_eq!(approximate_bytes(&hub_dir, "someone/else"), 0);
}

#[test]
fn install_never_writes_outside_the_hub_dir() {
    let server = FakeHub::start(files());
    server.add_decoys();
    let dir = tempfile::tempdir().unwrap();
    let hub_dir = dir.path().join("hub");
    Run::new().install(&server, &hub_dir).unwrap();
    let outside = all_files(dir.path())
        .into_iter()
        .filter(|f| !f.starts_with("hub/"))
        .collect::<Vec<_>>();
    assert!(outside.is_empty(), "{outside:?}");
    let inside = all_files(&hub_dir);
    assert!(
        inside
            .iter()
            .all(|f| !f.contains("evil") && !f.contains("README")),
        "{inside:?}"
    );
    assert_eq!(
        inside.len(),
        4 + 2,
        "four files and two refs/main: {inside:?}"
    );
}

#[test]
fn a_cancelled_install_keeps_partial_files_but_is_not_installed() {
    let server = FakeHub::start(files());
    let dir = tempfile::tempdir().unwrap();
    let hub_dir = dir.path().join("hub");
    let cancel = AtomicBool::new(false);
    let progress = |n: u64| {
        if n > 0 {
            cancel.store(true, Ordering::SeqCst);
        }
    };
    let on_total = |_: u64| {};
    let err = install_model(
        &hub(&server),
        &hub_dir,
        MODEL,
        &Transfer {
            cancel: &cancel,
            progress: &progress,
        },
        &on_total,
    )
    .unwrap_err();
    assert_eq!(err, DownloadError::Cancelled);
    assert!(locate(&hub_dir, MODEL).is_none());
    assert!(!repo_dir(&hub_dir, MODEL).join("refs/main").exists());
}

#[test]
fn a_stale_partial_from_an_older_commit_is_removed() {
    let server = FakeHub::start(files());
    let dir = tempfile::tempdir().unwrap();
    let hub_dir = dir.path().join("hub");
    let old = repo_dir(&hub_dir, MODEL)
        .join("snapshots")
        .join("0ldc0mm1t");
    std::fs::create_dir_all(&old).unwrap();
    std::fs::write(old.join("model.safetensors.incomplete"), vec![1u8; 1000]).unwrap();
    Run::new().install(&server, &hub_dir).unwrap();
    assert!(
        !old.join("model.safetensors.incomplete").exists(),
        "{:?}",
        all_files(&hub_dir)
    );
    assert!(locate(&hub_dir, MODEL).is_some());
}

#[test]
fn a_second_install_into_the_same_folder_is_refused_while_one_runs() {
    let server = FakeHub::start(files());
    let dir = tempfile::tempdir().unwrap();
    let hub_dir = dir.path().join("hub");
    server.stall_after(1_000_000);
    std::thread::scope(|scope| {
        let first = scope.spawn(|| Run::new().install(&server, &hub_dir));
        let started = std::time::Instant::now();
        while server.requests_matching("/resolve/").is_empty() {
            assert!(
                started.elapsed().as_secs() < 20,
                "the first install never started fetching"
            );
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let second = Run::new().install(&server, &hub_dir);
        assert_eq!(second, Err(DownloadError::Locked));
        // 解放すると偽サーバは接続を切る（最初の取得は途中で終わる）。
        server.release();
        assert_eq!(first.join().unwrap(), Err(DownloadError::Offline));
    });
    // 終わればロックは外れ、続きから取得できる。
    server.clear_faults();
    Run::new().install(&server, &hub_dir).unwrap();
    assert!(locate(&hub_dir, MODEL).is_some());
}

#[test]
fn the_installer_estimate_is_cached_and_refreshed_after_an_install() {
    let server = FakeHub::start(files());
    let dir = tempfile::tempdir().unwrap();
    let hub_dir = dir.path().join("hub");
    let installer = RealInstaller::with_hub(hub_dir.clone(), hub(&server));
    let before = installer.approximate_bytes(MODEL);
    assert_eq!(before, APPROX_MODEL_BYTES + APPROX_CODEC_BYTES);
    // 画面の状態の取得のたびに調べ直さない（短い間は同じ値）。
    place(&hub_dir, CODEC_ID, "weights.pth", b"x");
    assert_eq!(installer.approximate_bytes(MODEL), before);
    installer.install(MODEL, &DownloadControl::new()).unwrap();
    assert_eq!(
        installer.approximate_bytes(MODEL),
        0,
        "refreshed by the install"
    );
}
