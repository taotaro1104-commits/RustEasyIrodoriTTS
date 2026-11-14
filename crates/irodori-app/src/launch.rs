//! 起動処理：引数、データフォルダ、ポート、二重起動の防止、起動時に開く新しい台本、ブラウザの自動起動。

use std::fs::{File, OpenOptions, TryLockError};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::Parser;
use irodori_store::model::{new_project, uid};
use irodori_store::Store;

use crate::download::RealInstaller;
use crate::real_synth::RealSynth;
use crate::{build_router, logging, AppState};

pub const DEFAULT_PORT: u16 = 7860;
/// 使用中のとき、希望のポートから何番先まで空きを探すか。
pub const PORT_SEARCH: u16 = 100;

#[derive(Debug, Parser)]
#[command(
    name = "irodori-app",
    version,
    about = "Easy-Irodori-TTS（ローカルで動く音声作成アプリ）"
)]
pub struct Args {
    /// 台本・話者・設定を置くフォルダ（既定は実行ファイルと同じ場所）
    #[arg(long, default_value_os_t = default_data_dir())]
    pub data_dir: PathBuf,
    /// 待ち受けるポート（使用中なら次の空きを使う）
    #[arg(long, default_value_t = DEFAULT_PORT)]
    pub port: u16,
    /// ブラウザを自動で開かない
    #[arg(long)]
    pub no_browser: bool,
    /// モデルを探すフォルダ（HF キャッシュ互換。既定は `<data-dir>/models/huggingface/hub` で、v2.0 のものをそのまま使える）
    #[arg(long)]
    pub models_dir: Option<PathBuf>,
}

impl Args {
    /// モデルを探すフォルダ。
    pub fn hub_dir(&self) -> PathBuf {
        self.models_dir
            .clone()
            .unwrap_or_else(|| self.data_dir.join("models").join("huggingface").join("hub"))
    }
}

pub fn default_data_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// v2.0 と同じ構成のサブフォルダ（とログ置き場）を用意する。
pub fn prepare_dirs(root: &Path) -> Result<()> {
    for sub in ["projects", "voices", "config", "temp", "outputs", "logs"] {
        std::fs::create_dir_all(root.join(sub)).with_context(|| {
            format!("{sub} フォルダを作れません。保存先の権限と空き容量を確認してください。")
        })?;
    }
    Ok(())
}

/// `preferred` から順に空いているポートを探して、そのまま待ち受ける。
pub fn bind_listener(preferred: u16) -> Result<TcpListener> {
    for offset in 0..PORT_SEARCH {
        let Some(port) = preferred.checked_add(offset) else {
            break;
        };
        if let Ok(listener) = TcpListener::bind(("127.0.0.1", port)) {
            return Ok(listener);
        }
    }
    bail!("空いているポートが見つかりません。他のアプリを閉じるか、--port で別の番号を指定してください。")
}

pub fn pick_port(preferred: u16) -> Result<u16> {
    Ok(bind_listener(preferred)?.local_addr()?.port())
}

/// 二重起動を防ぐロック。放す（Drop）か、プロセスが終わると解除される。
#[derive(Debug)]
pub struct LockGuard {
    _files: Vec<File>,
}

/// 同じデータフォルダを同時に使わないための印。
/// `app.lock` はこのアプリ同士、`environment.lock` は v2.0（Python 版）との排他に使う（v2.0 が同じファイルをロックする）。
const LOCK_FILES: [&str; 2] = ["temp/app.lock", "temp/environment.lock"];

pub fn acquire_lock(root: &Path) -> Result<LockGuard> {
    let mut files = Vec::with_capacity(LOCK_FILES.len());
    for name in LOCK_FILES {
        let path = root.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .with_context(|| format!("起動の印（{name}）を作れません。"))?;
        match file.try_lock() {
            Ok(()) => files.push(file),
            Err(TryLockError::WouldBlock) => bail!("アプリはすでに起動しています。"),
            Err(TryLockError::Error(e)) => {
                return Err(e).with_context(|| format!("起動の印（{name}）を確認できません。"))
            }
        }
    }
    Ok(LockGuard { _files: files })
}

/// v2.0 の起動時の動き。`restore_input` が偽なら、毎回新しい台本を開く。
/// 前回の台本に文章があれば、「前回の入力を復元」できるよう `recovery_project` に記録する。
pub fn fresh_start(store: &Store) -> Result<()> {
    let settings = store.settings();
    if settings.restore_input {
        return Ok(());
    }
    if !settings.last_project.is_empty() {
        match store.load(Some(&settings.last_project)) {
            Ok(old) if old.cards.iter().any(|c| !c.text.trim().is_empty()) => {
                store.set_recovery_project(Some(&settings.last_project))?
            }
            Ok(_) => {}
            // 前回の台本が読めなくても起動は止めない（元のファイルには触れない）。
            Err(e) => tracing::warn!("前回の台本を読めませんでした: {}", e.message()),
        }
    }
    store.persist(&new_project("無題の台本"))?;
    Ok(())
}

/// サーバが応答するようになってからブラウザを開く。
pub fn open_browser_when_ready(url: String) {
    std::thread::spawn(move || {
        let addr = url
            .strip_prefix("http://")
            .and_then(|h| h.trim_end_matches('/').to_socket_addrs().ok())
            .and_then(|mut a| a.next());
        let Some(addr) = addr else { return };
        for _ in 0..100 {
            if TcpStream::connect_timeout(&addr, Duration::from_millis(500)).is_ok() {
                if let Err(e) = open::that(&url) {
                    tracing::warn!("ブラウザを開けませんでした: {e}");
                }
                return;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    });
}

pub async fn run(args: Args) -> Result<()> {
    prepare_dirs(&args.data_dir)?;
    let _lock = acquire_lock(&args.data_dir)?;
    logging::init_file_logging(&args.data_dir)?;
    let result = serve(&args).await;
    if let Err(e) = &result {
        // 起動できなかった理由を、画面を持たない起動（ダブルクリックなど）でも後から読めるようにする。
        tracing::error!("fatal: {e:#}");
    }
    result
}

async fn serve(args: &Args) -> Result<()> {
    let store = Arc::new(Store::open(&args.data_dir)?);
    // ポートが取れないときに、空の台本だけを増やさないよう、待ち受けてから新しい台本を開く。
    let listener = bind_listener(args.port)?;
    listener.set_nonblocking(true)?;
    fresh_start(&store)?;

    let port = listener.local_addr()?.port();
    let url = format!("http://127.0.0.1:{port}");
    let settings = store.settings();
    let synth = Arc::new(RealSynth::new(args.data_dir.clone(), args.hub_dir()));
    // トークンは画面にだけ埋め込む使い捨て（乱数 2 つ分）。
    let state = AppState::new(
        store,
        args.data_dir.clone(),
        format!("{}{}", uid(), uid()),
        settings.backend.clone(),
        synth,
        Arc::new(RealInstaller::new(args.hub_dir())),
    );
    println!("Easy-Irodori-TTS: {url}");
    tracing::info!(
        "listening on {url} (data: {}, models: {})",
        args.data_dir.display(),
        args.hub_dir().display()
    );
    if !args.no_browser && settings.auto_browser {
        open_browser_when_ready(url);
    }
    let listener = tokio::net::TcpListener::from_std(listener)?;
    axum::serve(listener, build_router(state)).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use irodori_store::model::{new_project, Project};
    use irodori_store::Store;
    use serde_json::json;
    use std::net::TcpListener;

    /// ポートを探すテストは、互いの確保・解放に当たらないよう 1 つずつ動かす。
    static PORTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn data_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        prepare_dirs(dir.path()).unwrap();
        dir
    }

    fn store_with_text(dir: &tempfile::TempDir, text: &str) -> (Store, String) {
        let store = Store::open(dir.path()).unwrap();
        let mut doc = new_project("前回");
        doc.cards[0].text = text.into();
        store.persist(&doc).unwrap();
        (store, doc.id)
    }

    #[test]
    fn args_defaults() {
        let args = Args::parse_from(["irodori-app"]);
        assert_eq!((args.port, args.no_browser), (7860, false));
        assert_eq!(args.data_dir, default_data_dir());
        assert_eq!(
            args.hub_dir(),
            default_data_dir()
                .join("models")
                .join("huggingface")
                .join("hub")
        );
        let args = Args::parse_from(["irodori-app", "--data-dir", "D", "--models-dir", "M"]);
        assert_eq!(
            (args.hub_dir(), args.data_dir),
            (PathBuf::from("M"), PathBuf::from("D"))
        );
        let args = Args::parse_from(["irodori-app", "--data-dir", "D"]);
        assert_eq!(
            args.hub_dir(),
            PathBuf::from("D")
                .join("models")
                .join("huggingface")
                .join("hub")
        );
        let args = Args::parse_from([
            "irodori-app",
            "--port",
            "9000",
            "--no-browser",
            "--data-dir",
            "X",
        ]);
        assert_eq!(
            (args.port, args.no_browser, args.data_dir),
            (9000, true, "X".into())
        );
    }

    #[test]
    fn prepare_dirs_creates_the_v2_layout_and_logs() {
        let dir = tempfile::tempdir().unwrap();
        prepare_dirs(dir.path()).unwrap();
        for sub in ["projects", "voices", "config", "temp", "outputs", "logs"] {
            assert!(dir.path().join(sub).is_dir(), "{sub}");
        }
    }

    #[test]
    fn pick_port_returns_the_preferred_port_when_it_is_free() {
        let _serial = PORTS.lock().unwrap_or_else(|e| e.into_inner());
        let free = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        assert_eq!(pick_port(free).unwrap(), free);
    }

    #[test]
    fn pick_port_skips_a_busy_port() {
        let _serial = PORTS.lock().unwrap_or_else(|e| e.into_inner());
        let busy = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = busy.local_addr().unwrap().port();
        let picked = pick_port(port).unwrap();
        assert!(
            picked > port && picked < port.saturating_add(PORT_SEARCH),
            "picked {picked} for busy {port}"
        );
    }

    #[test]
    fn second_lock_is_refused_with_a_clear_message() {
        let dir = data_dir();
        let _first = acquire_lock(dir.path()).unwrap();
        let err = acquire_lock(dir.path())
            .err()
            .expect("second lock must fail");
        assert_eq!(err.to_string(), "アプリはすでに起動しています。");
    }

    #[test]
    fn lock_is_released_on_drop() {
        let dir = data_dir();
        drop(acquire_lock(dir.path()).unwrap());
        assert!(acquire_lock(dir.path()).is_ok());
    }

    #[test]
    fn fresh_start_records_a_recovery_project_only_when_text_exists() {
        let dir = data_dir();
        let (store, old) = store_with_text(&dir, "前回の入力");
        fresh_start(&store).unwrap();
        let settings = store.settings();
        assert_eq!(settings.extra.get("recovery_project"), Some(&json!(old)));
        assert_ne!(settings.last_project, old, "a new project is opened");
        let current: Project = store.load(None).unwrap();
        assert_eq!(current.cards[0].text, "");
        assert_eq!(
            store.load(Some(&old)).unwrap().cards[0].text,
            "前回の入力",
            "the old project is untouched"
        );
    }

    #[test]
    fn fresh_start_does_not_record_blank_previous_input() {
        let dir = data_dir();
        let (store, old) = store_with_text(&dir, "  \n ");
        fresh_start(&store).unwrap();
        assert!(store.settings().extra.get("recovery_project").is_none());
        assert_ne!(store.settings().last_project, old);
    }

    #[test]
    fn fresh_start_is_skipped_when_restore_input_is_true() {
        let dir = data_dir();
        let (store, old) = store_with_text(&dir, "前回の入力");
        store
            .save_settings(&json!({"restore_input": true}))
            .unwrap();
        fresh_start(&store).unwrap();
        assert_eq!(store.settings().last_project, old);
        assert!(store.settings().extra.get("recovery_project").is_none());
    }

    #[test]
    fn fresh_start_survives_a_missing_previous_project() {
        let dir = data_dir();
        let (_store, old) = store_with_text(&dir, "消える");
        std::fs::remove_file(dir.path().join(format!("projects/{old}.json"))).unwrap();
        let store = Store::open(dir.path()).unwrap();
        fresh_start(&store).unwrap();
        assert_ne!(store.settings().last_project, old);
    }

    #[test]
    fn lock_is_refused_while_v2_holds_the_environment_lock() {
        // v2.0（Python）は同じ `temp/environment.lock` を取る。同じデータフォルダを同時に書くと、後から書いた側が勝って編集が消える。
        let dir = data_dir();
        let path = dir.path().join("temp/environment.lock");
        let v2 = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .unwrap();
        v2.try_lock().unwrap();
        let err = acquire_lock(dir.path()).expect_err("v2.0 is running");
        assert_eq!(err.to_string(), "アプリはすでに起動しています。");
        drop(v2);
        assert!(acquire_lock(dir.path()).is_ok(), "released once v2.0 stops");
    }

    #[test]
    fn running_app_blocks_v2_from_starting_on_the_same_folder() {
        let dir = data_dir();
        let _guard = acquire_lock(dir.path()).unwrap();
        let v2 = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(dir.path().join("temp/environment.lock"))
            .unwrap();
        assert!(
            v2.try_lock().is_err(),
            "v2.0 must see the environment lock as taken"
        );
    }
}
