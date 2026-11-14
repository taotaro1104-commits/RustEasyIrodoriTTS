//! 偽の Hugging Face。HF の API とファイル配信（Range・リダイレクト）を、ローカルの別スレッドで真似る。
//! ネットワークには出ない。

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Router;
use sha2::{Digest, Sha256};

pub const SHA: &str = "0123456789abcdef0123456789abcdef01234567";

#[derive(Clone)]
pub struct FakeFile {
    pub repo: String,
    pub path: String,
    pub bytes: Vec<u8>,
    /// LFS のファイル（API が `lfs.oid` = SHA-256 を返す）。
    pub lfs: bool,
}

impl FakeFile {
    pub fn new(repo: &str, path: &str, bytes: &[u8], lfs: bool) -> FakeFile {
        FakeFile {
            repo: repo.into(),
            path: path.into(),
            bytes: bytes.to_vec(),
            lfs,
        }
    }

    pub fn sha256(&self) -> String {
        Sha256::digest(&self.bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Recorded {
    pub path: String,
    pub range: Option<String>,
}

#[derive(Default)]
struct Behavior {
    ignore_range: bool,
    /// 先頭の N バイトだけ送って、接続を切る（Content-Length は全体のまま）。
    truncate_after: Option<usize>,
    html_with_200: bool,
    redirect_to: Option<String>,
    redirect_internally: bool,
    /// 何度でも自分自身へリダイレクトする。
    redirect_loop: bool,
    /// パスの末尾が一致したら、この状態コードを返す。
    status: HashMap<String, u16>,
    sha: Option<String>,
    lfs_oid: Option<String>,
    decoys: bool,
    /// 先頭の N バイトだけ送り、そのあと止めて、解放の合図まで待つ（取消のテスト用）。
    stall_after: Option<usize>,
}

struct Shared {
    files: Vec<FakeFile>,
    behavior: Behavior,
    requests: Vec<Recorded>,
    released: HashSet<u8>,
}

pub struct FakeHub {
    addr: SocketAddr,
    shared: Arc<Mutex<Shared>>,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl FakeHub {
    pub fn start(files: Vec<FakeFile>) -> FakeHub {
        let shared = Arc::new(Mutex::new(Shared {
            files,
            behavior: Behavior::default(),
            requests: Vec::new(),
            released: HashSet::new(),
        }));
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let state = Arc::clone(&shared);
        let thread = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async move {
                let listener = tokio::net::TcpListener::from_std(listener).unwrap();
                let app = Router::new().fallback(handle).with_state(state);
                axum::serve(listener, app)
                    .with_graceful_shutdown(async {
                        let _ = rx.await;
                    })
                    .await
                    .ok();
            });
        });
        FakeHub {
            addr,
            shared,
            shutdown: Some(tx),
            thread: Some(thread),
        }
    }

    pub fn port(&self) -> u16 {
        self.addr.port()
    }

    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub fn requests(&self) -> Vec<Recorded> {
        self.shared.lock().unwrap().requests.clone()
    }

    /// 記録した要求のうち、パスに `part` を含むもの。
    pub fn requests_matching(&self, part: &str) -> Vec<Recorded> {
        self.requests()
            .into_iter()
            .filter(|r| r.path.contains(part))
            .collect()
    }

    fn set(&self, f: impl FnOnce(&mut Behavior)) {
        f(&mut self.shared.lock().unwrap().behavior);
    }

    pub fn ignore_range(&self) {
        self.set(|b| b.ignore_range = true);
    }

    pub fn truncate_after(&self, bytes: Option<usize>) {
        self.set(|b| b.truncate_after = bytes);
    }

    pub fn serve_html_with_200(&self) {
        self.set(|b| b.html_with_200 = true);
    }

    /// `resolve` を、別のホストへのリダイレクトにする（`host` は `evil.example` など）。
    pub fn redirect_to(&self, host: &str) {
        let host = host.to_string();
        self.set(|b| b.redirect_to = Some(host));
    }

    /// `resolve` を、このサーバ内の別のパスへのリダイレクトにする（許可ホストへの追従の確認）。
    pub fn redirect_internally(&self) {
        self.set(|b| b.redirect_internally = true);
    }

    pub fn redirect_forever(&self) {
        self.set(|b| b.redirect_loop = true);
    }

    pub fn status(&self, path_suffix: &str, code: u16) {
        let key = path_suffix.to_string();
        self.set(|b| {
            b.status.insert(key, code);
        });
    }

    pub fn clear_status(&self) {
        self.set(|b| b.status.clear());
    }

    pub fn commit_sha(&self, sha: &str) {
        let sha = sha.to_string();
        self.set(|b| b.sha = Some(sha));
    }

    pub fn lfs_oid(&self, oid: &str) {
        let oid = oid.to_string();
        self.set(|b| b.lfs_oid = Some(oid));
    }

    /// 一覧に、頼んでいないファイルと `../evil` のような名前も混ぜる。
    pub fn add_decoys(&self) {
        self.set(|b| b.decoys = true);
    }

    pub fn clear_faults(&self) {
        self.set(|b| *b = Behavior::default());
    }

    /// 先頭の N バイトを送ったあと、`release` が呼ばれるまで本文を止める。
    pub fn stall_after(&self, bytes: usize) {
        self.set(|b| b.stall_after = Some(bytes));
    }

    pub fn release(&self) {
        self.shared.lock().unwrap().released.insert(0);
    }

    /// 配信中のファイルの中身を差し替える（すり替えのテスト）。
    pub fn replace_bytes(&self, repo: &str, path: &str, bytes: &[u8]) {
        let mut shared = self.shared.lock().unwrap();
        for f in shared
            .files
            .iter_mut()
            .filter(|f| f.repo == repo && f.path == path)
        {
            f.bytes = bytes.to_vec();
        }
    }
}

impl Drop for FakeHub {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn json_response(value: serde_json::Value) -> Response {
    (
        [(header::CONTENT_TYPE, "application/json")],
        value.to_string(),
    )
        .into_response()
}

fn status_of(code: u16) -> Response {
    StatusCode::from_u16(code)
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
        .into_response()
}

fn parse_range(value: &str) -> Option<usize> {
    value
        .strip_prefix("bytes=")?
        .strip_suffix('-')?
        .parse()
        .ok()
}

async fn handle(State(shared): State<Arc<Mutex<Shared>>>, request: Request) -> Response {
    let path = request.uri().path().to_string();
    let range = request
        .headers()
        .get(header::RANGE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let (
        files,
        forced,
        redirect_to,
        internal,
        forever,
        ignore_range,
        truncate,
        html,
        sha,
        oid,
        decoys,
        stall,
    ) = {
        let mut s = shared.lock().unwrap();
        s.requests.push(Recorded {
            path: path.clone(),
            range: range.clone(),
        });
        let b = &s.behavior;
        let forced = b
            .status
            .iter()
            .find(|(suffix, _)| path.ends_with(suffix.as_str()))
            .map(|(_, c)| *c);
        (
            s.files.clone(),
            forced,
            b.redirect_to.clone(),
            b.redirect_internally,
            b.redirect_loop,
            b.ignore_range,
            b.truncate_after,
            b.html_with_200,
            b.sha.clone().unwrap_or_else(|| SHA.to_string()),
            b.lfs_oid.clone(),
            b.decoys,
            b.stall_after,
        )
    };
    if let Some(code) = forced {
        return status_of(code);
    }
    let segments: Vec<&str> = path.trim_start_matches('/').split('/').collect();

    // GET /api/models/<org>/<name>  → {"sha": ...}
    if segments.len() == 4 && segments[..2] == ["api", "models"] {
        let repo = format!("{}/{}", segments[2], segments[3]);
        if !files.iter().any(|f| f.repo == repo) {
            return status_of(404);
        }
        return json_response(serde_json::json!({"id": repo, "sha": sha}));
    }
    // GET /api/models/<org>/<name>/tree/<sha>
    if segments.len() == 6 && segments[..2] == ["api", "models"] && segments[4] == "tree" {
        let repo = format!("{}/{}", segments[2], segments[3]);
        let mut entries: Vec<serde_json::Value> = files
            .iter()
            .filter(|f| f.repo == repo)
            .map(|f| {
                let mut e = serde_json::json!({"type": "file", "path": f.path, "size": f.bytes.len(), "oid": "1111111111111111111111111111111111111111"});
                if f.lfs {
                    e["lfs"] = serde_json::json!({"oid": oid.clone().unwrap_or_else(|| f.sha256()), "size": f.bytes.len(), "pointerSize": 134});
                }
                e
            })
            .collect();
        if entries.is_empty() {
            return status_of(404);
        }
        entries.push(serde_json::json!({"type": "directory", "path": "tokenizer"}));
        if decoys {
            for evil in ["../evil.bin", "README.md", "..\\evil2", "/abs/path.bin"] {
                entries.push(serde_json::json!({"type": "file", "path": evil, "size": 3}));
            }
        }
        return json_response(serde_json::Value::Array(entries));
    }
    // GET /<org>/<name>/resolve/<sha>/<path...>
    if segments.len() >= 5 && segments[2] == "resolve" {
        let repo = format!("{}/{}", segments[0], segments[1]);
        let file_path = segments[4..].join("/");
        if forever {
            return redirect(&format!("http://{}{}", host_of(&request), path));
        }
        if let Some(host) = redirect_to {
            return redirect(&format!("http://{host}/x/{file_path}"));
        }
        if internal {
            return redirect(&format!("/cdn/{repo}/{file_path}"));
        }
        return serve_file(
            &files,
            &repo,
            &file_path,
            range,
            ignore_range,
            truncate,
            html,
            stall,
            &shared,
        )
        .await;
    }
    // GET /cdn/<org>/<name>/<path...>  （リダイレクト先）
    if segments.len() >= 4 && segments[0] == "cdn" {
        let repo = format!("{}/{}", segments[1], segments[2]);
        let file_path = segments[3..].join("/");
        return serve_file(
            &files,
            &repo,
            &file_path,
            range,
            ignore_range,
            truncate,
            html,
            stall,
            &shared,
        )
        .await;
    }
    status_of(404)
}

fn host_of(request: &Request) -> String {
    request
        .headers()
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("127.0.0.1")
        .to_string()
}

fn redirect(location: &str) -> Response {
    let mut response = StatusCode::FOUND.into_response();
    response
        .headers_mut()
        .insert(header::LOCATION, HeaderValue::from_str(location).unwrap());
    response
}

#[allow(clippy::too_many_arguments)]
async fn serve_file(
    files: &[FakeFile],
    repo: &str,
    path: &str,
    range: Option<String>,
    ignore_range: bool,
    truncate: Option<usize>,
    html: bool,
    stall: Option<usize>,
    shared: &Arc<Mutex<Shared>>,
) -> Response {
    let Some(file) = files.iter().find(|f| f.repo == repo && f.path == path) else {
        return status_of(404);
    };
    if html {
        let page = b"<html><body>Service temporarily unavailable</body></html>".to_vec();
        return ([(header::CONTENT_TYPE, "text/html")], page).into_response();
    }
    let total = file.bytes.len();
    let start = if ignore_range {
        0
    } else {
        range.as_deref().and_then(parse_range).unwrap_or(0)
    };
    if start > total {
        return status_of(416);
    }
    let rest = file.bytes[start..].to_vec();
    let mut builder = Response::builder().header(header::CONTENT_LENGTH, rest.len());
    if start > 0 {
        builder = builder.status(StatusCode::PARTIAL_CONTENT).header(
            header::CONTENT_RANGE,
            format!("bytes {start}-{}/{total}", total - 1),
        );
    } else {
        builder = builder.status(StatusCode::OK);
    }
    let limit = truncate.or(stall);
    let body = match limit {
        Some(n) if n < rest.len() => {
            let head = rest[..n].to_vec();
            let stalls = stall.is_some() && truncate.is_none();
            let shared = Arc::clone(shared);
            let stream = futures_stream(head, stalls, shared);
            Body::from_stream(stream)
        }
        _ => Body::from(rest),
    };
    builder.body(body).unwrap()
}

/// 先頭を送ったあと、止める（`stalls`）か、エラーで接続を切る。
fn futures_stream(
    head: Vec<u8>,
    stalls: bool,
    shared: Arc<Mutex<Shared>>,
) -> impl futures_util::Stream<Item = Result<Vec<u8>, std::io::Error>> {
    futures_util::stream::unfold(0u8, move |state| {
        let head = head.clone();
        let shared = Arc::clone(&shared);
        async move {
            match state {
                0 => Some((Ok(head), 1)),
                1 if stalls => {
                    loop {
                        if shared.lock().unwrap().released.contains(&0) {
                            break;
                        }
                        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    }
                    None
                }
                1 => Some((Err(std::io::Error::other("connection cut")), 2)),
                _ => None,
            }
        }
    })
}
