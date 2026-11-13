//! 実際のバイナリを起動して確かめる。

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// ポート探しが互いのアプリに当たらないよう、このファイルのテストは 1 つずつ動かす。
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn spawn(dir: &std::path::Path, port: u16) -> Child {
    Command::new(env!("CARGO_BIN_EXE_irodori-app"))
        .args([
            "--data-dir",
            dir.to_str().unwrap(),
            "--port",
            &port.to_string(),
            "--no-browser",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
}

/// 簡単な GET を 1 回。`(状態コード, 本文)`。
fn try_get(port: u16, path: &str) -> Option<(u16, String)> {
    let mut stream = TcpStream::connect_timeout(
        &format!("127.0.0.1:{port}").parse().unwrap(),
        Duration::from_millis(1000),
    )
    .ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(3))).ok()?;
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
    )
    .ok()?;
    // 相手が閉じるときの切断エラーは無視し、受け取れた分で判定する。
    let mut bytes = Vec::new();
    let _ = stream.read_to_end(&mut bytes);
    let raw = String::from_utf8_lossy(&bytes).into_owned();
    let status = raw.split_whitespace().nth(1)?.parse().ok()?;
    Some((
        status,
        raw.split("\r\n\r\n").nth(1).unwrap_or("").to_string(),
    ))
}

/// 起動直後や負荷の高いときの接続失敗に備え、数回やり直す。
fn http_get(port: u16, path: &str) -> Option<(u16, String)> {
    for _ in 0..30 {
        if let Some(reply) = try_get(port, path) {
            return Some(reply);
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    None
}

fn wait_for_state(port: u16, child: &mut Child) -> (u16, String) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if let Some(status) = child.try_wait().unwrap() {
            panic!("the app exited early: {status}");
        }
        if let Some(reply) = try_get(port, "/api/state") {
            return reply;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("the app did not start in time");
}

struct Kill(Child);
impl Drop for Kill {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn the_binary_serves_state_and_refuses_a_second_instance() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let mut app = Kill(spawn(dir.path(), port));
    let (status, body) = wait_for_state(port, &mut app.0);
    assert_eq!(status, 200, "{body}");
    let state: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(state["project"]["cards"].as_array().unwrap().len(), 1);
    assert_eq!(http_get(port, "/").unwrap().0, 200);

    // 同じデータフォルダでの 2 つ目は断られる。
    let second = Command::new(env!("CARGO_BIN_EXE_irodori-app"))
        .args([
            "--data-dir",
            dir.path().to_str().unwrap(),
            "--port",
            &free_port().to_string(),
            "--no-browser",
        ])
        .output()
        .unwrap();
    assert!(!second.status.success());
    assert!(
        String::from_utf8_lossy(&second.stderr).contains("アプリはすでに起動しています。"),
        "{}",
        String::from_utf8_lossy(&second.stderr)
    );

    // ログには、成功した状態取得を残さない。
    let log = std::fs::read_to_string(dir.path().join("logs/editor.log")).unwrap();
    assert!(log.contains("GET / -> 200"), "{log}");
    assert!(!log.contains("/api/state -> 200"), "{log}");
}

#[test]
fn a_busy_port_moves_to_the_next_one() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let busy = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = busy.local_addr().unwrap().port();
    let mut app = Kill(spawn(dir.path(), port));
    // 起動時に標準出力へ出す URL の行から、実際のポートを知る。
    let mut line = String::new();
    BufReader::new(app.0.stdout.take().unwrap())
        .read_line(&mut line)
        .unwrap();
    let actual: u16 = line
        .trim()
        .rsplit(':')
        .next()
        .unwrap()
        .parse()
        .unwrap_or_else(|_| panic!("unexpected first line: {line:?}"));
    assert!(
        actual > port && actual < port + 100,
        "listening on {actual} although {port} is busy"
    );
    assert_eq!(http_get(actual, "/api/state").map(|r| r.0), Some(200));
}

#[test]
fn a_fatal_startup_error_is_written_to_the_log() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("config")).unwrap();
    std::fs::write(dir.path().join("config/settings.json"), b"{broken").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_irodori-app"))
        .args([
            "--data-dir",
            dir.path().to_str().unwrap(),
            "--port",
            &free_port().to_string(),
            "--no-browser",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("保存データを読み込めません"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let log = std::fs::read_to_string(dir.path().join("logs/editor.log")).unwrap();
    assert!(
        log.contains("fatal") && log.contains("保存データを読み込めません"),
        "the cause is kept for later: {log}"
    );
    assert_eq!(
        std::fs::read(dir.path().join("config/settings.json")).unwrap(),
        b"{broken",
        "the broken file is never overwritten"
    );
    assert!(
        !dir.path()
            .join("projects")
            .read_dir()
            .unwrap()
            .any(|e| e.is_ok()),
        "no empty project was created by the failed start"
    );
}
