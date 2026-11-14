//! Hugging Face のクライアント。ファイル一覧・サイズ・SHA-256 を API で得て、取得計画（`RepoPlan`）を作る。
//!
//! 通信先は許可ホスト（`huggingface.co` / `hf.co` とそのサブドメイン）の HTTPS だけ。サーバが返すパスは保存先に使わず、
//! 頼んだファイルの名前だけを拾う。

use std::io::Read;
use std::time::Duration;

use serde_json::Value;
use url::{Host, Url};

use super::DownloadError;

pub const HF_BASE: &str = "https://huggingface.co";
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const READ_TIMEOUT: Duration = Duration::from_secs(30);
/// メタデータ（JSON）の大きさの上限。これを超える応答は読まずに拒否する。
const MAX_JSON_BYTES: u64 = 8 * 1024 * 1024;
/// 重みのファイル。SHA-256 が分からない取得は受け付けない。
const HASH_REQUIRED_EXTENSIONS: [&str; 2] = [".safetensors", ".pth"];

/// どのホストへ出てよいか。`Local` はテスト専用（`127.0.0.1` だけを許す）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostPolicy {
    HuggingFace,
    Local,
}

/// 取得するファイル 1 つの計画。
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedFile {
    pub path: String,
    pub size: u64,
    /// LFS のファイルだけ（SHA-256 の 16 進、小文字 64 桁）。
    pub sha256: Option<String>,
}

/// 1 つのリポジトリの取得計画（固定したコミットのファイル一覧）。
#[derive(Debug, Clone, PartialEq)]
pub struct RepoPlan {
    pub repo: String,
    pub sha: String,
    pub files: Vec<PlannedFile>,
}

pub struct Hub {
    pub(crate) base: String,
    pub(crate) policy: HostPolicy,
    pub(crate) agent: ureq::Agent,
}

/// その URL へ要求を出してよいか（スキーム・ユーザー情報・ホスト）。
pub fn host_allowed(url: &str, policy: HostPolicy) -> bool {
    let Ok(parsed) = Url::parse(url) else {
        return false;
    };
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return false;
    }
    match policy {
        HostPolicy::HuggingFace => {
            if parsed.scheme() != "https" || parsed.port().is_some() {
                return false;
            }
            match parsed.host() {
                Some(Host::Domain(d)) => {
                    d == "huggingface.co"
                        || d.ends_with(".huggingface.co")
                        || d == "hf.co"
                        || d.ends_with(".hf.co")
                }
                _ => false,
            }
        }
        HostPolicy::Local => {
            matches!(parsed.scheme(), "http" | "https")
                && parsed.host() == Some(Host::Ipv4(std::net::Ipv4Addr::LOCALHOST))
        }
    }
}

fn is_hex(s: &str, len: usize) -> bool {
    s.len() == len && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// ログに出す URL から、署名つきクエリ・ユーザー情報・フラグメントを落とす。
pub(crate) fn redact_url(url: &str) -> String {
    match Url::parse(url) {
        Ok(mut parsed) => {
            parsed.set_query(None);
            parsed.set_fragment(None);
            let _ = parsed.set_username("");
            let _ = parsed.set_password(None);
            parsed.to_string()
        }
        Err(_) => "(invalid url)".to_string(),
    }
}

/// `limit` バイトまでの JSON を読む。超えたら拒否する（巨大な応答でメモリを使い切られないため）。
pub(crate) fn read_json_limited(reader: impl Read, limit: u64) -> Result<Value, DownloadError> {
    let mut body = Vec::new();
    reader.take(limit + 1).read_to_end(&mut body).map_err(|e| {
        tracing::warn!("download read error: {e}");
        DownloadError::Offline
    })?;
    if body.len() as u64 > limit {
        return Err(protocol(format!(
            "the response is larger than {limit} bytes"
        )));
    }
    serde_json::from_slice(&body).map_err(|e| protocol(format!("invalid json: {e}")))
}

fn protocol(detail: impl Into<String>) -> DownloadError {
    DownloadError::Protocol(detail.into())
}

/// ureq のエラーを取得のエラーへ。状態コードは `Http`、それ以外（接続できない等）は `Offline`。
pub(crate) fn map_ureq(error: ureq::Error) -> DownloadError {
    match error {
        ureq::Error::Status(code, _) => DownloadError::Http(code),
        ureq::Error::Transport(t) => {
            // ureq の表示は署名つき URL を含むので、種類と本文だけを記録する。
            tracing::warn!(
                "download transport error: {:?} {}",
                t.kind(),
                t.message().unwrap_or("")
            );
            DownloadError::Offline
        }
    }
}

impl Hub {
    pub fn huggingface() -> Hub {
        Hub::with_base(HF_BASE, HostPolicy::HuggingFace)
    }

    pub fn with_base(base: &str, policy: HostPolicy) -> Hub {
        let agent = ureq::AgentBuilder::new()
            .redirects(0)
            .timeout_connect(CONNECT_TIMEOUT)
            .timeout_read(READ_TIMEOUT)
            .user_agent("easy-irodori-tts-rust")
            .build();
        Hub {
            base: base.trim_end_matches('/').to_string(),
            policy,
            agent,
        }
    }

    pub(crate) fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    fn get_json(&self, url: &str) -> Result<Value, DownloadError> {
        if !host_allowed(url, self.policy) {
            return Err(protocol(format!("host not allowed: {}", redact_url(url))));
        }
        let response = self.agent.get(url).call().map_err(map_ureq)?;
        read_json_limited(response.into_reader(), MAX_JSON_BYTES)
            .map_err(|e| protocol(format!("{} ({})", e, redact_url(url))))
    }

    /// `repo` の最新コミットを固定し、`wanted` のファイルだけの取得計画を作る。
    pub fn plan(&self, repo: &str, wanted: &[&str]) -> Result<RepoPlan, DownloadError> {
        let info = self.get_json(&self.url(&format!("/api/models/{repo}")))?;
        let sha = info["sha"].as_str().unwrap_or_default();
        if !is_hex(sha, 40) {
            return Err(protocol(format!("malformed commit sha {sha:?}")));
        }
        let tree =
            self.get_json(&self.url(&format!("/api/models/{repo}/tree/{sha}?recursive=1")))?;
        let entries = tree
            .as_array()
            .ok_or_else(|| protocol("the file list is not an array"))?;
        let mut files = Vec::with_capacity(wanted.len());
        for path in wanted {
            let entry = entries
                .iter()
                .find(|e| e["type"] == "file" && e["path"].as_str() == Some(path))
                .ok_or_else(|| protocol(format!("{repo} has no file {path}")))?;
            let size = entry["size"]
                .as_u64()
                .ok_or_else(|| protocol(format!("{path} has no size")))?;
            let sha256 = match entry["lfs"]["oid"].as_str() {
                None => None,
                Some(oid) if is_hex(oid, 64) => Some(oid.to_string()),
                Some(oid) => return Err(protocol(format!("malformed sha256 {oid:?} for {path}"))),
            };
            if sha256.is_none() && HASH_REQUIRED_EXTENSIONS.iter().any(|x| path.ends_with(x)) {
                return Err(protocol(format!("{path} has no sha256 to verify against")));
            }
            files.push(PlannedFile {
                path: (*path).to_string(),
                size,
                sha256,
            });
        }
        Ok(RepoPlan {
            repo: repo.to_string(),
            sha: sha.to_string(),
            files,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_https_huggingface_hosts_are_allowed() {
        let ok = |u: &str| host_allowed(u, HostPolicy::HuggingFace);
        assert!(ok("https://huggingface.co/api/models/x/y"));
        assert!(ok("https://cdn-lfs.hf.co/repos/aa/bb"));
        assert!(ok(
            "https://cas-bridge.xethub.hf.co/xet-bridge-us/abc?X-Amz=1"
        ));
        assert!(ok("https://cdn-lfs.huggingface.co/x"));
        assert!(ok("https://hf.co/x"));
        for bad in [
            "http://huggingface.co/x",
            "https://evilhuggingface.co/x",
            "https://huggingface.co.evil.com/x",
            "https://user@huggingface.co/x",
            "https://user:pw@huggingface.co/x",
            "https://127.0.0.1/x",
            "https://[::1]/x",
            "https://hf.co.evil.com/",
            "https://nothf.co/",
            "ftp://huggingface.co/x",
            "not a url",
            "",
        ] {
            assert!(!ok(bad), "{bad}");
        }
    }

    #[test]
    fn the_local_policy_only_allows_loopback() {
        let ok = |u: &str| host_allowed(u, HostPolicy::Local);
        assert!(ok("http://127.0.0.1:4000/x"));
        assert!(!ok("http://evil.example/x"));
        assert!(!ok("http://huggingface.co/x"));
        assert!(!ok("http://user@127.0.0.1/x"));
    }

    #[test]
    fn only_the_default_https_port_is_allowed() {
        let ok = |u: &str| host_allowed(u, HostPolicy::HuggingFace);
        assert!(ok("https://huggingface.co:443/x"));
        for bad in [
            "https://huggingface.co:8443/x",
            "https://cdn-lfs.hf.co:1/x",
            "https://hf.co:80/x",
        ] {
            assert!(!ok(bad), "{bad}");
        }
    }

    #[test]
    fn json_larger_than_the_limit_is_refused() {
        let small = br#"{"a":[1,2,3]}"#;
        assert_eq!(read_json_limited(&small[..], 64).unwrap()["a"][2], 3);
        let big = format!("[{}]", "1,".repeat(100) + "1");
        let err = read_json_limited(big.as_bytes(), 64).unwrap_err();
        assert!(matches!(err, DownloadError::Protocol(_)), "{err:?}");
        let exact = br#"[1]"#;
        assert!(read_json_limited(&exact[..], 3).is_ok());
        assert!(read_json_limited(&exact[..], 2).is_err());
    }

    #[test]
    fn redact_drops_the_query_and_userinfo_from_logged_urls() {
        assert_eq!(
            redact_url("https://cas.hf.co/x/y?X-Amz-Signature=secret&a=b#frag"),
            "https://cas.hf.co/x/y"
        );
        assert_eq!(redact_url("https://u:p@hf.co/x"), "https://hf.co/x");
        assert_eq!(redact_url("not a url"), "(invalid url)");
    }

    #[test]
    fn hex_checks_are_strict() {
        assert!(is_hex(&"a".repeat(40), 40));
        assert!(!is_hex(&"A".repeat(40), 40));
        assert!(!is_hex(&"a".repeat(39), 40));
        assert!(!is_hex("../../", 6));
    }
}
