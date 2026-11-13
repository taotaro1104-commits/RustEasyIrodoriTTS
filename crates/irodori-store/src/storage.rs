//! 原子的な JSON 書き込みと、v2.0 互換の `Store`（台本・話者・設定）。
//!
//! `Store` は内部状態を `Mutex` で守る。編集クロージャの中から `Store` のメソッドを呼ばないこと
//! （ロックは再入不可。v2.0 の `RLock` とは異なる）。ファイルは読めなければエラーにし、**元のファイルを上書きしない**。

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use serde_json::{json, Value};

use crate::history::{input_state, History};
use crate::model::{new_project, uid, Project, Settings, Voice};
use crate::py_json::dumps_pretty;
use crate::{Result, StoreError};

const CORRUPT_PROJECT: &str = "台本を読み込めません。形式または保存場所を確認してください。";

fn invalid(msg: &str) -> StoreError {
    StoreError::Invalid(msg.into())
}

fn corrupt_file(path: &Path) -> StoreError {
    StoreError::Invalid(format!("保存データを読み込めません。元のファイルは変更していません: {}", path.display()))
}

/// 一時ファイルに書いて fsync し、rename で置き換える。失敗時は一時ファイルを消す。
pub fn atomic_json(path: &Path, v: &Value) -> Result<()> {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let text = dumps_pretty(v)?;
    let tmp = path.with_file_name(format!("{name}.{}.tmp", uid()));
    let result = (|| -> std::io::Result<()> {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(text.as_bytes())?;
        f.sync_all()?;
        std::fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result?;
    Ok(())
}

/// 無ければ `None`。BOM 付き UTF-8 も読める。壊れていれば元のファイルに触れずエラー。
pub fn read_json(path: &Path) -> Result<Option<Value>> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(path).map_err(|_| corrupt_file(path))?;
    let body = bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(&bytes);
    serde_json::from_slice(body).map(Some).map_err(|_| corrupt_file(path))
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProjectSummary {
    pub id: String,
    pub name: String,
}

pub(crate) struct Inner {
    pub root: PathBuf,
    pub settings: Settings,
    pub voices: Vec<Voice>,
    pub projects: HashMap<String, Project>,
    pub history: History,
}

pub struct Store {
    root: PathBuf,
    pub(crate) inner: Mutex<Inner>,
}

/// 32 桁の小文字 hex だけを許す（パス結合に使うため）。
pub fn valid_pid(pid: &str) -> bool {
    pid.len() == 32 && pid.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn project_path(root: &Path, pid: &str) -> Result<PathBuf> {
    if !valid_pid(pid) {
        return Err(invalid("台本IDが不正です。"));
    }
    Ok(root.join("projects").join(format!("{pid}.json")))
}

fn read_project(path: &Path, pid: &str) -> Result<Project> {
    let value = read_json(path)?;
    let obj = value.as_ref().and_then(Value::as_object).ok_or_else(|| invalid(CORRUPT_PROJECT))?;
    if obj.get("schema") != Some(&json!(2)) || obj.get("id").and_then(Value::as_str) != Some(pid) {
        return Err(invalid(CORRUPT_PROJECT));
    }
    let cards_ok = obj.get("cards").and_then(Value::as_array).is_some_and(|cards| {
        let ids: Vec<Option<&str>> = cards.iter().map(|c| c.get("id").and_then(Value::as_str)).collect();
        !ids.is_empty() && ids.iter().all(Option::is_some) && ids.iter().collect::<std::collections::HashSet<_>>().len() == ids.len()
    });
    if !cards_ok {
        return Err(invalid("台本のカード情報が不正です。"));
    }
    let mut project: Project = serde_json::from_value(value.unwrap()).map_err(|_| invalid(CORRUPT_PROJECT))?;
    for card in &mut project.cards {
        if card.status == "待機中" || card.status == "生成中" {
            card.status = "中断（再生成できます）".into();
        }
    }
    Ok(project)
}

impl Inner {
    pub(crate) fn load(&mut self, pid: Option<&str>) -> Result<Project> {
        let pid = match pid.filter(|p| !p.is_empty()) {
            Some(p) => p.to_string(),
            None => self.settings.last_project.clone(),
        };
        if pid.is_empty() {
            let doc = new_project("無題の台本");
            self.projects.insert(doc.id.clone(), doc.clone());
            self.persist(&doc)?;
            return Ok(doc);
        }
        if !self.projects.contains_key(&pid) {
            let doc = read_project(&project_path(&self.root, &pid)?, &pid)?;
            self.projects.insert(pid.clone(), doc);
        }
        Ok(self.projects[&pid].clone())
    }

    pub(crate) fn persist(&mut self, doc: &Project) -> Result<()> {
        let value = serde_json::to_value(doc).map_err(|e| StoreError::Internal(e.to_string()))?;
        atomic_json(&project_path(&self.root, &doc.id)?, &value)?;
        self.settings.last_project = doc.id.clone();
        self.write_settings()
    }

    pub(crate) fn write_settings(&self) -> Result<()> {
        let value = serde_json::to_value(&self.settings).map_err(|e| StoreError::Internal(e.to_string()))?;
        atomic_json(&self.root.join("config/settings.json"), &value)
    }
}

fn settings_invalid() -> StoreError {
    invalid("設定ファイルの形式が不正です。ファイルは変更していません。")
}

fn load_settings(root: &Path) -> Result<Settings> {
    let legacy = root.join("config/backend.txt");
    let backend = std::fs::read_to_string(&legacy).map(|t| t.trim_start_matches('\u{feff}').trim().to_string()).unwrap_or_else(|_| "cpu".into());
    let defaults = Settings::defaults(&backend);
    let saved = read_json(&root.join("config/settings.json"))?.unwrap_or_else(|| json!({}));
    let saved = saved.as_object().ok_or_else(settings_invalid)?;
    if saved.get("schema").map_or(false, |s| s != &json!(2)) {
        return Err(settings_invalid());
    }
    let mut merged = match serde_json::to_value(&defaults) {
        Ok(Value::Object(m)) => m,
        _ => return Err(settings_invalid()),
    };
    for (k, v) in saved {
        merged.insert(k.clone(), v.clone());
    }
    serde_json::from_value(Value::Object(merged)).map_err(|_| settings_invalid())
}

fn load_voices(root: &Path) -> Result<Vec<Voice>> {
    let value = read_json(&root.join("voices/index.json"))?.unwrap_or_else(|| json!([]));
    let bad = || invalid("話者一覧の形式が不正です。");
    let items = value.as_array().ok_or_else(bad)?;
    if items.iter().any(|v| v.get("id").is_none()) {
        return Err(bad());
    }
    serde_json::from_value(value).map_err(|_| bad())
}

impl Store {
    pub fn open(root: &Path) -> Result<Store> {
        let inner = Inner {
            root: root.to_path_buf(),
            settings: load_settings(root)?,
            voices: load_voices(root)?,
            projects: HashMap::new(),
            history: History::default(),
        };
        Ok(Store { root: root.to_path_buf(), inner: Mutex::new(inner) })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn settings(&self) -> Settings {
        self.lock().settings.clone()
    }

    pub fn voices(&self) -> Vec<Voice> {
        self.lock().voices.clone()
    }

    /// `pid` が `None`（または空）なら最後に開いた台本。無ければ新規作成して保存する。
    pub fn load(&self, pid: Option<&str>) -> Result<Project> {
        self.lock().load(pid)
    }

    pub fn persist(&self, doc: &Project) -> Result<()> {
        self.lock().persist(doc)
    }

    /// 楽観ロックつき更新。`revision` が合わなければ 409。変更が無ければ何も保存しない。
    pub fn update(&self, pid: &str, revision: i64, remember: bool, f: &mut dyn FnMut(&mut Project) -> Result<()>) -> Result<Project> {
        let mut inner = self.lock();
        let old = inner.load(Some(pid))?;
        if old.revision != revision {
            return Err(StoreError::Conflict("別の操作で台本が更新されています。保存済みの内容を読み直してください。".into()));
        }
        let mut doc = old.clone();
        f(&mut doc)?;
        if doc == old {
            return Ok(old);
        }
        let (old_input, changed) = (input_state(&old), input_state(&old) != input_state(&doc));
        doc.revision += 1;
        inner.persist(&doc)?;
        inner.projects.insert(pid.to_string(), doc.clone());
        if remember && changed {
            inner.history.push(pid, old_input);
        }
        inner.history.capture(pid, &old);
        Ok(doc)
    }

    /// 生成結果・状態の更新（編集操作ではないので revision は上げず、履歴にも積まない）。
    pub fn internal_update(&self, pid: &str, f: &mut dyn FnMut(&mut Project)) -> Result<Project> {
        let mut inner = self.lock();
        let mut doc = inner.load(Some(pid))?;
        f(&mut doc);
        inner.persist(&doc)?;
        inner.projects.insert(pid.to_string(), doc.clone());
        inner.history.capture(pid, &doc);
        Ok(doc)
    }

    pub fn list_projects(&self) -> Vec<ProjectSummary> {
        let dir = self.root.join("projects");
        let mut paths: Vec<PathBuf> = match std::fs::read_dir(&dir) {
            Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "json")).collect(),
            Err(_) => return Vec::new(),
        };
        paths.sort();
        paths
            .iter()
            .filter_map(|p| {
                let v = read_json(p).ok()??;
                Some(ProjectSummary { id: v.get("id")?.as_str()?.to_string(), name: v.get("name")?.as_str()?.to_string() })
            })
            .collect()
    }

    pub fn save_settings(&self, values: &Value) -> Result<()> {
        let mut inner = self.lock();
        let mut result = inner.settings.clone();
        let get = |k: &str| values.get(k);
        for key in ["auto_browser", "restore_input", "inherit", "auto_export", "custom_output", "custom_pattern", "txt"] {
            if let Some(v) = get(key) {
                let b = v.as_bool().ok_or_else(|| invalid("設定値が不正です。"))?;
                match key {
                    "auto_browser" => result.auto_browser = b,
                    "restore_input" => result.restore_input = b,
                    "inherit" => result.inherit = b,
                    "auto_export" => result.auto_export = b,
                    "custom_output" => result.custom_output = b,
                    "custom_pattern" => result.custom_pattern = b,
                    _ => result.txt = b,
                }
            }
        }
        let text_of = |v: &Value| v.as_str().map_or_else(|| v.to_string(), str::to_string);
        if !matches!(get("backend").map_or(result.backend.clone(), text_of).as_str(), "cpu" | "cu128") {
            return Err(invalid("CPUまたはNVIDIA GPUを選んでください。"));
        }
        if !matches!(get("theme").map_or(result.theme.clone(), text_of).as_str(), "light" | "dark" | "system") {
            return Err(invalid("画面の配色が不正です。"));
        }
        for key in ["backend", "output_dir", "pattern", "theme"] {
            if let Some(v) = get(key) {
                let s = text_of(v);
                match key {
                    "backend" => result.backend = s,
                    "output_dir" => result.output_dir = s,
                    "pattern" => result.pattern = s,
                    _ => result.theme = s,
                }
            }
        }
        if result.custom_output && !Path::new(&result.output_dir).is_absolute() {
            return Err(invalid("保存先は絶対パスで指定してください。"));
        }
        if result.custom_pattern && result.pattern.trim().is_empty() {
            return Err(invalid("ファイル名パターンを入力してください。"));
        }
        let saved = serde_json::to_value(&result).map_err(|e| StoreError::Internal(e.to_string()))?;
        atomic_json(&self.root.join("config/settings.json"), &saved)?;
        inner.settings = result;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Project, Voice};
    use crate::testutil::temp_root;
    use serde_json::json;

    const MESSAGE_CORRUPT: &str = "保存データを読み込めません。元のファイルは変更していません";

    fn open(dir: &tempfile::TempDir) -> Store {
        Store::open(dir.path()).unwrap()
    }

    fn write(dir: &tempfile::TempDir, rel: &str, bytes: &[u8]) {
        let p = dir.path().join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, bytes).unwrap();
    }

    #[test]
    fn atomic_json_leaves_no_temp_file_and_replaces_atomically() {
        let dir = temp_root();
        let path = dir.path().join("config/x.json");
        atomic_json(&path, &json!({"a": 1})).unwrap();
        atomic_json(&path, &json!({"a": 2, "b": "日本語"})).unwrap();
        let names: Vec<String> = std::fs::read_dir(dir.path().join("config")).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into()).collect();
        assert_eq!(names, vec!["x.json".to_string()]);
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text, crate::py_json::dumps_pretty(&json!({"a": 2, "b": "日本語"})).unwrap());
    }

    #[test]
    fn read_json_accepts_a_utf8_bom_and_missing_files_are_none() {
        let dir = temp_root();
        write(&dir, "config/bom.json", b"\xef\xbb\xbf{\"k\": 1}");
        assert_eq!(read_json(&dir.path().join("config/bom.json")).unwrap(), Some(json!({"k": 1})));
        assert_eq!(read_json(&dir.path().join("config/none.json")).unwrap(), None);
    }

    #[test]
    fn corrupt_project_is_reported_and_the_file_is_untouched() {
        let dir = temp_root();
        let pid = "0123456789abcdef0123456789abcdef";
        write(&dir, &format!("projects/{pid}.json"), b"{not json");
        let store = open(&dir);
        let err = store.load(Some(pid)).unwrap_err();
        assert!(err.message().starts_with(MESSAGE_CORRUPT), "{}", err.message());
        assert_eq!(std::fs::read(dir.path().join(format!("projects/{pid}.json"))).unwrap(), b"{not json");
    }

    #[test]
    fn corrupt_settings_and_voices_do_not_get_overwritten() {
        let dir = temp_root();
        write(&dir, "config/settings.json", b"garbage");
        assert!(Store::open(dir.path()).err().unwrap().message().starts_with(MESSAGE_CORRUPT));
        assert_eq!(std::fs::read(dir.path().join("config/settings.json")).unwrap(), b"garbage");
        let dir = temp_root();
        write(&dir, "voices/index.json", b"[1, 2");
        assert!(Store::open(dir.path()).is_err());
        assert_eq!(std::fs::read(dir.path().join("voices/index.json")).unwrap(), b"[1, 2");
        let dir = temp_root();
        write(&dir, "voices/index.json", br#"[{"no_id": true}]"#);
        assert_eq!(Store::open(dir.path()).err().unwrap().message(), "話者一覧の形式が不正です。");
    }

    #[test]
    fn settings_with_another_schema_are_refused() {
        let dir = temp_root();
        write(&dir, "config/settings.json", br#"{"schema": 3}"#);
        assert_eq!(Store::open(dir.path()).err().unwrap().message(), "設定ファイルの形式が不正です。ファイルは変更していません。");
    }

    #[test]
    fn pid_with_traversal_or_wrong_length_is_rejected() {
        let dir = temp_root();
        let store = open(&dir);
        for bad in ["../x", "..\\x", &"a".repeat(31), &"a".repeat(33), &"A".repeat(32), &"g".repeat(32), ""] {
            let e = store.load(Some(bad));
            if bad.is_empty() {
                continue; // 空は「未指定」扱い（last_project / 新規）
            }
            assert_eq!(e.err().unwrap().message(), "台本IDが不正です。", "{bad:?}");
        }
    }

    #[test]
    fn load_none_creates_and_remembers_a_project() {
        let dir = temp_root();
        let store = open(&dir);
        let p = store.load(None).unwrap();
        assert_eq!(p.cards.len(), 1);
        assert_eq!(store.settings().last_project, p.id);
        assert!(dir.path().join(format!("projects/{}.json", p.id)).is_file());
        let again = open(&dir).load(None).unwrap();
        assert_eq!(again.id, p.id);
    }

    #[test]
    fn update_bumps_revision_and_persists() {
        let dir = temp_root();
        let store = open(&dir);
        let p = store.load(None).unwrap();
        let updated = store.update(&p.id, 0, false, &mut |d: &mut Project| { d.name = "新しい名前".into(); Ok(()) }).unwrap();
        assert_eq!((updated.revision, updated.name.as_str()), (1, "新しい名前"));
        let reloaded = open(&dir).load(Some(&p.id)).unwrap();
        assert_eq!((reloaded.revision, reloaded.name.as_str()), (1, "新しい名前"));
    }

    #[test]
    fn stale_revision_is_a_conflict_and_changes_nothing() {
        let dir = temp_root();
        let store = open(&dir);
        let p = store.load(None).unwrap();
        store.update(&p.id, 0, false, &mut |d: &mut Project| { d.name = "A".into(); Ok(()) }).unwrap();
        let before = std::fs::read(dir.path().join(format!("projects/{}.json", p.id))).unwrap();
        let err = store.update(&p.id, 0, false, &mut |d: &mut Project| { d.name = "B".into(); Ok(()) }).unwrap_err();
        assert_eq!((err.status(), err.message()), (409, "別の操作で台本が更新されています。保存済みの内容を読み直してください。"));
        assert_eq!(std::fs::read(dir.path().join(format!("projects/{}.json", p.id))).unwrap(), before);
    }

    #[test]
    fn a_failing_edit_closure_changes_nothing() {
        let dir = temp_root();
        let store = open(&dir);
        let p = store.load(None).unwrap();
        let err = store.update(&p.id, 0, false, &mut |d: &mut Project| { d.name = "half".into(); Err(crate::StoreError::Invalid("x".into())) });
        assert!(err.is_err());
        let now = store.load(Some(&p.id)).unwrap();
        assert_eq!((now.revision, now.name.as_str()), (0, "無題の台本"));
    }

    #[test]
    fn no_op_update_keeps_the_revision() {
        let dir = temp_root();
        let store = open(&dir);
        let p = store.load(None).unwrap();
        let same = store.update(&p.id, 0, false, &mut |_d: &mut Project| Ok(())).unwrap();
        assert_eq!(same.revision, 0);
    }

    #[test]
    fn interrupted_cards_load_as_resumable() {
        let dir = temp_root();
        let mut p = crate::model::new_project("x");
        p.cards[0].status = "生成中".into();
        let mut second = crate::model::new_card(None);
        second.status = "待機中".into();
        p.cards.push(second);
        write(&dir, &format!("projects/{}.json", p.id), serde_json::to_string(&p).unwrap().as_bytes());
        let loaded = open(&dir).load(Some(&p.id)).unwrap();
        assert!(loaded.cards.iter().all(|c| c.status == "中断（再生成できます）"));
    }

    #[test]
    fn invalid_project_documents_are_refused_with_python_messages() {
        let dir = temp_root();
        let mut p = crate::model::new_project("x");
        let pid = p.id.clone();
        p.schema = 3;
        write(&dir, &format!("projects/{pid}.json"), serde_json::to_string(&p).unwrap().as_bytes());
        assert_eq!(open(&dir).load(Some(&pid)).err().unwrap().message(), "台本を読み込めません。形式または保存場所を確認してください。");
        p.schema = 2;
        p.cards.push(p.cards[0].clone());
        write(&dir, &format!("projects/{pid}.json"), serde_json::to_string(&p).unwrap().as_bytes());
        assert_eq!(open(&dir).load(Some(&pid)).err().unwrap().message(), "台本のカード情報が不正です。");
        p.cards.clear();
        write(&dir, &format!("projects/{pid}.json"), serde_json::to_string(&p).unwrap().as_bytes());
        assert_eq!(open(&dir).load(Some(&pid)).err().unwrap().message(), "台本のカード情報が不正です。");
        write(&dir, &format!("projects/{}.json", "b".repeat(32)), serde_json::to_string(&p).unwrap().as_bytes());
        assert_eq!(open(&dir).load(Some(&"b".repeat(32))).err().unwrap().message(), "台本を読み込めません。形式または保存場所を確認してください。");
    }

    #[test]
    fn internal_update_persists_without_bumping_the_revision() {
        let dir = temp_root();
        let store = open(&dir);
        let p = store.load(None).unwrap();
        let after = store.internal_update(&p.id, &mut |d: &mut Project| d.cards[0].status = "完了".into()).unwrap();
        assert_eq!((after.revision, after.cards[0].status.as_str()), (0, "完了"));
        assert_eq!(open(&dir).load(Some(&p.id)).unwrap().cards[0].status, "完了");
    }

    #[test]
    fn list_projects_returns_id_and_name_and_skips_broken_files() {
        let dir = temp_root();
        let store = open(&dir);
        let a = store.load(None).unwrap();
        write(&dir, &format!("projects/{}.json", "c".repeat(32)), b"broken");
        let list = store.list_projects();
        assert_eq!(list.len(), 1);
        assert_eq!((list[0].id.as_str(), list[0].name.as_str()), (a.id.as_str(), "無題の台本"));
    }

    #[test]
    fn legacy_backend_txt_sets_the_backend() {
        let dir = temp_root();
        write(&dir, "config/backend.txt", "\u{feff}cu128\n".as_bytes());
        assert_eq!(open(&dir).settings().backend, "cu128");
        let dir = temp_root();
        write(&dir, "config/backend.txt", b"xpu");
        assert_eq!(open(&dir).settings().backend, "cpu");
    }

    #[test]
    fn voices_are_loaded_with_unknown_fields_intact() {
        let dir = temp_root();
        write(&dir, "voices/index.json", br#"[{"id":"v1","revision":2,"name":"A","model":"m","audio":"voices/v1/a.wav","caption":"","future":1}]"#);
        let v: Vec<Voice> = open(&dir).voices();
        assert_eq!((v[0].id.as_str(), v[0].extra["future"].clone()), ("v1", json!(1)));
    }

    #[test]
    fn save_settings_validates_like_python() {
        let dir = temp_root();
        let store = open(&dir);
        let err = |v: serde_json::Value| store.save_settings(&v).unwrap_err().message().to_string();
        assert_eq!(err(json!({"auto_browser": "yes"})), "設定値が不正です。");
        assert_eq!(err(json!({"backend": "tpu"})), "CPUまたはNVIDIA GPUを選んでください。");
        assert_eq!(err(json!({"theme": "neon"})), "画面の配色が不正です。");
        assert_eq!(err(json!({"custom_output": true, "output_dir": "relative/dir"})), "保存先は絶対パスで指定してください。");
        assert_eq!(err(json!({"custom_pattern": true, "pattern": "   "})), "ファイル名パターンを入力してください。");
        store.save_settings(&json!({"theme": "dark", "txt": true, "pattern": "$連番$", "custom_pattern": true})).unwrap();
        let s = open(&dir).settings();
        assert_eq!((s.theme.as_str(), s.txt, s.pattern.as_str(), s.custom_pattern), ("dark", true, "$連番$", true));
        // 失敗した保存は何も変えない
        assert_eq!(store.settings().theme, "dark");
    }
}
