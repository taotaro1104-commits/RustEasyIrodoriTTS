//! 話者の保存・名前変更・削除（v2.0 の `easy_voices.py`）。
//!
//! 話者の WAV は版ごとに新しいファイルへ書き、古い版は消さない（生成待ちの要求が参照するため）。
//! 一覧（`voices/index.json`）の書き込みに失敗したら、メモリも WAV も元に戻す。

use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use crate::model::{uid, Voice};
use crate::postprocess;
use crate::storage::atomic_json;
use crate::{Result, Store, StoreError};

const NAME_MESSAGE: &str = "話者名を1～80文字で入力してください。";
const STALE_MESSAGE: &str = "話者が更新されています。一覧から読み直してください。";
const MIN_SECONDS: f64 = 1.0;
const MAX_SECONDS: f64 = 120.0;
const CAPTION_CHARS: usize = 2000;

fn invalid(message: &str) -> StoreError {
    StoreError::Invalid(message.into())
}

/// 前後の空白を除いた名前。1〜80 文字（コードポイント）でなければ拒否する。
fn clean_name(values: &Map<String, Value>) -> Result<String> {
    let name = values
        .get("name")
        .and_then(Value::as_str)
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let length = name.chars().count();
    if length == 0 || length > 80 {
        return Err(invalid(NAME_MESSAGE));
    }
    Ok(name)
}

/// `revision` が一致するか。Python と同じく `1.0 == 1` とみなす。
fn same_revision(sent: Option<&Value>, current: i64) -> bool {
    match sent {
        Some(Value::Number(n)) => {
            n.as_i64() == Some(current) || n.as_f64().is_some_and(|f| f == current as f64)
        }
        _ => false,
    }
}

/// フォルダ名に使える ID（パス結合に使うため、英数字・`-`・`_` だけ）。
fn safe_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

fn write_index(root: &Path, voices: &[Voice]) -> Result<()> {
    let value = serde_json::to_value(voices).map_err(|e| StoreError::Internal(e.to_string()))?;
    atomic_json(&root.join("voices/index.json"), &value)
}

/// 途中で失敗したときに、新しく置いた WAV（と、そのために作ったフォルダ）を片付ける。
struct Placed {
    file: PathBuf,
    folder: PathBuf,
    folder_is_new: bool,
    keep: bool,
}

impl Drop for Placed {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_file(&self.file);
            if self.folder_is_new {
                let _ = std::fs::remove_dir(&self.folder);
            }
        }
    }
}

impl Store {
    /// 話者を登録・更新する。`source` があればその WAV（1〜120 秒）を新しい版として置く。
    /// `allowed_models` に無いモデルは拒否する。
    pub fn save_voice(
        &self,
        values: &Map<String, Value>,
        source: Option<&Path>,
        allowed_models: &[&str],
    ) -> Result<Voice> {
        let name = clean_name(values)?;
        let model = values
            .get("model")
            .and_then(Value::as_str)
            .filter(|m| allowed_models.contains(m))
            .ok_or_else(|| invalid("使用するモデルを選んでください。"))?
            .to_string();
        let mut inner = self.lock();
        let requested = values
            .get("id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty());
        let existing = requested.and_then(|id| inner.voices.iter().find(|v| v.id == id).cloned());
        if requested.is_some() && existing.is_none() {
            return Err(invalid("編集対象の話者が見つかりません。"));
        }
        if let Some(old) = &existing {
            if !same_revision(values.get("revision"), old.revision) {
                return Err(invalid(STALE_MESSAGE));
            }
        }
        let vid = requested.map_or_else(uid, str::to_string);
        if !safe_id(&vid) {
            return Err(invalid("編集対象の話者が見つかりません。"));
        }
        let mut voice = existing.clone().unwrap_or_else(|| Voice {
            id: vid.clone(),
            revision: 0,
            audio: String::new(),
            name: String::new(),
            model: String::new(),
            caption: String::new(),
            sound_revision: None,
            creation: None,
            extra: Map::new(),
        });
        let mut placed = None;
        if let Some(source) = source {
            let seconds = postprocess::wav_seconds(source)?;
            if !(MIN_SECONDS..=MAX_SECONDS).contains(&seconds) {
                return Err(invalid("参照音声は1～120秒にしてください。"));
            }
            let relative = format!("voices/{vid}/{}.wav", uid());
            let target = inner.root.join(&relative);
            let folder = inner.root.join("voices").join(&vid);
            let folder_is_new = !folder.exists();
            std::fs::create_dir_all(&folder)?;
            let guard = Placed {
                file: target.clone(),
                folder,
                folder_is_new,
                keep: false,
            };
            postprocess::copy_as_pcm24(source, &target)?;
            voice.audio = relative;
            placed = Some(guard);
        } else if voice.audio.is_empty() {
            return Err(invalid(
                "参照音声を追加するか、生成した候補を選んでください。",
            ));
        }
        voice.name = name;
        voice.model = model;
        voice.caption = values
            .get("caption")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .chars()
            .take(CAPTION_CHARS)
            .collect();
        voice.revision += 1;
        voice.sound_revision = Some(voice.revision);
        if let Some(creation) = values.get("creation").filter(|c| !c.is_null()) {
            voice.creation = Some(creation.clone());
        }
        let mut voices: Vec<Voice> = inner
            .voices
            .iter()
            .filter(|v| v.id != vid)
            .cloned()
            .collect();
        voices.push(voice.clone());
        write_index(&inner.root, &voices)?;
        if let Some(guard) = placed.as_mut() {
            guard.keep = true;
        }
        inner.voices = voices;
        Ok(voice)
    }

    /// 名前だけを変える。音は変わらないので、`sound_revision` は（無ければ旧 `revision` で）固定する。
    pub fn rename_voice(&self, values: &Map<String, Value>) -> Result<Voice> {
        let name = clean_name(values)?;
        let mut inner = self.lock();
        let id = values.get("id").and_then(Value::as_str).unwrap_or_default();
        let mut voices = inner.voices.clone();
        let voice = voices
            .iter_mut()
            .find(|v| v.id == id)
            .ok_or_else(|| invalid("話者が見つかりません。選び直してください。"))?;
        if !same_revision(values.get("revision"), voice.revision) {
            return Err(invalid(STALE_MESSAGE));
        }
        if voice.name == name {
            return Ok(voice.clone());
        }
        voice.sound_revision.get_or_insert(voice.revision);
        voice.name = name;
        voice.revision += 1;
        let renamed = voice.clone();
        write_index(&inner.root, &voices)?;
        inner.voices = voices;
        Ok(renamed)
    }

    /// 一覧から外す。ファイルは消さない。無い ID でも成功する。
    pub fn delete_voice(&self, vid: &str) -> Result<()> {
        let mut inner = self.lock();
        let voices: Vec<Voice> = inner
            .voices
            .iter()
            .filter(|v| v.id != vid)
            .cloned()
            .collect();
        write_index(&inner.root, &voices)?;
        inner.voices = voices;
        Ok(())
    }
}

#[cfg(test)]
#[path = "voices_tests.rs"]
mod tests;
