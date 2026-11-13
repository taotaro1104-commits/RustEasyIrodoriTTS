//! 合成の取消と進捗のフック。
//!
//! 呼び出し側（アプリのジョブ管理）が `cancel` を立てると、サンプラは**次のステップの先頭**で
//! `IrodoriError::Cancelled` を返して止まる。`progress` には完了したステップ数が入る。
//! フックを渡さない（`Control::default()`）場合は何もしない。

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::{IrodoriError, Result};

/// 合成のどの段階にいるか。止まったとき（固まったとき）に、どこで止まったかを知るために外へ見せる。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Stage {
    Idle = 0,
    Reference = 1,
    Conditions = 2,
    Sampling = 3,
    Decode = 4,
}

impl Stage {
    pub fn name(self) -> &'static str {
        match self {
            Stage::Idle => "idle",
            Stage::Reference => "reference",
            Stage::Conditions => "conditions",
            Stage::Sampling => "sampling",
            Stage::Decode => "decode",
        }
    }

    pub fn from_usize(n: usize) -> Stage {
        match n {
            1 => Stage::Reference,
            2 => Stage::Conditions,
            3 => Stage::Sampling,
            4 => Stage::Decode,
            _ => Stage::Idle,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Control {
    /// `true` になったら止める。
    pub cancel: Option<Arc<AtomicBool>>,
    /// 完了したステップ数（現在の候補の中での数）。
    pub progress: Option<Arc<AtomicUsize>>,
    /// いまの段階（`Stage` の値）。
    pub stage: Option<Arc<AtomicUsize>>,
}

impl Control {
    /// 取消・進捗の両方を新しく作る。
    pub fn new() -> Control {
        Control { cancel: Some(Arc::new(AtomicBool::new(false))), progress: Some(Arc::new(AtomicUsize::new(0))), stage: Some(Arc::new(AtomicUsize::new(0))) }
    }

    pub fn cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(|c| c.load(Ordering::SeqCst))
    }

    /// 取消済みなら `Cancelled`。
    pub fn check(&self) -> Result<()> {
        if self.cancelled() {
            Err(IrodoriError::Cancelled)
        } else {
            Ok(())
        }
    }

    pub fn set_stage(&self, stage: Stage) {
        if let Some(s) = &self.stage {
            s.store(stage as usize, Ordering::SeqCst);
        }
    }

    pub fn stage(&self) -> Stage {
        self.stage.as_ref().map_or(Stage::Idle, |s| Stage::from_usize(s.load(Ordering::SeqCst)))
    }

    pub fn set_progress(&self, done: usize) {
        if let Some(p) = &self.progress {
            p.store(done, Ordering::SeqCst);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IrodoriError;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    #[test]
    fn default_control_is_a_no_op() {
        let c = Control::default();
        assert!(!c.cancelled());
        assert!(c.check().is_ok());
        c.set_progress(3); // 受け取り手がいなくても落ちない
    }

    #[test]
    fn check_is_ok_without_the_flag_being_set() {
        let c = Control::new();
        assert!(!c.cancelled());
        assert!(c.check().is_ok());
    }

    #[test]
    fn check_returns_cancelled_once_the_flag_is_set_from_another_thread() {
        let c = Control::new();
        let flag = Arc::clone(c.cancel.as_ref().unwrap());
        std::thread::spawn(move || flag.store(true, Ordering::SeqCst)).join().unwrap();
        assert!(c.cancelled());
        assert!(matches!(c.check(), Err(IrodoriError::Cancelled)));
    }

    #[test]
    fn set_progress_is_visible_to_other_holders() {
        let c = Control::new();
        let seen = Arc::clone(c.progress.as_ref().unwrap());
        c.set_progress(2);
        assert_eq!(seen.load(Ordering::SeqCst), 2);
        c.set_progress(5);
        assert_eq!(seen.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn cancelled_error_is_named_cancelled() {
        assert_eq!(IrodoriError::Cancelled.to_string(), "cancelled");
    }

    #[test]
    fn clones_share_the_same_flags() {
        let a = Control::new();
        let b = a.clone();
        a.cancel.as_ref().unwrap().store(true, Ordering::SeqCst);
        a.set_progress(1);
        assert!(b.cancelled());
        assert_eq!(b.progress.as_ref().unwrap().load(Ordering::SeqCst), 1);
    }

    #[test]
    fn stage_is_visible_to_other_holders_and_defaults_to_idle() {
        let c = Control::new();
        let seen = c.clone();
        assert_eq!(seen.stage(), Stage::Idle);
        c.set_stage(Stage::Decode);
        assert_eq!(seen.stage(), Stage::Decode);
        assert_eq!(Stage::Decode.name(), "decode");
        assert_eq!([Stage::Idle, Stage::Reference, Stage::Conditions, Stage::Sampling, Stage::Decode].map(Stage::name), ["idle", "reference", "conditions", "sampling", "decode"]);
        let none = Control::default();
        none.set_stage(Stage::Sampling); // 受け取り手がいなくても落ちない
        assert_eq!(none.stage(), Stage::Idle);
    }
}
