//! 合成の取消と進捗のフック。
//!
//! 呼び出し側（アプリのジョブ管理）が `cancel` を立てると、サンプラは**次のステップの先頭**で
//! `IrodoriError::Cancelled` を返して止まる。`progress` には完了したステップ数が入る。
//! フックを渡さない（`Control::default()`）場合は何もしない。

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::{IrodoriError, Result};

#[derive(Clone, Debug, Default)]
pub struct Control {
    /// `true` になったら止める。
    pub cancel: Option<Arc<AtomicBool>>,
    /// 完了したステップ数（現在の候補の中での数）。
    pub progress: Option<Arc<AtomicUsize>>,
}

impl Control {
    /// 取消・進捗の両方を新しく作る。
    pub fn new() -> Control {
        Control { cancel: Some(Arc::new(AtomicBool::new(false))), progress: Some(Arc::new(AtomicUsize::new(0))) }
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
}
