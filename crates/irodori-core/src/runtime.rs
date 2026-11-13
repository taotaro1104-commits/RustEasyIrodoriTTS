//! モデルの遅延ロードと、手動／アイドル自動アンロード。
//!
//! 生成は 1 本ずつ（`Mutex` で直列化）。アイドルが続いたらエンジン（＝常駐した重み）を解放し、
//! 次の呼び出しで再ロードする（上流は手動アンロードのみ）。

use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::pipeline::{Pipeline, SynthOutput, SynthRequest};
use crate::Result;

/// 合成エンジン。`Pipeline` が実装し、テストでは擬似エンジンを使う。
pub trait Engine {
    type Request;
    type Output;
    fn run(&self, req: &Self::Request) -> Result<Self::Output>;
}

impl Engine for Pipeline {
    type Request = SynthRequest;
    type Output = SynthOutput;
    fn run(&self, req: &SynthRequest) -> Result<SynthOutput> {
        self.synthesize(req)
    }
}

struct State<E> {
    engine: Option<E>,
    last_used: Instant,
}

pub struct Runtime<E: Engine> {
    loader: Box<dyn Fn() -> Result<E> + Send + Sync>,
    state: Mutex<State<E>>,
}

impl<E: Engine> Runtime<E> {
    /// `loader` は初回利用時（とアンロード後の再利用時）にエンジンを作る。
    pub fn new(loader: impl Fn() -> Result<E> + Send + Sync + 'static) -> Self {
        Self { loader: Box::new(loader), state: Mutex::new(State { engine: None, last_used: Instant::now() }) }
    }

    fn lock(&self) -> MutexGuard<'_, State<E>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn is_loaded(&self) -> bool {
        self.lock().engine.is_some()
    }

    /// 必要ならロードして実行する。ロード失敗はそのままエラーで返し、未ロードのままにする。
    pub fn run(&self, req: &E::Request) -> Result<E::Output> {
        let mut st = self.lock();
        if st.engine.is_none() {
            st.engine = Some((self.loader)()?);
        }
        let out = st.engine.as_ref().map(|e| e.run(req));
        st.last_used = Instant::now();
        out.expect("engine was just loaded")
    }

    pub fn unload(&self) {
        self.lock().engine = None;
    }

    /// 最後の利用から `max_idle` 以上たっていればアンロードする。アンロードしたら `true`。
    pub fn unload_if_idle(&self, max_idle: Duration) -> bool {
        self.unload_if_idle_at(Instant::now(), max_idle)
    }

    /// `unload_if_idle` の時刻注入版（テスト用）。
    pub fn unload_if_idle_at(&self, now: Instant, max_idle: Duration) -> bool {
        let mut st = self.lock();
        if st.engine.is_some() && now.saturating_duration_since(st.last_used) >= max_idle {
            st.engine = None;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IrodoriError;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    struct Fake {
        calls: Arc<AtomicUsize>,
    }

    impl Engine for Fake {
        type Request = &'static str;
        type Output = usize;
        fn run(&self, _req: &Self::Request) -> Result<usize> {
            Ok(self.calls.fetch_add(1, Ordering::SeqCst) + 1)
        }
    }

    fn runtime() -> (Runtime<Fake>, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let (loads, calls) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let (l, c) = (loads.clone(), calls.clone());
        let rt = Runtime::new(move || {
            l.fetch_add(1, Ordering::SeqCst);
            Ok(Fake { calls: c.clone() })
        });
        (rt, loads, calls)
    }

    #[test]
    fn starts_unloaded_and_loads_lazily_once() {
        let (rt, loads, _) = runtime();
        assert!(!rt.is_loaded());
        assert_eq!(rt.run(&"a").unwrap(), 1);
        assert_eq!(rt.run(&"b").unwrap(), 2);
        assert!(rt.is_loaded());
        assert_eq!(loads.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn unload_drops_the_engine_and_the_next_run_reloads() {
        let (rt, loads, _) = runtime();
        rt.run(&"a").unwrap();
        rt.unload();
        assert!(!rt.is_loaded());
        rt.run(&"b").unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn unload_is_idempotent() {
        let (rt, _, _) = runtime();
        rt.unload();
        rt.unload();
        assert!(!rt.is_loaded());
    }

    #[test]
    fn idle_unload_respects_the_threshold() {
        let (rt, _, _) = runtime();
        rt.run(&"a").unwrap();
        let t0 = Instant::now();
        assert!(!rt.unload_if_idle_at(t0, Duration::from_secs(300)), "recently used engine stays loaded");
        assert!(rt.is_loaded());
        assert!(rt.unload_if_idle_at(t0 + Duration::from_secs(301), Duration::from_secs(300)));
        assert!(!rt.is_loaded());
        assert!(!rt.unload_if_idle_at(t0 + Duration::from_secs(900), Duration::from_secs(300)), "nothing left to unload");
    }

    #[test]
    fn a_failing_loader_leaves_the_runtime_unloaded_and_reports_the_error() {
        let rt: Runtime<Fake> = Runtime::new(|| Err(IrodoriError::InvalidInput("boom".into())));
        let err = rt.run(&"a").unwrap_err().to_string();
        assert!(err.contains("boom"), "{err}");
        assert!(!rt.is_loaded());
    }
}
