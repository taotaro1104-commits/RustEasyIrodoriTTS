//! プロセスのピークメモリ（ワーキングセット）。

/// これまでのピーク常駐メモリ（MB）。取得できなければ `None`。
#[cfg(windows)]
pub fn peak_memory_mb() -> Option<f64> {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    // SAFETY: `counters` は有効な構造体で `cb` に正しいサイズを入れている。`GetCurrentProcess` は擬似ハンドルを返す。
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    (ok != 0).then(|| counters.PeakWorkingSetSize as f64 / (1024.0 * 1024.0))
}

/// Linux の `VmHWM`（ピーク RSS）。
#[cfg(not(windows))]
pub fn peak_memory_mb() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kb: f64 = status.lines().find(|l| l.starts_with("VmHWM:"))?.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_memory_is_reported_and_positive() {
        let mb = peak_memory_mb().expect("peak memory should be available on this platform");
        assert!(mb > 1.0, "{mb}");
    }
}
