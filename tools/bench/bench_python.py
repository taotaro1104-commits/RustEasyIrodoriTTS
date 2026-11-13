"""上流 Python 版（fp32, CPU）のピークメモリ / RTF を測る。Rust 版 `bench.sh` と同じ条件にしてある。

使い方（上流リポジトリの venv で、PYTHONPATH を上流のルートにして実行）:
    python tools/bench/bench_python.py --model-dir <dir> --codec-dir <dir> [--seconds 2]
"""
from __future__ import annotations

import argparse
import ctypes
import time
from ctypes import wintypes
from pathlib import Path


def peak_working_set_mb() -> float:
    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    c = Counters()
    c.cb = ctypes.sizeof(Counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise OSError("K32GetProcessMemoryInfo failed")
    return c.PeakWorkingSetSize / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--codec-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()

    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest

    t0 = time.time()
    rt = InferenceRuntime.from_key(RuntimeKey(
        checkpoint=str(args.model_dir / "model.safetensors"), model_device="cpu",
        codec_repo=str(args.codec_dir / "weights.pth"),
        model_precision="fp32", codec_device="cpu", codec_precision="fp32",
    ))
    load = time.time() - t0
    req = SamplingRequest(text="こんにちは、今日はいい天気ですね。", caption="落ち着いた女性の声",
                          no_ref=True, seconds=args.seconds, seed=1)
    t1 = time.time()
    result = rt.synthesize(req)
    synth = time.time() - t1
    audio = result.audio.shape[-1] / result.sample_rate
    print(f"load: {load:.2f}s, synthesis: {synth:.2f}s, RTF: {synth / audio:.2f}")
    print(f"peak memory: {peak_working_set_mb():.0f} MB")


if __name__ == "__main__":
    main()
