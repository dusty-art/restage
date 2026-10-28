"""Shared helpers for the Apple-silicon microbenchmarks.

All benchmarks write a JSON file into research/bench/results/ so the report
can be regenerated from raw numbers.
"""
import json
import os
import platform
import resource
import subprocess
import time

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def machine_info():
    info = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        info["chip"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        info["mem_bytes"] = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"]).decode().strip())
    except Exception:
        pass
    try:
        import mlx.core as mx
        info["mlx"] = mx.__version__
    except Exception:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
    except Exception:
        pass
    return info


def save(name, payload):
    payload = {"machine": machine_info(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"saved {path}")


def max_rss_gb():
    # ru_maxrss is bytes on macOS
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def _rusage_v4():
    """macOS proc_pid_rusage(RUSAGE_INFO_V4) -> tuple of uint64 fields (after uuid)."""
    import ctypes
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    buf = (ctypes.c_uint64 * 64)()
    rc = libproc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf))
    if rc != 0:
        raise OSError("proc_pid_rusage failed")
    return list(buf)[2:]  # skip 16-byte uuid


def footprint_gb():
    """Current physical footprint (includes Metal/IOAccelerator buffers)."""
    return _rusage_v4()[7] / 1e9


def peak_footprint_gb():
    """Lifetime max physical footprint of this process, in GB.

    On Apple silicon GPU buffers are attributed to the process footprint, so
    this is the number that matters for 'will it fit in 16 GB'.
    """
    return _rusage_v4()[28] / 1e9


# ---------------------------------------------------------------- MLX timing
def mlx_time(fn, warmup=3, iters=10, min_time=None):
    """Median wall time (s) of fn() including mx.eval of its outputs."""
    import mlx.core as mx
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


# -------------------------------------------------------------- torch timing
def torch_time(fn, warmup=3, iters=10):
    import torch
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def fmt_table(rows, cols):
    """Markdown table from list of dicts."""
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)
