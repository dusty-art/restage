"""Benchmark 4: differentiable splatting / soft rasterisation to 128x128.

N in {1k, 10k, 100k} primitives; methods gauss_dense, gauss_local, tri_soft
(see raster_ops.py). Libraries: MLX eager, MLX with mx.compile on the chunk
kernel, torch MPS eager. Forward (no grad) and forward+backward (MSE loss vs a
target image, grads w.r.t. all primitive parameters).
Also checks MLX vs torch produce the same image and gradients.

Usage: python bench_raster.py [method ...]
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fmt_table, save  # noqa: E402

HERE = os.path.abspath(__file__)
NS = [1000, 10000, 100000]


def child(lib, method, N):
    import numpy as np
    from common import footprint_gb, peak_footprint_gb
    import raster_ops as R
    base = footprint_gb()
    pn = R.make_params_np(N, method)
    target = np.random.default_rng(1).uniform(0, 1, (R.H, R.W, 3)).astype(np.float32)
    iters = 5 if N <= 10000 else 3
    out = {"lib": lib, "method": method, "N": N}
    if lib.startswith("mlx"):
        import mlx.core as mx
        mx.set_memory_limit(int(6e9))
        comp = lib == "mlx-compile"
        p = {k: mx.array(v) for k, v in pn.items()}
        tgt = mx.array(target)

        def loss_fn(p):
            img = R.mlx_render(p, method)
            return ((img - tgt) ** 2).mean()
        vg = mx.value_and_grad(loss_fn)
        render = lambda p: R.mlx_render(p, method)
        if comp:  # compile the whole forward and the whole value_and_grad graph
            vg, render = mx.compile(vg), mx.compile(render)
        if lib == "mlx-stream":  # eval per chunk: bounded memory at any N
            vg = lambda p: R.mlx_stream_value_and_grad(p, method, tgt)
            render = lambda p: R.mlx_stream_render(p, method)
        fwd = lambda: render(p)
        bwd = lambda: vg(p)
        for name, fn in [("fwd", fwd), ("fwdbwd", bwd)]:
            mx.eval(fn())
            mx.reset_peak_memory()
            ts = []
            for _ in range(iters):
                t0 = time.perf_counter()
                mx.eval(fn())
                ts.append(time.perf_counter() - t0)
            ts.sort()
            out[f"{name}_ms"] = round(ts[len(ts) // 2] * 1e3, 2)
            out[f"{name}_mem_GB"] = round(mx.get_peak_memory() / 1e9, 2)
        img = np.array(fwd())
        loss, g = vg(p)
        out["checksum_img"] = float(img.sum())
        out["checksum_grad"] = {k: float(mx.abs(v).sum()) for k, v in g.items()}
    else:
        import torch
        torch.mps.set_per_process_memory_fraction(0.5)
        p = {k: torch.tensor(v, device="mps", requires_grad=True) for k, v in pn.items()}
        tgt = torch.tensor(target, device="mps")

        def fwd():
            with torch.no_grad():
                return R.torch_render(p, method)

        def bwd():
            for v in p.values():
                v.grad = None
            loss = ((R.torch_render(p, method) - tgt) ** 2).mean()
            loss.backward()
            return loss
        for name, fn in [("fwd", fwd), ("fwdbwd", bwd)]:
            fn()
            torch.mps.synchronize()
            ts = []
            for _ in range(iters):
                t0 = time.perf_counter()
                fn()
                torch.mps.synchronize()
                ts.append(time.perf_counter() - t0)
            ts.sort()
            out[f"{name}_ms"] = round(ts[len(ts) // 2] * 1e3, 2)
        out["checksum_img"] = float(fwd().sum().cpu())
        bwd()
        out["checksum_grad"] = {k: float(v.grad.abs().sum().cpu()) for k, v in p.items()}
    out["peak_footprint_GB"] = round(peak_footprint_gb() - base, 2)
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        print(json.dumps(child(sys.argv[2], sys.argv[3], int(sys.argv[4]))))
        sys.exit(0)
    methods = sys.argv[1:] or ["gauss_local", "gauss_dense", "tri_soft"]
    rows = []
    for method in methods:
        for N in NS:
            for lib in ["mlx", "mlx-compile", "mlx-stream", "torch"]:
                if method == "gauss_local" and lib == "mlx-stream":
                    continue
                if method != "gauss_local" and lib in ("mlx", "mlx-compile") and N > 1000:
                    # one lazy graph over all chunks: ~40-80 B per pixel-primitive pair
                    # in backward -> ~6 GB at N=10k, >60 GB at N=100k; skipped (see report)
                    rows.append({"lib": lib, "method": method, "N": N, "fwd_ms": "skipped: one lazy graph, ~6 GB at 10k (measured), >60 GB at 100k (est.)"})
                    continue
                try:
                    r = subprocess.run([sys.executable, HERE, "child", lib, method, str(N)], capture_output=True,
                                       text=True, timeout=170)
                    ls = [l for l in r.stdout.splitlines() if l.startswith("{")]
                    row = json.loads(ls[-1]) if ls else {"lib": lib, "method": method, "N": N,
                                                         "fwd_ms": "FAIL " + (r.stderr.strip().splitlines() or ["?"])[-1][:150]}
                except subprocess.TimeoutExpired:
                    row = {"lib": lib, "method": method, "N": N, "fwd_ms": "TIMEOUT"}
                print(row, flush=True)
                rows.append(row)
    print(fmt_table(rows, ["method", "N", "lib", "fwd_ms", "fwdbwd_ms", "fwd_mem_GB", "fwdbwd_mem_GB",
                           "peak_footprint_GB", "checksum_img"]))
    save("raster_" + "_".join(methods), {"rows": rows})
