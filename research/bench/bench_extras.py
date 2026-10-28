"""Extras that ground specific recommendations in the report.

(a) MLX quantized matmul (4/8-bit affine, group 64) vs bf16 at ViT-S and
    LLM-like shapes, M in {1, 16, 1576}.
(b) ViT-S training-step cost vs token count (progressive resolution / token
    merging proxy): 112, 160, 224, 320 px at batch 32, MLX amp_bf16 compiled.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fmt_table, save  # noqa: E402


def quant_bench():
    import mlx.core as mx
    rows = []
    for (K, Nout) in [(384, 1536), (1536, 384), (4096, 4096)]:
        w = (mx.random.normal((Nout, K)) / K ** 0.5).astype(mx.bfloat16)
        qs = {b: mx.quantize(w, group_size=64, bits=b) for b in (4, 8)}
        for M in [1, 16, 1576]:
            x = mx.random.normal((M, K)).astype(mx.bfloat16)
            reps = 200 if M <= 16 else 40
            fns = {"bf16": lambda: [x @ w.T for _ in range(reps)]}
            for b, (wq, sc, bi) in qs.items():
                fns[f"q{b}"] = (lambda wq=wq, sc=sc, bi=bi, b=b: [
                    mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=64, bits=b) for _ in range(reps)])
            res = {"K": K, "N": Nout, "M": M}
            ref = x @ w.T
            for name, fn in fns.items():
                for _ in range(2):
                    mx.eval(fn())
                t0 = time.perf_counter()
                for _ in range(3):
                    mx.eval(fn())
                t = (time.perf_counter() - t0) / (3 * reps)
                res[f"{name}_us"] = round(t * 1e6, 1)
                if name != "bf16":
                    out = fn()[0]
                    res[f"{name}_relerr"] = round(float(mx.abs(out - ref).max() / mx.abs(ref).max()), 4)
            rows.append(res)
            print(res, flush=True)
    return rows


def res_bench():
    from bench_vit import run_cfg
    rows = []
    for img in [112, 160, 224, 320]:
        cfg = {"lib": "mlx", "precision": "amp_bf16", "compile": True, "depth": 12, "dim": 384, "heads": 6,
               "img": img, "batch": 32, "iters": 6}
        r = run_cfg(cfg)
        r["tokens"] = (img // 16) ** 2 + 1
        print(r, flush=True)
        rows.append(r)
    return rows


if __name__ == "__main__":
    q = quant_bench()
    print(fmt_table(q, list(q[0].keys())))
    r = res_bench()
    print(fmt_table(r, ["img", "tokens", "batch", "ms_step", "img_per_s", "TFLOPS", "lib_mem_GB"]))
    save("extras", {"quantized_matmul": q, "resolution_scaling": r})
