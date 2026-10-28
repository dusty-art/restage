"""Benchmark 2: scaled dot-product attention, MLX vs torch MPS.

B=1, H=6 heads, head_dim=64 (ViT-S head layout), L in {1k, 4k, 16k}.
Forward and forward+backward, fp16/bf16 (+fp32 at 1k/4k).
Each configuration runs in its own subprocess so the peak physical footprint
(which on Apple silicon includes GPU buffers) can be attributed to it.

Effective TFLOPS uses the FlashAttention convention: fwd = 4*B*H*L^2*D,
fwd+bwd = 3.5x fwd. A naive (materialised softmax) MLX baseline is included.
"""
import json
import os
import subprocess
import sys

from common import fmt_table, save

B, H, D = 1, 6, 64
REPEAT = {1024: 20, 4096: 4, 16384: 1}
CHILD = os.path.abspath(__file__)


def run_one(lib, L, mode, dtype, impl):
    import time
    from common import peak_footprint_gb, footprint_gb
    base = footprint_gb()
    if lib == "mlx":
        import mlx.core as mx
        mx.set_memory_limit(int(7e9))
        dt = {"fp32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}[dtype]
        q, k, v = [mx.random.normal((B, H, L, D)).astype(dt) for _ in range(3)]
        cot = mx.random.normal((B, H, L, D)).astype(dt)
        scale = D ** -0.5

        if impl == "fused":
            def attn(q, k, v):
                return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        else:
            def attn(q, k, v):
                s = (q * scale) @ k.transpose(0, 1, 3, 2)
                return mx.softmax(s.astype(mx.float32), axis=-1).astype(q.dtype) @ v

        R = REPEAT[L]  # several calls per eval to amortise submit/sync overhead
        if mode == "fwd":
            fn = lambda: [attn(q, k, v) for _ in range(R)]
        else:
            def fn():
                return [mx.vjp(attn, [q, k, v], [cot])[1] for _ in range(R)]
        mx.eval(q, k, v, cot)
        mx.reset_peak_memory()
        for _ in range(2):
            mx.eval(fn())
        iters = 10 if L <= 4096 else 4
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            mx.eval(fn())
            ts.append(time.perf_counter() - t0)
        peak_lib = mx.get_peak_memory() / 1e9
    else:
        import torch
        import torch.nn.functional as F
        torch.mps.set_per_process_memory_fraction(0.5)  # hard cap ~6.3 GB
        dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        req = mode == "fwdbwd"
        q, k, v = [torch.randn(B, H, L, D, device="mps", dtype=dt, requires_grad=req) for _ in range(3)]
        cot = torch.randn(B, H, L, D, device="mps", dtype=dt)
        if impl == "fused":
            attn = lambda q, k, v: F.scaled_dot_product_attention(q, k, v)
        else:
            def attn(q, k, v):
                s = (q * D ** -0.5) @ k.transpose(-1, -2)
                return torch.softmax(s.float(), -1).to(q.dtype) @ v
        R = REPEAT[L]
        if mode == "fwd":
            def fn():
                with torch.no_grad():
                    for _ in range(R):
                        o = attn(q, k, v)
                return o
        else:
            def fn():
                for _ in range(R):
                    q.grad = k.grad = v.grad = None
                    out = attn(q, k, v)
                    out.backward(cot)
                return q.grad
        for _ in range(2):
            fn()
        torch.mps.synchronize()
        iters = 10 if L <= 4096 else 4
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            torch.mps.synchronize()
            ts.append(time.perf_counter() - t0)
        peak_lib = torch.mps.driver_allocated_memory() / 1e9
    ts.sort()
    t = ts[len(ts) // 2] / REPEAT[L]
    flops = 4 * B * H * L * L * D * (1 if mode == "fwd" else 3.5)
    return {"lib": lib, "impl": impl, "L": L, "mode": mode, "dtype": dtype,
            "ms": round(t * 1e3, 2), "eff_TFLOPS": round(flops / t / 1e12, 2),
            "peak_footprint_GB": round(peak_footprint_gb() - base, 2),
            "lib_mem_GB": round(peak_lib, 2)}


def main():
    configs = []
    for L in [1024, 4096, 16384]:
        for mode in ["fwd", "fwdbwd"]:
            for dtype in (["fp16", "bf16", "fp32"] if L <= 4096 else ["fp16", "bf16"]):
                for lib in ["mlx", "torch"]:
                    configs.append((lib, L, mode, dtype, "fused"))
            if L <= 4096:
                configs.append(("mlx", L, mode, "fp16", "naive"))
                configs.append(("torch", L, mode, "fp16", "naive"))
    rows = []
    for c in configs:
        try:
            out = subprocess.run([sys.executable, CHILD, "child", json.dumps(c)], capture_output=True,
                                 text=True, timeout=170)
            line = [l for l in out.stdout.splitlines() if l.startswith("{")]
            if line:
                r = json.loads(line[-1])
            else:
                err = (out.stderr.strip().splitlines() or ["?"])[-1][:120]
                r = {"lib": c[0], "impl": c[4], "L": c[1], "mode": c[2], "dtype": c[3], "ms": "FAIL: " + err}
        except subprocess.TimeoutExpired:
            r = {"lib": c[0], "impl": c[4], "L": c[1], "mode": c[2], "dtype": c[3], "ms": "TIMEOUT"}
        print(r, flush=True)
        rows.append(r)
    print(fmt_table(rows, ["lib", "impl", "L", "mode", "dtype", "ms", "eff_TFLOPS", "peak_footprint_GB", "lib_mem_GB"]))
    save("sdpa", {"B": B, "H": H, "D": D, "rows": rows})


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "child":
        print(json.dumps(run_one(*json.loads(sys.argv[2]))))
    else:
        main()
