"""Benchmark 1: dense matmul throughput, MLX vs torch MPS, fp32/fp16/bf16.

Square N x N @ N x N for N in {1024, 2048, 4096}, plus two ViT-S-shaped
matmuls (batch 8 x 197 tokens, dim 384 -> 1536 and the QKV projection).
A chain of k matmuls is evaluated per timed call so that dispatch overhead
is amortised; the second operand is scaled by 1/sqrt(K) so values stay O(1)
in fp16.
"""
import math
import sys

from common import fmt_table, mlx_time, save, torch_time

SIZES = [1024, 2048, 4096]
SHAPES = [(n, n, n) for n in SIZES] + [(1576, 384, 1536), (1576, 384, 1152), (1576, 1536, 384)]
TARGET_FLOP = 1.5e11  # per timed call


def chain_len(m, k, n):
    return max(1, int(TARGET_FLOP / (2 * m * k * n)))


def bench_mlx(dtype_name):
    import mlx.core as mx
    dt = {"fp32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}[dtype_name]
    rows = []
    for (m, k, n) in SHAPES:
        a = mx.random.normal((m, k)).astype(dt)
        bs = [(mx.random.normal((k, n)) / math.sqrt(k)).astype(dt) for _ in range(4)]
        mx.eval(a, *bs)
        L = chain_len(m, k, n)

        if k == n:
            def fn():
                x = a
                for i in range(L):
                    x = x @ bs[i % 4]
                return x
        else:  # non-square: L independent products evaluated together
            def fn():
                return [a @ bs[i % 4] for i in range(L)]
        t = mlx_time(fn, warmup=2, iters=7)
        tflops = 2 * m * k * n * L / t / 1e12
        rows.append({"lib": "mlx", "dtype": dtype_name, "shape": f"{m}x{k}x{n}", "chain": L,
                     "ms_per_matmul": round(t / L * 1e3, 3), "TFLOPS": round(tflops, 2)})
        print(rows[-1], flush=True)
    return rows


def bench_torch(dtype_name):
    import torch
    dev = "mps"
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    rows = []
    for (m, k, n) in SHAPES:
        a = torch.randn(m, k, device=dev).to(dt)
        bs = [(torch.randn(k, n, device=dev) / math.sqrt(k)).to(dt) for _ in range(4)]
        L = chain_len(m, k, n)
        if k == n:
            def fn():
                x = a
                for i in range(L):
                    x = x @ bs[i % 4]
                return x
        else:
            outs = [None]

            def fn():
                for i in range(L):
                    outs[0] = a @ bs[i % 4]
                return outs[0]
        t = torch_time(fn, warmup=2, iters=7)
        tflops = 2 * m * k * n * L / t / 1e12
        rows.append({"lib": "torch-mps", "dtype": dtype_name, "shape": f"{m}x{k}x{n}", "chain": L,
                     "ms_per_matmul": round(t / L * 1e3, 3), "TFLOPS": round(tflops, 2)})
        print(rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    rows = []
    if which in ("both", "mlx"):
        for d in ["fp32", "fp16", "bf16"]:
            rows += bench_mlx(d)
    if which in ("both", "torch"):
        for d in ["fp32", "fp16", "bf16"]:
            rows += bench_torch(d)
    print(fmt_table(rows, ["lib", "dtype", "shape", "chain", "ms_per_matmul", "TFLOPS"]))
    save(f"matmul_{which}", {"rows": rows})
