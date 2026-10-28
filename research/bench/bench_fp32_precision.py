"""Is fp32 matmul on M5 true fp32 or a reduced-precision (TF32-like) path?

Compares MLX and torch-MPS fp32 matmul against a float64 NumPy reference and
reports max relative error. TF32 (10-bit mantissa) gives ~1e-3 relative error,
true fp32 gives ~1e-6. Also tries torch.set_float32_matmul_precision('high').
"""
import time

import numpy as np

from common import save

n = 2048
rng = np.random.default_rng(0)
A = rng.standard_normal((n, n)).astype(np.float32)
B = rng.standard_normal((n, n)).astype(np.float32)
ref = A.astype(np.float64) @ B.astype(np.float64)
scale = np.abs(ref).max()


def err(C):
    return float(np.abs(C.astype(np.float64) - ref).max() / scale)


rows = []
import mlx.core as mx

for dt_name, dt in [("fp32", mx.float32), ("fp16", mx.float16), ("bf16", mx.bfloat16)]:
    C = (mx.array(A).astype(dt) @ mx.array(B).astype(dt)).astype(mx.float32)
    rows.append({"lib": "mlx", "dtype": dt_name, "max_rel_err": err(np.array(C))})
    # CPU stream reference for MLX fp32
C = mx.matmul(mx.array(A), mx.array(B), stream=mx.cpu)
rows.append({"lib": "mlx-cpu", "dtype": "fp32", "max_rel_err": err(np.array(C))})

import torch

for prec in ["highest", "high", "medium"]:
    torch.set_float32_matmul_precision(prec)
    a = torch.from_numpy(A).to("mps")
    b = torch.from_numpy(B).to("mps")
    c = (a @ b).cpu().numpy()
    # time it too
    for _ in range(3):
        a @ b
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        a @ b
    torch.mps.synchronize()
    t = (time.perf_counter() - t0) / 10
    rows.append({"lib": f"torch-mps(precision={prec})", "dtype": "fp32", "max_rel_err": err(c),
                 "TFLOPS": round(2 * n ** 3 / t / 1e12, 2)})
for dt_name, dt in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
    c = (torch.from_numpy(A).to("mps", dt) @ torch.from_numpy(B).to("mps", dt)).float().cpu().numpy()
    rows.append({"lib": "torch-mps", "dtype": dt_name, "max_rel_err": err(c)})

# MLX exposes MLX_ENABLE_TF32; run a child process with it on and off.
import os
import subprocess
import sys

child = r'''
import mlx.core as mx, numpy as np, time, json
n=2048
A=np.random.default_rng(0).standard_normal((n,n)).astype(np.float32)
B=np.random.default_rng(1).standard_normal((n,n)).astype(np.float32)
ref=A.astype(np.float64)@B.astype(np.float64)
a,b=mx.array(A),mx.array(B)
err=float(np.abs(np.array(a@b)-ref).max()/np.abs(ref).max())
for _ in range(3): mx.eval(a@b)
t=time.perf_counter()
for _ in range(10): mx.eval(a@b)
t=(time.perf_counter()-t)/10
print(json.dumps({"max_rel_err":err,"TFLOPS":round(2*n**3/t/1e12,2)}))
'''
import json as _json

for v in ["0", "1"]:
    out = subprocess.check_output([sys.executable, "-c", child], env={**os.environ, "MLX_ENABLE_TF32": v})
    r = _json.loads(out.decode().strip().splitlines()[-1])
    rows.append({"lib": f"mlx(MLX_ENABLE_TF32={v})", "dtype": "fp32", **r})

for r in rows:
    print(r)
save("fp32_precision", {"n": n, "rows": rows})
