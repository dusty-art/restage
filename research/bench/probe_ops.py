"""Probe op coverage relevant to the project on torch MPS (no CPU fallback) and
MLX GPU. Prints a table: works on GPU / falls back / missing, with the
exception text. Forward and (where meaningful) backward are both exercised.
"""
import os
import sys
import traceback

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fmt_table, save  # noqa: E402


def t_probe():
    import torch
    import torch.nn.functional as F
    d = "mps"
    A = lambda *s: torch.randn(*s, device=d, requires_grad=True)
    tests = {
        "grid_sample 2D bilinear (+bwd)": lambda: F.grid_sample(A(1, 3, 32, 32), torch.rand(1, 8, 8, 2, device=d) * 2 - 1, align_corners=False).sum().backward(),
        "grid_sample 3D (volumetric, +bwd)": lambda: F.grid_sample(A(1, 3, 8, 8, 8), torch.rand(1, 4, 4, 4, 3, device=d) * 2 - 1, align_corners=False).sum().backward(),
        "interpolate bicubic antialias": lambda: F.interpolate(A(1, 3, 64, 64), size=(32, 32), mode="bicubic", antialias=True).sum().backward(),
        "conv3d (+bwd)": lambda: F.conv3d(A(1, 3, 8, 16, 16), A(8, 3, 3, 3, 3)).sum().backward(),
        "linalg.svd (+bwd)": lambda: torch.linalg.svd(A(16, 3, 3))[0].sum().backward(),
        "linalg.eigh": lambda: torch.linalg.eigh(torch.eye(3, device=d).repeat(4, 1, 1)),
        "linalg.inv (+bwd)": lambda: torch.linalg.inv(A(16, 4, 4) + 4 * torch.eye(4, device=d)).sum().backward(),
        "linalg.solve": lambda: torch.linalg.solve(torch.eye(4, device=d) * 3 + torch.randn(4, 4, device=d) * .1, torch.randn(4, 2, device=d)),
        "linalg.det (+bwd)": lambda: torch.linalg.det(A(16, 3, 3)).sum().backward(),
        "linalg.qr": lambda: torch.linalg.qr(torch.randn(8, 4, device=d)),
        "linalg.cholesky": lambda: torch.linalg.cholesky(torch.eye(4, device=d) * 2),
        "linalg.lstsq": lambda: torch.linalg.lstsq(torch.randn(8, 4, device=d), torch.randn(8, 2, device=d)),
        "scatter_reduce amax (+bwd)": lambda: torch.zeros(10, device=d).scatter_reduce(0, torch.randint(0, 10, (100,), device=d), A(100), "amax").sum().backward(),
        "index_put accumulate (+bwd)": lambda: torch.zeros(10, device=d).index_put((torch.randint(0, 10, (100,), device=d),), A(100), accumulate=True).sum().backward(),
        "sort / argsort 1M": lambda: torch.sort(torch.randn(1 << 20, device=d)),
        "topk k=64": lambda: torch.topk(torch.randn(4, 10000, device=d), 64),
        "unique": lambda: torch.unique(torch.randint(0, 50, (1000,), device=d)),
        "nonzero": lambda: torch.nonzero(torch.randn(1000, device=d) > 0),
        "searchsorted": lambda: torch.searchsorted(torch.sort(torch.randn(100, device=d))[0], torch.randn(10, device=d)),
        "cumsum (+bwd)": lambda: torch.cumsum(A(1000), 0).sum().backward(),
        "cdist (+bwd)": lambda: torch.cdist(A(100, 3), A(200, 3)).sum().backward(),
        "fft.rfft2": lambda: torch.fft.rfft2(torch.randn(64, 64, device=d)),
        "float64 tensor": lambda: torch.zeros(3, device=d, dtype=torch.float64),
        "complex64 matmul": lambda: torch.randn(4, 4, device=d, dtype=torch.complex64) @ torch.randn(4, 4, device=d, dtype=torch.complex64),
        "erfinv": lambda: torch.special.erfinv(torch.rand(10, device=d) * 1.8 - 0.9),
        "F.rms_norm (+bwd)": lambda: F.rms_norm(A(4, 64), (64,)).sum().backward(),
        "SDPA with bool mask (+bwd)": lambda: F.scaled_dot_product_attention(A(1, 2, 64, 32), A(1, 2, 64, 32), A(1, 2, 64, 32), attn_mask=torch.rand(64, 64, device=d) > 0.5).sum().backward(),
        "nn.functional.unfold": lambda: F.unfold(A(1, 3, 32, 32), 4, stride=4).sum().backward(),
        "max_pool3d (+bwd)": lambda: F.max_pool3d(A(1, 2, 8, 8, 8), 2).sum().backward(),
        "torch.histc": lambda: torch.histc(torch.randn(1000, device=d), bins=10),
        "bincount": lambda: torch.bincount(torch.randint(0, 10, (100,), device=d)),
    }
    rows = []
    for name, fn in tests.items():
        try:
            fn()
            torch.mps.synchronize()
            rows.append({"op": name, "torch_mps": "ok"})
        except Exception as e:  # noqa: BLE001
            msg = str(e).strip().splitlines()[0][:90]
            rows.append({"op": name, "torch_mps": "FAIL: " + msg})
    return rows


def m_probe():
    import mlx.core as mx
    r = mx.random.normal
    g = lambda f, *a: mx.eval(mx.grad(lambda *x: f(*x).sum())(*a))
    tests = {
        "grid_sample 2D bilinear (+bwd)": None,
        "grid_sample 3D (volumetric, +bwd)": None,
        "interpolate bicubic antialias": None,
        "conv3d (+bwd)": lambda: g(lambda x: mx.conv3d(x, r((8, 3, 3, 3, 3))), r((1, 8, 16, 16, 3))),
        "linalg.svd (+bwd)": lambda: mx.eval(mx.linalg.svd(r((16, 3, 3)))),
        "linalg.eigh": lambda: mx.eval(mx.linalg.eigh(mx.broadcast_to(mx.eye(3), (4, 3, 3)))),
        "linalg.inv (+bwd)": lambda: g(lambda x: mx.linalg.inv(x + 4 * mx.eye(4)), r((16, 4, 4))),
        "linalg.solve": lambda: mx.eval(mx.linalg.solve(mx.eye(4) * 3, r((4, 2)))),
        "linalg.det (+bwd)": None,
        "linalg.qr": lambda: mx.eval(mx.linalg.qr(r((8, 4)))),
        "linalg.cholesky": lambda: mx.eval(mx.linalg.cholesky(mx.eye(4) * 2)),
        "linalg.lstsq": None,
        "scatter_reduce amax (+bwd)": lambda: g(lambda v: mx.zeros((10,)).at[mx.random.randint(0, 10, (100,))].maximum(v), r((100,))),
        "index_put accumulate (+bwd)": lambda: g(lambda v: mx.zeros((10,)).at[mx.random.randint(0, 10, (100,))].add(v), r((100,))),
        "sort / argsort 1M": lambda: mx.eval(mx.sort(r((1 << 20,)))),
        "topk k=64": lambda: mx.eval(mx.topk(r((4, 10000)), 64)),
        "unique": lambda: mx.eval(mx.unique(mx.random.randint(0, 50, (1000,)), size=50)),  # static size required
        "nonzero": None,
        "searchsorted": lambda: mx.eval(mx.searchsorted(mx.sort(r((100,))), r((10,)))),
        "cumsum (+bwd)": lambda: g(lambda x: mx.cumsum(x), r((1000,))),
        "cdist (+bwd)": None,
        "fft.rfft2": lambda: mx.eval(mx.fft.rfft2(r((64, 64)))),
        "float64 tensor": lambda: mx.eval(mx.zeros((3,), dtype=mx.float64) + 1),
        "complex64 matmul": lambda: mx.eval(r((4, 4)).astype(mx.complex64) @ r((4, 4)).astype(mx.complex64)),
        "erfinv": lambda: mx.eval(mx.erfinv(mx.random.uniform(-0.9, 0.9, (10,)))),
        "F.rms_norm (+bwd)": lambda: g(lambda x: mx.fast.rms_norm(x, mx.ones((64,)), 1e-5), r((4, 64))),
        "SDPA with bool mask (+bwd)": lambda: g(lambda q: mx.fast.scaled_dot_product_attention(q, r((1, 2, 64, 32)), r((1, 2, 64, 32)), scale=0.17, mask=mx.random.uniform(shape=(64, 64)) > 0.5), r((1, 2, 64, 32))),
        "nn.functional.unfold": None,
        "max_pool3d (+bwd)": lambda: __import__("mlx.nn", fromlist=["x"]).MaxPool3d(2)(r((1, 8, 8, 8, 2))),
        "torch.histc": None,
        "bincount": None,
    }
    rows = {}
    # MLX linalg runs only on the CPU stream; check whether it is differentiable there
    x = r((16, 3, 3))
    lin = {
        "linalg.svd (+bwd)": lambda a: mx.linalg.svd(a, stream=mx.cpu)[1].sum(),
        "linalg.inv (+bwd)": lambda a: mx.linalg.inv(a + 4 * mx.eye(3), stream=mx.cpu).sum(),
        "linalg.eigh": lambda a: mx.linalg.eigh(a + a.transpose(0, 2, 1), stream=mx.cpu)[0].sum(),
        "linalg.qr": lambda a: mx.linalg.qr(a, stream=mx.cpu)[0].sum(),
        "linalg.cholesky": lambda a: mx.linalg.cholesky(a @ a.transpose(0, 2, 1) + mx.eye(3), stream=mx.cpu).sum(),
    }
    cpu_status = {}
    for name, f in lin.items():
        try:
            mx.eval(f(x))
            try:
                mx.eval(mx.grad(f)(x))
                cpu_status[name] = "; CPU stream fwd+bwd ok"
            except Exception as e:  # noqa: BLE001
                cpu_status[name] = "; CPU stream fwd ok, bwd: " + str(e).strip().splitlines()[0][:60]
        except Exception as e:  # noqa: BLE001
            cpu_status[name] = "; CPU stream FAIL " + str(e)[:60]
    for name, fn in tests.items():
        if fn is None:
            rows[name] = "no built-in op"
            continue
        try:
            out = fn()
            if out is not None:
                mx.eval(out)
            rows[name] = "ok"
        except Exception as e:  # noqa: BLE001
            msg = str(e).strip().splitlines()[0][:90]
            rows[name] = "FAIL: " + msg
        if name in cpu_status:
            rows[name] = ("GPU: not supported" if rows[name].startswith("FAIL") else rows[name]) + cpu_status[name]
    return rows


if __name__ == "__main__":
    rows = t_probe()
    m = m_probe()
    for r in rows:
        r["mlx_gpu"] = m.get(r["op"], "?")
    print(fmt_table(rows, ["op", "torch_mps", "mlx_gpu"]))
    save("op_probe", {"rows": rows})
