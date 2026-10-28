"""Core ML / Apple Neural Engine inference check for a ViT-S/16 encoder.

Converts the torch ViT-S (same as bench_vit) to an FP16 ML Program with
coremltools, then times prediction with compute units CPU_ONLY, CPU_AND_GPU,
CPU_AND_NE and ALL at batch 1 and 8. Where available, MLComputePlan reports
which device each op is preferred to run on. Compared against MLX bf16
forward at the same batch.
"""
import os
import sys
import tempfile
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import fmt_table, save  # noqa: E402
from vit_models import torch_model  # noqa: E402

import coremltools as ct  # noqa: E402



def static_vit(B, depth=12, dim=384, heads=6, img=224, patch=16, n_cls=1000):
    """Same ViT-S as vit_models.torch_model but with Python-int shapes so that
    torch.jit.trace does not emit aten::size/int casts (coremltools 9.0 fails
    on those with torch 2.15)."""
    import torch.nn as nn
    import torch.nn.functional as F
    g = img // patch
    N, hd = g * g + 1, dim // heads

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1, self.n2 = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim, eps=1e-6)
            self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
            self.fc1, self.fc2 = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)

        def forward(self, x):
            h = self.qkv(self.n1(x)).reshape(B, N, 3, heads, hd).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(h[0], h[1], h[2])
            x = x + self.proj(a.transpose(1, 2).reshape(B, N, dim))
            return x + self.fc2(F.gelu(self.fc1(self.n2(x))))

    class ViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(patch * patch * 3, dim)
            self.pos = nn.Parameter(torch.randn(1, N, dim) * 0.02)
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            self.blocks = nn.ModuleList([Block() for _ in range(depth)])
            self.norm, self.head = nn.LayerNorm(dim, eps=1e-6), nn.Linear(dim, n_cls)

        def forward(self, imgs):
            x = imgs.reshape(B, 3, g, patch, g, patch).permute(0, 2, 4, 3, 5, 1).reshape(B, g * g, patch * patch * 3)
            x = torch.cat([self.cls.expand(B, 1, dim), self.embed(x)], 1) + self.pos
            for b in self.blocks:
                x = b(x)
            return self.head(self.norm(x[:, 0]))

    return ViT()


rows = []
tmp = tempfile.mkdtemp(dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
for B in [1, 8]:
    model = static_vit(B).eval()
    ex = torch.randn(B, 3, 224, 224)
    with torch.no_grad():
        traced = torch.jit.trace(model, ex)
    t0 = time.perf_counter()
    ml = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=ex.shape)], convert_to="mlprogram",
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15)
    conv_s = time.perf_counter() - t0
    path = os.path.join(tmp, f"vits_b{B}.mlpackage")
    ml.save(path)
    ref = model(ex).detach().numpy()
    inp = {"x": ex.numpy()}
    plan_summary = None
    for cu in ["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"]:
        try:
            m = ct.models.MLModel(path, compute_units=getattr(ct.ComputeUnit, cu))
            out = list(m.predict(inp).values())[0]
            err = float(np.abs(out - ref).max() / np.abs(ref).max())
            for _ in range(3):
                m.predict(inp)
            n = 20
            t0 = time.perf_counter()
            for _ in range(n):
                m.predict(inp)
            dt = (time.perf_counter() - t0) / n
            rows.append({"batch": B, "compute_units": cu, "ms_per_call": round(dt * 1e3, 2),
                         "img_per_s": round(B / dt, 1), "rel_err_vs_fp32": round(err, 4)})
        except Exception as e:  # noqa: BLE001
            rows.append({"batch": B, "compute_units": cu, "ms_per_call": f"FAIL {type(e).__name__}: {str(e)[:100]}"})
        print(rows[-1], flush=True)
    if B == 1:
        try:
            from coremltools.models.compute_plan import MLComputePlan
            compiled = ct.models.MLModel(path, compute_units=ct.ComputeUnit.ALL).get_compiled_model_path()
            plan = MLComputePlan.load_from_path(path=compiled, compute_units=ct.ComputeUnit.ALL)
            prog = plan.model_structure.program
            counts = {}
            for fn in prog.functions.values():
                for op in fn.block.operations:
                    usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                    if usage is None:
                        continue
                    dev = type(usage.preferred_compute_device).__name__
                    counts[dev] = counts.get(dev, 0) + 1
            plan_summary = counts
            print("compute plan (ops per preferred device):", counts)
        except Exception as e:  # noqa: BLE001
            plan_summary = f"unavailable: {type(e).__name__}: {str(e)[:120]}"
            print(plan_summary)
        rows.append({"batch": 1, "compute_units": "plan(ALL)", "ms_per_call": str(plan_summary)})
    rows.append({"batch": B, "compute_units": "convert_seconds", "ms_per_call": round(conv_s, 1)})

# MLX reference at the same batch sizes
import mlx.core as mx  # noqa: E402
from mlx.utils import tree_map  # noqa: E402

from vit_models import mlx_forward, mlx_init  # noqa: E402

pb = tree_map(lambda p: p.astype(mx.bfloat16), mlx_init())
f = mx.compile(lambda p, x: mlx_forward(p, x))
for B in [1, 8]:
    xi = mx.random.normal((B, 224, 224, 3)).astype(mx.bfloat16)
    for _ in range(3):
        mx.eval(f(pb, xi))
    t0 = time.perf_counter()
    for _ in range(20):
        mx.eval(f(pb, xi))
    dt = (time.perf_counter() - t0) / 20
    rows.append({"batch": B, "compute_units": "MLX GPU bf16 (compiled)", "ms_per_call": round(dt * 1e3, 2),
                 "img_per_s": round(B / dt, 1)})
    print(rows[-1])
print(fmt_table(rows, ["batch", "compute_units", "ms_per_call", "img_per_s", "rel_err_vs_fp32"]))
save("coreml", {"coremltools": ct.__version__, "rows": rows})
import shutil  # noqa: E402

shutil.rmtree(tmp, ignore_errors=True)
