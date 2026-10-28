"""Benchmark 5: how big a model / batch can we train in 16 GB?

Runs ViT-S/B/L training steps (AdamW, 224 px) at several batch sizes, with and
without gradient checkpointing, in MLX (amp_bf16 = fp32 master weights + bf16
compute, mx.compile) and torch MPS (autocast bf16), each in a fresh process,
and records peak physical footprint. Configs are chosen to stay < ~7.5 GB;
larger ones are extrapolated in the report from the per-image slope.

Usage: python bench_memory.py [mlx|torch]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_vit import COLS, run_cfg  # noqa: E402
from common import fmt_table, save  # noqa: E402

S = {"depth": 12, "dim": 384, "heads": 6}
Bm = {"depth": 12, "dim": 768, "heads": 12}
Lm = {"depth": 24, "dim": 1024, "heads": 16}

MLX = [
    ("ViT-S", S, 8, False), ("ViT-S", S, 32, False), ("ViT-S", S, 32, True), ("ViT-S", S, 128, True),
    ("ViT-B", Bm, 8, False), ("ViT-B", Bm, 32, True),
    ("ViT-L", Lm, 8, True),
]
TORCH = [
    ("ViT-S", S, 32, False), ("ViT-S", S, 32, True), ("ViT-S", S, 128, True),
    ("ViT-B", Bm, 8, False), ("ViT-B", Bm, 32, True),
]

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "mlx"
    rows = []
    for name, arch, b, ck in (MLX if which == "mlx" else TORCH):
        cfg = {"lib": which, "precision": "amp_bf16", "compile": which == "mlx", "checkpoint": ck,
               "img": 224, "batch": b, "iters": 5 if (b >= 32 or name != "ViT-S") else 10, **arch}
        r = run_cfg(cfg, timeout=175)
        r["model"] = name
        print(r, flush=True)
        rows.append(r)
    print(fmt_table(rows, ["model"] + COLS))
    save(f"memory_{which}", {"rows": rows})
