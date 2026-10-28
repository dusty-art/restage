"""Benchmark 3 (and 5): ViT training-step time and memory, MLX vs torch MPS.

Default: ViT-S/16 (12 layers, dim 384, 6 heads), 224 px, batch 8, AdamW,
1000-way cross-entropy, synthetic data held on the GPU.
Each config runs in its own subprocess so peak physical footprint (which
includes GPU buffers on Apple silicon) is attributable to it.

Usage:
  python bench_vit.py                 # the standard ViT-S b8 comparison
  python bench_vit.py child '<json>'  # one config (used internally / bench_memory.py)
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fmt_table, save  # noqa: E402

HERE = os.path.abspath(__file__)


def run_mlx(cfg):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_map, tree_flatten
    from vit_models import mlx_forward, mlx_init

    mx.set_memory_limit(int(cfg.get("mem_limit_gb", 7.5) * 1e9))
    B, img = cfg["batch"], cfg["img"]
    depth, dim, heads = cfg["depth"], cfg["dim"], cfg["heads"]
    prec, ckpt = cfg["precision"], cfg.get("checkpoint", False)
    params = mlx_init(depth=depth, dim=dim, heads=heads, img=img)
    if prec == "bf16":
        params = tree_map(lambda p: p.astype(mx.bfloat16), params)
    opt = optim.AdamW(learning_rate=1e-4)
    opt.init(params)
    x = mx.random.normal((B, img, img, 3))
    y = mx.random.randint(0, 1000, (B,))
    if prec == "bf16":
        x = x.astype(mx.bfloat16)
    mx.eval(params, opt.state, x, y)

    def loss_fn(params, x, y):
        if prec == "amp_bf16":  # fp32 master weights, bf16 compute
            params = tree_map(lambda p: p.astype(mx.bfloat16), params)
            x = x.astype(mx.bfloat16)
        logits = mlx_forward(params, x, heads=heads, checkpoint=ckpt)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y).mean()

    vg = mx.value_and_grad(loss_fn)

    def step(params, x, y):
        loss, grads = vg(params, x, y)
        params = opt.apply_gradients(grads, params)
        return loss, params

    if cfg.get("compile"):
        step = mx.compile(step, inputs=[opt.state], outputs=[opt.state])

    mx.reset_peak_memory()
    t_first = time.perf_counter()
    loss, params = step(params, x, y)
    mx.eval(loss, params, opt.state)
    t_first = time.perf_counter() - t_first
    for _ in range(2):
        loss, params = step(params, x, y)
        mx.eval(loss, params, opt.state)
    ts = []
    for _ in range(cfg.get("iters", 10)):
        t0 = time.perf_counter()
        loss, params = step(params, x, y)
        mx.eval(loss, params, opt.state)
        ts.append(time.perf_counter() - t0)
    n_par = sum(v.size for _, v in tree_flatten(params))
    return ts, t_first, mx.get_peak_memory() / 1e9, n_par, float(loss)


def run_torch(cfg):
    import torch
    import torch.nn.functional as F
    from vit_models import torch_model

    torch.mps.set_per_process_memory_fraction(cfg.get("mem_fraction", 0.6))
    B, img = cfg["batch"], cfg["img"]
    prec = cfg["precision"]
    model = torch_model(depth=cfg["depth"], dim=cfg["dim"], heads=cfg["heads"], img=img,
                        checkpoint=cfg.get("checkpoint", False)).to("mps")
    if prec == "bf16":
        model = model.to(torch.bfloat16)
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        fused = True
    except Exception:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        fused = False
    fwd = torch.compile(model) if cfg.get("compile") else model
    x = torch.randn(B, 3, img, img, device="mps", dtype=torch.bfloat16 if prec == "bf16" else torch.float32)
    y = torch.randint(0, 1000, (B,), device="mps")
    amp = {"amp_fp16": torch.float16, "amp_bf16": torch.bfloat16}.get(prec)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("mps", dtype=amp or torch.float32, enabled=amp is not None):
            loss = F.cross_entropy(fwd(x).float(), y)
        loss.backward()
        opt.step()
        return loss

    t_first = time.perf_counter()
    step()
    torch.mps.synchronize()
    t_first = time.perf_counter() - t_first
    for _ in range(2):
        step()
    torch.mps.synchronize()
    ts = []
    for _ in range(cfg.get("iters", 10)):
        t0 = time.perf_counter()
        loss = step()
        torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    n_par = sum(p.numel() for p in model.parameters())
    return ts, t_first, torch.mps.driver_allocated_memory() / 1e9, n_par, float(loss) if fused or True else 0


def child(cfg):
    from common import peak_footprint_gb, footprint_gb
    from vit_models import train_flops_per_image
    base = footprint_gb()
    ts, t_first, lib_mem, n_par, loss = (run_mlx if cfg["lib"] == "mlx" else run_torch)(cfg)
    ts.sort()
    t = ts[len(ts) // 2]
    flops = train_flops_per_image(cfg["depth"], cfg["dim"], img=cfg["img"]) * cfg["batch"]
    return {**cfg, "ms_step": round(t * 1e3, 1), "img_per_s": round(cfg["batch"] / t, 1),
            "TFLOPS": round(flops / t / 1e12, 2), "first_step_s": round(t_first, 2),
            "peak_footprint_GB": round(peak_footprint_gb() - base, 2), "lib_mem_GB": round(lib_mem, 2),
            "params_M": round(n_par / 1e6, 1), "loss": round(loss, 3)}


def run_cfg(cfg, timeout=175):
    try:
        out = subprocess.run([sys.executable, HERE, "child", json.dumps(cfg)], capture_output=True, text=True,
                             timeout=timeout)
        lines = [l for l in out.stdout.splitlines() if l.startswith("{")]
        if lines:
            return json.loads(lines[-1])
        err = [l for l in out.stderr.strip().splitlines() if l.strip()]
        msg = next((l for l in reversed(err) if "Error" in l or "error" in l), err[-1] if err else "?")
        return {**cfg, "ms_step": "FAIL: " + msg[:160]}
    except subprocess.TimeoutExpired:
        return {**cfg, "ms_step": f"TIMEOUT>{timeout}s"}


VITS = {"depth": 12, "dim": 384, "heads": 6, "img": 224, "batch": 8}
COLS = ["lib", "precision", "compile", "checkpoint", "batch", "ms_step", "img_per_s", "TFLOPS", "first_step_s",
        "peak_footprint_GB", "lib_mem_GB", "params_M"]

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "child":
        print(json.dumps(child(json.loads(sys.argv[2]))))
        sys.exit(0)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfgs = []
    if which in ("all", "mlx"):
        for prec in ["fp32", "amp_bf16", "bf16"]:
            for comp in [False, True]:
                cfgs.append({"lib": "mlx", "precision": prec, "compile": comp, **VITS})
    if which in ("all", "torch"):
        for prec in ["fp32", "amp_fp16", "amp_bf16", "bf16"]:
            cfgs.append({"lib": "torch", "precision": prec, "compile": False, **VITS})
    if which in ("all", "torch_compile"):
        for prec in ["fp32", "amp_bf16"]:
            cfgs.append({"lib": "torch", "precision": prec, "compile": True, **VITS})
    rows = []
    for c in cfgs:
        r = run_cfg(c)
        print(r, flush=True)
        rows.append(r)
    print(fmt_table(rows, COLS))
    save(f"vit_{which}", {"rows": rows})
