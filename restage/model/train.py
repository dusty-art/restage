"""Train the Restage compiler on packed synthetic clips.

    python -m restage.model.train --data data --steps 20000 --batch 2 --out runs/v1

DINOv2-S runs frozen on the fly (so augmentations are free); the compiler is trained with AdamW, warm-up and
cosine decay, gradient clipping, and periodic validation. Checkpoints are MLX safetensors plus a JSON config.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from restage.model.losses import WEIGHTS, compute_losses, prepare_targets
from restage.model.net import Config, Restage, count_params

ROOT = Path(__file__).resolve().parents[2]
SKIP_MX = {"frames", "meta"}


def to_mx(batch: dict) -> dict:
    out = {}
    for k, v in batch.items():
        if k in SKIP_MX or not isinstance(v, np.ndarray):
            continue
        out[k] = mx.array(v)
    return out


class Features:
    """Frozen DINOv2-S over the valid keyframes of a batch → (cls [B,K,384], patches [B,K,9,16,384])."""

    def __init__(self, dtype=mx.bfloat16):
        from restage.data.dinov2 import DinoV2S
        self.net = DinoV2S.load()
        self.dtype = dtype
        if hasattr(self.net, "set_dtype"):
            self.net.set_dtype(dtype)

    def __call__(self, frames: np.ndarray, key_mask: np.ndarray):
        B, K = key_mask.shape
        idx = np.nonzero(key_mask.reshape(-1))[0]
        flat = frames.reshape(B * K, *frames.shape[2:])[idx]
        cls_l, pat_l = [], []
        for i in range(0, len(flat), 64):
            c, p = self.net(mx.array(flat[i:i + 64]))
            cls_l.append(c.astype(mx.float32))
            pat_l.append(p.astype(mx.float32))
        c = mx.concatenate(cls_l)
        p = mx.concatenate(pat_l)
        cls = mx.zeros((B * K, c.shape[-1]))
        pat = mx.zeros((B * K,) + p.shape[1:])
        ii = mx.array(idx)
        cls[ii] = c
        pat[ii] = p
        return mx.stop_gradient(cls.reshape(B, K, -1)), mx.stop_gradient(pat.reshape((B, K) + p.shape[1:]))


def lr_at(step, base, warmup, total, floor=0.1):
    if step < warmup:
        return base * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


def save(model, cfg, path: Path, extra: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), dict(tree_flatten(model.parameters())))
    path.with_suffix(".json").write_text(json.dumps({"config": asdict(cfg), **(extra or {})}, indent=1))


def load(path: Path) -> tuple[Restage, Config]:
    meta = json.loads(Path(path).with_suffix(".json").read_text())
    c = meta["config"]
    c["grid"] = tuple(c["grid"])
    cfg = Config(**c)
    model = Restage(cfg)
    model.update(tree_unflatten(list(mx.load(str(path)).items())))
    return model, cfg


def make_step(model, feats_fn, seed=0):
    rng = np.random.default_rng(seed)

    def loss_fn(model, feats, batch, bm, tgt, render):
        out = model(feats, bm)
        total, parts, metrics, _ = compute_losses(out, batch, bm, tgt, render=render, rng=rng)
        return total, (parts, metrics)

    vg = nn.value_and_grad(model, loss_fn)

    def step(batch, render=False):
        feats = feats_fn(batch["frames"], batch["key_mask"])
        bm = to_mx(batch)
        tgt = prepare_targets(batch)
        (loss, (parts, metrics)), grads = vg(model, feats, batch, bm, tgt, render)
        return loss, parts, metrics, grads
    return step


def evaluate(model, feats_fn, loader, n_batches: int) -> dict:
    acc: dict[str, list[float]] = {}
    for _ in range(n_batches):
        batch = next(loader)
        feats = feats_fn(batch["frames"], batch["key_mask"])
        bm = to_mx(batch)
        out = model(feats, bm)
        total, parts, metrics, _ = compute_losses(out, batch, bm, prepare_targets(batch))
        mx.eval(total, parts, metrics)
        for k, v in {"loss": total, **parts, **metrics}.items():
            acc.setdefault(k, []).append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "v1"))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--resume", default="")
    ap.add_argument("--fake", action="store_true", help="random batches that follow the data contract (smoke test)")
    ap.add_argument("--mem-gb", type=float, default=10.0)
    ap.add_argument("--render-from", type=int, default=-1, help="step at which the re-render loss switches on (-1 = never)")
    args = ap.parse_args()
    mx.set_memory_limit(int(args.mem_gb * (1 << 30)))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.fake:
        from restage.model.fake import fake_loader, fake_features
        train_it, val_it, feats_fn = fake_loader(args.batch, 0), fake_loader(args.batch, 1), fake_features
    else:
        from restage.data.loader import make_loader
        train_it = make_loader(Path(args.data) / "train", batch_size=args.batch, augment=True, seed=0)
        val_it = make_loader(Path(args.data) / "val", batch_size=args.batch, augment=False, seed=1)
        feats_fn = Features()

    cfg = Config()
    model = Restage(cfg)
    start = 0
    if args.resume:
        model, cfg = load(Path(args.resume))
        start = json.loads(Path(args.resume).with_suffix(".json").read_text()).get("step", 0)
    print(f"Restage compiler: {count_params(model) / 1e6:.1f}M parameters; weights {WEIGHTS}", flush=True)
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=args.wd, betas=[0.9, 0.98])
    step_fn = make_step(model, feats_fn)
    log = open(out_dir / "log.jsonl", "a")
    best = float("inf")
    t0 = time.time()
    for it in range(start, args.steps):
        opt.learning_rate = lr_at(it, args.lr, args.warmup, args.steps)
        batch = next(train_it)
        render = args.render_from >= 0 and it >= args.render_from
        loss, parts, metrics, grads = step_fn(batch, render=render)
        if not all(np.isfinite(float(v)) for v in [loss]):
            print(f"step {it}: non-finite loss, batch skipped", flush=True)
            continue
        grads, gnorm = optim.clip_grad_norm(grads, args.clip)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss, parts, metrics)
        if it % 25 == 0:
            rec = {"step": it, "loss": float(loss), "gnorm": float(gnorm), "lr": float(opt.learning_rate),
                   "s_per_step": (time.time() - t0) / 25 if it else None,
                   **{k: round(float(v), 4) for k, v in parts.items()}, **{k: round(float(v), 4) for k, v in metrics.items()}}
            t0 = time.time()
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if (it + 1) % args.eval_every == 0 or it + 1 == args.steps:
            ev = evaluate(model, feats_fn, val_it, args.eval_batches)
            ev["step"] = it + 1
            print("VAL", json.dumps({k: round(v, 4) for k, v in ev.items()}), flush=True)
            log.write(json.dumps({"val": ev}) + "\n")
            log.flush()
            if ev["loss"] < best:
                best = ev["loss"]
                save(model, cfg, out_dir / "best.safetensors", {"step": it + 1, "val": ev})
        if (it + 1) % args.save_every == 0:
            save(model, cfg, out_dir / "last.safetensors", {"step": it + 1})
    save(model, cfg, out_dir / "last.safetensors", {"step": args.steps})


if __name__ == "__main__":
    main()
