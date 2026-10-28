"""Check the re-render loss on one real validation batch: loss values, timing, finite gradients."""
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from restage.data.loader import Loader  # noqa: E402
from restage.model.losses import compute_losses, prepare_targets  # noqa: E402
from restage.model.net import Config, Restage  # noqa: E402
from restage.model.train import Features, to_mx  # noqa: E402


def main():
    mx.set_memory_limit(8 << 30)
    ld = Loader("data/val", batch_size=2, augment=False, seed=0, epoch=True, workers=1)
    batch = next(iter(ld))
    feats = Features()(batch["frames"], batch["key_mask"])
    bm, tgt = to_mx(batch), prepare_targets(batch)
    model = Restage(Config())
    mx.eval(model.parameters())

    def lf(m, render):
        out = m(feats, bm)
        total, L, met, _ = compute_losses(out, batch, bm, tgt, render=render, rng=np.random.default_rng(0))
        return total, (L, met)

    for render in (False, True):
        vg = nn.value_and_grad(model, lambda m: lf(m, render))
        for _ in range(3):
            t = time.time()
            (tot, (L, met)), g = vg(model)
            mx.eval(tot, g)
            dt = time.time() - t
        gmax = max(float(mx.abs(v).max()) for _, v in tree_flatten(g))
        print(f"render={render}: step {dt * 1e3:.0f} ms, total {float(tot):.3f}, grads finite {np.isfinite(gmax)} (max {gmax:.3g})")
        if render:
            print({k: round(float(v), 4) for k, v in L.items() if k.startswith("render")},
                  {k: round(float(v), 4) for k, v in met.items() if k.startswith("r")})
    ld.close()


if __name__ == "__main__":
    main()
