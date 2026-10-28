"""Sustained-throughput check: does the M5 throttle over a few minutes?

Trains ViT-S/16 (MLX, amp_bf16, mx.compile, batch 32, 224 px) for ~150 s and
reports step time per 15 s window. Also prints ViT-S inference throughput
(forward only, bf16, batch 64) for the encoder-at-inference budget.
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_map  # noqa: E402

from common import save  # noqa: E402
from vit_models import mlx_forward, mlx_init, train_flops_per_image  # noqa: E402

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 150
B = 32
params = mlx_init()
opt = optim.AdamW(learning_rate=1e-4)
opt.init(params)
x = mx.random.normal((B, 224, 224, 3))
y = mx.random.randint(0, 1000, (B,))


def loss_fn(params, x, y):
    params = tree_map(lambda p: p.astype(mx.bfloat16), params)
    return nn.losses.cross_entropy(mlx_forward(params, x.astype(mx.bfloat16)).astype(mx.float32), y).mean()


vg = mx.value_and_grad(loss_fn)


def step(params, x, y):
    loss, g = vg(params, x, y)
    return loss, opt.apply_gradients(g, params)


step = mx.compile(step, inputs=[opt.state], outputs=[opt.state])
for _ in range(3):
    loss, params = step(params, x, y)
    mx.eval(loss, params, opt.state)

windows, cur, t_start, t_win = [], [], time.perf_counter(), time.perf_counter()
while time.perf_counter() - t_start < DURATION:
    t0 = time.perf_counter()
    loss, params = step(params, x, y)
    mx.eval(loss, params, opt.state)
    cur.append(time.perf_counter() - t0)
    if time.perf_counter() - t_win > 15:
        windows.append({"t_s": round(time.perf_counter() - t_start), "ms_step": round(1e3 * sum(cur) / len(cur), 1),
                        "img_per_s": round(B * len(cur) / sum(cur), 1)})
        print(windows[-1], flush=True)
        cur, t_win = [], time.perf_counter()

try:
    therm = subprocess.check_output(["pmset", "-g", "therm"]).decode()
except Exception as e:
    therm = str(e)

# inference throughput (forward only, bf16 weights)
pb = tree_map(lambda p: p.astype(mx.bfloat16), params)
fwd = mx.compile(lambda p, x: mlx_forward(p, x))
xi = mx.random.normal((64, 224, 224, 3)).astype(mx.bfloat16)
for _ in range(3):
    mx.eval(fwd(pb, xi))
t0 = time.perf_counter()
for _ in range(10):
    mx.eval(fwd(pb, xi))
inf_ips = 640 / (time.perf_counter() - t0)
print("inference img/s (ViT-S bf16 b64):", round(inf_ips, 1))
flops = train_flops_per_image(12, 384) * B
save("sustained", {"windows": windows, "pmset_therm": therm, "batch": B,
                   "steady_TFLOPS": round(flops / (windows[-1]["ms_step"] / 1e3) / 1e12, 2) if windows else None,
                   "vit_s_inference_img_per_s_b64_bf16": round(inf_ips, 1)})
