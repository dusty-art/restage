"""PyTorch custom Metal kernel via torch.mps.compile_shader, wrapped in an
autograd.Function (fused SiLU forward + backward), checked against
torch.nn.functional.silu and timed.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

from common import save  # noqa: E402

SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void silu_fwd(device float* y [[buffer(0)]], const device float* x [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    float v = x[i];
    y[i] = v / (1.0f + exp(-v));
}
kernel void silu_bwd(device float* gx [[buffer(0)]], const device float* x [[buffer(1)]],
                     const device float* gy [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    float v = x[i];
    float s = 1.0f / (1.0f + exp(-v));
    gx[i] = gy[i] * s * (1.0f + v * (1.0f - s));
}
"""
lib = torch.mps.compile_shader(SRC)


class SiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        y = torch.empty_like(x)
        lib.silu_fwd(y, x, threads=x.numel())
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, gy):
        (x,) = ctx.saved_tensors
        gx = torch.empty_like(x)
        lib.silu_bwd(gx, x, gy.contiguous(), threads=x.numel())
        return gx


x = torch.randn(1 << 24, device="mps", requires_grad=True)
y = SiLU.apply(x)
y.sum().backward()
g1 = x.grad.clone()
x.grad = None
y_ref = torch.nn.functional.silu(x)
y_ref.sum().backward()
res = {"fwd_max_abs_err": float((y - y_ref).abs().max()), "bwd_max_abs_err": float((g1 - x.grad).abs().max())}


def t(fn):
    for _ in range(3):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / 20 * 1e3


with torch.no_grad():
    res["custom_fwd_ms"] = round(t(lambda: SiLU.apply(x)), 3)
    res["torch_silu_fwd_ms"] = round(t(lambda: torch.nn.functional.silu(x)), 3)
print(res)
save("torch_shader", res)
