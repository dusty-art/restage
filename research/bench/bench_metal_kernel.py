"""Benchmark 6: does mx.fast.metal_kernel work? Tiny example + a real one.

Part A: a tiny elementwise custom kernel (fused x*sigmoid(x)), checked
        against MLX ops.
Part B: a differentiable SoftRas-style triangle rasteriser as two custom
        Metal kernels wrapped in mx.custom_function with a hand-written VJP:
          forward : one thread per pixel, loop over triangles, accumulate
                    sum(w*col) and sum(w)   (w = prod_j sigmoid(d_j) * zw)
          backward: one thread per (triangle, pixel-block), loop over the
                    block's pixels, accumulate 19 per-triangle gradients in
                    registers, write partials (no atomics), sum in MLX.
        Same maths as raster_ops.tri_soft, so gradients are checked
        against MLX autodiff of the plain-op version. Optional culling
        (skip a triangle for a pixel once an edge sigmoid < ~1e-5).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx  # noqa: E402

import raster_ops as R  # noqa: E402
from common import fmt_table, save  # noqa: E402

HW = R.H * R.W
NB = 16  # pixel blocks in backward

# ------------------------------------------------------------------ part A
silu_kernel = mx.fast.metal_kernel(
    name="my_silu",
    input_names=["x"],
    output_names=["y"],
    source="""
        uint i = thread_position_in_grid.x;
        T v = x[i];
        y[i] = v / (T(1) + metal::exp(-v));
    """,
)


def part_a():
    x = mx.random.normal((1 << 24,))
    y = silu_kernel(inputs=[x], template=[("T", mx.float32)], grid=(x.size, 1, 1), threadgroup=(256, 1, 1),
                    output_shapes=[x.shape], output_dtypes=[x.dtype])[0]
    ref = x * mx.sigmoid(x)
    err = float(mx.abs(y - ref).max())

    def t(fn):
        for _ in range(3):
            mx.eval(fn())
        t0 = time.perf_counter()
        for _ in range(20):
            mx.eval(fn())
        return (time.perf_counter() - t0) / 20 * 1e3
    tk = t(lambda: silu_kernel(inputs=[x], template=[("T", mx.float32)], grid=(x.size, 1, 1),
                               threadgroup=(256, 1, 1), output_shapes=[x.shape], output_dtypes=[x.dtype])[0])
    tr = t(lambda: x * mx.sigmoid(x))
    tc = t(lambda: mx.compile(lambda a: a * mx.sigmoid(a))(x))
    gbps = 2 * x.size * 4 / (tk / 1e3) / 1e9
    return {"max_abs_err": err, "custom_ms": round(tk, 3), "mlx_ops_ms": round(tr, 3), "mlx_compiled_ms": round(tc, 3),
            "custom_GBps": round(gbps, 1)}


# ------------------------------------------------------------------ part B
FWD_SRC = """
    uint pix = thread_position_in_grid.x;
    if (pix >= HWN) return;
    float px = float(pix % WIDTH) + 0.5f;
    float py = float(pix / WIDTH) + 0.5f;
    int N = tri_shape[0];
    float n0 = 0.f, n1 = 0.f, n2 = 0.f, den = 0.f;
    for (int t = 0; t < N; ++t) {
        const device float* T = tri + t * 19;
        float cov = 1.0f;
        bool skip = false;
        for (int j = 0; j < 3; ++j) {
            float d = ((px - T[2*j]) * T[6 + 2*j + 1] - (py - T[2*j + 1]) * T[6 + 2*j]) * T[12 + j];
            if (CULL && d < -11.5f) { skip = true; break; }
            cov *= 1.0f / (1.0f + metal::exp(-d));
        }
        if (skip) continue;
        float w = cov * T[15];
        n0 += w * T[16]; n1 += w * T[17]; n2 += w * T[18]; den += w;
    }
    acc[pix * 4 + 0] = n0; acc[pix * 4 + 1] = n1; acc[pix * 4 + 2] = n2; acc[pix * 4 + 3] = den;
"""

BWD_SRC = """
    uint t = thread_position_in_grid.x;
    uint b = thread_position_in_grid.y;
    int N = tri_shape[0];
    if (int(t) >= N) return;
    const device float* T = tri + t * 19;
    float v[6], e[6], L[3];
    for (int k = 0; k < 6; ++k) { v[k] = T[k]; e[k] = T[6 + k]; }
    for (int k = 0; k < 3; ++k) L[k] = T[12 + k];
    float zw = T[15], c0 = T[16], c1 = T[17], c2 = T[18];
    float g[19];
    for (int k = 0; k < 19; ++k) g[k] = 0.f;
    int per = (HWN + NBLK - 1) / NBLK;
    int p0 = int(b) * per, p1 = min(HWN, p0 + per);
    for (int pix = p0; pix < p1; ++pix) {
        float px = float(pix % WIDTH) + 0.5f;
        float py = float(pix / WIDTH) + 0.5f;
        float A[3], B[3], s[3];
        bool skip = false;
        for (int j = 0; j < 3; ++j) {
            A[j] = px - v[2*j]; B[j] = py - v[2*j + 1];
            float d = (A[j] * e[2*j + 1] - B[j] * e[2*j]) * L[j];
            if (CULL && d < -11.5f) { skip = true; break; }
            s[j] = 1.0f / (1.0f + metal::exp(-d));
        }
        if (skip) continue;
        float cov = s[0] * s[1] * s[2];
        float w = cov * zw;
        const device float* G = gacc + pix * 4;
        float gw = G[0] * c0 + G[1] * c1 + G[2] * c2 + G[3];
        g[16] += w * G[0]; g[17] += w * G[1]; g[18] += w * G[2];
        g[15] += gw * cov;
        float gcov = gw * zw;
        for (int j = 0; j < 3; ++j) {
            float other = s[(j + 1) % 3] * s[(j + 2) % 3];
            float gd = gcov * other * s[j] * (1.0f - s[j]);
            g[2*j]       += gd * (-e[2*j + 1] * L[j]);
            g[2*j + 1]   += gd * ( e[2*j] * L[j]);
            g[6 + 2*j]   += gd * (-B[j] * L[j]);
            g[6 + 2*j + 1] += gd * ( A[j] * L[j]);
            g[12 + j]    += gd * (A[j] * e[2*j + 1] - B[j] * e[2*j]);
        }
    }
    device float* O = partial + (int(b) * N + int(t)) * 19;
    for (int k = 0; k < 19; ++k) O[k] = g[k];
"""

fwd_k = mx.fast.metal_kernel(name="softtri_fwd", input_names=["tri"], output_names=["acc"], source=FWD_SRC)
bwd_k = mx.fast.metal_kernel(name="softtri_bwd", input_names=["tri", "gacc"], output_names=["partial"], source=BWD_SRC)


def make_softtri(cull):
    tmpl = [("HWN", HW), ("WIDTH", R.W), ("NBLK", NB), ("CULL", bool(cull))]

    @mx.custom_function
    def softtri_accum(tri):
        return fwd_k(inputs=[tri], template=tmpl, grid=(HW, 1, 1), threadgroup=(256, 1, 1),
                     output_shapes=[(HW, 4)], output_dtypes=[mx.float32])[0]

    @softtri_accum.vjp
    def softtri_vjp(primals, cotangent, output):
        # single-input custom_function: MLX passes the primal array itself, not a tuple
        tri = primals[0] if isinstance(primals, (list, tuple)) else primals
        N = tri.shape[0]
        part = bwd_k(inputs=[tri, cotangent], template=tmpl, grid=(N, NB, 1), threadgroup=(64, 1, 1),
                     output_shapes=[(NB, N, 19)], output_dtypes=[mx.float32])[0]
        return part.sum(axis=0)

    return softtri_accum


def kernel_loss(p, tgt, accum):
    prims_fn, _, bg_w = R._mlx_setup("tri_soft")
    v, e, L, zw, col = prims_fn(p)
    N = v.shape[0]
    tri = mx.concatenate([v.reshape(N, 6), e.reshape(N, 6), L, zw[:, None], col], axis=1)
    acc = accum(tri)
    img = (acc[:, :3] / (acc[:, 3:] + bg_w + R.EPS)).reshape(R.H, R.W, 3)
    return ((img - tgt) ** 2).mean(), img


def part_b():
    rows = []
    tgt = mx.array(np.random.default_rng(1).uniform(0, 1, (R.H, R.W, 3)).astype(np.float32))
    for N in [1000, 10000, 100000]:
        pn = R.make_params_np(N, "tri_soft")
        p = {k: mx.array(v) for k, v in pn.items()}
        ref_loss = ref_g = None
        if N <= 10000:  # reference gradients from the plain-op MLX version
            ref_loss, ref_g = R.mlx_stream_value_and_grad(p, "tri_soft", tgt)
            mx.eval(ref_loss, ref_g)
        for cull in [False, True]:
            accum = make_softtri(cull)
            vg = mx.value_and_grad(lambda q: kernel_loss(q, tgt, accum)[0])
            fwd = lambda: kernel_loss(p, tgt, accum)[1]
            iters = 5 if N <= 10000 else 3
            res = {"N": N, "cull": cull}
            for name, fn in [("fwd", fwd), ("fwdbwd", lambda: vg(p))]:
                mx.eval(fn())
                mx.reset_peak_memory()
                ts = []
                for _ in range(iters):
                    t0 = time.perf_counter()
                    mx.eval(fn())
                    ts.append(time.perf_counter() - t0)
                ts.sort()
                res[f"{name}_ms"] = round(ts[len(ts) // 2] * 1e3, 2)
                res[f"{name}_mem_GB"] = round(mx.get_peak_memory() / 1e9, 3)
            if ref_g is not None:
                loss, g = vg(p)
                rel = {k: float(mx.abs(g[k] - ref_g[k]).max() / (mx.abs(ref_g[k]).max() + 1e-20)) for k in g}
                res["loss_rel_err"] = float(abs(loss - ref_loss) / abs(ref_loss))
                res["grad_max_rel_err"] = max(rel.values())
            print(res, flush=True)
            rows.append(res)
    return rows


if __name__ == "__main__":
    a = part_a()
    print("part A:", a)
    rows = part_b()
    print(fmt_table(rows, ["N", "cull", "fwd_ms", "fwdbwd_ms", "fwdbwd_mem_GB", "loss_rel_err", "grad_max_rel_err"]))
    save("metal_kernel", {"part_a_silu": a, "part_b_softtri": rows})
