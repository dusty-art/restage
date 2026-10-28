"""Plain-op differentiable 2D splatting / soft rasterisation, in MLX and torch.

Three methods, all producing a 128x128 RGB image and fully differentiable
w.r.t. per-primitive parameters:

  gauss_dense : every pixel x every Gaussian (O(HW*N)), normalised weighted sum.
                Chunked over N; each chunk is gradient-checkpointed so memory
                stays O(HW*chunk).
  gauss_local : each Gaussian splats onto a fixed KxK (9x9) footprint around its
                centre via scatter-add (O(N*K^2)). This is what a tile-based
                splatter does, minus sorting.
  tri_soft    : SoftRas-style soft triangles (O(HW*N)): coverage = product of
                3 edge sigmoids, depth-softmax aggregation with a background
                term, flat per-triangle colour (low-poly look). Chunked +
                checkpointed like gauss_dense.

Order-independent aggregation (sum of weights) is used everywhere so chunks
can be summed; true front-to-back alpha compositing needs per-pixel sorting
and is what custom kernels are for.
"""
import math

H = W = 128
SIGMA_TRI = 0.7      # px, edge softness
GAMMA = 0.05         # depth softmax temperature
EPS = 1e-6
K = 9                # local footprint


def make_params_np(N, method, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    if method.startswith("gauss"):
        mu = rng.uniform(0, W, (N, 2)).astype(np.float32)
        # sigma ~ 0.6-2.5 px so the 9x9 footprint (+-4px) holds ~>2 sigma
        log_s = np.log(rng.uniform(0.6, 2.0, (N, 2))).astype(np.float32)
        rot = rng.uniform(0, math.pi, (N, 1)).astype(np.float32)
        col = rng.normal(0, 1, (N, 3)).astype(np.float32)
        op = rng.normal(0, 1, (N, 1)).astype(np.float32)
        return {"mu": mu, "log_s": log_s, "rot": rot, "col": col, "op": op}
    else:
        c = rng.uniform(0, W, (N, 1, 2))
        r = max(2.0, 1.5 * W / math.sqrt(N))
        v = (c + rng.normal(0, r, (N, 3, 2))).astype(np.float32)
        z = rng.uniform(0.1, 0.9, (N, 1)).astype(np.float32)
        col = rng.normal(0, 1, (N, 3)).astype(np.float32)
        return {"v": v, "z": z, "col": col}


# ======================================================================= MLX

def _mlx_setup(method):
    """Returns (prims_fn, chunk_fn, bg_w) for the dense methods."""
    import mlx.core as mx
    if method == "gauss_dense":
        def prims_fn(p):
            s = mx.exp(p["log_s"])
            c, sn = mx.cos(p["rot"][:, 0]), mx.sin(p["rot"][:, 0])
            isx, isy = 1 / (s[:, 0] ** 2), 1 / (s[:, 1] ** 2)
            conic = mx.stack([c * c * isx + sn * sn * isy, c * sn * (isx - isy), sn * sn * isx + c * c * isy], 1)
            return [p["mu"], conic, mx.sigmoid(p["op"][:, 0]), mx.sigmoid(p["col"])]

        def chunk_fn(px, py, mu, conic, op, col):
            dx = px - mu[:, 0]
            dy = py - mu[:, 1]
            q = conic[:, 0] * dx * dx + 2 * conic[:, 1] * dx * dy + conic[:, 2] * dy * dy
            w = op * mx.exp(-0.5 * q)
            return w @ col, w.sum(axis=1)
        return prims_fn, chunk_fn, None

    def prims_fn(p):  # tri_soft
        v = p["v"]
        e = mx.roll(v, -1, axis=1) - v                                   # [N,3,2]
        area = e[:, 0, 0] * e[:, 1, 1] - e[:, 0, 1] * e[:, 1, 0]
        sgn = mx.stop_gradient(mx.where(area >= 0, 1.0, -1.0))
        inv_len = mx.rsqrt((e * e).sum(-1) + EPS) * sgn[:, None] / SIGMA_TRI  # [N,3]
        zw = mx.exp((1.0 - p["z"][:, 0]) / GAMMA)
        return [v, e, inv_len, zw, mx.sigmoid(p["col"])]

    def chunk_fn(px, py, v, e, inv_len, zw, col):
        cov = None
        for j in range(3):
            d = ((px - v[:, j, 0]) * e[:, j, 1] - (py - v[:, j, 1]) * e[:, j, 0]) * inv_len[:, j]
            sj = mx.sigmoid(d)
            cov = sj if cov is None else cov * sj
        w = cov * zw
        return w @ col, w.sum(axis=1)
    return prims_fn, chunk_fn, math.exp(EPS / GAMMA)


def _mlx_pixels():
    import mlx.core as mx
    ys, xs = mx.meshgrid(mx.arange(H, dtype=mx.float32) + 0.5, mx.arange(W, dtype=mx.float32) + 0.5, indexing="ij")
    return xs.reshape(-1, 1), ys.reshape(-1, 1)


def mlx_render(p, method, chunk=1024, compile_chunk=False):
    import mlx.core as mx
    if method == "gauss_local":
        return _mlx_gauss_local(p)
    px, py = _mlx_pixels()
    prims_fn, chunk_fn, bg_w = _mlx_setup(method)
    prims = prims_fn(p)
    f = mx.compile(chunk_fn) if compile_chunk else chunk_fn
    N = prims[0].shape[0]
    if N > chunk:
        f = mx.checkpoint(f)
    acc_c, acc_w = 0.0, 0.0
    for s0 in range(0, N, chunk):
        cc, ww = f(px, py, *[t[s0:s0 + chunk] for t in prims])
        acc_c, acc_w = acc_c + cc, acc_w + ww
    if bg_w is not None:
        acc_w = acc_w + bg_w  # black background
    img = acc_c / (acc_w[:, None] + EPS)
    return img.reshape(H, W, 3)


def mlx_stream_render(p, method, chunk=1024, compile_chunk=True):
    """Forward only, evaluated chunk by chunk (bounded memory)."""
    import mlx.core as mx
    if method == "gauss_local":
        return _mlx_gauss_local(p)
    px, py = _mlx_pixels()
    prims_fn, chunk_fn, bg_w = _mlx_setup(method)
    f = mx.compile(chunk_fn) if compile_chunk else chunk_fn
    prims = prims_fn(p)
    acc_c, acc_w = mx.zeros((H * W, 3)), mx.zeros((H * W,))
    for s0 in range(0, prims[0].shape[0], chunk):
        cc, ww = f(px, py, *[t[s0:s0 + chunk] for t in prims])
        acc_c, acc_w = acc_c + cc, acc_w + ww
        mx.eval(acc_c, acc_w)
    acc_w = acc_w + (bg_w or 0.0)
    return (acc_c / (acc_w[:, None] + EPS)).reshape(H, W, 3)


def mlx_stream_value_and_grad(p, method, tgt, chunk=1024, compile_chunk=True):
    """Memory-bounded dense render + backward for MLX.

    MLX builds one lazy graph per eval, so chunk + mx.checkpoint does NOT
    bound memory (all chunks' recomputed intermediates are alive in the same
    backward graph). Instead: (1) forward chunk by chunk with an mx.eval per
    chunk, (2) get dL/d(accumulators) analytically via autodiff of the cheap
    normalisation + loss, (3) per-chunk mx.vjp with an eval per chunk,
    (4) back-propagate from primitive attributes to raw parameters.
    Returns (loss, grads) like mx.value_and_grad.
    """
    import mlx.core as mx
    px, py = _mlx_pixels()
    prims_fn, chunk_fn, bg_w = _mlx_setup(method)
    f = mx.compile(chunk_fn) if compile_chunk else chunk_fn
    prims = prims_fn(p)
    mx.eval(prims)
    N = prims[0].shape[0]
    acc_c = mx.zeros((H * W, 3))
    acc_w = mx.zeros((H * W,))
    for s0 in range(0, N, chunk):
        cc, ww = f(px, py, *[t[s0:s0 + chunk] for t in prims])
        acc_c, acc_w = acc_c + cc, acc_w + ww
        mx.eval(acc_c, acc_w)

    def head(ac, aw):
        aw = aw + (bg_w or 0.0)
        img = (ac / (aw[:, None] + EPS)).reshape(H, W, 3)
        return ((img - tgt) ** 2).mean()
    loss, (g_c, g_w) = mx.value_and_grad(head, argnums=(0, 1))(acc_c, acc_w)
    mx.eval(loss, g_c, g_w)
    gparts = [[] for _ in prims]
    for s0 in range(0, N, chunk):
        sl = [t[s0:s0 + chunk] for t in prims]
        _, vj = mx.vjp(lambda *a: f(px, py, *a), sl, [g_c, g_w])
        mx.eval(vj)
        for i, g in enumerate(vj):
            gparts[i].append(g)
    g_prims = [mx.concatenate(g, 0) for g in gparts]
    keys = list(p.keys())
    _, gl = mx.vjp(lambda *a: prims_fn(dict(zip(keys, a))), [p[k] for k in keys], g_prims)
    return loss, dict(zip(keys, gl))


def _mlx_gauss_local(p):
    import mlx.core as mx
    N = p["mu"].shape[0]
    r = K // 2
    o = mx.arange(-r, r + 1, dtype=mx.float32)
    oy, ox = mx.meshgrid(o, o, indexing="ij")
    offs = mx.stack([ox.reshape(-1), oy.reshape(-1)], 1)                 # [K*K,2]
    mu = p["mu"]
    s = mx.exp(p["log_s"])
    c, sn = mx.cos(p["rot"][:, 0]), mx.sin(p["rot"][:, 0])
    isx, isy = 1 / (s[:, 0] ** 2), 1 / (s[:, 1] ** 2)
    a, b, cc = c * c * isx + sn * sn * isy, c * sn * (isx - isy), sn * sn * isx + c * c * isy
    base = mx.stop_gradient(mx.floor(mu))
    pix = base[:, None, :] + offs[None]                                  # [N,KK,2]
    valid = ((pix[..., 0] >= 0) & (pix[..., 0] < W) & (pix[..., 1] >= 0) & (pix[..., 1] < H))
    d = pix + 0.5 - mu[:, None, :]
    dx, dy = d[..., 0], d[..., 1]
    q = a[:, None] * dx * dx + 2 * b[:, None] * dx * dy + cc[:, None] * dy * dy
    w = mx.sigmoid(p["op"]) * mx.exp(-0.5 * q) * valid
    col = mx.sigmoid(p["col"])
    vals = mx.concatenate([w[..., None] * col[:, None, :], w[..., None]], -1).reshape(-1, 4)
    idx = (mx.clip(pix[..., 1], 0, H - 1) * W + mx.clip(pix[..., 0], 0, W - 1)).astype(mx.int32).reshape(-1)
    acc = mx.zeros((H * W, 4)).at[idx].add(vals)
    img = acc[:, :3] / (acc[:, 3:] + EPS)
    return img.reshape(H, W, 3)


# ===================================================================== torch

def torch_render(p, method, chunk=1024):
    import torch
    from torch.utils.checkpoint import checkpoint
    dev = p[next(iter(p))].device
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32) + 0.5,
                            torch.arange(W, device=dev, dtype=torch.float32) + 0.5, indexing="ij")
    px, py = xs.reshape(-1, 1), ys.reshape(-1, 1)

    if method == "gauss_local":
        return _torch_gauss_local(p)

    if method == "gauss_dense":
        s = torch.exp(p["log_s"])
        c, sn = torch.cos(p["rot"][:, 0]), torch.sin(p["rot"][:, 0])
        isx, isy = 1 / s[:, 0] ** 2, 1 / s[:, 1] ** 2
        conic = torch.stack([c * c * isx + sn * sn * isy, c * sn * (isx - isy), sn * sn * isx + c * c * isy], 1)
        prims = [p["mu"], conic, torch.sigmoid(p["op"][:, 0]), torch.sigmoid(p["col"])]

        def chunk_fn(px, py, mu, conic, op, col):
            dx = px - mu[:, 0]
            dy = py - mu[:, 1]
            q = conic[:, 0] * dx * dx + 2 * conic[:, 1] * dx * dy + conic[:, 2] * dy * dy
            w = op * torch.exp(-0.5 * q)
            return w @ col, w.sum(1)
        bg_w = None
    else:
        v = p["v"]
        e = torch.roll(v, -1, dims=1) - v
        area = e[:, 0, 0] * e[:, 1, 1] - e[:, 0, 1] * e[:, 1, 0]
        sgn = torch.where(area >= 0, 1.0, -1.0).detach()
        inv_len = torch.rsqrt((e * e).sum(-1) + EPS) * sgn[:, None] / SIGMA_TRI
        zw = torch.exp((1.0 - p["z"][:, 0]) / GAMMA)
        prims = [v, e, inv_len, zw, torch.sigmoid(p["col"])]

        def chunk_fn(px, py, v, e, inv_len, zw, col):
            cov = None
            for j in range(3):
                d = ((px - v[:, j, 0]) * e[:, j, 1] - (py - v[:, j, 1]) * e[:, j, 0]) * inv_len[:, j]
                sj = torch.sigmoid(d)
                cov = sj if cov is None else cov * sj
            w = cov * zw
            return w @ col, w.sum(1)
        bg_w = math.exp(EPS / GAMMA)

    N = prims[0].shape[0]
    acc_c, acc_w = 0.0, 0.0
    for s0 in range(0, N, chunk):
        args = [px, py] + [t[s0:s0 + chunk] for t in prims]
        if N > chunk and torch.is_grad_enabled():
            cc, ww = checkpoint(chunk_fn, *args, use_reentrant=False)
        else:
            cc, ww = chunk_fn(*args)
        acc_c, acc_w = acc_c + cc, acc_w + ww
    if bg_w is not None:
        acc_w = acc_w + bg_w
    img = acc_c / (acc_w[:, None] + EPS)
    return img.reshape(H, W, 3)


def _torch_gauss_local(p):
    import torch
    dev = p["mu"].device
    r = K // 2
    o = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    oy, ox = torch.meshgrid(o, o, indexing="ij")
    offs = torch.stack([ox.reshape(-1), oy.reshape(-1)], 1)
    mu = p["mu"]
    s = torch.exp(p["log_s"])
    c, sn = torch.cos(p["rot"][:, 0]), torch.sin(p["rot"][:, 0])
    isx, isy = 1 / s[:, 0] ** 2, 1 / s[:, 1] ** 2
    a, b, cc = c * c * isx + sn * sn * isy, c * sn * (isx - isy), sn * sn * isx + c * c * isy
    base = torch.floor(mu).detach()
    pix = base[:, None, :] + offs[None]
    valid = ((pix[..., 0] >= 0) & (pix[..., 0] < W) & (pix[..., 1] >= 0) & (pix[..., 1] < H)).float()
    d = pix + 0.5 - mu[:, None, :]
    dx, dy = d[..., 0], d[..., 1]
    q = a[:, None] * dx * dx + 2 * b[:, None] * dx * dy + cc[:, None] * dy * dy
    w = torch.sigmoid(p["op"]) * torch.exp(-0.5 * q) * valid
    col = torch.sigmoid(p["col"])
    vals = torch.cat([w[..., None] * col[:, None, :], w[..., None]], -1).reshape(-1, 4)
    idx = (pix[..., 1].clamp(0, H - 1) * W + pix[..., 0].clamp(0, W - 1)).long().reshape(-1)
    acc = torch.zeros(H * W, 4, device=dev).index_add(0, idx, vals)
    img = acc[:, :3] / (acc[:, 3:] + EPS)
    return img.reshape(H, W, 3)
