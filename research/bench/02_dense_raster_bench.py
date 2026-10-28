"""Micro-benchmark: dense (untiled) soft rasterisation of triangles and dense
Gaussian splatting, forward+backward, in pure MLX and pure PyTorch-MPS on this Mac.
No custom kernels. Cost is O(pixels x primitives). Reports ms/iter and peak memory."""
import sys, time, math
import numpy as np

CONFIGS = [  # (H, W, n_prims)
    (64, 64, 256),
    (128, 128, 512),
    (128, 128, 1024),
    (128, 128, 2048),
    (256, 256, 512),
]
SIGMA, GAMMA = 1e-2, 1e-1
rng = np.random.default_rng(0)


def make_tris(T):
    c = rng.uniform(-0.8, 0.8, (T, 1, 2))
    v = c + rng.normal(0, 0.08, (T, 3, 2))
    z = rng.uniform(1.0, 5.0, (T, 3))
    col = rng.uniform(0, 1, (T, 3))
    return v.astype(np.float32), z.astype(np.float32), col.astype(np.float32)


def make_gauss(K):
    mu = rng.uniform(-0.9, 0.9, (K, 2))
    s = rng.uniform(0.01, 0.06, (K, 2))
    th = rng.uniform(0, math.pi, (K,))
    op = rng.uniform(0.3, 0.9, (K,))
    col = rng.uniform(0, 1, (K, 3))
    z = rng.uniform(1, 5, (K,))
    return [a.astype(np.float32) for a in (mu, s, th, op, col, z)]


# ---------------------------------------------------------------- MLX
def run_mlx():
    import mlx.core as mx

    def pix(H, W):
        ys, xs = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
        return mx.array(np.stack([xs, ys], -1).reshape(-1, 2).astype(np.float32))

    def softras(v, z, col, P):
        # v (T,3,2) z (T,3) col (T,3) P (N,2)
        p = P[:, None, None, :]                      # N,1,1,2
        a = v[None]                                  # 1,T,3,2
        b = mx.roll(v, -1, axis=1)[None]
        e = b - a
        rel = p - a
        cross = e[..., 0] * rel[..., 1] - e[..., 1] * rel[..., 0]   # N,T,3
        elen = mx.sqrt((e ** 2).sum(-1) + 1e-12)
        area = (v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1]) - \
               (v[:, 1, 1] - v[:, 0, 1]) * (v[:, 2, 0] - v[:, 0, 0])   # T
        orient = mx.sign(area)[None, :, None]
        d = cross / elen * orient
        sd = mx.min(d, axis=-1)                                     # N,T
        D = mx.sigmoid(sd / SIGMA)
        bary = mx.roll(cross, -1, axis=-1) / (area[None, :, None] + 1e-8)
        bary = mx.clip(bary, 0, 1)
        bary = bary / (bary.sum(-1, keepdims=True) + 1e-8)
        zp = (bary * z[None]).sum(-1)                                # N,T
        zn = (5.0 - zp) / 4.0                                        # near -> 1
        logit = mx.log(D + 1e-6) + zn / GAMMA
        bg = mx.full((logit.shape[0], 1), 1e-3 / GAMMA)
        w = mx.softmax(mx.concatenate([logit, bg], 1), axis=1)
        rgb = w[:, :-1] @ col + w[:, -1:] * 1.0
        sil = 1 - mx.exp(mx.sum(mx.log(1 - D + 1e-6), axis=1))
        return rgb, sil

    def splat(mu, s, th, op, col, z, P):
        order = mx.argsort(z)
        mu, s, th, op, col = mu[order], s[order], th[order], op[order], col[order]
        c, sn = mx.cos(th), mx.sin(th)
        d = P[:, None, :] - mu[None]                 # N,K,2
        u = (c * d[..., 0] + sn * d[..., 1]) / s[:, 0]
        w_ = (-sn * d[..., 0] + c * d[..., 1]) / s[:, 1]
        alpha = op * mx.exp(-0.5 * (u * u + w_ * w_))
        alpha = mx.clip(alpha, 0, 0.99)
        logT = mx.cumsum(mx.log(1 - alpha), axis=1) - mx.log(1 - alpha)  # exclusive
        wgt = alpha * mx.exp(logT)
        return wgt @ col

    out = []
    for H, W, T in CONFIGS:
        P = pix(H, W)
        v, z, col = (mx.array(a) for a in make_tris(T))
        tgt = mx.array(rng.uniform(0, 1, (H * W, 3)).astype(np.float32))

        def loss_tri(v, z, col):
            rgb, sil = softras(v, z, col, P)
            return mx.mean((rgb - tgt) ** 2) + mx.mean(sil)
        vg = mx.value_and_grad(loss_tri, argnums=(0, 1, 2))
        res = {}
        for name, fn in [("eager", vg), ("compiled", mx.compile(vg))]:
            try:
                mx.reset_peak_memory()
                for _ in range(2):
                    l, g = fn(v, z, col); mx.eval(l, g)
                t0 = time.perf_counter(); n = 5
                for _ in range(n):
                    l, g = fn(v, z, col); mx.eval(l, g)
                dt = (time.perf_counter() - t0) / n * 1e3
                res[name] = (dt, mx.get_peak_memory() / 2**30)
            except Exception as ex:  # noqa
                res[name] = (float("nan"), str(ex)[:60])
        out.append(("mlx", "softras", H, W, T, res))

        g = [mx.array(a) for a in make_gauss(T)]
        def loss_g(mu, s, th, op, col):
            return mx.mean((splat(mu, s, th, op, col, g[5], P) - tgt) ** 2)
        vg2 = mx.value_and_grad(loss_g, argnums=(0, 1, 2, 3, 4))
        res = {}
        for name, fn in [("eager", vg2), ("compiled", mx.compile(vg2))]:
            try:
                mx.reset_peak_memory()
                for _ in range(2):
                    l, gr = fn(*g[:5]); mx.eval(l, gr)
                t0 = time.perf_counter(); n = 5
                for _ in range(n):
                    l, gr = fn(*g[:5]); mx.eval(l, gr)
                dt = (time.perf_counter() - t0) / n * 1e3
                res[name] = (dt, mx.get_peak_memory() / 2**30)
            except Exception as ex:  # noqa
                res[name] = (float("nan"), str(ex)[:60])
        out.append(("mlx", "dense-gs", H, W, T, res))
        mx.clear_cache()
    return out


# ---------------------------------------------------------------- torch MPS
def run_torch():
    import torch
    dev = torch.device("mps")

    def pix(H, W):
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
        return torch.stack([xs, ys], -1).reshape(-1, 2).to(dev)

    def softras(v, z, col, P):
        p = P[:, None, None, :]
        a = v[None]
        b = torch.roll(v, -1, dims=1)[None]
        e = b - a
        rel = p - a
        cross = e[..., 0] * rel[..., 1] - e[..., 1] * rel[..., 0]
        elen = torch.sqrt((e ** 2).sum(-1) + 1e-12)
        area = (v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1]) - \
               (v[:, 1, 1] - v[:, 0, 1]) * (v[:, 2, 0] - v[:, 0, 0])
        orient = torch.sign(area)[None, :, None]
        d = cross / elen * orient
        sd = d.min(-1).values
        D = torch.sigmoid(sd / SIGMA)
        bary = torch.roll(cross, -1, dims=-1) / (area[None, :, None] + 1e-8)
        bary = bary.clamp(0, 1)
        bary = bary / (bary.sum(-1, keepdim=True) + 1e-8)
        zp = (bary * z[None]).sum(-1)
        zn = (5.0 - zp) / 4.0
        logit = torch.log(D + 1e-6) + zn / GAMMA
        bg = torch.full((logit.shape[0], 1), 1e-3 / GAMMA, device=dev)
        w = torch.softmax(torch.cat([logit, bg], 1), 1)
        rgb = w[:, :-1] @ col + w[:, -1:] * 1.0
        sil = 1 - torch.exp(torch.log(1 - D + 1e-6).sum(1))
        return rgb, sil

    out = []
    for H, W, T in CONFIGS:
        P = pix(H, W)
        v, z, col = (torch.tensor(a, device=dev, requires_grad=True) for a in make_tris(T))
        tgt = torch.rand(H * W, 3, device=dev)
        try:
            for i in range(7):
                if i == 2:
                    torch.mps.synchronize(); t0 = time.perf_counter()
                rgb, sil = softras(v, z, col, P)
                l = ((rgb - tgt) ** 2).mean() + sil.mean()
                l.backward()
            torch.mps.synchronize()
            dt = (time.perf_counter() - t0) / 5 * 1e3
            mem = torch.mps.driver_allocated_memory() / 2**30
            out.append(("torch-mps", "softras", H, W, T, {"eager": (dt, mem)}))
        except Exception as ex:  # noqa
            out.append(("torch-mps", "softras", H, W, T, {"eager": (float("nan"), str(ex)[:60])}))
        del v, z, col
        torch.mps.empty_cache()
    return out


if __name__ == "__main__":
    which = sys.argv[1]
    rows = run_mlx() if which == "mlx" else run_torch()
    for r in rows:
        lib, kind, H, W, T, res = r
        s = "  ".join(f"{k}: {v[0]:.1f} ms, peak {v[1] if isinstance(v[1], str) else f'{v[1]:.2f} GB'}" for k, v in res.items())
        print(f"{lib:10s} {kind:9s} {H}x{W} prims={T:5d}  pairs={H*W*T/1e6:6.1f}M  {s}", flush=True)
