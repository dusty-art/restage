"""Timing and memory of the renderer on the realistic training workload.

Workload (per job = one keyframe view): a ground plane (16 x 16 cells, 512 triangles),
24 rounded boxes (the set, 2592 triangles), 8 humanoid slots (5376 triangles) and 2 object
boxes (216 triangles): 8696 triangles, 4889 vertices. Some slots are absent (opacity 0).
Attributes C = 16 (RGB from region colours, an 11-way instance one-hot, 2 spare).
B = 64 jobs, rendered at 64 x 36 and 128 x 72.

Run:  .venv/bin/python -m restage.render.bench_render [--quick] [--stages]
"""
from __future__ import annotations

import argparse
import json
import math
import time

import mlx.core as mx
import numpy as np

from .camera import look_at, project
from .fk import axis_angle_to_mat, take_static
from .raster import rasterize
from .shapes import ground_plane, humanoid, merge_meshes, rounded_boxes

N_BOX, N_PEOPLE, N_OBJ, C = 24, 8, 2, 16


def make_params(B: int, seed: int = 0) -> dict:
    """Random but plausible scene parameters (numpy), the decoder's outputs for B views."""
    rng = np.random.default_rng(seed)
    f32 = np.float32
    # set: 4 walls of a 8 x 8 m room + furniture boxes
    centers = np.zeros((B, N_BOX, 3), f32)
    sizes = np.zeros((B, N_BOX, 3), f32)
    walls = [((0, 1.4, -4), (8, 2.8, 0.2)), ((0, 1.4, 4), (8, 2.8, 0.2)), ((-4, 1.4, 0), (0.2, 2.8, 8)),
             ((4, 1.4, 0), (0.2, 2.8, 8))]
    for k, (c, s) in enumerate(walls):
        centers[:, k], sizes[:, k] = c, s
    nf = N_BOX - 4
    sizes[:, 4:] = rng.uniform([0.3, 0.3, 0.3], [1.8, 1.6, 1.2], (B, nf, 3))
    centers[:, 4:, [0, 2]] = rng.uniform(-3.4, 3.4, (B, nf, 2))
    centers[:, 4:, 1] = sizes[:, 4:, 1] / 2
    yaw = rng.uniform(-math.pi, math.pi, (B, N_BOX)).astype(f32)
    yaw[:, :4] = 0
    rnd = rng.uniform(0, 0.6, (B, N_BOX)).astype(f32)
    rnd[:, :4] = 0
    box_present = (rng.random((B, N_BOX)) < 0.8).astype(f32)
    box_present[:, :4] = 1
    # people
    P = B * N_PEOPLE
    shape = rng.normal(1, 0.06, (P, 4)).astype(f32)
    joint_aa = rng.normal(0, 0.25, (P, 22, 3)).astype(f32)
    root_yaw = rng.uniform(-math.pi, math.pi, P).astype(f32)
    root_pos = np.stack([rng.uniform(-2.5, 2.5, P), np.zeros(P), rng.uniform(-2.5, 2.5, P)], -1).astype(f32)
    person_present = (rng.random((B, N_PEOPLE)) < 0.6).astype(f32)
    # objects
    obj_c = np.stack([rng.uniform(-2, 2, (B, N_OBJ)), rng.uniform(0.1, 1.0, (B, N_OBJ)),
                      rng.uniform(-2, 2, (B, N_OBJ))], -1).astype(f32)
    obj_s = rng.uniform(0.1, 0.4, (B, N_OBJ, 3)).astype(f32)
    # cameras: inside the room, looking at the middle
    ang = rng.uniform(-math.pi, math.pi, B)
    rad = rng.uniform(2.5, 3.6, B)
    eye = np.stack([rad * np.sin(ang), rng.uniform(1.2, 2.0, B), rad * np.cos(ang)], -1).astype(f32)
    target = np.stack([rng.uniform(-0.5, 0.5, B), rng.uniform(0.6, 1.2, B), rng.uniform(-0.5, 0.5, B)], -1).astype(f32)
    fov = rng.uniform(40, 60, B).astype(f32)
    colors = rng.uniform(0.1, 0.9, (B, N_BOX + N_PEOPLE * 5 + N_OBJ, 3)).astype(f32)
    return dict(centers=centers, sizes=sizes, yaw=yaw, rnd=rnd, box_present=box_present, shape=shape,
                joint_aa=joint_aa, root_yaw=root_yaw, root_pos=root_pos, person_present=person_present,
                obj_c=obj_c, obj_s=obj_s, eye=eye, target=target, fov=fov, colors=colors)


def build_scene(p: dict, B: int):
    """Scene parameters (MLX) -> (verts [B, V, 3] world, faces, attrs [B, V, C], opacity [B, V], cam)."""
    P = B * N_PEOPLE
    gv, gf, _ = ground_plane(p["eye"], extent=40.0, n=16)
    bv, bf, bm = rounded_boxes(p["centers"], p["sizes"], p["yaw"], p["rnd"])
    ov, of, om = rounded_boxes(p["obj_c"], p["obj_s"], mx.zeros((B, N_OBJ)), mx.full((B, N_OBJ), 0.3))
    jr = axis_angle_to_mat(p["joint_aa"])
    rr = axis_angle_to_mat(mx.stack([mx.zeros(P), p["root_yaw"], mx.zeros(P)], -1))
    hv, hf, hm = humanoid(p["shape"], jr, rr, p["root_pos"])
    Vh = hv.shape[1]
    hv = hv.reshape(B, N_PEOPLE * Vh, 3)
    hf_all = (hf[None] + (np.arange(N_PEOPLE) * Vh)[:, None, None]).reshape(-1, 3)
    verts, faces, offs = merge_meshes((gv, gf), (bv, bf), (ov, of), (hv, hf_all))
    V = verts.shape[1]
    # per-vertex attributes: rgb, instance one-hot (0 = set, 1..8 people, 9..10 objects), 2 spare
    inst = np.zeros(V, np.int32)
    col_idx = np.zeros(V, np.int32)
    o_g, o_b, o_o, o_h = offs
    col_idx[o_b:o_o] = bm["box"]
    inst[o_o:o_h] = 1 + N_PEOPLE + om["box"]
    col_idx[o_o:o_h] = N_BOX + N_PEOPLE * 5 + om["box"]
    slot = np.repeat(np.arange(N_PEOPLE), Vh)
    inst[o_h:] = 1 + slot
    col_idx[o_h:] = N_BOX + slot * 5 + np.tile(hm["region"], N_PEOPLE)
    rgb = take_static(p["colors"], col_idx, 1)
    onehot = mx.array(np.eye(11, dtype=np.float32)[inst])
    attrs = mx.concatenate([rgb, mx.broadcast_to(onehot[None], (B, V, 11)), mx.zeros((B, V, 2))], -1)
    op_set = mx.concatenate([mx.ones((B, gv.shape[1])), mx.repeat(p["box_present"], bm["n_vert_per_box"], 1),
                             mx.ones((B, ov.shape[1])), mx.repeat(p["person_present"], Vh, 1)], 1)
    camR = look_at(p["eye"], p["target"])
    return verts, faces, attrs, op_set, (camR, p["eye"], p["fov"])


def _time(fn, iters):
    mx.eval(fn())
    mx.synchronize()
    mx.reset_peak_memory()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3), float(np.min(ts) * 1e3), mx.get_peak_memory() / 1e9


def run(B=64, sizes=((36, 64), (72, 128)), iters=10, sigma=0.5, gamma=0.05):
    mx.set_memory_limit(8 << 30)
    pn = make_params(B)
    p = {k: mx.array(v) for k, v in pn.items()}
    verts_w, faces, attrs, opacity, (camR, eye, fov) = build_scene(p, B)
    mx.eval(verts_w, attrs, opacity)
    F = faces.shape[0]
    rows = []
    for H, W in sizes:
        ndc = project(verts_w, camR, eye, fov, W / H)
        mx.eval(ndc)
        rng = np.random.default_rng(1)
        tgt_a = mx.array((rng.random((B, H, W)) < 0.5).astype(np.float32))
        tgt_d = mx.array(rng.uniform(1, 6, (B, H, W)).astype(np.float32))
        tgt_c = mx.array(rng.uniform(0, 1, (B, H, W, C)).astype(np.float32))
        kw = dict(height=H, width=W, sigma=sigma, gamma=gamma)
        out = rasterize(ndc, faces, attrs, opacity, **kw)
        mx.eval(out)
        n_ent, cap = int(out.n_entries.item()), out.capacity

        def loss_fn(v, a, o):
            r = rasterize(v, faces, a, o, **kw)
            return (mx.abs(r.alpha - tgt_a).mean() + (r.alpha * mx.abs(r.depth - tgt_d)).mean()
                    + ((r.attrs - tgt_c) ** 2).mean())

        vg = mx.value_and_grad(loss_fn, argnums=(0, 1, 2))

        def fwd():
            r = rasterize(ndc, faces, attrs, opacity, **kw)
            return r.alpha, r.depth, r.attrs

        t_f = _time(fwd, iters)
        t_fb = _time(lambda: vg(ndc, attrs, opacity), iters)
        # full pipeline: scene parameters -> meshes -> projection -> raster -> loss, grads to params
        keys = ["centers", "sizes", "yaw", "rnd", "shape", "joint_aa", "root_pos", "eye", "fov", "colors",
                "person_present"]

        def full_loss(*vals):
            q = dict(p)
            q.update(dict(zip(keys, vals)))
            vw, fc, at, op, (cR, ey, fv) = build_scene(q, B)
            v = project(vw, cR, ey, fv, W / H)
            r = rasterize(v, fc, at, op, **kw)
            return (mx.abs(r.alpha - tgt_a).mean() + (r.alpha * mx.abs(r.depth - tgt_d)).mean()
                    + ((r.attrs - tgt_c) ** 2).mean())

        vg_full = mx.value_and_grad(full_loss, argnums=tuple(range(len(keys))))
        vals = [p[k] for k in keys]
        t_full = _time(lambda: vg_full(*vals), iters)
        vg_comp = mx.compile(vg_full)
        t_comp = _time(lambda: vg_comp(*vals), iters)
        _, grads = vg_comp(*vals)
        finite = all(bool(mx.all(mx.isfinite(g)).item()) for g in grads)
        row = {"B": B, "H": H, "W": W, "faces_per_job": F, "C": C, "entries": n_ent, "capacity": cap,
               "fwd_ms": round(t_f[0], 2), "fwd_min_ms": round(t_f[1], 2), "fwd_peak_GB": round(t_f[2], 3),
               "fwdbwd_ms": round(t_fb[0], 2), "fwdbwd_min_ms": round(t_fb[1], 2),
               "fwdbwd_peak_GB": round(t_fb[2], 3),
               "full_step_ms": round(t_full[0], 2), "full_step_min_ms": round(t_full[1], 2),
               "full_step_compiled_ms": round(t_comp[0], 2), "full_step_compiled_min_ms": round(t_comp[1], 2),
               "full_peak_GB": round(t_full[2], 3), "grads_finite": finite}
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


def stages(B=64, H=36, W=64, iters=10):
    """Per-kernel timings (ms) of one forward + backward on the realistic workload."""
    from . import raster as R
    mx.set_memory_limit(8 << 30)
    p = {k: mx.array(v) for k, v in make_params(B).items()}
    vw, faces, attrs, op, (cR, ey, fv) = build_scene(p, B)
    v = project(vw, cR, ey, fv, W / H)
    pix = mx.stack([(v[..., 0] + 1) * W / 2, (1 - v[..., 1]) * H / 2, v[..., 2], op], -1)
    mx.eval(pix, attrs)
    F, V = len(faces), v.shape[1]
    topo = R._topology(np.asarray(faces, np.int32), V)
    ntx, nty = -(-W // R.TILE_W), -(-H // R.TILE_H)
    NT, cap, C4 = ntx * nty, R.default_capacity(B, F, H, W), -(-attrs.shape[-1] // 4)
    K = 4 * C4 + 4
    params = mx.array([0.5, 20.0, R.cut_distance(0.5, 1e-3), 0.05, -1.0], mx.float32)
    tmpl = [("C4", C4), ("F", F), ("V", V), ("W", W), ("H", H), ("NTX", ntx), ("NT", NT), ("TW", R.TILE_W),
            ("TH", R.TILE_H), ("CULL", 1), ("AMODE", 0), ("NP", max(32, 1 << (3 * K - 1).bit_length()))]
    k = R._kernels(R.FAST_MATH)
    tg = (R._TGS, 1, 1)
    out = {}
    rects, counts, tri = k["setup"](inputs=[pix, topo.faces, params], template=tmpl, grid=(B * F, 1, 1),
                                    threadgroup=(256, 1, 1), output_shapes=[(B * F,), (B * F,), (B * F * 16,)],
                                    output_dtypes=[mx.uint32, mx.int32, mx.float32])
    out["setup"] = _time(lambda: k["setup"](inputs=[pix, topo.faces, params], template=tmpl, grid=(B * F, 1, 1),
                                            threadgroup=(256, 1, 1),
                                            output_shapes=[(B * F,), (B * F,), (B * F * 16,)],
                                            output_dtypes=[mx.uint32, mx.int32, mx.float32]), iters)[1]
    csum = mx.cumsum(counts)
    counts = mx.where(csum <= cap, counts, 0)
    rects = mx.where(csum <= cap, rects, mx.array(255, mx.uint32))
    offs = csum - counts
    nthr = (B * NT * 32, 1, 1)

    def cnt():
        return k["count"](inputs=[rects], template=tmpl, grid=nthr, threadgroup=tg, output_shapes=[(B * NT,)],
                          output_dtypes=[mx.int32])[0]
    out["bin_count"] = _time(cnt, iters)[1]
    tcount = cnt()
    tstart = mx.cumsum(tcount) - tcount

    def fill():
        return k["fill"](inputs=[rects, tstart], template=tmpl, grid=nthr, threadgroup=tg, output_shapes=[(cap,)],
                         output_dtypes=[mx.int32])[0]
    out["bin_fill"] = _time(fill, iters)[1]
    tlist = fill()

    def fwd():
        return k["fwd"](inputs=[tri, attrs, tlist, tstart, tcount, params], template=tmpl, grid=nthr, threadgroup=tg,
                        output_shapes=[(B, H, W), (B, H, W), (B, H, W, 4 * C4), (B, H, W, 2)],
                        output_dtypes=[mx.float32] * 4)
    out["forward"] = _time(fwd, iters)[1]
    _, de, ai, aux = fwd()
    g = [mx.ones((B, H, W)), mx.ones((B, H, W)), mx.ones((B, H, W, 4 * C4))]

    def bwd():
        return k["bwd"](inputs=[tri, attrs, tlist, tstart, tcount, rects, offs, params, g[0], g[1], g[2], de, ai, aux],
                        template=tmpl, grid=nthr, threadgroup=tg, output_shapes=[(cap * 3 * K,)],
                        output_dtypes=[mx.float32])[0]
    out["backward"] = _time(bwd, iters)[1]
    part = bwd()
    out["gather"] = _time(lambda: k["gather"](inputs=[part, offs, counts, topo.inc_ptr, topo.inc_idx], template=tmpl,
                                              grid=(B * V * (K // 4), 1, 1), threadgroup=(256, 1, 1),
                                              output_shapes=[(B, V, K)], output_dtypes=[mx.float32])[0], iters)[1]
    out = {kk: round(vv, 2) for kk, vv in out.items()}
    print(json.dumps({"H": H, "W": W, "stage_min_ms": out}), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer iterations")
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--stages", action="store_true", help="also print per-kernel timings")
    a = ap.parse_args()
    run(B=a.B, iters=4 if a.quick else 10)
    if a.stages:
        for hw in ((36, 64), (72, 128)):
            stages(a.B, *hw, iters=4 if a.quick else 10)
