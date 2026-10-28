"""Tests for restage.render: FK, cameras, shapes, export and the soft rasteriser.

Run: .venv/bin/python -m pytest tests/test_render.py -q
"""
import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")      # exact float32 matmuls for the reference maths

import math  # noqa: E402

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from restage import scene as SC  # noqa: E402
from restage.render import export as E  # noqa: E402
from restage.render import fk  # noqa: E402
from restage.render import shapes as S  # noqa: E402
from restage.render.camera import look_at, project  # noqa: E402
from restage.render.raster import rasterize, rasterize_reference  # noqa: E402

mx.set_memory_limit(8 << 30)


def _rand_quats(rng, shape):
    q = rng.normal(size=shape + (4,))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _np(a):
    return np.array(a)


# ----------------------------------------------------------------------------- rotations and FK

def test_quaternion_helpers_match_numpy():
    rng = np.random.default_rng(0)
    a, b = _rand_quats(rng, (50,)), _rand_quats(rng, (50,))
    v = rng.normal(size=(50, 3))
    A, Bq, Vv = (mx.array(x.astype(np.float32)) for x in (a, b, v))
    np.testing.assert_allclose(_np(fk.quat_mul(A, Bq)), SC.quat_mul(a, b), atol=1e-6)
    np.testing.assert_allclose(_np(fk.quat_rotate(A, Vv)), SC.quat_rotate(a, v), atol=1e-5)
    np.testing.assert_allclose(_np(fk.quat_to_mat(A)), SC.quat_to_mat(a), atol=1e-6)
    np.testing.assert_allclose(_np(fk.quat_normalize(A * 3.0)), a, atol=1e-6)
    m = SC.quat_to_mat(a)
    q2 = _np(fk.mat_to_quat(mx.array(m.astype(np.float32))))
    np.testing.assert_allclose(q2, SC.mat_to_quat(m), atol=1e-5)          # same branch choice as numpy
    np.testing.assert_allclose(SC.quat_to_mat(q2), m, atol=1e-5)


def test_rot6d_roundtrip_and_orthonormal():
    rng = np.random.default_rng(1)
    x = mx.array(rng.normal(size=(100, 6)).astype(np.float32))
    R = fk.rot6d_to_mat(x)
    Rn = _np(R)
    np.testing.assert_allclose(np.einsum("nji,njk->nik", Rn, Rn), np.broadcast_to(np.eye(3), Rn.shape), atol=1e-5)
    np.testing.assert_allclose(np.linalg.det(Rn), 1.0, atol=1e-5)
    np.testing.assert_allclose(_np(fk.rot6d_to_mat(fk.mat_to_rot6d(R))), Rn, atol=1e-5)
    aa = rng.normal(size=(20, 3))
    Ra = _np(fk.axis_angle_to_mat(mx.array(aa.astype(np.float32))))
    ang = np.linalg.norm(aa, axis=-1)
    np.testing.assert_allclose(SC.quat_to_mat(SC.quat_axis_angle(aa, ang)), Ra, atol=1e-5)
    np.testing.assert_allclose(_np(fk.yaw_matrix(mx.array([0.7]))), SC.quat_to_mat(SC.quat_euler(y=0.7))[None],
                               atol=1e-6)


@pytest.mark.parametrize("batched_offsets", [False, True])
def test_forward_kinematics_matches_scene(batched_offsets):
    rng = np.random.default_rng(2)
    T = 16
    pose = _rand_quats(rng, (T, 22))
    root_q = _rand_quats(rng, (T,))
    root_p = rng.normal(size=(T, 3))
    if batched_offsets:
        offs = SC.OFFSETS[None] * rng.uniform(0.8, 1.2, (T, 1, 1))
        ref = [SC.forward_kinematics(pose[t:t + 1], root_p[t:t + 1], root_q[t:t + 1], offs[t]) for t in range(T)]
        ref_p = np.concatenate([r[0] for r in ref]), np.concatenate([r[1] for r in ref])
        off_in = mx.array(offs.astype(np.float32))
    else:
        ref_p = SC.forward_kinematics(pose, root_p, root_q)
        off_in = None
    Rw, pw = fk.forward_kinematics(mx.array(SC.quat_to_mat(pose).astype(np.float32)),
                                   mx.array(SC.quat_to_mat(root_q).astype(np.float32)),
                                   mx.array(root_p.astype(np.float32)), off_in)
    np.testing.assert_allclose(_np(pw), ref_p[0], atol=2e-5)
    np.testing.assert_allclose(_np(Rw), SC.quat_to_mat(ref_p[1]), atol=2e-5)


# ----------------------------------------------------------------------------- camera

def test_projection_conventions_and_gen_parity():
    from restage.gen.cinema import project as gen_project
    rng = np.random.default_rng(3)
    B, N = 6, 40
    eye = rng.uniform(-3, 3, (B, 3)).astype(np.float32)
    tgt = rng.uniform(-1, 1, (B, 3)).astype(np.float32)
    R = look_at(mx.array(eye), mx.array(tgt))
    fov = rng.uniform(30, 70, B).astype(np.float32)
    pts = rng.uniform(-4, 4, (B, N, 3)).astype(np.float32)
    out = _np(project(mx.array(pts), R, mx.array(eye), mx.array(fov), 16 / 9))
    q = SC.mat_to_quat(_np(R))
    for b in range(B):
        x, y, z = gen_project(eye[b], q[b], pts[b].astype(float), fov[b], 16 / 9)
        ok = z > 1e-3
        np.testing.assert_allclose(out[b, ok, 0], x[ok], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(out[b, ok, 1], y[ok], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(out[b, :, 2], z, atol=1e-4)
    # the target projects to the image centre at positive depth; +Y world is up in the image
    o = _np(project(mx.array(tgt[:, None]), R, mx.array(eye), mx.array(fov), 16 / 9))
    np.testing.assert_allclose(o[:, 0, :2], 0, atol=1e-5)
    assert (o[:, 0, 2] > 0).all()
    cam = mx.array([[0.0, 1.0, 5.0]])
    Rc = look_at(cam, mx.array([[0.0, 1.0, 0.0]]))
    up = _np(project(mx.array([[[0.0, 2.0, 0.0], [1.0, 1.0, 0.0]]]), Rc, cam, mx.array([50.0]), 1.0))
    assert up[0, 0, 1] > 0 and abs(up[0, 0, 0]) < 1e-6         # above the axis -> +y in NDC
    assert up[0, 1, 0] > 0                                      # +X world is to the right (camera looks down -Z)


# ----------------------------------------------------------------------------- shapes

def _closed_and_outward(pos, faces):
    from collections import Counter
    cnt = Counter()
    for a, b, c in faces:
        for e in ((a, b), (b, c), (c, a)):
            cnt[(int(e[0]), int(e[1]))] += 1
    bad = sum(1 for (a, b), k in cnt.items() if k != 1 or cnt.get((b, a), 0) != 1)
    p = pos[faces]
    vol = np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6
    return bad == 0, vol


def test_rounded_boxes_topology_and_shape():
    B, N = 2, 3
    c = mx.array(np.array([[[0, 0.5, 0], [2, 1, 0], [-2, 1, 1]]] * B, np.float32))
    s = mx.array(np.array([[[1, 1, 1], [2, 1, 0.5], [0.4, 2, 0.4]]] * B, np.float32))
    yaw = mx.array(np.array([[0.0, 0.3, -1.0]] * B, np.float32))
    for r in (0.0, 0.5, 1.0):
        v, f, meta = S.rounded_boxes(c, s, yaw, mx.full((B, N), r))
        assert v.shape == (B, N * 56, 3) and f.shape == (N * 108, 3)
        assert meta["n_vert_per_box"] == 56 and meta["n_face_per_box"] == 108
        assert f.dtype == np.int32 and f.min() == 0 and f.max() == N * 56 - 1
        vn = _np(v)
        for k in range(N):
            closed, vol = _closed_and_outward(vn[0], f[k * 108:(k + 1) * 108])
            assert closed and vol > 0
            box_vol = np.prod(_np(s)[0, k])
            if r == 0.0:
                assert abs(vol - box_vol) / box_vol < 2e-3           # sharp box
            elif r == 1.0:
                # inscribed ellipsoid: every vertex lies on it
                R = SC.quat_to_mat(SC.quat_euler(y=float(_np(yaw)[0, k])))
                loc = (vn[0, k * 56:(k + 1) * 56] - _np(c)[0, k]) @ R / (_np(s)[0, k] / 2)
                np.testing.assert_allclose(np.linalg.norm(loc, axis=-1), 1.0, atol=1e-5)
                assert vol < box_vol * math.pi / 6 * 1.0001
    # gradients are finite at both ends of the roundness range
    g = mx.grad(lambda rr: S.rounded_boxes(c, s, yaw, rr)[0].sum())(mx.array([[0.0, 1.0, 0.5]] * B))
    assert np.isfinite(_np(g)).all()


def test_ground_plane():
    v, f, _ = S.ground_plane(mx.array([[1.0, 1.5, -2.0]]), extent=10.0, n=8)
    vn = _np(v)[0]
    assert vn.shape == (81, 3) and f.shape == (128, 3)
    n = np.cross(vn[f[:, 1]] - vn[f[:, 0]], vn[f[:, 2]] - vn[f[:, 0]])
    assert (n[:, 1] > 0).all()                                  # faces up
    np.testing.assert_allclose(vn[:, 1], 0.0)
    np.testing.assert_allclose([vn[:, 0].min(), vn[:, 0].max()], [-9.0, 11.0], atol=1e-5)


def test_humanoid_template():
    t = S.HUMANOID_TEMPLATE
    F, V = len(t["faces"]), len(t["joint"])
    assert F <= 700 and F == 672 and V == 393
    assert set(np.unique(t["region"])) == {0, 1, 2, 3, 4}
    assert set(np.unique(t["joint"])) == set(range(22)) - {SC.JID["l_clavicle"], SC.JID["r_clavicle"]}
    for h in (0.9, 1.0, 1.1):
        pos, faces, offs = S.humanoid_rest_numpy([h, 1, 1, 1])
        assert abs(pos[:, 1].max() - 1.78 * h) < 0.02 * h               # standing height
        assert abs(pos[:, 1].min()) < 1e-5                            # soles on the ground
        assert pos[:, 2].max() > 0.15 and pos[:, 2].min() > -0.15     # feet point to +Z
    for pi, name in enumerate(t["part_names"]):
        if name in ("head", "hair"):
            continue
        vi = np.nonzero(t["part"] == pi)[0]
        fs = faces[np.isin(faces, vi).all(1)]
        closed, vol = _closed_and_outward(pos, fs)
        assert closed and vol > 0, name
    hv = np.isin(np.arange(V), np.nonzero(np.isin(t["part"], [t["part_names"].index("head"),
                                                                t["part_names"].index("hair")]))[0])
    fs = faces[hv[faces].all(1)]
    assert _closed_and_outward(pos, fs)[1] > 0
    # shape parameters do what they say
    base, _, _ = S.humanoid_rest_numpy([1, 1, 1, 1])
    wide, _, offs_w = S.humanoid_rest_numpy([1, 1, 1, 1.2])
    big_head, _, _ = S.humanoid_rest_numpy([1, 1, 1.2, 1])
    fat, _, _ = S.humanoid_rest_numpy([1, 1.3, 1, 1])
    assert wide[:, 0].max() > base[:, 0].max() + 0.02
    assert big_head[:, 1].max() > base[:, 1].max() + 0.02
    thigh = t["part"] == t["part_names"].index("l_thigh")
    assert np.ptp(fat[thigh, 0]) > 1.2 * np.ptp(base[thigh, 0])


def _walk_pose(T):
    J = SC.JID
    out = []
    for t in range(T):
        q = np.tile(np.array([0, 0, 0, 1.0]), (22, 1))
        a = 0.5 * math.sin(t)
        q[J["l_shoulder"]] = SC.quat_euler(z=-1.2)
        q[J["r_shoulder"]] = SC.quat_euler(z=1.2)
        q[J["l_hip"]], q[J["r_hip"]] = SC.quat_euler(x=-a), SC.quat_euler(x=a)
        q[J["l_knee"]] = SC.quat_euler(x=0.4)
        q[J["l_elbow"]] = SC.quat_euler(y=0.5)
        q[J["neck"]] = SC.quat_euler(y=0.3)
        q[J["pelvis"]] = SC.quat_euler(y=0.1 * t)
        out.append(q)
    return np.array(out)


def test_humanoid_export_roundtrip_and_png(tmp_path):
    """Posed MLX humanoid == the format's LBS of the exported rest mesh; write a .restage + PNGs."""
    T = 3
    rng = np.random.default_rng(4)
    pose_q = _walk_pose(T)
    jr = SC.quat_to_mat(pose_q).astype(np.float32)
    rr = SC.quat_to_mat(SC.quat_euler(y=np.linspace(0, 0.6, T))).astype(np.float32)
    rp = np.stack([np.linspace(-0.5, 0.5, T), np.zeros(T), np.zeros(T)], -1).astype(np.float32)
    shape = np.array([1.05, 1.1, 0.95, 1.08], np.float32)
    v, f, meta = S.humanoid(mx.array(np.tile(shape, (T, 1))), mx.array(jr), mx.array(rr), mx.array(rp))
    v = _np(v)
    cols = np.array([[0.8, 0.6, 0.45], [0.7, 0.1, 0.1], [0.1, 0.15, 0.4], [0.9, 0.9, 0.9], [0.1, 0.06, 0.03]],
                    np.float32)
    mesh, skel = E.humanoid_to_numpy(shape, cols, "person0")
    assert mesh["joints"].shape == (len(v[0]), 4) and (mesh["weights"][:, 0] == 1).all()
    track = E.pose_track(jr, rr, rp)
    track["state"] = np.full(T, 2, np.uint8)
    ent = {"id": "e0", "name": "p", "class": "person", "location": "loc0", "mesh": "person0",
           "skeleton": skel["id"], "track": track}
    eye = np.array([[2.5, 1.6, 3.5]], np.float32)
    cR = _np(look_at(mx.array(eye), mx.array([[0, 1.0, 0]])))
    cam = E.camera_track(np.repeat(cR, T, 0), np.repeat(eye, T, 0), np.full(T, 45.0))
    bv, bf, bm = S.rounded_boxes(mx.array([[[-1.5, 0.4, -1.5], [1.5, 0.5, -1.0], [0, 3, 0]]]),
                                 mx.array([[[1.0, 0.8, 0.6], [0.8, 1.0, 0.8], [1, 1, 1]]]),
                                 mx.array([[0.3, -0.4, 0.0]]), mx.array([[0.2, 0.8, 0.0]]))
    boxes = E.rounded_boxes_to_numpy(_np(bv[0]), bf, bm, rng.uniform(0.2, 0.8, (3, 3)), [True, True, False])
    assert len(boxes["indices"]) == 216 and len(boxes["positions"]) == 112
    ground = E.ground_to_numpy(10, 4)
    doc = E.scene_doc([ground, boxes, mesh], [{"id": "ground", "mesh": "ground"}, {"id": "boxes", "mesh": "set_boxes"}],
                      [ent], [cam], T, skeletons=[skel])
    path = tmp_path / "render_export.restage"
    SC.save(path, doc)
    d2 = SC.load(path)
    m2 = d2["meshes"][2]
    sk = d2["skeletons"][0]
    tr = d2["entities"][0]["track"]
    off = np.array([j["offset"] for j in sk["joints"]])
    par = np.array([j["parent"] for j in sk["joints"]])
    jp, jq = SC.forward_kinematics(tr["pose"].astype(float), tr["position"].astype(float),
                                   tr["rotation"].astype(float), off, par)
    rest = np.zeros((22, 3))
    for j in range(22):
        rest[j] = (rest[par[j]] if par[j] >= 0 else 0) + off[j]
    J = m2["joints"][:, 0].astype(int)
    lbs = np.einsum("tnij,nj->tni", SC.quat_to_mat(jq)[:, J], m2["positions"] - rest[J]) + jp[:, J]
    np.testing.assert_allclose(lbs, v, atol=1e-5)
    # PNG through the soft rasteriser (front view, close-up) and, if moderngl works, the offline renderer
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    front = np.array([[0.0, 1.0, 3.2]], np.float32)
    fR = _np(look_at(mx.array(front), mx.array([[0, 0.95, 0]])))[0]
    pv, pf, _ = S.humanoid(mx.array(shape[None]), mx.array(SC.quat_to_mat(pose_q[:1]).astype(np.float32)),
                           mx.array(np.eye(3, dtype=np.float32)[None]), mx.zeros((1, 3)))
    img = E.preview_image(_np(pv[0]), pf, E.region_colors(meta["region"], cols), fR, front[0], 40.0, 288, 512)
    plt.imsave(tmp_path / "humanoid_preview.png", img)
    # it looks like a person: regions stack head > torso > legs > shoes in the image, and it is tall
    r = rasterize(project(pv, mx.array(fR[None]), mx.array(front), mx.array([40.0]), 512 / 288), pf,
                  mx.array(np.eye(5, dtype=np.float32)[meta["region"]][None]), height=288, width=512, sigma=0.3)
    a, reg = _np(r.alpha[0]), _np(r.attrs[0]).argmax(-1)
    rows = np.nonzero(a.max(1) > 0.5)[0]
    cols_ = np.nonzero(a.max(0) > 0.5)[0]
    assert np.ptp(rows) > 1.0 * np.ptp(cols_)                   # taller than wide (arms down)
    ymean = [np.nonzero((reg == k) & (a > 0.5))[0].mean() for k in range(5)]
    assert ymean[4] < ymean[1] < ymean[2] < ymean[3]            # hair above top above bottom above shoes
    try:
        from restage.gen.render import render_png
        render_png(d2, 1, str(tmp_path / "scene_gen.png"), 320, 180)
    except Exception as e:  # noqa: BLE001 - optional: needs an OpenGL context
        pytest.skip(f"offline renderer unavailable: {e}")
    assert (tmp_path / "scene_gen.png").exists()


# ----------------------------------------------------------------------------- rasteriser

def _random_tris(rng, B, F, z=(1.0, 3.0), spread=0.25, C=3):
    V = 3 * F
    c = rng.uniform(-0.9, 0.9, (B, F, 1, 2))
    v = c + rng.normal(0, spread, (B, F, 3, 2))
    zz = rng.uniform(*z, (B, F, 1)) + rng.uniform(-0.2, 0.2, (B, F, 3))
    verts = np.concatenate([v, zz[..., None]], -1).reshape(B, V, 3).astype(np.float32)
    faces = np.arange(V).reshape(F, 3)
    attrs = rng.normal(0, 1, (B, V, C)).astype(np.float32)
    op = rng.uniform(0.3, 1.0, (B, V)).astype(np.float32)
    return verts, faces, attrs, op


def _weights(rng, B, H, W, C):
    return [mx.array(rng.normal(0, 1, s).astype(np.float32)) for s in [(B, H, W), (B, H, W), (B, H, W, C)]]


def _loss(fn, gw, faces, kw):
    def f(vv, aa, oo):
        o = fn(vv, faces, aa, oo, **kw)
        al, de, at = (o.alpha, o.depth, o.attrs) if hasattr(o, "alpha") else o
        return (al * gw[0]).sum() + (de * gw[1]).sum() + (at * gw[2]).sum()
    return f


def _rel(a, b):
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


@pytest.mark.parametrize("alpha_mode,cull,fast,C", [("sum", False, True, 3), ("prob", False, True, 5),
                                                    ("sum", True, False, 8), ("prob", True, False, 1)])
def test_raster_matches_dense_reference(alpha_mode, cull, fast, C):
    rng = np.random.default_rng(5)
    B, F, H, W = 2, 40, 20, 28
    verts, faces, attrs, op = _random_tris(rng, B, F, C=C)
    kw = dict(height=H, width=W, sigma=0.8, gamma=0.3, cull_backfaces=cull, alpha_mode=alpha_mode)
    args = [mx.array(verts), mx.array(attrs), mx.array(op)]
    out = rasterize(args[0], faces, args[1], args[2], fast_math=fast, **kw)
    ref = rasterize_reference(args[0], faces, args[1], args[2], **kw)
    assert int(out.n_entries.item()) <= out.capacity
    for a, b in zip((out.alpha, out.depth, out.attrs), ref):
        assert _rel(_np(a), _np(b)) < 1e-4
    gw = _weights(rng, B, H, W, C)
    gk = mx.grad(_loss(lambda *a, **k: rasterize(*a, fast_math=fast, **k), gw, faces, kw), argnums=(0, 1, 2))(*args)
    gr = mx.grad(_loss(rasterize_reference, gw, faces, kw), argnums=(0, 1, 2))(*args)
    for a, b in zip(gk, gr):
        a, b = _np(a), _np(b)
        assert np.isfinite(a).all()
        assert _rel(a, b) < 1e-3
    # per component of the vertex gradient as well (x, y, depth)
    for k in range(3):
        assert _rel(_np(gk[0])[..., k], _np(gr[0])[..., k]) < 1e-3


def test_raster_shared_mesh_matches_reference():
    """A shared-vertex mesh (box + person), real projection, occlusion and culling."""
    rng = np.random.default_rng(6)
    B, H, W = 2, 18, 32
    bv, bf, bm = S.rounded_boxes(mx.array(rng.uniform([-1, 0.3, -1], [1, 0.8, 1], (B, 2, 3)).astype(np.float32)),
                                 mx.array(rng.uniform(0.4, 1.0, (B, 2, 3)).astype(np.float32)),
                                 mx.array(rng.uniform(-1, 1, (B, 2)).astype(np.float32)), mx.full((B, 2), 0.4), n=2)
    hv, hf, hm = S.humanoid(mx.ones((B, 4)), mx.broadcast_to(mx.eye(3), (B, 22, 3, 3)),
                            mx.broadcast_to(mx.eye(3), (B, 3, 3)), mx.array([[0, 0, 0.8], [0.3, 0, -0.5]]))
    verts, faces, _ = S.merge_meshes((bv, bf), (hv, hf))
    eye = mx.array([[0.5, 1.4, 4.0], [-2.0, 1.0, 3.0]])
    ndc = project(verts, look_at(eye, mx.array([[0, 0.8, 0]] * B)), eye, mx.array([45.0, 55.0]), W / H)
    V = verts.shape[1]
    attrs = mx.array(rng.uniform(0, 1, (B, V, 4)).astype(np.float32))
    op = mx.array(rng.uniform(0.5, 1, (B, V)).astype(np.float32))
    kw = dict(height=H, width=W, sigma=0.6, gamma=0.1)
    out = rasterize(ndc, faces, attrs, op, **kw)
    ref = rasterize_reference(ndc, faces, attrs, op, **kw)
    for a, b in zip((out.alpha, out.depth, out.attrs), ref):
        assert _rel(_np(a), _np(b)) < 1e-4
    assert _np(out.alpha).max() > 0.99
    gw = _weights(rng, B, H, W, 4)
    gk = mx.grad(_loss(rasterize, gw, faces, kw), argnums=(0, 1, 2))(ndc, attrs, op)
    gr = mx.grad(_loss(rasterize_reference, gw, faces, kw), argnums=(0, 1, 2))(ndc, attrs, op)
    for a, b in zip(gk, gr):
        assert _rel(_np(a), _np(b)) < 1e-3


def test_raster_finite_differences():
    """Analytic gradients against central differences on a tiny case (safe math)."""
    rng = np.random.default_rng(7)
    B, F, H, W, C = 1, 3, 12, 16, 2
    verts, faces, attrs, op = _random_tris(rng, B, F, spread=0.35, C=C)
    kw = dict(height=H, width=W, sigma=1.0, gamma=0.5, cull_backfaces=False, fast_math=False, cutoff=1e-6)
    gw = _weights(rng, B, H, W, C)
    f = _loss(rasterize, gw, faces, kw)
    args = [mx.array(verts), mx.array(attrs), mx.array(op)]
    grads = [_np(g) for g in mx.grad(f, argnums=(0, 1, 2))(*args)]
    h = {0: 2e-3, 1: 1e-2, 2: 1e-2}
    for ai, x in enumerate([verts, attrs, op]):
        flat = x.reshape(-1)
        idx = rng.choice(flat.size, size=min(12, flat.size), replace=False)
        for i in idx:
            xp, xm = flat.copy(), flat.copy()
            xp[i] += h[ai]
            xm[i] -= h[ai]
            a_p, a_m = list(args), list(args)
            a_p[ai], a_m[ai] = mx.array(xp.reshape(x.shape)), mx.array(xm.reshape(x.shape))
            fd = (f(*a_p).item() - f(*a_m).item()) / (2 * h[ai])
            an = grads[ai].reshape(-1)[i]
            assert abs(fd - an) <= 2e-2 * max(1.0, abs(an)), (ai, i, fd, an)


def test_raster_near_plane_no_nans():
    """Triangles crossing / behind the near plane are dropped with finite values and gradients."""
    rng = np.random.default_rng(8)
    B = 2
    eye = mx.array([[0.0, 1.0, 0.0]] * B)
    R = look_at(eye, mx.array([[0.0, 1.0, -1.0]] * B))
    # a ground grid around and under the camera, a quad crossing the camera plane, one exactly at z = 0
    gv, gf, _ = S.ground_plane(eye, extent=5.0, n=6, power=1.0)
    quad = mx.array(np.array([[[-1, 0.5, 1], [1, 0.5, 1], [1, 1.5, -2], [-1, 1.5, -2], [0.2, 1.0, 0.0],
                               [0.3, 1.1, 0.0], [0.1, 1.2, -1.0]]] * B, np.float32))
    qf = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]])
    verts, faces, _ = S.merge_meshes((gv, gf), (quad, qf))
    attrs = mx.array(rng.uniform(0, 1, (B, verts.shape[1], 3)).astype(np.float32))

    def loss(vw, at):
        v = project(vw, R, eye, mx.array([60.0] * B), 16 / 9)
        r = rasterize(v, faces, at, height=36, width=64, cull_backfaces=False)
        return r.alpha.sum() + r.depth.sum() + r.attrs.sum(), r

    (val, r), (g_v, g_a) = mx.value_and_grad(lambda a, b: loss(a, b), argnums=(0, 1))(verts, attrs)
    for x in (val, r.alpha, r.depth, r.attrs, g_v, g_a):
        assert np.isfinite(_np(x)).all()
    assert _np(r.alpha).max() > 0.9                 # the visible ground still renders
    assert np.abs(_np(g_v)).max() > 0


def test_raster_determinism():
    rng = np.random.default_rng(9)
    B, F, H, W, C = 8, 600, 36, 64, 6
    verts, faces, attrs, op = _random_tris(rng, B, F, spread=0.08, C=C)
    gw = _weights(rng, B, H, W, C)
    f = mx.value_and_grad(_loss(rasterize, gw, faces, dict(height=H, width=W)), argnums=(0, 1, 2))
    args = [mx.array(verts), mx.array(attrs), mx.array(op)]
    r1 = [_np(x) for x in rasterize(args[0], faces, args[1], args[2], height=H, width=W)[:3]]
    v1, g1 = f(*args)
    for _ in range(2):
        r2 = [_np(x) for x in rasterize(args[0], faces, args[1], args[2], height=H, width=W)[:3]]
        v2, g2 = f(*args)
        for a, b in zip(r1, r2):
            assert np.array_equal(a, b)
        assert v1.item() == v2.item()
        for a, b in zip(g1, g2):
            assert np.array_equal(_np(a), _np(b))


def test_raster_capacity_overflow_and_compile():
    rng = np.random.default_rng(10)
    B, F, H, W, C = 2, 200, 36, 64, 4
    verts, faces, attrs, op = _random_tris(rng, B, F, spread=0.2, C=C)
    args = [mx.array(verts), mx.array(attrs), mx.array(op)]
    full = rasterize(args[0], faces, args[1], args[2], height=H, width=W)
    small = rasterize(args[0], faces, args[1], args[2], height=H, width=W, max_entries=100)
    assert int(small.n_entries.item()) == int(full.n_entries.item()) > 100 == small.capacity
    assert np.isfinite(_np(small.alpha)).all() and _np(small.alpha).sum() < _np(full.alpha).sum()
    gw = _weights(rng, B, H, W, C)
    f = _loss(rasterize, gw, faces, dict(height=H, width=W))
    g_eager = mx.grad(f, argnums=(0, 1, 2))(*args)
    g_comp = mx.compile(mx.grad(f, argnums=(0, 1, 2)))(*args)
    for a, b in zip(g_eager, g_comp):
        np.testing.assert_allclose(_np(a), _np(b), rtol=1e-5, atol=1e-6)


def test_opacity_zero_fades_and_gets_gradient():
    """An absent (opacity 0) occluder neither shows nor occludes but receives a fade-in gradient."""
    back = np.array([[-0.8, -0.8, 4.0], [0.8, -0.8, 4.0], [0.0, 0.8, 4.0]], np.float32)
    front = back.copy()
    front[:, 2] = 2.0
    front[:, :2] *= 0.5
    verts = mx.array(np.concatenate([back, front])[None])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    attrs = mx.array(np.array([[1, 0]] * 3 + [[0, 1]] * 3, np.float32)[None])

    def run(o):
        return rasterize(verts, faces, attrs, o, height=24, width=24, sigma=0.5)

    op = mx.array(np.array([[1, 1, 1, 0, 0, 0]], np.float32))
    r = run(op)
    np.testing.assert_allclose(_np(r.attrs)[0, 12, 12], [1, 0], atol=1e-6)      # the back triangle shows
    np.testing.assert_allclose(_np(r.depth)[0, 12, 12], 4.0, atol=1e-4)
    g = mx.grad(lambda o: run(o).attrs[0, 12, 12, 1])(op)
    assert _np(g)[0, 3:].sum() > 0                                               # raising opacity shows it


def test_full_pipeline_gradients_compile():
    """Scene parameters -> meshes -> projection -> raster -> loss: finite grads, eager == compiled,
    and bitwise reproducible."""
    from restage.render.bench_render import build_scene, make_params
    B, H, W = 3, 36, 64
    p = {k: mx.array(v) for k, v in make_params(B, seed=3).items()}
    keys = ["centers", "sizes", "yaw", "rnd", "shape", "joint_aa", "root_pos", "eye", "target", "fov", "colors",
            "person_present"]
    tgt = mx.array(np.random.default_rng(0).uniform(1, 5, (B, H, W)).astype(np.float32))

    def loss(*vals):
        q = dict(p)
        q.update(dict(zip(keys, vals)))
        vw, fc, at, op, (cR, ey, fv) = build_scene(q, B)
        r = rasterize(project(vw, cR, ey, fv, W / H), fc, at, op, height=H, width=W)
        return (r.alpha * mx.abs(r.depth - tgt)).mean() + r.attrs.mean() + r.alpha.mean()

    vals = [p[k] for k in keys]
    f = mx.value_and_grad(loss, argnums=tuple(range(len(keys))))
    v1, g1 = f(*vals)
    v2, g2 = mx.compile(f)(*vals)
    assert abs(v1.item() - v2.item()) < 1e-5
    for k, a, b in zip(keys, g1, g2):
        a, b = _np(a), _np(b)
        assert np.isfinite(a).all(), k
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6, err_msg=k)
    assert all(np.abs(_np(g)).max() > 0 for g in g1)
    _, g3 = f(*vals)                                   # bitwise reproducible end to end
    for a, b in zip(g1, g3):
        assert np.array_equal(_np(a), _np(b))
