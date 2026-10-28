"""Tests for the data pipeline (restage.data): DINOv2 port, packing, the batch loader, the cut detector.

    .venv/bin/python -m pytest tests/test_data.py -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from restage import scene as S
from restage.data import cuts as C
from restage.data import loader as LD
from restage.data import pack as P

ROOT = Path(__file__).resolve().parents[1]

CONTRACT = {   # key: (dtype, shape with K / S / L / E / T placeholders)
    "frames": (np.uint8, ("K", 126, 224, 3)), "key_frame": (np.int32, ("K",)), "key_shot": (np.int32, ("K",)),
    "fps": (np.int32, ()), "n_frames": (np.int32, ()), "shot_range": (np.int32, ("S", 2)),
    "shot_loc": (np.int32, ("S",)), "shot_size": (np.int32, ("S",)), "shot_move": (np.int32, ("S",)),
    "cam_rot": (np.float32, ("K", 3, 3)), "cam_pos": (np.float32, ("K", 3)), "cam_fov": (np.float32, ("K",)),
    "loc_points": (np.float32, ("L", 2048, 3)), "loc_sky": (np.float32, ("L", 3)), "loc_ground": (np.float32, ("L", 3)),
    "ent_class": (np.int32, ("E",)), "ent_loc": (np.int32, ("E",)), "ent_present": (np.uint8, ("E", "K")),
    "ent_visible": (np.uint8, ("E", "K")), "ent_pos": (np.float32, ("E", "K", 3)), "ent_rot": (np.float32, ("E", "K", 4)),
    "ent_pose": (np.float32, ("E", "K", 22, 4)), "ent_height": (np.float32, ("E",)), "ent_size": (np.float32, ("E", 3)),
    "ent_colors": (np.float32, ("E", 5, 3)), "depth": (np.float16, ("K", 36, 64)), "ids": (np.uint8, ("K", 36, 64)),
    "thumbs": (np.uint8, ("T", 18, 32, 3)),
}


def _gen(out, genre, seed, seconds):
    cmd = [sys.executable, "-m", "restage.gen", "--genre", genre, "--n", "1", "--seconds", str(seconds), "--res",
           "448x252", "--aux-res", "320x180", "--seed", str(seed), "--out", str(out), "--name", f"{genre}_{seed}"]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    return out / f"{genre}_{seed}"


@pytest.fixture(scope="module")
def samples(tmp_path_factory):
    """Generator samples (one indoor, one multi-location, one with many entities), packed."""
    d = tmp_path_factory.mktemp("gen")
    out = {}
    # seeds outside every dataset split's range (make_dataset.SPLIT_BASE), so the cut detector never saw them
    for genre, seed, secs in (("talking", 5000011, 7), ("hiking", 5000004, 12), ("driving", 5000002, 8),
                              ("interview", 5000004, 8)):
        base = _gen(d, genre, seed, secs)
        clip = P.pack(base, d / f"{genre}.npz")
        out[genre] = {"base": base, "clip": P.load_clip(d / f"{genre}.npz"), "raw": clip}
    return out


# ---------------------------------------------------------------- DINOv2

@pytest.fixture(scope="module")
def dino():
    from restage.data.dinov2 import DinoV2S
    return DinoV2S


def test_bicubic_matches_torch():
    torch = pytest.importorskip("torch")
    from restage.data.dinov2 import resample_pos_embed
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(1, 1 + 37 * 37, 8)).astype(np.float32)
    ref = torch.nn.functional.interpolate(torch.from_numpy(pos[:, 1:].reshape(1, 37, 37, 8)).permute(0, 3, 1, 2),
                                          size=(9, 16), mode="bicubic", align_corners=False)
    ref = ref.permute(0, 2, 3, 1).reshape(1, -1, 8).numpy()
    out = resample_pos_embed(pos, (9, 16))
    assert np.abs(out[:, 1:] - ref).max() < 1e-5
    assert np.array_equal(out[:, :1], pos[:, :1])


def test_dinov2_shapes_and_bf16(dino):
    import mlx.core as mx
    x = np.random.default_rng(0).integers(0, 256, (3, 126, 224, 3), dtype=np.uint8)
    for dt in (mx.float32, mx.bfloat16):
        m = dino.load(dtype=dt)
        c, p = m(x)
        mx.eval(c, p)
        assert c.shape == (3, 384) and p.shape == (3, 9, 16, 384)
        assert c.dtype == dt and p.dtype == dt
        assert np.isfinite(np.array(p.astype(mx.float32))).all()
    # float input in 0..1 gives the same result as u8
    m = dino.load()
    c1, _ = m(x)
    c2, _ = m(x.astype(np.float32) / 255)
    assert np.abs(np.array(c1) - np.array(c2)).max() < 3e-2          # TF32 matmuls amplify 1-ulp input changes


def test_dinov2_parity_with_transformers(dino, samples):
    pytest.importorskip("transformers")
    from restage.data.dinov2 import HF_DIR, parity
    if not (HF_DIR / "model.safetensors").exists():
        pytest.skip("HF checkpoint not downloaded")
    imgs = samples["talking"]["clip"]["frames"][::6][:6]
    r = parity(images=imgs)
    # fp32 matmuls run as TF32 on the M5 by default; bf16 is mixed precision (fp32 residual stream)
    assert r["fp32"]["cls_cos_min"] > 0.9999 and r["fp32"]["patch_cos_median"] > 0.9999
    assert r["fp32"]["patch_cos_p01"] > 0.999 and r["fp32"]["patch_cos_min"] > 0.97
    assert r["bf16"]["cls_cos_min"] > 0.999 and r["bf16"]["patch_cos_median"] > 0.999
    assert r["bf16"]["patch_cos_p01"] > 0.99 and r["bf16"]["patch_cos_min"] > 0.95
    assert r["fp32"]["patch_mean_abs"] < 0.02 and r["bf16"]["patch_mean_abs"] < 0.08


def test_dinov2_exact_parity_without_tf32():
    """With true fp32 matmuls (MLX_ENABLE_TF32=0) the port matches transformers to ~1e-4."""
    pytest.importorskip("transformers")
    from restage.data.dinov2 import HF_DIR
    if not (HF_DIR / "model.safetensors").exists():
        pytest.skip("HF checkpoint not downloaded")
    import os
    code = "import json; from restage.data.dinov2 import parity; print(json.dumps(parity(n=3)))"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True,
                         env={**os.environ, "MLX_ENABLE_TF32": "0"}).stdout
    r = json.loads(out.strip().splitlines()[-1])
    assert r["fp32"]["cls_max_abs"] < 2e-3 and r["fp32"]["patch_max_abs"] < 2e-3, r["fp32"]


# ---------------------------------------------------------------- packing

def test_select_keyframes():
    sr = np.array([[0, 45], [45, 46], [46, 120], [120, 127]])
    kf, ks = P.select_keyframes(sr, 30)
    for si, (s, e) in enumerate(sr):
        f = kf[ks == si]
        assert f[0] == s and f[-1] == e - 1 and np.all(np.diff(f) > 0) and np.all((f >= s) & (f < e))
        assert len(f) >= min(e - s, int(np.ceil((e - s) / 7.5)))
    assert np.all(np.diff(kf) > 0)
    f = kf[ks == 2]           # 74 frames: 0, 8, 15, 22, 30, ... every 7.5 frames + last
    assert list(f[:5] - 46) == [0, 8, 15, 22, 30] and f[-1] == 119


def test_downsample_labels_no_blur():
    K, h, w = 2, 180, 320
    depth = np.full((K, h, w), 10.0, np.float32)
    ids = np.zeros((K, h, w), np.int32)
    depth[:, :, :37] = 0                              # sky on the left
    ids[:, 50:120, 100:203] = 3                       # an entity at 2 m with an edge mid-block
    depth[:, 50:120, 100:203] = 2.0
    depth[:, 120:, :] = np.linspace(5, 6, w)          # a slanted floor
    d, i = P.downsample_labels(depth, ids)
    assert d.shape == (K, 36, 64) and i.shape == (K, 36, 64) and d.dtype == np.float16 and i.dtype == np.uint8
    # ids = the sample at the 5x5 block centre
    assert np.array_equal(i, ids[:, 2::5, 2::5])
    # every depth value is an input depth of the same class, never a blend across the edge
    for k in range(K):
        for y in range(36):
            for x in range(64):
                blk_d = depth[k, 5 * y:5 * y + 5, 5 * x:5 * x + 5]
                blk_i = ids[k, 5 * y:5 * y + 5, 5 * x:5 * x + 5]
                c = blk_i[2, 2]
                if c > 0:
                    assert np.isclose(float(d[k, y, x]), 2.0)
                elif depth[k, 5 * y + 2, 5 * x + 2] == 0:
                    assert d[k, y, x] == 0
                else:
                    vals = blk_d[(blk_i == 0) & (blk_d > 0)]
                    assert np.min(np.abs(vals - float(d[k, y, x]))) < 1e-2


def test_pack_contract(samples):
    for name, s in samples.items():
        c = s["raw"]
        K, S_, L, E, T = len(c["key_frame"]), len(c["shot_range"]), len(c["loc_points"]), len(c["ent_class"]), int(c["n_frames"])
        dims = {"K": K, "S": S_, "L": L, "E": E, "T": T}
        for k, (dt, shp) in CONTRACT.items():
            assert k in c, k
            a = np.asarray(c[k])
            assert a.dtype == dt, (name, k, a.dtype)
            assert a.shape == tuple(dims.get(x, x) for x in shp), (name, k, a.shape)
        meta = json.loads(str(c["meta"]))
        assert meta["genre"] == name and "seed" in meta and "producer" in meta
        assert np.all(c["key_shot"][1:] >= c["key_shot"][:-1])
        assert np.all(np.isin(c["ent_class"], [1, 2, 3, 4]))
        assert np.all(c["ent_visible"] <= c["ent_present"])
        assert np.allclose(np.linalg.det(c["cam_rot"]), 1, atol=1e-4)
        assert np.allclose(np.linalg.norm(c["ent_rot"], axis=-1), 1, atol=1e-4)
        assert c["ids"].max() <= E
        people = c["ent_class"] == 1
        assert np.all((c["ent_height"][people] > 1.4) & (c["ent_height"][people] < 2.1))
        assert np.all(c["ent_colors"] >= 0) and np.all(c["ent_colors"] <= 1)
        # loc_points lie on the set: inside the bounds, and the floor / ground is sampled
        doc = S.load(f"{s['base']}.restage")
        for li, loc in enumerate(doc["locations"]):
            lo, hi = np.array(loc["bounds"][0]), np.array(loc["bounds"][1])
            pts = c["loc_points"][li]
            assert np.all(pts >= lo - 0.5) and np.all(pts <= hi + 0.5)


def test_person_colours_from_regions(samples):
    """Region colours agree with the person's vertex colours: skin = the head / neck colour, etc."""
    c = samples["talking"]["raw"]
    doc = S.load(f"{samples['talking']['base']}.restage")
    meshes = {m["id"]: m for m in doc["meshes"]}
    for e, ent in enumerate(doc["entities"]):
        if ent["class"] != "person":
            continue
        m = meshes[ent["mesh"]]
        col = np.asarray(m["colors"])[:, :3]
        jv = np.asarray(m["joints"])[:, 0]
        for slot, joint in ((0, "neck"), (3, "l_toe"), (2, "pelvis")):
            vals = col[jv == S.JID[joint]]
            assert np.min(np.abs(vals - c["ent_colors"][e, slot]).max(1)) < 0.02, (slot, joint)


def test_camera_convention(samples):
    """Entity body centres projected with the stored camera land on that entity's pixels in `ids`; wrong
    conventions do not."""
    def looks_plus_z(pts, R, c, fov, *a):
        return P.project(pts, R @ np.diag([-1.0, 1, -1]), c, fov)

    def transposed(pts, R, c, fov, *a):
        return P.project(pts, R.T, c, fov)

    def flipped_v(pts, R, c, fov, *a):
        u, v, z = P.project(pts, R, c, fov)
        return u, P.LAB_H - v, z
    tot_h = tot_n = 0
    for name, s in samples.items():
        c = s["clip"]
        hits, n = P.projection_hits(c)
        assert n > 5, name
        assert hits / n >= 0.95, (name, hits, n)
        tot_h, tot_n = tot_h + hits, tot_n + n
        h0, n0 = P.projection_hits(c, project_fn=looks_plus_z)
        assert n0 == 0 or h0 / n0 < 0.5
    # summed over the samples, other conventions score much lower
    for fn in (transposed, flipped_v):
        h, n = 0, 0
        for s in samples.values():
            a, b = P.projection_hits(s["clip"], project_fn=fn)
            h, n = h + a, n + b
        assert n == 0 or h / n < 0.9 * tot_h / tot_n


def test_roots_on_ground_for_standing_people(samples):
    """A walking person's root is at the feet: it projects at the bottom of the entity's pixels."""
    c = samples["hiking"]["clip"]
    rows = []
    key_loc = c["shot_loc"][c["key_shot"]]
    for k in range(len(c["key_frame"])):
        for e in np.where(c["ent_visible"][:, k] & (c["ent_loc"] == key_loc[k]))[0]:
            m = c["ids"][k] == e + 1
            if m.sum() < 20:
                continue
            u, v, z = P.project(c["ent_pos"][e, k], c["cam_rot"][k], c["cam_pos"][k], c["cam_fov"][k])
            ys = np.where(m)[0]
            rows.append((v - ys.min()) / max(ys.max() + 1 - ys.min(), 1))
    assert len(rows) > 5
    assert np.median(rows) > 0.85


# ---------------------------------------------------------------- loader

def _sample(clip, keys=None, **kw):
    if keys is None:
        keys, _ = LD.window_from(clip["key_shot"], clip["shot_loc"], 0)
    return LD.make_sample(clip, keys, np.random.default_rng(kw.pop("seed", 0)), **kw)


def _batch_hits(s):
    cen = LD.entity_centres(s["ent_pos"].astype(float), s["ent_rot"].astype(float), s["ent_class"], s["ent_height"])
    key_loc = s["shot_loc"][s["key_shot"]]
    thick = np.where(s["ent_class"] == 1, 0.25 * s["ent_height"] / 1.75, 0.6 * s["ent_size"].max(1)) + 0.1
    hits = n = 0
    for k in np.where(s["key_mask"])[0]:
        u, v, z = P.project(cen[:, k], s["cam_rot"][k], s["cam_pos"][k], s["cam_fov"][k])
        for e in np.where(s["ent_mask"] & s["ent_visible"][:, k] & (s["ent_loc"] == key_loc[k]))[0]:
            if z[e] < 0.05 or (s["ids"][k] == e + 1).sum() < 3:
                continue
            x, y = int(np.floor(u[e])), int(np.floor(v[e]))
            if not (0 <= x < 64 and 0 <= y < 36):
                continue
            hit = bool((s["ids"][k, max(0, y - 1):y + 2, max(0, x - 1):x + 2] == e + 1).any())
            if not hit and 0 < s["depth"][k, y, x] < z[e] - thick[e]:
                continue
            hits, n = hits + hit, n + 1
    return hits, n


def test_gauge_fixing(samples):
    for name in ("hiking", "talking", "interview"):
        c = samples[name]["clip"]
        for s0 in range(len(c["shot_loc"])):
            keys, _ = LD.window_from(c["key_shot"], c["shot_loc"], s0)
            s = _sample(c, keys)
            n_loc = int(s["loc_mask"].sum())
            key_loc = s["shot_loc"][s["key_shot"]]
            assert s["is_anchor"].sum() == n_loc
            for l in range(n_loc):
                a = int(np.argmax(s["key_mask"] & (key_loc == l)))
                assert s["is_anchor"][a]
                R, p = s["cam_rot"][a], s["cam_pos"][a]
                fwd = -R[:, 2]
                assert abs(p[0]) < 1e-4 and abs(p[2]) < 1e-4
                assert abs(fwd[0]) < 1e-4 and fwd[2] < 0           # looks toward -Z
                orig = c["cam_rot"][keys[a]]
                assert abs(p[1] - c["cam_pos"][keys[a], 1]) < 1e-4        # height kept
                assert abs(fwd[1] - (-orig[:, 2])[1]) < 1e-5              # pitch kept
                assert abs(R[1, 0] - orig[1, 0]) < 1e-5                   # roll kept (camera x-axis height)
            # rigid per location: distances between cameras and entities are unchanged
            for l in range(n_loc):
                km = np.where(s["key_mask"] & (key_loc == l))[0]
                d_new = np.linalg.norm(s["cam_pos"][km][:, None] - s["cam_pos"][km][None], axis=-1)
                d_old = np.linalg.norm(c["cam_pos"][keys[km]][:, None] - c["cam_pos"][keys[km]][None], axis=-1)
                assert np.allclose(d_new, d_old, atol=1e-3)


def test_projection_after_crops_and_mirror(samples):
    for name in ("talking", "hiking", "driving", "interview"):
        c = samples[name]["clip"]
        for mirror_p in (0.0, 1.0):
            H = N = 0
            for seed in range(8):
                rng = np.random.default_rng(seed)
                keys, _ = LD.window_from(c["key_shot"], c["shot_loc"], int(rng.integers(len(c["shot_loc"]))), rng)
                s = _sample(c, keys, augment=True, p_crop=1.0, p_mirror=mirror_p, seed=seed, zoom=(1.15, 1.6))
                assert bool(s["mirrored"]) == (mirror_p == 1.0)
                assert np.all(s["shot_zoom"][s["shot_mask"]] > 1.1)
                h, n = _batch_hits(s)
                H, N = H + h, N + n
            assert N > 5 and H / N >= 0.93, (name, mirror_p, H, N)


def test_crop_updates_fov_and_depth(samples):
    c = samples["talking"]["clip"]
    keys, _ = LD.window_from(c["key_shot"], c["shot_loc"], 0)
    plain = _sample(c, keys)
    s = _sample(c, keys, augment=True, p_crop=1.0, p_mirror=0.0, jitter=0.0)
    K = int(plain["key_mask"].sum())
    z = s["shot_zoom"][s["key_shot"][:K]]
    t0 = np.tan(np.radians(plain["cam_fov"][:K]) / 2)
    t1 = np.tan(np.radians(s["cam_fov"][:K]) / 2)
    assert np.allclose(t0 / t1, z, rtol=1e-4)


def test_crop_depth_is_exact():
    """After a rotated zoom, every labelled pixel back-projects (new camera, new depth) to a point whose view depth
    in the old camera equals the old depth of its source pixel, and which projects into that source pixel."""
    rng = np.random.default_rng(3)
    K = 1
    yy, xx = np.mgrid[0:36, 0:64]
    s = {"frames": rng.integers(0, 255, (K, 126, 224, 3), dtype=np.uint8),
         "ids": ((xx // 8 + yy // 6) % 3).astype(np.uint8)[None], "depth": (2 + xx * 0.1 + yy * 0.05)[None].astype(np.float32),
         "cam_rot": S.quat_to_mat(S.quat_euler(-0.2, 0.8, 0.05))[None], "cam_pos": np.array([[1.0, 1.6, -2.0]]),
         "cam_fov": np.array([45.0]), "ent_class": np.zeros(0, int), "ent_pos": np.zeros((0, K, 3)),
         "ent_rot": np.zeros((0, K, 4)), "ent_height": np.zeros(0), "ent_visible": np.zeros((0, K), bool)}
    old = {k: np.array(v, copy=True) for k, v in s.items()}
    z, R_rel = LD.sample_crop(np.random.default_rng(7), 45.0, zmin=1.5, zmax=1.6)
    assert not np.allclose(R_rel, np.eye(3))
    LD.apply_crop(s, [0], z, R_rel)
    R1, c1, f1 = s["cam_rot"][0], s["cam_pos"][0], s["cam_fov"][0]
    F1 = 18 / np.tan(np.radians(f1) / 2)
    d = np.stack([(xx + 0.5 - 32) / F1, -(yy + 0.5 - 18) / F1, -np.ones_like(xx, float)], -1)
    X = c1 + (d * s["depth"][0][..., None]) @ R1.T                  # world points of the new pixels
    u, v, zo = P.project(X, old["cam_rot"][0], old["cam_pos"][0], 45.0)
    xi, yi = np.floor(u).astype(int), np.floor(v).astype(int)
    assert xi.min() >= 0 and xi.max() < 64 and yi.min() >= 0 and yi.max() < 36
    assert np.allclose(zo, old["depth"][0][yi, xi], rtol=1e-4)    # same surface depth
    assert np.array_equal(s["ids"][0], old["ids"][0][yi, xi])       # ids follow the same source pixels


def test_pure_zoom_crop_is_centred_resample():
    rng = np.random.default_rng(0)
    K = 1
    s = {"frames": rng.integers(0, 255, (K, 126, 224, 3), dtype=np.uint8),
         "ids": np.zeros((K, 36, 64), np.uint8), "depth": np.full((K, 36, 64), 4.0, np.float32),
         "cam_rot": np.eye(3)[None].copy(), "cam_pos": np.zeros((K, 3)), "cam_fov": np.array([40.0]),
         "ent_class": np.zeros(0, int), "ent_pos": np.zeros((0, K, 3)), "ent_rot": np.zeros((0, K, 4)),
         "ent_height": np.zeros(0), "ent_visible": np.zeros((0, K), bool)}
    s["ids"][0, 16:20, 30:34] = 1
    LD.apply_crop(s, [0], 2.0, np.eye(3))
    assert np.isclose(s["cam_fov"][0], np.degrees(2 * np.arctan(np.tan(np.radians(20)) / 2)))
    assert np.allclose(s["depth"], 4.0)                        # no rotation: view depth unchanged
    ys, xs = np.where(s["ids"][0] == 1)                        # the 4x4 blob doubles around the centre
    assert ys.min() >= 13 and ys.max() <= 22 and xs.min() >= 27 and xs.max() <= 37 and len(ys) >= 40


def test_mirror_pose_matches_mirrored_skeleton():
    """FK of the mirrored pose == mirror image of the FK joints with left / right swapped."""
    rng = np.random.default_rng(1)
    T = 4
    pose = S.quat_euler(rng.normal(0, 0.5, (T, 22)), rng.normal(0, 0.5, (T, 22)), rng.normal(0, 0.5, (T, 22)))
    root_q = S.quat_euler(rng.normal(0, 0.3, T), rng.normal(0, 2, T), rng.normal(0, 0.3, T))
    root_p = rng.normal(0, 2, (T, 3))
    jp, _ = S.forward_kinematics(pose, root_p, root_q)
    s = {"frames": np.zeros((T, 2, 2, 3), np.uint8), "depth": np.zeros((T, 2, 2)), "ids": np.zeros((T, 2, 2)),
         "cam_rot": np.tile(np.eye(3), (T, 1, 1)), "cam_pos": np.zeros((T, 3)), "loc_points": np.zeros((1, 4, 3)),
         "ent_pos": root_p[None].copy(), "ent_rot": root_q[None].copy(), "ent_pose": pose[None].copy(),
         "ent_class": np.array([1])}
    LD.mirror(s)
    jm, _ = S.forward_kinematics(s["ent_pose"][0], s["ent_pos"][0], s["ent_rot"][0])
    expect = jp[:, LD.MIRROR_PERM] * np.array([-1, 1, 1])
    assert np.abs(jm - expect).max() < 1e-6
    # camera: the mirrored camera sees the mirrored point at the flipped pixel
    R = S.quat_to_mat(S.quat_euler(0.2, 0.7, 0.1))
    cpos = np.array([0.3, 1.5, 2.0])
    X = np.array([0.5, 1.2, -1.0])
    u, v, _ = P.project(X, R, cpos, 50.0)
    M = np.diag([-1.0, 1, 1])
    u2, v2, _ = P.project(M @ X, M @ R @ M, M @ cpos, 50.0)
    assert np.isclose(u2, P.LAB_W - u) and np.isclose(v2, v)


def test_masks_consistent(samples):
    for name, sm in samples.items():
        c = sm["clip"]
        for seed in range(4):
            rng = np.random.default_rng(seed)
            keys, _ = LD.window_from(c["key_shot"], c["shot_loc"], int(rng.integers(len(c["shot_loc"]))), rng)
            s = _sample(c, keys, augment=True, seed=seed, emax=4 if name == "driving" else LD.EMAX)
            K, Sn, L, E = int(s["key_mask"].sum()), int(s["shot_mask"].sum()), int(s["loc_mask"].sum()), int(s["ent_mask"].sum())
            for k, n in (("key_mask", K), ("shot_mask", Sn), ("loc_mask", L), ("ent_mask", E)):
                assert np.all(s[k][:n]) and not np.any(s[k][n:]), k             # valid entries come first
            assert K == len(keys) and K <= LD.KMAX and Sn <= LD.SMAX and L <= LD.LMAX
            assert np.all(np.diff(s["key_shot"][:K]) >= 0) and s["key_shot"][K - 1] == Sn - 1
            assert np.all(s["key_time"][:K] >= 0) and np.all(np.diff(s["key_time"][:K]) > 0)
            assert set(s["shot_loc"][:Sn]) == set(range(L))
            assert np.all(s["ent_loc"][:E] < L)
            assert not s["frames"][K:].any() and not s["depth"][K:].any() and not s["ids"][K:].any()
            assert not s["is_anchor"][K:].any()
            assert not s["ent_present"][E:].any() and not s["ent_present"][:, K:].any()
            assert np.all(s["ent_visible"] <= s["ent_present"])
            assert np.all(s["ent_present"][:E].any(1))                         # kept entities appear in the window
            assert np.all(np.isin(s["ent_class"][:E], [1, 2, 3, 4])) and not s["ent_class"][E:].any()
            u = np.unique(s["ids"])
            assert np.all((u <= E) | (u == LD.IGNORE_ID)), u
            assert np.allclose(np.linalg.det(s["cam_rot"]), 1, atol=1e-4)
            assert np.allclose(np.linalg.norm(s["ent_rot"], axis=-1), 1, atol=1e-4)
            assert np.allclose(np.linalg.norm(s["ent_pose"], axis=-1), 1, atol=1e-4)
            if name == "driving":
                assert E == 4                                                    # overflow: the 4 most visible kept


def test_epoch_windows_cover_clip(samples):
    for sm in samples.values():
        c = sm["clip"]
        ws = LD.epoch_windows(c["key_shot"], c["shot_loc"])
        allk = np.concatenate(ws)
        assert np.array_equal(allk, np.arange(len(c["key_frame"])))
        for w in ws:
            assert len(w) <= LD.KMAX


def test_loader_prefetch(samples, tmp_path):
    files = [tmp_path / f"{n}.npz" for n in samples]
    for n, f in zip(samples, files):
        np.savez(f, **{k: v for k, v in samples[n]["raw"].items()})
    with LD.Loader(files, batch_size=3, augment=True, workers=2, mode="process") as dl:
        b1, b2 = next(dl), next(dl)
    for b in (b1, b2):
        assert b["frames"].shape == (3, LD.KMAX, 126, 224, 3) and b["ent_pose"].shape == (3, LD.EMAX, LD.KMAX, 22, 4)
        assert b["loc_points"].shape == (3, LD.LMAX, 2048, 3) and b["shot_loc"].shape == (3, LD.SMAX)
    dl = LD.Loader(files, batch_size=4, augment=False, epoch=True, workers=2, mode="thread")
    n_win = sum(len(LD.epoch_windows(samples[n]["clip"]["key_shot"], samples[n]["clip"]["shot_loc"])) for n in samples)
    batches = list(dl)
    assert len(batches) == len(dl) == -(-n_win // 4)
    assert sum(int(b["batch_mask"].sum()) for b in batches) == n_win


# ---------------------------------------------------------------- cut detector

def test_cut_features_and_nms():
    rng = np.random.default_rng(0)
    a = np.repeat(rng.integers(0, 255, (1, 18, 32, 3), dtype=np.uint8), 20, 0)
    b = np.repeat(rng.integers(0, 255, (1, 18, 32, 3), dtype=np.uint8), 20, 0)
    f = C.frame_features(np.concatenate([a, b]))
    assert f.shape == (40, C.N_FEAT)
    assert np.argmax(f[:, 2]) == 20 and np.argmax(f[:, 0]) == 20 and f[20, 0] > 0.02 and f[19, 2] == 0
    x = C.window_inputs(f)
    assert x.shape == (40, (2 * C.WIN + 1) * 2 * C.N_FEAT)
    p = np.zeros(30)
    p[[5, 6, 7, 20]] = [0.6, 0.9, 0.7, 0.8]
    assert C.pick_cuts(p, 0.5, 4) == [6, 20]
    assert C.shots_from_cuts([6, 20], 30) == [[0, 6], [6, 20], [20, 30]]


def test_detect_cuts_on_video(samples):
    if not (C.CKPT / "cut_mlp.json").exists():
        pytest.skip("cut detector not trained (python -m restage.data.cuts train)")
    det = C.CutDetector.load()
    tp = fp = fn = 0
    for name, sm in samples.items():
        shots = C.detect_cuts(f"{sm['base']}.mp4", det)
        T = int(sm["clip"]["n_frames"])
        assert shots[0][0] == 0 and shots[-1][1] == T and all(a[1] == b[0] for a, b in zip(shots, shots[1:]))
        pred = [s for s, _ in shots[1:]]
        true = [int(s) for s, _ in sm["clip"]["shot_range"][1:]]
        ign = C.ignore_mask(sm["clip"]["meta"].get("cam_jumps", []), T)
        r = C.f1_score([p for p in pred if not (ign[p] and p not in true)], true, 1)
        tp, fp, fn = tp + r["tp"], fp + r["fp"], fn + r["fn"]
        # frames and thumbnails give the same answer as the video path
        assert C.detect_cuts(sm["clip"]["thumbs"], det) == shots
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    assert f1 > 0.8, (tp, fp, fn)
