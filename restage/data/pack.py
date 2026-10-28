"""Pack one generator sample (.restage + .mp4 + _aux.npz) into a training clip (docs/TRAINING_DATA.md).

    python -m restage.data.pack data/scratch/talking_00003 data/train/talking_00003.npz

* Video: decoded once with ffmpeg at 30 fps, 224x126 (area filter); keyframes are taken from it and
  every frame is reduced to a 32x18 thumbnail (cv2 INTER_AREA) for the cut detector.
* Keyframes: 4 per second inside each shot plus every shot's first and last frame.
* Cameras: from the scene's camera tracks (camera-to-world rotation, position, vertical fov).
* depth / ids at 64x36 from the aux file (rendered at an integer multiple, normally 320x180 = 5x):
  ids are point-sampled at the 64x36 pixel centre (nearest; set props 1000+k -> 0); depth is the
  lower median of the block's pixels that belong to the same class as the centre sample (sky,
  set, or that entity), so it is an actual surface depth and never blurs across an object edge.
* loc_points: 2048 area-weighted surface samples of the location's props (ground included; the
  `sky`/`cloud` classes and distant scenery reaching > 20% beyond the location bounds excluded),
  restricted to the filmed region: x/z within `REGION_RADIUS` of the location's camera and entity
  positions (whole indoor sets fit inside it).
* Entities: height, rest bbox, colours per region (people: skin, top, bottom, shoes, hair from vertex
  colours grouped by skinning joint and height; objects: dominant colour in slot 0).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from .. import scene as S

FFMPEG = "/opt/homebrew/bin/ffmpeg"
KEY_W, KEY_H = 224, 126
THUMB_W, THUMB_H = 32, 18
LAB_W, LAB_H = 64, 36
N_POINTS = 2048
KEY_RATE = 4                       # keyframes per second inside a shot
REGION_RADIUS = 30.0               # metres around the cameras / entities of a location used for loc_points
SKIP_CLASSES = {"sky", "cloud"}     # backdrop classes not sampled into loc_points
GROUND_CLASSES = {"floor", "ground", "terrain", "road", "sidewalk", "voxel_chunk", "stage", "grass", "water",
                  "room", "crosswalk", "markings", "rug"}

SHOT_SIZES = ["extreme_wide", "wide", "medium", "closeup", "extreme_closeup", "insert"]
SHOT_MOVES = ["static", "pan", "tilt", "dolly", "truck", "orbit", "handheld", "crane", "drone", "fpv"]
ENT_CLASSES = {"person": 1, "animal": 2, "vehicle": 3, "object": 4}
J = S.JID
# person colour regions: skinning joints
REGIONS = {"skin": ("neck", "l_wrist", "r_wrist"), "top": ("spine1", "spine2", "chest"),
           "bottom": ("pelvis", "l_hip", "r_hip"), "shoes": ("l_ankle", "r_ankle", "l_toe", "r_toe"),
           "hair": ("head",)}


class BadSample(ValueError):
    """The generator output cannot be packed (non-finite tracks, missing files, ...)."""


# ---------------------------------------------------------------- video

def decode_video(path, width=KEY_W, height=KEY_H, fps=30) -> np.ndarray:
    """All frames of a video as u8 [T, height, width, 3] RGB (ffmpeg, area downscale)."""
    cmd = [FFMPEG, "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={width}:{height}:flags=area",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(out) // (width * height * 3)
    return np.frombuffer(out, np.uint8, n * width * height * 3).reshape(n, height, width, 3)


def thumbnails(frames: np.ndarray) -> np.ndarray:
    """[T, 126, 224, 3] -> [T, 18, 32, 3] (area)."""
    return np.stack([cv2.resize(f, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA) for f in frames])


def select_keyframes(shot_range, fps=30, rate=KEY_RATE):
    """Video frame indices and shot indices of keyframes: `rate` per second inside each shot plus each
    shot's first and last frame."""
    kf, ks = [], []
    step = fps / rate
    for si, (s, e) in enumerate(shot_range):
        fr = {int(s + round(k * step)) for k in range(int(np.ceil((e - s) / step)) + 1)}
        fr = sorted(f for f in fr if s <= f < e) + [int(e - 1)]
        fr = sorted(set(fr))
        kf += fr
        ks += [si] * len(fr)
    return np.array(kf, np.int32), np.array(ks, np.int32)


# ---------------------------------------------------------------- labels

def downsample_labels(depth: np.ndarray, ids: np.ndarray, out_hw=(LAB_H, LAB_W)):
    """depth f [K, h, w] (0 = sky), ids int [K, h, w] (0 set / sky, e + 1 entity) -> (depth f16, ids u8) at out_hw.

    ids: nearest sample at the output pixel centre. depth: lower median of the pixels in the output pixel's
    footprint that have the same class as the centre sample (classes: sky, set, entity e)."""
    K, h, w = ids.shape
    oh, ow = out_hw
    fy, fx = h / oh, w / ow
    if abs(fy - round(fy)) > 1e-6 or abs(fx - round(fx)) > 1e-6:
        # non-integer factor: plain nearest sampling at pixel centres
        yy = np.clip(((np.arange(oh) + 0.5) * fy).astype(int), 0, h - 1)
        xx = np.clip(((np.arange(ow) + 0.5) * fx).astype(int), 0, w - 1)
        return depth[:, yy][:, :, xx].astype(np.float16), ids[:, yy][:, :, xx].astype(np.uint8)
    fy, fx = int(round(fy)), int(round(fx))
    d = depth.astype(np.float32).reshape(K, oh, fy, ow, fx).transpose(0, 1, 3, 2, 4).reshape(K, oh, ow, fy * fx)
    i = ids.astype(np.int32).reshape(K, oh, fy, ow, fx).transpose(0, 1, 3, 2, 4).reshape(K, oh, ow, fy * fx)
    cls = np.where(i > 0, i, np.where(d > 0, 0, -1))           # -1 sky, 0 set, e+1 entity
    centre = (fy // 2) * fx + fx // 2                            # exact centre for odd factors
    c = cls[..., centre]
    same = cls == c[..., None]
    dd = np.where(same, d, np.inf)
    dd.sort(-1)
    n = same.sum(-1)
    med = np.take_along_axis(dd, ((n - 1) // 2)[..., None], -1)[..., 0]
    out_d = np.where(c < 0, 0.0, med)
    out_i = np.maximum(c, 0)
    if out_i.max() > 255:
        raise BadSample("more than 254 entities")
    return out_d.astype(np.float16), out_i.astype(np.uint8)


def project(points, cam_rot, cam_pos, fov_deg, width=LAB_W, height=LAB_H):
    """World points [..., 3] -> pixel (u, v) and view depth z (> 0 in front), with camera-to-world rotation
    cam_rot [3, 3] (camera looks down -Z, +Y up) and vertical fov. Pixel (0, 0) is the top-left corner."""
    pc = (np.asarray(points, np.float64) - cam_pos) @ np.asarray(cam_rot, np.float64)    # = R^T (p - c)
    z = -pc[..., 2]
    f = (height / 2) / np.tan(np.radians(fov_deg) / 2)
    zz = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = width / 2 + f * pc[..., 0] / zz
    v = height / 2 - f * pc[..., 1] / zz
    return u, v, z


# ---------------------------------------------------------------- geometry helpers

def _mesh_world(mesh, pos=(0, 0, 0), rot=(0, 0, 0, 1), scale=(1, 1, 1)):
    v = np.asarray(mesh["positions"], np.float64) * np.asarray(scale, np.float64)
    q = np.broadcast_to(np.asarray(rot, np.float64), (len(v), 4))
    return S.quat_rotate(q, v) + np.asarray(pos, np.float64)


def _faces(mesh):
    return np.asarray(mesh["indices"]).astype(np.int64).reshape(-1, 3)


def _face_colors(mesh, mat):
    if "colors" in mesh:
        c = np.asarray(mesh["colors"])[:, :3].astype(np.float64)
        if np.asarray(mesh["colors"]).dtype == np.uint8:
            c = c / 255
    else:
        c = np.ones((len(mesh["positions"]), 3))
    c = c * np.asarray((mat or {}).get("color", [1, 1, 1]), np.float64)
    return c[_faces(mesh)].mean(1)


def _tri_area_normal(v, f):
    cr = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    a = np.linalg.norm(cr, axis=1)
    return 0.5 * a, cr / np.maximum(a, 1e-12)[:, None]


def _srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def dominant_color(colors, areas, levels=16):
    """Area-weighted dominant colour: group faces by quantised sRGB colour, return the mean (linear) colour of
    the group with the largest area."""
    colors = np.asarray(colors, np.float64)
    areas = np.asarray(areas, np.float64)
    if len(colors) == 0 or areas.sum() <= 0:
        return None
    q = np.minimum((_srgb(colors) * levels).astype(np.int64), levels - 1)
    key = (q[:, 0] * levels + q[:, 1]) * levels + q[:, 2]
    tot = np.bincount(key, weights=areas)
    best = int(np.argmax(tot))
    sel = key == best
    return (colors[sel] * areas[sel, None]).sum(0) / areas[sel].sum()


def person_appearance(mesh, mat):
    """(colours [5, 3] skin, top, bottom, shoes, hair; height = rest top of the skull (skin-coloured head))."""
    v = np.asarray(mesh["positions"], np.float64)
    f = _faces(mesh)
    col = _face_colors(mesh, mat)
    area, nrm = _tri_area_normal(v, f)
    jv = np.asarray(mesh["joints"]).astype(np.int64)
    wv = np.asarray(mesh["weights"], np.float64)
    vj = jv[np.arange(len(jv)), wv.argmax(1)]          # dominant joint per vertex
    fj = vj[f[:, 0]]
    cy = v[f].mean(1)[:, 1]
    out = np.zeros((5, 3))
    sel = np.isin(fj, [J[n] for n in REGIONS["skin"]])
    skin = dominant_color(col[sel], area[sel])
    if skin is None:
        skin = dominant_color(col, area)
    out[0] = skin
    sel = np.isin(fj, [J[n] for n in REGIONS["top"]])
    front = sel & (nrm[:, 2] > 0.3)                    # the front of the torso (a backpack hangs at the back)
    c = dominant_color(col[front], area[front]) if front.any() else None
    out[1] = c if c is not None else (dominant_color(col[sel], area[sel]) if sel.any() else skin)
    for k, name in ((2, "bottom"), (3, "shoes")):
        sel = np.isin(fj, [J[n] for n in REGIONS[name]])
        c = dominant_color(col[sel], area[sel]) if sel.any() else None
        out[k] = c if c is not None else skin
    head = fj == J["head"]
    if head.any():
        y0, y1 = cy[head].min(), cy[head].max()
        top = head & (cy >= y1 - 0.12 * (y1 - y0))
        c = dominant_color(col[top], area[top])
        out[4] = c if c is not None else skin
        hv = np.zeros(len(v), bool)
        hv[f[head].ravel()] = True
        sk = hv & (np.abs(np.asarray(mesh["colors"])[:, :3] - skin).max(1) < 0.02)
        height = float(v[sk, 1].max()) if sk.any() else float(v[hv, 1].max())
    else:
        out[4] = skin
        height = float(v[:, 1].max())
    return out, height


def object_appearance(mesh, mat):
    v = np.asarray(mesh["positions"], np.float64)
    f = _faces(mesh)
    area, _ = _tri_area_normal(v, f)
    out = np.zeros((5, 3))
    c = dominant_color(_face_colors(mesh, mat), area)
    if c is not None:
        out[0] = c
    return out


def sample_surface(tris, n, rng, keep=None, max_rounds=6):
    """Area-weighted samples on triangles [F, 3, 3]; `keep(points) -> bool mask` rejects samples outside a region."""
    a, _ = _tri_area_normal(tris.reshape(-1, 3), np.arange(len(tris) * 3).reshape(-1, 3))
    if a.sum() <= 0:
        return None
    p = a / a.sum()
    got = []
    cnt = 0
    m = max(4 * n, 8192)
    for _ in range(max_rounds):
        idx = rng.choice(len(tris), m, p=p)
        r1, r2 = rng.random(m), rng.random(m)
        s = np.sqrt(r1)
        w0, w1, w2 = 1 - s, s * (1 - r2), s * r2
        t = tris[idx]
        pts = t[:, 0] * w0[:, None] + t[:, 1] * w1[:, None] + t[:, 2] * w2[:, None]
        if keep is not None:
            pts = pts[keep(pts)]
        got.append(pts)
        cnt += len(pts)
        if cnt >= n:
            break
        m *= 4
    pts = np.concatenate(got)
    if len(pts) == 0:
        return None
    sel = rng.choice(len(pts), n, replace=len(pts) < n)
    return pts[sel]


def camera_jumps(doc, rel_fov=0.06, ratio=2.5, min_move=0.3, ratio_move=5.0):
    """Frames t (inside a shot, t > start) where the camera turns or moves much more between t-1 and t than around
    it: generator framing jumps that look like cuts in the video but are not edits (see GEN_REQUESTS.md #3).
    A turn counts when its excess over the local median (+-5 frames) exceeds `rel_fov` of the vertical fov and
    it is > `ratio` x that median; a move when it exceeds `min_move` metres and `ratio_move` x the local median."""
    out = []
    cams = {c["shot"]: c for c in doc["cameras"]}
    for s in doc["shots"]:
        tr = cams[s["id"]]["track"]
        q = np.asarray(tr["rotation"], np.float64)
        p = np.asarray(tr["position"], np.float64)
        fov = np.asarray(tr["fov"], np.float64)
        if len(q) < 3:
            continue
        R = S.quat_to_mat(q / np.linalg.norm(q, axis=1, keepdims=True))
        f = -R[:, :, 2]
        ang = np.degrees(np.arccos(np.clip((f[1:] * f[:-1]).sum(1), -1, 1)))
        dp = np.linalg.norm(np.diff(p, axis=0), axis=1)
        for i in range(len(ang)):
            nb = np.concatenate([ang[max(0, i - 5):i], ang[i + 1:i + 6]])
            nbp = np.concatenate([dp[max(0, i - 5):i], dp[i + 1:i + 6]])
            med = float(np.median(nb)) if len(nb) else 0.0
            medp = float(np.median(nbp)) if len(nbp) else 0.0
            turn = ang[i] - med > rel_fov * fov[i + 1] and ang[i] > ratio * med
            move = dp[i] > min_move and dp[i] > ratio_move * medp
            if turn or move:
                out.append(int(s["start"] + i + 1))
    return out


# ---------------------------------------------------------------- packing

def _paths(base):
    base = Path(base)
    if base.suffix in (".restage", ".mp4", ".npz"):
        base = base.with_suffix("")
    stem = base.name[:-4] if base.name.endswith("_aux") else base.name
    d = base.parent
    return d / f"{stem}.restage", d / f"{stem}.mp4", d / f"{stem}_aux.npz", stem


def pack(base, out_path=None, seed=None, verbose=False) -> dict:
    """Pack `<base>.restage`, `<base>.mp4`, `<base>_aux.npz`. Returns the clip dict (and writes it to out_path)."""
    t0 = time.time()
    rs_path, mp4_path, aux_path, stem = _paths(base)
    for p in (rs_path, mp4_path, aux_path):
        if not p.exists():
            raise BadSample(f"missing {p}")
    doc = S.load(rs_path)
    fps, T = int(doc["fps"]), int(doc["n_frames"])
    seed = int(doc.get("meta", {}).get("seed", 0)) if seed is None else seed
    rng = np.random.default_rng(seed + 12345)

    # ---- shots, keyframes
    shots = doc["shots"]
    loc_ids = [l["id"] for l in doc["locations"]]
    shot_range = np.array([[s["start"], s["end"]] for s in shots], np.int32)
    if shot_range[0, 0] != 0 or shot_range[-1, 1] != T or np.any(shot_range[1:, 0] != shot_range[:-1, 1]):
        raise BadSample("shots do not tile the video")
    shot_loc = np.array([loc_ids.index(s["location"]) for s in shots], np.int32)
    shot_size = np.array([SHOT_SIZES.index(s["size"]) if s["size"] in SHOT_SIZES else -1 for s in shots], np.int32)
    shot_move = np.array([SHOT_MOVES.index(s["move"]) if s["move"] in SHOT_MOVES else -1 for s in shots], np.int32)
    key_frame, key_shot = select_keyframes(shot_range, fps)
    K = len(key_frame)

    # ---- cameras at keyframes
    cams = {c["shot"]: c for c in doc["cameras"]}
    cam_rot = np.zeros((K, 3, 3), np.float32)
    cam_pos = np.zeros((K, 3), np.float32)
    cam_fov = np.zeros(K, np.float32)
    for k, (f, si) in enumerate(zip(key_frame, key_shot)):
        tr = cams[shots[si]["id"]]["track"]
        i = f - shot_range[si, 0]
        q = np.asarray(tr["rotation"][i], np.float64)
        cam_rot[k] = S.quat_to_mat(q / np.linalg.norm(q))
        cam_pos[k] = tr["position"][i]
        cam_fov[k] = tr["fov"][i]
    if not (np.isfinite(cam_rot).all() and np.isfinite(cam_pos).all() and np.isfinite(cam_fov).all()):
        bad = sorted({shots[si]["id"] for si in key_shot[~np.isfinite(cam_pos).all(1) | ~np.isfinite(cam_fov)]})
        raise BadSample(f"non-finite camera track in shots {bad}")
    if not np.all((cam_fov > 1) & (cam_fov < 170)):
        raise BadSample("camera fov out of range")

    # ---- video
    frames_all = decode_video(mp4_path, fps=fps)
    if abs(len(frames_all) - T) > 2:
        raise BadSample(f"video has {len(frames_all)} frames, scene has {T}")
    if len(frames_all) < T:
        frames_all = np.concatenate([frames_all, np.repeat(frames_all[-1:], T - len(frames_all), 0)])
    frames_all = frames_all[:T]
    thumbs = thumbnails(frames_all)
    frames = np.ascontiguousarray(frames_all[key_frame])
    del frames_all

    # ---- aux labels
    with np.load(aux_path) as aux:
        afr = aux["frames"] if "frames" in aux.files else np.arange(T)
        pos_of = -np.ones(T, np.int64)
        pos_of[afr] = np.arange(len(afr))
        sel = pos_of[key_frame]
        if (sel < 0).any():
            raise BadSample("aux file misses keyframes")
        ids_hr = aux["ids"][sel].astype(np.int32)
        depth_hr = aux["depth"][sel].astype(np.float32)
    E = len(doc["entities"])
    ids_hr = np.where((ids_hr >= 1) & (ids_hr <= E), ids_hr, 0)            # props (1000 + k) -> set
    depth, ids = downsample_labels(depth_hr, ids_hr)
    del ids_hr, depth_hr

    # ---- locations
    meshes = {m["id"]: m for m in doc["meshes"]}
    mats = {m["id"]: m for m in doc["materials"]}
    L = len(doc["locations"])
    loc_points = np.zeros((L, N_POINTS, 3), np.float32)
    loc_sky = np.zeros((L, 3), np.float32)
    loc_ground = np.zeros((L, 3), np.float32)
    ents = doc["entities"]
    for li, loc in enumerate(doc["locations"]):
        env = loc.get("environment", {})
        loc_sky[li] = env.get("sky", [0.6, 0.7, 0.9])
        # filmed region: cameras of this location's shots + its entities while it is on screen
        on = np.zeros(T, bool)
        xs = []
        for si, s in enumerate(shots):
            if shot_loc[si] == li:
                on[s["start"]:s["end"]] = True
                xs.append(np.asarray(cams[s["id"]]["track"]["position"], np.float64))
        for e in ents:
            if e["location"] == loc["id"]:
                st = np.asarray(e["track"]["state"])
                m = on & (st > 0)
                if m.any():
                    xs.append(np.asarray(e["track"]["position"], np.float64)[m])
        lo_b, hi_b = np.asarray(loc["bounds"][0], float), np.asarray(loc["bounds"][1], float)
        if xs:
            xs = np.concatenate(xs)
            xs = xs[np.isfinite(xs).all(1)]
        if len(xs):
            rlo = np.maximum(xs.min(0) - REGION_RADIUS, lo_b - 1e-3)
            rhi = np.minimum(xs.max(0) + REGION_RADIUS, hi_b + 1e-3)
        else:
            rlo, rhi = lo_b, hi_b

        def keep(p, rlo=rlo, rhi=rhi):
            return (p[:, 0] >= rlo[0]) & (p[:, 0] <= rhi[0]) & (p[:, 2] >= rlo[2]) & (p[:, 2] <= rhi[2])
        tris, gcol, garea = [], [], []
        for pr in loc.get("props", []):
            if pr.get("class") in SKIP_CLASSES:
                continue
            mesh = meshes[pr["mesh"]]
            vw = _mesh_world(mesh, pr.get("position", (0, 0, 0)), pr.get("rotation", (0, 0, 0, 1)),
                             pr.get("scale", (1, 1, 1)))
            ext = 0.2 * (hi_b - lo_b)
            if np.any(vw.min(0)[[0, 2]] < (lo_b - ext)[[0, 2]]) or np.any(vw.max(0)[[0, 2]] > (hi_b + ext)[[0, 2]]):
                continue                                    # distant scenery (mountain ring) beyond the set
            f = _faces(mesh)
            t = vw[f]
            bb_lo, bb_hi = t.min(1), t.max(1)
            inside = (bb_hi[:, 0] >= rlo[0]) & (bb_lo[:, 0] <= rhi[0]) & (bb_hi[:, 2] >= rlo[2]) & (bb_lo[:, 2] <= rhi[2])
            tris.append(t[inside])
            if pr.get("class") in GROUND_CLASSES:
                a, n = _tri_area_normal(vw, f)
                c = _face_colors(mesh, mats.get(mesh.get("material")))
                cen = t.mean(1)
                g = inside & (np.abs(n[:, 1]) > 0.8) & keep(cen)
                if pr.get("class") == "room":
                    g &= cen[:, 1] < lo_b[1] + 0.3            # floor of the room shell, not the ceiling
                gcol.append(c[g])
                garea.append(a[g])
        tris = np.concatenate(tris) if tris else np.zeros((0, 3, 3))
        pts = sample_surface(tris, N_POINTS, rng, keep) if len(tris) else None
        if pts is None:
            raise BadSample(f"location {loc['id']} has no surface in its filmed region")
        loc_points[li] = pts
        gc = dominant_color(np.concatenate(gcol), np.concatenate(garea)) if gcol and sum(len(x) for x in gcol) else None
        loc_ground[li] = gc if gc is not None else env.get("ground", [0.35, 0.3, 0.25])

    # ---- entities
    skel = {s["id"]: s for s in doc.get("skeletons", [])}
    ent_class = np.zeros(E, np.int32)
    ent_loc = np.zeros(E, np.int32)
    ent_ident = np.zeros(E, np.int32)
    ent_present = np.zeros((E, K), np.uint8)
    ent_visible = np.zeros((E, K), np.uint8)
    ent_pos = np.zeros((E, K, 3), np.float32)
    ent_rot = np.zeros((E, K, 4), np.float32)
    ent_rot[..., 3] = 1
    ent_pose = np.zeros((E, K, 22, 4), np.float32)
    ent_pose[..., 3] = 1
    ent_height = np.zeros(E, np.float32)
    ent_size = np.zeros((E, 3), np.float32)
    ent_colors = np.zeros((E, 5, 3), np.float32)
    idents = []
    for ei, e in enumerate(ents):
        cls = e.get("class", "object")
        ent_class[ei] = ENT_CLASSES.get(cls, 4)
        ent_loc[ei] = loc_ids.index(e["location"])
        ident = e.get("identity", e["id"])
        if ident not in idents:
            idents.append(ident)
        ent_ident[ei] = idents.index(ident)
        tr = e["track"]
        st = np.asarray(tr["state"])[key_frame]
        ent_present[ei] = st > 0
        ent_visible[ei] = st == 2
        p = np.asarray(tr["position"], np.float32)[key_frame]
        q = np.asarray(tr["rotation"], np.float32)[key_frame]
        if not (np.isfinite(p[st > 0]).all() and np.isfinite(q[st > 0]).all()):
            raise BadSample(f"non-finite track for entity {e['id']}")
        ent_pos[ei] = np.nan_to_num(p)
        qn = np.linalg.norm(q, axis=1, keepdims=True)
        ent_rot[ei] = np.where(qn > 1e-6, q / np.maximum(qn, 1e-6), [0, 0, 0, 1])
        sk = skel.get(e.get("skeleton")) if e.get("skeleton") else None
        humanoid = cls == "person" and sk is not None and (sk.get("base") == "humanoid22" or
                                                           [j["name"] for j in sk["joints"]] == S.JOINTS)
        if humanoid and "pose" in tr:
            ps = np.asarray(tr["pose"], np.float32)[key_frame]
            pn = np.linalg.norm(ps, axis=-1, keepdims=True)
            ent_pose[ei] = np.where(pn > 1e-6, ps / np.maximum(pn, 1e-6), [0, 0, 0, 1])
        mesh = meshes[e["mesh"]]
        mat = mats.get(mesh.get("material"))
        v = np.asarray(mesh["positions"], np.float64)
        sc = np.median(np.asarray(tr["scale"], np.float64), 0) if "scale" in tr else np.ones(3)
        ent_size[ei] = (v.max(0) - v.min(0)) * sc
        app = e.get("appearance") if isinstance(e.get("appearance"), dict) else None
        if cls == "person" and app is not None and all(k in app for k in ("skin", "top", "bottom", "shoes", "hair")):
            ent_colors[ei] = [app[k] for k in ("skin", "top", "bottom", "shoes", "hair")]     # generator metadata
            ent_height[ei] = float(app["height"]) if "height" in app else person_appearance(mesh, mat)[1] * sc[1]
        elif cls == "person" and "joints" in mesh:
            col, h = person_appearance(mesh, mat)
            ent_colors[ei] = col
            ent_height[ei] = h * sc[1]
        else:
            ent_colors[ei] = object_appearance(mesh, mat)
            ent_height[ei] = ent_size[ei, 1]

    meta = {"name": stem, "genre": doc.get("meta", {}).get("genre"), "seed": doc.get("meta", {}).get("seed"),
            "title": doc.get("meta", {}).get("title"), "producer": doc.get("meta", {}).get("producer"),
            "source": {"restage": str(rs_path), "video": str(mp4_path), "aux": str(aux_path)},
            "video_size": doc.get("meta", {}).get("source", {}).get("width"),
            "continuity": len(doc.get("continuity", [])), "cam_jumps": camera_jumps(doc),
            "locations": [l["name"] for l in doc["locations"]],
            "entities": [e.get("name", e["id"]) for e in ents], "pack_version": 1}
    with np.load(aux_path) as aux:
        if "video_size" in aux.files:
            meta["video_size"] = [int(x) for x in aux["video_size"]]
        meta["aux_size"] = [int(aux["ids"].shape[2]), int(aux["ids"].shape[1])]
    clip = {
        "frames": frames, "key_frame": key_frame, "key_shot": key_shot,
        "fps": np.int32(fps), "n_frames": np.int32(T),
        "shot_range": shot_range, "shot_loc": shot_loc, "shot_size": shot_size, "shot_move": shot_move,
        "cam_rot": cam_rot, "cam_pos": cam_pos, "cam_fov": cam_fov,
        "loc_points": loc_points, "loc_sky": loc_sky, "loc_ground": loc_ground,
        "ent_class": ent_class, "ent_loc": ent_loc, "ent_ident": ent_ident,
        "ent_present": ent_present, "ent_visible": ent_visible,
        "ent_pos": ent_pos, "ent_rot": ent_rot, "ent_pose": ent_pose,
        "ent_height": ent_height, "ent_size": ent_size, "ent_colors": ent_colors,
        "depth": depth, "ids": ids, "thumbs": thumbs,
        "meta": np.array(json.dumps(meta)),
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **clip)
        tmp.replace(out_path)
    if verbose:
        print(f"{stem}: K={K} S={len(shots)} L={L} E={E} in {time.time() - t0:.2f}s")
    return clip


def load_clip(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d


# ---------------------------------------------------------------- checks

def entity_centres(clip_or_doc_entity_rows, clip=None):
    """Body-centre points [E, K, 3]: people -> pelvis (root + R_root (0, 0.53 h, 0)), objects -> bbox centre
    (root + R_root (0, h / 2, 0)); these lie inside the body, unlike the root of a person (on the ground)."""
    c = clip_or_doc_entity_rows if clip is None else clip
    E, K = c["ent_pos"].shape[:2]
    up = np.zeros((E, K, 3))
    frac = np.where(c["ent_class"] == 1, 0.53, 0.5)
    up[..., 1] = (c["ent_height"] * frac)[:, None]
    return c["ent_pos"].astype(np.float64) + S.quat_rotate(c["ent_rot"].astype(np.float64), up)


def projection_hits(clip, points=None, radius=1, min_pixels=3, project_fn=None, detail=False):
    """Camera-convention check. For every visible entity of the keyframe's location whose body centre projects
    inside the 64x36 label image and is not hidden behind something nearer (per the depth map), test whether a
    pixel of that entity's id lies within `radius` pixels. Returns (hits, total) (+ the misses if detail)."""
    pts = entity_centres(clip) if points is None else points
    project_fn = project_fn or project
    hits = total = 0
    misses = []
    E, K = clip["ent_present"].shape
    key_loc = clip["shot_loc"][clip["key_shot"]]
    # the centre lies about this far behind the visible surface (people: body depth, not the T-pose span)
    thick = np.where(clip["ent_class"] == 1, 0.25 * clip["ent_height"] / 1.75, 0.6 * clip["ent_size"].max(1)) + 0.1
    for k in range(K):
        u, v, z = project_fn(pts[:, k], clip["cam_rot"][k], clip["cam_pos"][k], clip["cam_fov"][k])
        for e in range(E):
            if not clip["ent_visible"][e, k] or clip["ent_loc"][e] != key_loc[k] or z[e] <= 0.05:
                continue
            if (clip["ids"][k] == e + 1).sum() < min_pixels:
                continue
            x, y = int(np.floor(u[e])), int(np.floor(v[e]))
            if not (0 <= x < LAB_W and 0 <= y < LAB_H):
                continue
            win = clip["ids"][k, max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1]
            hit = bool((win == e + 1).any())
            d = float(clip["depth"][k, y, x])
            if not hit and 0 < d < z[e] - thick[e]:
                continue                                    # occluded by a nearer surface: no evidence
            total += 1
            hits += hit
            if not hit:
                misses.append((k, e))
    return (hits, total, misses) if detail else (hits, total)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="path without extension: <base>.restage, <base>.mp4, <base>_aux.npz")
    ap.add_argument("out")
    a = ap.parse_args()
    c = pack(a.base, a.out, verbose=True)
    print("projection hits", projection_hits(c))
