"""Batch loader for packed clips (docs/TRAINING_DATA.md, "Batch from the loader").

    from restage.data.loader import Loader, make_loader
    with Loader("data/train", batch_size=8, augment=True, workers=4) as dl:
        for step, batch in zip(range(1000), dl):     # dict of numpy arrays, fixed shapes
            ...
    Loader("data/val", batch_size=8, augment=False, epoch=True)   # every window of every clip once
    it = make_loader("data/train", batch_size=8)                  # infinite iterator: next(it)

Worker processes use the spawn start method: a training script that creates a Loader must guard its entry
point with `if __name__ == "__main__":` (or pass mode="thread").

Per sample:
1. **Window**: a run of whole consecutive shots whose keyframes fit in Kmax (and <= Smax shots, <= Lmax
   locations), starting at a random shot (training) or covering the clip shot by shot (epoch mode). A single
   shot with more than Kmax keyframes is cut to a random run of Kmax consecutive keyframes.
2. **Entities**: those of the window's locations that are present in the window; if more than Emax, the ones
   visible in the most keyframes are kept and the pixels of the others get id 255 (ignore).
3. **Crop-cuts** (augment): each shot, with probability `p_crop`, is re-framed by a virtual camera at the same
   position: zoom z in [1.15, 2] (fov' = 2 atan(tan(fov / 2) / z)) and a small yaw / pitch toward an
   off-centre crop. Frames are warped with the exact homography (bilinear), ids and depth with nearest
   sampling (depth is re-expressed along the new optical axis), `cam_rot` / `cam_fov` updated, and an entity
   that leaves the frame is no longer visible.
4. **Gauge** per location on its first keyframe in the window: translate by (-p0.x, 0, -p0.z), rotate about
   +Y by -psi (psi = heading of the anchor camera's -Z), applied to cameras, entity roots and orientations and
   loc_points.
5. **Mirror** (augment, p = 0.5): x -> -x in every gauge. Frames / depth / ids flip horizontally; rotations
   R -> M R M (quaternion (x, y, z, w) -> (x, -y, -z, w)); left / right joints swap with their local rotations
   reflected the same way.
6. **Colour jitter** (augment): per clip gain / contrast / saturation / white balance / gamma, plus a small
   per-shot white-balance and gain change.
7. **Padding** to Kmax=48 / Smax=12 / Lmax=4 / Emax=10 with masks.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import queue
import threading
from pathlib import Path

import cv2
import numpy as np

from ..scene import JOINTS, quat_mul, quat_rotate
from .pack import LAB_H, LAB_W, project

KMAX, SMAX, LMAX, EMAX = 48, 12, 4, 10
IGNORE_ID = 255
PAD_FOV = 50.0          # padded keyframes get a valid fov so log(fov) stays finite under a mask
FRAME_KEYS = ("frames", "key_frame", "key_shot", "cam_rot", "cam_pos", "cam_fov", "depth", "ids")
CLIP_KEYS = FRAME_KEYS + ("fps", "shot_range", "shot_loc", "shot_size", "shot_move", "loc_points", "loc_sky",
                          "loc_ground", "ent_class", "ent_loc", "ent_ident", "ent_present", "ent_visible", "ent_pos",
                          "ent_rot", "ent_pose", "ent_height", "ent_size", "ent_colors")
# left <-> right joint permutation of humanoid22
MIRROR_PERM = np.array([JOINTS.index(("r_" + n[2:]) if n.startswith("l_") else ("l_" + n[2:]) if n.startswith("r_")
                                     else n) for n in JOINTS])
M_X = np.diag([-1.0, 1.0, 1.0])


# ---------------------------------------------------------------- small maths

def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def quat_y(a):
    return np.array([0.0, np.sin(a / 2), 0.0, np.cos(a / 2)])


def reflect_quat(q):
    """Rotation conjugated by the mirror x -> -x: (x, y, z, w) -> (x, -y, -z, w)."""
    q = np.array(q, copy=True)
    q[..., 1:3] *= -1
    return q


def heading(R):
    """Yaw psi of a camera (camera-to-world R): R_y(psi) (0, 0, -1) is the ground projection of its -Z axis
    (its +Y axis when looking straight down or up)."""
    h = -R[:, 2]
    if h[0] ** 2 + h[2] ** 2 < 1e-6:                    # looking straight down: image-up points forward
        h = R[:, 1] if R[1, 2] > 0 else -R[:, 1]
    return float(np.arctan2(-h[0], -h[2]))


def entity_centres(pos, rot, ent_class, height):
    """Body centres [E, K, 3] (see pack.entity_centres)."""
    up = np.zeros(pos.shape)
    up[..., 1] = (height * np.where(ent_class == 1, 0.53, 0.5))[:, None]
    return pos + quat_rotate(rot, up)


# ---------------------------------------------------------------- clip access

class ClipSource:
    """Lazy access to packed clips with a small per-process LRU cache of the arrays the loader needs."""

    def __init__(self, files, cache=8):
        self.files = [str(f) for f in files]
        self.cache, self._lru = cache, {}
        self._key_shot = {}

    def key_shot(self, i):
        if i not in self._key_shot:
            with np.load(self.files[i]) as z:
                self._key_shot[i] = (z["key_shot"].copy(), z["shot_loc"].copy())
        return self._key_shot[i]

    def get(self, i):
        if i in self._lru:
            c = self._lru.pop(i)
            self._lru[i] = c
            return c
        with np.load(self.files[i], allow_pickle=False) as z:
            c = {k: z[k] for k in CLIP_KEYS if k in z.files}
            c["meta"] = json.loads(str(z["meta"])) if "meta" in z.files else {}
        if "ent_ident" not in c:
            c["ent_ident"] = np.arange(len(c["ent_class"]), dtype=np.int32)
        self._key_shot[i] = (c["key_shot"], c["shot_loc"])
        self._lru[i] = c
        while len(self._lru) > self.cache:
            self._lru.pop(next(iter(self._lru)))
        return c


def list_clips(path):
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.glob("*.npz") if not f.name.endswith(".tmp.npz"))


# ---------------------------------------------------------------- windows

def window_from(key_shot, shot_loc, s0, rng=None, kmax=KMAX, smax=SMAX, lmax=LMAX):
    """Keyframe indices of the window that starts at shot s0 and greedily takes whole following shots."""
    n = np.bincount(key_shot, minlength=len(shot_loc))
    first = np.concatenate([[0], np.cumsum(n)])
    if n[s0] > kmax:                                     # one long shot: a run of kmax keyframes inside it
        off = 0 if rng is None else int(rng.integers(0, n[s0] - kmax + 1))
        return np.arange(first[s0] + off, first[s0] + off + kmax), s0 + 1
    s1, total, locs = s0 + 1, n[s0], {int(shot_loc[s0])}
    while s1 < len(n) and total + n[s1] <= kmax and s1 - s0 < smax and len(locs | {int(shot_loc[s1])}) <= lmax:
        total += n[s1]
        locs.add(int(shot_loc[s1]))
        s1 += 1
    return np.arange(first[s0], first[s1]), s1


def epoch_windows(key_shot, shot_loc, kmax=KMAX, smax=SMAX, lmax=LMAX):
    """Deterministic windows covering every keyframe once (long shots split into consecutive runs)."""
    n = np.bincount(key_shot, minlength=len(shot_loc))
    first = np.concatenate([[0], np.cumsum(n)])
    out, s = [], 0
    while s < len(n):
        if n[s] > kmax:
            for a in range(first[s], first[s + 1], kmax):
                out.append(np.arange(a, min(a + kmax, first[s + 1])))
            s += 1
            continue
        keys, s = window_from(key_shot, shot_loc, s, None, kmax, smax, lmax)
        out.append(keys)
    return out


# ---------------------------------------------------------------- augmentations

def _crop_maps(W, H, fov, z, R_rel):
    """For an output image W x H of the virtual camera (zoom z, rotation R_rel in the old camera frame): source
    pixel coordinates (map_x, map_y) in the old image and the depth factor 1 / (-d_z)."""
    F = (H / 2) / np.tan(np.radians(fov) / 2)
    Fn = F * z
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d_new = np.stack([(uu - W / 2) / Fn, -(vv - H / 2) / Fn, -np.ones_like(uu)], -1)
    d = d_new @ R_rel.T                                   # ray in the old camera frame
    nz = -d[..., 2]
    mx_ = W / 2 + F * d[..., 0] / nz - 0.5
    my_ = H / 2 - F * d[..., 1] / nz - 0.5
    return mx_.astype(np.float32), my_.astype(np.float32), (1.0 / nz).astype(np.float32)


def sample_crop(rng, fov, aspect=16 / 9, zmin=1.15, zmax=2.0):
    """Random zoom and off-centre direction whose frustum stays inside the original image."""
    z = float(rng.uniform(zmin, zmax))
    t = np.tan(np.radians(fov) / 2)
    th, tw = t, t * aspect
    ax, ay = rng.uniform(-1, 1), rng.uniform(-1, 1)
    for _ in range(12):
        yaw = np.arctan(ax * tw * (1 - 1 / z))
        pitch = np.arctan(ay * th * (1 - 1 / z))
        R = rot_y(yaw) @ rot_x(pitch)
        # corners of the new frustum, projected into the old image (normalised coordinates)
        c = np.array([[sx * tw / z, sy * th / z, -1] for sx in (-1, 1) for sy in (-1, 1)]) @ R.T
        x, y = c[:, 0] / -c[:, 2], c[:, 1] / -c[:, 2]
        if np.all(np.abs(x) <= tw) and np.all(np.abs(y) <= th) and np.all(c[:, 2] < 0):
            return z, R
        ax, ay = ax * 0.7, ay * 0.7
    return z, np.eye(3)


def apply_crop(s, keys_of_shot, z, R_rel):
    """Re-frame the keyframes `keys_of_shot` of sample dict `s` (world frame) with zoom z and rotation R_rel."""
    H, W = s["frames"].shape[1:3]
    for k in keys_of_shot:
        fov = float(s["cam_fov"][k])                     # per keyframe (zooming shots)
        mxf, myf, _ = _crop_maps(W, H, fov, z, R_rel)
        s["frames"][k] = cv2.remap(s["frames"][k], mxf, myf, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mxl, myl, dfac = _crop_maps(LAB_W, LAB_H, fov, z, R_rel)
        xi = np.clip(np.floor(mxl + 0.5).astype(int), 0, LAB_W - 1)
        yi = np.clip(np.floor(myl + 0.5).astype(int), 0, LAB_H - 1)
        old_ids = s["ids"][k].copy()
        s["ids"][k] = old_ids[yi, xi]
        s["depth"][k] = s["depth"][k][yi, xi] * dfac
        R_old = s["cam_rot"][k].copy()
        s["cam_rot"][k] = R_old @ R_rel
        s["cam_fov"][k] = np.degrees(2 * np.arctan(np.tan(np.radians(fov) / 2) / z))
        # visibility: an entity cropped out of the frame is not visible any more
        E = len(s["ent_class"])
        if E:
            cnt_old = np.bincount(old_ids.ravel(), minlength=256)[1:E + 1]
            cnt_new = np.bincount(s["ids"][k].ravel(), minlength=256)[1:E + 1]
            cen = entity_centres(s["ent_pos"][:, k:k + 1], s["ent_rot"][:, k:k + 1], s["ent_class"], s["ent_height"])
            u, v, zz = project(cen[:, 0], s["cam_rot"][k], s["cam_pos"][k], s["cam_fov"][k])
            inside = (zz > 0) & (u >= 0) & (u < LAB_W) & (v >= 0) & (v < LAB_H)
            s["ent_visible"][:, k] &= (cnt_new > 0) | ((cnt_old == 0) & inside)


def gauge_fix(s, key_loc):
    """Per-location gauge on the location's first keyframe (in place). Returns the anchor flags [K]."""
    K = len(key_loc)
    is_anchor = np.zeros(K, bool)
    for l in np.unique(key_loc):
        a = int(np.argmax(key_loc == l))
        is_anchor[a] = True
        R0, p0 = s["cam_rot"][a].astype(np.float64), s["cam_pos"][a].astype(np.float64)
        psi = heading(R0)
        G = rot_y(-psi)
        qG = quat_y(-psi)
        t = np.array([p0[0], 0.0, p0[2]])
        km = key_loc == l
        s["cam_rot"][km] = G @ s["cam_rot"][km]
        s["cam_pos"][km] = (s["cam_pos"][km] - t) @ G.T
        em = s["ent_loc"] == l
        if em.any():
            s["ent_pos"][em] = (s["ent_pos"][em] - t) @ G.T
            s["ent_rot"][em] = quat_mul(np.broadcast_to(qG, s["ent_rot"][em].shape), s["ent_rot"][em])
        s["loc_points"][l] = (s["loc_points"][l] - t) @ G.T
    return is_anchor


def mirror(s):
    """x -> -x in every gauge (in place)."""
    s["frames"] = np.ascontiguousarray(s["frames"][:, :, ::-1])
    s["depth"] = np.ascontiguousarray(s["depth"][:, :, ::-1])
    s["ids"] = np.ascontiguousarray(s["ids"][:, :, ::-1])
    s["cam_rot"] = M_X @ s["cam_rot"] @ M_X
    s["cam_pos"] = s["cam_pos"] @ M_X
    s["ent_pos"] = s["ent_pos"] @ M_X
    s["ent_rot"] = reflect_quat(s["ent_rot"])
    people = s["ent_class"] == 1
    if people.any():
        s["ent_pose"][people] = reflect_quat(s["ent_pose"][people][:, :, MIRROR_PERM])
    s["loc_points"] = s["loc_points"] @ M_X


def color_jitter(frames, key_shot, rng, strength=1.0):
    x = frames.astype(np.float32) / 255
    g = 1 + rng.uniform(-0.25, 0.25) * strength
    wb = 1 + rng.uniform(-0.08, 0.08, 3) * strength
    gamma = 1 + rng.uniform(-0.15, 0.15) * strength
    sat = 1 + rng.uniform(-0.3, 0.3) * strength
    con = 1 + rng.uniform(-0.2, 0.2) * strength
    per_shot = 1 + rng.uniform(-0.04, 0.04, (key_shot.max() + 1, 3)) * strength
    x = x * (g * wb) * per_shot[key_shot][:, None, None, :]
    x = np.clip(x, 0, 1) ** gamma
    m = x.mean(-1, keepdims=True)
    x = m + (x - m) * sat
    x = 0.5 + (x - 0.5) * con
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------- one sample

def make_sample(clip, keys, rng=None, augment=False, p_crop=0.35, p_mirror=0.5, jitter=1.0, zoom=(1.15, 2.0),
                kmax=KMAX, smax=SMAX, lmax=LMAX, emax=EMAX):
    """Build one padded sample (dict of numpy arrays without the batch axis) from a clip and window keys."""
    rng = rng or np.random.default_rng()
    keys = np.asarray(keys)
    K = len(keys)
    shots_w = np.unique(clip["key_shot"][keys])                    # original shot indices, in order
    shot_map = -np.ones(len(clip["shot_loc"]), int)
    shot_map[shots_w] = np.arange(len(shots_w))
    key_shot = shot_map[clip["key_shot"][keys]]
    locs_w = []
    for sh in shots_w:
        if clip["shot_loc"][sh] not in locs_w:
            locs_w.append(int(clip["shot_loc"][sh]))
    loc_map = -np.ones(len(clip["loc_points"]), int)
    loc_map[locs_w] = np.arange(len(locs_w))
    shot_loc = loc_map[clip["shot_loc"][shots_w]]
    key_loc = shot_loc[key_shot]
    # entities
    pres = clip["ent_present"][:, keys] > 0
    vis = clip["ent_visible"][:, keys] > 0
    cand = np.where(np.isin(clip["ent_loc"], locs_w) & pres.any(1))[0]
    if len(cand) > emax:
        score = vis[cand].sum(1) * 1000 + pres[cand].sum(1)
        cand = np.sort(cand[np.argsort(-score, kind="stable")[:emax]])
    E = len(cand)
    s = {"frames": clip["frames"][keys].copy(), "cam_rot": clip["cam_rot"][keys].astype(np.float64),
         "cam_pos": clip["cam_pos"][keys].astype(np.float64), "cam_fov": clip["cam_fov"][keys].astype(np.float64),
         "depth": clip["depth"][keys].astype(np.float32),
         "loc_points": clip["loc_points"][locs_w].astype(np.float64),
         "ent_class": clip["ent_class"][cand], "ent_loc": loc_map[clip["ent_loc"][cand]],
         "ent_pos": clip["ent_pos"][cand][:, keys].astype(np.float64),
         "ent_rot": clip["ent_rot"][cand][:, keys].astype(np.float64),
         "ent_pose": clip["ent_pose"][cand][:, keys].astype(np.float64),
         "ent_present": pres[cand], "ent_visible": vis[cand].copy(), "ent_height": clip["ent_height"][cand]}
    # ids: original entity e -> slot j (+1); dropped entities -> 255
    lut = np.zeros(256, np.uint8)
    nE = len(clip["ent_class"])
    lut[1:nE + 1] = IGNORE_ID
    lut[cand + 1] = np.arange(1, E + 1)
    s["ids"] = lut[clip["ids"][keys]]
    shot_zoom = np.ones(len(shots_w), np.float32)
    if augment and p_crop > 0:
        for j in range(len(shots_w)):
            if rng.random() < p_crop:
                kk = np.where(key_shot == j)[0]
                z, R_rel = sample_crop(rng, float(s["cam_fov"][kk].min()), zmin=zoom[0], zmax=zoom[1])
                apply_crop(s, kk, z, R_rel)
                shot_zoom[j] = z
    is_anchor = gauge_fix(s, key_loc)
    mirrored = bool(augment and rng.random() < p_mirror)
    if mirrored:
        mirror(s)
    if augment and jitter > 0:
        s["frames"] = color_jitter(s["frames"], key_shot, rng, jitter)
    fps = float(clip["fps"])
    key_time = (clip["key_frame"][keys] - clip["key_frame"][keys[0]]) / fps

    # ---- pad
    H, W = s["frames"].shape[1:3]
    L, S = len(locs_w), len(shots_w)
    out = {"frames": np.zeros((kmax, H, W, 3), np.uint8), "key_mask": np.zeros(kmax, bool),
           "key_shot": np.zeros(kmax, np.int32), "key_time": np.zeros(kmax, np.float32),
           "key_frame": np.zeros(kmax, np.int32), "shot_mask": np.zeros(smax, bool),
           "shot_loc": np.zeros(smax, np.int32), "shot_size": np.zeros(smax, np.int32),
           "shot_move": np.zeros(smax, np.int32), "shot_zoom": np.ones(smax, np.float32),
           "loc_mask": np.zeros(lmax, bool),
           "is_anchor": np.zeros(kmax, bool), "cam_rot": np.tile(np.eye(3, dtype=np.float32), (kmax, 1, 1)),
           "cam_pos": np.zeros((kmax, 3), np.float32), "cam_fov": np.full(kmax, PAD_FOV, np.float32),
           "loc_points": np.zeros((lmax, clip["loc_points"].shape[1], 3), np.float32),
           "loc_sky": np.zeros((lmax, 3), np.float32), "loc_ground": np.zeros((lmax, 3), np.float32),
           "ent_mask": np.zeros(emax, bool), "ent_class": np.zeros(emax, np.int32),
           "ent_loc": np.zeros(emax, np.int32), "ent_ident": np.zeros(emax, np.int32),
           "ent_present": np.zeros((emax, kmax), bool), "ent_visible": np.zeros((emax, kmax), bool),
           "ent_pos": np.zeros((emax, kmax, 3), np.float32), "ent_rot": np.zeros((emax, kmax, 4), np.float32),
           "ent_pose": np.zeros((emax, kmax, 22, 4), np.float32), "ent_height": np.zeros(emax, np.float32),
           "ent_size": np.zeros((emax, 3), np.float32), "ent_colors": np.zeros((emax, 5, 3), np.float32),
           "depth": np.zeros((kmax, LAB_H, LAB_W), np.float32), "ids": np.zeros((kmax, LAB_H, LAB_W), np.uint8),
           "mirrored": np.array(mirrored)}
    out["ent_rot"][..., 3] = 1
    out["ent_pose"][..., 3] = 1
    out["frames"][:K] = s["frames"]
    out["key_mask"][:K] = True
    out["key_shot"][:K] = key_shot
    out["key_time"][:K] = key_time
    out["key_frame"][:K] = clip["key_frame"][keys]
    out["shot_mask"][:S] = True
    out["shot_loc"][:S] = shot_loc
    out["shot_size"][:S] = clip["shot_size"][shots_w]
    out["shot_move"][:S] = clip["shot_move"][shots_w]
    out["shot_zoom"][:S] = shot_zoom
    out["loc_mask"][:L] = True
    out["is_anchor"][:K] = is_anchor
    out["cam_rot"][:K] = s["cam_rot"]
    out["cam_pos"][:K] = s["cam_pos"]
    out["cam_fov"][:K] = s["cam_fov"]
    out["loc_points"][:L] = s["loc_points"]
    out["loc_sky"][:L] = clip["loc_sky"][locs_w]
    out["loc_ground"][:L] = clip["loc_ground"][locs_w]
    out["depth"][:K] = s["depth"]
    out["ids"][:K] = s["ids"]
    if E:
        out["ent_mask"][:E] = True
        out["ent_class"][:E] = s["ent_class"]
        out["ent_loc"][:E] = s["ent_loc"]
        idents = clip["ent_ident"][cand]
        out["ent_ident"][:E] = np.unique(idents, return_inverse=True)[1]
        out["ent_present"][:E, :K] = s["ent_present"]
        out["ent_visible"][:E, :K] = s["ent_visible"] & s["ent_present"]
        out["ent_pos"][:E, :K] = s["ent_pos"]
        out["ent_rot"][:E, :K] = s["ent_rot"]
        out["ent_pose"][:E, :K] = s["ent_pose"]
        out["ent_height"][:E] = clip["ent_height"][cand]
        out["ent_size"][:E] = clip["ent_size"][cand]
        out["ent_colors"][:E] = clip["ent_colors"][cand]
    return out


def collate(samples):
    return {k: np.stack([s[k] for s in samples]) for k in samples[0]}


# ---------------------------------------------------------------- iteration

class Sampler:
    """Produces batches (pure numpy; used inside the worker threads / processes)."""

    def __init__(self, files, batch_size=8, augment=True, seed=0, epoch=False, cache=8, **aug):
        self.src = ClipSource(files, cache)
        self.B, self.augment, self.epoch, self.aug = batch_size, augment, epoch, aug
        self.rng = np.random.default_rng(seed)
        self._plan = None

    def plan(self):
        """Epoch mode: every (clip, window) pair, in order."""
        if self._plan is None:
            self._plan = []
            for i in range(len(self.src.files)):
                ks, sl = self.src.key_shot(i)
                for keys in epoch_windows(ks, sl):
                    self._plan.append((i, keys))
        return self._plan

    def random_window(self):
        i = int(self.rng.integers(len(self.src.files)))
        ks, sl = self.src.key_shot(i)
        s0 = int(self.rng.integers(len(sl)))
        keys, _ = window_from(ks, sl, s0, self.rng)
        return i, keys

    def batch(self, items):
        samples, ids = [], []
        for i, keys in items:
            samples.append(make_sample(self.src.get(i), keys, self.rng, self.augment, **self.aug))
            ids.append(i)
        b = collate(samples)
        b["clip_id"] = np.array(ids, np.int32)
        return b

    def batch_at(self, j):
        """Epoch mode: batch j (the last one is filled up with repeats, marked by batch_mask)."""
        plan = self.plan()
        items = plan[j * self.B:(j + 1) * self.B]
        n = len(items)
        items = items + [items[-1]] * (self.B - n)
        b = self.batch(items)
        b["batch_mask"] = np.arange(self.B) < n
        return b

    def n_batches(self):
        return -(-len(self.plan()) // self.B)

    def random_batch(self):
        b = self.batch([self.random_window() for _ in range(self.B)])
        b["batch_mask"] = np.ones(self.B, bool)
        return b


def _worker(files, cfg, wid, nw, q, stop):
    smp = Sampler(files, seed=[cfg["seed"], wid], **{k: v for k, v in cfg.items() if k != "seed"})
    j = wid
    while not stop.is_set():
        if cfg["epoch"]:
            if j >= smp.n_batches():
                q.put(None)
                return
            b = smp.batch_at(j)
            j += nw
        else:
            b = smp.random_batch()
        while not stop.is_set():
            try:
                q.put(b, timeout=0.2)
                break
            except queue.Full:
                pass


class Loader:
    """Iterator of batches with background prefetch (worker processes by default, or threads)."""

    def __init__(self, path, batch_size=8, augment=True, seed=0, workers=4, prefetch=6, epoch=False, mode="process",
                 cache=8, **aug):
        self.files = list_clips(path) if isinstance(path, (str, Path)) else [Path(f) for f in path]
        if not self.files:
            raise FileNotFoundError(f"no packed clips in {path}")
        self.cfg = dict(batch_size=batch_size, augment=augment, seed=seed, epoch=epoch, cache=cache, **aug)
        self.workers, self.prefetch, self.mode = max(1, workers), prefetch, mode
        self._procs = []
        self._started = False

    def _start(self):
        if self.mode == "process":
            ctx = mp.get_context("spawn")
            self._q, self._stop = ctx.Queue(self.prefetch), ctx.Event()
            mk = ctx.Process
        else:
            self._q, self._stop = queue.Queue(self.prefetch), threading.Event()
            mk = threading.Thread
        files = [str(f) for f in self.files]
        for w in range(self.workers):
            p = mk(target=_worker, args=(files, self.cfg, w, self.workers, self._q, self._stop), daemon=True)
            p.start()
            self._procs.append(p)
        self._done = 0
        self._started = True

    def __iter__(self):
        if not self._started:
            self._start()
        return self

    def __next__(self):
        if not self._started:
            self._start()
        while True:
            b = self._q.get()
            if b is None:
                self._done += 1
                if self._done >= self.workers:
                    self.close()
                    raise StopIteration
                continue
            return b

    def __len__(self):
        if not self.cfg["epoch"]:
            raise TypeError("infinite loader")
        return Sampler(self.files, **self.cfg).n_batches()

    def close(self):
        if not self._started:
            return
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
        except Exception:
            pass
        for p in self._procs:
            p.join(timeout=2)
            if self.mode == "process" and p.is_alive():
                p.terminate()
        self._procs = []
        self._started = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def make_loader(path, batch_size=8, augment=True, seed=0, workers=3, mode="process", **kw):
    """An infinite batch iterator (random windows) over the clips in `path`, prefetched in the background.
    `next(it)` returns a dict of numpy arrays. Use `Loader(..., epoch=True)` to visit every window once."""
    return Loader(path, batch_size=batch_size, augment=augment, seed=seed, workers=workers, mode=mode, **kw)
