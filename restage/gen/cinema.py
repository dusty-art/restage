"""Shot planner: real film grammar turned into per-frame cameras.

A shot is planned from a `ShotSpec` (size, move, subject, angle and lens preferences) and
solved into a camera track by `CameraSolver`:

* **Framing**: the shot size sets how much of the subject the frame height covers (wide = full
  body, medium = waist up, close-up = head and shoulders, ...); the lens (vertical fov, from a
  35 mm-equivalent focal length per size) and the distance follow from that. Eyes go near the
  upper third with a little headroom; subjects sit on a vertical third with look room on the side
  they face (or centred for direct address).
* **Placement**: candidate viewpoints around the preferred angle are scored and rejected if the
  camera is inside or too close to geometry or a person, if the subject would be occluded too
  often, or if they break the 180 degree rule of a conversation.
* **Moves**: static, pan, tilt, dolly, truck, orbit, handheld (band-limited 1/f shake plus
  breathing and walking bob), crane, drone and first-person (attached to a head or a car).

Genre templates (`plan_*`) produce shot lists with genre-typical sizes, moves and shot lengths
(log-normal, roughly 1.5-6 s), cut on the beat for dance, shot / reverse-shot over the shoulder
for conversations, jump cuts and inserts for vlogs, POV / follow / drone for hiking, dashcam /
chase / roadside for driving, and first / third person for voxel gameplay.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..scene import JID, quat_euler, quat_mul, quat_rotate, quat_to_mat
from .motion import minjerk, quat_continuous, wrap, yaw_of_dir
from .spatial import sphere_occludes

SENSOR_H = 20.25   # mm: 16:9 crop of a 36 mm wide full-frame sensor

LENS = {  # 35 mm-equivalent focal length ranges (mm) per shot size
    "extreme_wide": (14, 24), "wide": (18, 35), "medium": (28, 50), "closeup": (45, 90),
    "extreme_closeup": (70, 135), "insert": (40, 85),
}


def fov_from_focal(f_mm):
    return float(np.degrees(2 * np.arctan(SENSOR_H / 2 / f_mm)))


def focal_from_fov(fov):
    return SENSOR_H / 2 / np.tan(np.radians(fov) / 2)


# ---------------------------------------------------------------- subjects

@dataclass
class Subject:
    """Per-frame anchor points of something the camera frames (arrays [T, 3])."""
    eid: str
    kind: str
    pts: dict
    fwd: np.ndarray
    height: float
    radius: float
    spheres: np.ndarray = None    # [T, K, 3] occluder spheres
    sphere_r: np.ndarray = None   # [K]
    body_fwd: np.ndarray = None   # [T, 3] facing of the body (heading), fwd is where the head looks


def person_subject(eid, track, scale=1.0):
    jw = track["joints_world"].astype(np.float64)
    jr = track["joint_rot_world"].astype(np.float64)
    T = len(jw)
    head = jw[:, JID["head"]]
    hq = jr[:, JID["head"]]
    eye = head + quat_rotate(hq, np.broadcast_to([0, 0.11 * scale, 0.08 * scale], (T, 3)))
    top = head + quat_rotate(hq, np.broadcast_to([0, 0.24 * scale, 0.0], (T, 3)))
    fwd = quat_rotate(hq, np.broadcast_to([0, 0, 1.0], (T, 3)))
    feet = np.minimum(jw[:, JID["l_ankle"]], jw[:, JID["r_ankle"]]) * [1, 0, 1] + \
        [0, 1, 0] * np.minimum(jw[:, JID["l_toe"], 1:2], jw[:, JID["r_toe"], 1:2]) - [0, 0.02, 0]
    feet[:, [0, 2]] = (jw[:, JID["l_ankle"], [0, 2]] + jw[:, JID["r_ankle"], [0, 2]]) / 2
    pts = {"eye": eye, "top": top, "head": head + quat_rotate(hq, np.broadcast_to([0, 0.11 * scale, 0], (T, 3))),
           "chest": jw[:, JID["chest"]], "hip": jw[:, JID["pelvis"]], "feet": feet,
           "center": (jw[:, JID["spine2"]] + jw[:, JID["pelvis"]]) / 2,
           "hands": (jw[:, JID["l_wrist"]] + jw[:, JID["r_wrist"]]) / 2,
           "l_wrist": jw[:, JID["l_wrist"]], "r_wrist": jw[:, JID["r_wrist"]]}
    idx = [JID[j] for j in ("head", "chest", "spine1", "pelvis", "l_knee", "r_knee", "l_ankle", "r_ankle",
                            "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_shoulder", "r_shoulder")]
    sph = jw[:, idx].copy()
    sph[:, 0] = pts["head"]
    # generous radii: cameras should keep clear sight lines past hair, shoulders and chair backs
    r = np.array([0.17, 0.24, 0.2, 0.2, 0.1, 0.1, 0.08, 0.08, 0.08, 0.08, 0.07, 0.07, 0.12, 0.12]) * scale
    body = quat_rotate(jr[:, JID["chest"]], np.broadcast_to([0, 0, 1.0], (T, 3))) * [1, 0, 1]
    body /= np.maximum(np.linalg.norm(body, axis=1, keepdims=True), 1e-9)
    return Subject(eid, "person", pts, fwd, 1.78 * scale, 0.3 * scale, sph, r, body)


def object_subject(eid, track, height, radius, kind="object", center_h=None):
    p = track["position"].astype(np.float64)
    T = len(p)
    q = track["rotation"].astype(np.float64)
    ch = height / 2 if center_h is None else center_h
    c = p + [0, ch, 0]
    fwd = quat_rotate(q, np.broadcast_to([0, 0, 1.0], (T, 3)))
    pts = {"center": c, "top": p + [0, height, 0], "feet": p, "eye": c, "head": c, "chest": c, "hip": c, "hands": c}
    n = 3 if radius > 0.6 else 1
    if n == 3:
        sph = np.stack([c - fwd * radius * 0.6, c, c + fwd * radius * 0.6], 1)
        r = np.full(3, max(height, radius * 0.5) * 0.55)
    else:
        sph = c[:, None]
        r = np.array([max(radius, height / 2)])
    return Subject(eid, kind, pts, fwd, height, radius, sph, r, fwd)


# ---------------------------------------------------------------- shots

@dataclass
class ShotSpec:
    start: int                     # video frames [start, end)
    end: int
    loc: str
    size: str
    move: str
    subject: str | None = None
    world: np.ndarray = None       # world frame per video frame of this shot
    params: dict = field(default_factory=dict)
    transition: str = "cut"

    @property
    def n(self):
        return self.end - self.start


def shot_lengths(total, rng, median=3.0, sigma=0.45, lo=1.5, hi=6.0, fps=30, quantum=None):
    """Split `total` frames into shot lengths drawn from a clipped log-normal (seconds)."""
    out = []
    left = total
    while left > 0:
        L = float(np.clip(rng.lognormal(np.log(median), sigma), lo, hi))
        n = int(round(L * fps))
        if quantum:
            n = max(quantum, int(round(n / quantum)) * quantum)
        if left - n < lo * fps * 0.8:
            n = left
        out.append(min(n, left))
        left -= out[-1]
    return out


# ---------------------------------------------------------------- camera noise

def colored_noise(n, fps, rng, f_lo=0.1, f_hi=6.0, alpha=1.0, dims=3):
    """Band-limited 1/f^alpha noise, unit std per dimension."""
    m = max(n, 16)
    freqs = np.fft.rfftfreq(m * 2, 1 / fps)
    spec = (rng.normal(size=(len(freqs), dims)) + 1j * rng.normal(size=(len(freqs), dims)))
    amp = np.where((freqs >= f_lo) & (freqs <= f_hi), 1.0 / np.maximum(freqs, f_lo) ** alpha, 0.0)
    amp *= 1 / (1 + np.exp((freqs - f_hi) * 3))
    x = np.fft.irfft(spec * amp[:, None], axis=0)[:n]
    x -= x.mean(0)
    return x / np.maximum(x.std(0), 1e-9)


def handheld(n, fps, rng, strength=1.0, walking_hz=None):
    """Handheld camera shake: (rotation offsets [n, 3] radians (pitch, yaw, roll), position offsets [n, 3])."""
    rot = colored_noise(n, fps, rng, 0.08, 7.0, 1.1) * np.radians([0.55, 0.7, 0.45]) * strength
    t = np.arange(n) / fps
    breath = np.sin(2 * np.pi * rng.uniform(0.2, 0.3) * t + rng.uniform(0, 6))
    rot[:, 0] += np.radians(0.2) * breath * strength
    pos = colored_noise(n, fps, rng, 0.05, 4.0, 1.3) * np.array([0.012, 0.01, 0.012]) * strength
    pos[:, 1] += 0.004 * breath * strength
    if walking_hz:
        ph = 2 * np.pi * walking_hz * t + rng.uniform(0, 6)
        pos[:, 1] += 0.022 * strength * (np.cos(ph) + 0.25 * np.cos(2 * ph))
        pos[:, 0] += 0.012 * strength * np.sin(ph / 2)
        rot[:, 2] += np.radians(0.8) * strength * np.sin(ph / 2)
        rot[:, 0] += np.radians(0.5) * strength * np.cos(ph)
    return rot, pos


# ---------------------------------------------------------------- aiming

def aim_rotation(cam, target, ndc=(0.0, 0.0), fov=40.0, aspect=16 / 9, roll=0.0):
    """Quaternion (camera looks down -Z, no roll unless given) so `target` projects at `ndc`.

    Exact for a yaw-pitch camera; broadcasts over leading dims.
    """
    cam, target = np.asarray(cam, float), np.asarray(target, float)
    ndc = np.asarray(ndc, float)
    tv = np.tan(np.radians(np.asarray(fov, float)) / 2)
    th = tv * aspect
    u = np.stack(np.broadcast_arrays(ndc[..., 0] * th, ndc[..., 1] * tv, -np.ones_like(tv)), -1)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    w = target - cam
    w = w / np.maximum(np.linalg.norm(w, axis=-1, keepdims=True), 1e-9)
    c = -u[..., 2]
    A_ = np.sqrt(u[..., 1] ** 2 + c ** 2)
    phi = np.arctan2(u[..., 1], c)
    pitch = np.arcsin(np.clip(w[..., 1] / A_, -1, 1)) - phi
    vx = u[..., 0]
    vz = u[..., 1] * np.sin(pitch) + u[..., 2] * np.cos(pitch)
    yaw = np.arctan2(w[..., 0], w[..., 2]) - np.arctan2(vx, vz)
    return quat_euler(pitch, yaw, roll + np.zeros_like(pitch)), pitch, yaw


def project(cam_p, cam_q, pts, fov, aspect=16 / 9):
    """NDC (x, y) and depth of world points for camera tracks (broadcast over frames)."""
    rel = pts - cam_p
    R = quat_to_mat(cam_q)
    pc = np.einsum("...ji,...j->...i", R, rel)
    z = -pc[..., 2]
    tv = np.tan(np.radians(fov) / 2)
    x = pc[..., 0] / np.maximum(z, 1e-6) / (tv * aspect)
    y = pc[..., 1] / np.maximum(z, 1e-6) / tv
    return x, y, z


# ---------------------------------------------------------------- solver

SIZE_SPAN = {  # (bottom anchor, top anchor, extra span factor) for people
    "extreme_wide": ("feet", "top", 6.0), "wide": ("feet", "top", 1.45), "medium": ("hip", "top", 1.25),
    "closeup": ("chest", "top", 1.3), "extreme_closeup": ("eye", "top", 1.5), "insert": ("center", "top", 1.8),
}


class CameraSolver:
    def __init__(self, world, subjects: dict, fps=30, aspect=16 / 9, rng=None, eye_heights=None):
        self.world = world
        self.subjects = subjects
        self.fps = fps
        self.aspect = aspect
        self.rng = np.random.default_rng() if rng is None else rng

    # ------------------------------------------------------------ helpers
    def _span(self, sub: Subject, size, frames):
        if sub.kind != "person":
            base = max(sub.height, sub.radius * 2)
            k = {"extreme_wide": 8.0, "wide": 2.6, "medium": 1.8, "closeup": 1.3, "extreme_closeup": 0.9, "insert": 1.6}[size]
            return base * k, sub.pts["center"][frames].mean(0), 0.0
        lo, hi, k = SIZE_SPAN[size]
        f = frames
        top = sub.pts["top"][f, 1].mean()
        bot = sub.pts[lo][f, 1].mean()
        if size == "extreme_closeup":
            span = 0.3 * sub.height / 1.78
        elif size == "closeup":
            span = (top - bot) * k
        else:
            span = max(top - bot, 0.3) * k
        return span, None, None

    def _aim_point(self, sub, size, f):
        if sub.kind != "person":
            return sub.pts["center"][f], 0.0
        if size in ("wide", "extreme_wide"):
            c = (sub.pts["top"][f] + sub.pts["feet"][f]) / 2
            return c, (0.05 if size == "wide" else 0.0)
        return sub.pts["eye"][f], None

    def _others(self, f, exclude=()):
        sph, rad = [], []
        for eid, s in self.subjects.items():
            if eid in exclude or s.spheres is None:
                continue
            sph.append(s.spheres[f])
            rad.append(np.broadcast_to(s.sphere_r, s.spheres.shape[1:2]))
        if not sph:
            return None, None
        return np.concatenate(sph, axis=-2), np.concatenate(rad, axis=-1)

    def _visibility(self, cams, sub, frames, keys=("eye", "chest", "hip"), exclude_self=True):
        """Fraction of (frame, point) pairs with a clear line of sight. cams [n, 3] per frame."""
        pts = np.stack([sub.pts[k][frames] for k in keys], 1)
        n, k = pts.shape[:2]
        c = np.repeat(cams[:, None], k, 1).reshape(-1, 3)
        p = pts.reshape(-1, 3)
        blocked = self.world.blocked(c, p)
        sph, rad = self._others(frames, (sub.eid,) if exclude_self else ())
        if sph is not None:
            for i in range(n):
                ci, pi = c[i * k:(i + 1) * k], p[i * k:(i + 1) * k]
                blocked[i * k:(i + 1) * k] |= sphere_occludes(ci, pi, sph[i], rad)
        return 1 - blocked.mean()

    def valid_positions(self, cams, frames, margin=0.3, ignore=()):
        """True per frame where the camera is clear of geometry and people."""
        ok = ~self.world.inside(cams, margin=margin, ground_margin=0.25)
        sph, rad = self._others(frames, ignore)
        if sph is not None:
            d = np.linalg.norm(cams[:, None] - sph, axis=-1) - rad
            ok &= d.min(1) > 0.35
        return ok

    # ------------------------------------------------------------ main entry
    def solve(self, spec: ShotSpec):
        if spec.move == "fpv":
            return self._fpv(spec)
        if spec.move == "drone":
            return self._drone(spec)
        sub = self.subjects.get(spec.subject) if spec.subject else None
        if sub is None:
            return self._establishing(spec)
        return self._framed(spec, sub)

    def _lens(self, size, p):
        lo, hi = LENS[size]
        f = p.get("focal") or float(np.exp(self.rng.uniform(np.log(lo), np.log(hi))))
        return f

    def _framed(self, spec, sub):
        """Search a viewpoint for a framed subject, then apply the move."""
        rng = self.rng
        p = spec.params
        fr = spec.world
        n = len(fr)
        size = spec.size
        span, _, _ = self._span(sub, size, fr)
        if p.get("two_shot"):
            other = self.subjects[p["two_shot"]]
            a, b = sub.pts["head"][fr].mean(0), other.pts["head"][fr].mean(0)
            width = np.linalg.norm((a - b)[[0, 2]]) * 1.6 + 0.6
            span = max(span, width / self.aspect)
        focal = self._lens(size, p)
        fov0 = fov_from_focal(focal)
        # preferred azimuth relative to the subject's facing (0 = frontal)
        facing = (sub.fwd[fr] * 0.4 + sub.body_fwd[fr] * 0.6).mean(0)
        face_yaw = float(yaw_of_dir(facing[0], facing[2]))
        az_pref = p.get("azimuth", rng.choice([-1, 1]) * rng.uniform(15, 45))
        az_world = p.get("azimuth_world")
        h_pref = p.get("height")
        spread = p.get("spread", 35.0)
        best = None
        aim_pts = []
        for f in fr[:: max(1, n // 6)]:
            aim_pts.append(self._aim_point(sub, size, f)[0])
        aim_mean = np.mean(aim_pts, 0)
        if p.get("two_shot"):
            other = self.subjects[p["two_shot"]]
            aim_mean = (aim_mean + np.mean([self._aim_point(other, size, f)[0] for f in fr[::max(1, n // 6)]], 0)) / 2
        motion = np.ptp(sub.pts["hip"][fr][:, [0, 2]], axis=0).max() if n > 1 else 0.0
        sample = fr[:: max(1, n // 8)]
        # cameras that travel with the subject are placed relative to where it is at the start
        moving_cam = spec.move in ("follow", "orbit") or (spec.move == "truck" and motion > 1.0)
        aim_track = np.array([self._aim_point(sub, size, f)[0] for f in sample])
        if moving_cam:
            aim_mean = aim_track[0]
            f0 = sub.body_fwd[fr[: max(2, n // 10)]].mean(0)
            face_yaw = float(yaw_of_dir(f0[0], f0[2]))
        if p.get("fixed_pos") is not None:
            cam = np.asarray(p["fixed_pos"], float) + rng.normal(0, p.get("jitter", 0.0), 3)
            dist = np.linalg.norm(aim_mean - cam)
            fov = float(np.clip(np.degrees(2 * np.arctan(span / 2 / max(dist, 0.1))), 8, 80))
            return self._apply_move(spec, sub, cam, fov, False)
        if p.get("roadside"):
            hip = sub.pts["hip"][fr]
            mid_p = hip[n // 2]
            d = sub.fwd[fr[n // 2]]
            # a pan must stand far enough to the side to follow the car smoothly (turn rate at the
            # closest point = speed / distance); a locked-off camera can stand anywhere
            speed = float(np.linalg.norm(np.diff(hip[:, [0, 2]], axis=0), axis=1).mean() * self.fps) if n > 1 else 0.0
            d_pan = max(7.0, 1.2 * speed / (np.radians(MAX_TURN_DEG * 0.85) * self.fps))
            best_r = None
            for d_min in ((d_pan, 7.0) if spec.move == "pan" else (7.0,)):
                for _ in range(24):
                    side_v = np.array([d[2], 0, -d[0]]) * rng.choice([-1, 1])
                    cam = mid_p + side_v * rng.uniform(d_min, d_min * 1.6) + np.array([d[0], 0, d[2]]) * rng.uniform(-5, 25)
                    cam[1] = float(self.world.ground_at(cam[0], cam[2])) + p.get("height", 1.5)
                    if self.world.inside(cam[None], margin=0.5)[0]:
                        continue
                    vis = self._visibility(np.broadcast_to(cam, (len(sample), 3)).copy(), sub, sample,
                                           keys=("center", "top"))
                    if best_r is None or vis > best_r[0]:
                        best_r = (vis, cam, d_min)
                    if vis > 0.95:
                        break
                if best_r is not None and best_r[0] > 0.6:
                    break
            if best_r is not None and spec.move == "pan" and best_r[2] < d_pan:
                spec.move = "static"       # too close for a smooth pan: lock off and let the car pass
            if best_r is not None:
                cam = best_r[1]
                dist = np.linalg.norm(hip - cam, axis=1).min()
                fov = float(np.clip(np.degrees(2 * np.arctan(span / 2 / max(dist, 1.0))), 20, 70))
                return self._apply_move(spec, sub, cam, fov, False)
        for attempt in range(4):
            cands = []
            d_pref = span / (2 * np.tan(np.radians(fov0) / 2))
            for _ in range(48 if attempt == 0 else 72):
                # distance varies independently; the lens keeps the framing (clamped to real lenses)
                k = np.exp(rng.uniform(np.log(0.8), np.log(1.25))) if attempt == 0 else \
                    np.exp(rng.uniform(np.log(0.35), np.log(1.8)))
                dist = max(d_pref * k, p.get("min_dist", 0.0))
                fov = float(np.degrees(2 * np.arctan(span / 2 / dist)))
                if not 7.0 <= fov <= 78.0:
                    continue
                if size in ("wide", "extreme_wide") and motion > 0.5 and spec.move == "static":
                    dist *= 1 + motion / max(span, 1) * 0.6
                sp = spread * (0.5 + attempt * 0.9)
                if attempt >= 2:
                    yaw = rng.uniform(0, 2 * np.pi)        # anywhere around the subject
                elif az_world is not None:
                    yaw = np.radians(az_world + rng.normal(0, sp))
                else:
                    yaw = face_yaw + np.radians(az_pref + rng.normal(0, sp))
                if h_pref is not None:
                    h = h_pref + rng.normal(0, 0.08 + 0.1 * attempt)
                else:
                    eye_h = sub.pts["eye"][fr, 1].mean() - sub.pts["feet"][fr, 1].mean() if sub.kind == "person" else sub.height
                    base = {"wide": rng.uniform(0.7, 1.05), "extreme_wide": rng.uniform(0.8, 1.6),
                            "medium": rng.uniform(0.9, 1.03), "closeup": rng.uniform(0.95, 1.03),
                            "extreme_closeup": 1.0, "insert": 1.0}[size]
                    h = eye_h * base if sub.kind == "person" else max(0.5, sub.height * rng.uniform(1.0, 2.5))
                horiz = dist * np.array([np.sin(yaw), np.cos(yaw)])
                cam = np.array([aim_mean[0] + horiz[0], 0.0, aim_mean[2] + horiz[1]])
                feet_y = sub.pts["feet"][fr, 1].mean()
                cam[1] = max(feet_y + h, float(self.world.ground_at(cam[0], cam[2])) + 0.35)
                cands.append((cam, fov, yaw, h))
            scored = []
            # a camera that stays put must be far enough from a moving subject's path that a smooth
            # pan (turn rate = speed / distance) can follow it
            hip_s = sub.pts["hip"][sample]
            speed = float(np.linalg.norm(np.diff(sub.pts["hip"][fr][:, [0, 2]], axis=0), axis=1).max() * self.fps) \
                if n > 1 else 0.0
            d_pass = max(0.8, 1.15 * speed / (np.radians(MAX_TURN_DEG * 0.8) * self.fps))
            for cam, fov, yaw, h in cands:
                if not moving_cam and attempt < 3 and np.linalg.norm((hip_s - cam)[:, [0, 2]], axis=1).min() < d_pass:
                    continue
                if moving_cam:
                    cams = aim_track + (cam - aim_track[0])
                    cams[:, 1] = cam[1] + (aim_track[:, 1] - aim_track[0, 1])
                else:
                    cams = np.broadcast_to(cam, (len(sample), 3)).copy()
                ok = self.valid_positions(cams, sample, margin=0.3)
                if not ok.all():
                    continue
                if p.get("side_of") is not None and attempt < 3:
                    a, b, sgn = p["side_of"]
                    if np.sign(_cross2(np.array(b)[[0, 2]] - np.array(a)[[0, 2]], cam[[0, 2]] - np.array(a)[[0, 2]])) != sgn:
                        continue
                vis = self._visibility(cams, sub, sample)
                if p.get("two_shot"):
                    vis = min(vis, self._visibility(cams, self.subjects[p["two_shot"]], sample))
                if vis < (0.8, 0.55, 0.4, 0.0)[attempt]:
                    continue
                dev = abs(wrap(yaw - (face_yaw + np.radians(az_pref)))) if az_world is None else \
                    abs(wrap(yaw - np.radians(az_world)))
                # prefer seeing the face: penalise viewpoints from behind unless asked for
                front = np.cos(wrap(yaw - face_yaw))
                score = vis * 2 - dev * 0.5 - abs(np.log(fov / fov0)) * 0.6 + rng.uniform(0, 0.05)
                if attempt >= 2 and sub.kind == "person":
                    score += 0.8 * front
                scored.append((score, cam, fov, yaw))
            if scored:
                best = max(scored, key=lambda x: x[0])
                break
        if best is None:
            # last resort: look from straight in front at a close distance, wide lens
            fov = 60.0
            dist = span / (2 * np.tan(np.radians(fov) / 2))
            cam = aim_mean + np.array([facing[0], 0, facing[2]]) * dist
            cam[1] = aim_mean[1]
            best = (0, cam, fov, face_yaw)
            info_fallback = True
        else:
            info_fallback = False
        _, cam, fov, yaw = best
        return self._apply_move(spec, sub, cam, fov, info_fallback)

    def _screen_x(self, sub, cam, fr, p):
        """Horizontal placement: a third with look room, or centred for direct address.

        The composition is chosen once per shot from where the subject faces on average (an
        operator does not reframe on every glance), and it is a continuous function of that
        facing, so the camera never snaps between thirds inside a shot.
        """
        if p.get("center"):
            return np.zeros(len(fr))
        x = p.get("screen_x")
        if x is not None:
            return np.full(len(fr), x)
        d = sub.pts["eye"][fr] - cam
        right = np.stack([-d[:, 2], np.zeros(len(fr)), d[:, 0]], 1)
        right /= np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-9)
        facing = sub.fwd[fr] * 0.4 + sub.body_fwd[fr] * 0.6
        look = float((facing * right).sum(1).mean())
        k = p.get("thirds", 0.3)
        return np.full(len(fr), -k * float(np.clip(look / 0.35, -1.0, 1.0)))

    def _aim_track(self, spec, sub, cams, fov, lag=0.0):
        """Per-frame orientation keeping the subject framed (with optional operator lag)."""
        fr = spec.world
        n = len(fr)
        size = spec.size
        p = spec.params
        aim = np.array([self._aim_point(sub, size, f)[0] for f in fr])
        if p.get("two_shot"):
            other = self.subjects[p["two_shot"]]
            aim = (aim + np.array([self._aim_point(other, size, f)[0] for f in fr])) / 2
        span, _, _ = self._span(sub, size, fr)
        if sub.kind == "person" and size not in ("wide", "extreme_wide"):
            dist = np.linalg.norm(aim - cams, axis=1)
            half = dist * np.tan(np.radians(fov) / 2)
            head_top = 0.13 * sub.height / 1.78
            y_n = np.clip(0.86 - head_top / np.maximum(half, 1e-3), 0.12, 0.42)
        else:
            y_n = np.full(n, 0.0 if size == "extreme_wide" else 0.04)
        if p.get("two_shot"):
            x_n = np.zeros(n)
        else:
            x_n = self._screen_x(sub, cams.mean(0), fr, p)
        if lag > 0:
            # operator lag in seconds of subject motion: fine for people, but a car at 27 m/s would be
            # aimed metres behind itself, so fast subjects are followed nearly without lag
            if sub.kind != "person":
                lag = min(lag, 0.03)
            aim = _lag_filter(aim, lag * self.fps)
        aim = gaussian_filter1d(aim, 1.5, axis=0, mode="nearest") if n > 3 else aim
        q, pitch, yaw = aim_rotation(cams, aim, np.stack([x_n, y_n], 1), fov, self.aspect)
        return q, pitch, yaw

    def _apply_move(self, spec, sub, cam, fov, fallback):
        rng = self.rng
        fr = spec.world
        n = len(fr)
        move = spec.move
        p = spec.params
        t = np.arange(n) / self.fps
        u = minjerk(t / max(t[-1], 1e-6)) if n > 1 else np.zeros(n)
        aim0 = sub.pts["center" if sub.kind != "person" else "chest"][fr].mean(0)
        to = cam - aim0
        to[1] = 0
        dist = max(np.linalg.norm(to), 0.3)
        dirh = to / dist
        right = np.array([-dirh[2], 0, dirh[0]])
        cams = np.broadcast_to(cam, (n, 3)).copy()
        fovs = np.full(n, fov)
        lag = 0.0
        hh = p.get("handheld", move == "handheld")
        amp_moves = {"dolly", "orbit", "crane"} | ({"truck"} if np.ptp(sub.pts["hip"][fr][:, [0, 2]], axis=0).max() <= 1.0
                                                    else set())
        if move in amp_moves:
            # draw the move once, then flip or shrink it if the path would enter geometry
            k_dolly = p.get("dolly", rng.choice([-1, 1]) * rng.uniform(0.15, 0.3))        # + = push in
            k_truck = p.get("truck", rng.choice([-1, 1]) * rng.uniform(0.4, 1.2))
            k_orbit = p.get("orbit", rng.choice([-1, 1]) * rng.uniform(12, 22) * max(t[-1], 1))
            k_crane = p.get("crane", rng.choice([-1, 1]) * rng.uniform(0.8, 1.8))
            c_orb = gaussian_filter1d(sub.pts["chest" if sub.kind == "person" else "center"][fr], 8, axis=0, mode="nearest")

            def path(a):
                if move == "dolly":
                    out = aim0 + dirh * (dist * (1 - a * k_dolly * u))[:, None]
                    out[:, 1] = cam[1]
                elif move == "truck":
                    out = cam + right[None] * (a * k_truck * (u - 0.5))[:, None]
                elif move == "orbit":
                    ang = np.radians(a * k_orbit) * (u - 0.5)
                    base = np.arctan2(dirh[0], dirh[2])
                    out = c_orb + np.stack([np.sin(base + ang), np.zeros(n), np.cos(base + ang)], 1) * dist
                    out[:, 1] = cam[1]
                else:
                    out = np.broadcast_to(cam, (n, 3)).copy()
                    out[:, 1] = np.maximum(cam[1] + a * k_crane * (u - 0.5), self.world.ground_at(cam[0], cam[2]) + 0.4)
                return out
            for a in (1.0, -1.0, 0.5, -0.5, 0.0):
                cams = path(a)
                if not self.world.inside(cams[::3], margin=0.15, ground_margin=0.15).any() and \
                        self.valid_positions(cams[::3], fr[::3], margin=0.15).all():
                    break
            if a == 0.0 and move != "crane":
                cams = path(0.0)
        elif move == "truck":
            # track alongside the moving subject
            off = cam - sub.pts["hip"][fr[0]]
            cams = sub.pts["hip"][fr] + off
            cams[:, 1] = cam[1]
            cams = gaussian_filter1d(cams, 6, axis=0, mode="nearest")
        elif move == "follow":
            # keep the camera offset from the moving subject (behind / ahead), with operator lag
            off = cam - sub.pts["hip"][fr[0]]
            hd = sub.body_fwd[fr]
            yaw_s = np.unwrap(np.arctan2(hd[:, 0], hd[:, 2]))
            yaw_s = gaussian_filter1d(yaw_s, 20, mode="nearest")
            lim = np.radians(MAX_TURN_DEG * 0.5)        # the operator walks around, never swings
            for i in range(1, n):
                yaw_s[i] = yaw_s[i - 1] + np.clip(yaw_s[i] - yaw_s[i - 1], -lim, lim)
            yaw_s = gaussian_filter1d(yaw_s, 3, mode="nearest")
            rel = np.array([off[0], off[2]])
            yaw0 = yaw_s[0]
            c, s_ = np.cos(yaw_s - yaw0), np.sin(yaw_s - yaw0)
            ox = rel[0] * c + rel[1] * s_
            oz = -rel[0] * s_ + rel[1] * c
            base = _lag_filter(sub.pts["hip"][fr], (0.18 if sub.kind == "person" else 0.02) * self.fps)
            cams = base + np.stack([ox, np.zeros(n), oz], 1)
            gy = self.world.ground_at(cams[:, 0], cams[:, 2])
            cams[:, 1] = gaussian_filter1d(gy, 6, mode="nearest") + (cam[1] - float(self.world.ground_at(cam[0], cam[2])))
            lag = 0.15
        elif move == "pan":
            lag = 0.25
            if np.ptp(sub.pts["hip"][fr][:, [0, 2]], axis=0).max() < 0.4 and n > 20:
                # scripted pan settling on the subject
                q_end, _, yaw_end = self._aim_track(spec, sub, cams, fovs)
                swing = np.radians(p.get("pan", rng.choice([-1, 1]) * rng.uniform(12, 30)))
                swing = np.sign(swing) * min(abs(swing), np.radians(MAX_TURN_DEG * 0.6) * 0.8 * n / 1.875)
                q, pitch, yaw = self._aim_track(spec, sub, cams, fovs)
                yaw = yaw + swing * (1 - minjerk(t / max(t[-1] * 0.8, 1e-3)))
                q = quat_euler(pitch, yaw, np.zeros(n))
                return self._finish(spec, cams, q, fovs, hh, fallback)
        elif move == "tilt":
            q, pitch, yaw = self._aim_track(spec, sub, cams, fovs)
            tilt = np.radians(p.get("tilt", rng.uniform(10, 25))) * (1 - minjerk(t / max(t[-1] * 0.85, 1e-3)))
            q = quat_euler(pitch - tilt, yaw, np.zeros(n))
            return self._finish(spec, cams, q, fovs, hh, fallback)
        if move != "static":
            q, _, _ = self._aim_track(spec, sub, cams, fovs, lag)
        else:
            # a locked-off tripod: one orientation for the whole shot, aimed at the average framing;
            # a moving subject travels through the frame
            q, _, _ = self._aim_track(spec, sub, cams, fovs)
            q = np.broadcast_to(_mean_quat(q), q.shape).copy()
        return self._finish(spec, cams, q, fovs, hh, fallback, p.get("walk_hz") if move == "follow" else None)

    def _finish(self, spec, cams, q, fovs, hh, fallback, walking_hz=None):
        n = len(cams)
        if hh:
            strength = spec.params.get("shake", self.rng.uniform(0.6, 1.4))
            rot, pos = handheld(n, self.fps, self.rng, strength, walking_hz)
            q = quat_mul(q, quat_euler(rot[:, 0], rot[:, 1], rot[:, 2]))
            cams = cams + pos
        # keep cameras out of geometry: pull toward the subject when needed
        info = {"fallback": fallback}
        bad = self.world.inside(cams, margin=0.12, ground_margin=0.15)
        if bad.any() and spec.subject in self.subjects:
            sub = self.subjects[spec.subject]
            tgt = sub.pts["chest" if sub.kind == "person" else "center"][spec.world]
            for i in np.where(bad)[0]:
                for k in np.linspace(0.1, 0.9, 9):
                    c = cams[i] + (tgt[i] - cams[i]) * k
                    if not self.world.inside(c[None], margin=0.12, ground_margin=0.15)[0]:
                        cams[i] = c
                        break
            cams = gaussian_filter1d(cams, 2, axis=0, mode="nearest")
            info["pulled"] = int(bad.sum())
        return _finalize(cams, q, fovs, info)

    # ------------------------------------------------------------ special moves
    def _establishing(self, spec):
        """No subject: frame the location focus from a valid wide viewpoint (static, pan or drone-like)."""
        rng = self.rng
        p = spec.params
        n = spec.n
        focus = np.asarray(p.get("focus", [0, 1, 0]), float)
        fov = p.get("fov", fov_from_focal(rng.uniform(16, 28)))
        best = None
        for _ in range(80):
            yaw = rng.uniform(0, 2 * np.pi)
            dist = p.get("dist", rng.uniform(2.0, 6.0))
            cam = focus + np.array([np.sin(yaw) * dist, 0, np.cos(yaw) * dist])
            cam[1] = p.get("height", float(self.world.ground_at(cam[0], cam[2])) + rng.uniform(1.4, 2.2))
            if self.world.inside(cam[None], margin=0.35)[0]:
                continue
            d = np.linalg.norm(cam - focus)
            score = d + rng.uniform(0, 0.5)
            if best is None or score > best[0]:
                best = (score, cam)
        cam = best[1] if best else focus + [0, 0.5, 2.0]
        cams = np.broadcast_to(cam, (n, 3)).copy()
        t = np.arange(n) / self.fps
        q, pitch, yaw = aim_rotation(cams, np.broadcast_to(focus, (n, 3)), np.zeros((n, 2)), fov, self.aspect)
        if spec.move == "pan":
            yaw = yaw + np.radians(rng.choice([-1, 1]) * 15) * (0.5 - minjerk(t / max(t[-1], 1e-6)))
            q = quat_euler(pitch, yaw, np.zeros(n))
        return self._finish(spec, cams, q, np.full(n, fov), spec.params.get("handheld", False), best is None)

    def _fpv(self, spec):
        """First person: attached to a person's eyes (with gaze / mouse-look) or a vehicle dash."""
        p = spec.params
        fr = spec.world
        n = len(fr)
        rng = self.rng
        host = p["host_track"]
        t = np.arange(n) / self.fps
        if p.get("vehicle"):
            pos = host["position"][fr].astype(np.float64)
            q = host["rotation"][fr].astype(np.float64)
            off = np.asarray(p.get("mount", [0.0, 1.2, 0.6]), float)
            cams = pos + quat_rotate(q, np.broadcast_to(off, (n, 3)))
            fwd = quat_rotate(q, np.broadcast_to([0, 0, 1.0], (n, 3)))
            yaw = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2])) + np.pi   # camera -Z = vehicle +Z
            pitch = np.full(n, np.radians(p.get("pitch", 0.0)))
            body = host.get("pose")
            roll = np.zeros(n)
            if body is not None:
                bq = body[fr, 1].astype(np.float64)
                roll = 2 * np.arcsin(np.clip(bq[:, 2], -1, 1)) * 0.6
                pitch = pitch - 2 * np.arcsin(np.clip(bq[:, 0], -1, 1)) * 0.6
            vib = colored_noise(n, self.fps, rng, 2.0, 12.0, 0.5) * np.radians(0.08)
            q = quat_euler(pitch + vib[:, 0], yaw + vib[:, 1] * 0.5, roll + vib[:, 2])
            fov = np.full(n, p.get("fov", 52.0))
            return _finalize(cams, q, fov, {"attached": True})
        jw = host["joints_world"][fr].astype(np.float64)
        jr = host["joint_rot_world"][fr].astype(np.float64)
        s = p.get("scale", 1.0)
        head = jw[:, JID["head"]]
        hq = jr[:, JID["head"]]
        cams = head + quat_rotate(hq, np.broadcast_to([0, 0.1 * s, 0.17 * s], (n, 3)))
        # gaze: walking direction smoothed + look-around (mouse-look for games)
        root_q = host["rotation"][fr].astype(np.float64)
        fwd = quat_rotate(root_q, np.broadcast_to([0, 0, 1.0], (n, 3)))
        yaw = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2]))
        yaw = gaussian_filter1d(yaw, 8, mode="nearest")
        look = np.zeros((n, 2))
        game = p.get("game", False)
        tt = 0.0
        while tt < t[-1]:
            dur = rng.uniform(0.6, 2.0) if game else rng.uniform(1.5, 4.0)
            m = (t >= tt) & (t < tt + dur)
            amp = (0.35 if game else 0.3)
            look[m] = [rng.normal(0, amp), rng.normal(-0.05, 0.14 if game else 0.1)]
            tt += dur
        # game mouse-look: quick flicks (short smoothing); POV heads: slower
        look = gaussian_filter1d(look, (0.18 if game else 0.25) * self.fps, axis=0, mode="nearest")
        yaw_c = yaw + look[:, 0] + np.pi
        pitch_c = np.clip(look[:, 1], -0.6, 0.5)
        if not game:
            pitch_c = pitch_c - 0.12
        # a head (or a player's mouse) turns fast but not instantly: rate-limit both axes, then ease
        lim = np.radians(MAX_TURN_DEG * 0.7)
        for arr in (yaw_c, pitch_c):
            for i in range(1, n):
                arr[i] = arr[i - 1] + np.clip(arr[i] - arr[i - 1], -lim, lim)
        yaw_c = gaussian_filter1d(yaw_c, 2.0, mode="nearest")
        pitch_c = gaussian_filter1d(pitch_c, 2.0, mode="nearest")
        roll = np.zeros(n)
        if game:
            # head bob from the avatar's steps, but no roll (game cameras stay level)
            bob = cams[:, 1] - gaussian_filter1d(cams[:, 1], 10, mode="nearest")
            cams[:, 1] = gaussian_filter1d(cams[:, 1], 10, mode="nearest") + bob * 0.6
        else:
            rot, pos = handheld(n, self.fps, rng, 0.6)
            pitch_c, yaw_c, roll = pitch_c + rot[:, 0], yaw_c + rot[:, 1], rot[:, 2]
        q = quat_euler(pitch_c, yaw_c, roll)
        fov = np.full(n, p.get("fov", 62.0))
        return _finalize(cams, q, fov, {"attached": True})

    def _drone(self, spec):
        """Drone: high, slow and smooth; follows or reveals the subject; orbit or fly-by."""
        p = spec.params
        rng = self.rng
        fr = spec.world
        n = len(fr)
        t = np.arange(n) / self.fps
        sub = self.subjects.get(spec.subject)
        target = sub.pts["center"][fr] if sub is not None else np.broadcast_to(np.asarray(p.get("focus", [0, 0, 0])), (n, 3))
        target = gaussian_filter1d(target, 12, axis=0, mode="nearest")
        alt = p.get("altitude", rng.uniform(6, 18))
        dist = p.get("dist", rng.uniform(8, 20))
        style = p.get("style", rng.choice(["follow", "orbit", "flyby", "reveal"]))
        base_yaw = p.get("yaw", rng.uniform(0, 2 * np.pi))
        if sub is not None:
            hd = sub.body_fwd[fr].mean(0)
            behind = float(np.arctan2(hd[0], hd[2])) + np.pi
            best_y = None
            sample = fr[:: max(1, n // 6)]
            for k in range(10):
                yy = behind + rng.normal(0, 0.5 + 0.25 * k)
                cam_s = target[:: max(1, n // 6)] + np.array([np.sin(yy), 0, np.cos(yy)]) * dist + [0, alt, 0]
                vis = self._visibility(cam_s[:len(sample)], sub, sample, keys=("head", "center"))
                if best_y is None or vis > best_y[0]:
                    best_y = (vis, yy)
                if vis > 0.9:
                    break
            base_yaw = best_y[1]
        if style == "orbit":
            ang = base_yaw + np.radians(rng.choice([-1, 1]) * rng.uniform(5, 9)) * t
        elif style == "flyby":
            ang = np.full(n, base_yaw + rng.choice([-1, 1]) * 1.2)
        else:
            ang = np.full(n, base_yaw)
        off = np.stack([np.sin(ang), np.zeros(n), np.cos(ang)], 1) * dist
        cams = target + off
        if style == "flyby":
            side = np.array([np.cos(base_yaw), 0, -np.sin(base_yaw)])
            cams = cams + side * (rng.uniform(2, 4) * (t - t[-1] / 2))[:, None]
        if style == "reveal":
            cams = cams + (target - cams) * (0.35 * minjerk(t / max(t[-1], 1e-6)))[:, None] * [1, 0, 1]
        # keep a horizontal standoff: looking almost straight down makes the heading unstable
        hz = (cams - target)[:, [0, 2]]
        hd_ = np.linalg.norm(hz, axis=1, keepdims=True)
        stand = max(6.0, 0.8 * alt)
        away = np.where(hd_ > 1e-3, hz / np.maximum(hd_, 1e-3), np.array([[np.sin(base_yaw), np.cos(base_yaw)]]))
        cams[:, [0, 2]] = target[:, [0, 2]] + away * np.sqrt(hd_ ** 2 + stand ** 2)      # smooth, no kink
        gy = self.world.ground_at(cams[:, 0], cams[:, 2])
        rise = alt + (rng.uniform(-4, 6) * minjerk(t / max(t[-1], 1e-6)) if style == "reveal" else 0)
        cams[:, 1] = np.maximum(gaussian_filter1d(gy, 15, mode="nearest") + 4, target[:, 1] + rise)
        cams = gaussian_filter1d(cams, 4, axis=0, mode="nearest")
        fov = np.full(n, p.get("fov", rng.uniform(32, 50)))
        x_n = np.zeros(n) if style != "flyby" else np.full(n, rng.choice([-0.2, 0.2]))
        q, pitch, yaw = aim_rotation(cams, target, np.stack([x_n, np.full(n, -0.05)], 1), fov, self.aspect)
        bank = np.clip(gaussian_filter1d(np.gradient(np.unwrap(yaw)) * self.fps, 10, mode="nearest") * -0.15, -0.15, 0.15)
        q = quat_euler(pitch, yaw, bank)
        return self._finish(spec, cams, q, fov, False, False)


MAX_TURN_DEG = 1.9     # per-frame rotation limit of every camera inside a shot (a real head / operator)


def limit_turn_rate(q, max_deg=MAX_TURN_DEG):
    """Follow an orientation track but turn at most `max_deg` per frame (slerp toward the target).

    Deliberately fast targets (a car passing close to a roadside camera, a game flick) make the
    camera lag and catch up instead of snapping, so a turn inside a shot never looks like a cut.
    """
    q = quat_continuous(np.asarray(q, float))
    out = q.copy()
    lim = np.radians(max_deg)
    for i in range(1, len(q)):
        prev, tgt = out[i - 1], q[i]
        d = float(np.dot(prev, tgt))
        if d < 0:
            tgt, d = -tgt, -d
        ang = 2 * np.arccos(min(1.0, d))
        if ang > lim:
            w = lim / ang
            th = ang / 2
            r = (np.sin((1 - w) * th) * prev + np.sin(w * th) * tgt) / np.sin(th)
            out[i] = r / np.linalg.norm(r)
        else:
            out[i] = tgt
    return out


def _finalize(cams, q, fovs, info):
    """Common exit of every camera solver: finite values, sign-continuous and turn-rate-limited rotations."""
    cams = np.asarray(cams, np.float64)
    q = np.asarray(q, np.float64)
    fovs = np.broadcast_to(np.asarray(fovs, np.float64), (len(cams),)).copy()
    if not (np.isfinite(cams).all() and np.isfinite(q).all() and np.isfinite(fovs).all()) or \
            np.abs(cams).max() > 1e5:
        raise FloatingPointError("camera solver produced non-finite or diverging values")
    q = quat_continuous(q / np.linalg.norm(q, axis=1, keepdims=True))
    raw = turn_degrees(q)
    info["max_turn_raw"] = float(raw.max()) if len(raw) else 0.0    # before limiting: tests catch snaps here
    if len(q) > 2 and raw.max() > MAX_TURN_DEG:
        # limit, smooth the resulting rate changes, limit again (the second pass rarely engages)
        q = limit_turn_rate(q)
        q = quat_continuous(gaussian_filter1d(q, 1.5, axis=0, mode="nearest"))
        q = limit_turn_rate(q / np.linalg.norm(q, axis=1, keepdims=True))
        info["turn_limited"] = int((raw > MAX_TURN_DEG).sum())
    return cams.astype(np.float32), q.astype(np.float32), fovs.astype(np.float32), info


def turn_degrees(q):
    """Rotation angle between consecutive orientations of a track, in degrees."""
    q = np.asarray(q, np.float64)
    return np.degrees(2 * np.arccos(np.clip(np.abs((q[1:] * q[:-1]).sum(1)), 0, 1)))


def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _lag_filter(x, tau_frames):
    """Camera-operator lag: two cascaded first-order lags (smooth, never overshoots, always stable).

    For a target moving at constant velocity the follower trails by 2 * tau_frames frames.
    """
    if tau_frames <= 0:
        return np.asarray(x, float)
    x = np.asarray(x, float)
    a = 1.0 - np.exp(-1.0 / tau_frames)
    y1 = x[0].copy()
    y2 = x[0].copy()
    out = np.empty_like(x)
    for i in range(len(x)):
        y1 = y1 + a * (x[i] - y1)
        y2 = y2 + a * (y1 - y2)
        out[i] = y2
    return out


def _mean_quat(q):
    q = quat_continuous(q)
    m = q.mean(0)
    return m / np.linalg.norm(m)


# ---------------------------------------------------------------- genre templates

def plan_talking(f0, f1, fps, rng, host, a_cam, objects, loc):
    """Vlog / talking head over video frames [f0, f1): A-roll medium / close-up from the camera the
    host talks to (jump cuts = same camera, punched in), plus inserts of objects and cutaways.

    a_cam: dict(pos, azimuth_world, handheld) of the main camera the host addresses.
    """
    shots = []
    f = f0
    prev = None
    i = 0
    while f < f1:
        L = int(round(np.clip(rng.lognormal(np.log(3.0), 0.4), 1.4, 5.5) * fps))
        r = rng.random()
        kind = "aroll"
        if i > 0 and r < 0.13 and objects:
            kind, L = "insert", int(round(rng.uniform(1.0, 2.0) * fps))
        elif i > 0 and r < 0.22:
            kind = "wide"
        elif i > 0 and r < 0.3:
            kind, L = "angle", int(round(rng.uniform(1.5, 3.0) * fps))
        if f1 - (f + L) < 1.2 * fps:
            L = f1 - f
        L = min(L, f1 - f)
        if kind == "insert":
            spec = ShotSpec(f, f + L, loc, "insert", str(rng.choice(["static", "handheld", "dolly"])), str(rng.choice(objects)),
                            params={"azimuth": float(rng.uniform(-60, 60)), "spread": 50})
        elif kind == "wide":
            spec = ShotSpec(f, f + L, loc, "wide", str(rng.choice(["static", "pan", "handheld"])), host,
                            params={"azimuth_world": a_cam["azimuth_world"] + rng.choice([-1, 1]) * rng.uniform(25, 55),
                                    "spread": 25})
        elif kind == "angle":
            spec = ShotSpec(f, f + L, loc, "closeup", "static", host,
                            params={"azimuth_world": a_cam["azimuth_world"] + rng.choice([-1, 1]) * rng.uniform(30, 55),
                                    "spread": 15})
        else:
            # A-roll: same camera; consecutive A-roll shots are jump cuts that change the crop
            size = str(rng.choice(["medium", "closeup"])) if prev is None else ("closeup" if prev == "medium" else "medium")
            spec = ShotSpec(f, f + L, loc, size, "handheld" if a_cam.get("handheld") else "static", host,
                            params={"fixed_pos": a_cam["pos"], "jitter": 0.015, "center": a_cam.get("center", True),
                                    "handheld": a_cam.get("handheld", False), "shake": 0.45, "a_roll": True})
            prev = size
        shots.append(spec)
        f += L
        i += 1
    return shots


def fill_gaps(shots, n_frames):
    """Make shots contiguous over [0, n_frames) by stretching neighbours."""
    shots = sorted(shots, key=lambda s: s.start)
    shots[0].start = 0
    for a, b in zip(shots[:-1], shots[1:]):
        if b.start != a.end:
            a.end = b.start
    shots[-1].end = n_frames
    return [s for s in shots if s.end > s.start]


def plan_broll(f0, f1, fps, rng, subject, loc):
    """Cutaway footage in another location: wide / follow / drone of the subject."""
    shots = []
    k = f0
    for L in shot_lengths(f1 - f0, rng, 2.6, 0.35, 1.4, 4.5, fps):
        size, move = [("wide", "pan"), ("medium", "follow"), ("extreme_wide", "drone"), ("wide", "handheld")][rng.integers(4)]
        shots.append(ShotSpec(k, k + L, loc, size, move, subject,
                              params={"azimuth": float(rng.choice([-1, 1]) * rng.uniform(20, 140)), "spread": 40,
                                      "handheld": move != "drone", "walk_hz": 1.8 if move == "follow" else None}))
        k += L
    return shots


def plan_interview(n_frames, fps, rng, a, b, turns, loc, pos_a, pos_b):
    """Two-person conversation: establishing two-shot, then shot / reverse-shot singles and OTS
    on the speaker, occasional reactions, all on one side of the line of action (180 degree rule).

    turns: list of (start_frame, end_frame, speaker_id).
    """
    side = rng.choice([-1, 1])
    line = (np.asarray(pos_a, float), np.asarray(pos_b, float))
    # camera sides: the reference is the line a -> b; every camera keeps the same sign
    shots = []
    first = int(round(rng.uniform(2.5, 4.0) * fps))
    first = min(first, n_frames)
    shots.append(ShotSpec(0, first, loc, "wide", rng.choice(["static", "dolly", "pan"]), a,
                          params={"two_shot": b, "side_of": (line[0], line[1], side), "spread": 60,
                                  "azimuth_world": _perp_azimuth(line, side), "focal": rng.uniform(20, 30)}))
    f = first

    def speaker_at(fr):
        for t0, t1, spk in turns:
            if t0 <= fr < t1:
                return spk
        return a

    while f < n_frames:
        L = int(round(np.clip(rng.lognormal(np.log(3.0), 0.4), 1.5, 6.0) * fps))
        L = min(L, n_frames - f)
        if n_frames - (f + L) < 1.2 * fps:
            L = n_frames - f
        spk = speaker_at(f + L // 3)
        listener = b if spk == a else a
        r = rng.random()
        if r < 0.12:
            subj, other, kind = listener, spk, "reaction"
        else:
            subj, other, kind = spk, listener, "speaker"
        r2 = rng.random()
        p_s = line[0] if subj == a else line[1]
        p_o = line[1] if subj == a else line[0]
        # camera near the other person, on the chosen side: looking at subj
        az = float(np.degrees(np.arctan2(*(p_o - p_s)[[0, 2]])))
        sgn_off = side * (1 if subj == a else -1)
        if r2 < 0.35 and kind == "speaker":
            size, extra = "medium", {"azimuth_world": az + sgn_off * rng.uniform(10, 18), "spread": 6, "ots": other}
        elif r2 < 0.8:
            size, extra = rng.choice(["closeup", "medium"]), {"azimuth_world": az + sgn_off * rng.uniform(22, 40), "spread": 10}
        elif r2 < 0.9:
            size, extra = "extreme_closeup", {"azimuth_world": az + sgn_off * rng.uniform(25, 40), "spread": 8}
            L = min(L, int(2.5 * fps)) if n_frames - (f + min(L, int(2.5 * fps))) >= 1.2 * fps else L
        else:
            size, extra = "wide", {"two_shot": other, "azimuth_world": _perp_azimuth(line, side), "spread": 40}
        move = rng.choice(["static", "static", "static", "dolly", "handheld"])
        params = {"side_of": (line[0], line[1], side), "thirds": 0.3, **extra}
        shots.append(ShotSpec(f, f + L, loc, size, move, subj, params=params))
        f += L
    return shots, side


def _perp_azimuth(line, side):
    a, b = line
    d = (b - a)[[0, 2]]
    perp = np.array([-d[1], d[0]]) * side
    # cross(d, cam - a) has sign `side` for cameras on that side; perp points there
    if np.sign(d[0] * perp[1] - d[1] * perp[0]) != side:
        perp = -perp
    return float(np.degrees(np.arctan2(perp[0], perp[1])))


def plan_dance(n_frames, fps, rng, dancers, loc, bpm, beat0, focus):
    beat = 60.0 / bpm * fps
    shots = []
    f = 0
    first = True
    while f < n_frames:
        beats = int(rng.choice([2, 4, 4, 4, 4, 8, 8] if not first else [4, 8]))
        end = int(round(beat0 * fps + beat * np.round((f + beats * beat - beat0 * fps) / beat)))
        end = max(end, f + int(beat * 2))
        if n_frames - end < beat * 3:
            end = n_frames
        end = min(end, n_frames)
        lead = str(rng.choice(dancers))
        r = rng.random()
        if first or r < 0.3:
            size, move = "wide", rng.choice(["static", "orbit", "dolly", "crane"])
            params = {"azimuth": rng.uniform(-35, 35), "spread": 30}
            if len(dancers) > 1:
                params["two_shot"] = str([d for d in dancers if d != lead][0])
        elif r < 0.65:
            size, move = "medium", rng.choice(["handheld", "orbit", "static", "truck"])
            params = {"azimuth": rng.uniform(-60, 60), "spread": 30}
        elif r < 0.85:
            size, move = "closeup", rng.choice(["handheld", "static", "orbit"])
            params = {"azimuth": rng.uniform(-50, 50), "spread": 25}
        else:
            size, move = "wide", rng.choice(["tilt", "handheld"])
            params = {"azimuth": rng.uniform(-80, 80), "spread": 40, "height": rng.uniform(0.4, 0.8)}
        shots.append(ShotSpec(f, end, loc, size, move, lead, params=params))
        f = end
        first = False
    return shots


def plan_hiking(n_frames, fps, rng, hikers, segments):
    """segments: list of (loc_id, video_start, video_end) per location visited."""
    shots = []
    for loc, s0, s1 in segments:
        f = s0
        first = True
        while f < s1:
            L = int(round(np.clip(rng.lognormal(np.log(3.4), 0.4), 1.6, 6.0) * fps))
            if s1 - (f + L) < 1.5 * fps:
                L = s1 - f
            L = min(L, s1 - f)
            lead = str(rng.choice(hikers[loc]))
            r = rng.random()
            if first and rng.random() < 0.5:
                spec = ShotSpec(f, f + L, loc, "extreme_wide", "drone", lead, params={"style": rng.choice(["reveal", "follow", "orbit"])})
            elif first:
                spec = ShotSpec(f, f + L, loc, "wide", rng.choice(["static", "pan"]), lead,
                                params={"azimuth": rng.choice([-1, 1]) * rng.uniform(20, 70), "spread": 30})
            elif r < 0.25:
                spec = ShotSpec(f, f + L, loc, "medium", "fpv", lead, params={"pov": True})
            elif r < 0.5:
                spec = ShotSpec(f, f + L, loc, rng.choice(["wide", "medium"]), "follow", lead,
                                params={"azimuth": 180 + rng.normal(0, 15), "spread": 20, "handheld": True,
                                        "height": rng.uniform(1.4, 1.7), "walk_hz": 1.8})
            elif r < 0.65:
                spec = ShotSpec(f, f + L, loc, "medium", "follow", lead,
                                params={"azimuth": rng.normal(0, 20), "spread": 20, "handheld": True, "walk_hz": 1.8,
                                        "height": rng.uniform(1.4, 1.7)})
            elif r < 0.8:
                spec = ShotSpec(f, f + L, loc, "wide", rng.choice(["static", "pan"]), lead,
                                params={"azimuth": rng.choice([-1, 1]) * rng.uniform(40, 100), "spread": 30})
            elif r < 0.86:
                spec = ShotSpec(f, f + L, loc, "extreme_wide", "drone", lead, params={})
            else:
                spec = ShotSpec(f, f + L, loc, "closeup", "follow", lead,
                                params={"azimuth": rng.choice([-1, 1]) * rng.uniform(20, 50), "spread": 15, "handheld": True,
                                        "walk_hz": 1.8})
            shots.append(spec)
            f += L
            first = False
    return shots


def plan_driving(n_frames, fps, rng, ego, loc, others=()):
    shots = []
    f = 0
    first = True
    while f < n_frames:
        L = int(round(np.clip(rng.lognormal(np.log(3.2), 0.4), 1.5, 6.0) * fps))
        if n_frames - (f + L) < 1.5 * fps:
            L = n_frames - f
        L = min(L, n_frames - f)
        r = rng.random()
        if first or r < 0.3:
            spec = ShotSpec(f, f + L, loc, "medium", "fpv", ego, params={"vehicle": True, "fov": rng.uniform(44, 56)})
        elif r < 0.55:
            spec = ShotSpec(f, f + L, loc, "medium", "follow", ego,
                            params={"azimuth": 180 + rng.normal(0, 12), "spread": 10, "height": rng.uniform(2.0, 3.2),
                                    "focal": rng.uniform(22, 32), "handheld": False})
        elif r < 0.75:
            spec = ShotSpec(f, f + L, loc, "wide", "pan", ego,
                            params={"azimuth": rng.choice([-1, 1]) * rng.uniform(35, 75), "spread": 15,
                                    "height": rng.uniform(1.0, 1.8), "focal": rng.uniform(24, 50), "roadside": True})
        elif r < 0.9:
            spec = ShotSpec(f, f + L, loc, "extreme_wide", "drone", ego, params={"style": rng.choice(["follow", "flyby"]),
                                                                                  "altitude": rng.uniform(15, 40)})
        else:
            spec = ShotSpec(f, f + L, loc, "wide", "static", ego,
                            params={"spread": 15, "height": rng.uniform(1.2, 2.5), "focal": rng.uniform(20, 30),
                                    "roadside": True})
        shots.append(spec)
        f += L
        first = False
    return shots


def plan_voxel(n_frames, fps, rng, player, segments, extra_subjects=()):
    shots = []
    for loc, s0, s1 in segments:
        f = s0
        first = True
        while f < s1:
            L = int(round(np.clip(rng.lognormal(np.log(4.0), 0.35), 2.0, 7.0) * fps))
            if s1 - (f + L) < 1.5 * fps:
                L = s1 - f
            L = min(L, s1 - f)
            r = rng.random()
            if first or r < 0.6:
                spec = ShotSpec(f, f + L, loc, "medium", "fpv", player, params={"game": True, "fov": 70.0})
            elif r < 0.85:
                spec = ShotSpec(f, f + L, loc, "wide", "follow", player,
                                params={"azimuth": 180 + rng.normal(0, 20), "spread": 15, "height": rng.uniform(2.2, 3.0),
                                        "focal": 22})
            else:
                spec = ShotSpec(f, f + L, loc, "wide", "orbit", player, params={"azimuth": rng.uniform(-90, 90), "spread": 40,
                                                                                 "height": rng.uniform(1.6, 2.5)})
            shots.append(spec)
            f += L
            first = False
    return shots
