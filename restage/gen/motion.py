"""Procedural, physically plausible motion for Restage entities.

Humanoids are animated in layers, all vectorised over frames:

1. **Base layer** (legs and pelvis): a timeline of actions (stand, walk / run along a path,
   turn in place, sit down / stand up, dance stepping, jump). Every action produces foot
   *plants* (where and when a foot is on the ground) and a desired pelvis trajectory. Feet are
   placed by a footstep planner whose stride follows the speed, plants never move while in
   contact (heel strike / toe off pivot about fixed contact points), the pelvis is lowered
   wherever a leg could not reach its plant, and legs are solved with analytic two-bone IK.
2. **Upper layer**: spine lean and counter-rotation, breathing, arm swing, and overlays blended
   on top (talking gestures, waving, pointing, holding / reaching, dance arms), solved with
   two-bone IK for the arms in the chest frame.
3. **Head**: look-at targets distributed over neck and head, plus nods.

The result is an entity track in the format of docs/SCENE_FORMAT.md: root position [T,3],
root rotation [T,4] (heading), pose [T,22,4] (local joint rotations relative to the rest pose),
plus extras used by the generator (contacts, morph weights, world joint positions).

Vehicles follow road paths with a kinematic bicycle model (steering angle from curvature,
wheel spin from distance, body roll / pitch from accelerations). Objects: ballistic balls
with restitution and rolling, objects attached to a hand and set down, rolling objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

from ..scene import JID, OFFSETS, forward_kinematics, mat_to_quat, quat_axis_angle, quat_euler, quat_mul, \
    quat_rotate

G = 9.81

# ---------------------------------------------------------------- small math helpers


def minjerk(u):
    u = np.clip(u, 0, 1)
    return u ** 3 * (10 - 15 * u + 6 * u * u)


def smoothstep(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def bump(u):
    """Smooth bump on [-1, 1] (0 outside), peak 1 at 0."""
    u = np.clip(np.abs(u), 0, 1)
    return (1 - u * u) ** 2


def quat_conj(q):
    q = np.asarray(q, float)
    return np.concatenate([-q[..., :3], q[..., 3:]], -1)


def quat_norm(q):
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def quat_continuous(q, axis=0):
    """Flip signs along `axis` so consecutive quaternions are in the same hemisphere."""
    q = np.array(q, float)
    qm = np.moveaxis(q, axis, 0)
    for i in range(1, len(qm)):
        d = (qm[i] * qm[i - 1]).sum(-1, keepdims=True)
        qm[i] = np.where(d < 0, -qm[i], qm[i])
    return np.moveaxis(qm, 0, axis)


def quat_nlerp(a, b, t):
    """Normalised lerp with hemisphere alignment; t broadcasts (…, 1) or (…)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = np.asarray(t, float)
    if t.ndim == a.ndim - 1:
        t = t[..., None]
    d = (a * b).sum(-1, keepdims=True)
    b = np.where(d < 0, -b, b)
    return quat_norm(a * (1 - t) + b * t)


def quat_slerp(a, b, t):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = np.asarray(t, float)
    if t.ndim == a.ndim - 1:
        t = t[..., None]
    d = (a * b).sum(-1, keepdims=True)
    b = np.where(d < 0, -b, b)
    d = np.abs(d).clip(-1, 1)
    th = np.arccos(d)
    s = np.sin(th)
    small = s < 1e-5
    w0 = np.where(small, 1 - t, np.sin((1 - t) * th) / np.where(small, 1, s))
    w1 = np.where(small, t, np.sin(t * th) / np.where(small, 1, s))
    return quat_norm(a * w0 + b * w1)


def smooth_quats(q, sigma):
    """Gaussian smoothing of a quaternion sequence (time on axis 0)."""
    if sigma <= 0:
        return q
    q = quat_continuous(q)
    return quat_norm(gaussian_filter1d(q, sigma, axis=0, mode="nearest"))


def yaw_quat(yaw):
    yaw = np.asarray(yaw, float)
    return np.stack([np.zeros_like(yaw), np.sin(yaw / 2), np.zeros_like(yaw), np.cos(yaw / 2)], -1)


def yaw_of_dir(dx, dz):
    """Heading angle such that R_y(yaw) maps +Z onto (dx, ., dz)."""
    return np.arctan2(dx, dz)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def mat_from_frames(src, dst):
    """Rotation matrix mapping orthonormal frame src (…,3 cols) to dst (…,3 cols)."""
    return dst @ np.swapaxes(src, -1, -2)


def _normalize(v, fallback=(0.0, 0.0, 1.0)):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    fb = np.broadcast_to(np.asarray(fallback, float), v.shape)
    return np.where(n > 1e-8, v / np.maximum(n, 1e-12), fb)


def two_bone_ik(a, target, pole, l1, l2, bone_dir, bend_dir, reach=0.9995):
    """Analytic two-bone IK, vectorised over leading dims.

    a: upper joint world position; target: desired end position; pole: world direction toward
    which the middle joint should protrude; bone_dir / bend_dir: rest bone direction and rest
    protrusion direction (unit, perpendicular). Returns world rotation matrices of the upper and
    middle joints (relative to the rest pose, whose joint rotations are identity).
    """
    d = target - a
    dist = np.linalg.norm(d, axis=-1, keepdims=True)
    u = _normalize(d, (0, -1, 0))
    dc = np.clip(dist, abs(l1 - l2) + 1e-4, (l1 + l2) * reach)
    cos_a = np.clip((l1 * l1 + dc * dc - l2 * l2) / (2 * l1 * dc), -1, 1)
    sin_a = np.sqrt(1 - cos_a * cos_a)
    n = pole - (pole * u).sum(-1, keepdims=True) * u
    alt = np.cross(u, np.broadcast_to([1.0, 0, 0], u.shape))
    n = np.where(np.linalg.norm(n, axis=-1, keepdims=True) > 1e-6, n, alt)
    n = _normalize(n)
    upper = u * cos_a + n * sin_a
    mid = a + upper * l1
    end = a + u * dc
    lower = _normalize(end - mid)
    H = _normalize(np.cross(u, n))
    n1 = np.cross(H, upper)
    n2 = np.cross(H, lower)
    b = np.asarray(bone_dir, float)
    c = np.asarray(bend_dir, float)
    src = np.stack([b, c, np.cross(b, c)], -1)
    R1 = mat_from_frames(src, np.stack([upper, n1, H], -1))
    R2 = mat_from_frames(src, np.stack([lower, n2, H], -1))
    return R1, R2, end


# ---------------------------------------------------------------- paths

class Path:
    """Smooth ground path in XZ, parameterised by arc length (Catmull-Rom through waypoints)."""

    def __init__(self, points, smooth=True, spacing=0.05):
        pts = np.asarray(points, float)
        if pts.shape[1] == 3:
            pts = pts[:, [0, 2]]
        keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6])
        pts = pts[keep]
        if len(pts) == 1:
            pts = np.concatenate([pts, pts + [0, 1e-3]])
        if smooth and len(pts) > 2:
            pts = _catmull_rom(pts, 12)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        n = max(2, int(np.ceil(s[-1] / spacing)) + 1)
        ss = np.linspace(0, s[-1], n)
        self.p = np.stack([np.interp(ss, s, pts[:, 0]), np.interp(ss, s, pts[:, 1])], 1)
        self.s = ss
        self.length = float(ss[-1])
        d = np.gradient(self.p, axis=0)
        self._yaw = np.unwrap(yaw_of_dir(d[:, 0], d[:, 1]))

    def pos(self, s):
        s = np.clip(s, 0, self.length)
        return np.stack([np.interp(s, self.s, self.p[:, 0]), np.interp(s, self.s, self.p[:, 1])], -1)

    def heading(self, s):
        return np.interp(np.clip(s, 0, self.length), self.s, self._yaw)

    def curvature(self, s, ds=0.5):
        return (self.heading(s + ds) - self.heading(s - ds)) / (2 * ds)


def _catmull_rom(p, n_per):
    P = np.concatenate([2 * p[:1] - p[1:2], p, 2 * p[-1:] - p[-2:-1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        t = np.linspace(0, 1, n_per, endpoint=False)[:, None]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 +
                          (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(p[-1:])
    return np.concatenate(out)


def speed_profile(length, v, accel, fps):
    """Arc length s(t) for a smooth start-cruise-stop profile covering `length` exactly.

    v(t) = v * (S(t / r) - S((t - length / v) / r)) with S = smoothstep and ramp time r: the two
    ramps have the same shape, so the integral is exactly v * length / v = length, also for short
    walks where the ramps overlap and the peak speed stays below v.
    Returns (t, s, v) sampled at 4x the frame rate.
    """
    v = max(1e-3, float(v))
    r = max(0.35, v / accel * 1.5)
    t1 = length / v
    dt = 1.0 / fps / 4
    t = np.arange(0.0, t1 + r + dt, dt)
    vel = v * (smoothstep(t / r) - smoothstep((t - t1) / r))
    s = np.concatenate([[0.0], np.cumsum((vel[1:] + vel[:-1]) / 2 * dt)])
    s *= length / max(s[-1], 1e-9)
    return t, s, vel


# ---------------------------------------------------------------- humanoid body constants

@dataclass
class Body:
    scale: float = 1.0

    def __post_init__(self):
        s = self.scale
        self.off = OFFSETS.astype(np.float64) * s
        self.l_thigh = float(np.linalg.norm(self.off[JID["l_knee"]]))
        self.l_shin = float(np.linalg.norm(self.off[JID["l_ankle"]]))
        self.l_upper = float(np.linalg.norm(self.off[JID["l_elbow"]]))
        self.l_fore = float(np.linalg.norm(self.off[JID["l_wrist"]]))
        self.pelvis_h = float(self.off[0, 1])
        self.ankle_h = float(self.off[0, 1] + self.off[JID["l_hip"], 1] + self.off[JID["l_knee"], 1] +
                             self.off[JID["l_ankle"], 1])
        self.hip_x = float(self.off[JID["l_hip"], 0])
        self.leg = self.l_thigh + self.l_shin
        self.heel = np.array([0, -self.ankle_h, -0.06 * s])     # sole contact points in the foot frame
        self.ball = np.array([0, -self.ankle_h, 0.13 * s])
        self.eye = np.array([0, 0.115 * s, 0.09 * s])           # eye point in the head frame


@dataclass
class Plant:
    t_on: float
    t_off: float
    pos: np.ndarray              # ground point under the ankle (flat foot)
    yaw: float
    heel_strike: float = 0.0     # toe-up angle at touch-down (rad, >= 0)
    toe_off: float = 0.0         # heel-up angle at lift-off (rad, >= 0)
    lift: float = 0.08           # swing apex height above the straight line to the next plant
    ground_pitch: float = 0.0
    heel_raise: object = None    # optional callable t -> heel-up angle while planted (dance)


@dataclass
class Overlay:
    kind: str
    t0: float
    t1: float
    fade: float = 0.3
    params: dict = field(default_factory=dict)


GAITS = {
    # duty: fraction of a stride a foot is on the ground; width: lateral foot offset (x scale)
    "walk": dict(duty=0.62, width=0.07, heel=0.3, toe=0.55, lift=0.07, accel=1.2, sway=0.022, bob=0.018,
                 yaw=0.07, roll=0.05, lean=0.05),
    "run": dict(duty=0.36, width=0.04, heel=0.12, toe=0.7, lift=0.2, accel=3.0, sway=0.012, bob=0.035,
                yaw=0.1, roll=0.04, lean=0.16),
}


def step_frequency(v, gait, scale):
    if gait == "run":
        f = 2.55 + 0.12 * v
    else:
        f = 1.05 + 0.52 * v
    f = np.clip(f, 1.3 if gait == "walk" else 2.5, 2.3 if gait == "walk" else 3.4)
    return f / np.sqrt(scale)


DANCE_STYLES = ("bounce", "sway", "disco", "hiphop", "rave", "groove")


class HumanoidAnimator:
    """Builds a humanoid22 track from a timeline of actions. Times are in seconds."""

    def __init__(self, n_frames, fps=30, scale=1.0, rng=None, pos=(0.0, 0.0), yaw=0.0, ground=None,
                 energy=1.0):
        self.T, self.fps = int(n_frames), float(fps)
        self.b = Body(scale)
        self.s = scale
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.ground = ground if ground is not None else (lambda x, z: np.zeros(np.broadcast(x, z).shape))
        self.energy = energy
        self.t = 0.0
        self.yaw = float(yaw)
        self.center = np.array(pos, float)
        w = 0.085 * scale
        self.plants = {0: [], 1: []}   # 0 = left, 1 = right
        stagger = self.rng.uniform(-0.04, 0.06) * scale
        for foot, side in ((0, 1), (1, -1)):
            self.plants[foot].append(self._mk_plant(0.0, self.center, self.yaw, side * w,
                                                    stagger if foot == 0 else -stagger * 0.3))
        T = self.T
        self.pelvis = np.full((T, 3), np.nan)
        self.pelvis_yaw = np.full(T, np.nan)
        self.pelvis_rot = np.zeros((T, 3))       # pitch, yaw offset, roll
        self.lean = np.zeros(T)
        self.spine_extra = np.zeros((T, 3))      # pitch, yaw, roll summed over the spine
        self.phase = np.full(T, np.nan)
        self.gait_amp = np.zeros(T)
        self.run_w = np.zeros(T)
        self.sit_w = np.zeros(T)
        self.speed = np.zeros(T)
        self.ankle_override = {}                 # frame -> (foot, ankle pos) for jump flight
        self.overlays: list[Overlay] = []
        self.look: list[tuple] = []              # (t0, t1, target fn or point)
        self.speaking = np.zeros(T)
        self.mode = "stand"
        self.seat = None
        self._stand_from = 0.0

    # ------------------------------------------------------------ helpers
    def _f(self, t):
        return int(round(t * self.fps))

    def _frames(self, t0, t1):
        a, b = max(0, self._f(t0)), min(self.T, self._f(t1))
        return np.arange(a, b)

    def _ground_pitch(self, p, yaw):
        fwd = np.array([np.sin(yaw), np.cos(yaw)]) * 0.15
        h1 = float(self.ground(p[0] + fwd[0], p[2] + fwd[1]))
        h0 = float(self.ground(p[0] - fwd[0], p[2] - fwd[1]))
        return -np.arctan2(h1 - h0, 0.3)

    def _mk_plant(self, t_on, center_xz, yaw, lateral, forward=0.0, **kw):
        c = np.asarray(center_xz, float)
        x = c[0] + np.cos(yaw) * lateral + np.sin(yaw) * forward
        z = c[1] - np.sin(yaw) * lateral + np.cos(yaw) * forward
        y = float(self.ground(x, z))
        p = np.array([x, y, z])
        return Plant(t_on, np.inf, p, yaw, ground_pitch=self._ground_pitch(p, yaw), **kw)

    def _last(self, foot):
        return self.plants[foot][-1]

    def _close_stand(self, t1):
        """Write a standing pelvis for [self._stand_from, t1) (weight shifts, slight sway)."""
        fr = self._frames(self._stand_from, t1)
        if not len(fr):
            return
        if self.mode == "sit":
            self._write_seated(fr)
            return
        L, R = self._last(0).pos, self._last(1).pos
        mid = (L + R) / 2
        yaw = self.yaw
        t = fr / self.fps
        rng = self.rng
        # slow weight shifts between the feet every few seconds, plus a small sway
        k = np.zeros(len(fr))
        tt = t[0]
        while tt < t[-1] + 1:
            dur = rng.uniform(2.5, 6.0)
            target = rng.uniform(-0.6, 0.6)
            k += target * (smoothstep((t - tt) / 0.8) - smoothstep((t - tt - dur) / 0.8))
            tt += dur
        k = gaussian_filter1d(k, 6, mode="nearest")
        lat = np.array([np.cos(yaw), -np.sin(yaw)])
        half = (L - R)[[0, 2]] / 2
        sway = 0.012 * self.s * np.sin(2 * np.pi * t / rng.uniform(4, 7) + rng.uniform(0, 6))
        xz = mid[[0, 2]] + half[None] * k[:, None] * 0.35 + lat[None] * sway[:, None]
        self.pelvis[fr, 0] = xz[:, 0]
        self.pelvis[fr, 2] = xz[:, 1]
        self.pelvis[fr, 1] = mid[1] + self.b.pelvis_h - 0.012 * self.s - 0.006 * self.s * np.abs(k)
        self.pelvis_yaw[fr] = yaw
        self.pelvis_rot[fr, 2] = 0.05 * k
        self.pelvis_rot[fr, 1] = 0.04 * k
        self.lean[fr] = 0.02

    def _begin(self):
        """Finish the standing period before a new action."""
        self._close_stand(self.t)

    def _end(self, t_end, mode="stand"):
        self.t = t_end
        self._stand_from = t_end
        self.mode = mode

    # ------------------------------------------------------------ base actions
    def stand(self, duration):
        """Keep standing (weight shifts handled automatically)."""
        if self.mode == "sit":
            self.t += duration
            return self
        self.t += duration
        return self

    def wait(self, duration):
        self.t += duration
        return self

    def walk(self, waypoints, speed=1.3, gait="walk", end_speed=0.0, stop=True):
        """Walk or run through waypoints [(x, z), ...] starting from the current position."""
        if self.mode == "sit":
            self.stand_up()
        g = GAITS[gait]
        start = (self._last(0).pos + self._last(1).pos)[[0, 2]] / 2
        path = Path(np.concatenate([[start], np.asarray(waypoints, float)[:, [0, -1]]]))
        if path.length < 0.2:
            return self
        h0 = path.heading(min(0.4, path.length / 2))
        if abs(wrap(h0 - self.yaw)) > np.radians(35):
            self.turn_to(h0)
            start = (self._last(0).pos + self._last(1).pos)[[0, 2]] / 2
            path = Path(np.concatenate([[start], np.asarray(waypoints, float)[:, [0, -1]]]))
        self._begin()
        t0 = self.t
        ts, ss, vs = speed_profile(path.length, speed, g["accel"], self.fps)
        t_end = t0 + ts[-1]
        s_of = lambda t: np.interp(t - t0, ts, ss)
        v_of = lambda t: np.interp(t - t0, ts, vs)
        side = {0: 1, 1: -1}
        w = g["width"] * self.s
        # first foot: the one that is further behind relative to the walking direction
        d = np.array([np.sin(h0), np.cos(h0)])
        back = [float(np.dot(self._last(f).pos[[0, 2]] - start, d)) for f in (0, 1)]
        foot = 0 if back[0] <= back[1] else 1
        first = foot
        landings = []
        t_lift = t0 + 0.12
        f_now = step_frequency(max(speed * 0.5, 0.5), gait, self.s)
        P = 1.0 / f_now
        swing = (1 - g["duty"]) * 2 * P
        t_land = t_lift + swing
        prev_plant = self._last(foot)
        prev_plant.t_off = t_lift
        prev_plant.toe_off = 0.0
        n_steps = 0
        while True:
            v = max(v_of(t_land), 0.25 * speed)
            P = 1.0 / step_frequency(v, gait, self.s)
            stance = g["duty"] * 2 * P
            t_mid = t_land + stance / 2
            s_mid = s_of(t_mid)
            final = s_mid >= path.length - 0.25 * v * P or t_mid >= t_end
            if final:
                s_mid = path.length
            hd = path.heading(s_mid)
            xz = path.pos(s_mid)
            lateral = side[foot] * (w if not final else 0.085 * self.s)
            pl = self._mk_plant(t_land, xz, hd, lateral, heel_strike=g["heel"] * min(1, v / 1.2),
                                lift=g["lift"] * self.s * (0.6 + 0.4 * min(1, v / max(speed, 1e-3))))
            prev = self._last(foot)
            pl.toe_off = 0.0
            prev.toe_off = g["toe"] * min(1, v / 1.2) if np.linalg.norm(pl.pos - prev.pos) > 0.15 else 0.0
            if np.linalg.norm(pl.pos - prev.pos) < 0.12:
                pl.heel_strike = 0.0
            self.plants[foot].append(pl)
            landings.append((t_land, foot))
            n_steps += 1
            other = 1 - foot
            # other foot lifts so that it lands one step period later
            v2 = max(v_of(t_land + P), 0.25 * speed)
            P2 = 1.0 / step_frequency(v2, gait, self.s)
            stance2 = g["duty"] * 2 * P2
            swing2 = 2 * P2 - stance2
            t_land_next = t_land + P
            t_lift_other = max(t_land_next - swing2, self._last(other).t_on + 0.15)
            self._last(other).t_off = t_lift_other
            pl.t_off = np.inf
            if final:
                # closing step of the other foot next to this one
                hd_end = hd
                cl = self._mk_plant(t_land_next, path.pos(path.length), hd_end, side[other] * 0.085 * self.s,
                                    lift=0.05 * self.s)
                prev_o = self._last(other)
                if np.linalg.norm(cl.pos - prev_o.pos) < 0.04:
                    prev_o.t_off = np.inf
                else:
                    prev_o.toe_off = 0.25 if np.linalg.norm(cl.pos - prev_o.pos) > 0.3 else 0.0
                    self.plants[other].append(cl)
                    landings.append((t_land_next, other))
                t_done = max(t_land_next + 0.15, t_end)
                break
            foot = other
            t_land = t_land_next
            if n_steps > 400:
                t_done = t_land
                break
        # pelvis trajectory
        fr = self._frames(t0, t_done)
        t = fr / self.fps
        s_t = s_of(t)
        xz = path.pos(s_t)
        yaw = path.heading(s_t)
        v_t = v_of(t)
        lt = np.array([l[0] for l in landings])
        lv = np.array([n + (0 if first == 0 else 1) for n in range(len(landings))], float)
        if len(lt) >= 2:
            per = lt[1] - lt[0]
            lt = np.concatenate([[lt[0] - per], lt, [lt[-1] + per]])
            lv = np.concatenate([[lv[0] - 1], lv, [lv[-1] + 1]])
        phase = np.interp(t, lt, lv)
        amp = np.clip(v_t / max(speed, 1e-3), 0, 1) * smoothstep((t - t0) / 0.3) * \
            (1 - smoothstep((t - (t_done - 0.35)) / 0.35))
        duty = g["duty"]
        lat = np.stack([np.cos(yaw), -np.sin(yaw)], 1)
        sway = g["sway"] * self.s * np.cos(np.pi * (phase - duty)) * amp
        xz = xz + lat * sway[:, None]
        if gait == "run":
            bob = -g["bob"] * np.cos(2 * np.pi * (phase - duty / 2)) * amp
            base_h = self.b.pelvis_h - 0.05 * self.s * amp
        else:
            bob = g["bob"] * np.cos(2 * np.pi * (phase - duty / 2)) * amp
            base_h = self.b.pelvis_h - 0.015 * self.s - 0.01 * amp * self.s
        gy = self.ground(xz[:, 0], xz[:, 1])
        # support height: follow the planted feet rather than the terrain under the pelvis
        self.pelvis[fr, 0], self.pelvis[fr, 2] = xz[:, 0], xz[:, 1]
        self.pelvis[fr, 1] = gaussian_filter1d(gy, 4, mode="nearest") + base_h + bob * self.s
        self.pelvis_yaw[fr] = yaw
        self.pelvis_rot[fr, 1] = -g["yaw"] * np.cos(np.pi * phase) * amp
        self.pelvis_rot[fr, 2] = g["roll"] * np.cos(np.pi * (phase - duty)) * amp
        self.pelvis_rot[fr, 0] = 0.03 * amp
        self.lean[fr] = (g["lean"] + 0.03 * np.clip(v_t, 0, 5)) * amp + 0.02
        self.phase[fr] = phase
        self.gait_amp[fr] = amp * (0.6 + 0.4 * min(1.0, speed / 1.4))
        self.run_w[fr] = (1.0 if gait == "run" else 0.0) * smoothstep(amp * 1.5)
        self.speed[fr] = v_t
        self.yaw = float(path.heading(path.length))
        self._end(t_done)
        return self

    def turn_to(self, yaw, step_time=0.38):
        """Turn in place to face `yaw` with small stepping."""
        if self.mode == "sit":
            return self
        d = wrap(yaw - self.yaw)
        if abs(d) < np.radians(8):
            self.yaw = self.yaw + d
            return self
        self._begin()
        t0 = self.t
        center = (self._last(0).pos + self._last(1).pos)[[0, 2]] / 2
        M = max(2, int(np.ceil(abs(d) / np.radians(70))) + 1)
        lead = 0 if d > 0 else 1   # turning left: left foot first
        side = {0: 1, 1: -1}
        swing = step_time * 0.75
        t = t0 + 0.05
        for i in range(1, M + 1):
            foot = lead if i % 2 == 1 else 1 - lead
            y = self.yaw + d * min(1.0, i / (M - 1))
            prev = self._last(foot)
            prev.t_off = t
            pl = self._mk_plant(t + swing, center, y, side[foot] * 0.085 * self.s, lift=0.045 * self.s)
            self.plants[foot].append(pl)
            t += step_time
        t_end = t + swing - step_time + 0.15
        fr = self._frames(t0, t_end)
        u = minjerk((fr / self.fps - t0) / max(t_end - t0 - 0.1, 0.1))
        self.pelvis[fr, 0], self.pelvis[fr, 2] = center[0], center[1]
        self.pelvis[fr, 1] = float(self.ground(center[0], center[1])) + self.b.pelvis_h - 0.02 * self.s
        self.pelvis_yaw[fr] = self.yaw + d * u
        self.lean[fr] = 0.03
        self.yaw = self.yaw + d
        self._end(t_end)
        return self

    def face(self, point):
        c = (self._last(0).pos + self._last(1).pos) / 2
        return self.turn_to(float(yaw_of_dir(point[0] - c[0], point[-1] - c[2])))

    def sit(self, seat, seat_yaw, duration=1.4, approach=True):
        """Sit on a seat (world point of the sitting surface) facing seat_yaw.

        If `approach`, first walk to stand in front of the seat and turn around.
        """
        seat = np.asarray(seat, float)
        fwd = np.array([np.sin(seat_yaw), np.cos(seat_yaw)])
        stand_xz = seat[[0, 2]] + fwd * 0.4 * self.s
        if approach:
            cur = (self._last(0).pos + self._last(1).pos)[[0, 2]] / 2
            if np.linalg.norm(cur - stand_xz) > 0.25:
                self.walk([stand_xz], speed=1.0)
            self.turn_to(seat_yaw)
        self._begin()
        t0 = self.t
        fr = self._frames(t0, t0 + duration)
        u = (fr / self.fps - t0) / duration
        start = self.pelvis[fr[0] - 1] if fr[0] > 0 and not np.isnan(self.pelvis[fr[0] - 1, 0]) else \
            np.array([stand_xz[0], seat[1] * 0 + self.b.pelvis_h, stand_xz[1]])
        seat_pel = np.array([seat[0] - fwd[0] * 0.05, seat[1] + 0.135 * self.s, seat[2] - fwd[1] * 0.05])
        back = minjerk(u / 0.9)
        down = minjerk((u - 0.08) / 0.92)
        self.pelvis[fr] = start + (seat_pel - start) * np.stack([back, down, back], 1)
        self.pelvis_yaw[fr] = seat_yaw
        self.lean[fr] = 0.75 * np.sin(np.pi * np.clip(u / 0.95, 0, 1)) ** 1.3 + 0.08 * u
        self.pelvis_rot[fr, 0] = 0.25 * np.sin(np.pi * np.clip(u, 0, 1)) - 0.12 * smoothstep(u)
        self.sit_w[fr] = smoothstep((u - 0.3) / 0.7)
        self.seat = (seat_pel, seat_yaw)
        self.yaw = seat_yaw
        # feet: plant both feet comfortably in front of the seat
        self._end(t0 + duration, mode="sit")
        return self

    def _write_seated(self, fr):
        seat_pel, yaw = self.seat
        t = fr / self.fps
        self.pelvis[fr] = seat_pel + np.array([0, 0.004, 0]) * np.sin(2 * np.pi * t / 4.0)[:, None]
        self.pelvis_yaw[fr] = yaw
        self.pelvis_rot[fr, 0] = -0.12
        self.lean[fr] = 0.08 + 0.03 * np.sin(2 * np.pi * t / 9.0)
        self.sit_w[fr] = 1.0

    def sit_still(self, seat, seat_yaw):
        """Start the animation already seated (feet planted in front of the seat)."""
        seat = np.asarray(seat, float)
        fwd = np.array([np.sin(seat_yaw), np.cos(seat_yaw)])
        foot_xz = seat[[0, 2]] + fwd * 0.42 * self.s
        for foot, side in ((0, 1), (1, -1)):
            self.plants[foot][-1] = self._mk_plant(0.0, foot_xz, seat_yaw, side * 0.11 * self.s,
                                                   forward=self.rng.uniform(-0.05, 0.05))
        seat_pel = np.array([seat[0] - fwd[0] * 0.05, seat[1] + 0.135 * self.s, seat[2] - fwd[1] * 0.05])
        self.seat = (seat_pel, seat_yaw)
        self.yaw = seat_yaw
        self.mode = "sit"
        self._stand_from = self.t
        return self

    def stand_up(self, duration=1.3):
        if self.mode != "sit":
            return self
        self._close_stand(self.t)
        t0 = self.t
        fr = self._frames(t0, t0 + duration)
        u = (fr / self.fps - t0) / duration
        seat_pel, yaw = self.seat
        L, R = self._last(0).pos, self._last(1).pos
        mid = (L + R) / 2
        end = np.array([mid[0], mid[1] + self.b.pelvis_h - 0.012 * self.s, mid[2]])
        fwd = minjerk((u - 0.1) / 0.8)
        up = minjerk((u - 0.2) / 0.8)
        self.pelvis[fr] = seat_pel + (end - seat_pel) * np.stack([fwd, up, fwd], 1)
        self.pelvis_yaw[fr] = yaw
        self.lean[fr] = 0.8 * np.sin(np.pi * np.clip(u / 0.9, 0, 1)) ** 1.1 + 0.02
        self.pelvis_rot[fr, 0] = 0.2 * np.sin(np.pi * np.clip(u, 0, 1)) - 0.12 * (1 - smoothstep(u))
        self.sit_w[fr] = 1 - smoothstep(u / 0.5)
        self.seat = None
        self._end(t0 + duration, mode="stand")
        return self

    def jump(self, height=0.3, forward=0.0):
        """Counter-movement jump in place (or forward), with crouch, flight and landing."""
        if self.mode == "sit":
            self.stand_up()
        self._begin()
        t0 = self.t
        crouch, push = 0.32, 0.16
        v0 = np.sqrt(2 * G * height)
        flight = 2 * v0 / G
        land = 0.45
        t_take = t0 + crouch + push
        t_land = t_take + flight
        t_end = t_land + land
        L0, R0 = self._last(0), self._last(1)
        c0 = (L0.pos + R0.pos) / 2
        fwd = np.array([np.sin(self.yaw), 0, np.cos(self.yaw)])
        c1 = c0 + fwd * forward
        stand_h = self.b.pelvis_h - 0.012 * self.s
        fr = self._frames(t0, t_end)
        t = fr / self.fps
        y = np.zeros(len(fr))
        x = np.zeros((len(fr), 3))
        dip = 0.16 * self.s
        ext = self.b.pelvis_h - 0.01 * self.b.leg + 0.04 * self.s   # pelvis height at full extension, heels up
        for i, tt in enumerate(t):
            if tt < t0 + crouch:
                u = (tt - t0) / crouch
                y[i] = stand_h - dip * minjerk(u)
                x[i] = c0
            elif tt < t_take:
                u = (tt - t0 - crouch) / push
                y[i] = stand_h - dip + (ext + 0.02 * self.s - stand_h + dip) * smoothstep(u)
                x[i] = c0 + (c1 - c0) * 0.05 * u
            elif tt < t_land:
                u = tt - t_take
                y[i] = ext + 0.02 * self.s + v0 * u - 0.5 * G * u * u
                x[i] = c0 + (c1 - c0) * (0.05 + 0.9 * u / flight)
            else:
                u = (tt - t_land) / land
                y[i] = stand_h + (ext + 0.02 * self.s - stand_h) * (1 - smoothstep(u / 0.25)) - \
                    0.13 * self.s * bump((u - 0.35) / 0.35)
                x[i] = c1
        y = gaussian_filter1d(y, 1.0, mode="nearest")
        self.pelvis[fr, 0] = x[:, 0]
        self.pelvis[fr, 2] = x[:, 2]
        self.pelvis[fr, 1] = y + c0[1]
        self.pelvis_yaw[fr] = self.yaw
        lean = np.where(t < t_take, 0.45 * np.sin(np.pi * np.clip((t - t0) / (crouch + push), 0, 1)),
                        0.25 * bump((t - t_land - 0.1) / 0.35))
        self.lean[fr] = lean
        self.overlays.append(Overlay("jump_arms", t0, t_end, 0.1, {"t_take": t_take, "t_land": t_land}))
        # feet: lift at take-off, land at t_land
        for foot, pl in ((0, L0), (1, R0)):
            pl.t_off = t_take
            pl.toe_off = 0.35
            new = Plant(t_land, np.inf, pl.pos + (c1 - c0), pl.yaw, heel_strike=0.0, lift=0.0,
                        ground_pitch=pl.ground_pitch)
            self.plants[foot].append(new)
        self._jumps = getattr(self, "_jumps", []) + [(t_take, t_land, c0, c1)]
        self._end(t_end)
        return self

    def dance(self, duration, style="bounce", bpm=110.0, beat0=0.0, center=None, facing=None):
        """Dance to a tempo: stepping on beats, pelvis bounce, style-specific arms."""
        if self.mode == "sit":
            self.stand_up()
        if center is not None and np.linalg.norm(np.asarray(center)[[0, -1]] - (self._last(0).pos + self._last(1).pos)[[0, 2]] / 2) > 0.3:
            self.walk([np.asarray(center)[[0, -1]]], speed=1.1)
        if facing is not None:
            self.turn_to(facing)
        self._begin()
        t0 = self.t
        t1 = t0 + duration
        beat = 60.0 / bpm
        rng = self.rng
        c = (self._last(0).pos + self._last(1).pos) / 2
        yaw = self.yaw
        lat = np.array([np.cos(yaw), 0, -np.sin(yaw)])
        # beats after t0 aligned with the global beat grid
        k0 = int(np.ceil((t0 + 0.3 - beat0) / beat))
        beats = beat0 + beat * np.arange(k0, int((t1 - beat0) / beat))
        width = 0.11 * self.s
        pattern = {"bounce": "touch", "sway": "side", "disco": "touch", "hiphop": "rock", "rave": "touch",
                   "groove": "side"}[style]
        step_len = {"touch": 0.28, "side": 0.38, "rock": 0.25}[pattern] * self.s
        offset = 0.0
        # stepping: one foot moves on (most) beats
        turn_bar = rng.integers(2, 4) if style in ("disco", "groove", "hiphop") else 99
        i = 0
        center_track = []
        while i < len(beats) - 1:
            tb = beats[i]
            bar = i // 4
            if bar and bar % turn_bar == 0 and i % 4 == 0 and i + 4 < len(beats):
                # a quick turn over one bar, stepping on every beat
                for kk in range(1, 5):
                    foot = (kk + 1) % 2
                    yy = yaw + 2 * np.pi * min(1.0, kk / 3.0)
                    prev = self._last(foot)
                    prev.t_off = beats[i + kk - 1] - beat * 0.45
                    cc = c + lat * offset
                    pl = self._mk_plant(beats[i + kk - 1], cc[[0, 2]], yy, (1 if foot == 0 else -1) * width,
                                        lift=0.06 * self.s)
                    self.plants[foot].append(pl)
                center_track.append((beats[i], offset, 1))
                i += 4
                continue
            phase4 = i % 4
            if pattern == "touch" or pattern == "side":
                direction = 1 if (i // 2) % 2 == 0 else -1
                foot = 0 if direction > 0 else 1
                if phase4 % 2 == 1:
                    foot = 1 - foot
                if phase4 % 2 == 0:
                    offset += direction * step_len
                fpos_lat = offset + (1 if foot == 0 else -1) * width * (1.0 if phase4 % 2 == 0 else 0.55)
                fwd_off = 0.0
            else:   # rock: step forward and back alternately
                foot = i % 2
                fwd_off = (0.22 * self.s if (i // 2) % 2 == 0 else -0.12 * self.s) if phase4 % 2 == 0 else 0.0
                fpos_lat = offset + (1 if foot == 0 else -1) * width
            prev = self._last(foot)
            prev.t_off = tb - beat * 0.45
            pl = self._mk_plant(tb, c[[0, 2]], yaw, fpos_lat, forward=fwd_off, lift=0.07 * self.s)
            if style in ("bounce", "rave", "hiphop"):
                ph = tb
                pl.heel_raise = (lambda tt, ph=ph: 0.25 * bump((tt - ph - beat * 0.5) / (beat * 0.5)))
            self.plants[foot].append(pl)
            center_track.append((tb, offset, 0))
            i += 1
        # close the feet back to a neutral stance at the end
        t_close = beats[-1] if len(beats) else t0 + 0.5
        for foot in (0, 1):
            prev = self._last(foot)
            prev.t_off = t_close + foot * 0.3
            pl = self._mk_plant(t_close + foot * 0.3 + 0.3, (c + lat * offset)[[0, 2]], yaw,
                                (1 if foot == 0 else -1) * 0.085 * self.s, lift=0.05 * self.s)
            self.plants[foot].append(pl)
        t_end = max(t1, t_close + 0.8)
        fr = self._frames(t0, t_end)
        t = fr / self.fps
        # pelvis follows the support centre (smoothed), bounces down on the beat
        ct = np.array([[tt, off] for tt, off, _ in center_track]) if center_track else np.array([[t0, 0.0]])
        off_t = np.interp(t, ct[:, 0], ct[:, 1])
        off_t = gaussian_filter1d(off_t, 0.18 * self.fps, mode="nearest")
        en = self.energy
        bphase = (t - beat0) / beat
        amp_in = smoothstep((t - t0) / 0.6) * (1 - smoothstep((t - t_close) / 0.6))
        bounce = {"bounce": 0.045, "sway": 0.02, "disco": 0.03, "hiphop": 0.055, "rave": 0.06, "groove": 0.035}[style]
        bounce *= self.s * en * amp_in
        down = 0.5 + 0.5 * np.cos(2 * np.pi * bphase)
        sway = {"bounce": 0.02, "sway": 0.06, "disco": 0.04, "hiphop": 0.03, "rave": 0.02, "groove": 0.05}[style]
        sw = sway * self.s * np.sin(np.pi * bphase) * amp_in
        pos = c[None] + lat[None] * (off_t + sw)[:, None]
        self.pelvis[fr, 0], self.pelvis[fr, 2] = pos[:, 0], pos[:, 2]
        self.pelvis[fr, 1] = c[1] + self.b.pelvis_h - 0.035 * self.s * amp_in - bounce * down
        turn = np.zeros(len(fr))
        for tb, _, is_turn in center_track:
            if is_turn:
                turn += 2 * np.pi * minjerk((t - tb + beat * 0.3) / (beat * 3.2))
        self.pelvis_yaw[fr] = yaw + turn
        self.pelvis_rot[fr, 2] = 0.12 * np.sin(np.pi * bphase) * amp_in * (sway / 0.04)
        self.pelvis_rot[fr, 1] = 0.08 * np.sin(np.pi * bphase + 0.5) * amp_in
        self.pelvis_rot[fr, 0] = 0.06 * down * amp_in
        self.lean[fr] = 0.05 + 0.08 * down * amp_in * (style in ("hiphop", "rave"))
        self.phase[fr] = np.nan
        self.overlays.append(Overlay("dance_arms", t0, t_end, 0.4, {"style": style, "bpm": bpm, "beat0": beat0,
                                                                     "energy": en}))
        self.overlays.append(Overlay("dance_head", t0, t_end, 0.4, {"style": style, "bpm": bpm, "beat0": beat0}))
        self.yaw = yaw + (turn[-1] if len(turn) else 0.0)
        self._end(t_end)
        return self

    # ------------------------------------------------------------ upper-body overlays
    def talk(self, t0, t1, intensity=1.0, rest="auto", table_y=None):
        """Talking: speech flag (mouth), beat and illustrative hand gestures, head nods, shoulder motion."""
        self.overlays.append(Overlay("talk", t0, t1, 0.3, {"intensity": intensity, "rest": rest, "table_y": table_y,
                                                            "seed": int(self.rng.integers(1 << 30))}))
        fr = self._frames(t0, t1)
        self.speaking[fr] = 1.0
        return self

    def listen(self, t0, t1):
        self.overlays.append(Overlay("listen", t0, t1, 0.4, {"seed": int(self.rng.integers(1 << 30))}))
        return self

    def wave(self, t0, duration=2.0, side="r"):
        self.overlays.append(Overlay("wave", t0, t0 + duration, 0.35, {"side": side}))
        return self

    def point(self, t0, duration, target, side="r"):
        self.overlays.append(Overlay("point", t0, t0 + duration, 0.35, {"side": side, "target": np.asarray(target, float)}))
        return self

    def hold(self, t0, t1, side="r", kind="carry"):
        """Hold an object in one hand ('carry' at waist, 'phone' in front of the chest)."""
        self.overlays.append(Overlay("hold", t0, t1, 0.35, {"side": side, "kind": kind}))
        return self

    def reach(self, t_at, target, side="r", duration=1.2):
        """Reach so the hand is at `target` (world) at time t_at."""
        self.overlays.append(Overlay("reach", t_at - duration / 2, t_at + duration / 2, duration * 0.4,
                                     {"side": side, "target": np.asarray(target, float), "t_at": t_at}))
        return self

    def drink(self, t0, duration=2.5, side="r"):
        self.overlays.append(Overlay("drink", t0, t0 + duration, 0.6, {"side": side}))
        return self

    def look_at(self, t0, t1, target):
        """Look at a world point or a callable frame -> point during [t0, t1)."""
        self.look.append((t0, t1, target))
        return self

    # ------------------------------------------------------------ build
    def build(self) -> dict:
        self._close_stand(max(self.t, self.T / self.fps + 0.1))
        T, b, s = self.T, self.b, self.s
        t = np.arange(T) / self.fps
        # fill pelvis gaps (should not happen) by holding
        for arr in (self.pelvis, self.pelvis_yaw):
            m = np.isnan(arr.reshape(T, -1)[:, 0])
            if m.all():
                arr[:] = 0
            elif m.any():
                idx = np.where(~m)[0]
                fill = np.interp(np.arange(T), idx, np.arange(len(idx)))
                arr[m] = arr[idx[np.round(fill[m]).astype(int)]]
        yaw = gaussian_filter1d(np.unwrap(self.pelvis_yaw), 2.0, mode="nearest")
        ankle, frot, toe, contact = self._feet(t)
        pel = self.pelvis.copy()
        pel[:, [0, 2]] = gaussian_filter1d(pel[:, [0, 2]], 1.5, axis=0, mode="nearest")
        pel[:, 1] = gaussian_filter1d(pel[:, 1], 1.2, mode="nearest")
        prot = gaussian_filter1d(self.pelvis_rot, 2.0, axis=0, mode="nearest")
        pel_q = quat_euler(prot[:, 0], yaw + prot[:, 1], prot[:, 2])
        # jump flight: feet follow the pelvis with tucked legs
        for (t_take, t_land, c0, c1) in getattr(self, "_jumps", []):
            fr = np.where((t > t_take) & (t < t_land))[0]
            if not len(fr):
                continue
            u = (t[fr] - t_take) / (t_land - t_take)
            for foot, sx in ((0, 1), (1, -1)):
                hip = pel[fr] + quat_rotate(pel_q[fr], np.broadcast_to(b.off[JID["l_hip" if foot == 0 else "r_hip"]],
                                                                         (len(fr), 3)))
                tuck = 0.12 * s * np.sin(np.pi * u)
                lo = hip[:, 1] - b.leg * 0.985 + tuck
                start = ankle[fr[0] - 1, foot] if fr[0] > 0 else ankle[fr[0], foot]
                end = ankle[min(fr[-1] + 1, T - 1), foot]
                xz = start + (end - start) * minjerk(u)[:, None]
                ankle[fr, foot, 0], ankle[fr, foot, 2] = xz[:, 0], xz[:, 2]
                ankle[fr, foot, 1] = np.maximum(lo, xz[:, 1])
        # lower the pelvis wherever a leg cannot reach its foot target
        for it in range(2):
            excess = np.zeros(T)
            for foot, hipj in ((0, "l_hip"), (1, "r_hip")):
                hip = pel + quat_rotate(pel_q, np.broadcast_to(b.off[JID[hipj]], (T, 3)))
                d_xz = np.linalg.norm((hip - ankle[:, foot])[:, [0, 2]], axis=1)
                lmax = b.leg * 0.992
                y_max = ankle[:, foot, 1] + np.sqrt(np.maximum(lmax * lmax - d_xz * d_xz, 0.0))
                excess = np.maximum(excess, hip[:, 1] - y_max)
            excess = np.maximum(excess, 0)
            if excess.max() < 1e-4:
                break
            ex = gaussian_filter1d(maximum_filter1d(excess, 9, mode="nearest"), 1.6, mode="nearest")
            pel[:, 1] -= ex + 0.002 * (ex > 0)
        # ---- pose
        pose = np.zeros((T, 22, 4))
        pose[..., 3] = 1
        root_q = yaw_quat(yaw)
        root_pos = pel - quat_rotate(root_q, np.broadcast_to(b.off[0], (T, 3)))
        pose[:, 0] = quat_mul(quat_conj(root_q), pel_q)
        # spine: lean, counter-rotation to the pelvis twist, breathing
        lean = gaussian_filter1d(self.lean, 2.0, mode="nearest")
        breath = 0.012 * np.sin(2 * np.pi * t / 4.2 + self.rng.uniform(0, 6))
        spine = np.stack([lean + breath * 0.5 - prot[:, 0] * 0.8, -prot[:, 1] * 0.9, -prot[:, 2] * 0.7], 1)
        spine_add = self._spine_overlays(t)
        spine = spine + spine_add + self.spine_extra
        for j, w in (("spine1", 0.3), ("spine2", 0.3), ("chest", 0.4)):
            pose[:, JID[j]] = quat_euler(spine[:, 0] * w, spine[:, 1] * w, spine[:, 2] * w)
        pose[:, JID["chest"]] = quat_mul(pose[:, JID["chest"]], quat_euler(-breath * 0.6, 0, 0))
        # legs
        self._solve_legs(pose, root_pos, root_q, ankle, frot, toe)
        # arms: default then overlays
        self._solve_arms(pose, root_pos, root_q, t)
        # head
        self._solve_head(pose, root_pos, root_q, t)
        pos_w, rot_w = forward_kinematics(pose, root_pos, root_q, b.off)
        jaw = self._mouth(t)
        return {"position": root_pos.astype(np.float32), "rotation": root_q.astype(np.float32),
                "pose": quat_continuous(pose).astype(np.float32), "contacts": contact.astype(np.uint8),
                "joints_world": pos_w.astype(np.float32), "joint_rot_world": rot_w.astype(np.float32),
                "morph": jaw[:, None].astype(np.float32), "morph_names": ["jawOpen"], "scale_skel": s,
                "foot_targets": ankle.astype(np.float32), "speaking": self.speaking.copy()}

    # ------------------------------------------------------------ feet from plants
    def _feet(self, t):
        T, b = self.T, self.b
        ankle = np.zeros((T, 2, 3))
        pitch = np.zeros((T, 2))
        fyaw = np.zeros((T, 2))
        toe = np.zeros((T, 2))
        contact = np.zeros((T, 2), bool)
        for foot in (0, 1):
            pls = self.plants[foot]
            # sort and drop zero-length plants
            for i, pl in enumerate(pls):
                if i + 1 < len(pls):
                    pl.t_off = min(pl.t_off, pls[i + 1].t_on - 0.08)
            for i, pl in enumerate(pls):
                t_off = pl.t_off if np.isfinite(pl.t_off) else 1e9
                m = (t >= pl.t_on) & (t <= t_off)
                if i == 0:
                    m |= t < pl.t_on
                tt = t[m]
                th, piv = self._roll(pl, tt)
                ankle[m, foot], pitch[m, foot] = self._ankle_from_roll(pl, th, piv)
                fyaw[m, foot] = pl.yaw
                toe[m, foot] = -np.maximum(th, 0) * 0.9
                contact[m, foot] = True
                if i + 1 < len(pls) and np.isfinite(pl.t_off):
                    nx = pls[i + 1]
                    ms = (t > t_off) & (t < nx.t_on)
                    if not ms.any():
                        continue
                    a0, p0 = self._ankle_from_roll(pl, *self._roll(pl, np.array([t_off])))
                    a1, p1 = self._ankle_from_roll(nx, *self._roll(nx, np.array([nx.t_on])))
                    u = (t[ms] - t_off) / max(nx.t_on - t_off, 1e-3)
                    k = minjerk(u)[:, None]
                    pos = a0 + (a1 - a0) * k
                    dist = float(np.linalg.norm((a1 - a0)[0, [0, 2]]))
                    lift = nx.lift if dist > 0.05 else min(nx.lift, 0.03)
                    lift = lift * min(1.0, 0.4 + dist / 0.5)
                    pos[:, 1] += lift * np.sin(np.pi * np.clip(u, 0, 1) ** 0.85)
                    # never through the ground
                    gy = self.ground(pos[:, 0], pos[:, 2]) + b.ankle_h * 0.9
                    pos[:, 1] = np.maximum(pos[:, 1], gy)
                    ankle[ms, foot] = pos
                    dy = wrap(nx.yaw - pl.yaw)
                    fyaw[ms, foot] = pl.yaw + dy * minjerk(u)
                    pitch[ms, foot] = p0[0] + (p1[0] - p0[0]) * smoothstep(u)
                    toe[ms, foot] = 0.0
        frot = quat_euler(pitch, fyaw, np.zeros_like(pitch))
        return ankle, frot, toe, contact

    def _roll(self, pl, tt):
        """Foot pitch (positive = heel up) and pivot for a plant at times tt."""
        th = np.zeros(len(tt))
        piv = np.zeros(len(tt), int)   # 0 heel / flat, 1 ball
        if pl.heel_strike > 0:
            u = (tt - pl.t_on) / 0.12
            th = np.where(u < 1, -pl.heel_strike * (1 - smoothstep(u)), th)
        if pl.toe_off > 0 and np.isfinite(pl.t_off):
            dur = min(0.28, max(0.08, (pl.t_off - pl.t_on) * 0.4))
            u = (tt - (pl.t_off - dur)) / dur
            m = u > 0
            th = np.where(m, pl.toe_off * smoothstep(u), th)
            piv = np.where(m, 1, piv)
        if pl.heel_raise is not None:
            hr = np.array([pl.heel_raise(x) for x in tt]) if len(tt) else np.zeros(0)
            raise_m = (piv == 0) & (th >= 0)
            th = np.where(raise_m, np.maximum(th, hr), th)
            piv = np.where(raise_m & (hr > 1e-3), 1, piv)
        return th, piv

    def _ankle_from_roll(self, pl, th, piv):
        b = self.b
        qy = yaw_quat(np.full(len(th), pl.yaw))
        gp = pl.ground_pitch
        qg = quat_mul(qy, quat_euler(np.full(len(th), gp), 0, 0))
        q = quat_mul(qy, quat_euler(gp + th, 0, 0))
        heel_c = pl.pos + quat_rotate(qg, np.broadcast_to(b.heel + [0, b.ankle_h, 0], (len(th), 3)))
        ball_c = pl.pos + quat_rotate(qg, np.broadcast_to(b.ball + [0, b.ankle_h, 0], (len(th), 3)))
        a_heel = heel_c + quat_rotate(q, np.broadcast_to(-b.heel, (len(th), 3)))
        a_ball = ball_c + quat_rotate(q, np.broadcast_to(-b.ball, (len(th), 3)))
        a = np.where((piv == 1)[:, None], a_ball, a_heel)
        return a, gp + th

    # ------------------------------------------------------------ IK solvers
    def _solve_legs(self, pose, root_pos, root_q, ankle, frot, toe):
        b = self.b
        T = self.T
        pos_w, rot_w = forward_kinematics(pose, root_pos, root_q, b.off)
        pel_q = rot_w[:, 0]
        for foot, (hip, knee, ank, toej) in enumerate((("l_hip", "l_knee", "l_ankle", "l_toe"),
                                                       ("r_hip", "r_knee", "r_ankle", "r_toe"))):
            a = pos_w[:, JID[hip]]
            fwd_foot = quat_rotate(frot[:, foot], np.broadcast_to([0, 0, 1.0], (T, 3)))
            fwd_pel = quat_rotate(pel_q, np.broadcast_to([0, 0, 1.0], (T, 3)))
            side = 1 if foot == 0 else -1
            out = quat_rotate(pel_q, np.broadcast_to([side * 0.15, 0, 0], (T, 3)))
            pole = _normalize(fwd_foot * 0.6 + fwd_pel * 0.4 + out)
            R1, R2, _ = two_bone_ik(a, ankle[:, foot], pole, b.l_thigh, b.l_shin, [0, -1, 0], [0, 0, 1])
            q1, q2 = mat_to_quat(R1), mat_to_quat(R2)
            pose[:, JID[hip]] = quat_mul(quat_conj(pel_q), q1)
            pose[:, JID[knee]] = quat_mul(quat_conj(q1), q2)
            pose[:, JID[ank]] = quat_mul(quat_conj(q2), frot[:, foot])
            pose[:, JID[toej]] = quat_euler(toe[:, foot], 0, 0)

    def _arm_ik(self, pose, root_pos, root_q, side, target_w, pole_w, twist=None, wrist=None, frames=None):
        """Solve one arm for world-space hand targets; returns local quats for shoulder, elbow, wrist."""
        b = self.b
        pre = "l_" if side == "l" else "r_"
        pos_w, rot_w = forward_kinematics(pose, root_pos, root_q, b.off)
        sl = slice(None) if frames is None else frames
        clav_q = rot_w[sl, JID[pre + "clavicle"]]
        a = pos_w[sl, JID[pre + "shoulder"]]
        sx = 1.0 if side == "l" else -1.0
        # target is the wrist; hands extend ~0.09 m beyond it
        R1, R2, _ = two_bone_ik(a, target_w, pole_w, b.l_upper, b.l_fore, [sx, 0, 0], [0, 0, -1], reach=0.995)
        q1, q2 = mat_to_quat(R1), mat_to_quat(R2)
        if twist is not None:
            q2 = quat_mul(q2, quat_axis_angle(np.broadcast_to([sx, 0, 0], q2.shape[:-1] + (3,)), twist))
        sh = quat_mul(quat_conj(clav_q), q1)
        el = quat_mul(quat_conj(q1), q2)
        wr = np.zeros_like(sh)
        wr[..., 3] = 1
        if wrist is not None:
            wr = quat_euler(np.zeros_like(wrist[..., 0]), wrist[..., 0] * sx, wrist[..., 1] * sx)
        return sh, el, wr

    def _chest_frame(self, pose, root_pos, root_q):
        pos_w, rot_w = forward_kinematics(pose, root_pos, root_q, self.b.off)
        return pos_w, rot_w

    def _solve_arms(self, pose, root_pos, root_q, t):
        T, b, s = self.T, self.b, self.s
        pos_w, rot_w = self._chest_frame(pose, root_pos, root_q)
        chest_p, chest_q = pos_w[:, JID["chest"]], rot_w[:, JID["chest"]]
        pel_q = rot_w[:, 0]
        to_w = lambda local, q=chest_q, p=chest_p: p + quat_rotate(q, local)
        phase = np.nan_to_num(self.phase)
        amp = self.gait_amp
        run = gaussian_filter1d(self.run_w, 3, mode="nearest")
        sit = gaussian_filter1d(self.sit_w, 2, mode="nearest")
        for side, sx, pre in (("l", 1.0, "l_"), ("r", -1.0, "r_")):
            shoulder = pos_w[:, JID[pre + "shoulder"]]
            # hanging arm with walking / running swing (in pelvis-yaw frame so it swings forward)
            swing = (-1 if side == "l" else 1) * np.cos(np.pi * phase) * amp
            a_walk = 0.32 * swing + 0.05
            a_run = 0.7 * swing + 0.25
            ang = a_walk * (1 - run) + a_run * run
            L = (b.l_upper + b.l_fore)
            dist = L * (0.985 - 0.06 * np.maximum(swing, 0) * (1 - run) - 0.42 * run)
            q_body = yaw_quat(np.arctan2(*quat_rotate(pel_q, np.broadcast_to([0, 0, 1.0], (T, 3)))[:, [0, 2]].T))
            local = np.stack([sx * (0.035 + 0.02 * run) * s * np.ones(T), -np.cos(ang) * dist,
                              np.sin(ang) * dist], 1)
            tgt = shoulder + quat_rotate(q_body, local)
            # seated: hands rest on the thighs
            if sit.max() > 0:
                kn = pos_w[:, JID[pre + "knee"]]
                hp = pos_w[:, JID[pre + "hip"]]
                on_thigh = hp + (kn - hp) * 0.62 + np.array([0, 0.075 * s, 0]) - \
                    quat_rotate(q_body, np.broadcast_to([sx * 0.02 * s, 0, 0], (T, 3)))
                tgt = tgt * (1 - sit[:, None]) + on_thigh * sit[:, None]
            pole = quat_rotate(q_body, np.broadcast_to([sx * 0.3, -0.2, -1.0], (T, 3)))
            pole = _normalize(pole * (1 - run[:, None]) + quat_rotate(q_body, np.broadcast_to([sx * 0.2, -0.3, -1], (T, 3)))
                              * run[:, None])
            wrist = np.stack([0.15 * np.ones(T) * sit, np.zeros(T)], 1)
            sh, el, wr = self._arm_ik(pose, root_pos, root_q, side, tgt, pole, wrist=wrist)
            pose[:, JID[pre + "shoulder"]], pose[:, JID[pre + "elbow"]], pose[:, JID[pre + "wrist"]] = sh, el, wr
        # overlays
        for ov in self.overlays:
            if ov.kind in ("dance_head",):
                continue
            fr = self._frames(ov.t0, ov.t1)
            if not len(fr):
                continue
            tt = t[fr]
            w = smoothstep((tt - ov.t0) / ov.fade) * (1 - smoothstep((tt - (ov.t1 - ov.fade)) / ov.fade))
            res = self._overlay_targets(ov, tt, fr, pos_w, rot_w)
            if res is None:
                continue
            for side, spec in res.items():
                if side == "clav":
                    for sd, pre in (("l", "l_"), ("r", "r_")):
                        cl = spec[sd]
                        q = quat_euler(np.zeros(len(fr)), np.zeros(len(fr)), cl * (1 if sd == "l" else -1))
                        pose[fr, JID[pre + "clavicle"]] = quat_nlerp(pose[fr, JID[pre + "clavicle"]], q, w)
                    continue
                pre = side + "_"
                tgt_local, pole_local = spec["target"], spec["pole"]
                frame = spec.get("frame", "chest")
                if frame == "chest":
                    tgt_w = to_w(tgt_local, chest_q[fr], chest_p[fr])
                    pole_w = quat_rotate(chest_q[fr], pole_local)
                else:
                    tgt_w, pole_w = tgt_local, quat_rotate(chest_q[fr], pole_local)
                ww = w * spec.get("weight", 1.0)
                sub = pose[fr].copy()
                sh, el, wr = self._arm_ik(pose[fr], root_pos[fr], root_q[fr], side, tgt_w, _normalize(pole_w),
                                          twist=spec.get("twist"), wrist=spec.get("wrist"))
                for j, q in ((pre + "shoulder", sh), (pre + "elbow", el), (pre + "wrist", wr)):
                    pose[fr, JID[j]] = quat_nlerp(sub[:, JID[j]], q, ww)

    def _spine_overlays(self, t):
        out = np.zeros((self.T, 3))
        for ov in self.overlays:
            fr = self._frames(ov.t0, ov.t1)
            if not len(fr):
                continue
            tt = t[fr]
            w = smoothstep((tt - ov.t0) / ov.fade) * (1 - smoothstep((tt - (ov.t1 - ov.fade)) / ov.fade))
            if ov.kind == "talk":
                rng = np.random.default_rng(ov.params["seed"] + 1)
                n = len(fr)
                noise = gaussian_filter1d(rng.normal(0, 1, (n, 3)), 0.6 * self.fps, axis=0, mode="nearest")
                noise /= max(noise.std(), 1e-6)
                out[fr] += w[:, None] * noise * np.array([0.025, 0.05, 0.03]) * ov.params["intensity"]
            elif ov.kind == "dance_arms":
                p = ov.params
                bp = (tt - p["beat0"]) / (60 / p["bpm"])
                st = p["style"]
                amp = {"hiphop": (0.12, 0.15, 0.06), "rave": (0.08, 0.06, 0.03), "disco": (0.04, 0.2, 0.1),
                       "sway": (0.02, 0.1, 0.12), "bounce": (0.06, 0.08, 0.05), "groove": (0.05, 0.14, 0.1)}[st]
                out[fr, 0] += w * amp[0] * (0.5 + 0.5 * np.cos(2 * np.pi * bp)) * p["energy"]
                out[fr, 1] += w * amp[1] * np.sin(np.pi * bp / 2) * p["energy"]
                out[fr, 2] += w * amp[2] * np.sin(np.pi * bp) * p["energy"]
            elif ov.kind == "drink":
                u = (tt - ov.t0) / (ov.t1 - ov.t0)
                out[fr, 0] += -0.08 * bump((u - 0.5) / 0.35) * w
            elif ov.kind == "reach":
                out[fr, 0] += 0.25 * w
        return out

    def _overlay_targets(self, ov, tt, fr, pos_w, rot_w):
        """Hand targets for an overlay: {side: {target, pole, [twist], [wrist], [frame]}}."""
        s = self.s
        n = len(tt)
        p = ov.params
        one = np.ones(n)
        if ov.kind == "talk":
            return self._talk_targets(ov, tt, fr, pos_w, rot_w)
        if ov.kind == "listen":
            return None
        if ov.kind == "wave":
            sd = p["side"]
            sx = 1 if sd == "l" else -1
            osc = np.sin(2 * np.pi * 2.2 * (tt - ov.t0))
            tgt = np.stack([sx * (0.36 + 0.07 * osc) * s * one, 0.42 * s * one, 0.12 * s * one], 1)
            pole = np.broadcast_to([sx * 1.0, -0.6, -0.2], (n, 3))
            wrist = np.stack([0.25 * osc, 0.0 * one], 1)
            return {sd: {"target": tgt, "pole": pole, "wrist": wrist, "twist": -0.8 * sx * one}}
        if ov.kind == "point":
            sd = p["side"]
            pre = sd + "_"
            sh = pos_w[fr, JID[pre + "shoulder"]]
            d = _normalize(p["target"] - sh)
            tgt = sh + d * (self.b.l_upper + self.b.l_fore) * 0.99
            return {sd: {"target": tgt, "pole": np.broadcast_to([0, -1, -0.3], (n, 3)), "frame": "world"}}
        if ov.kind == "hold":
            sd = p["side"]
            sx = 1 if sd == "l" else -1
            if p["kind"] == "phone":
                tgt = np.stack([sx * 0.06 * s * one, -0.02 * s * one, 0.33 * s * one], 1)
                return {sd: {"target": tgt, "pole": np.broadcast_to([sx * 0.3, -1, -0.2], (n, 3)),
                             "wrist": np.stack([-0.4 * one, 0 * one], 1)}}
            tgt = np.stack([sx * 0.2 * s * one, -0.26 * s * one, 0.3 * s * one], 1)
            return {sd: {"target": tgt, "pole": np.broadcast_to([sx * 0.4, -1, -0.4], (n, 3)),
                         "twist": -0.6 * sx * one}}
        if ov.kind == "reach":
            sd = p["side"]
            return {sd: {"target": np.broadcast_to(p["target"], (n, 3)), "pole": np.broadcast_to([0.3, -1, -0.3], (n, 3)),
                         "frame": "world"}}
        if ov.kind == "drink":
            sd = p["side"]
            sx = 1 if sd == "l" else -1
            u = (tt - ov.t0) / (ov.t1 - ov.t0)
            k = smoothstep(u / 0.35) * (1 - smoothstep((u - 0.7) / 0.3))
            near = np.stack([sx * 0.2 * s * one, -0.26 * s * one, 0.3 * s * one], 1)
            mouth = np.stack([sx * 0.06 * s * one, 0.22 * s * one, 0.17 * s * one], 1)
            tgt = near + (mouth - near) * k[:, None]
            return {sd: {"target": tgt, "pole": np.broadcast_to([sx * 0.5, -1, -0.2], (n, 3)),
                         "twist": -0.6 * sx * one}}
        if ov.kind == "jump_arms":
            t_take = p["t_take"]
            # arms swing back during the crouch, up through take-off, then down for landing
            u = tt - t_take
            back = smoothstep((tt - ov.t0) / 0.3) * (1 - smoothstep((u + 0.05) / 0.12))
            up = smoothstep((u + 0.05) / 0.12) * (1 - smoothstep((tt - p["t_land"]) / 0.3))
            res = {}
            for sd, sx in (("l", 1), ("r", -1)):
                tgt = np.stack([sx * 0.22 * s * one, (-0.35 * back + 0.35 * up - 0.4 * (1 - back - up)) * s,
                                (-0.3 * back + 0.25 * up + 0.05) * s], 1)
                res[sd] = {"target": tgt, "pole": np.broadcast_to([sx * 0.5, -0.5, -1], (n, 3))}
            return res
        if ov.kind == "dance_arms":
            return self._dance_targets(ov, tt)
        return None

    def _talk_targets(self, ov, tt, fr, pos_w, rot_w):
        rng = np.random.default_rng(ov.params["seed"])
        s = self.s
        n = len(tt)
        inten = ov.params["intensity"]
        seated = self.sit_w[fr].mean() > 0.5
        # gesture phrases: (start, end, type)
        phrases = []
        t = ov.t0 + rng.uniform(0.0, 0.8)
        while t < ov.t1 - 0.8:
            dur = rng.uniform(1.0, 3.2)
            kind = rng.choice(["beat2", "beatL", "beatR", "open", "explainL", "explainR", "self", "shrug"],
                              p=[0.22, 0.14, 0.18, 0.12, 0.1, 0.12, 0.06, 0.06])
            phrases.append((t, min(t + dur, ov.t1 - 0.2), kind))
            t += dur + rng.uniform(0.3, 1.6) / max(inten, 0.3)
        res = {"l": {"target": np.zeros((n, 3)), "pole": np.zeros((n, 3)), "twist": np.zeros(n), "weight": np.zeros(n),
                     "wrist": np.zeros((n, 2))},
               "r": {"target": np.zeros((n, 3)), "pole": np.zeros((n, 3)), "twist": np.zeros(n), "weight": np.zeros(n),
                     "wrist": np.zeros((n, 2))}}
        base_y = -0.12 if not seated else -0.1
        for sd, sx in (("l", 1), ("r", -1)):
            res[sd]["target"][:] = [sx * 0.14 * s, base_y * s, 0.3 * s]
            res[sd]["pole"][:] = [sx * 0.5, -1, -0.3]
        clav = {"l": np.zeros(n), "r": np.zeros(n)}
        for (a, b_, kind) in phrases:
            m = (tt >= a - 0.4) & (tt <= b_ + 0.4)
            if not m.any():
                continue
            u_in = smoothstep((tt[m] - a + 0.35) / 0.35)
            u_out = 1 - smoothstep((tt[m] - b_) / 0.35)
            env = u_in * u_out
            sides = {"beat2": "lr", "beatL": "l", "beatR": "r", "open": "lr", "explainL": "l", "explainR": "r",
                     "self": rng.choice(["l", "r"]), "shrug": "lr"}[kind]
            # stroke rhythm ~ syllable stress
            rate = rng.uniform(2.0, 3.2)
            beats = np.maximum(0, np.sin(2 * np.pi * rate * (tt[m] - a))) ** 2
            for sd in sides:
                sx = 1 if sd == "l" else -1
                if kind.startswith("beat"):
                    pos = np.array([sx * rng.uniform(0.08, 0.2), rng.uniform(-0.12, 0.0), rng.uniform(0.3, 0.38)]) * s
                    off = np.stack([0 * beats, -0.045 * beats, 0.015 * beats], 1) * s * inten
                    tw = -0.3 * sx
                elif kind == "open":
                    pos = np.array([sx * rng.uniform(0.25, 0.34), rng.uniform(-0.15, -0.05), rng.uniform(0.25, 0.32)]) * s
                    off = np.stack([sx * 0.03 * beats, -0.02 * beats, 0 * beats], 1) * s
                    tw = 1.1 * sx
                elif kind.startswith("explain"):
                    pos = np.array([sx * 0.12, rng.uniform(-0.06, 0.04), 0.36]) * s
                    sweep = np.sin(2 * np.pi * rng.uniform(0.4, 0.8) * (tt[m] - a))
                    off = np.stack([sx * 0.1 * sweep, -0.03 * beats, 0 * beats], 1) * s
                    tw = 0.4 * sx
                elif kind == "self":
                    pos = np.array([sx * 0.04, -0.02, 0.15]) * s
                    off = np.zeros((m.sum(), 3))
                    tw = -0.2 * sx
                else:   # shrug
                    pos = np.array([sx * 0.3, -0.25, 0.2]) * s
                    off = np.zeros((m.sum(), 3))
                    tw = 1.2 * sx
                    clav[sd][m] = np.maximum(clav[sd][m], 0.12 * env)
                tgt = pos[None] + off
                cur = res[sd]["target"][m]
                res[sd]["target"][m] = cur + (tgt - cur) * env[:, None]
                res[sd]["twist"][m] = res[sd]["twist"][m] + (tw - res[sd]["twist"][m]) * env
                res[sd]["weight"][m] = np.maximum(res[sd]["weight"][m], env)
                res[sd]["wrist"][m, 0] = -0.2 * beats * env
        for sd in "lr":
            res[sd]["target"] = gaussian_filter1d(res[sd]["target"], 2.0, axis=0, mode="nearest")
            res[sd]["weight"] = gaussian_filter1d(res[sd]["weight"], 2.0, mode="nearest") * min(1.0, 0.6 + 0.5 * inten)
        res["clav"] = {"l": gaussian_filter1d(clav["l"], 2, mode="nearest"),
                       "r": gaussian_filter1d(clav["r"], 2, mode="nearest")}
        return res

    def _dance_targets(self, ov, tt):
        p = ov.params
        s = self.s
        n = len(tt)
        beat = 60 / p["bpm"]
        bp = (tt - p["beat0"]) / beat
        st = p["style"]
        en = p["energy"]
        one = np.ones(n)
        res = {}
        ph = np.cos(2 * np.pi * bp)          # +1 on the beat
        half = np.sin(np.pi * bp)            # alternates every beat
        bar = np.floor(bp / 4)
        for sd, sx in (("l", 1), ("r", -1)):
            if st == "bounce":    # elbows bent, fists pumping alternately
                alt = sx * half
                tgt = np.stack([sx * 0.22 * one, -0.12 + 0.1 * alt * en, 0.2 + 0.05 * ph], 1)
                pole = np.broadcast_to([sx * 0.6, -1, -0.4], (n, 3))
                tw = -0.4 * sx * one
            elif st == "sway":    # arms raised overhead, swaying side to side
                tgt = np.stack([sx * 0.2 + 0.16 * np.sin(np.pi * bp / 2), 0.62 + 0.03 * ph * one, 0.12 * one], 1)
                pole = np.broadcast_to([sx * 1, 0, -0.2], (n, 3))
                tw = 0.6 * sx * one
            elif st == "disco":   # alternating point up across / down
                up = (np.floor(bp / 2) % 2 == 0).astype(float)
                up = gaussian_filter1d(up, 3, mode="nearest")
                mine = up if sd == "r" else 1 - up
                tgt_up = np.array([sx * 0.25, 0.48, 0.18])
                tgt_dn = np.array([sx * 0.12, -0.3, 0.18])
                tgt = tgt_dn[None] + (tgt_up - tgt_dn)[None] * mine[:, None]
                tgt[:, 0] += sx * 0.03 * ph
                pole = np.broadcast_to([sx * 0.8, -0.6, -0.4], (n, 3))
                tw = 0 * one
            elif st == "hiphop":  # arms crossing and opening on the beat
                cross = 0.5 + 0.5 * np.cos(np.pi * bp)
                tgt = np.stack([sx * (0.05 + 0.25 * (1 - cross)), -0.05 + 0.05 * ph, 0.28 + 0.05 * cross], 1)
                pole = np.broadcast_to([sx * 0.7, -1, -0.3], (n, 3))
                tw = -0.6 * sx * one
            elif st == "rave":    # fist pumps up on the beat, alternating hands every bar
                lead = ((bar % 2 == 0) == (sd == "r")).astype(float)
                lead = gaussian_filter1d(lead, 3, mode="nearest")
                pump = 0.5 + 0.5 * ph
                tgt_hi = np.stack([sx * 0.2 * one, 0.25 + 0.25 * pump, 0.2 * one], 1)
                tgt_lo = np.stack([sx * 0.2 * one, -0.15 * one + 0.05 * ph, 0.2 * one], 1)
                tgt = tgt_lo + (tgt_hi - tgt_lo) * lead[:, None]
                pole = np.broadcast_to([sx * 0.6, -1, -0.3], (n, 3))
                tw = -0.3 * sx * one
            else:                 # groove: rolling arms in front + claps every 4th beat
                ang = 2 * np.pi * bp / 2
                tgt = np.stack([sx * 0.08 + 0.06 * np.cos(ang + (0 if sd == "l" else np.pi)),
                                -0.08 + 0.07 * np.sin(ang + (0 if sd == "l" else np.pi)), 0.3 * one], 1)
                clap = bump(((bp % 4) - 3.0) / 0.5)
                tgt[:, 0] = tgt[:, 0] * (1 - clap) + sx * 0.03 * clap
                pole = np.broadcast_to([sx * 0.6, -1, -0.3], (n, 3))
                tw = 0.5 * sx * one
            res[sd] = {"target": tgt * s, "pole": pole, "twist": tw}
        return res

    def _solve_head(self, pose, root_pos, root_q, t):
        T, b = self.T, self.b
        pos_w, rot_w = forward_kinematics(pose, root_pos, root_q, b.off)
        head_p = pos_w[:, JID["head"]] + np.array([0, 0.1 * self.s, 0])
        chest_q = rot_w[:, JID["chest"]]
        # default gaze: ahead along the heading, a little down while walking
        fwd = quat_rotate(root_q, np.broadcast_to([0, 0, 1.0], (T, 3)))
        default = head_p + fwd * 4.0 + np.array([0, -0.3, 0]) * (self.gait_amp[:, None] > 0.1)
        target = default.copy()
        wsum = np.zeros(T)
        for (t0, t1, tg) in self.look:
            fr = self._frames(t0, t1)
            if not len(fr):
                continue
            if callable(tg):
                pts = np.array([tg(int(f)) for f in fr], float)
            else:
                pts = np.broadcast_to(np.asarray(tg, float), (len(fr), 3))
            w = smoothstep((t[fr] - t0) / 0.35) * (1 - smoothstep((t[fr] - (t1 - 0.35)) / 0.35))
            target[fr] = target[fr] * (1 - w[:, None]) + pts * w[:, None]
            wsum[fr] = np.maximum(wsum[fr], w)
        # idle glances when nothing is specified
        rng = np.random.default_rng(int(self.rng.integers(1 << 30)))
        glance = np.zeros((T, 2))
        tt = 0.0
        while tt < t[-1]:
            dur = rng.uniform(1.5, 5.0)
            g = rng.normal(0, [0.25, 0.08])
            m = (t >= tt) & (t < tt + dur)
            glance[m] = g
            tt += dur
        glance = gaussian_filter1d(glance, 0.15 * self.fps, axis=0, mode="nearest") * (1 - wsum[:, None]) * 0.7
        d = target - head_p
        dl = quat_rotate(quat_conj(chest_q), d)
        yaw = np.arctan2(dl[:, 0], dl[:, 2]) + glance[:, 0]
        pitch = -np.arctan2(dl[:, 1], np.linalg.norm(dl[:, [0, 2]], axis=1)) + glance[:, 1]
        yaw = np.clip(np.unwrap(yaw), -1.3, 1.3)
        pitch = np.clip(pitch, -0.6, 0.75)
        # saccade-like but smooth head motion
        yaw = gaussian_filter1d(yaw, 2.5, mode="nearest")
        pitch = gaussian_filter1d(pitch, 2.5, mode="nearest")
        nod = np.zeros(T)
        tilt = np.zeros(T)
        for ov in self.overlays:
            fr = self._frames(ov.t0, ov.t1)
            if not len(fr):
                continue
            tt_ = t[fr]
            w = smoothstep((tt_ - ov.t0) / ov.fade) * (1 - smoothstep((tt_ - (ov.t1 - ov.fade)) / ov.fade))
            if ov.kind in ("talk", "listen"):
                r = np.random.default_rng(ov.params["seed"] + 7)
                k = 0.06 if ov.kind == "talk" else 0.04
                nn = gaussian_filter1d(r.normal(0, 1, len(fr)), 0.12 * self.fps, mode="nearest")
                nod[fr] += w * k * nn / max(nn.std(), 1e-6)
                tl = gaussian_filter1d(r.normal(0, 1, len(fr)), 0.8 * self.fps, mode="nearest")
                tilt[fr] += w * 0.05 * tl / max(tl.std(), 1e-6)
            elif ov.kind == "dance_head":
                bp = (tt_ - ov.params["beat0"]) / (60 / ov.params["bpm"])
                nod[fr] += w * 0.1 * np.cos(2 * np.pi * bp)
                tilt[fr] += w * 0.06 * np.sin(np.pi * bp)
        pose[:, JID["neck"]] = quat_euler(pitch * 0.4 + nod * 0.3, yaw * 0.4, tilt * 0.4)
        pose[:, JID["head"]] = quat_euler(pitch * 0.6 + nod * 0.7, yaw * 0.6, tilt * 0.6)

    def _mouth(self, t):
        """jawOpen weights from the speaking flag: syllable-rate opening with pauses."""
        rng = np.random.default_rng(int(self.rng.integers(1 << 30)))
        sp = gaussian_filter1d(self.speaking, 2, mode="nearest")
        syl = 0.5 + 0.5 * np.sin(2 * np.pi * 4.3 * t + 2.0 * np.sin(2 * np.pi * 0.7 * t))
        env = gaussian_filter1d((rng.random(len(t)) > 0.15).astype(float), 3, mode="nearest")
        return np.clip(sp * syl * env * 0.9, 0, 1)


# ---------------------------------------------------------------- tracks helpers

def static_track(T, pos, yaw=0.0, q=None):
    pos = np.broadcast_to(np.asarray(pos, float), (T, 3)).copy()
    rot = np.broadcast_to(yaw_quat(yaw) if q is None else np.asarray(q, float), (T, 4)).copy()
    return {"position": pos.astype(np.float32), "rotation": rot.astype(np.float32)}


def time_remap(track: dict, frames: np.ndarray) -> dict:
    """Resample a track (dict of [T_world, ...] arrays) at integer world frames (video time skips)."""
    out = {}
    for k, v in track.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) > frames.max():
            out[k] = v[frames]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------- vehicles

def drive(path: Path, n_frames, fps, dims, speed=12.0, t_start=0.0, s_start=0.0, accel=2.0, stops=(),
          ground=None, lane_offset=0.0, kind="car"):
    """Vehicle track along a path: heading = path tangent, steering from curvature.

    Returns a track with a 6-joint pose (chassis, body, 4 wheels) or bike pose (frame, body,
    2 wheels, crank). `stops` is a list of (arc length, wait seconds) for traffic lights.
    """
    T = int(n_frames)
    t = np.arange(T) / fps
    ground = ground if ground is not None else (lambda x, z: np.zeros(np.broadcast(x, z).shape))
    # integrate speed with smooth accelerations, stopping at requested arc positions
    v_cur, s = 0.0 if t_start > -1 else speed, s_start
    if t_start < 0:
        v_cur = speed
    ss = np.zeros(T)
    vs = np.zeros(T)
    stops = sorted(stops)
    wait_until = -1.0
    dt = 1.0 / fps
    for i in range(T):
        tt = t[i]
        target = speed if tt >= t_start else 0.0
        for (s_stop, wait) in stops:
            if s < s_stop:
                brake_d = v_cur * v_cur / (2 * accel * 0.8) + 0.5
                if s_stop - s < brake_d:
                    target = 0.0 if s_stop - s > 0.3 else 0.0
                    if v_cur < 0.2 and wait_until < 0:
                        wait_until = tt + wait
                break
        if wait_until > 0:
            if tt < wait_until:
                target = 0.0
            else:
                wait_until = -1.0
                stops = [st for st in stops if st[0] > s + 1.0]
        dv = np.clip(target - v_cur, -accel * 1.3 * dt, accel * dt)
        v_cur = max(0.0, v_cur + dv)
        s = min(path.length, s + v_cur * dt)
        ss[i], vs[i] = s, v_cur
    vs = gaussian_filter1d(vs, 3, mode="nearest")
    xz = path.pos(ss)
    yaw = path.heading(ss)
    if lane_offset:
        xz = xz + np.stack([np.cos(yaw), -np.sin(yaw)], 1) * lane_offset
    y = ground(xz[:, 0], xz[:, 1])
    pos = np.stack([xz[:, 0], y, xz[:, 1]], 1)
    yaw_s = gaussian_filter1d(yaw, 1.5, mode="nearest")
    rot = yaw_quat(yaw_s)
    kappa = gaussian_filter1d(path.curvature(ss), 3, mode="nearest")
    wb = dims["axle_f"] - dims["axle_r"]
    steer = np.arctan(wb * kappa)
    a_long = np.gradient(vs, dt)
    a_lat = vs * vs * kappa
    spin = ss / dims["wheel_r"]
    roll = np.clip(gaussian_filter1d(0.012 * a_lat, 4, mode="nearest"), -0.08, 0.08)
    pitch = np.clip(gaussian_filter1d(-0.01 * a_long, 4, mode="nearest"), -0.05, 0.05)
    if kind == "bike":
        J = 5
        pose = np.zeros((T, J, 4))
        pose[..., 3] = 1
        lean = np.clip(np.arctan(vs * vs * kappa / G), -0.6, 0.6)
        pose[:, 0] = quat_euler(np.zeros(T), np.zeros(T), -lean)
        pose[:, 2] = quat_mul(quat_euler(0, steer * 0.8, 0), quat_euler(spin, 0, 0))
        pose[:, 3] = quat_euler(spin, 0, 0)
        pose[:, 4] = quat_euler(spin * 0.4, 0, 0)
    else:
        pose = np.zeros((T, 6, 4))
        pose[..., 3] = 1
        pose[:, 1] = quat_euler(pitch, np.zeros(T), roll)
        for j, steer_on in ((2, True), (3, True), (4, False), (5, False)):
            q = quat_euler(spin, 0, 0)
            if steer_on:
                q = quat_mul(yaw_quat(steer), q)
            pose[:, j] = q
    return {"position": pos.astype(np.float32), "rotation": rot.astype(np.float32),
            "pose": quat_continuous(pose).astype(np.float32), "speed": vs, "arc": ss}


# ---------------------------------------------------------------- objects

def ball_flight(n_frames, fps, p0, v0, radius=0.11, restitution=0.7, friction=0.6, t0=0.0, ground=None,
                bounds=None, rolling_resistance=0.35):
    """Ballistic ball with bounces (restitution, tangential friction), then rolling to rest.

    Before t0 the ball rests at p0. `bounds` = ((xmin, zmin), (xmax, zmax)) adds walls.
    Rotation integrates the rolling angular velocity.
    """
    T = int(n_frames)
    ground = ground if ground is not None else (lambda x, z: np.zeros(np.broadcast(x, z).shape))
    pos = np.zeros((T, 3))
    rot = np.zeros((T, 4))
    rot[:, 3] = 1
    p = np.array(p0, float)
    v = np.array(v0, float)
    q = np.array([0, 0, 0, 1.0])
    sub = 8
    dt = 1.0 / fps / sub
    rolling = False
    for i in range(T):
        tt = i / fps
        if tt >= t0:
            for _ in range(sub):
                gy = float(ground(p[0], p[2])) + radius
                if not rolling:
                    v[1] -= G * dt
                    v *= (1 - 0.02 * dt)
                p += v * dt
                gy = float(ground(p[0], p[2])) + radius
                if p[1] <= gy:
                    p[1] = gy
                    if not rolling:
                        if abs(v[1]) < 0.6:
                            rolling = True
                            v[1] = 0
                        else:
                            v[1] = -v[1] * restitution
                            v[[0, 2]] *= friction ** 0.5
                if rolling:
                    p[1] = gy
                    sp = np.linalg.norm(v[[0, 2]])
                    if sp > 1e-4:
                        dec = min(sp, rolling_resistance * dt)
                        v[[0, 2]] *= (sp - dec) / sp
                    v[1] = 0
                if bounds is not None:
                    (x0, z0), (x1, z1) = bounds
                    for ax, lo, hi in ((0, x0 + radius, x1 - radius), (2, z0 + radius, z1 - radius)):
                        if p[ax] < lo:
                            p[ax], v[ax] = lo, -v[ax] * restitution
                        elif p[ax] > hi:
                            p[ax], v[ax] = hi, -v[ax] * restitution
                # rotation from rolling / spinning
                vh = np.array([v[0], 0, v[2]])
                w = np.cross([0, 1.0, 0], vh) / radius
                ang = np.linalg.norm(w) * dt
                if ang > 1e-9:
                    q = quat_mul(quat_axis_angle(w, ang), q)
        pos[i] = p
        rot[i] = q / np.linalg.norm(q)
    return {"position": pos.astype(np.float32), "rotation": quat_continuous(rot).astype(np.float32)}


def roll_object(n_frames, fps, p0, direction, v0, radius, t0=0.0, decel=0.4, ground=None, axis_up=True):
    """Object rolling on the ground (no slipping): decelerating at `decel` m/s^2."""
    T = int(n_frames)
    ground = ground if ground is not None else (lambda x, z: np.zeros(np.broadcast(x, z).shape))
    t = np.arange(T) / fps
    tau = np.maximum(0, t - t0)
    t_stop = v0 / decel
    d = np.where(tau < t_stop, v0 * tau - 0.5 * decel * tau * tau, v0 * t_stop / 2)
    dirv = np.asarray(direction, float)
    dirv = dirv / np.linalg.norm(dirv)
    xz = np.asarray(p0, float)[[0, 2]] + d[:, None] * dirv[[0, 2]]
    y = ground(xz[:, 0], xz[:, 1]) + radius
    pos = np.stack([xz[:, 0], y, xz[:, 1]], 1)
    axis = np.cross([0, 1.0, 0], dirv)
    rot = quat_axis_angle(np.broadcast_to(axis, (T, 3)), d / radius)
    return {"position": pos.astype(np.float32), "rotation": rot.astype(np.float32)}


def attached_track(holder: dict, joint: int, intervals, rest_pos, rest_yaw=0.0, grip=(0.0, 0.0, 0.0),
                   upright=True):
    """Track of an object picked up / carried by a joint (e.g. a hand) and put down.

    holder: humanoid track (with 'joints_world', 'joint_rot_world'). intervals: list of
    (frame_pick, frame_place, place_pos or None). Between intervals the object rests where it
    was last put down. The relative offset to the joint is captured at the pick frame, so
    there is no pop at pick-up; at the place frame it is released where it is.
    """
    T = len(holder["position"])
    pos = np.broadcast_to(np.asarray(rest_pos, float), (T, 3)).copy()
    rot = np.broadcast_to(yaw_quat(rest_yaw), (T, 4)).copy()
    jp = holder["joints_world"][:, joint].astype(np.float64)
    jq = holder["joint_rot_world"][:, joint].astype(np.float64)
    cur_p, cur_q = np.asarray(rest_pos, float), yaw_quat(rest_yaw)
    last = 0
    for (f0, f1, place) in intervals:
        f0, f1 = int(np.clip(f0, 0, T - 1)), int(np.clip(f1, 0, T))
        pos[last:f0] = cur_p
        rot[last:f0] = cur_q
        rel_p = quat_rotate(quat_conj(jq[f0]), cur_p - jp[f0])
        rel_q = quat_mul(quat_conj(jq[f0]), cur_q)
        seg = slice(f0, f1)
        pos[seg] = jp[seg] + quat_rotate(jq[seg], np.broadcast_to(rel_p, (f1 - f0, 3)))
        q = quat_mul(jq[seg], np.broadcast_to(rel_q, (f1 - f0, 4)))
        if upright:
            fwd = quat_rotate(q, np.broadcast_to([0, 0, 1.0], (f1 - f0, 3)))
            q = yaw_quat(np.arctan2(fwd[:, 0], fwd[:, 2]))
        rot[seg] = q
        cur_p = pos[f1 - 1].copy() if f1 > f0 else cur_p
        cur_q = rot[f1 - 1].copy() if f1 > f0 else cur_q
        if place is not None and f1 < T:
            cur_p = np.asarray(place, float)
        last = f1
    pos[last:] = cur_p
    rot[last:] = cur_q
    return {"position": pos.astype(np.float32), "rotation": quat_continuous(rot).astype(np.float32)}
