"""Compose a whole synthetic video as a `.restage` scene document with exact ground truth.

`generate(genre, seed, seconds)` picks 1-3 locations, casts entities (people, objects,
vehicles), performs them on a *world* timeline (`motion.py`), plans and solves the edit
(`cinema.py`), maps world time to video time (time skips at cuts for hiking / driving /
gameplay), computes each entity's per-frame `state` (0 absent, 1 present but not seen,
2 visible: camera frustum + coarse occlusion by the set and other entities), optionally
injects continuity errors between cuts and records them, and assembles the document of
docs/SCENE_FORMAT.md. Everything is deterministic given the seed.

A person who appears in several locations is one entity per location (the format has one
location per entity); the copies share `identity` and are absent (state 0) while the edit is
elsewhere. See FORMAT_NOTES.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import assets as A
from . import cinema as C
from . import sets as S
from .motion import DANCE_STYLES, HumanoidAnimator, Path, attached_track, ball_flight, drive, quat_continuous, \
    static_track
from .spatial import sphere_occludes
from ..scene import JID, quat_rotate, quat_to_mat, skeleton_doc

GENRES = ("talking", "interview", "dance", "hiking", "driving", "voxel")
PRODUCER = "restage-gen 0.1 (ground truth)"


# ---------------------------------------------------------------- document builder

class DocBuilder:
    def __init__(self, fps, n_frames, title, seed, genre):
        self.fps, self.T = fps, n_frames
        self.meta = {"title": title, "producer": PRODUCER, "seed": int(seed), "genre": genre}
        self.materials = {"m_matte": {"id": "m_matte", "color": [1.0, 1.0, 1.0], "roughness": 0.9, "metalness": 0.0,
                                      "emissive": [0.0, 0.0, 0.0], "flat": True}}
        self.meshes, self.skeletons, self.locations = [], [], []
        self.entities, self.cameras, self.shots, self.continuity = [], [], [], []
        self._mesh_ids = set()

    def material_for(self, name):
        if name not in self.materials:
            if name.startswith("m_glow_"):
                h = name[len("m_glow_"):]
                c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
                self.materials[name] = {"id": name, "color": [0.0, 0.0, 0.0], "roughness": 0.9, "metalness": 0.0,
                                        "emissive": c, "flat": True}
            else:
                self.materials[name] = {"id": name, "color": [1.0, 1.0, 1.0], "roughness": 0.9, "metalness": 0.0,
                                        "emissive": [0.0, 0.0, 0.0], "flat": True}
        return name

    def add_mesh(self, mid, m):
        if mid in self._mesh_ids:
            return mid
        v = len(m["positions"])
        e = {"id": mid, "positions": np.asarray(m["positions"], np.float32),
             "indices": np.asarray(m["indices"], np.uint16 if v < 65536 else np.uint32),
             "colors": np.asarray(m["colors"], np.float32),
             "material": self.material_for(m.get("material", "m_matte"))}
        if "joints" in m:
            e["joints"] = np.asarray(m["joints"], np.uint8)
            e["weights"] = np.asarray(m["weights"], np.float32)
        if m.get("morphs"):
            e["morphs"] = [{"name": mo["name"], "positions": np.asarray(mo["positions"], np.float32)} for mo in m["morphs"]]
        self.meshes.append(e)
        self._mesh_ids.add(mid)
        return mid

    def add_location(self, ls: S.LocationSet):
        for mid, m in ls.meshes.items():
            self.add_mesh(mid, m)
        self.locations.append(ls.loc)

    def doc(self):
        return {"format": "restage.scene", "version": 1, "fps": self.fps, "n_frames": self.T, "meta": self.meta,
                "materials": list(self.materials.values()), "meshes": self.meshes, "skeletons": self.skeletons,
                "locations": self.locations, "entities": self.entities, "cameras": self.cameras, "shots": self.shots,
                "continuity": self.continuity}


# ---------------------------------------------------------------- cast

@dataclass
class Ent:
    eid: str
    name: str
    cls: str
    loc: str
    mesh: dict
    track: dict = None             # world-time track
    skeleton: dict = None
    subject: C.Subject = None
    identity: str = None
    params: dict = field(default_factory=dict)
    animator: HumanoidAnimator = None
    present: np.ndarray = None     # world... video-time presence mask (filled later)
    height: float = 0.3
    radius: float = 0.1
    entity_class: str = "object"


class Cast:
    def __init__(self, rng, fps, W):
        self.rng, self.fps, self.W = rng, fps, W
        self.ents: list[Ent] = []
        self._n = 0

    def _id(self):
        e = f"e{self._n}"
        self._n += 1
        return e

    def person(self, loc, name, params=None, identity=None, mesh=None, scale=None, style="realistic", **anim_kw):
        if mesh is None:
            P = params or A.humanoid_params(self.rng, style)
            P["style"] = style
            mesh, s = A.humanoid(P, self.rng)
        else:
            P, s = params, scale
        eid = self._id()
        e = Ent(eid, name, "person", loc, mesh, identity=identity or eid, params={"appearance": P, "scale": s},
                entity_class="person", height=P["height"], radius=0.3)
        e.skeleton = skeleton_doc(s, f"sk_{eid}")
        e.skeleton["base"] = "humanoid22"
        e.skeleton["scale"] = float(s)
        e.animator = HumanoidAnimator(self.W, self.fps, s, np.random.default_rng(self.rng.integers(1 << 31)), **anim_kw)
        self.ents.append(e)
        return e

    def obj(self, loc, name, mesh, track, height, radius, cls="object"):
        eid = self._id()
        e = Ent(eid, name, cls, loc, mesh, track=track, identity=eid, entity_class=cls, height=height, radius=radius)
        self.ents.append(e)
        return e

    def build_people(self):
        for e in self.ents:
            if e.animator is not None and e.track is None:
                e.track = e.animator.build()

    def subjects(self, loc):
        out = {}
        for e in self.ents:
            if e.loc != loc:
                continue
            if e.subject is None:
                if e.cls == "person":
                    e.subject = C.person_subject(e.eid, e.track, e.params["scale"])
                else:
                    e.subject = C.object_subject(e.eid, e.track, e.height, e.radius, e.cls, e.params.get("center_h"))
            out[e.eid] = e.subject
        return out


def _ground_obj(mesh):
    p = mesh["positions"]
    return float(p[:, 1].max() - p[:, 1].min()), float(np.linalg.norm(p[:, [0, 2]].max(0) - p[:, [0, 2]].min(0)) / 2)


def place_objects(cast, ls, rng, n=3, kinds=("mug", "book", "bottle", "phone", "plant")):
    """Small movable objects on the set's surfaces, as entities (so they can move or vanish)."""
    out = []
    surfaces = list(ls.anchors.get("surfaces", []))
    rng.shuffle(surfaces)
    used = []
    for i in range(min(n, len(surfaces) * 2)):
        if not surfaces:
            break
        sf = surfaces[i % len(surfaces)]
        kind = str(rng.choice(kinds))
        mesh = {"mug": lambda: A.mug(rng), "book": lambda: A.book(rng), "bottle": lambda: A.bottle(rng),
                "phone": lambda: A.phone(), "plant": lambda: A.plant(rng, 0.3), "laptop": lambda: A.laptop(rng)}[kind]()
        mesh = A.merge(mesh, *mesh.get("extras", []))
        mesh.pop("extras", None)
        h, r = _ground_obj(mesh)
        for _ in range(12):
            local = (rng.uniform(-0.5, 0.5, 2) * np.asarray(sf["size"]))
            yaw = sf["yaw"]
            dx = local[0] * np.cos(yaw) + local[1] * np.sin(yaw)
            dz = -local[0] * np.sin(yaw) + local[1] * np.cos(yaw)
            pos = np.array([sf["pos"][0] + dx, sf["pos"][1] + 0.002, sf["pos"][2] + dz])
            if all(np.linalg.norm(pos[[0, 2]] - u[[0, 2]]) > 0.18 for u in used):
                break
        used.append(pos)
        tr = static_track(cast.W, pos, rng.uniform(0, 2 * np.pi))
        out.append(cast.obj(ls.id, kind, mesh, tr, h, r))
    return out


# ---------------------------------------------------------------- edit timing helpers

def world_map(shots, time_skip=None):
    """World frame per video frame, per shot (time skips add world-time jumps at some cuts)."""
    w = 0
    for i, s in enumerate(shots):
        if i > 0 and time_skip is not None:
            w += int(time_skip(i, s))
        s.world = np.arange(w, w + s.n)
        w += s.n
    return w


# ---------------------------------------------------------------- visibility

def compute_states(cast: Cast, shots, cams, T, fps, aspect, worlds, video_tracks, present):
    """state per entity [T]: 0 absent, 1 present but not visible, 2 visible in the frame."""
    states = {}
    loc_of_frame = np.empty(T, object)
    shot_of = np.zeros(T, int)
    for i, s in enumerate(shots):
        loc_of_frame[s.start:s.end] = s.loc
        shot_of[s.start:s.end] = i
    cam_p = np.zeros((T, 3))
    cam_q = np.zeros((T, 4))
    cam_f = np.zeros(T)
    for i, s in enumerate(shots):
        p, q, f, _ = cams[i]
        cam_p[s.start:s.end], cam_q[s.start:s.end], cam_f[s.start:s.end] = p, q, f
    # sample points per entity in video time
    pts = {}
    for e in cast.ents:
        tr = video_tracks[e.eid]
        if e.cls == "person":
            jw = tr["joints_world"].astype(np.float64)
            idx = [JID[j] for j in ("head", "neck", "chest", "spine1", "pelvis", "l_elbow", "r_elbow", "l_wrist",
                                    "r_wrist", "l_knee", "r_knee", "l_ankle", "r_ankle", "l_shoulder", "r_shoulder",
                                    "head")]
            P = jw[:, idx].copy()
            P[:, 0] += [0, 0.1 * e.params["scale"], 0]
            P[:, -1] += [0, 0.2 * e.params["scale"], 0]
            pts[e.eid] = P
        else:
            p = tr["position"].astype(np.float64)
            q = tr["rotation"].astype(np.float64)
            ch = e.params.get("center_h", e.height / 2)
            if e.cls == "vehicle":     # a 3x3x3 lattice over the body box
                w = e.params.get("width", e.radius * 0.8)
                g = np.array([[x, y, z] for x in (-0.42, 0, 0.42) for y in (0.15, 0.5, 0.85) for z in (-0.45, 0, 0.45)])
                off = g * [w, e.height, e.radius * 2] - [0, 0, 0]
                off[:, 1] -= ch
            else:
                off = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0]], float)
                off = off * [e.radius * 0.8, e.height * 0.45, e.radius * 0.8]
            c = p + [0, ch, 0]
            P = c[:, None] + quat_rotate(np.repeat(q[:, None], len(off), 1), np.broadcast_to(off, (len(p), len(off), 3)))
            pts[e.eid] = P
    spheres = {}
    for e in cast.ents:
        if e.cls == "person":
            spheres[e.eid] = (pts[e.eid][:, [0, 2, 3, 4, 9, 10]], np.array([0.12, 0.17, 0.15, 0.15, 0.08, 0.08]) *
                              e.params["scale"])
        elif e.cls == "vehicle":     # three spheres along the body (lattice points: centre column, mid height)
            w = e.params.get("width", e.radius * 0.8)
            spheres[e.eid] = (pts[e.eid][:, [12, 13, 14]], np.full(3, min(w, e.height) * 0.5))
        else:
            spheres[e.eid] = (pts[e.eid][:, :1], np.array([max(e.radius, e.height / 2) * 0.8]))
    for e in cast.ents:
        st = np.where(present[e.eid], 1, 0).astype(np.uint8)
        P = pts[e.eid]
        n_pts = P.shape[1]
        here = (loc_of_frame == e.loc) & present[e.eid]
        frames = np.where(here)[0]
        if len(frames):
            pf = P[frames]
            cp = cam_p[frames][:, None]
            rel = pf - cp
            R = quat_to_mat(cam_q[frames])
            pc = np.einsum("tji,tkj->tki", R, rel)
            z = -pc[..., 2]
            tv = np.tan(np.radians(cam_f[frames]) / 2)[:, None]
            x = pc[..., 0] / np.maximum(z, 1e-6) / (tv * aspect)
            y = pc[..., 1] / np.maximum(z, 1e-6) / tv
            inside = (z > 0.1) & (np.abs(x) < 1.0) & (np.abs(y) < 1.0)
            # attached first-person cameras do not "see" their own head
            for fi, f in enumerate(frames):
                sh = shots[shot_of[f]]
                if sh.params.get("host") == e.eid and e.cls == "person":
                    inside[fi, :3] = False      # a first-person camera does not see its own head
            vis = inside.copy()
            cand = np.where(inside.any(1))[0]
            if len(cand):
                ff = frames[cand]
                c = np.repeat(cam_p[ff][:, None], n_pts, 1).reshape(-1, 3)
                p = pf[cand].reshape(-1, 3)
                world = worlds[e.loc]
                blk = world.blocked(c, p, shrink=0.08).reshape(len(cand), n_pts)
                # other entities as spheres
                others = [o for o in cast.ents if o.eid != e.eid and o.loc == e.loc]
                for ci, f in enumerate(ff):
                    sph, rad = [], []
                    for o in others:
                        if not present[o.eid][f] or shots[shot_of[f]].params.get("host") == o.eid:
                            continue
                        sph.append(spheres[o.eid][0][f])
                        rad.append(spheres[o.eid][1])
                    if sph:
                        blk[ci] |= sphere_occludes(np.broadcast_to(cam_p[f], (n_pts, 3)).copy(), pf[cand[ci]],
                                                   np.concatenate(sph), np.concatenate(rad))
                vis[cand] &= ~blk
            # projected size: entities under ~2% of the frame height (7 px at 360p) count as not visible
            extent = 1.7 if e.cls == "person" else max(e.height, 2 * e.radius)
            size = extent / np.maximum(z.min(1), 0.1) / (2 * tv[:, 0])
            # people need two visible sample points, unless they are close (over-the-shoulder, POV hands)
            need = np.where((e.cls == "person") & (z.min(1) > 2.0), 2, 1)
            seen = (vis.sum(1) >= need) & (size > 0.02)
            st[frames] = np.where(seen, 2, 1)
        states[e.eid] = st
    return states


# ---------------------------------------------------------------- genre scenarios

def _video_tracks(cast, shots, T):
    world_of = np.zeros(T, int)
    for s in shots:
        world_of[s.start:s.end] = s.world
    out = {}
    for e in cast.ents:
        tr = {}
        for k, v in e.track.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == cast.W:
                tr[k] = v[world_of]
            else:
                tr[k] = v
        out[e.eid] = tr
    return out, world_of


def _seat_person(cast, ls, seat, name, rng, look=None, **kw):
    p = cast.person(ls.id, name, pos=tuple(np.asarray(seat["pos"])[[0, 2]]), yaw=seat["yaw"])
    p.animator.sit_still(seat["pos"], seat["yaw"])
    return p


def scene_talking(rng, T, fps):
    loc_name = str(rng.choice(["living_room", "kitchen", "office", "bedroom"]))
    ls = S.make(loc_name, "loc0", rng)
    sets = [ls]
    W = T
    cast = Cast(rng, fps, W)
    seats = [s for s in ls.anchors["seats"] if s["kind"] in ("couch", "office_chair", "bed", "chair", "armchair")]
    standing = loc_name == "kitchen" or not seats or rng.random() < 0.25
    if standing:
        spot = ls.anchors["stand"][0] if ls.anchors["stand"] else {"pos": [0, 0, 0], "yaw": 0.0}
        host = cast.person(ls.id, "host", pos=(spot["pos"][0], spot["pos"][2]), yaw=spot["yaw"])
        eye_h = host.params["appearance"]["height"] * 0.93
        host_pos = np.array([spot["pos"][0], eye_h, spot["pos"][2]])
        facing = spot["yaw"]
    else:
        seat = seats[len(seats) // 2] if loc_name == "living_room" else seats[0]
        host = _seat_person(cast, ls, seat, "host", rng)
        eye_h = seat["pos"][1] + 0.135 * host.params["scale"] + 0.72 * host.params["scale"]
        host_pos = np.array([seat["pos"][0], eye_h, seat["pos"][2]])
        facing = seat["yaw"]
    # main camera ("A-cam") the host addresses: in front, a little off-axis, at eye level
    a_cam = None
    for _ in range(60):
        az = facing + np.radians(rng.normal(0, 8))
        dist = rng.uniform(1.4, 2.3)
        pos = host_pos + np.array([np.sin(az) * dist, rng.uniform(-0.08, 0.06), np.cos(az) * dist])
        if not ls.world.inside(pos[None], margin=0.3)[0] and not ls.world.blocked(pos[None], host_pos[None])[0]:
            a_cam = {"pos": pos, "azimuth_world": float(np.degrees(az)), "handheld": bool(rng.random() < 0.3),
                     "center": bool(rng.random() < 0.6)}
            break
    if a_cam is None:
        pos = host_pos + np.array([np.sin(facing), 0, np.cos(facing)]) * 1.2
        a_cam = {"pos": pos, "azimuth_world": float(np.degrees(facing)), "handheld": False, "center": True}
    an = host.animator
    t_end = T / fps
    # performance: greeting wave, talking with pauses, maybe a sip from a mug
    objs = place_objects(cast, ls, rng, n=int(rng.integers(2, 4)))
    t = 0.0
    if rng.random() < 0.5:
        an.wave(0.3, 1.8, "r")
    while t < t_end:
        dur = rng.uniform(2.5, 7.0)
        an.talk(t, min(t + dur, t_end), intensity=rng.uniform(0.6, 1.2))
        t += dur + rng.uniform(0.3, 1.0)
    an.look_at(0, t_end + 1, a_cam["pos"])
    if standing and rng.random() < 0.5:
        an.stand(t_end * 0.45)
        tgt = ls.anchors.get("focus", [0, 1, 0])
        an.point(t_end * 0.45, 1.6, tgt, "l")
    # a sip from a mug: reach, pick up, drink, put it back (the mug follows the hand)
    sip = None
    if rng.random() < 0.7:
        sip = _plan_sip(cast, ls, host, rng, t_end)
    # a ball thrown in from off-screen bouncing across the room
    ball = None
    events = []
    if loc_name in ("living_room", "kitchen") and rng.random() < 0.5:
        ball = _thrown_ball(cast, ls, rng, T, fps)
        f0 = ball.params["present_from"] + 1
        # the host glances at the ball as it bounces by; the edit cuts to a wide pan following it
        an.look_at(f0 / fps + 0.25, f0 / fps + 2.4, lambda f, b=ball: b.track["position"][min(f, len(b.track["position"]) - 1)])
        events.append((max(0, f0 - 8), min(T, f0 + int(2.6 * fps)), ball.eid))
    cast.build_people()
    if sip is not None:
        mug, f_pick, f_place, surface_y, side = sip
        wrist = JID[side + "_wrist"]
        tr = attached_track(host.track, wrist, [(f_pick, f_place, None)], mug.track["position"][0], 0.0)
        # released mugs settle on the surface
        tr["position"][f_place:, 1] = surface_y
        mug.track = tr
    if ball is not None:
        objs.append(ball)
    # b-roll: a cutaway of the host walking somewhere else while the voice-over continues
    if not events and T >= 10 * fps and rng.random() < 0.35:
        b_len = int(rng.uniform(3.0, 5.0) * fps)
        b0 = int(rng.uniform(0.3, 0.6) * T)
        events.append((b0, min(T, b0 + b_len), _broll_host(cast, sets, host, rng, T, fps)))
    shots = []
    f = 0
    for (e0, e1, eid) in events:
        if e0 - f >= int(1.4 * fps):
            shots += C.plan_talking(f, e0, fps, rng, host.eid, a_cam, [o.eid for o in objs if o.cls == "object"], ls.id)
        else:
            e0 = f
        ent = next(x for x in cast.ents if x.eid == eid)
        if ent.loc != ls.id:
            shots += C.plan_broll(e0, e1, fps, rng, eid, ent.loc)
        else:
            shots.append(C.ShotSpec(e0, e1, ls.id, "extreme_wide", "pan", eid, params={"spread": 40, "azimuth": 0.0,
                                                                                       "height": 1.5}))
        f = e1
    if T - f > 0:
        shots += C.plan_talking(f, T, fps, rng, host.eid, a_cam, [o.eid for o in objs if o.cls == "object"], ls.id)
    shots = C.fill_gaps(shots, T)
    world_map(shots)
    return finish(cast, sets, shots, T, fps, rng, "talking", f"{ls.loc['name']} vlog")


def _broll_host(cast, sets, host, rng, T, fps):
    """Second location for vlog b-roll: the same person (shared identity) walking a street or trail."""
    name = str(rng.choice(["city_street", "forest_trail"]))
    ls2 = S.make(name, f"loc{len(sets)}", rng)
    sets.append(ls2)
    if name == "city_street":
        sw = np.asarray(ls2.anchors["sidewalks"][int(rng.integers(2))], float)
        a, b = (sw[0], sw[1]) if rng.random() < 0.5 else (sw[1], sw[0])
        start = a + (b - a) * rng.uniform(0.15, 0.35)
        waypoints = [a + (b - a) * 0.95]
        yaw = float(np.arctan2(*(b - a)))
    else:
        trail = ls2.anchors["trail"]
        s0 = rng.uniform(0.1, 0.3) * trail.length
        start = trail.pos(s0)
        waypoints = trail.pos(np.arange(s0 + 2, min(trail.length - 5, s0 + 80), 2.0))
        yaw = float(trail.heading(s0))
    h2 = cast.person(ls2.id, host.name, params=host.params["appearance"], mesh=host.mesh, scale=host.params["scale"],
                     identity=host.identity, pos=tuple(start), yaw=yaw, ground=ls2.world.ground)
    h2.animator.walk(np.asarray(waypoints), speed=rng.uniform(1.1, 1.4))
    h2.track = h2.animator.build()
    return h2.eid


def _plan_sip(cast, ls, host, rng, t_end):
    """Put a mug within reach of the host and schedule reach / drink / put back overlays."""
    an = host.animator
    pel = an.seat[0] if an.seat is not None else np.array([*an.center[:1], 1.0, an.center[1]])
    yaw = an.yaw
    fwd = np.array([np.sin(yaw), 0, np.cos(yaw)])
    best = None
    for sf in ls.anchors.get("surfaces", []):
        p = np.asarray(sf["pos"], float)
        for side, sx in (("r", -1), ("l", 1)):
            right = np.array([np.cos(yaw), 0, -np.sin(yaw)]) * sx
            spot = pel + fwd * 0.38 + right * 0.22
            spot[1] = p[1]
            half = np.asarray(sf["size"]) / 2
            local = spot - p
            c, s_ = np.cos(sf["yaw"]), np.sin(sf["yaw"])
            lx, lz = local[0] * c - local[2] * s_, local[0] * s_ + local[2] * c
            lx, lz = np.clip(lx, -half[0], half[0]), np.clip(lz, -half[1], half[1])
            snap = p + np.array([lx * c + lz * s_, 0, -lx * s_ + lz * c])
            shoulder_h = pel[1] + 0.45 * host.params["scale"]
            d = np.linalg.norm(snap - (pel + [0, 0.45, 0]))
            if d < 0.62 * host.params["scale"] + 0.1 and snap[1] < shoulder_h and snap[1] > pel[1] - 0.6:
                if best is None or d < best[0]:
                    best = (d, snap, side)
    if best is None:
        return None
    _, spot, side = best
    mesh = A.mug(rng)
    h, r = _ground_obj(mesh)
    mug = cast.obj(ls.id, "mug", mesh, static_track(cast.W, spot + [0, 0.002, 0], rng.uniform(0, 6)), h, r)
    t1 = float(rng.uniform(2.0, max(2.5, t_end - 6.0)))
    grip = spot + [0, 0.05, 0] - (spot - pel) * np.array([1, 0, 1]) / max(np.linalg.norm((spot - pel)[[0, 2]]), 1e-6) * 0.07
    an.reach(t1, grip, side, 1.3)
    an.hold(t1 + 0.4, t1 + 3.4, side, "carry")
    an.drink(t1 + 0.6, 2.4, side)
    an.reach(t1 + 3.6, grip, side, 1.3)
    fps = cast.fps
    return mug, int(round(t1 * fps)), int(round((t1 + 3.6) * fps)), float(spot[1] + 0.002), side


def _thrown_ball(cast, ls, rng, T, fps):
    """A ball thrown in from one side of the room: ballistic flight, bounces, rolling."""
    room = ls.anchors["room"]
    w, d = room["w"], room["d"]
    side = rng.choice([-1, 1])
    p0 = np.array([side * (w / 2 - 0.3), rng.uniform(0.9, 1.4), rng.uniform(-d / 4, d / 3)])
    v0 = np.array([-side * rng.uniform(2.5, 4.0), rng.uniform(0.5, 2.0), rng.uniform(-1.0, 1.0)])
    t0 = float(rng.uniform(1.0, max(1.5, T / fps - 6)))
    mesh = A.ball(rng, 0.11)
    tr = ball_flight(T, fps, p0, v0, 0.11, restitution=rng.uniform(0.6, 0.78), t0=t0,
                     bounds=((-w / 2, -d / 2), (w / 2, d / 2)))
    b = cast.obj(ls.id, "ball", mesh, tr, 0.22, 0.11)      # origin at the ball centre (it spins about it)
    b.params["present_from"] = int(round(t0 * fps)) - 1
    b.params["center_h"] = 0.0
    return b


def scene_interview(rng, T, fps):
    loc_name = str(rng.choice(["podcast_studio", "podcast_studio", "living_room"]))
    ls = S.make(loc_name, "loc0", rng)
    cast = Cast(rng, fps, T)
    seats = ls.anchors["seats"]
    if loc_name == "living_room":
        arm = next(s for s in seats if s["kind"] == "armchair")
        couch = [s for s in seats if s["kind"] == "couch"]
        cs = min(couch, key=lambda s: np.linalg.norm(np.asarray(s["pos"]) - np.asarray(arm["pos"])))
        pair = [cs, arm]
    else:
        pair = seats[:2]
    a = _seat_person(cast, ls, pair[0], "host", rng)
    b = _seat_person(cast, ls, pair[1], "guest", rng)
    eye = lambda e, s: np.array([s["pos"][0], s["pos"][1] + 0.86 * e.params["scale"], s["pos"][2]])
    pa, pb = eye(a, pair[0]), eye(b, pair[1])
    # turn taking
    turns = []
    t = 0
    spk = a if rng.random() < 0.6 else b
    while t < T:
        L = int(np.clip(rng.lognormal(np.log(4.5), 0.5), 1.5, 10) * fps)
        turns.append((t, min(T, t + L), spk.eid))
        t += L + int(rng.uniform(0.1, 0.5) * fps)
        spk = b if spk is a else a
    for t0, t1, sid in turns:
        s_, l_ = (a, b) if sid == a.eid else (b, a)
        s_.animator.talk(t0 / fps, t1 / fps, intensity=rng.uniform(0.7, 1.2))
        l_.animator.listen(t0 / fps, t1 / fps)
    a.animator.look_at(0, T / fps + 1, pb)
    b.animator.look_at(0, T / fps + 1, pa)
    place_objects(cast, ls, rng, n=int(rng.integers(1, 3)), kinds=("mug", "bottle", "mug"))
    cast.build_people()
    shots, side = C.plan_interview(T, fps, rng, a.eid, b.eid, turns, ls.id, pa, pb)
    world_map(shots)
    return finish(cast, [ls], shots, T, fps, rng, "interview", f"{ls.loc['name']} conversation")


def scene_dance(rng, T, fps):
    loc_name = str(rng.choice(["dance_studio", "stage"]))
    ls = S.make(loc_name, "loc0", rng)
    cast = Cast(rng, fps, T)
    n = int(rng.choice([1, 1, 2, 3]))
    x0, z0, x1, z1 = ls.anchors["dance_floor"]
    cz = (z0 + z1) / 2
    bpm = float(rng.uniform(90, 128))
    beat0 = float(rng.uniform(0, 60 / bpm))
    style = str(rng.choice(DANCE_STYLES))
    dancers = []
    ground = ls.world.ground
    spacing = 1.6
    for i in range(n):
        x = (i - (n - 1) / 2) * spacing + rng.uniform(-0.2, 0.2)
        z = cz + rng.uniform(-0.6, 0.6) - (0.8 if (n == 3 and i == 1) else 0)
        d = cast.person(ls.id, f"dancer {i + 1}", pos=(x, z), yaw=ls.anchors["dance_facing"], ground=ground,
                        energy=float(rng.uniform(0.8, 1.2)))
        st = style if rng.random() < 0.75 else str(rng.choice(DANCE_STYLES))
        intro = rng.uniform(0.0, 1.0)
        d.animator.stand(intro)
        d.animator.dance(T / fps - intro + 0.5, st, bpm, beat0)
        dancers.append(d)
    cast.build_people()
    shots = C.plan_dance(T, fps, rng, [d.eid for d in dancers], ls.id, bpm, beat0, ls.anchors["focus"])
    world_map(shots)
    state, _ = finish(cast, [ls], shots, T, fps, rng, "dance", f"{ls.loc['name']} dance")
    state["meta_extra"] = {"music": {"bpm": round(bpm, 3), "beat0": round(beat0, 4)}}
    return state, None


def _trail_walk(anim, trail, s0, s1, speed, rng, stops=True):
    """Walk a person along a trail polyline from arc s0 to s1, with occasional pauses to look around."""
    s = s0
    while s < s1 - 1:
        seg = min(s1, s + rng.uniform(18, 45))
        ss = np.arange(s, seg, 2.0)[1:]
        if len(ss) < 1:
            break
        pts = trail.pos(ss)
        anim.walk(pts, speed=speed * rng.uniform(0.9, 1.1))
        s = seg
        if stops and rng.random() < 0.4:
            anim.stand(rng.uniform(1.0, 3.0))
            if rng.random() < 0.5:
                p = trail.pos(min(trail.length, s + 30))
                anim.point(anim.t - 1.0, 1.2, [p[0], 2.0, p[1]], "r")
    return anim


def scene_hiking(rng, T, fps):
    n_loc = int(rng.choice([1, 2, 2, 3]))
    variants = list(rng.permutation(S.TRAIL_VARIANTS))[:n_loc]
    seg_len = [int(T * w) for w in rng.dirichlet(np.ones(n_loc) * 4)]
    seg_len[-1] = T - sum(seg_len[:-1])
    sets, segments = [], []
    f = 0
    for i, v in enumerate(variants):
        sets.append(S.nature_trail(f"loc{i}", rng, v))
        segments.append((f"loc{i}", f, f + seg_len[i]))
        f += seg_len[i]
    # world timeline: video time plus skips at some cuts
    W = T * 4
    cast = Cast(rng, fps, W)
    n_hikers = int(rng.choice([1, 1, 2]))
    looks = [A.humanoid_params(rng) for _ in range(n_hikers)]
    for P in looks:
        P["backpack"] = True
        P["top"] = str(rng.choice(["jacket", "longsleeve", "tshirt"]))
        P["bottom"] = str(rng.choice(["pants", "shorts"]))
    meshes = [A.humanoid(P, rng) for P in looks]
    hikers = {}
    for ls in sets:
        trail = ls.anchors["trail"]
        hikers[ls.id] = []
        s0 = rng.uniform(0.05, 0.25) * trail.length
        for h in range(n_hikers):
            sh = s0 - h * rng.uniform(2.5, 4.0)
            p0 = trail.pos(max(sh, 0.5))
            e = cast.person(ls.id, f"hiker {h + 1}", params=looks[h], mesh=meshes[h][0], scale=meshes[h][1],
                            identity=f"hiker{h}", pos=(p0[0], p0[1]), yaw=float(trail.heading(max(sh, 0.5))),
                            ground=ls.world.ground)
            _trail_walk(e.animator, trail, max(sh, 0.5), trail.length - 5, rng.uniform(1.1, 1.45), rng, stops=(h == 0))
            hikers[ls.id].append(e.eid)
    cast.build_people()
    shots = C.plan_hiking(T, fps, rng, hikers, segments)
    # time skips: a cut inside a location skips 2-12 s of walking; a new location restarts its own clock

    def skip(i, s):
        if s.loc != shots[i - 1].loc:
            return 0
        return int(rng.uniform(2, 12) * fps) if rng.random() < 0.6 else 0
    w = 0
    for i, s in enumerate(shots):
        if i == 0 or s.loc != shots[i - 1].loc:
            w = int(rng.uniform(1, 4) * fps)
        else:
            w += skip(i, s)
        w = min(w, W - s.n - 1)
        s.world = np.arange(w, w + s.n)
        w += s.n
    return finish(cast, sets, shots, T, fps, rng, "hiking", " to ".join(v for v in variants) + " hike",
                  time_skips=True)


def scene_driving(rng, T, fps):
    loc_name = str(rng.choice(["highway", "highway", "city_street"]))
    ls = S.make(loc_name, "loc0", rng)
    W = T * 4 if loc_name == "highway" else T * 2
    cast = Cast(rng, fps, W)
    roads = ls.anchors["roads"]
    y_off = 0.06 if loc_name == "highway" else 0.0
    ground = (lambda x, z, g=ls.world.ground: g(x, z) + y_off)
    main = [r for r in roads if not r.get("cross")]
    ego_road = main[0] if loc_name == "city_street" else main[int(rng.integers(len(main)))]
    lanes = {}
    for k, r in enumerate(main):
        lanes[k] = Path(np.asarray(r["points"])[:, [0, 2]], smooth=True, spacing=0.5)
    ego_k = main.index(ego_road)
    ego_path = lanes[ego_k]
    kinds = ["sedan", "hatch", "suv"]
    m, dims = A.car(rng, str(rng.choice(kinds)))
    m = A.merge(m, *m.get("extras", []))
    speed = ego_road["speed"] * rng.uniform(0.85, 1.05)
    ego_tr = drive(ego_path, W, fps, dims, speed=speed, t_start=-1, s_start=rng.uniform(5, 30), ground=ground)
    ego = cast.obj(ls.id, "ego car", m, ego_tr, dims["height"], dims["length"] / 2, "vehicle")
    ego.skeleton = A.vehicle_skeleton_doc("car", dims, f"sk_{ego.eid}")
    ego.params["dims"] = dims
    ego.params["width"] = dims["width"]
    # traffic: same-lane cars share a speed so nobody overlaps
    for k, path in lanes.items():
        n_cars = int(rng.integers(1, 4))
        lane_speed = speed if k == ego_k else main[k]["speed"] * rng.uniform(0.85, 1.1)
        taken = [ego_tr["arc"][0]] if k == ego_k else []
        for _ in range(n_cars):
            for _ in range(20):
                s0 = rng.uniform(0, min(path.length * 0.6, 40 + lane_speed * W / fps * 0.3))
                if all(abs(s0 - x) > 25 for x in taken):
                    break
            taken.append(s0)
            truck = rng.random() < 0.2
            mm, dd = A.truck(rng) if truck else A.car(rng, str(rng.choice(kinds)))
            mm = A.merge(mm, *mm.get("extras", []))
            tr = drive(path, W, fps, dd, speed=lane_speed, t_start=-1, s_start=s0, ground=ground)
            o = cast.obj(ls.id, "truck" if truck else "car", mm, tr, dd["height"], dd["length"] / 2, "vehicle")
            o.skeleton = A.vehicle_skeleton_doc("car", dd, f"sk_{o.eid}")
            o.params["width"] = dd["width"]
    if loc_name == "city_street":
        for i in range(int(rng.integers(2, 6))):
            sw = ls.anchors["sidewalks"][int(rng.integers(2))]
            a, b = np.asarray(sw[0], float), np.asarray(sw[1], float)
            if rng.random() < 0.5:
                a, b = b, a
            u0 = rng.uniform(0.2, 0.6)
            p0 = a + (b - a) * u0
            e = cast.person(ls.id, f"pedestrian {i + 1}", pos=tuple(p0), yaw=float(np.arctan2(*(b - a))), ground=ls.world.ground)
            e.animator.walk([a + (b - a) * min(0.98, u0 + 0.5)], speed=rng.uniform(1.1, 1.5))
    cast.build_people()
    shots = C.plan_driving(T, fps, rng, ego.eid, ls.id)

    def skip(i, s):
        return int(rng.uniform(2, 10) * fps) if (rng.random() < 0.5 and loc_name == "highway") else 0
    world_map(shots, skip)
    # keep the world timeline inside the simulated range
    over = max(s.world[-1] for s in shots) - (W - 1)
    if over > 0:
        for s in shots:
            s.world = np.clip(s.world - over, 0, W - 1)
    for s in shots:
        if s.move == "fpv":
            s.params["host_track"] = ego.track
            s.params["host"] = ego.eid
            s.params["mount"] = dims["dash"]
    return finish(cast, [ls], shots, T, fps, rng, "driving", f"{ls.loc['name']} drive", time_skips=True)


def _flat_region_path(hm, origin, start_xz, rng, n_pts=6, radius=12, vox=None):
    """Random walk waypoints on voxel cells at the same height as the start (no climbing)."""
    X, Z = hm.shape
    ix, iz = int(start_xz[0] - origin[0]), int(start_xz[1] - origin[2])
    h0 = hm[ix, iz]
    from collections import deque
    seen = np.zeros_like(hm, bool)
    q = deque([(ix, iz)])
    seen[ix, iz] = True
    cells = []
    while q:
        x, z = q.popleft()
        cells.append((x, z))
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = x + dx, z + dz
            if 1 <= a < X - 1 and 1 <= b < Z - 1 and not seen[a, b] and hm[a, b] == h0 and abs(a - ix) + abs(b - iz) < radius * 2:
                seen[a, b] = True
                q.append((a, b))
    cells = np.array(cells)
    # interior cells only (all 8 neighbours flat) so feet never hang over an edge
    ok = np.array([np.all(hm[max(x - 1, 0):x + 2, max(z - 1, 0):z + 2] == h0) for x, z in cells])
    if vox is not None:
        # nothing solid at body height (tree trunks, walls) in the cell or its neighbours
        hi = int(round(h0 - origin[1]))          # first air cell above the walkable surface
        from scipy.ndimage import binary_dilation
        body = binary_dilation((vox[:, hi:hi + 3, :] > 0).any(1), np.ones((5, 5), bool))   # 2 blocks of clearance
        ok &= ~body[cells[:, 0], cells[:, 1]]
    cells = cells[ok]
    return cells, h0


def scene_voxel(rng, T, fps):
    n_loc = int(rng.choice([1, 2]))
    sets = [S.voxel_world(f"loc{i}", rng) for i in range(n_loc)]
    seg_len = [T] if n_loc == 1 else [int(T * rng.uniform(0.45, 0.6))]
    if n_loc == 2:
        seg_len.append(T - seg_len[0])
    segments, f = [], 0
    for i, ls in enumerate(sets):
        segments.append((ls.id, f, f + seg_len[i]))
        f += seg_len[i]
    W = T * 3
    cast = Cast(rng, fps, W)
    P = A.humanoid_params(rng, "blocky")
    P["height"] = 1.8
    mesh, s = A.humanoid(P, rng)
    players = {}
    for ls in sets:
        hm, origin = ls.anchors["heightmap"], ls.anchors["voxel_origin"]
        house = ls.anchors["house"]
        lo, hi = np.asarray(house["lo"]), np.asarray(house["hi"])
        door = np.asarray(house["door"])
        start = door[[0, 2]] + [0, 4.5]
        cells, h0 = _flat_region_path(hm, origin, start, rng, vox=ls.anchors["voxels"])
        cxz = cells + 0.5 + origin[[0, 2]]
        outside = ~((cxz[:, 0] > lo[0] - 4) & (cxz[:, 0] < hi[0] + 4) & (cxz[:, 1] > lo[2] - 4) & (cxz[:, 1] < hi[2] + 4))
        if outside.sum() >= 12:
            cxz = cxz[outside]
        if len(cxz) < 12:     # tiny plateau: walk on any flat cell around the start
            cells, h0 = _flat_region_path(hm, origin, start, rng, radius=20)
            cxz = cells + 0.5 + origin[[0, 2]] if len(cells) else np.asarray([start], float)
        e = cast.person(ls.id, "player", params=P, mesh=mesh, scale=s, identity="player", pos=tuple(start), yaw=0.0,
                        ground=ls.world.ground)
        an = e.animator
        cur = np.asarray(start, float)
        while an.t < W / fps - 4:
            # pick a reachable waypoint 4-10 m away with a straight line over flat cells
            for _ in range(30):
                tgt = cxz[rng.integers(len(cxz))]
                d = np.linalg.norm(tgt - cur)
                if 3 < d < 11:
                    line = cur[None] + (tgt - cur)[None] * np.linspace(0, 1, 24)[:, None]
                    ii = np.floor(line - origin[[0, 2]]).astype(int)
                    if np.all(hm[ii[:, 0], ii[:, 1]] == h0) and np.all(np.min(np.linalg.norm(line[:, None] - cxz[None], axis=2), 1) < 0.8):
                        break
            else:
                an.stand(1.0)
                continue
            r = rng.random()
            if r < 0.15:
                an.jump(0.45)
            elif r < 0.3:
                an.stand(rng.uniform(0.5, 1.5))
            if np.linalg.norm(tgt - cur) > 1:
                an.walk([tgt], speed=rng.choice([1.4, 2.6, 4.3]) if r > 0.5 else 1.4,
                        gait="run" if r > 0.85 else "walk")
                cur = np.asarray(tgt, float)
        players[ls.id] = e.eid
    cast.build_people()
    shots = C.plan_voxel(T, fps, rng, None, segments)
    for sh in shots:
        sh.subject = players[sh.loc]
    w = 0
    for i, sh in enumerate(shots):
        if i == 0 or sh.loc != shots[i - 1].loc:
            w = int(rng.uniform(0, 2) * fps)
        else:
            w += int(rng.uniform(2, 10) * fps) if rng.random() < 0.6 else 0
        w = min(w, W - sh.n - 1)
        sh.world = np.arange(w, w + sh.n)
        w += sh.n
    for sh in shots:
        if sh.move == "fpv":
            e = next(x for x in cast.ents if x.eid == sh.subject)
            sh.params.update({"host_track": e.track, "host": e.eid, "scale": e.params["scale"]})
    return finish(cast, sets, shots, T, fps, rng, "voxel", "voxel world gameplay", time_skips=True)


SCENES = {"talking": scene_talking, "interview": scene_interview, "dance": scene_dance, "hiking": scene_hiking,
          "driving": scene_driving, "voxel": scene_voxel}


# ---------------------------------------------------------------- shared finishing

def finish(cast: Cast, sets, shots, T, fps, rng, genre, title, time_skips=False, errors=False):
    """Solve cameras, map to video time, compute states, assemble the document."""
    worlds = {ls.id: ls.world for ls in sets}
    aspect = 16 / 9
    cams = []
    for s in shots:
        if s.move == "fpv" and "host_track" not in s.params:
            e = next(x for x in cast.ents if x.eid == s.subject)
            s.params.update({"host_track": e.track, "host": e.eid, "scale": e.params.get("scale", 1.0)})
        solver = C.CameraSolver(worlds[s.loc], cast.subjects(s.loc), fps, aspect, rng)
        cams.append(solver.solve(s))
    video, world_of = _video_tracks(cast, shots, T)
    # presence: entities exist in their location; identity copies are absent while the edit is elsewhere
    loc_of = np.empty(T, object)
    for s in shots:
        loc_of[s.start:s.end] = s.loc
    multi = {}
    for e in cast.ents:
        multi.setdefault(e.identity, set()).add(e.loc)
    present = {}
    for e in cast.ents:
        if len(multi[e.identity]) > 1:
            present[e.eid] = loc_of == e.loc
        else:
            present[e.eid] = np.ones(T, bool)
        if "present_from" in e.params:
            present[e.eid] = present[e.eid] & (world_of >= e.params["present_from"])
    return {"cast": cast, "sets": sets, "shots": shots, "cams": cams, "video": video, "world_of": world_of,
            "present": present, "T": T, "fps": fps, "genre": genre, "title": title, "worlds": worlds,
            "time_skips": time_skips, "rng": rng}, None


def assemble(state, seed, errors=False, video_name=None, width=640, height=360):
    cast, sets, shots, cams = state["cast"], state["sets"], state["shots"], state["cams"]
    T, fps = state["T"], state["fps"]
    rng = np.random.default_rng(seed + 7919)
    video, present = state["video"], state["present"]
    continuity = []
    if errors:
        before = compute_states(cast, shots, cams, T, fps, 16 / 9, state["worlds"], video, present)
        continuity = inject_errors(cast, shots, video, present, rng, T, fps, seen=before)
    states = compute_states(cast, shots, cams, T, fps, 16 / 9, state["worlds"], video, present)
    db = DocBuilder(fps, T, state["title"], seed, state["genre"])
    db.meta.update(state.get("meta_extra", {}))
    if video_name:
        db.meta["source"] = {"video": video_name, "width": width, "height": height}
    for ls in sets:
        db.add_location(ls)
    for e in cast.ents:
        mid = db.add_mesh(f"mesh_{e.eid}" if "mesh_id" not in e.params else e.params["mesh_id"], e.mesh)
        tr = video[e.eid]
        track = {"state": states[e.eid], "position": np.asarray(tr["position"], np.float32),
                 "rotation": np.asarray(quat_continuous(tr["rotation"]), np.float32)}
        if "pose" in tr:
            track["pose"] = np.asarray(tr["pose"], np.float32)
        if "morph" in tr and e.mesh.get("morphs"):
            track["morph"] = np.asarray(tr["morph"], np.float32)
        if "contacts" in tr:
            track["contacts"] = np.asarray(tr["contacts"], np.uint8)
        ent = {"id": e.eid, "name": e.name, "class": e.entity_class, "location": e.loc, "mesh": mid, "track": track,
               "identity": e.identity}
        if e.skeleton is not None:
            db.skeletons.append(e.skeleton)
            ent["skeleton"] = e.skeleton["id"]
        db.entities.append(ent)
    world_prev = None
    for i, (s, (p, q, f, info)) in enumerate(zip(shots, cams)):
        sid, cid = f"s{i}", f"cam{i}"
        db.cameras.append({"id": cid, "shot": sid, "track": {"position": p, "rotation": q, "fov": f}})
        sh = {"id": sid, "start": int(s.start), "end": int(s.end), "location": str(s.loc), "camera": cid,
              "size": str(s.size), "move": str(_move_name(s)), "transition": str(s.transition)}
        if s.subject:
            sh["subject"] = str(s.subject)
        if s.params.get("host"):
            sh["attached_to"] = s.params["host"]
        sh["world_frame"] = int(s.world[0])
        if world_prev is not None and s.loc == shots[i - 1].loc and int(s.world[0]) != world_prev:
            sh["time_skip"] = round((int(s.world[0]) - world_prev) / fps, 3)
        world_prev = int(s.world[-1]) + 1
        sh["focal_mm"] = round(float(C.focal_from_fov(float(np.mean(f)))), 1)
        db.shots.append(sh)
    db.continuity = continuity
    return db.doc()


def _move_name(s):
    m = s.move
    if m == "follow":
        return "handheld" if s.params.get("handheld") else "truck"
    return m


# ---------------------------------------------------------------- continuity errors

def inject_errors(cast, shots, video, present, rng, T, fps, n=None, seen=None):
    """Deliberate continuity mistakes at cuts; tracks are modified in place and recorded.

    With `seen` (states before the errors), the changed thing is chosen among what is on
    screen around the cut, so the mistake is visible in the video.
    """
    out = []
    objs = [e for e in cast.ents if e.cls == "object"]
    people = [e for e in cast.ents if e.cls == "person"]
    cuts = [i for i in range(1, len(shots)) if shots[i].loc == shots[i - 1].loc]
    if not cuts:
        return out
    n = n or int(rng.integers(1, 4))
    chosen = sorted(rng.choice(cuts, size=min(n, len(cuts)), replace=False))
    for ci in chosen:
        f = shots[ci].start
        loc = shots[ci].loc
        a, b = shots[ci - 1], shots[ci]

        def on(e, sh):
            return seen is None or (seen[e.eid][sh.start:sh.end] == 2).mean() > 0.3
        lo = [o for o in objs if o.loc == loc]
        kinds = []
        cand = {"moved": [o for o in lo if on(o, a) and on(o, b)], "appeared": [o for o in lo if on(o, b)],
                "vanished": [o for o in lo if on(o, a)]}
        for k, v in cand.items():
            if v:
                kinds += [k] * (2 if k == "moved" else 1)
        lp = [p for p in people if p.loc == loc and on(p, a) and on(p, b)]
        if lp:
            kinds += ["wardrobe"]
        if not kinds:
            continue
        kind = str(rng.choice(kinds))
        if kind in ("moved", "appeared", "vanished"):
            pool = cand[kind]
            o = pool[int(rng.integers(len(pool)))]
            tr = video[o.eid]
            if kind == "moved":
                d = rng.uniform(0.15, 0.45) * np.array([np.cos(a := rng.uniform(0, 2 * np.pi)), 0, np.sin(a)])
                tr["position"] = tr["position"].copy()
                tr["position"][f:] += d.astype(np.float32)
                detail = f"{o.name} is {np.linalg.norm(d):.2f} m from where it was when the shot cut away"
                sev = float(np.clip(np.linalg.norm(d) / 0.5, 0.2, 1.0))
            elif kind == "appeared":
                present[o.eid] = present[o.eid].copy()
                present[o.eid][:f] = False
                detail = f"{o.name} was not there before the cut"
                sev = 0.7
            else:
                present[o.eid] = present[o.eid].copy()
                present[o.eid][f:] = False
                detail = f"{o.name} disappears at the cut"
                sev = 0.7
            out.append({"kind": kind, "entity": o.eid, "from_shot": f"s{ci - 1}", "to_shot": f"s{ci}", "frame": int(f),
                        "detail": detail, "severity": round(sev, 2)})
        else:
            p = lp[int(rng.integers(len(lp)))]
            P = dict(p.params["appearance"])
            old = P["top_color"] if P["top"] != "jacket" else P["jacket_color"]
            new = A.rgb(A.CLOTH_COLORS[int(rng.integers(len(A.CLOTH_COLORS)))])
            if np.abs(new - old).sum() < 0.2:
                new = A.shade(1 - old, 1.0)
            if P["top"] == "jacket":
                P["jacket_color"] = new
            else:
                P["top_color"] = new
            mesh, s = A.humanoid(P, np.random.default_rng(int(rng.integers(1 << 30))))
            # same geometry as the original, only colours differ: rebuild with the original rng state is not
            # available, so recolour the original mesh where it had the old top colour instead
            m2 = dict(p.mesh)
            col = p.mesh["colors"].copy()
            hit = np.all(np.abs(col - old[None]) < 1e-4, axis=1)
            col[hit] = new
            m2["colors"] = col
            twin = Ent(cast._id(), p.name, "person", p.loc, m2, track=p.track, skeleton=None, identity=p.identity,
                       params=dict(p.params), entity_class="person", height=p.height, radius=p.radius)
            twin.skeleton = dict(p.skeleton, id=f"sk_{twin.eid}")
            cast.ents.append(twin)
            video[twin.eid] = video[p.eid]
            present[twin.eid] = present[p.eid].copy()
            present[twin.eid][:f] = False
            present[p.eid] = present[p.eid].copy()
            present[p.eid][f:] = False
            out.append({"kind": "wardrobe", "entity": p.eid, "from_shot": f"s{ci - 1}", "to_shot": f"s{ci}", "frame": int(f),
                        "detail": f"{p.name}'s top changes colour at the cut (continues as {twin.eid})",
                        "severity": 0.8, "replacement": twin.eid})
    return out


# ---------------------------------------------------------------- entry point

def generate(genre="random", seed=0, seconds=15.0, fps=30, errors=False, video_name=None, width=640, height=360):
    """Generate a complete scene document. Returns (doc, info) where info holds generator extras."""
    rng = np.random.default_rng(seed)
    if genre == "random":
        genre = str(rng.choice(GENRES))
    T = int(round(seconds * fps))
    state, _ = SCENES[genre](rng, T, fps)
    doc = assemble(state, seed, errors, video_name, width, height)
    return doc, state
