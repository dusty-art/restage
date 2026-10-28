"""Location generators: static sets (one per place a video visits) with props, environment and anchors.

Every generator returns a `LocationSet`:

* `loc`: the location dict for the scene document (id, name, kind, bounds, environment, props),
* `meshes`: prop meshes by id (deduplicated, so 300 trees share a handful of meshes),
* `world`: a spatial.World for camera placement and visibility (boxes, room interior, ground),
* `anchors`: semantic spots for scenarios: seats, surfaces for small objects, free floor,
  walking paths, road lanes, the dance floor, viewpoints ...

All sizes are realistic metres. Rooms are centred at the origin with the floor at y = 0 and
walls at x = +-w/2, z = +-d/2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

from . import assets as A
from .motion import Path
from .spatial import World
from ..scene import quat_axis_angle, quat_mul


@dataclass
class LocationSet:
    loc: dict
    meshes: dict
    world: World
    anchors: dict = field(default_factory=dict)

    @property
    def id(self):
        return self.loc["id"]

    @property
    def kind(self):
        return self.loc["kind"]


class SetBuilder:
    def __init__(self, loc_id, name, kind, rng):
        self.id, self.name, self.kind, self.rng = loc_id, name, kind, rng
        self.meshes: dict = {}
        self.props: list = []
        self.world = World()
        self.anchors: dict = {"seats": [], "surfaces": [], "stand": [], "inserts": []}
        self._n = 0
        self._cache: dict = {}

    def mesh_id(self, mesh, key=None):
        if key is not None and key in self._cache:
            return self._cache[key]
        mid = f"{self.id}_m{len(self.meshes)}"
        self.meshes[mid] = mesh
        if key is not None:
            self._cache[key] = mid
        return mid

    def add(self, cls, mesh, pos=(0, 0, 0), yaw=0.0, scale=1.0, key=None, collide=True, occluder=True, name=None,
            q=None, cast_shadow=True):
        """Place a mesh as a prop; registers glowing sub-meshes and world colliders."""
        q = A.yaw_q(yaw) if q is None else np.asarray(q, float)
        sc = np.broadcast_to(np.asarray(scale, float), (3,)).astype(float)
        pid = name or f"{cls}{self._n}"
        self._n += 1
        mid = self.mesh_id(_strip(mesh), key)
        self.props.append({"id": pid, "class": cls, "mesh": mid, "position": [float(x) for x in pos],
                           "rotation": [float(x) for x in q], "scale": [float(x) for x in sc]})
        if not cast_shadow:
            self.props[-1]["cast_shadow"] = False
        for gi, g in enumerate(mesh.get("extras", [])):
            gid = self.mesh_id(g, None if key is None else (key, "glow", gi))
            self.props.append({"id": f"{pid}_glow{gi}", "class": cls, "mesh": gid, "position": [float(x) for x in pos],
                               "rotation": [float(x) for x in q], "scale": [float(x) for x in sc]})
        if collide:
            for b in A.colliders(mesh):
                lo, hi = A._xform_box(b, pos, q, sc)
                self.world.add_box(lo, hi, pid, occluder)
            from ..scene import quat_rotate
            for sp in mesh.get("spheres", []):
                c = quat_rotate(np.asarray(q, float), np.asarray(sp[:3], float) * sc) + np.asarray(pos, float)
                self.world.add_sphere(c, sp[3] * float(sc.max()))
        return pid

    def finish(self, bounds, environment) -> LocationSet:
        loc = {"id": self.id, "name": self.name, "kind": self.kind,
               "bounds": [[float(x) for x in bounds[0]], [float(x) for x in bounds[1]]],
               "environment": environment, "props": self.props}
        return LocationSet(loc, self.meshes, self.world, self.anchors)


def _strip(m):
    return {k: v for k, v in m.items() if k not in ("extras", "seat", "seats", "spheres")}


def _seat_world(local, pos, yaw):
    q = A.yaw_q(yaw)
    from ..scene import quat_rotate
    return (quat_rotate(q, np.asarray(local, float)) + np.asarray(pos, float)).tolist()


# ---------------------------------------------------------------- environments

def environment(kind, rng, time_of_day=None):
    """Sky colour, ambient, sun direction / intensity and fog for a location kind."""
    tod = time_of_day or rng.choice(["day", "day", "day", "overcast", "golden"])
    az = rng.uniform(0, 2 * np.pi)
    if kind in ("indoor", "stage"):
        el = np.radians(rng.uniform(40, 70))
        sky = A.rgb(["#c9d6e3", "#d8d2c4", "#b8c4cf"][rng.integers(3)])
        sun = float(rng.uniform(0.45, 0.7)) if kind == "indoor" else float(rng.uniform(0.3, 0.5))
        amb = float(rng.uniform(0.42, 0.55)) if kind == "indoor" else float(rng.uniform(0.3, 0.42))
        fog = None
    else:
        if tod == "golden":
            el, sky, sun, amb = np.radians(rng.uniform(14, 26)), A.rgb("#f2c49b"), rng.uniform(1.0, 1.15), rng.uniform(0.5, 0.6)
        elif tod == "overcast":
            el, sky, sun, amb = np.radians(rng.uniform(40, 70)), A.rgb("#b9c2cb"), rng.uniform(0.35, 0.55), rng.uniform(0.55, 0.65)
        else:
            el, sky, sun, amb = np.radians(rng.uniform(30, 65)), A.rgb(["#8fbbe8", "#9cc6ee", "#7fb0e0"][rng.integers(3)]), \
                rng.uniform(0.85, 1.05), rng.uniform(0.35, 0.45)
        fog = None
        if kind == "nature":
            fog = {"color": [float(x) for x in A.mix(sky, A.rgb("#dfe6ea"), 0.4)], "near": 25.0, "far": 160.0}
        elif kind == "street":
            fog = {"color": [float(x) for x in A.mix(sky, A.rgb("#dfe6ea"), 0.4)], "near": 120.0, "far": 600.0}
        elif kind == "voxel":
            fog = {"color": [float(x) for x in sky], "near": 30.0, "far": 70.0}
    sun_dir = [float(-np.cos(el) * np.sin(az)), float(-np.sin(el)), float(-np.cos(el) * np.cos(az))]
    ground = {"indoor": "#8a7f72", "stage": "#3a3a40", "street": "#77786f", "nature": "#6f7a52", "voxel": "#6a8a4a"}.get(
        kind, "#77786f")
    return {"sky": [float(x) for x in sky], "ground": [float(x) for x in A.rgb(ground)], "ambient": float(amb),
            "sun_dir": sun_dir, "sun": float(sun), "fog": fog, "time_of_day": str(tod)}


# ---------------------------------------------------------------- rooms

WALLS = ["#e9e4da", "#d7dde2", "#e6d8c3", "#cfd8c8", "#efe9df", "#c8d3dc", "#e3d1c9", "#9eb0b8", "#ded7cf"]
FLOORS_WOOD = ["#a8794e", "#8a6040", "#c49a6c", "#b5835a", "#7a5538"]
FLOORS_TILE = ["#d9d4cc", "#bfc4c8", "#e8e2d6", "#a7a39b"]


def _room_base(b: SetBuilder, w, d, h, floor_kind, floor_c, wall_c, rng, accent=None):
    walls = [wall_c] * 4
    if accent is not None:
        walls[0] = accent
    shell = A.room_shell(w, d, h, floor_c, wall_c, floor_kind=floor_kind, rng=rng, wall_colors=walls)
    b.add("room", shell, collide=False, cast_shadow=False)
    b.world.interior = (np.array([-w / 2, 0.0, -d / 2]), np.array([w / 2, h, d / 2]))
    b.anchors["room"] = {"w": w, "d": d, "h": h}


def _window(b, wall, along, w=1.2, h=1.3, sill=0.9, outside=None):
    """Place a window on a wall: wall in {'back', 'front', 'left', 'right'}; along = coordinate along the wall."""
    room = b.anchors["room"]
    W, D = room["w"], room["d"]
    pos, yaw = {"back": ((along, 0, -D / 2 + 0.005), 0.0), "front": ((along, 0, D / 2 - 0.005), np.pi),
                "left": ((-W / 2 + 0.005, 0, along), np.pi / 2), "right": ((W / 2 - 0.005, 0, along), -np.pi / 2)}[wall]
    out = outside if outside is not None else A.rgb(["#cfe3f2", "#dbe9f2", "#e8efe0"][b.rng.integers(3)])
    b.add("window", A.window(w, h, sill, outside=out), pos, yaw, collide=False)


def _on_wall(b, cls, mesh, wall, along, y, collide=False, depth_off=0.005):
    room = b.anchors["room"]
    W, D = room["w"], room["d"]
    pos, yaw = {"back": ((along, y, -D / 2 + depth_off), 0.0), "front": ((along, y, D / 2 - depth_off), np.pi),
                "left": ((-W / 2 + depth_off, y, along), np.pi / 2), "right": ((W / 2 - depth_off, y, along), -np.pi / 2)}[wall]
    return b.add(cls, mesh, pos, yaw, collide=collide)


def _surface(b, pos, size, yaw, kind, prop):
    b.anchors["surfaces"].append({"pos": [float(x) for x in pos], "size": [float(x) for x in size], "yaw": float(yaw),
                                  "kind": kind, "prop": prop})


def living_room(loc_id="loc0", rng=None) -> LocationSet:
    rng = np.random.default_rng() if rng is None else rng
    b = SetBuilder(loc_id, "living room", "indoor", rng)
    w, d, h = rng.uniform(5.0, 7.0), rng.uniform(4.4, 6.0), rng.uniform(2.6, 3.0)
    wall = A.rgb(WALLS[rng.integers(len(WALLS))])
    accent = A.rgb(["#5b6c5d", "#3d5a6c", "#a35d4a", "#d9a441", "#6d597a"][rng.integers(5)]) if rng.random() < 0.4 else None
    _room_base(b, w, d, h, "wood", A.rgb(FLOORS_WOOD[rng.integers(len(FLOORS_WOOD))]), wall, rng, accent)
    cx = rng.uniform(-0.5, 0.5)
    couch = A.couch(rng, 3)
    cz = -d / 2 + 0.5
    pid = b.add("couch", couch, (cx, 0, cz), 0.0)
    for s in couch["seats"]:
        b.anchors["seats"].append({"pos": _seat_world(s, (cx, 0, cz), 0.0), "yaw": 0.0, "kind": "couch", "prop": pid})
    b.add("rug", A.rug(rng, 2.6, 1.8), (cx, 0, cz + 1.25), 0.0, collide=False)
    tz = cz + 1.25
    tbl = A.table(rng, 1.1, 0.6, 0.42)
    tid = b.add("table", tbl, (cx, 0, tz), 0.0)
    _surface(b, (cx, 0.42, tz), (1.0, 0.5), 0.0, "coffee_table", tid)
    # TV on a low cabinet against the front wall
    stand = A.box((1.6, 0.5, 0.4), (0, 0.25, 0), A.rgb(A.WOOD[rng.integers(len(A.WOOD))]))
    sid = b.add("cabinet", stand, (cx, 0, d / 2 - 0.22), np.pi)
    b.add("tv", A.tv(1.3), (cx, 0.52, d / 2 - 0.25), np.pi)
    _surface(b, (cx + 0.6, 0.5, d / 2 - 0.22), (0.3, 0.3), np.pi, "cabinet", sid)
    side = rng.choice([-1, 1])
    arm = A.couch(rng, 1)
    ax, az = cx + side * (1.75 + rng.uniform(0, 0.3)), tz - 0.1
    ayaw = -side * np.pi / 2
    aid = b.add("armchair", arm, (ax, 0, az), ayaw)
    b.anchors["seats"].append({"pos": _seat_world(arm["seat"], (ax, 0, az), ayaw), "yaw": float(ayaw), "kind": "armchair", "prop": aid})
    b.add("lamp", A.lamp(rng, "floor"), (cx - side * 1.55, 0, cz - 0.1), 0.0)
    b.add("plant", A.plant(rng, rng.uniform(0.9, 1.5)), (-side * (w / 2 - 0.4), 0, -d / 2 + 0.4), rng.uniform(0, 6))
    b.add("plant", A.plant(rng, rng.uniform(0.5, 1.0)), (side * (w / 2 - 0.35), 0, d / 2 - 0.4), rng.uniform(0, 6))
    sh = A.shelf(rng, 0.9, 1.9, 0.32)
    shx = -side * (w / 2 - 0.17)
    b.add("shelf", sh, (shx, 0, rng.uniform(-0.5, 0.8)), side * np.pi / 2)
    for k in range(rng.integers(1, 4)):
        _on_wall(b, "picture", A.picture_frame(rng, rng.uniform(0.4, 0.9), rng.uniform(0.35, 0.7)), "back",
                 cx + (k - 1) * 0.95 + rng.uniform(-0.1, 0.1), rng.uniform(1.45, 1.75))
    _window(b, "right" if side < 0 else "left", rng.uniform(-0.8, 0.8), 1.4, 1.4, 0.85)
    if rng.random() < 0.6 and w > 5.6:
        _window(b, "back", float(np.clip(cx + side * 2.1, -w / 2 + 0.8, w / 2 - 0.8)), 1.0, 1.4, 0.85)
    b.add("door", A.door(), (side * (w / 2 - 0.8), 0, d / 2 - 0.01), np.pi, collide=False)
    b.anchors["free_floor"] = [cx - 1.4, tz + 0.6, cx + 1.4, d / 2 - 0.7]
    b.anchors["focus"] = [cx, 0.9, cz + 0.3]
    b.anchors["stand"].append({"pos": [cx, 0, tz + 1.1], "yaw": float(np.pi)})
    b.anchors["stand"].append({"pos": [cx + side * 0.9, 0, tz + 1.2], "yaw": 0.0})
    return b.finish(([-w / 2, 0, -d / 2], [w / 2, h, d / 2]), environment("indoor", rng))


def kitchen(loc_id="loc0", rng=None) -> LocationSet:
    rng = np.random.default_rng() if rng is None else rng
    b = SetBuilder(loc_id, "kitchen", "indoor", rng)
    w, d, h = rng.uniform(4.6, 6.0), rng.uniform(3.8, 5.0), rng.uniform(2.6, 2.9)
    wall = A.rgb(WALLS[rng.integers(len(WALLS))])
    _room_base(b, w, d, h, "tiles", A.rgb(FLOORS_TILE[rng.integers(len(FLOORS_TILE))]), wall, rng)
    fside = rng.choice([-1, 1])
    run = w - 1.3
    rx = -fside * 0.6
    cab = A.rgb(["#f4f1ea", "#3d5a6c", "#7b8c6e", "#c49a6c", "#2f3e46", "#d8cbb5"][rng.integers(6)])
    top = A.rgb(["#e8e6e1", "#3a3a3a", "#c9b79c", "#8a8e93"][rng.integers(4)])
    cid = b.add("counter", A.counter(rng, run, 0.62, 0.9, top, cab, sink=True, stove=True), (rx, 0, -d / 2 + 0.31), 0.0)
    _surface(b, (rx - run * 0.05, 0.9, -d / 2 + 0.35), (run * 0.35, 0.4), 0.0, "counter", cid)
    b.add("cabinet", A.upper_cabinets(rng, run, 0.35, 0.7, cab), (rx, 1.5, -d / 2 + 0.175), 0.0)
    b.add("fridge", A.fridge(rng), (fside * (w / 2 - 0.4), 0, -d / 2 + 0.38), 0.0)
    # window above the sink: give it a backsplash-free spot (between cabinets in real kitchens)
    _window(b, rng.choice(["left", "right"]) if rng.random() < 0.5 else ("right" if fside < 0 else "left"),
            rng.uniform(-0.4, 0.6), 1.2, 1.2, 1.0)
    # dining table with chairs
    tw, td = rng.uniform(1.2, 1.6), 0.8
    tx, tz = rng.uniform(-0.4, 0.4), rng.uniform(0.3, 0.8)
    tid = b.add("table", A.table(rng, tw, td, 0.75), (tx, 0, tz), 0.0)
    _surface(b, (tx, 0.75, tz), (tw - 0.2, td - 0.2), 0.0, "table", tid)
    chair_c = A.rgb(A.WOOD[rng.integers(len(A.WOOD))])
    for k, (dx, dz, yaw) in enumerate([(-tw / 4, -td / 2 - 0.3, 0.0), (tw / 4, -td / 2 - 0.3, 0.0),
                                       (-tw / 4, td / 2 + 0.3, np.pi), (tw / 4, td / 2 + 0.3, np.pi)]):
        if rng.random() < 0.85:
            ch = A.chair(rng, chair_c)
            cpid = b.add("chair", ch, (tx + dx, 0, tz + dz), yaw)
            b.anchors["seats"].append({"pos": _seat_world(ch["seat"], (tx + dx, 0, tz + dz), yaw), "yaw": float(yaw),
                                       "kind": "chair", "prop": cpid})
    b.add("plant", A.plant(rng, rng.uniform(0.35, 0.6)), (rx + run * 0.38, 0.9, -d / 2 + 0.3), 0.0)
    b.add("plant", A.plant(rng, rng.uniform(0.8, 1.3)), (-fside * (w / 2 - 0.4), 0, d / 2 - 0.4), 0.0)
    b.add("picture", A.picture_frame(rng, 0.5, 0.4), (fside * (w / 2 - 0.005), 1.6, rng.uniform(0.2, 1.0)),
          -fside * np.pi / 2, collide=False)
    b.add("door", A.door(), (fside * (w / 2 - 0.9), 0, d / 2 - 0.01), np.pi, collide=False)
    b.anchors["free_floor"] = [-w / 2 + 0.7, -d / 2 + 0.95, w / 2 - 0.7, tz - td / 2 - 0.7]
    b.anchors["stand"].append({"pos": [rx, 0, -d / 2 + 1.05], "yaw": 0.0})       # in front of the counter, facing out
    b.anchors["stand"].append({"pos": [rx + 0.5, 0, -d / 2 + 0.95], "yaw": float(np.pi)})   # facing the counter
    b.anchors["focus"] = [rx, 1.1, -d / 2 + 0.8]
    return b.finish(([-w / 2, 0, -d / 2], [w / 2, h, d / 2]), environment("indoor", rng))


def office(loc_id="loc0", rng=None) -> LocationSet:
    rng = np.random.default_rng() if rng is None else rng
    b = SetBuilder(loc_id, "office", "indoor", rng)
    w, d, h = rng.uniform(3.6, 5.0), rng.uniform(3.4, 4.6), rng.uniform(2.6, 2.9)
    wall = A.rgb(WALLS[rng.integers(len(WALLS))])
    _room_base(b, w, d, h, "wood" if rng.random() < 0.6 else "plain", A.rgb(FLOORS_WOOD[rng.integers(len(FLOORS_WOOD))])
               if rng.random() < 0.6 else A.rgb("#8e9aaf"), wall, rng, A.rgb("#3d5a6c") if rng.random() < 0.3 else None)
    dx, dz = rng.uniform(-0.3, 0.3), -d / 2 + 1.35
    dsk = A.desk(rng, 1.4, 0.7)
    did = b.add("desk", dsk, (dx, 0, dz), 0.0)
    _surface(b, (dx + 0.35, 0.755, dz + 0.1), (0.5, 0.35), 0.0, "desk", did)
    ch = A.office_chair(rng)
    cz = dz - 0.62
    chid = b.add("chair", ch, (dx, 0, cz), 0.0)
    b.anchors["seats"].append({"pos": _seat_world(ch["seat"], (dx, 0, cz), 0.0), "yaw": 0.0, "kind": "office_chair", "prop": chid})
    lp = A.laptop(rng)
    b.add("laptop", lp, (dx - 0.1, 0.755, dz + 0.02), np.pi, collide=False)
    for k in range(2):
        b.add("shelf", A.shelf(rng, 0.8, rng.uniform(1.6, 2.0), 0.3), (-0.55 + 1.1 * k + dx * 0.3, 0, -d / 2 + 0.16), 0.0)
    b.add("plant", A.plant(rng, rng.uniform(0.9, 1.4)), (w / 2 - 0.4, 0, -d / 2 + 0.4), 0.0)
    b.add("plant", A.plant(rng, rng.uniform(0.3, 0.5)), (dx - 0.55, 0.755, dz - 0.15), 0.0, collide=False)
    if rng.random() < 0.5:
        _on_wall(b, "whiteboard", A.whiteboard(1.6, 0.9), rng.choice(["left", "right"]), rng.uniform(-0.6, 0.6), 1.5)
    else:
        _on_wall(b, "picture", A.picture_frame(rng, 0.7, 0.5), rng.choice(["left", "right"]), rng.uniform(-0.6, 0.6), 1.6)
    _window(b, rng.choice(["left", "right"]), rng.uniform(-0.9, 0.2) if d > 4 else 0.0, 1.2, 1.3, 0.9)
    b.add("lamp", A.lamp(rng, "floor"), (-w / 2 + 0.4, 0, -d / 2 + 0.45), 0.0)
    b.add("door", A.door(), (rng.choice([-1, 1]) * (w / 2 - 0.8), 0, d / 2 - 0.01), np.pi, collide=False)
    b.anchors["free_floor"] = [-w / 2 + 0.6, dz + 0.75, w / 2 - 0.6, d / 2 - 0.6]
    b.anchors["focus"] = [dx, 1.1, cz]
    b.anchors["stand"].append({"pos": [dx + 0.9, 0, dz + 0.9], "yaw": float(np.pi)})
    return b.finish(([-w / 2, 0, -d / 2], [w / 2, h, d / 2]), environment("indoor", rng))


def bedroom(loc_id="loc0", rng=None) -> LocationSet:
    rng = np.random.default_rng() if rng is None else rng
    b = SetBuilder(loc_id, "bedroom", "indoor", rng)
    w, d, h = rng.uniform(3.6, 4.8), rng.uniform(3.6, 4.6), rng.uniform(2.5, 2.8)
    wall = A.rgb(WALLS[rng.integers(len(WALLS))])
    _room_base(b, w, d, h, "wood", A.rgb(FLOORS_WOOD[rng.integers(len(FLOORS_WOOD))]), wall, rng,
               A.rgb(["#6d597a", "#5b6c5d", "#9eb0b8"][rng.integers(3)]) if rng.random() < 0.4 else None)
    bx = rng.uniform(-0.4, 0.4)
    bw, bl = rng.choice([1.4, 1.6, 1.8]), 2.05
    bz = -d / 2 + bl / 2 + 0.08
    bid = b.add("bed", A.bed(rng, bw, bl), (bx, 0, bz), 0.0)
    b.anchors["seats"].append({"pos": [bx, 0.55, bz + bl / 2 - 0.35], "yaw": 0.0, "kind": "bed", "prop": bid})
    for side in (-1, 1):
        nx = bx + side * (bw / 2 + 0.3)
        nid = b.add("nightstand", A.nightstand(rng), (nx, 0, -d / 2 + 0.25), 0.0)
        b.add("lamp", A.lamp(rng, "desk"), (nx, 0.5, -d / 2 + 0.25), 0.0, collide=False)
        _surface(b, (nx + 0.08, 0.5, -d / 2 + 0.3), (0.15, 0.15), 0.0, "nightstand", nid)
    b.add("rug", A.rug(rng, 2.0, 1.4), (bx, 0, bz + bl / 2 + 0.3), 0.0, collide=False)
    wside = rng.choice([-1, 1])
    b.add("wardrobe", A.wardrobe(rng, 1.2), (wside * (w / 2 - 0.3), 0, d / 2 - 1.0), -wside * np.pi / 2)
    _on_wall(b, "picture", A.picture_frame(rng, rng.uniform(0.7, 1.0), rng.uniform(0.5, 0.6)), "back", bx, 1.55)
    _window(b, "left" if wside > 0 else "right", rng.uniform(-0.5, 0.5), 1.2, 1.3, 0.9)
    b.add("plant", A.plant(rng, rng.uniform(0.6, 1.1)), (-wside * (w / 2 - 0.35), 0, d / 2 - 0.4), 0.0)
    ch = A.chair(rng)
    cx = -wside * (w / 2 - 0.6)
    cid = b.add("chair", ch, (cx, 0, bz + 0.6), wside * np.pi / 2)
    b.anchors["seats"].append({"pos": _seat_world(ch["seat"], (cx, 0, bz + 0.6), wside * np.pi / 2),
                               "yaw": float(wside * np.pi / 2), "kind": "chair", "prop": cid})
    b.add("door", A.door(), (wside * (w / 2 - 1.6) if w > 4 else 0.0, 0, d / 2 - 0.01), np.pi, collide=False)
    b.anchors["free_floor"] = [-w / 2 + 0.7, bz + bl / 2 + 0.2, w / 2 - 0.9, d / 2 - 0.5]
    b.anchors["focus"] = [bx, 0.9, bz + 0.6]
    return b.finish(([-w / 2, 0, -d / 2], [w / 2, h, d / 2]), environment("indoor", rng))


def podcast_studio(loc_id="loc0", rng=None) -> LocationSet:
    """Interview / podcast set: table with two microphones, two facing chairs, acoustic panels."""
    rng = np.random.default_rng() if rng is None else rng
    b = SetBuilder(loc_id, "podcast studio", "indoor", rng)
    w, d, h = rng.uniform(5.5, 7.0), rng.uniform(4.8, 6.0), 3.0
    wall = A.rgb(["#2b2d33", "#3a3536", "#2f3e46", "#4a4440", "#e9e4da"][rng.integers(5)])
    _room_base(b, w, d, h, "wood", A.rgb(FLOORS_WOOD[rng.integers(len(FLOORS_WOOD))]), wall, rng)
    panel_c = A.rgb(["#3d405b", "#264653", "#6d597a", "#7a3b3b", "#3a3a3a", "#335c67"][rng.integers(6)])
    for wall_name, span in (("back", w), ("left", d), ("right", d)):
        n = int(span // 1.3)
        for i in range(n):
            if rng.random() < 0.8:
                along = -span / 2 + span / n * (i + 0.5)
                _on_wall(b, "panel", A.acoustic_panel(0.6, 0.6, panel_c, rng), wall_name, along, rng.choice([1.3, 1.7, 2.1]))
    b.add("rug", A.rug(rng, 3.2, 2.4), (0, 0, 0.2), 0.0, collide=False)
    tid = b.add("table", A.table(rng, 1.5, 0.85, 0.74), (0, 0, 0.2), np.pi / 2)
    _surface(b, (0, 0.74, 0.2), (0.6, 1.2), np.pi / 2, "table", tid)
    b.anchors["table"] = {"pos": [0, 0.74, 0.2], "size": [0.85, 1.5]}
    seats = []
    for side in (-1, 1):
        ch = A.office_chair(rng, A.rgb(["#2a2c30", "#5a3a22", "#3d405b"][rng.integers(3)]))
        pos = (side * 0.95, 0, 0.2)
        yaw = -side * np.pi / 2
        cid = b.add("chair", ch, pos, yaw)
        seats.append({"pos": _seat_world(ch["seat"], pos, yaw), "yaw": float(yaw), "kind": "office_chair", "prop": cid})
        # desk microphone in front of each seat, angled toward the speaker
        b.add("microphone", A.microphone("desk"), (side * 0.42, 0.74, 0.2 + rng.uniform(-0.12, 0.12)), side * np.pi / 2,
              collide=False, key="mic")
    b.anchors["seats"] = seats
    b.anchors["pair"] = True
    sx = rng.uniform(-1.0, 1.0)
    b.add("shelf", A.shelf(rng, 1.0, 1.9, 0.32), (sx, 0, -d / 2 + 0.17), 0.0)
    b.add("plant", A.plant(rng, rng.uniform(1.0, 1.6)), (sx + rng.choice([-1, 1]) * 1.2, 0, -d / 2 + 0.45), 0.0)
    b.add("lamp", A.lamp(rng, "floor"), (-w / 2 + 0.5, 0, -d / 2 + 0.5), 0.0)
    b.add("lamp", A.lamp(rng, "floor"), (w / 2 - 0.5, 0, -d / 2 + 0.5), 0.0)
    neon = A.emissive(A.box((1.2, 0.08, 0.03)), A.rgb(["#ff4fa0", "#38d6ff", "#ffb347", "#7cff6b"][rng.integers(4)]))
    b.add("sign", neon, (sx * -0.6, 2.2, -d / 2 + 0.03), 0.0, collide=False)
    b.add("speaker", A.speaker(0.9), (w / 2 - 0.45, 0, d / 2 - 0.7), np.pi, collide=True)
    b.add("door", A.door(color=A.rgb("#3a3a3a")), (-w / 2 + 1.0, 0, d / 2 - 0.01), np.pi, collide=False)
    b.anchors["free_floor"] = [-w / 2 + 0.8, 1.2, w / 2 - 0.8, d / 2 - 0.6]
    b.anchors["focus"] = [0, 1.1, 0.2]
    env = environment("indoor", rng)
    env["ambient"] *= 0.85
    return b.finish(([-w / 2, 0, -d / 2], [w / 2, h, d / 2]), env)


def dance_studio(loc_id="loc0", rng=None) -> LocationSet:
    rng = np.random.default_rng() if rng is None else rng
    b = SetBuilder(loc_id, "dance studio", "indoor", rng)
    w, d, h = rng.uniform(8.0, 12.0), rng.uniform(7.0, 10.0), rng.uniform(3.2, 4.0)
    wall = A.rgb(WALLS[rng.integers(len(WALLS))])
    _room_base(b, w, d, h, "wood", A.rgb(["#c49a6c", "#b5835a", "#d2b48c"][rng.integers(3)]), wall, rng)
    b.add("mirror", A.mirror_wall(w - 1.0, 2.2), (0, 1.25, -d / 2 + 0.01), 0.0, collide=False)
    b.add("barre", A.barre(min(6, w - 2)), (0, 0, -d / 2 + 0.25), 0.0, collide=False)
    for side in (-1, 1):
        b.add("speaker", A.speaker(1.1), (side * (w / 2 - 0.5), 0, -d / 2 + 0.6), 0.0)
        for k in range(2):
            _window(b, "left" if side < 0 else "right", -d / 4 + k * d / 2.2, 1.4, 1.6, 0.9)
    b.add("plant", A.plant(rng, 1.2), (w / 2 - 0.5, 0, d / 2 - 0.5), 0.0)
    b.add("bench", A.bench(rng, 1.8), (-w / 2 + 1.5, 0, d / 2 - 0.4), np.pi)
    b.add("door", A.door(), (w / 2 - 1.5, 0, d / 2 - 0.01), np.pi, collide=False)
    b.anchors["dance_floor"] = [-w / 2 + 1.5, -d / 2 + 1.5, w / 2 - 1.5, d / 2 - 1.5]
    b.anchors["dance_facing"] = 0.0   # dancers face the front (+Z), mirror behind them
    b.anchors["focus"] = [0, 1.0, 0]
    b.anchors["free_floor"] = b.anchors["dance_floor"]
    return b.finish(([-w / 2, 0, -d / 2], [w / 2, h, d / 2]), environment("indoor", rng))


def stage(loc_id="loc0", rng=None) -> LocationSet:
    """Concert stage: raised platform, truss with lights, speakers, backdrop, audience floor."""
    rng = np.random.default_rng() if rng is None else rng
    b = SetBuilder(loc_id, "stage", "stage", rng)
    W, D, H = 30.0, 26.0, 12.0
    floor = A.grid_patch(W, D, 1, 1, 0.0, color=A.rgb("#2a2a2e"))
    b.add("floor", floor, collide=False)
    sw, sd, sh = rng.uniform(10, 14), rng.uniform(6, 8), rng.uniform(0.9, 1.2)
    sz = -D / 2 + sd / 2 + 2.0
    plat = A.merge(A.box((sw, sh, sd), (0, sh / 2, 0), A.rgb("#1d1d20")),
                   A.box((sw + 0.02, 0.04, sd + 0.02), (0, sh, 0), A.rgb(["#3a2f2a", "#24262b", "#4a3b30"][rng.integers(3)])))
    b.add("stage", plat, (0, 0, sz), 0.0)
    back = A.box((sw + 4, 7.0, 0.3), (0, 3.5, 0), A.rgb(["#141416", "#1d1630", "#10202a"][rng.integers(3)]))
    b.add("backdrop", back, (0, 0, sz - sd / 2 - 0.4), 0.0)
    colors = [A.rgb(c) for c in ("#ff4fa0", "#38d6ff", "#ffb347", "#a066ff", "#7cff6b", "#ffffff")]
    tr = A.truss(sw + 2, 0.35)
    for k, zz in enumerate((sz - sd / 2 + 0.5, sz + sd / 2 - 0.3)):
        b.add("truss", tr, (0, 5.5, zz), 0.0, key="truss", collide=False)
        for i in range(6):
            x = -sw / 2 + (i + 0.5) * sw / 6
            light = A.stage_light(colors[rng.integers(len(colors))])
            b.add("light", light, (x, 5.2, zz), 0.0, collide=False,
                  q=quat_mul(A.yaw_q(np.pi if k == 0 else 0.0), quat_axis_angle([1, 0, 0], 0.9)))
    for side in (-1, 1):
        b.add("truss", A.truss(5.5, 0.35), (side * (sw / 2 + 1), 2.75, sz + sd / 2 - 0.3), 0.0,
              q=quat_axis_angle([0, 0, 1], np.pi / 2), collide=False)
        b.add("speaker", A.merge(A.speaker(1.4), A.transform(A.speaker(1.4), (0, 1.4, 0))), (side * (sw / 2 + 1.2), 0, sz + sd / 2 + 0.6), 0.0)
    screen = A.emissive(A.box((sw * 0.6, 3.0, 0.1)), A.rgb(["#22344d", "#3b1d4a", "#1d3b35"][rng.integers(3)]))
    b.add("screen", screen, (0, sh + 3.0, sz - sd / 2 - 0.2), 0.0, collide=False)
    b.anchors["dance_floor"] = [-sw / 2 + 1.2, sz - sd / 2 + 1.2, sw / 2 - 1.2, sz + sd / 2 - 1.0]
    b.anchors["dance_facing"] = 0.0
    b.anchors["stage_h"] = sh
    b.anchors["focus"] = [0, sh + 1.0, sz]
    box_lo, box_hi = np.array([-sw / 2, sz - sd / 2]), np.array([sw / 2, sz + sd / 2])

    def ground(x, z):
        x, z = np.asarray(x, float), np.asarray(z, float)
        on = (x > box_lo[0]) & (x < box_hi[0]) & (z > box_lo[1]) & (z < box_hi[1])
        return np.where(on, sh, 0.0)
    b.world.ground = ground
    # remove the platform collider (ground handles it) so cameras can go low in front of it
    keep = [i for i, n in enumerate(b.world.names) if not n.startswith("stage")]
    b.world._lo = [b.world._lo[i] for i in keep]
    b.world._hi = [b.world._hi[i] for i in keep]
    b.world.names = [b.world.names[i] for i in keep]
    b.world.occ = [b.world.occ[i] for i in keep]
    b.world.lo, b.world.hi, b.world.occluder = np.array(b.world._lo), np.array(b.world._hi), np.array(b.world.occ)
    b.world.interior = (np.array([-W / 2, 0, -D / 2]), np.array([W / 2, H, D / 2]))
    b.anchors["free_floor"] = b.anchors["dance_floor"]
    env = environment("stage", rng)
    # dark house, strong frontal key light from the lighting rig over the audience
    env["sky"] = [float(x) for x in A.rgb(["#26283a", "#2a2236", "#1f2a33"][rng.integers(3)])]
    env["ground"] = [float(x) for x in A.rgb("#4a4650")]
    env["ambient"] = float(rng.uniform(0.6, 0.8))
    env["sun"] = float(rng.uniform(1.1, 1.35))
    az = rng.uniform(-0.5, 0.5)
    env["sun_dir"] = [float(np.sin(az) * 0.5), -0.75, -0.65]
    return b.finish(([-W / 2, 0, -D / 2], [W / 2, H, D / 2]), env)


# ---------------------------------------------------------------- outdoor: street and highway

def city_street(loc_id="loc0", rng=None, length=180.0) -> LocationSet:
    """Two-lane city street along X with parking lanes, sidewalks, buildings and a cross street at x = 0."""
    rng = np.random.default_rng() if rng is None else rng
    b = SetBuilder(loc_id, "city street", "street", rng)
    L = length
    lane, park, walk = 3.4, 2.3, 3.6
    road_half = lane + park
    curb_h = 0.15
    ground_c = A.rgb("#6f7a5c")
    b.add("ground", A.grid_patch(L + 80, L + 80, 1, 1, -0.02, color=ground_c), collide=False)
    cross_half = 3.5                      # half-width of the cross street
    front = road_half + walk              # building line of the main street
    gap = cross_half + walk               # where main-street buildings start along x
    b.add("road", A.road_strip(L, 2, lane, dashed=False), (0, 0, 0), 0.0, collide=False)
    for side in (-1, 1):
        b.add("road", A.box((L, 0.02, park), (0, 0, 0), A.rgb("#3a3c40")), (0, 0, side * (lane + park / 2)), collide=False)
        for sx in (-1, 1):
            seg = L / 2 - cross_half
            b.add("sidewalk", A.sidewalk(seg, walk, curb_h), (sx * (cross_half + seg / 2), 0, side * (road_half + walk / 2)),
                  0.0, collide=False, key="main_walk")
    lc = L / 2 - road_half
    for side in (-1, 1):
        b.add("road", A.road_strip(lc, 2, cross_half, dashed=False), (0, 0, side * (road_half + lc / 2)), np.pi / 2,
              collide=False, key="cross")
        for sx in (-1, 1):
            seg = L / 2 - front
            b.add("sidewalk", A.sidewalk(seg, walk, curb_h), (sx * (cross_half + walk / 2), 0, side * (front + seg / 2)),
                  np.pi / 2, collide=False, key="cross_walk")
    for side in (-1, 1):
        b.add("crosswalk", A.crosswalk(2 * lane, 3.0), (side * (cross_half + 1.6), 0, 0), 0.0, collide=False, key="cw")
        b.add("crosswalk", A.crosswalk(2 * cross_half, 3.0), (0, 0, side * (road_half + 1.7)), np.pi / 2, collide=False,
              key="cw2")
    # buildings
    for side in (-1, 1):
        for x0, x1 in ((-L / 2 - 20, -gap - 0.5), (gap + 0.5, L / 2 + 20)):
            x = x0
            while x < x1 - 6:
                bw = min(rng.uniform(8, 20), x1 - x)
                bd = rng.uniform(10, 16)
                floors = int(rng.integers(2, 9))
                bld = A.building(rng, bw, bd, floors)
                yaw = 0.0 if side < 0 else np.pi
                pos = (x + bw / 2, 0, side * (front + bd / 2 + 0.05))
                b.add("building", bld, pos, yaw)
                x += bw + rng.choice([0.0, 0.0, 1.5])
        for sx in (-1, 1):   # corner buildings along the cross street
            for k in range(3):
                bw, bd = rng.uniform(10, 18), rng.uniform(10, 14)
                zc = side * (front + 28 + k * 20)
                b.add("building", A.building(rng, bw, bd, int(rng.integers(3, 9))), (sx * (gap + bd / 2 + 0.05), 0, zc),
                      -sx * np.pi / 2)
    # street furniture
    lamp = A.street_lamp(6.0)
    for side in (-1, 1):
        for x in np.arange(-L / 2 + 8, L / 2 - 4, 24.0):
            if abs(x) < gap + 2:
                continue
            b.add("street_lamp", lamp, (x, curb_h, side * (road_half + 0.45)), np.pi if side > 0 else 0.0, key="lamp")
        for x in np.arange(-L / 2 + 20, L / 2 - 4, 16.0):
            if abs(x) < gap + 3:
                continue
            if rng.random() < 0.7:
                t = A.tree(rng, "broadleaf", rng.uniform(5, 7))
                b.add("tree", t, (x + rng.uniform(-1, 1), curb_h, side * (road_half + 1.1)), rng.uniform(0, 6),
                      key=("st_tree", int(rng.integers(4))))
            elif rng.random() < 0.5:
                b.add("bench", A.bench(rng), (x, curb_h, side * (front - 0.6)), 0.0 if side > 0 else np.pi, key="bench")
            else:
                b.add("trash_can", A.trash_can(), (x, curb_h, side * (road_half + 0.6)), 0.0, key="trash")
    for sx in (-1, 1):
        for sz in (-1, 1):
            st = "green" if sx * sz > 0 else "red"
            b.add("traffic_light", A.traffic_light(st), (sx * (gap - 0.5), curb_h, sz * (road_half + 0.5)),
                  np.pi / 2 * (1 if sx < 0 else -1), key=("tl", st))
    # parked cars
    for side in (-1, 1):
        x = -L / 2 + rng.uniform(3, 10)
        while x < L / 2 - 5:
            if abs(x) > gap + 3 and rng.random() < 0.55:
                cm, dims = A.car(rng, rng.choice(["sedan", "hatch", "suv"]))
                cm = A.merge(cm, *cm.get("extras", []))
                cm.pop("joints", None)
                cm.pop("weights", None)
                b.add("car", cm, (x, 0.0, side * (lane + park / 2)), np.pi / 2 if side > 0 else -np.pi / 2)
            x += rng.uniform(6, 9)

    def ground(x, z):
        az = np.abs(np.asarray(z, float))
        ax = np.abs(np.asarray(x, float))
        on_walk = ((az > road_half) & (az < front) & (ax > cross_half)) | \
                  ((ax > cross_half) & (ax < gap) & (az >= front))
        on_road = (az <= road_half) | ((ax <= cross_half) & (az > road_half))
        return np.where(on_walk, curb_h, np.where(on_road, 0.01, -0.02))
    b.world.ground = ground
    b.anchors["roads"] = [
        {"points": [[-L / 2 + 2, 0, lane / 2], [L / 2 - 2, 0, lane / 2]], "dir": 1, "speed": 11.0},
        {"points": [[L / 2 - 2, 0, -lane / 2], [-L / 2 + 2, 0, -lane / 2]], "dir": -1, "speed": 11.0},
        {"points": [[-cross_half / 2, 0, -L / 2 + 5], [-cross_half / 2, 0, L / 2 - 5]], "dir": 1, "speed": 9.0,
         "cross": True},
    ]
    b.anchors["sidewalks"] = [[[-L / 2 + 5, side * (road_half + walk * 0.6)], [L / 2 - 5, side * (road_half + walk * 0.6)]]
                              for side in (-1, 1)]
    b.anchors["curb"] = {"road_half": road_half, "front": front, "walk": walk, "curb_h": curb_h, "gap": gap}
    b.anchors["focus"] = [0, 1.5, 0]
    return b.finish(([-L / 2 - 40, -1, -L / 2 - 40], [L / 2 + 40, 40, L / 2 + 40]), environment("street", rng))


def _road_spline(rng, length=1800.0, step=200.0):
    pts = [np.array([0.0, 0.0])]
    heading = 0.0
    for _ in range(int(length / step)):
        heading += rng.uniform(-0.35, 0.35)
        pts.append(pts[-1] + step * np.array([np.sin(heading), np.cos(heading)]))
    pts = np.array(pts)
    pts -= pts.mean(0)
    return Path(pts, spacing=2.0)


def highway(loc_id="loc0", rng=None, length=1800.0) -> LocationSet:
    """Divided four-lane highway winding through hilly countryside."""
    rng = np.random.default_rng() if rng is None else rng
    b = SetBuilder(loc_id, "highway", "street", rng)
    path = _road_spline(rng, length)
    lane, median, shoulder = 3.6, 1.2, 2.0
    half = 2 * lane + median / 2 + shoulder
    # terrain over the road's bounding box
    lo, hi = path.p.min(0) - 220, path.p.max(0) + 220
    cell = 8.0
    nx, nz = int((hi[0] - lo[0]) / cell) + 1, int((hi[1] - lo[1]) / cell) + 1
    hills = A.value_noise((nz, nx), 40, rng, 4) * 35 + A.value_noise((nz, nx), 10, rng, 2) * 6
    X, Z = np.meshgrid(lo[0] + np.arange(nx) * cell, lo[1] + np.arange(nz) * cell)
    from scipy.spatial import cKDTree
    tree = cKDTree(path.p)
    dist, idx = tree.query(np.stack([X.ravel(), Z.ravel()], 1))
    dist, idx = dist.reshape(X.shape), idx.reshape(X.shape)
    road_h = gaussian_filter(np.interp(path.s, path.s[::50], rng.normal(0, 4, len(path.s[::50]))), 60)
    k = np.clip((dist - half - 4) / 45, 0, 1)
    k = k * k * (3 - 2 * k)
    hgt = road_h[idx] * (1 - k) + (road_h[idx] + hills - hills.min() * 0.3) * k
    hf = A.Heightfield(hgt, lo[0], lo[1], cell)
    grass, dirt = A.rgb("#6f8f4a"), A.rgb("#8a7a55")

    def color_fn(x, y, z, slope):
        c = np.where((slope > 0.12)[:, None], dirt[None], grass[None])
        return c * (0.92 + 0.08 * np.sin(x * 0.05 + z * 0.03))[:, None]
    b.add("terrain", A.terrain_mesh(hf, color_fn, rng, 0.06), collide=False)
    ground = lambda x, z: hf(x, z)
    b.world.ground = ground
    # road surface and markings
    P = np.stack([path.p[:, 0], np.zeros(len(path.p)), path.p[:, 1]], 1)
    road = A.road_ribbon(P, 2 * half, A.rgb("#3b3d41"), 0.06, ground)
    marks = [A.road_ribbon(P, median, A.rgb("#55575c"), 0.075, ground)]
    tdir = np.gradient(path.p, axis=0)
    tdir /= np.linalg.norm(tdir, axis=1, keepdims=True)
    nrm = np.stack([tdir[:, 1], -tdir[:, 0]], 1)    # left-hand normal in xz (x, z): +X when heading +Z
    white = A.rgb("#e8e8e2")
    for off in (median / 2 + 0.2, median / 2 + 2 * lane - 0.15):
        for sgn in (-1, 1):
            pts = P.copy()
            pts[:, 0] += nrm[:, 0] * off * sgn
            pts[:, 2] += nrm[:, 1] * off * sgn
            marks.append(A.road_ribbon(pts, 0.15, white, 0.08, ground))
    for sgn in (-1, 1):
        pts = P.copy()
        pts[:, 0] += nrm[:, 0] * (median / 2 + lane) * sgn
        pts[:, 2] += nrm[:, 1] * (median / 2 + lane) * sgn
        pts[:, 1] = ground(pts[:, 0], pts[:, 2])
        marks.append(A.dashed_ribbon(pts, 0.14, white, 3.0, 9.0, 0.08))
    b.add("road", road, collide=False)
    b.add("markings", A.merge(*marks), collide=False)
    # guard rails as a ribbon of posts along both edges (merged)
    rails = []
    for sgn in (-1, 1):
        off = half + 0.4
        pts = P[::3].copy()
        pts[:, 0] += nrm[::3, 0] * off * sgn
        pts[:, 2] += nrm[::3, 1] * off * sgn
        pts[:, 1] = ground(pts[:, 0], pts[:, 2])
        for a, c in zip(pts[:-1], pts[1:]):
            rails.append(A.limb(a + [0, 0.6, 0], c + [0, 0.6, 0], 0.12, 0.12, 4, A.rgb("#b8bcc2")))
            rails.append(A.limb(a, a + [0, 0.7, 0], 0.05, 0.05, 4, A.rgb("#8a8e93")))
    b.add("guardrail", A.merge(*rails), collide=False)
    # trees and rocks beyond the verges
    variants = {kd: [A.tree(rng, kd) for _ in range(4)] for kd in ("pine", "broadleaf")}
    n_trees = int(length / 4)
    placed = 0
    for _ in range(n_trees * 3):
        if placed >= n_trees:
            break
        s = rng.uniform(0, path.length)
        side = rng.choice([-1, 1])
        off = rng.uniform(half + 12, half + 150) * side
        i = int(s / path.length * (len(path.p) - 1))
        x, z = path.p[i] + nrm[i] * off
        kd = "pine" if rng.random() < 0.5 else "broadleaf"
        vi = int(rng.integers(4))
        b.add("tree", variants[kd][vi], (x, float(ground(x, z)) - 0.1, z), rng.uniform(0, 6), scale=rng.uniform(0.8, 1.3),
              key=(kd, vi))
        placed += 1
    for _ in range(int(length / 40)):
        s = rng.uniform(0, path.length)
        i = int(s / path.length * (len(path.p) - 1))
        off = rng.uniform(half + 8, half + 60) * rng.choice([-1, 1])
        x, z = path.p[i] + nrm[i] * off
        b.add("rock", A.rock(rng, rng.uniform(0.6, 2.5)), (x, float(ground(x, z)) - 0.2, z), rng.uniform(0, 6))
    # overhead sign gantries
    for s in np.linspace(path.length * 0.2, path.length * 0.8, 3):
        i = int(s / path.length * (len(path.p) - 1))
        c = path.p[i]
        y = float(ground(*c))
        yaw = float(path.heading(s))
        g = A.merge(A.box((0.3, 6.5, 0.3), (half + 0.8, 3.25, 0), A.rgb("#8a8e93")),
                    A.box((0.3, 6.5, 0.3), (-half - 0.8, 3.25, 0), A.rgb("#8a8e93")),
                    A.box((2 * half + 2, 0.4, 0.4), (0, 6.3, 0), A.rgb("#8a8e93")),
                    A.box((5.0, 2.0, 0.1), (half / 2, 5.1, -0.25), A.rgb("#1f6b3a")),
                    A.box((4.8, 0.12, 0.11), (half / 2, 5.4, -0.26), A.rgb("#f2f2f2")))
        b.add("sign", g, (c[0], y, c[1]), yaw + np.pi, collide=True)
    b.add("backdrop", A.mountain_ring(rng, radius=float(np.linalg.norm(hi - lo)) * 0.65, n=16, h=(90, 260)),
          (float((lo[0] + hi[0]) / 2), 0, float((lo[1] + hi[1]) / 2)), collide=False, cast_shadow=False)
    roads = []
    for sgn, direction in ((-1, 1), (1, -1)):       # right-hand traffic: drive on the right of the median
        for ln in range(2):
            off = (median / 2 + lane * (ln + 0.5)) * sgn
            pts = np.stack([path.p[:, 0] + nrm[:, 0] * off, np.zeros(len(path.p)), path.p[:, 1] + nrm[:, 1] * off], 1)
            if direction < 0:
                pts = pts[::-1]
            roads.append({"points": pts[::5].tolist(), "dir": direction, "speed": float(rng.uniform(24, 31)), "lane": ln})
    b.anchors["roads"] = roads
    b.anchors["road_path"] = path
    b.anchors["focus"] = [0, 1, 0]
    env = environment("street", rng)
    env["fog"] = {"color": [float(x) for x in A.mix(np.array(env["sky"]), A.rgb("#e6ecef"), 0.5)], "near": 150.0,
                  "far": 900.0}
    return b.finish(([lo[0], -20, lo[1]], [hi[0], 300, hi[1]]), env)


# ---------------------------------------------------------------- nature trail

TRAIL_VARIANTS = ("forest", "mountain", "desert")


def nature_trail(loc_id="loc0", rng=None, variant="forest", size=240.0) -> LocationSet:
    """Hiking trail through a heightfield terrain with trees, rocks, bushes and grass."""
    rng = np.random.default_rng() if rng is None else rng
    b = SetBuilder(loc_id, f"{variant} trail", "nature", rng)
    cell = 2.0
    n = int(size / cell) + 1
    lo = -size / 2
    amp = {"forest": 7.0, "mountain": 26.0, "desert": 5.0}[variant]
    base = A.value_noise((n, n), 60, rng, 5, 0.5)
    if variant == "mountain":
        ridge = 1 - np.abs(A.value_noise((n, n), 70, rng, 4, 0.55))
        H = base * amp * 0.6 + ridge ** 2 * amp
    elif variant == "desert":
        X_, Z_ = np.meshgrid(np.arange(n), np.arange(n))
        dunes = np.sin(X_ * 0.09 + A.value_noise((n, n), 30, rng, 2) * 3) * 1.6
        H = base * amp + dunes
    else:
        H = base * amp
    # trail: winding path across the terrain
    k_pts = 7
    xs = np.linspace(-size * 0.42, size * 0.42, k_pts)
    zs = np.cumsum(rng.uniform(-1, 1, k_pts)) * size * 0.08
    zs -= zs.mean()
    trail = Path(np.stack([xs, zs], 1), spacing=1.0)
    X, Z = np.meshgrid(lo + np.arange(n) * cell, lo + np.arange(n) * cell)
    from scipy.spatial import cKDTree
    tree = cKDTree(trail.p)
    dist, idx = tree.query(np.stack([X.ravel(), Z.ravel()], 1))
    dist, idx = dist.reshape(X.shape), idx.reshape(X.shape)
    # trail height: terrain smoothed along the path (gentle grade), terrain blends into it
    hf0 = A.Heightfield(H, lo, lo, cell)
    th = gaussian_filter(hf0(trail.p[:, 0], trail.p[:, 1]), 12)
    kk = np.clip((dist - 1.8) / 7.0, 0, 1)
    kk = kk * kk * (3 - 2 * kk)
    H = th[idx] * (1 - kk) + H * kk
    hf = A.Heightfield(H, lo, lo, cell)
    pal = {"forest": ("#5d7f3a", "#4c6b33", "#7a6a4a", "#8a8378", None),
           "mountain": ("#6f8048", "#7d7a6e", "#8a7a5a", "#8d8a85", "#f2f4f7"),
           "desert": ("#d9b77e", "#c99a62", "#c2955a", "#b07a4a", None)}[variant]
    g1, g2, trail_c, rock_c, snow_c = (A.rgb(c) if c else None for c in pal)
    hmax = H.max()

    def color_fn(x, y, z, slope):
        d_tr, _ = tree.query(np.stack([x, z], 1))
        noise = (np.sin(x * 0.21 + np.sin(z * 0.13) * 2) + np.sin(z * 0.17 - x * 0.05)) * 0.25 + 0.5
        c = g1[None] * (1 - noise[:, None]) + g2[None] * noise[:, None]
        c = np.where((slope > 0.25)[:, None], rock_c[None], c)
        if snow_c is not None:
            c = np.where((y > hmax * 0.72)[:, None] & (slope < 0.5)[:, None], snow_c[None], c)
        c = np.where((d_tr < 1.3)[:, None], trail_c[None], c)
        return c
    b.add("terrain", A.terrain_mesh(hf, color_fn, rng, 0.05), collide=False)
    b.world.ground = lambda x, z: hf(x, z)
    # vegetation and rocks: denser near the trail, none on it
    kinds = {"forest": [("pine", 0.45), ("broadleaf", 0.35), ("birch", 0.15), ("dead", 0.05)],
             "mountain": [("pine", 0.85), ("dead", 0.15)],
             "desert": [("cactus", 0.6), ("dead", 0.25), ("palm", 0.15)]}[variant]
    density = {"forest": 520, "mountain": 220, "desert": 110}[variant]
    variants = {kd: [A.tree(rng, kd) for _ in range(4)] for kd, _ in kinds}
    kd_names = [k for k, _ in kinds]
    kd_p = np.array([p for _, p in kinds])

    def scatter(count, min_d, max_d, fn):
        placed, tries = 0, 0
        while placed < count and tries < count * 20:
            tries += 1
            x, z = rng.uniform(lo + 4, -lo - 4, 2)
            d, _ = tree.query([x, z])
            if d < min_d or d > max_d:
                continue
            if rng.random() > np.exp(-d / 45):
                continue
            fn(x, z, d)
            placed += 1

    def put_tree(x, z, d):
        kd = kd_names[rng.choice(len(kd_names), p=kd_p)]
        if variant == "mountain" and hf(x, z) > hmax * 0.7:
            return
        vi = int(rng.integers(4))
        b.add("tree", variants[kd][vi], (x, float(hf(x, z)) - 0.15, z), rng.uniform(0, 6), scale=rng.uniform(0.75, 1.3),
              key=(kd, vi))
    scatter(density, 3.5, 110, put_tree)
    rock_meshes = [A.rock(rng, s, rock_c) for s in (0.4, 0.8, 1.4, 2.5, 4.0)]
    n_rocks = {"forest": 60, "mountain": 140, "desert": 90}[variant]

    def put_rock(x, z, d):
        vi = int(rng.integers(len(rock_meshes) if d > 6 else 2))
        b.add("rock", rock_meshes[vi], (x, float(hf(x, z)) - 0.2, z), rng.uniform(0, 6), scale=rng.uniform(0.7, 1.4),
              key=("rock", vi))
    scatter(n_rocks, 2.2, 90, put_rock)
    if variant != "desert":
        bushes = [A.bush(rng) for _ in range(4)]

        def put_bush(x, z, d):
            vi = int(rng.integers(4))
            b.add("bush", bushes[vi], (x, float(hf(x, z)) - 0.05, z), rng.uniform(0, 6), key=("bush", vi))
        scatter({"forest": 160, "mountain": 70}[variant], 2.0, 60, put_bush)
    # grass tufts / flowers merged into a few chunk meshes near the trail
    tufts = []
    gcol = {"forest": "#5e8c3a", "mountain": "#7a8c4a", "desert": "#a89a5a"}[variant]
    for _ in range({"forest": 900, "mountain": 500, "desert": 200}[variant]):
        s = rng.uniform(0, trail.length)
        p = trail.pos(s)
        hd = trail.heading(s)
        off = rng.choice([-1, 1]) * rng.uniform(1.5, 9)
        x, z = p[0] + np.cos(hd) * off, p[1] - np.sin(hd) * off
        m = A.grass_tuft(rng, A.rgb(gcol)) if rng.random() > 0.12 or variant == "desert" else A.flower(rng)
        tufts.append(A.transform(m, (x, float(hf(x, z)) - 0.02, z), A.yaw_q(rng.uniform(0, 6))))
    for i in range(0, len(tufts), 300):
        b.add("grass", A.merge(*tufts[i:i + 300]), collide=False)
    # distant backdrop
    ring = A.mountain_ring(rng, radius=size * 1.6, n=14,
                           h={"forest": (40, 90), "mountain": (120, 300), "desert": (25, 70)}[variant],
                           color=A.rgb({"forest": "#56664e", "mountain": "#6f6a66", "desert": "#b98a5e"}[variant]),
                           snow=variant == "mountain")
    b.add("backdrop", ring, collide=False, cast_shadow=False)
    if variant != "desert":
        # tree line at the terrain edge so the horizon is not bare
        cone_c = A.rgb("#2f4f2f" if variant == "forest" else "#3a5a3a")
        edge = []
        for a in np.linspace(0, 2 * np.pi, 140, endpoint=False):
            r = size * 0.5 * rng.uniform(0.92, 1.0)
            x, z = np.cos(a) * r, np.sin(a) * r
            hh = rng.uniform(8, 16)
            edge.append(A.transform(A.cone(hh * 0.3, hh, 6, A.jitter_color(cone_c, rng, 0.1)), (x, float(hf(x, z)) - 1, z)))
        b.add("tree", A.merge(*edge), collide=False, name="treeline")
    tp = np.stack([trail.p[:, 0], hf(trail.p[:, 0], trail.p[:, 1]), trail.p[:, 1]], 1)
    b.anchors["trail"] = trail
    b.anchors["trail_points"] = tp
    b.anchors["heightfield"] = hf
    b.anchors["focus"] = tp[len(tp) // 2].tolist()
    env = environment("nature", rng)
    if variant == "desert":
        env["sky"] = [float(x) for x in A.rgb("#b7d3ea")]
        env["sun"] = 1.05
        env["fog"] = {"color": [float(x) for x in A.rgb("#e8dccb")], "near": 60.0, "far": 320.0}
    elif variant == "mountain":
        env["fog"] = {"color": [float(x) for x in A.mix(np.array(env["sky"]), A.rgb("#e6ecef"), 0.5)], "near": 80.0,
                      "far": 520.0}
    return b.finish(([lo, float(H.min()) - 1, lo], [-lo, float(H.max()) + 60, -lo]), env)


# ---------------------------------------------------------------- voxel world

def voxel_world(loc_id="loc0", rng=None, size=(72, 40, 72)) -> LocationSet:
    """Block world: noise terrain with grass / dirt / stone / sand / water, trees, a small house and clouds."""
    rng = np.random.default_rng() if rng is None else rng
    b = SetBuilder(loc_id, "voxel world", "voxel", rng)
    X, Y, Z = size
    ids = A.BLOCK_ID
    vox = np.zeros((X, Y, Z), np.uint8)
    hm = A.value_noise((X, Z), 28, rng, 3, 0.45)
    hm = np.round(13 + hm * 7 + A.value_noise((X, Z), 8, rng, 1) * 1.2).astype(int)
    # a flat plateau for the house and play area
    cx, cz = X // 2 + int(rng.integers(-6, 6)), Z // 2 + int(rng.integers(-6, 6))
    plateau_h = int(np.median(hm[cx - 10:cx + 10, cz - 10:cz + 10]))
    xx, zz = np.meshgrid(np.arange(X), np.arange(Z), indexing="ij")
    rr = np.sqrt((xx - cx) ** 2 + (zz - cz) ** 2)
    blend = np.clip((rr - 11) / 7, 0, 1)
    hm = np.round(plateau_h * (1 - blend) + hm * blend).astype(int)
    sea = plateau_h - 3
    for x in range(X):
        for z in range(Z):
            h = hm[x, z]
            vox[x, :max(h - 3, 0), z] = ids["stone"]
            vox[x, max(h - 3, 0):h, z] = ids["dirt"]
            top = ids["grass"] if h > sea + 1 else ids["sand"]
            vox[x, h, z] = top
            if h < sea:
                vox[x, h + 1:sea + 1, z] = ids["water"]
    # house: cobble foundation, plank walls, glass windows, door opening, stepped roof
    hw, hd, wall_h = int(rng.integers(5, 7)), int(rng.integers(5, 7)), 4
    x0, z0 = cx - hw, cz - hd
    x1, z1 = cx + hw, cz + hd
    y0 = plateau_h + 1
    vox[x0:x1 + 1, plateau_h, z0:z1 + 1] = ids["cobble"]
    for y in range(y0, y0 + wall_h):
        for x in range(x0, x1 + 1):
            for z in range(z0, z1 + 1):
                edge = x in (x0, x1) or z in (z0, z1)
                if edge:
                    corner = x in (x0, x1) and z in (z0, z1)
                    vox[x, y, z] = ids["log"] if corner else ids["planks"]
    for x in range(x0 + 2, x1 - 1, 3):
        vox[x, y0 + 1:y0 + 3, z0] = ids["glass"]
        vox[x, y0 + 1:y0 + 3, z1] = ids["glass"]
    for z in range(z0 + 2, z1 - 1, 3):
        vox[x0, y0 + 1:y0 + 3, z] = ids["glass"]
    vox[cx, y0:y0 + 2, z1] = 0          # door opening facing +Z
    for k in range(hw + 2):           # gable roof sloping down to +-x, gables at z0 / z1
        y = y0 + wall_h + k
        xa, xb = x0 - 1 + k, x1 + 1 - k
        if xa > xb or y >= Y:
            break
        vox[xa, y, z0 - 1:z1 + 2] = ids["roof"]
        vox[xb, y, z0 - 1:z1 + 2] = ids["roof"]
        if xb - xa <= 1:
            vox[xa:xb + 1, y, z0 - 1:z1 + 2] = ids["roof"]
        if xb - xa > 1:
            vox[xa + 1:xb, y, z0] = ids["planks"]
            vox[xa + 1:xb, y, z1] = ids["planks"]
    # lamp posts and a path of gravel from the door
    for z in range(z1 + 1, min(Z - 1, z1 + 8)):
        vox[cx, hm[cx, z], z] = ids["gravel"]
    # trees away from the house
    trees = 0
    for _ in range(400):
        if trees >= int(X * Z / 160):
            break
        x, z = int(rng.integers(3, X - 3)), int(rng.integers(3, Z - 3))
        if rr[x, z] < 16 or vox[x, hm[x, z], z] != ids["grass"]:
            continue
        th = int(rng.integers(4, 7))
        base = hm[x, z] + 1
        if base + th + 3 >= Y:
            continue
        vox[x - 2:x + 3, base + th - 2:base + th, z - 2:z + 3] = np.where(
            vox[x - 2:x + 3, base + th - 2:base + th, z - 2:z + 3] == 0, ids["leaves"],
            vox[x - 2:x + 3, base + th - 2:base + th, z - 2:z + 3])
        vox[x - 1:x + 2, base + th:base + th + 2, z - 1:z + 2] = ids["leaves"]
        vox[x, base:base + th, z] = ids["log"]
        trees += 1
    origin = np.array([-X / 2, -float(plateau_h + 1), -Z / 2])   # plateau top at y = 0
    var = (A.value_noise((X, Z), 6, rng, 1) > 0.25).astype(np.uint8)[:, None, :] * np.ones((1, Y, 1), np.uint8)
    var = var + (A.value_noise((X, Z), 3, rng, 1) > 0.5)[:, None, :].astype(np.uint8)
    # mesh once, then split quads into chunk props (keeps the viewer's culling effective)
    m = A.greedy_mesh(vox, 1.0, origin, var)
    f = m["indices"]
    cen = m["positions"][f].mean(1) - origin
    ch = 24
    key = (np.floor(cen[:, 0] / ch).astype(int) * 1000 + np.floor(cen[:, 2] / ch).astype(int))
    for kk in np.unique(key):
        sel = f[key == kk]
        used = np.unique(sel)
        remap = np.zeros(len(m["positions"]), np.int64)
        remap[used] = np.arange(len(used))
        b.add("voxel_chunk", A.make_mesh(m["positions"][used], remap[sel], m["colors"][used]), collide=False)
    # flowers and clouds
    fl = []
    for _ in range(140):
        x, z = int(rng.integers(0, X)), int(rng.integers(0, Z))
        if vox[x, hm[x, z], z] == ids["grass"] and vox[x, hm[x, z] + 1, z] == 0 and rr[x, z] > 4:
            c = A.rgb(["#d23c3c", "#e8c547", "#f4f1de", "#9b5de5"][rng.integers(4)])
            p = origin + [x + 0.5, hm[x, z] + 1, z + 0.5]
            fl.append(A.transform(A.merge(A.box((0.08, 0.4, 0.08), (0, 0.2, 0), A.rgb("#3f7f35")),
                                          A.box((0.22, 0.18, 0.22), (0, 0.45, 0), c)), p))
    if fl:
        b.add("flower", A.merge(*fl), collide=False)
    cl = []
    for _ in range(16):
        cxx, czz = rng.uniform(-X, X), rng.uniform(-Z, Z)
        for _ in range(int(rng.integers(2, 5))):
            cl.append(A.box((rng.uniform(6, 14), 1.0, rng.uniform(4, 9)), (cxx + rng.uniform(-6, 6), 30 + rng.uniform(0, 1),
                                                                              czz + rng.uniform(-6, 6)), A.rgb("#f6f8fa")))
    b.add("cloud", A.merge(*cl), collide=False, cast_shadow=False)
    solid = A.SOLID[vox]
    b.world.voxels = (solid, origin, 1.0)
    hm_top = hm.astype(float) + 1 + origin[1]          # walkable surface height per column

    def ground(x, z):
        x, z = np.asarray(x, float), np.asarray(z, float)
        ix = np.clip(np.floor(x - origin[0]).astype(int), 0, X - 1)
        iz = np.clip(np.floor(z - origin[2]).astype(int), 0, Z - 1)
        return hm_top[ix, iz]
    b.world.ground = ground
    b.anchors["heightmap"] = hm_top
    b.anchors["voxel_origin"] = origin
    b.anchors["house"] = {"lo": (origin + [x0, y0, z0]).tolist(), "hi": (origin + [x1 + 1, y0 + wall_h, z1 + 1]).tolist(),
                          "door": (origin + [cx + 0.5, y0, z1 + 1.5]).tolist()}
    b.anchors["plateau"] = {"center": (origin + [cx + 0.5, plateau_h + 1, cz + 0.5]).tolist(), "radius": 11.0}
    b.anchors["water_level"] = float(sea + 1 + origin[1])
    b.anchors["voxels"] = vox
    b.anchors["focus"] = (origin + [cx, plateau_h + 2, cz]).tolist()
    env = environment("voxel", rng, "day")
    return b.finish((origin.tolist(), (origin + [X, Y + 20, Z]).tolist()), env)


GENERATORS = {"living_room": living_room, "kitchen": kitchen, "office": office, "bedroom": bedroom,
              "podcast_studio": podcast_studio, "dance_studio": dance_studio, "stage": stage, "city_street": city_street,
              "highway": highway, "forest_trail": lambda i, r: nature_trail(i, r, "forest"),
              "mountain_trail": lambda i, r: nature_trail(i, r, "mountain"),
              "desert_trail": lambda i, r: nature_trail(i, r, "desert"), "voxel_world": voxel_world}


def make(name, loc_id="loc0", rng=None) -> LocationSet:
    return GENERATORS[name](loc_id, rng)
