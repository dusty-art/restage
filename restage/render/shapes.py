"""Fixed-topology, differentiable mesh builders (MLX).

Every builder returns ``(verts, faces, meta)``:

* ``verts``: [B, V, 3] MLX array, world coordinates (metres, +Y up);
* ``faces``: [F, 3] static numpy int32, counter-clockwise seen from outside
  (three.js / glTF front faces), indices into V;
* ``meta``: dict of static per-vertex numpy arrays (e.g. ``box``, ``joint``, ``region``).

Topology never depends on the parameters, so batched meshes share ``faces`` and
compiled graphs are reused. Characters face +Z in their rest pose (SCENE_FORMAT.md).
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from ..scene import JID, OFFSETS, PARENTS
from .fk import forward_kinematics, matvec3, rest_joint_positions, skin_rigid, yaw_matrix

__all__ = [
    "cube_grid", "rounded_boxes", "ground_plane", "humanoid", "humanoid_offsets", "humanoid_rest_local",
    "humanoid_rest_numpy", "merge_meshes", "tile_faces", "HUMANOID_TEMPLATE", "REGIONS", "SHAPE_PARAMS",
    "N_SHAPE",
]

# ----------------------------------------------------------------------------- unit cube grid


def cube_grid(n: int = 3):
    """Surface of the cube [-1, 1]^3 subdivided into n x n quads per face.

    Returns (u [V, 3] float32 with V = 6n^2 + 2, faces [12 n^2, 3] int32 outward CCW,
    face_id [F] in 0..5 for +X, -X, +Y, -Y, +Z, -Z).
    """
    idx = {}
    pts = []

    def vid(i, j, k):
        key = (i, j, k)
        if key not in idx:
            idx[key] = len(pts)
            pts.append([-1 + 2 * i / n, -1 + 2 * j / n, -1 + 2 * k / n])
        return idx[key]

    faces, fid = [], []
    # (fixed axis, value, axis a, axis b): the quad grid spans axes a, b on the face
    for f_i, (ax, val) in enumerate([(0, n), (0, 0), (1, n), (1, 0), (2, n), (2, 0)]):
        a, b = [d for d in range(3) if d != ax]
        for s in range(n):
            for t in range(n):
                corners = []
                for ds, dt in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    c = [0, 0, 0]
                    c[ax], c[a], c[b] = val, s + ds, t + dt
                    corners.append(vid(*c))
                q = [corners[0], corners[1], corners[2], corners[3]]
                faces += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
                fid += [f_i, f_i]
    u = np.array(pts, np.float32)
    f = _orient_outward(u, np.array(faces, np.int32), np.zeros(3))
    return u, f, np.array(fid, np.int32)


def _orient_outward(p, f, center):
    """Flip triangles so their normals point away from `center` ([3] or per-face [F, 3])."""
    a, b, c = p[f[:, 0]], p[f[:, 1]], p[f[:, 2]]
    n = np.cross(b - a, c - a)
    out = (a + b + c) / 3 - center
    flip = (n * out).sum(-1) < 0
    f = f.copy()
    f[flip] = f[flip][:, [0, 2, 1]]
    return f


def _superellipsoid_np(u, roundness):
    """Radially project cube-surface points onto |x|^p + |y|^p + |z|^p = 1 with p = 2 / roundness (numpy)."""
    p = 2.0 / np.clip(roundness, 1e-3, 1.0)
    norm = (np.abs(u) ** p).sum(-1, keepdims=True) ** (1.0 / p)
    return u / norm


# ----------------------------------------------------------------------------- rounded boxes

_BOX_CACHE: dict = {}


def _box_template(n):
    t = _BOX_CACHE.get(n)
    if t is None:
        u, f, fid = cube_grid(n)
        au = np.abs(u)
        logu = np.where(au > 0, np.log(np.maximum(au, 1e-30)), -1e4).astype(np.float32)
        t = (u, f, fid, logu)
        _BOX_CACHE[n] = t
    return t


def rounded_boxes(center: mx.array, size: mx.array, yaw: mx.array, roundness: mx.array, n: int = 3):
    """N oriented rounded boxes per job, identical topology.

    Args:
      center: [B, N, 3] box centres (metres); size: [B, N, 3] full extents along the box's local
        x, y, z; yaw: [B, N] rotation about +Y (radians, ``fk.yaw_matrix``); roundness: [B, N]
        in 0..1. A cube-surface grid with n x n quads per face is projected radially onto the
        superellipsoid ``|x|^p + |y|^p + |z|^p = 1`` with ``p = 2 / roundness``: 0 gives the
        sharp box (exactly at roundness <= 1e-3 up to 0.06 %), 1 the inscribed ellipsoid.
      n: subdivisions per face edge (static; 3 gives 56 vertices and 108 triangles per box).

    Returns:
      verts [B, N * Vb, 3], faces [N * Fb, 3] (box k uses vertices k*Vb ... (k+1)*Vb - 1),
      meta {"box": [V] box index, "face_id": [F] cube side 0..5, "n_vert_per_box": Vb,
      "n_face_per_box": Fb}.
    """
    u, f, fid, logu = _box_template(n)
    B, N = center.shape[:2]
    Vb = u.shape[0]
    p = 2.0 / mx.clip(roundness, 1e-3, 1.0)                                  # [B, N]
    lu = mx.array(logu)                                                     # [Vb, 3]
    s = mx.exp(p[..., None, None] * lu[None, None]).sum(-1)                 # [B, N, Vb], >= 1
    inv_norm = mx.exp(-mx.log(s) / p[..., None])                            # 1 / ||u||_p
    v = mx.array(u)[None, None] * inv_norm[..., None]                       # [B, N, Vb, 3]
    v = v * (0.5 * size)[:, :, None, :]
    R = yaw_matrix(yaw)                                                     # [B, N, 3, 3]
    v = matvec3(R[:, :, None], v) + center[:, :, None, :]
    faces = tile_faces(f, N, Vb)
    meta = {"box": np.repeat(np.arange(N, dtype=np.int32), Vb), "face_id": np.tile(fid, N),
            "n_vert_per_box": Vb, "n_face_per_box": f.shape[0]}
    return v.reshape(B, N * Vb, 3), faces, meta


# ----------------------------------------------------------------------------- ground plane

def ground_plane(center: mx.array, extent: float = 60.0, n: int = 16, power: float = 2.5, y: float = 0.0):
    """Square ground grid on the plane y = `y`, facing +Y, centred (in x, z) on `center`.

    Grid coordinates are ``extent * sign(s) * |s|^power`` for s uniform in [-1, 1], so cells are
    small near the centre and large far away. Pass the camera position as `center` for
    training renders: the rasteriser drops triangles that cross the near plane, and with
    small cells under the camera only invisible ones are lost.

    Args:
      center: [B, 3] (only x and z are used). extent: half size (metres). n: cells per side
        (even). power: spacing exponent (1 = uniform).
    Returns:
      verts [B, (n+1)^2, 3], faces [2 n^2, 3], meta {"grid": (n+1, n+1)}.
    """
    s = np.linspace(-1.0, 1.0, n + 1)
    g = (extent * np.sign(s) * np.abs(s) ** power).astype(np.float32)
    gx, gz = np.meshgrid(g, g, indexing="ij")                    # [n+1, n+1]: x index i, z index j
    base = np.stack([gx, np.full_like(gx, y), gz], -1).reshape(-1, 3)
    faces = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = i * (n + 1) + j, i * (n + 1) + j + 1, (i + 1) * (n + 1) + j + 1, (i + 1) * (n + 1) + j
            faces += [[a, b, c], [a, c, d]]          # (x0,z0) -> (x0,z1) -> (x1,z1): normal +Y
    faces = np.array(faces, np.int32)
    off = mx.stack([center[:, 0], mx.zeros_like(center[:, 0]), center[:, 2]], -1)
    verts = mx.array(base)[None] + off[:, None, :]
    return verts, faces, {"grid": (n + 1, n + 1)}


# ----------------------------------------------------------------------------- mesh utilities

def tile_faces(faces: np.ndarray, n: int, n_vert: int) -> np.ndarray:
    """Repeat a face list for n copies of a mesh with n_vert vertices each -> [n * F, 3]."""
    faces = np.asarray(faces, np.int32)
    return (faces[None] + (np.arange(n, dtype=np.int32) * n_vert)[:, None, None]).reshape(-1, 3)


def merge_meshes(*meshes):
    """Concatenate meshes of the same batch: each (verts [B, Vi, 3], faces [Fi, 3]) ->
    (verts [B, sum Vi, 3], faces [sum Fi, 3], vertex offsets list)."""
    vs, fs, offs = [], [], []
    o = 0
    for v, f in meshes:
        vs.append(v)
        fs.append(np.asarray(f, np.int32) + o)
        offs.append(o)
        o += v.shape[1]
    return mx.concatenate(vs, 1), np.concatenate(fs, 0), offs


# ----------------------------------------------------------------------------- humanoid template

REGIONS = ("skin", "top", "bottom", "shoes", "hair")
SKIN, TOP, BOTTOM, SHOES, HAIR = range(5)
SHAPE_PARAMS = ("height", "girth", "head", "shoulder")
N_SHAPE = len(SHAPE_PARAMS)

# scale channels, functions of (h, g, hs, sw) = shape[:, :4]
_CH = {"h": 0, "hg": 1, "hsw": 2, "hhs": 3, "hgm": 4}
_N_CH = 5
_SHOULDER_JOINTS = (JID["l_clavicle"], JID["l_shoulder"], JID["r_clavicle"], JID["r_shoulder"])

_OCT_C = 0.5     # chamfer of the octagonal cross-sections
_OCTAGON = np.array([[1, _OCT_C], [_OCT_C, 1], [-_OCT_C, 1], [-1, _OCT_C], [-1, -_OCT_C], [-_OCT_C, -1],
                     [_OCT_C, -1], [1, -_OCT_C]], np.float32)
_SQUARE = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], np.float32)

X3, Y3, Z3 = np.eye(3, dtype=np.float32)


def _channels(shape: mx.array) -> mx.array:
    """[B, >=4] shape (height scale, girth, head size, shoulder width; 1 = average) -> [B, 5] scales."""
    h, g, hs, sw = shape[:, 0], shape[:, 1], shape[:, 2], shape[:, 3]
    return mx.stack([h, h * g, h * sw, h * hs, h * (0.5 + 0.5 * g)], -1)


def humanoid_offsets(shape):
    """Skeleton offsets [B, 22, 3] (MLX) or [22, 3] (numpy, for one shape [4]): ``OFFSETS * height``,
    with the x offsets of both clavicles and shoulders also scaled by the shoulder width."""
    if isinstance(shape, mx.array):
        h, sw = shape[:, 0], shape[:, 3]
        mask = np.zeros((22, 3), np.float32)
        mask[list(_SHOULDER_JOINTS), 0] = 1
        scale = 1 + mx.array(mask)[None] * (sw[:, None, None] - 1)
        return mx.array(OFFSETS)[None] * h[:, None, None] * scale
    shape = np.asarray(shape, np.float64)
    off = OFFSETS.astype(np.float64) * shape[0]
    off[list(_SHOULDER_JOINTS), 0] *= shape[3]
    return off.astype(np.float32)


class _Builder:
    """Accumulates static per-vertex terms: local = To*S[io] + Tl*S[il] + Tw*S[iw] + Td*S[id]."""

    def __init__(self):
        self.T = [[], [], [], []]
        self.ch = [[], [], [], []]
        self.joint, self.region, self.part, self.faces = [], [], [], []
        self.part_names = []
        self.nv = 0

    def add(self, name, joint, terms, chans, faces, region):
        """terms: 4 arrays [n, 3] (rest metres at shape 1); chans: 4 channel names or [n] arrays;
        region: int or [n] array; faces local to this part (oriented later)."""
        n = terms[0].shape[0]
        pid = len(self.part_names)
        self.part_names.append(name)
        for k in range(4):
            self.T[k].append(np.asarray(terms[k], np.float32))
            c = chans[k]
            self.ch[k].append(np.full(n, _CH[c], np.int32) if isinstance(c, str) else np.asarray(c, np.int32))
        self.joint.append(np.full(n, JID[joint], np.int32))
        self.region.append(np.broadcast_to(np.asarray(region, np.int32), (n,)).copy())
        self.part.append(np.full(n, pid, np.int32))
        self.faces.append(np.asarray(faces, np.int32) + self.nv)
        self.nv += n

    def prism(self, name, joint, region, El, Ew, Ed, rings, poly=_OCTAGON, origin=(0, 0, 0), wch="hg", dch="hg",
              lch="h", och="h"):
        """Tapered prism along El. rings: (l, half_w, half_d[, dy]) in metres from `origin`;
        the cross-section `poly` (unit square bounded) spans Ew (width) and Ed (depth); dy shifts a
        ring along Ed (scaled like the origin)."""
        El, Ew, Ed = (np.asarray(a, np.float32) for a in (El, Ew, Ed))
        k = len(poly)
        To, Tl, Tw, Td = [], [], [], []
        for ring in rings:
            l, w, d = ring[:3]
            dy = ring[3] if len(ring) > 3 else 0.0
            for px, py in poly:
                To.append(np.asarray(origin, np.float32) + Ed * dy)
                Tl.append(El * l)
                Tw.append(Ew * (w * px))
                Td.append(Ed * (d * py))
        faces = []
        R = len(rings)
        for r in range(R - 1):
            for i in range(k):
                a, b = r * k + i, r * k + (i + 1) % k
                c, d = (r + 1) * k + (i + 1) % k, (r + 1) * k + i
                faces += [[a, b, c], [a, c, d]]
        for r in (0, R - 1):
            for i in range(1, k - 1):
                faces.append([r * k, r * k + i, r * k + i + 1])
        terms = [np.array(t) for t in (To, Tl, Tw, Td)]
        pos = sum(terms)
        faces = _orient_prism(pos, np.array(faces, np.int32), El, R, k)
        self.add(name, joint, terms, [och, lch, wch, dch], faces, region)


def _orient_prism(pos, faces, El, R, k):
    """Orient side faces away from the axis and caps along -El / +El."""
    ring_c = pos.reshape(R, k, 3).mean(1)
    a, b, c = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    cen = (a + b + c) / 3
    # closest axis point: project the centroid on the polyline of ring centres
    ref = np.empty_like(cen)
    for i, x in enumerate(cen):
        best, bd = ring_c[0], 1e9
        for r in range(R - 1):
            p0, p1 = ring_c[r], ring_c[r + 1]
            e = p1 - p0
            t = np.clip(np.dot(x - p0, e) / max(np.dot(e, e), 1e-12), 0, 1)
            q = p0 + t * e
            dd = np.linalg.norm(x - q)
            if dd < bd:
                best, bd = q, dd
        ref[i] = best
    n_side = 2 * k * (R - 1)
    ref[n_side:n_side + (k - 2)] = ring_c[0] + El * 1.0           # first cap faces -El: reference beyond it
    ref[n_side + (k - 2):] = ring_c[-1] - El * 1.0
    return _orient_outward(pos, faces, ref)


def _build_humanoid():
    bld = _Builder()
    up, down = Y3, -Y3
    # ---- torso (along +Y; width X, depth Z)
    bld.prism("pelvis", "pelvis", BOTTOM, up, X3, Z3,
              [(-0.13, 0.145, 0.092), (-0.06, 0.165, 0.105), (0.08, 0.155, 0.098)])
    bld.prism("spine1", "spine1", TOP, up, X3, Z3, [(-0.03, 0.150, 0.095), (0.13, 0.152, 0.097)])
    bld.prism("spine2", "spine2", TOP, up, X3, Z3, [(-0.01, 0.155, 0.098), (0.13, 0.168, 0.104)])
    bld.prism("chest", "chest", TOP, up, X3, Z3,
              [(-0.01, 0.17, 0.104), (0.10, 0.195, 0.108), (0.175, 0.15, 0.075)],
              wch=np.array([_CH["hg"]] * 8 + [_CH["hsw"]] * 16), dch="hg")
    bld.prism("neck", "neck", SKIN, up, X3, Z3, [(-0.03, 0.05, 0.052), (0.12, 0.046, 0.048)], wch="hgm", dch="hgm")
    # ---- head: 3x3 rounded box, hair on the top / back faces (separate vertices along the hairline)
    u, f, _ = cube_grid(3)
    v = _superellipsoid_np(u, 0.55)
    cen = v[f].mean(1)
    hair_f = (cen[:, 1] > 0.3) | ((cen[:, 2] < -0.3) & (cen[:, 1] > -0.55))
    half = np.array([0.09, 0.118, 0.104], np.float32)
    hc = np.array([0.0, 0.105, 0.012], np.float32)
    for sel, region, infl in ((~hair_f, SKIN, 1.0), (hair_f, HAIR, 1.06)):
        ff = f[sel]
        used = np.unique(ff)
        remap = -np.ones(len(v), np.int64)
        remap[used] = np.arange(len(used))
        vv = v[used].copy()
        if infl != 1.0:          # puff the hair out, but keep the hairline vertices shared with the face
            seam = np.isin(used, np.unique(f[~hair_f]))
            vv[~seam] *= infl
        To = np.broadcast_to(hc, vv.shape)
        bld.add("hair" if region == HAIR else "head", "head",
                [To, Y3 * (half[1] * vv[:, 1:2]), X3 * (half[0] * vv[:, 0:1]), Z3 * (half[2] * vv[:, 2:3])],
                ["hhs"] * 4, remap[ff], region)
    # ---- arms (T-pose along +-X; cross-section width along Z, depth along Y)
    for side, pre in ((1.0, "l_"), (-1.0, "r_")):
        ax = X3 * side
        bld.prism(pre + "upperarm", pre + "shoulder", TOP, ax, Z3, Y3,
                  [(-0.04, 0.056, 0.058), (0.10, 0.053, 0.055), (0.28, 0.043, 0.045)])
        bld.prism(pre + "forearm", pre + "elbow", SKIN, ax, Z3, Y3, [(-0.01, 0.042, 0.043), (0.25, 0.031, 0.027)])
        bld.prism(pre + "hand", pre + "wrist", SKIN, ax, Z3, Y3, [(0.0, 0.042, 0.017), (0.17, 0.037, 0.012)],
                  poly=_SQUARE, wch="h", dch="h")
    # ---- legs (along -Y; width X, depth Z) and shoes (along +Z; width X, height Y)
    for side, pre in ((1.0, "l_"), (-1.0, "r_")):
        bld.prism(pre + "thigh", pre + "hip", BOTTOM, down, X3, Z3,
                  [(-0.08, 0.085, 0.088), (0.12, 0.080, 0.083), (0.43, 0.055, 0.058)])
        bld.prism(pre + "shin", pre + "knee", BOTTOM, down, X3, Z3,
                  [(-0.01, 0.055, 0.058), (0.13, 0.054, 0.062), (0.40, 0.037, 0.040)])
        bld.prism(pre + "foot", pre + "ankle", SHOES, Z3, X3, Y3,
                  [(-0.06, 0.046, 0.05, -0.01), (0.135, 0.052, 0.03, -0.03)], poly=_SQUARE, wch="hgm", dch="h")
        bld.prism(pre + "toe", pre + "toe", SHOES, Z3, X3, Y3,
                  [(-0.01, 0.052, 0.02, 0.01), (0.075, 0.045, 0.013, 0.003)], poly=_SQUARE, wch="hgm", dch="h")
    T = [np.concatenate(t, 0) for t in bld.T]
    ch = [np.concatenate(c, 0) for c in bld.ch]
    return {
        "T": T, "ch": ch,
        "joint": np.concatenate(bld.joint), "region": np.concatenate(bld.region),
        "part": np.concatenate(bld.part), "part_names": bld.part_names,
        "faces": np.concatenate(bld.faces, 0).astype(np.int32),
    }


HUMANOID_TEMPLATE = _build_humanoid()


def humanoid_rest_local(shape: mx.array) -> mx.array:
    """Vertex positions [B, V, 3] in the frame of their joint (``v_rest - rest_joint``) for shapes [B, 4]."""
    t = HUMANOID_TEMPLATE
    S = _channels(shape)                                       # [B, 5]
    loc = 0
    for k in range(4):
        onehot = mx.array(np.eye(_N_CH, dtype=np.float32)[t["ch"][k]])      # [V, 5] static
        s = (S[:, None, :] * onehot[None]).sum(-1)             # [B, V], deterministic backward (no scatter)
        loc = loc + mx.array(t["T"][k])[None] * s[..., None]
    return loc


def humanoid(shape: mx.array, joint_rot: mx.array, root_rot: mx.array, root_pos: mx.array):
    """Low-poly person on the humanoid22 skeleton, rigidly skinned (one joint per vertex).

    Args:
      shape: [B, 4] multipliers, 1 = average adult (``SHAPE_PARAMS``): height scale (scales the
        skeleton offsets; 1 is about 1.78 m tall), girth (limb and torso thickness), head size,
        shoulder width (also scales the clavicle and shoulder offsets).
      joint_rot: [B, 22, 3, 3] local joint rotations relative to the rest T-pose (humanoid22
        order; the format's ``pose`` track as matrices). root_rot: [B, 3, 3]; root_pos: [B, 3]
        (the entity transform; the rest pose stands on y = 0 facing +Z).

    Returns:
      verts [B, V, 3], faces [F, 3], meta with static per-vertex "joint" (humanoid22 index),
      "region" (0 skin, 1 top, 2 bottom, 3 shoes, 4 hair), "part", plus "part_names" and the
      posed joints "joint_rot" [B, 22, 3, 3], "joint_pos" [B, 22, 3] (MLX).
    """
    t = HUMANOID_TEMPLATE
    local = humanoid_rest_local(shape)
    offs = humanoid_offsets(shape)
    Rw, pw = forward_kinematics(joint_rot, root_rot, root_pos, offs, PARENTS)
    verts = skin_rigid(local, t["joint"], Rw, pw)
    meta = {"joint": t["joint"], "region": t["region"], "part": t["part"], "part_names": t["part_names"],
            "joint_rot": Rw, "joint_pos": pw}
    return verts, t["faces"], meta


def humanoid_rest_numpy(shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rest-pose mesh for one shape [4] in entity space (feet on y = 0, facing +Z):
    (positions [V, 3], faces [F, 3], skeleton offsets [22, 3]) as numpy, the bind pose for export."""
    shape = np.asarray(shape, np.float32).reshape(1, -1)
    local = np.array(humanoid_rest_local(mx.array(shape)))[0]
    offs = humanoid_offsets(shape[0])
    rest = rest_joint_positions(offs.astype(np.float64))
    pos = local + rest[HUMANOID_TEMPLATE["joint"]]
    return pos.astype(np.float32), HUMANOID_TEMPLATE["faces"], offs
