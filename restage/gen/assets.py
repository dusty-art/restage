"""Low-poly mesh builders for the Restage procedural generator.

Every builder returns a *mesh dict*::

    {"positions": f32 [V, 3], "indices": u32 [F, 3], "colors": f32 [V, 3],
     # optional
     "joints": u8 [V, 4], "weights": f32 [V, 4],      # rigid skinning (weight 1 on joints[:, 0])
     "morphs": [{"name": str, "positions": f32 [V, 3]}],
     "material": "m_matte" | "m_glow_xxxxxx",           # material id used when the mesh is written
     "colliders": [[min xyz], [max xyz]] boxes in mesh-local space}

Conventions follow docs/SCENE_FORMAT.md: metres, +Y up, objects and characters face +Z,
triangles are counter-clockwise seen from outside (three.js front faces), colours are
linear RGB. Room shells are single-sided and face inward so a viewer can look in from outside.

All assets are original: simple shapes and an original colour palette.
"""
from __future__ import annotations

import numpy as np

from ..scene import HUMANOID22, JID, quat_axis_angle, quat_rotate

# ---------------------------------------------------------------- colour helpers


def srgb_to_linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, np.float64), 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def rgb(h: str) -> np.ndarray:
    """'#rrggbb' (sRGB, as a designer would pick it) to linear RGB."""
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return srgb_to_linear([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]).astype(np.float32)


def shade(c, k: float):
    """Scale a linear colour (k < 1 darker, k > 1 lighter), clipped to 0..1."""
    return np.clip(np.asarray(c, np.float32) * k, 0, 1).astype(np.float32)


def mix(a, b, t: float):
    return (np.asarray(a, np.float32) * (1 - t) + np.asarray(b, np.float32) * t).astype(np.float32)


def jitter_color(c, rng, amount=0.08):
    """Small brightness / hue variation of a linear colour."""
    c = np.asarray(c, np.float32)
    k = 1 + rng.uniform(-amount, amount)
    tint = 1 + rng.uniform(-amount / 2, amount / 2, 3)
    return np.clip(c * k * tint, 0, 1).astype(np.float32)


# ---------------------------------------------------------------- mesh core

def make_mesh(positions, indices, colors=None, color=(0.8, 0.8, 0.8)) -> dict:
    p = np.asarray(positions, np.float32).reshape(-1, 3)
    f = np.asarray(indices, np.uint32).reshape(-1, 3)
    if colors is None:
        colors = np.broadcast_to(np.asarray(color, np.float32), p.shape)
    return {"positions": p, "indices": f, "colors": np.array(colors, np.float32).reshape(-1, 3)}


def empty_mesh() -> dict:
    return make_mesh(np.zeros((0, 3)), np.zeros((0, 3)))


def _orient_outward(p, f, center):
    """Flip triangles of a convex solid so their normals point away from `center`."""
    a, b, c = p[f[:, 0]], p[f[:, 1]], p[f[:, 2]]
    n = np.cross(b - a, c - a)
    out = ((a + b + c) / 3 - center)
    flip = (n * out).sum(1) < 0
    f = f.copy()
    f[flip] = f[flip][:, [0, 2, 1]]
    return f


def _convex(p, f, color, center=None):
    p = np.asarray(p, np.float32)
    f = np.asarray(f, np.int64)
    center = p.mean(0) if center is None else np.asarray(center, np.float32)
    return make_mesh(p, _orient_outward(p, f, center), color=color)


def merge(*meshes) -> dict:
    """Concatenate meshes (skin attributes are kept when every part has them)."""
    meshes = [m for m in meshes if m is not None and len(m["positions"])]
    if not meshes:
        return empty_mesh()
    out = {"positions": np.concatenate([m["positions"] for m in meshes]).astype(np.float32),
           "colors": np.concatenate([m["colors"] for m in meshes]).astype(np.float32)}
    idx, base = [], 0
    for m in meshes:
        idx.append(m["indices"].astype(np.int64) + base)
        base += len(m["positions"])
    out["indices"] = np.concatenate(idx).astype(np.uint32)
    if all("joints" in m for m in meshes):
        out["joints"] = np.concatenate([m["joints"] for m in meshes]).astype(np.uint8)
        out["weights"] = np.concatenate([m["weights"] for m in meshes]).astype(np.float32)
    names = {mo["name"] for m in meshes for mo in m.get("morphs", [])}
    if names:
        out["morphs"] = []
        for name in sorted(names):
            parts = []
            for m in meshes:
                d = next((mo["positions"] for mo in m.get("morphs", []) if mo["name"] == name), None)
                parts.append(np.zeros_like(m["positions"]) if d is None else d)
            out["morphs"].append({"name": name, "positions": np.concatenate(parts).astype(np.float32)})
    cols = [c for m in meshes for c in m.get("colliders", [])]
    if cols:
        out["colliders"] = cols
    mats = {m.get("material", "m_matte") for m in meshes}
    if len(mats) == 1 and "m_matte" not in mats:
        out["material"] = mats.pop()
    return out


def transform(m: dict, t=(0.0, 0.0, 0.0), q=None, s=None) -> dict:
    """Return a copy of `m` scaled by s, rotated by quaternion q, then translated by t."""
    out = dict(m)
    p = m["positions"].astype(np.float64)
    if s is not None:
        p = p * np.asarray(s, np.float64)
    if q is not None:
        p = quat_rotate(np.broadcast_to(np.asarray(q, np.float64), p.shape[:-1] + (4,)), p)
    out["positions"] = (p + np.asarray(t, np.float64)).astype(np.float32)
    if s is not None and np.prod(np.sign(np.broadcast_to(np.asarray(s, float), (3,)))) < 0:
        out["indices"] = m["indices"][:, [0, 2, 1]].copy()   # mirrored: keep CCW winding
    if "morphs" in m:
        out["morphs"] = []
        for mo in m["morphs"]:
            d = mo["positions"].astype(np.float64)
            if s is not None:
                d = d * np.asarray(s, np.float64)
            if q is not None:
                d = quat_rotate(np.broadcast_to(np.asarray(q, np.float64), d.shape[:-1] + (4,)), d)
            out["morphs"].append({"name": mo["name"], "positions": d.astype(np.float32)})
    if "colliders" in m:
        out["colliders"] = [_xform_box(b, t, q, s) for b in m["colliders"]]
    return out


def _xform_box(box_, t, q, s):
    lo, hi = np.asarray(box_[0], float), np.asarray(box_[1], float)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    if s is not None:
        corners = corners * np.asarray(s, float)
    if q is not None:
        corners = quat_rotate(np.broadcast_to(np.asarray(q, float), (8, 4)), corners)
    corners = corners + np.asarray(t, float)
    return [corners.min(0).tolist(), corners.max(0).tolist()]


def yaw_q(angle: float) -> np.ndarray:
    return quat_axis_angle([0, 1, 0], angle)


def recolor(m: dict, color) -> dict:
    out = dict(m)
    out["colors"] = np.broadcast_to(np.asarray(color, np.float32), m["positions"].shape).copy()
    return out


def facet(m: dict, rng=None, jitter: float = 0.0) -> dict:
    """Unshare vertices so every triangle owns its corners; optionally jitter per-face brightness."""
    f = m["indices"].astype(np.int64)
    flat = f.reshape(-1)
    out = {"positions": m["positions"][flat].copy(), "colors": m["colors"][flat].copy(),
           "indices": np.arange(len(flat), dtype=np.uint32).reshape(-1, 3)}
    if "joints" in m:
        out["joints"], out["weights"] = m["joints"][flat].copy(), m["weights"][flat].copy()
    if "morphs" in m:
        out["morphs"] = [{"name": mo["name"], "positions": mo["positions"][flat].copy()} for mo in m["morphs"]]
    for k in ("material", "colliders"):
        if k in m:
            out[k] = m[k]
    if jitter and rng is not None:
        k = 1 + rng.uniform(-jitter, jitter, len(f))
        out["colors"] = np.clip(out["colors"] * np.repeat(k, 3)[:, None], 0, 1).astype(np.float32)
    return out


def skin(m: dict, joint: int) -> dict:
    """Rigidly bind every vertex of `m` to one joint."""
    out = dict(m)
    v = len(m["positions"])
    j = np.zeros((v, 4), np.uint8)
    j[:, 0] = joint
    w = np.zeros((v, 4), np.float32)
    w[:, 0] = 1
    out["joints"], out["weights"] = j, w
    return out


def with_colliders(m: dict, boxes=None) -> dict:
    out = dict(m)
    if boxes is None:
        p = m["positions"]
        boxes = [[p.min(0).tolist(), p.max(0).tolist()]]
    out["colliders"] = [[list(map(float, b[0])), list(map(float, b[1]))] for b in boxes]
    return out


def colliders(m: dict):
    """Local collision boxes of a mesh (explicit, or its bounding box)."""
    if m.get("colliders"):
        return m["colliders"]
    p = m["positions"]
    if not len(p):
        return []
    return [[p.min(0).tolist(), p.max(0).tolist()]]


def emissive(m: dict, color) -> dict:
    """Mark a mesh as glowing with a given linear colour (uses an emissive material)."""
    out = dict(m)
    c = np.clip(np.asarray(color, float), 0, 1)
    out["material"] = "m_glow_" + "".join(f"{int(round(x * 255)):02x}" for x in c)
    out["colors"] = np.broadcast_to(np.asarray(color, np.float32), m["positions"].shape).copy()
    return out


def signed_volume(m: dict) -> float:
    p, f = m["positions"].astype(np.float64), m["indices"].astype(np.int64)
    return float(np.einsum("ij,ij->i", p[f[:, 0]], np.cross(p[f[:, 1]], p[f[:, 2]])).sum() / 6)


# ---------------------------------------------------------------- primitives

def box(size=(1, 1, 1), center=(0, 0, 0), color=(0.8, 0.8, 0.8)) -> dict:
    """Axis-aligned box with 4 vertices per face (crisp flat colours)."""
    h = np.asarray(size, np.float32) / 2
    c = np.asarray(center, np.float32)
    pos, idx = [], []
    for axis in range(3):
        for sgn in (1, -1):
            n = np.zeros(3)
            n[axis] = sgn
            u = np.zeros(3)
            u[(axis + 1) % 3] = 1
            v = np.cross(n, u)
            base = len(pos)
            for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                pos.append(c + (n + su * u + sv * v) * h)
            idx += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    return make_mesh(pos, idx, color=color)


def box_minmax(lo, hi, color=(0.8, 0.8, 0.8)) -> dict:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return box(hi - lo, (lo + hi) / 2, color)


def tapered_box(bottom=(1, 1), top=(1, 1), y0=0.0, y1=1.0, color=(0.8, 0.8, 0.8), top_shift=(0.0, 0.0),
                center=(0.0, 0.0)) -> dict:
    """Frustum of a box: (w, d) footprint at y0 and at y1, top optionally shifted in x/z."""
    cx, cz = center
    pts = []
    for (w, d), y, (sx, sz) in ((bottom, y0, (0, 0)), (top, y1, top_shift)):
        for ix, iz in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            pts.append([cx + sx + ix * w / 2, y, cz + sz + iz * d / 2])
    quads = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    p = np.array(pts, np.float32)
    # unshared corners per face for crisp colours
    pos, idx = [], []
    for q in quads:
        b = len(pos)
        pos += [p[i] for i in q]
        idx += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    pos = np.array(pos, np.float32)
    return make_mesh(pos, _orient_outward(pos, np.array(idx), p.mean(0)), color=color)


def _ring(n, r, y, phase=0.0, rz=None):
    a = phase + 2 * np.pi * np.arange(n) / n
    return np.stack([r * np.cos(a), np.full(n, y), (r if rz is None else rz) * np.sin(a)], 1)


def cylinder(r=0.5, h=1.0, n=8, color=(0.8, 0.8, 0.8), r_top=None, y0=0.0, phase=None, caps=True,
             rz=None) -> dict:
    """Vertical (optionally tapered, elliptic) prism from y0 to y0 + h."""
    r_top = r if r_top is None else r_top
    phase = np.pi / n if phase is None else phase
    if r_top <= 1e-6:
        return cone(r, h, n, color, y0, phase)
    k = rz / r if rz is not None else 1.0
    b = _ring(n, r, y0, phase, r * k)
    t = _ring(n, r_top, y0 + h, phase, r_top * k)
    pos, idx = [], []
    for i in range(n):
        j = (i + 1) % n
        base = len(pos)
        pos += [b[i], b[j], t[j], t[i]]
        idx += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    if caps:
        for ring in (b, t):
            base = len(pos)
            pos += list(ring)
            idx += [[base, base + i, base + i + 1] for i in range(1, n - 1)]
    pos = np.array(pos, np.float32)
    return make_mesh(pos, _orient_outward(pos, np.array(idx), [0, y0 + h / 2, 0]), color=color)


def cone(r=0.5, h=1.0, n=8, color=(0.8, 0.8, 0.8), y0=0.0, phase=None) -> dict:
    phase = np.pi / n if phase is None else phase
    b = _ring(n, r, y0, phase)
    apex = np.array([0, y0 + h, 0])
    pos, idx = [], []
    for i in range(n):
        base = len(pos)
        pos += [b[i], b[(i + 1) % n], apex]
        idx.append([base, base + 1, base + 2])
    base = len(pos)
    pos += list(b)
    idx += [[base, base + i, base + i + 1] for i in range(1, n - 1)]
    pos = np.array(pos, np.float32)
    return make_mesh(pos, _orient_outward(pos, np.array(idx), [0, y0 + h / 3, 0]), color=color)


_ICO = None


def _icosahedron():
    global _ICO
    if _ICO is None:
        t = (1 + 5 ** 0.5) / 2
        v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t],
                      [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
        f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
                      [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
                      [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
        _ICO = (v / np.linalg.norm(v, axis=1, keepdims=True), f)
    return _ICO


def sphere_points(subdiv=1):
    v, f = _icosahedron()
    v = list(v)
    for _ in range(subdiv):
        cache, nf = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = (v[a] + v[b]) / 2
                v.append(m / np.linalg.norm(m))
                cache[key] = len(v) - 1
            return cache[key]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = np.array(nf)
    return np.array(v), np.array(f)


def icosphere(r=0.5, subdiv=1, color=(0.8, 0.8, 0.8), center=(0, 0, 0), scale=(1, 1, 1), rng=None,
              jitter=0.0) -> dict:
    """Low-poly sphere (subdiv 0: 20 faces, 1: 80 faces); optional radial jitter for rocks / foliage."""
    v, f = sphere_points(subdiv)
    rad = np.full(len(v), r, float)
    if jitter and rng is not None:
        rad *= 1 + rng.uniform(-jitter, jitter, len(v))
    p = v * rad[:, None] * np.asarray(scale, float) + np.asarray(center, float)
    return make_mesh(p, _orient_outward(p, f, center), color=color)


def _basis_to(d):
    """Quaternion rotating +Y onto direction d."""
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    y = np.array([0.0, 1.0, 0.0])
    c = float(np.dot(y, d))
    if c > 1 - 1e-9:
        return np.array([0, 0, 0, 1.0])
    if c < -1 + 1e-9:
        return np.array([1.0, 0, 0, 0])
    ax = np.cross(y, d)
    return quat_axis_angle(ax, np.arccos(np.clip(c, -1, 1)))


def limb(a, b, ra=0.05, rb=None, n=6, color=(0.8, 0.8, 0.8), rz_scale=1.0, cap=0.0) -> dict:
    """Tapered prism from point a to point b (radius ra at a, rb at b), optional pointed caps."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rb = ra if rb is None else rb
    L = float(np.linalg.norm(b - a))
    m = cylinder(ra, L, n, color, r_top=rb, rz=ra * rz_scale)
    if cap > 0:
        tip_lo = np.array([0, -cap * ra, 0])
        tip_hi = np.array([0, L + cap * rb, 0])
        ring_lo = _ring(n, ra, 0, np.pi / n, ra * rz_scale)
        ring_hi = _ring(n, rb, L, np.pi / n, rb * rz_scale)
        pos = list(ring_lo) + list(ring_hi) + [tip_lo, tip_hi]
        idx = []
        for i in range(n):
            j = (i + 1) % n
            idx += [[i, j, n + j], [i, n + j, n + i], [i, j, 2 * n], [n + i, n + j, 2 * n + 1]]
        pos = np.array(pos, np.float32)
        m = make_mesh(pos, _orient_outward(pos, np.array(idx), [0, L / 2, 0]), color=color)
    return transform(m, a, _basis_to(b - a))


def quad(corners, color=(0.8, 0.8, 0.8)) -> dict:
    """Single-sided quad; corners in CCW order seen from the visible side."""
    return make_mesh(corners, [[0, 1, 2], [0, 2, 3]], color=color)


def grid_patch(w, d, nx, nz, y=0.0, color=(0.8, 0.8, 0.8), colors_fn=None, center=(0, 0)) -> dict:
    """Upward-facing tiled floor (unshared, so each tile can have its own colour)."""
    xs = np.linspace(-w / 2, w / 2, nx + 1) + center[0]
    zs = np.linspace(-d / 2, d / 2, nz + 1) + center[1]
    pos, idx, col = [], [], []
    for i in range(nx):
        for k in range(nz):
            c = color if colors_fn is None else colors_fn(i, k)
            b = len(pos)
            pos += [[xs[i], y, zs[k]], [xs[i], y, zs[k + 1]], [xs[i + 1], y, zs[k + 1]], [xs[i + 1], y, zs[k]]]
            idx += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
            col += [c] * 4
    return make_mesh(pos, idx, col)


# ---------------------------------------------------------------- humanoid

SKIN_TONES = ["#f6d5b8", "#eec4a0", "#e0ac84", "#c98f65", "#b07450", "#8f5a3c", "#6e4430", "#4f3022"]
HAIR_COLORS = ["#1c130d", "#0e0b09", "#3b2618", "#6b4a32", "#a37b52", "#d6b98a", "#8a3b22", "#9c9690",
               "#e8e0d2"]
CLOTH_COLORS = ["#2f4b7c", "#c0392b", "#e8e6e1", "#26262b", "#5c7a3a", "#d9a441", "#7d3c98", "#1f7a8c",
                "#e07a5f", "#3d405b", "#81b29a", "#f2cc8f", "#9a8c98", "#b5838d", "#6d597a", "#335c67",
                "#e09f3e", "#540b0e", "#2a9d8f", "#264653", "#f4f1de", "#8d99ae", "#ef476f", "#118ab2"]
PANTS_COLORS = ["#2b3a55", "#1d1f24", "#3f4a3c", "#5b4636", "#7a6a55", "#4a5568", "#2f3e46", "#6b705c",
                "#d8cbb5", "#283d3b"]
SHOE_COLORS = ["#f2f2f2", "#1a1a1a", "#5a3a22", "#8a8a8a", "#2c3e50", "#a33b2b"]


def humanoid_params(rng: np.random.Generator, style: str = "realistic") -> dict:
    """Random but plausible appearance parameters for `humanoid()`."""
    height = float(np.clip(rng.normal(1.72, 0.09), 1.52, 1.98))
    feminine = bool(rng.random() < 0.5)
    return {
        "style": style,
        "height": height,
        "build": float(np.clip(rng.normal(1.0, 0.1), 0.82, 1.3)),
        "feminine": feminine,
        "skin": rgb(SKIN_TONES[rng.integers(len(SKIN_TONES))]),
        "hair_color": rgb(HAIR_COLORS[rng.integers(len(HAIR_COLORS))]),
        "hair": str(rng.choice(["short", "long", "bun", "ponytail", "bob"] if feminine
                               else ["short", "buzz", "short", "bald", "long", "curly"])),
        "beard": bool(not feminine and rng.random() < 0.3),
        "top_color": rgb(CLOTH_COLORS[rng.integers(len(CLOTH_COLORS))]),
        "top": str(rng.choice(["tshirt", "longsleeve", "jacket", "tank"], p=[0.4, 0.3, 0.2, 0.1])),
        "jacket_color": rgb(CLOTH_COLORS[rng.integers(len(CLOTH_COLORS))]),
        "bottom": str(rng.choice(["pants", "shorts", "skirt"] if feminine else ["pants", "shorts"],
                                 p=[0.6, 0.2, 0.2] if feminine else [0.8, 0.2])),
        "bottom_color": rgb(PANTS_COLORS[rng.integers(len(PANTS_COLORS))]),
        "shoe_color": rgb(SHOE_COLORS[rng.integers(len(SHOE_COLORS))]),
        "hat": str(rng.choice(["none", "cap", "beanie"], p=[0.75, 0.15, 0.1])),
        "hat_color": rgb(CLOTH_COLORS[rng.integers(len(CLOTH_COLORS))]),
        "backpack": False,
    }


def rest_joint_positions(scale: float = 1.0) -> np.ndarray:
    """World positions of humanoid22 joints in the rest (T-) pose for a skeleton of a given scale."""
    p = np.zeros((len(HUMANOID22), 3))
    for j, (_, par, off) in enumerate(HUMANOID22):
        p[j] = (p[par] if par >= 0 else 0) + np.asarray(off) * scale
    return p


def _head_sphere(cx, cy, cz, rx, ry, rz, subdiv=1):
    """Unit icosphere shaped as a head: narrower jaw, flatter back."""
    v, f = sphere_points(subdiv)
    v = v.copy()
    low = v[:, 1] < 0
    v[low, 0] *= 1 - 0.28 * (-v[low, 1])            # jaw narrows
    v[low, 2] = np.where(v[low, 2] > 0, v[low, 2] * (1 - 0.1 * (-v[low, 1])), v[low, 2] * 0.85)
    v[v[:, 2] < 0, 2] *= 0.95                         # flatter back of the head
    return v, f, np.array([cx, cy, cz]), np.array([rx, ry, rz])


def _sphere_shell(v, f, center, radii, inflate, keep, color):
    """Faces of a (deformed) unit sphere scaled by radii*inflate, selected by keep(unit centroid)."""
    cen = v[f].mean(1)
    sel = keep(cen)
    p = v * radii * inflate + center
    ff = f[sel]
    used = np.unique(ff)
    remap = -np.ones(len(p), int)
    remap[used] = np.arange(len(used))
    pp = p[used]
    return make_mesh(pp, _orient_outward(pp, remap[ff], center), color=color)


def humanoid(params: dict | None = None, rng=None) -> tuple[dict, float]:
    """A low-poly person rigidly skinned to humanoid22, in the rest pose (T-pose, facing +Z).

    Returns (mesh, skeleton_scale). The mesh is built for a skeleton whose offsets are
    humanoid22 OFFSETS * skeleton_scale (use scene.skeleton_doc(scale)).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    P = humanoid_params(rng) if params is None else params
    s = P["height"] / 1.78
    J = rest_joint_positions(s)
    blocky = P.get("style") == "blocky"
    b = P["build"]
    fem = P["feminine"]
    skin_c, bot_c = P["skin"], P["bottom_color"]
    top_c = P["jacket_color"] if P["top"] == "jacket" else P["top_color"]
    shoe_c, hair_c = P["shoe_color"], P["hair_color"]
    sleeve_c = P["jacket_color"] if P["top"] == "jacket" else top_c
    parts: list[dict] = []

    def add(m, joint):
        parts.append(skin(m, JID[joint] if isinstance(joint, str) else joint))

    shoulder_w = (0.36 if not fem else 0.33) * b ** 0.6
    hip_w = (0.31 if not fem else 0.335) * b ** 0.7
    belly = 1.0 + max(0.0, b - 1.0) * 0.9
    y = lambda j: J[JID[j]][1]

    if blocky:
        n_limb = 4
    else:
        n_limb = 6

    # ---- torso (pelvis, spine1, spine2, chest)
    pel = J[JID["pelvis"]]
    add(tapered_box((hip_w, 0.19 * b * s), (hip_w * 0.95, 0.19 * b * s), pel[1] - 0.115 * s, pel[1] + 0.08 * s,
                    bot_c), "pelvis")
    add(box((hip_w * 0.98, 0.025 * s, 0.195 * b * s), (0, pel[1] + 0.07 * s, 0), shade(bot_c, 0.55)), "pelvis")
    if P["bottom"] == "skirt":
        add(tapered_box((hip_w * 1.02, 0.2 * b * s), (hip_w * 1.55, 0.3 * b * s), pel[1] - 0.06 * s,
                        pel[1] - 0.36 * s, bot_c), "pelvis")
    torso = [("spine1", (0.29, 0.18 * belly), (0.30, 0.19 * belly), -0.05, 0.13),
             ("spine2", (0.31, 0.19 * belly), (0.34 if not fem else 0.31, 0.205), -0.02, 0.135),
             ("chest", (0.345 if not fem else 0.315, 0.21 if not fem else 0.225), (shoulder_w + 0.03, 0.18), -0.015, 0.16)]
    for name, bot, top, lo, hi in torso:
        jy = y(name)
        w0, d0 = bot[0] * b ** 0.7 * s, bot[1] * b ** 0.6 * s
        w1, d1 = top[0] * (b ** 0.6 if name != "chest" else 1) * s, top[1] * b ** 0.6 * s
        if name == "chest":
            w1 = top[0] * s
        add(tapered_box((w0, d0), (w1, d1), jy + lo * s, jy + hi * s, top_c, top_shift=(0, -0.005 * s)), name)
    if P["top"] == "jacket":   # open jacket: shirt strip down the front
        for name, bot, top, lo, hi in torso:
            jy = y(name)
            d_front = (top[1] if name != "chest" else 0.2) * b ** 0.6 * s / 2
            add(box((0.075 * s, (hi - lo) * s, 0.01 * s), (0, jy + (lo + hi) / 2 * s, d_front + 0.003 * s),
                    P["top_color"]), name)
    if not blocky:   # trapezius slope from the shoulders to the neck
        cy = y("chest")
        add(tapered_box((shoulder_w + 0.03 * s, 0.17 * b ** 0.6 * s), (0.15 * s, 0.11 * s), cy + 0.155 * s,
                        cy + 0.205 * s, top_c, top_shift=(0, -0.01 * s)), "chest")
    if P.get("backpack"):
        add(box((0.28 * s, 0.38 * s, 0.14 * s), (0, y("spine2") + 0.06 * s, -0.17 * s * b ** 0.6),
                P["jacket_color"]), "chest")

    # ---- neck and head
    nk = J[JID["neck"]]
    add(cylinder(0.052 * s * b ** 0.4, 0.13 * s, 6 if not blocky else 4, skin_c, y0=nk[1] - 0.04 * s), "neck")
    hd = J[JID["head"]]
    hc = np.array([0, hd[1] + 0.105 * s, 0.012 * s])
    if blocky:
        hs = 0.23 * s
        add(box((hs, hs, hs), hc, skin_c), "head")
        add(box((hs * 1.04, hs * 0.35, hs * 1.04), hc + [0, hs * 0.36, -0.005], hair_c), "head")
        add(box((hs * 1.04, hs * 0.7, hs * 0.2), hc + [0, hs * 0.12, -hs * 0.45], hair_c), "head")
        for side in (-1, 1):
            add(box((0.04 * s, 0.03 * s, 0.01 * s), hc + [side * 0.05 * s, 0.0, hs / 2 + 0.004],
                    rgb("#f4f4f4")), "head")
            add(box((0.02 * s, 0.03 * s, 0.012 * s), hc + [side * 0.06 * s, 0.0, hs / 2 + 0.006],
                    rgb("#2b4da8")), "head")
        add(box((0.07 * s, 0.018 * s, 0.01 * s), hc + [0, -0.06 * s, hs / 2 + 0.004], shade(skin_c, 0.55)), "head")
    else:
        rx, ry, rz = 0.088 * s, 0.118 * s, 0.103 * s
        v, f, c, rad = _head_sphere(*hc, rx, ry, rz)
        head = make_mesh(v * rad + c, _orient_outward(v * rad + c, f, c), color=skin_c)
        add(head, "head")
        # face: eyes, brows, nose, mouth (with a jawOpen morph), ears
        eye_y, eye_z = hc[1] + 0.012 * s, hc[2] + rz * 0.9
        for side in (-1, 1):
            add(box((0.024 * s, 0.014 * s, 0.014 * s), (side * 0.034 * s, eye_y, eye_z), rgb("#f2efe9")), "head")
            add(box((0.012 * s, 0.013 * s, 0.01 * s), (side * 0.034 * s, eye_y, eye_z + 0.0045 * s),
                    rgb("#2a1c14")), "head")
            add(box((0.03 * s, 0.008 * s, 0.012 * s), (side * 0.035 * s, eye_y + 0.022 * s, eye_z - 0.001),
                    shade(hair_c, 0.9)), "head")
            add(box((0.016 * s, 0.04 * s, 0.026 * s), (side * rx * 1.02, hc[1] - 0.002 * s, hc[2] - 0.01 * s),
                    shade(skin_c, 0.93)), "head")
        add(tapered_box((0.026 * s, 0.03 * s), (0.014 * s, 0.006 * s), hc[1] - 0.03 * s, hc[1] + 0.01 * s,
                        shade(skin_c, 0.9), top_shift=(0, -0.012 * s), center=(0, hc[2] + rz * 0.93)), "head")
        mouth = box((0.036 * s, 0.008 * s, 0.01 * s), (0, hc[1] - 0.058 * s, hc[2] + rz * 0.76),
                    mix(rgb("#7a3434"), skin_c, 0.45))
        d = np.zeros_like(mouth["positions"])
        d[mouth["positions"][:, 1] < hc[1] - 0.058 * s, 1] = -0.016 * s
        mouth["morphs"] = [{"name": "jawOpen", "positions": d}]
        inner = box((0.03 * s, 0.005 * s, 0.006 * s), (0, hc[1] - 0.059 * s, hc[2] + rz * 0.755),
                    rgb("#3a1414"))
        di = np.zeros_like(inner["positions"])
        di[:, 1] = -0.009 * s
        inner["morphs"] = [{"name": "jawOpen", "positions": di}]
        add(mouth, "head")
        add(inner, "head")
        # hair: shells over the head sphere (hairline in front, ear-level or longer at the back)
        style = P["hair"]
        if style != "bald":
            infl = {"buzz": 1.035, "short": 1.07, "curly": 1.13, "bob": 1.1, "long": 1.09}.get(style, 1.075)
            front = {"buzz": 0.42, "short": 0.36, "curly": 0.3}.get(style, 0.32)
            low = {"bob": -0.55, "long": -0.6}.get(style)

            def keep(cn, front=front, low=low):
                top = (cn[:, 1] > front - 0.25 * np.clip(-cn[:, 2], 0, 1) * 2.4) & \
                      ((cn[:, 2] < 0.55) | (cn[:, 1] > front + 0.1))
                if low is None:
                    return top
                face = (cn[:, 2] > 0.35) & (cn[:, 1] < front + 0.05)
                return top | ((cn[:, 1] > low) & ~face & (cn[:, 2] < 0.5))
            shell = _sphere_shell(v, f, c, rad, infl, keep, hair_c)
            if style == "curly":
                shell = facet(shell, rng, 0.12)
            add(shell, "head")
            if style == "long":
                add(tapered_box((0.15 * s, 0.045 * s), (0.17 * s, 0.07 * s), hc[1] - 0.3 * s, hc[1] - 0.02 * s,
                                hair_c, center=(0, hc[2] - 0.07 * s)), "head")
            elif style == "bun":
                add(icosphere(0.05 * s, 0, hair_c, hc + [0, 0.07 * s, -0.1 * s]), "head")
            elif style == "ponytail":
                add(limb(hc + [0, 0.03 * s, -0.1 * s], hc + [0, -0.2 * s, -0.13 * s], 0.03 * s, 0.018 * s, 5,
                         hair_c), "head")
        if P.get("beard"):
            def keep_b(cn):
                return (cn[:, 1] < -0.15) & (cn[:, 2] > -0.1) & ~((cn[:, 1] > -0.55) & (np.abs(cn[:, 0]) < 0.3)
                                                                  & (cn[:, 2] > 0.7))
            add(_sphere_shell(v, f, c, rad, 1.04, keep_b, shade(hair_c, 0.9)), "head")
        hat = P.get("hat", "none")
        if hat == "cap":
            add(_sphere_shell(v, f, c, rad, 1.12, lambda cn: cn[:, 1] > 0.3, P["hat_color"]), "head")
            add(box((0.15 * s, 0.012 * s, 0.1 * s), (0, hc[1] + 0.06 * s, hc[2] + rz + 0.035 * s),
                    shade(P["hat_color"], 0.85)), "head")
        elif hat == "beanie":
            add(_sphere_shell(v, f, c, rad, 1.14, lambda cn: cn[:, 1] > 0.2, P["hat_color"]), "head")

    # ---- arms (T-pose along +-X)
    for side, pre in ((1, "l_"), (-1, "r_")):
        sh, el, wr = J[JID[pre + "shoulder"]], J[JID[pre + "elbow"]], J[JID[pre + "wrist"]]
        ra = 0.05 * s * b ** 0.7
        if blocky:
            add(box((abs(el[0] - sh[0]) + 0.08 * s, 0.1 * s, 0.1 * s), ((sh + el) / 2), sleeve_c), pre + "shoulder")
            add(box((abs(wr[0] - el[0]), 0.09 * s, 0.09 * s), ((el + wr) / 2), sleeve_c if P["top"] != "tank"
                    else skin_c), pre + "elbow")
            add(box((0.1 * s, 0.085 * s, 0.085 * s), wr + [side * 0.05 * s, 0, 0], skin_c), pre + "wrist")
            continue
        add(icosphere(ra * 1.18, 0, sleeve_c if P["top"] != "tank" else skin_c, sh, scale=(1, 1, 1)), pre + "shoulder")
        if P["top"] in ("longsleeve", "jacket"):
            add(limb(sh, el, ra, ra * 0.82, n_limb, sleeve_c), pre + "shoulder")
            add(icosphere(ra * 0.8, 0, sleeve_c, el), pre + "elbow")
            add(limb(el, wr, ra * 0.8, ra * 0.62, n_limb, sleeve_c), pre + "elbow")
        else:
            mid = sh + (el - sh) * (0.45 if P["top"] == "tshirt" else 0.0)
            if P["top"] == "tshirt":
                add(limb(sh, mid + [side * 0.01 * s, 0, 0], ra * 1.12, ra * 1.05, n_limb, sleeve_c), pre + "shoulder")
            add(limb(mid, el, ra * 0.92, ra * 0.78, n_limb, skin_c), pre + "shoulder")
            add(icosphere(ra * 0.78, 0, skin_c, el), pre + "elbow")
            add(limb(el, wr, ra * 0.78, ra * 0.6, n_limb, skin_c), pre + "elbow")
        # hand: palm, fingers, thumb (palm faces -Y in the T-pose, thumb toward +Z)
        add(box((0.085 * s, 0.028 * s, 0.078 * s), wr + [side * 0.045 * s, 0, 0], skin_c), pre + "wrist")
        add(box((0.07 * s, 0.022 * s, 0.07 * s), wr + [side * 0.12 * s, -0.003 * s, -0.002 * s],
                shade(skin_c, 0.96)), pre + "wrist")
        add(limb(wr + [side * 0.02 * s, -0.005 * s, 0.035 * s], wr + [side * 0.07 * s, -0.012 * s, 0.07 * s],
                 0.013 * s, 0.011 * s, 4, skin_c), pre + "wrist")

    # ---- legs
    for side, pre in ((1, "l_"), (-1, "r_")):
        hp, kn, an, to = (J[JID[pre + n]] for n in ("hip", "knee", "ankle", "toe"))
        rl = 0.074 * s * b ** 0.75 * (1.04 if fem else 1)
        if blocky:
            add(box((0.13 * s, abs(hp[1] - kn[1]) + 0.05 * s, 0.13 * s), (hp + kn) / 2, bot_c), pre + "hip")
            add(box((0.12 * s, abs(kn[1] - an[1]), 0.12 * s), (kn + an) / 2, bot_c), pre + "knee")
            add(box((0.12 * s, 0.09 * s, 0.22 * s), an + [0, -0.015 * s, 0.03 * s], shoe_c), pre + "ankle")
            continue
        shorts = P["bottom"] in ("shorts", "skirt")
        thigh_c = bot_c if P["bottom"] != "skirt" else skin_c
        if P["bottom"] == "shorts":
            mid = hp + (kn - hp) * 0.55
            add(limb(hp + [0, 0.04 * s, 0], mid, rl * 1.05, rl * 0.95, n_limb, bot_c), pre + "hip")
            add(limb(mid, kn, rl * 0.85, rl * 0.68, n_limb, skin_c), pre + "hip")
        else:
            add(limb(hp + [0, 0.04 * s, 0], kn, rl, rl * 0.7, n_limb, thigh_c), pre + "hip")
        shin_c = skin_c if shorts else bot_c
        add(icosphere(rl * 0.7, 0, shin_c, kn), pre + "knee")
        add(limb(kn, an + [0, 0.02 * s, 0], rl * 0.68, rl * 0.5, n_limb, shin_c), pre + "knee")
        # shoe: heel block on the ankle, toe cap on the toe joint; sole touches y = 0 at rest
        add(tapered_box((0.095 * s, 0.17 * s), (0.085 * s, 0.13 * s), 0.0, an[1] + 0.025 * s, shoe_c,
                        center=(an[0], an[2] + 0.04 * s), top_shift=(0, -0.015 * s)), pre + "ankle")
        add(box((0.098 * s, 0.012 * s, 0.175 * s), (an[0], 0.006 * s, an[2] + 0.04 * s), shade(shoe_c, 0.6)
                if shoe_c.mean() > 0.4 else rgb("#d8d8d8")), pre + "ankle")
        add(tapered_box((0.095 * s, 0.085 * s), (0.08 * s, 0.06 * s), 0.0, 0.055 * s, shoe_c,
                        center=(to[0], to[2] + 0.035 * s), top_shift=(0, -0.01 * s)), pre + "toe")

    m = merge(*parts)
    return m, s


def humanoid_rest_height(params) -> float:
    return params["height"]


# ---------------------------------------------------------------- quadruped (dog)

DOG_SKELETON = [  # name, parent, offset (metres, faces +Z)
    ("root", -1, (0.0, 0.42, 0.0)), ("spine", 0, (0.0, 0.02, 0.18)), ("neck", 1, (0.0, 0.06, 0.12)),
    ("head", 2, (0.0, 0.12, 0.06)), ("tail", 0, (0.0, 0.03, -0.2)),
    ("fl_upper", 1, (0.08, -0.04, 0.06)), ("fl_lower", 5, (0.0, -0.19, 0.0)), ("fl_paw", 6, (0.0, -0.17, 0.0)),
    ("fr_upper", 1, (-0.08, -0.04, 0.06)), ("fr_lower", 8, (0.0, -0.19, 0.0)), ("fr_paw", 9, (0.0, -0.17, 0.0)),
    ("bl_upper", 0, (0.08, -0.03, -0.17)), ("bl_lower", 11, (0.0, -0.2, 0.0)), ("bl_paw", 12, (0.0, -0.16, 0.0)),
    ("br_upper", 0, (-0.08, -0.03, -0.17)), ("br_lower", 14, (0.0, -0.2, 0.0)), ("br_paw", 15, (0.0, -0.16, 0.0)),
]


def dog_skeleton_doc(scale=1.0, sid="dog17"):
    return {"id": sid, "joints": [{"name": n, "parent": p, "offset": [x * scale for x in o]}
                                  for n, p, o in DOG_SKELETON]}


def dog(rng=None, scale=1.0) -> dict:
    """Low-poly dog rigidly skinned to DOG_SKELETON (faces +Z)."""
    rng = np.random.default_rng(0) if rng is None else rng
    fur = rgb(["#b07a45", "#2b2420", "#e8dcc6", "#8a6a4a", "#d9b27c"][rng.integers(5)])
    dark = shade(fur, 0.6)
    J = np.zeros((len(DOG_SKELETON), 3))
    for i, (_, p, o) in enumerate(DOG_SKELETON):
        J[i] = (J[p] if p >= 0 else 0) + np.asarray(o) * scale
    s = scale
    parts = []

    def add(m, j):
        parts.append(skin(m, j))
    add(tapered_box((0.2 * s, 0.26 * s), (0.2 * s, 0.24 * s), J[0][1] - 0.09 * s, J[0][1] + 0.1 * s, fur,
                    center=(0, -0.05 * s)), 0)
    add(tapered_box((0.21 * s, 0.24 * s), (0.2 * s, 0.22 * s), J[1][1] - 0.12 * s, J[1][1] + 0.09 * s, fur,
                    center=(0, J[1][2] + 0.0)), 1)
    add(limb(J[2] + [0, -0.04 * s, -0.02 * s], J[3] + [0, -0.02 * s, 0], 0.07 * s, 0.06 * s, 6, fur), 2)
    hc = J[3] + [0, 0.02 * s, 0.04 * s]
    add(box((0.15 * s, 0.13 * s, 0.15 * s), hc, fur), 3)
    add(box((0.09 * s, 0.07 * s, 0.11 * s), hc + [0, -0.03 * s, 0.11 * s], shade(fur, 0.9)), 3)
    add(box((0.04 * s, 0.03 * s, 0.03 * s), hc + [0, -0.005 * s, 0.17 * s], rgb("#151515")), 3)
    for side in (-1, 1):
        add(box((0.02 * s, 0.02 * s, 0.01 * s), hc + [side * 0.04 * s, 0.025 * s, 0.075 * s], rgb("#151515")), 3)
        add(tapered_box((0.05 * s, 0.03 * s), (0.02 * s, 0.015 * s), hc[1] + 0.05 * s, hc[1] + 0.12 * s, dark,
                        center=(hc[0] + side * 0.05 * s, hc[2] - 0.02 * s)), 3)
    add(limb(J[4], J[4] + [0, 0.12 * s, -0.16 * s], 0.025 * s, 0.012 * s, 4, fur), 4)
    for base in (5, 8, 11, 14):
        add(limb(J[base] + [0, 0.05 * s, 0], J[base + 1], 0.045 * s, 0.035 * s, 5, fur), base)
        add(limb(J[base + 1], J[base + 2], 0.032 * s, 0.028 * s, 5, fur), base + 1)
        add(box((0.055 * s, 0.035 * s, 0.075 * s), J[base + 2] + [0, -0.0225 * s, 0.015 * s], dark), base + 2)
    return merge(*parts)


# ---------------------------------------------------------------- vehicles

CAR_COLORS = ["#b3202a", "#1f4e8c", "#e6e6e6", "#202225", "#8c9196", "#2e6b3f", "#d9a61c", "#5d2e5e",
              "#f0f0ea", "#3c3f44", "#7a1c1c", "#4a7fb5"]


def vehicle_skeleton_doc(kind: str, dims: dict, sid: str) -> dict:
    """Chassis root, a body joint (roll / pitch on the suspension) and four wheel joints."""
    hw, r, zf, zr = dims["track"] / 2, dims["wheel_r"], dims["axle_f"], dims["axle_r"]
    joints = [("chassis", -1, (0, 0, 0)), ("body", 0, (0, dims["body_pivot"], 0)),
              ("wheel_fl", 0, (hw, r, zf)), ("wheel_fr", 0, (-hw, r, zf)),
              ("wheel_rl", 0, (hw, r, zr)), ("wheel_rr", 0, (-hw, r, zr))]
    if kind == "bike":
        joints = [("frame", -1, (0, 0, 0)), ("body", 0, (0, dims["body_pivot"], 0)),
                  ("wheel_f", 0, (0, r, zf)), ("wheel_r", 0, (0, r, zr)), ("crank", 0, (0, 0.3, 0.0))]
    return {"id": sid, "joints": [{"name": n, "parent": p, "offset": [float(x) for x in o]} for n, p, o in joints]}


def _wheel(r, w, color_tire, color_hub, n=10):
    tire = transform(cylinder(r, w, n, color_tire, y0=-w / 2), q=quat_axis_angle([0, 0, 1], np.pi / 2))
    hub = transform(cylinder(r * 0.55, w * 1.04, n, color_hub, y0=-w * 0.52), q=quat_axis_angle([0, 0, 1], np.pi / 2))
    spoke = box((w * 1.08, r * 0.18, r * 1.3), (0, 0, 0), shade(color_hub, 0.7))
    return merge(tire, hub, spoke)


def car(rng=None, kind: str = "sedan") -> tuple[dict, dict]:
    """Low-poly car (sedan / hatch / suv) facing +Z. Returns (mesh, dims) for the skeleton."""
    rng = np.random.default_rng(0) if rng is None else rng
    paint = rgb(CAR_COLORS[rng.integers(len(CAR_COLORS))])
    L, W, Hb = {"sedan": (4.5, 1.8, 0.62), "hatch": (4.0, 1.75, 0.64), "suv": (4.7, 1.9, 0.8)}[kind]
    r = 0.33 if kind != "suv" else 0.38
    clear = 0.18 if kind != "suv" else 0.24
    dims = {"length": L, "width": W, "wheel_r": r, "track": W - 0.22, "axle_f": L * 0.31,
            "axle_r": -L * 0.31, "body_pivot": 0.55, "height": 0.0}
    piv = dims["body_pivot"]
    glass = rgb("#1b2633")
    body = []
    y0, y1 = clear, clear + Hb
    body.append(tapered_box((W, L), (W * 0.98, L * 0.97), y0, y1, paint))
    # hood slope / bumpers
    body.append(box((W * 0.96, 0.16, 0.12), (0, y0 + 0.12, L / 2 + 0.03), shade(paint, 0.35)))
    body.append(box((W * 0.96, 0.16, 0.12), (0, y0 + 0.12, -L / 2 - 0.03), shade(paint, 0.35)))
    cab_len = {"sedan": 0.45, "hatch": 0.55, "suv": 0.62}[kind]
    cab_shift = {"sedan": -0.06, "hatch": -0.12, "suv": -0.1}[kind]
    ch = {"sedan": 0.52, "hatch": 0.55, "suv": 0.62}[kind]
    cz = cab_shift * L
    cab_bot = (W * 0.92, L * cab_len)
    cab_top = (W * 0.76, L * cab_len * (0.62 if kind == "sedan" else 0.75))
    body.append(tapered_box(cab_bot, cab_top, y1, y1 + ch, glass, center=(0, cz),
                            top_shift=(0, -0.05 * L if kind != "sedan" else 0)))
    # roof and pillars in paint colour over the glass
    roof_z = cz + (-0.05 * L if kind != "sedan" else 0)
    body.append(box((cab_top[0] + 0.02, 0.05, cab_top[1] + 0.02), (0, y1 + ch + 0.02, roof_z), paint))
    body.append(box((W * 0.93, 0.06, L * cab_len * 0.98), (0, y1 + 0.03, cz), paint))  # belt line
    # lights
    for side in (-1, 1):
        body.append(emissive(box((0.32, 0.1, 0.04), (side * (W / 2 - 0.25), y1 - 0.12, L / 2 + 0.005)),
                             rgb("#fff4d6")))
        body.append(emissive(box((0.3, 0.09, 0.04), (side * (W / 2 - 0.22), y1 - 0.1, -L / 2 - 0.005)),
                             rgb("#c41414")))
        body.append(box((0.06, 0.07, 0.14), (side * (W * 0.46 + 0.06), y1 + 0.12, cz + cab_bot[1] / 2 - 0.05),
                        paint))
    body.append(box((W * 0.5, 0.12, 0.03), (0, y0 + 0.28, L / 2 + 0.005), rgb("#1a1a1a")))   # grille
    body.append(box((0.5, 0.11, 0.02), (0, y0 + 0.3, -L / 2 - 0.02), rgb("#e8e8e0")))          # plate
    lights = [b for b in body if b.get("material", "m_matte") != "m_matte"]
    plain = [b for b in body if b.get("material", "m_matte") == "m_matte"]
    parts = [transform(merge(*plain), (0, -piv, 0))]
    glow = [transform(b, (0, -piv, 0)) for b in lights]
    mesh = merge(*[skin(p, 1) for p in parts])
    wheel = _wheel(r, 0.22, rgb("#161616"), rgb("#9a9ea3"))
    wheels = []
    for j, (x, z) in enumerate([(1, dims["axle_f"]), (-1, dims["axle_f"]), (1, dims["axle_r"]), (-1, dims["axle_r"])]):
        wheels.append(skin(wheel, 2 + j))
    mesh = merge(mesh, *wheels)
    # emissive light meshes become separate sub-meshes (different material); store under "extras"
    mesh["extras"] = [skin(g, 1) for g in glow]
    mesh["colliders"] = [[[-W / 2, 0, -L / 2], [W / 2, y1 + ch + 0.05, L / 2]]]
    dims["height"] = y1 + ch + 0.05
    dims["dash"] = [0.0, y1 + ch - 0.13, cz + cab_bot[1] / 2 - 0.3]     # behind the windscreen, mirror height
    # vertices are in joint-local space for wheels; convert to rest (mesh) space
    _bake_rest(mesh, dims, piv)
    return mesh, dims


def _bake_rest(mesh, dims, piv):
    """Wheel / body parts were built around their joint; move them to the rest pose of the vehicle skeleton."""
    hw, r = dims["track"] / 2, dims["wheel_r"]
    off = {1: np.array([0, piv, 0]), 2: np.array([hw, r, dims["axle_f"]]), 3: np.array([-hw, r, dims["axle_f"]]),
           4: np.array([hw, r, dims["axle_r"]]), 5: np.array([-hw, r, dims["axle_r"]])}
    for m in [mesh] + mesh.get("extras", []):
        j = m["joints"][:, 0]
        p = m["positions"].astype(np.float64)
        for k, o in off.items():
            p[j == k] += o
        m["positions"] = p.astype(np.float32)


def truck(rng=None) -> tuple[dict, dict]:
    """Box truck facing +Z with the same vehicle skeleton as cars (dual rear wheels share joints)."""
    rng = np.random.default_rng(0) if rng is None else rng
    paint = rgb(["#e6e6e6", "#1f4e8c", "#b3202a", "#d9a61c", "#2e6b3f"][rng.integers(5)])
    box_c = rgb(["#f2f2ee", "#d8d8d0", "#c9c2b0"][rng.integers(3)])
    L, W, r = 7.2, 2.3, 0.48
    dims = {"length": L, "width": W, "wheel_r": r, "track": W - 0.3, "axle_f": L / 2 - 1.2, "axle_r": -L / 2 + 1.6,
            "body_pivot": 1.0, "height": 3.4}
    piv = dims["body_pivot"]
    glass = rgb("#1b2633")
    parts = [box((W, 0.3, L - 0.2), (0, 0.55, 0), rgb("#26282b")),
             box((W, 1.6, 1.9), (0, 1.5, L / 2 - 0.95), paint),
             tapered_box((W * 0.98, 1.6), (W * 0.94, 1.2), 2.3, 2.95, paint, center=(0, L / 2 - 1.05),
                         top_shift=(0, -0.2)),
             box((W * 0.9, 0.55, 0.05), (0, 2.55, L / 2 - 0.33), glass),
             box((W + 0.04, 2.6, L - 2.1), (0, 2.0, -1.0), box_c),
             box((W * 0.9, 0.2, 0.1), (0, 0.9, L / 2 + 0.02), rgb("#303030"))]
    for side in (-1, 1):
        parts.append(box((0.05, 0.5, 0.9), (side * W / 2, 2.3, L / 2 - 0.9), glass))
    lights = []
    for side in (-1, 1):
        lights.append(emissive(box((0.3, 0.15, 0.04), (side * (W / 2 - 0.3), 1.0, L / 2 + 0.01)), rgb("#fff4d6")))
        lights.append(emissive(box((0.2, 0.25, 0.04), (side * (W / 2 - 0.2), 0.95, -L / 2 + 0.03)), rgb("#c41414")))
    mesh = skin(transform(merge(*parts), (0, -piv, 0)), 1)
    wheel = _wheel(r, 0.3, rgb("#161616"), rgb("#8a8e93"))
    ws = []
    for j, (x, z) in enumerate([(1, dims["axle_f"]), (-1, dims["axle_f"]), (1, dims["axle_r"]), (-1, dims["axle_r"])]):
        ws.append(skin(wheel, 2 + j))
        if j >= 2:
            ws.append(skin(transform(wheel, (0, 0, 1.1)), 2 + j))
    mesh = merge(mesh, *ws)
    mesh["extras"] = [skin(transform(g, (0, -piv, 0)), 1) for g in lights]
    mesh["colliders"] = [[[-W / 2, 0, -L / 2], [W / 2, 3.4, L / 2]]]
    _bake_rest(mesh, dims, piv)
    return mesh, dims


def bike(rng=None) -> tuple[dict, dict]:
    """Bicycle facing +Z: frame, two wheels and a crank, each rigidly skinned to the bike skeleton."""
    rng = np.random.default_rng(0) if rng is None else rng
    frame_c = rgb(["#c0392b", "#2f4b7c", "#1d1f24", "#e8e6e1", "#2a9d8f"][rng.integers(5)])
    r = 0.34
    dims = {"length": 1.75, "width": 0.6, "wheel_r": r, "track": 0, "axle_f": 0.52, "axle_r": -0.52,
            "body_pivot": 0.0, "height": 1.1}
    tire, rim = rgb("#151515"), rgb("#b0b4b8")

    def ring(rad, thick, n=14):
        parts = []
        for i in range(n):
            a0, a1 = 2 * np.pi * i / n, 2 * np.pi * (i + 1) / n
            p0 = np.array([0, rad * np.cos(a0), rad * np.sin(a0)])
            p1 = np.array([0, rad * np.cos(a1), rad * np.sin(a1)])
            parts.append(limb(p0, p1, thick, thick, 4, tire))
        return merge(*parts)
    wheel = merge(ring(r - 0.02, 0.022), limb([-0.03, 0, 0], [0.03, 0, 0], 0.03, 0.03, 6, rim),
                  *[limb([0, 0, 0], [0, (r - 0.03) * np.cos(a), (r - 0.03) * np.sin(a)], 0.004, 0.004, 3, rim)
                    for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)])
    bb = np.array([0, 0.3, 0.0])       # bottom bracket
    seat = np.array([0, 0.82, -0.18])
    head = np.array([0, 0.8, 0.38])
    rear, front = np.array([0, r, -0.52]), np.array([0, r, 0.52])
    tube = 0.018
    frame = merge(limb(bb, seat, tube, tube, 5, frame_c), limb(bb, head, tube, tube, 5, frame_c),
                  limb(seat + [0, -0.06, 0.02], head + [0, -0.04, 0], tube, tube, 5, frame_c),
                  limb(bb, rear, tube * 0.8, tube * 0.8, 5, frame_c),
                  limb(seat + [0, -0.06, 0.02], rear, tube * 0.7, tube * 0.7, 5, frame_c),
                  limb(head, front, tube, tube * 0.8, 5, frame_c),
                  limb(head, head + [0, 0.17, -0.05], tube, tube, 5, rgb("#2a2a2a")),
                  limb(head + [0.28, 0.17, -0.08], head + [-0.28, 0.17, -0.08], 0.014, 0.014, 5, rgb("#2a2a2a")),
                  limb(seat + [0, -0.05, 0], seat + [0, 0.06, 0], 0.014, 0.014, 4, rgb("#2a2a2a")),
                  box((0.12, 0.05, 0.24), seat + [0, 0.08, 0.0], rgb("#1a1a1a")))
    crank = merge(limb([0.07, 0, 0], [0.07, -0.17, 0], 0.012, 0.012, 4, rim),
                  limb([-0.07, 0, 0], [-0.07, 0.17, 0], 0.012, 0.012, 4, rim),
                  box((0.09, 0.025, 0.1), [0.12, -0.17, 0], rgb("#202020")),
                  box((0.09, 0.025, 0.1), [-0.12, 0.17, 0], rgb("#202020")),
                  limb([-0.08, 0, 0], [0.08, 0, 0], 0.02, 0.02, 6, rim))
    mesh = merge(skin(frame, 1), skin(transform(wheel, front), 2), skin(transform(wheel, rear), 3),
                 skin(transform(crank, bb), 4))
    mesh["colliders"] = [[[-0.3, 0, -0.9], [0.3, 1.05, 0.9]]]
    dims["crank_pos"] = bb.tolist()
    dims["seat"] = (seat + [0, 0.1, 0]).tolist()
    dims["handlebar"] = (head + [0, 0.17, -0.08]).tolist()
    return mesh, dims


# ---------------------------------------------------------------- furniture and props

WOOD = ["#8b5e3c", "#a0703f", "#6b4a2e", "#c49a6c", "#5a3b24", "#d2b48c"]
FABRIC = ["#4a5d7e", "#7b8c6e", "#a35d4a", "#d8c3a5", "#3e3e46", "#6f4e7c", "#b8a07e", "#2f5d62", "#c97c5d",
          "#8e9aaf", "#e0b1a0", "#5b6c5d"]


def _pick(rng, pal):
    return rgb(pal[rng.integers(len(pal))])


def table(rng=None, w=1.4, d=0.8, h=0.75, round_top=False, color=None) -> dict:
    rng = np.random.default_rng(0) if rng is None else rng
    c = _pick(rng, WOOD) if color is None else color
    t = 0.04
    if round_top:
        top = cylinder(w / 2, t, 12, c, y0=h - t)
        legs = [cylinder(0.04, h - t, 6, shade(c, 0.8)), cylinder(0.25, 0.03, 8, shade(c, 0.8))]
    else:
        top = box((w, t, d), (0, h - t / 2, 0), c)
        legs = [box((0.05, h - t, 0.05), (sx * (w / 2 - 0.06), (h - t) / 2, sz * (d / 2 - 0.06)), shade(c, 0.85))
                for sx in (-1, 1) for sz in (-1, 1)]
        legs.append(box((w - 0.16, 0.06, 0.02), (0, h - t - 0.03, d / 2 - 0.06), shade(c, 0.85)))
    m = merge(top, *legs)
    return with_colliders(m, [[[-w / 2, h - t - 0.06, -d / 2], [w / 2, h, d / 2]],
                              [[-w / 2 + 0.02, 0, -d / 2 + 0.02], [w / 2 - 0.02, h, d / 2 - 0.02]]])


def chair(rng=None, color=None, seat_h=0.46, cushion=None) -> dict:
    """Dining chair facing +Z (sitter faces +Z, backrest at -Z)."""
    rng = np.random.default_rng(0) if rng is None else rng
    c = _pick(rng, WOOD) if color is None else color
    w, d = 0.46, 0.46
    parts = [box((w, 0.04, d), (0, seat_h - 0.02, 0), c)]
    if cushion is not None:
        parts.append(box((w - 0.04, 0.04, d - 0.04), (0, seat_h + 0.02, 0.01), cushion))
    for sx in (-1, 1):
        for sz in (-1, 1):
            parts.append(box((0.04, seat_h - 0.04, 0.04), (sx * (w / 2 - 0.03), (seat_h - 0.04) / 2,
                                                            sz * (d / 2 - 0.03)), shade(c, 0.85)))
        parts.append(box((0.04, 0.46, 0.04), (sx * (w / 2 - 0.03), seat_h + 0.23, -d / 2 + 0.03), shade(c, 0.85)))
    parts.append(box((w, 0.16, 0.03), (0, seat_h + 0.38, -d / 2 + 0.03), c))
    parts.append(box((w - 0.1, 0.05, 0.02), (0, seat_h + 0.2, -d / 2 + 0.03), c))
    m = merge(*parts)
    m["seat"] = [0.0, seat_h + (0.04 if cushion is not None else 0), 0.02]
    return with_colliders(m, [[[-w / 2, 0, -d / 2], [w / 2, seat_h, d / 2]],
                              [[-w / 2, seat_h, -d / 2], [w / 2, seat_h + 0.48, -d / 2 + 0.06]]])


def office_chair(rng=None, color=None) -> dict:
    rng = np.random.default_rng(0) if rng is None else rng
    c = rgb("#2a2c30") if color is None else color
    metal = rgb("#8a8e93")
    parts = [box((0.5, 0.08, 0.48), (0, 0.48, 0.02), c),
             tapered_box((0.46, 0.07), (0.42, 0.06), 0.58, 1.08, c, center=(0, -0.22), top_shift=(0, -0.04)),
             cylinder(0.03, 0.38, 6, metal, y0=0.07), box((0.1, 0.05, 0.12), (0, 0.44, 0), metal)]
    for a in np.linspace(0, 2 * np.pi, 5, endpoint=False):
        tip = np.array([np.cos(a) * 0.3, 0.06, np.sin(a) * 0.3])
        parts.append(limb([0, 0.08, 0], tip, 0.02, 0.016, 4, metal))
        parts.append(icosphere(0.03, 0, rgb("#111111"), tip - [0, 0.03, 0]))
    for side in (-1, 1):
        parts.append(box((0.05, 0.03, 0.26), (side * 0.27, 0.68, 0.02), c))
        parts.append(box((0.03, 0.18, 0.03), (side * 0.27, 0.58, -0.05), metal))
    m = merge(*parts)
    m["seat"] = [0.0, 0.52, 0.04]
    return with_colliders(m, [[[-0.3, 0, -0.3], [0.3, 0.52, 0.3]], [[-0.25, 0.52, -0.3], [0.25, 1.1, -0.15]]])


def couch(rng=None, seats=3, color=None) -> dict:
    """Sofa facing +Z. Seat cushions top at ~0.45 m."""
    rng = np.random.default_rng(0) if rng is None else rng
    c = _pick(rng, FABRIC) if color is None else color
    w = 0.62 * seats + 0.36
    d = 0.92
    parts = [box((w, 0.24, d), (0, 0.2, 0), shade(c, 0.9)),
             box((w, 0.5, 0.22), (0, 0.55, -d / 2 + 0.11), c)]
    for side in (-1, 1):
        parts.append(box((0.18, 0.36, d), (side * (w / 2 - 0.09), 0.44, 0), shade(c, 0.95)))
    cw = (w - 0.36) / seats
    for i in range(seats):
        x = -w / 2 + 0.18 + cw * (i + 0.5)
        parts.append(box((cw - 0.02, 0.13, d - 0.26), (x, 0.385, 0.1), shade(c, 1.05)))
        parts.append(box((cw - 0.04, 0.36, 0.14), (x, 0.62, -d / 2 + 0.29), shade(c, 1.02)))
    for sx in (-1, 1):
        for sz in (-1, 1):
            parts.append(box((0.05, 0.08, 0.05), (sx * (w / 2 - 0.06), 0.04, sz * (d / 2 - 0.06)), rgb("#2b2016")))
    m = merge(*parts)
    m["seat"] = [0.0, 0.45, 0.12]
    m["seats"] = [[-w / 2 + 0.18 + cw * (i + 0.5), 0.45, 0.12] for i in range(seats)]
    return with_colliders(m, [[[-w / 2, 0, -d / 2], [w / 2, 0.45, d / 2]], [[-w / 2, 0.45, -d / 2], [w / 2, 0.8, -d / 2 + 0.32]],
                              [[-w / 2, 0.45, -d / 2], [-w / 2 + 0.18, 0.62, d / 2]], [[w / 2 - 0.18, 0.45, -d / 2], [w / 2, 0.62, d / 2]]])


def armchair(rng=None, color=None) -> dict:
    m = couch(rng, 1, color)
    return m


def bed(rng=None, w=1.6, length=2.05, color=None) -> dict:
    """Bed with headboard at -Z."""
    rng = np.random.default_rng(0) if rng is None else rng
    frame_c = _pick(rng, WOOD)
    sheet = rgb(["#f1efe9", "#dfe7ef", "#e9e1d3"][rng.integers(3)])
    blanket = _pick(rng, FABRIC) if color is None else color
    parts = [box((w + 0.08, 0.28, length + 0.06), (0, 0.18, 0), frame_c),
             box((w, 0.22, length), (0, 0.43, 0), sheet),
             box((w + 0.06, 0.06, length * 0.62), (0, 0.55, length * 0.17), blanket),
             box((w + 0.08, 0.95, 0.07), (0, 0.5, -length / 2 - 0.03), shade(frame_c, 0.9))]
    for side in (-1, 1):
        parts.append(box((w * 0.42, 0.12, 0.38), (side * w * 0.24, 0.6, -length / 2 + 0.26), sheet))
    m = merge(*parts)
    return with_colliders(m, [[[-w / 2 - 0.05, 0, -length / 2 - 0.07], [w / 2 + 0.05, 0.6, length / 2 + 0.03]],
                              [[-w / 2 - 0.05, 0, -length / 2 - 0.07], [w / 2 + 0.05, 0.98, -length / 2]]])


def shelf(rng=None, w=0.9, h=1.8, d=0.32, books=True) -> dict:
    rng = np.random.default_rng(0) if rng is None else rng
    c = _pick(rng, WOOD)
    parts = [box((w, h, 0.02), (0, h / 2, -d / 2 + 0.01), shade(c, 0.8))]
    for side in (-1, 1):
        parts.append(box((0.025, h, d), (side * (w / 2 - 0.0125), h / 2, 0), c))
    n = max(3, int(h / 0.38))
    for i in range(n + 1):
        y = 0.04 + i * (h - 0.06) / n
        parts.append(box((w, 0.025, d), (0, y, 0), c))
        if books and i < n:
            x = -w / 2 + 0.04
            while x < w / 2 - 0.08:
                if rng.random() < 0.2:
                    x += rng.uniform(0.04, 0.12)
                    continue
                bw = rng.uniform(0.025, 0.05)
                bh = rng.uniform(0.18, 0.3)
                bc = jitter_color(_pick(rng, CLOTH_COLORS), rng, 0.15)
                parts.append(box((bw, bh, d * rng.uniform(0.6, 0.85)), (x + bw / 2, y + 0.0125 + bh / 2, 0.01), bc))
                x += bw + 0.003
    return with_colliders(merge(*parts))


def lamp(rng=None, kind="floor") -> dict:
    rng = np.random.default_rng(0) if rng is None else rng
    metal = rgb(["#2a2a2a", "#b8a27a", "#d0d0d0"][rng.integers(3)])
    shade_c = rgb(["#f3e7cf", "#e9e4da", "#d9c6a5"][rng.integers(3)])
    if kind == "floor":
        parts = [cylinder(0.16, 0.03, 10, metal), cylinder(0.015, 1.45, 6, metal, y0=0.03),
                 cylinder(0.22, 0.28, 10, shade_c, r_top=0.15, y0=1.38)]
        glow = emissive(cylinder(0.12, 0.02, 8, y0=1.37), rgb("#fff1c8"))
    else:
        parts = [cylinder(0.08, 0.02, 8, metal), cylinder(0.01, 0.3, 5, metal, y0=0.02),
                 cylinder(0.13, 0.16, 8, shade_c, r_top=0.09, y0=0.28)]
        glow = emissive(cylinder(0.07, 0.015, 8, y0=0.27), rgb("#fff1c8"))
    m = merge(*parts)
    m["extras"] = [glow]
    top = 1.66 if kind == "floor" else 0.44
    return with_colliders(m, [[[-0.2, 0, -0.2], [0.2, top, 0.2]]] if kind != "floor" else
                          [[[-0.06, 0, -0.06], [0.06, 1.4, 0.06]], [[-0.22, 1.38, -0.22], [0.22, 1.66, 0.22]]])


def plant(rng=None, h=None, pot_color=None) -> dict:
    """Potted plant: tapered pot with soil and low-poly leaf clusters."""
    rng = np.random.default_rng(0) if rng is None else rng
    h = rng.uniform(0.4, 1.4) if h is None else h
    pot_c = rgb(["#b5651d", "#e8e4dc", "#2f3e46", "#c9b79c", "#7a4b35"][rng.integers(5)]) if pot_color is None else pot_color
    leaf = rgb(["#3c7a3a", "#4f8f3a", "#2f6b46", "#5e9a4b"][rng.integers(4)])
    pr = 0.1 + 0.08 * h
    ph = pr * 1.3
    parts = [cylinder(pr * 0.75, ph, 8, pot_c, r_top=pr), cylinder(pr * 0.92, 0.02, 8, rgb("#3a2a1c"), y0=ph - 0.03)]
    kind = rng.integers(3)
    if kind == 0:     # bushy
        for _ in range(4 + int(h * 4)):
            c = np.array([rng.uniform(-1, 1) * pr, ph + rng.uniform(0.1, 0.9) * (h - ph), rng.uniform(-1, 1) * pr])
            parts.append(icosphere(rng.uniform(0.12, 0.2) * (0.5 + h / 2), 0, jitter_color(leaf, rng, 0.15), c, rng=rng,
                                   jitter=0.2))
    elif kind == 1:   # leaves on stems
        for i in range(7 + int(h * 4)):
            a = rng.uniform(0, 2 * np.pi)
            top = np.array([np.cos(a) * rng.uniform(0.1, 0.35) * h, ph + rng.uniform(0.4, 1.0) * (h - ph),
                            np.sin(a) * rng.uniform(0.1, 0.35) * h])
            parts.append(limb([0, ph, 0], top, 0.008, 0.006, 3, shade(leaf, 0.7)))
            parts.append(transform(icosphere(0.09 * (0.6 + h / 2), 0, jitter_color(leaf, rng, 0.15), scale=(1, 0.3, 0.55)),
                                   top, yaw_q(-a)))
    else:             # tall leaves (snake-plant like)
        for i in range(6 + int(h * 3)):
            a = rng.uniform(0, 2 * np.pi)
            r0 = rng.uniform(0, pr * 0.6)
            base = np.array([np.cos(a) * r0, ph - 0.02, np.sin(a) * r0])
            top = base + [np.cos(a) * 0.08, h - ph, np.sin(a) * 0.08] * np.array([1, rng.uniform(0.7, 1), 1])
            parts.append(limb(base, top, 0.035, 0.004, 4, jitter_color(leaf, rng, 0.2), rz_scale=0.3))
    return with_colliders(merge(*parts), [[[-pr, 0, -pr], [pr, h, pr]]])


def mug(rng=None, color=None) -> dict:
    rng = np.random.default_rng(0) if rng is None else rng
    c = _pick(rng, ["#f2f2f2", "#c0392b", "#2f4b7c", "#e09f3e", "#2a9d8f", "#26262b"]) if color is None else color
    body = cylinder(0.042, 0.1, 10, c)
    coffee = cylinder(0.036, 0.003, 10, rgb("#3b2416"), y0=0.092)
    handle = merge(limb([0.04, 0.075, 0], [0.068, 0.07, 0], 0.008, 0.008, 4, c),
                   limb([0.068, 0.07, 0], [0.068, 0.03, 0], 0.008, 0.008, 4, c),
                   limb([0.068, 0.03, 0], [0.04, 0.025, 0], 0.008, 0.008, 4, c))
    return merge(body, coffee, handle)


def ball(rng=None, r=0.11, color=None, kind="soccer") -> dict:
    rng = np.random.default_rng(0) if rng is None else rng
    if kind == "soccer":
        m = icosphere(r, 1, rgb("#f4f4f4"))
        m = facet(m)
        f = m["positions"].reshape(-1, 3, 3).mean(1)
        v, _ = sphere_points(0)
        dark = (np.abs((f / np.linalg.norm(f, axis=1, keepdims=True)) @ v.T) > 0.97).any(1)
        m["colors"].reshape(-1, 3, 3)[dark] = rgb("#1a1a1a")
        return m
    c = _pick(rng, ["#d35400", "#c0392b", "#f1c40f", "#3498db"]) if color is None else color
    return facet(icosphere(r, 1, c), rng, 0.05)


def laptop(rng=None, open_angle=1.85) -> dict:
    """Laptop with the hinge at z = -0.11, screen facing +Z."""
    rng = np.random.default_rng(0) if rng is None else rng
    shell = rgb(["#c8ccd0", "#2a2c30", "#8a8e93"][rng.integers(3)])
    base = merge(box((0.33, 0.018, 0.23), (0, 0.009, 0), shell),
                 box((0.29, 0.002, 0.11), (0, 0.019, -0.03), rgb("#1e1f22")),
                 box((0.1, 0.002, 0.06), (0, 0.019, 0.07), shade(shell, 0.9)))
    lid = merge(box((0.33, 0.22, 0.008), (0, 0.11, 0), shell))
    screen = emissive(box((0.3, 0.19, 0.002), (0, 0.11, 0.005)), rgb("#7fa7d9"))
    q = quat_axis_angle([1, 0, 0], -(open_angle - np.pi / 2))
    lid = transform(lid, (0, 0.018, -0.115), q)
    screen = transform(screen, (0, 0.018, -0.115), q)
    m = merge(base, lid)
    m["extras"] = [screen]
    return m


def door(w=0.9, h=2.05, color=None) -> dict:
    """Door in the XY plane facing +Z (frame + panel + handle), thin enough to sit on a wall."""
    c = rgb("#e9e4da") if color is None else color
    frame_c = rgb("#f4f1ea")
    parts = [box((w, h, 0.04), (0, h / 2, 0.02), c)]
    for x in (-w / 2 - 0.04, w / 2 + 0.04):
        parts.append(box((0.08, h + 0.08, 0.06), (x, (h + 0.08) / 2, 0.03), frame_c))
    parts.append(box((w + 0.16, 0.08, 0.06), (0, h + 0.04, 0.03), frame_c))
    for y in (0.5, 1.4):
        parts.append(box((w * 0.7, 0.5, 0.01), (0, y, 0.045), shade(c, 0.95)))
    parts.append(icosphere(0.03, 0, rgb("#b8a27a"), (w / 2 - 0.1, 1.0, 0.07)))
    return merge(*parts)


def window(w=1.2, h=1.3, sill=0.9, outside=None, frame=None) -> dict:
    """Wall window facing +Z: frame, mullions and a glowing 'outside' pane (sky / light)."""
    frame_c = rgb("#f4f1ea") if frame is None else frame
    out_c = rgb("#cfe3f2") if outside is None else outside
    parts = []
    t = 0.06
    for x in (-w / 2, w / 2):
        parts.append(box((t, h + t, 0.08), (x, sill + h / 2, 0.04), frame_c))
    for y in (sill, sill + h):
        parts.append(box((w + t, t, 0.08), (0, y, 0.04), frame_c))
    parts.append(box((0.03, h, 0.05), (0, sill + h / 2, 0.035), frame_c))
    parts.append(box((w, 0.03, 0.05), (0, sill + h * 0.62, 0.035), frame_c))
    parts.append(box((w + 0.16, 0.04, 0.16), (0, sill - 0.02, 0.08), frame_c))
    m = merge(*parts)
    m["extras"] = [emissive(box((w, h, 0.01), (0, sill + h / 2, 0.012)), out_c)]
    return m


def picture_frame(rng=None, w=0.6, h=0.45) -> dict:
    """Wall picture facing +Z with an original abstract composition of colour blocks."""
    rng = np.random.default_rng(0) if rng is None else rng
    fc = rgb(["#2a2a2a", "#c49a6c", "#f4f1ea", "#8b5e3c"][rng.integers(4)])
    parts = [box((w, h, 0.03), (0, 0, 0.015), fc), box((w - 0.06, h - 0.06, 0.005), (0, 0, 0.032),
                                                        rgb("#f2efe6"))]
    pal = [rgb(x) for x in rng.choice(CLOTH_COLORS, 4, replace=False)]
    for _ in range(rng.integers(3, 7)):
        bw, bh = rng.uniform(0.08, 0.4) * w, rng.uniform(0.08, 0.5) * h
        cx, cy = rng.uniform(-1, 1) * (w - 0.1 - bw) / 2, rng.uniform(-1, 1) * (h - 0.1 - bh) / 2
        parts.append(box((bw, bh, 0.004), (cx, cy, 0.036 + rng.uniform(0, 0.002)), pal[rng.integers(4)]))
    return merge(*parts)


def microphone(kind="desk", color=None) -> dict:
    """Podcast microphone: 'desk' (small stand) or 'arm' (boom arm clamped at the table edge, mic at +Z)."""
    c = rgb("#1d1d1f") if color is None else color
    metal = rgb("#6d7075")
    capsule = merge(limb([0, 0, 0], [0, 0.16, 0], 0.03, 0.028, 8, c, cap=0.6),
                    limb([0, 0.03, 0], [0, 0.13, 0], 0.032, 0.032, 8, rgb("#3a3b3f")))
    if kind == "desk":     # short table stand, capsule tilted toward the speaker (+Z)
        parts = [cylinder(0.07, 0.015, 10, metal), cylinder(0.008, 0.09, 5, metal, y0=0.015),
                 transform(capsule, (0, 0.1, 0.02), quat_axis_angle([1, 0, 0], 0.9))]
        return merge(*parts)
    parts = [box((0.05, 0.08, 0.05), (0, 0.0, 0), metal), limb([0, 0.03, 0], [0, 0.3, 0.15], 0.01, 0.01, 5, metal),
             limb([0, 0.3, 0.15], [0, 0.28, 0.42], 0.01, 0.01, 5, metal),
             transform(capsule, (0, 0.2, 0.44), quat_axis_angle([1, 0, 0], 0.25))]
    return merge(*parts)


def counter(rng=None, w=2.4, d=0.62, h=0.9, top_color=None, cab_color=None, sink=False, stove=False) -> dict:
    """Kitchen base cabinets with a countertop, front at +Z."""
    rng = np.random.default_rng(0) if rng is None else rng
    cab = _pick(rng, ["#f4f1ea", "#3d5a6c", "#7b8c6e", "#c49a6c", "#2f3e46", "#d8cbb5"]) if cab_color is None else cab_color
    top = _pick(rng, ["#e8e6e1", "#3a3a3a", "#c9b79c", "#8a8e93"]) if top_color is None else top_color
    parts = [box((w, h - 0.14, d - 0.04), (0, 0.1 + (h - 0.14) / 2, -0.02), cab),
             box((w, 0.1, d - 0.1), (0, 0.05, -0.05), shade(cab, 0.5)),
             box((w + 0.02, 0.04, d + 0.02), (0, h - 0.02, 0), top)]
    n = max(1, int(round(w / 0.6)))
    for i in range(n):
        x = -w / 2 + w / n * (i + 0.5)
        parts.append(box((w / n - 0.02, h - 0.2, 0.01), (x, 0.12 + (h - 0.2) / 2, d / 2 - 0.035), shade(cab, 1.04)))
        parts.append(box((0.12, 0.015, 0.02), (x, h - 0.16, d / 2 - 0.02), rgb("#9a9ea3")))
    if sink:
        parts.append(box((0.5, 0.01, 0.38), (w * 0.2, h + 0.001, 0), rgb("#9fa5ab")))
        parts.append(limb([w * 0.2, h, -0.24], [w * 0.2, h + 0.3, -0.24], 0.015, 0.015, 5, rgb("#c8ccd0")))
        parts.append(limb([w * 0.2, h + 0.3, -0.24], [w * 0.2, h + 0.3, -0.08], 0.012, 0.012, 5, rgb("#c8ccd0")))
    if stove:
        parts.append(box((0.6, 0.01, 0.52), (-w * 0.25, h + 0.001, 0), rgb("#1c1c1c")))
        for dx in (-0.14, 0.14):
            for dz in (-0.12, 0.12):
                parts.append(transform(cylinder(0.08, 0.006, 8, rgb("#3a3a3a")), (-w * 0.25 + dx, h + 0.005, dz)))
    return with_colliders(merge(*parts), [[[-w / 2, 0, -d / 2], [w / 2, h, d / 2]]])


def upper_cabinets(rng=None, w=2.4, d=0.35, h=0.7, color=None) -> dict:
    rng = np.random.default_rng(0) if rng is None else rng
    c = _pick(rng, ["#f4f1ea", "#3d5a6c", "#7b8c6e", "#c49a6c"]) if color is None else color
    parts = [box((w, h, d), (0, h / 2, 0), c)]
    n = max(1, int(round(w / 0.5)))
    for i in range(n):
        x = -w / 2 + w / n * (i + 0.5)
        parts.append(box((w / n - 0.02, h - 0.04, 0.01), (x, h / 2, d / 2 + 0.005), shade(c, 1.04)))
        parts.append(box((0.015, 0.1, 0.02), (x + (w / n / 2 - 0.05) * (1 if i % 2 else -1), 0.12, d / 2 + 0.015),
                         rgb("#9a9ea3")))
    return merge(*parts)


def fridge(rng=None) -> dict:
    c = rgb(["#e8e8e6", "#c8ccd0", "#2a2c30"][(rng or np.random.default_rng(0)).integers(3)])
    parts = [box((0.75, 1.85, 0.7), (0, 0.925, 0), c), box((0.74, 0.01, 0.01), (0, 1.25, 0.355), shade(c, 0.6)),
             box((0.03, 0.4, 0.04), (0.3, 1.5, 0.37), rgb("#9a9ea3")), box((0.03, 0.4, 0.04), (0.3, 0.9, 0.37), rgb("#9a9ea3"))]
    return merge(*parts)


def tv(w=1.3) -> dict:
    h = w * 9 / 16
    m = merge(box((w, h, 0.05), (0, h / 2, 0), rgb("#111214")), box((0.3, 0.02, 0.2), (0, -0.01, 0), rgb("#222")))
    m["extras"] = [emissive(box((w - 0.04, h - 0.04, 0.005), (0, h / 2, 0.026)), rgb("#22344d"))]
    return m


def rug(rng=None, w=2.4, d=1.7) -> dict:
    rng = np.random.default_rng(0) if rng is None else rng
    a, b = _pick(rng, FABRIC), _pick(rng, FABRIC)
    parts = [box((w, 0.012, d), (0, 0.006, 0), a), box((w - 0.25, 0.014, d - 0.25), (0, 0.007, 0), b),
             box((w - 0.5, 0.016, d - 0.5), (0, 0.008, 0), a)]
    return merge(*parts)


def nightstand(rng=None) -> dict:
    rng = np.random.default_rng(0) if rng is None else rng
    c = _pick(rng, WOOD)
    return merge(box((0.45, 0.5, 0.4), (0, 0.25, 0), c), box((0.4, 0.18, 0.01), (0, 0.32, 0.2), shade(c, 1.1)),
                 box((0.08, 0.015, 0.02), (0, 0.32, 0.21), rgb("#9a9ea3")))


def wardrobe(rng=None, w=1.2) -> dict:
    rng = np.random.default_rng(0) if rng is None else rng
    c = _pick(rng, WOOD + ["#f4f1ea"])
    return merge(box((w, 2.0, 0.58), (0, 1.0, 0), c), box((0.01, 1.9, 0.01), (0, 1.0, 0.295), shade(c, 0.6)),
                 box((0.02, 0.2, 0.03), (-0.05, 1.05, 0.3), rgb("#9a9ea3")), box((0.02, 0.2, 0.03), (0.05, 1.05, 0.3),
                                                                                 rgb("#9a9ea3")))


def desk(rng=None, w=1.4, d=0.7) -> dict:
    rng = np.random.default_rng(0) if rng is None else rng
    c = _pick(rng, WOOD + ["#f4f1ea", "#2a2c30"])
    leg = rgb("#2a2c30")
    parts = [box((w, 0.035, d), (0, 0.735, 0), c)]
    for sx in (-1, 1):
        parts.append(box((0.05, 0.72, d - 0.06), (sx * (w / 2 - 0.05), 0.36, 0), leg))
    parts.append(box((w - 0.1, 0.3, 0.02), (0, 0.55, -d / 2 + 0.03), leg))
    return with_colliders(merge(*parts), [[[-w / 2, 0.62, -d / 2], [w / 2, 0.755, d / 2]],
                                          [[-w / 2, 0, -d / 2], [w / 2, 0.755, d / 2]]])


def monitor(w=0.6) -> dict:
    h = w * 0.58
    m = merge(box((0.22, 0.015, 0.18), (0, 0.0075, 0), rgb("#2a2c30")), box((0.05, 0.16, 0.03), (0, 0.09, -0.04),
                                                                            rgb("#2a2c30")),
              box((w, h, 0.03), (0, 0.17 + h / 2, -0.02), rgb("#151618")))
    m["extras"] = [emissive(box((w - 0.03, h - 0.03, 0.004), (0, 0.17 + h / 2, -0.004)), rgb("#9cc0e8"))]
    return m


def whiteboard(w=1.8, h=1.0) -> dict:
    parts = [box((w, h, 0.03), (0, 0, 0.015), rgb("#c8ccd0")), box((w - 0.05, h - 0.05, 0.005), (0, 0, 0.03),
                                                                    rgb("#fafafa"))]
    rng = np.random.default_rng(int(w * 100))
    for _ in range(5):
        x, y = rng.uniform(-0.7, 0.5) * w / 2, rng.uniform(-0.6, 0.6) * h / 2
        parts.append(box((rng.uniform(0.1, 0.5), 0.012, 0.002), (x, y, 0.034), rgb(["#2f4b7c", "#c0392b", "#222"][_ % 3])))
    return merge(*parts)


def acoustic_panel(w=0.6, h=0.6, color=None, rng=None) -> dict:
    rng = np.random.default_rng(0) if rng is None else rng
    c = _pick(rng, ["#3d405b", "#264653", "#6d597a", "#2f3e46", "#7a3b3b", "#3a3a3a"]) if color is None else color
    parts = []
    n = 4
    for i in range(n):
        for k in range(n):
            depth = 0.03 + 0.03 * ((i + k) % 2)
            parts.append(box((w / n - 0.004, h / n - 0.004, depth), (-w / 2 + w / n * (i + 0.5), -h / 2 + h / n * (k + 0.5),
                                                                     depth / 2), shade(c, 0.9 + 0.12 * ((i + k) % 2))))
    return merge(*parts)


def speaker(h=1.0) -> dict:
    w = h * 0.5
    parts = [box((w, h, w * 0.9), (0, h / 2, 0), rgb("#1a1a1c"))]
    for y, r in ((h * 0.7, w * 0.32), (h * 0.3, w * 0.38)):
        parts.append(transform(cylinder(r, 0.02, 10, rgb("#2e2f33")), (0, y, w * 0.45), quat_axis_angle([1, 0, 0], np.pi / 2)))
    return merge(*parts)


def stage_light(color=None) -> dict:
    """Spotlight can facing +Z with a glowing lens."""
    c = rgb("#ffd27a") if color is None else color
    body = transform(cylinder(0.12, 0.3, 8, rgb("#1a1a1c"), r_top=0.14, y0=-0.15), q=quat_axis_angle([1, 0, 0], np.pi / 2))
    m = merge(body, box((0.32, 0.03, 0.03), (0, 0.17, 0), rgb("#2a2a2a")))
    m["extras"] = [emissive(transform(cylinder(0.12, 0.01, 8, y0=0), (0, 0, 0.15), quat_axis_angle([1, 0, 0], np.pi / 2)), c)]
    return m


def truss(length=8.0, size=0.3) -> dict:
    """Square aluminium truss along X centred at the origin."""
    metal = rgb("#b8bcc2")
    parts = []
    s = size / 2
    corners = [(s, s), (s, -s), (-s, -s), (-s, s)]
    for y, z in corners:
        parts.append(limb([-length / 2, y, z], [length / 2, y, z], 0.02, 0.02, 4, metal))
    n = int(length / size)
    for i in range(n):
        x0, x1 = -length / 2 + i * size, -length / 2 + (i + 1) * size
        for (y0, z0), (y1, z1) in zip(corners, corners[1:] + corners[:1]):
            parts.append(limb([x0, y0, z0], [x1, y1, z1], 0.008, 0.008, 3, metal))
    return merge(*parts)


def mirror_wall(w=6.0, h=2.2) -> dict:
    parts = [box((w, h, 0.02), (0, 0, 0.01), rgb("#a9c3cf"))]
    for x in np.linspace(-w / 2, w / 2, int(w / 1.2) + 1):
        parts.append(box((0.03, h, 0.03), (x, 0, 0.02), rgb("#d0d0d0")))
    return merge(*parts)


def barre(length=4.0) -> dict:
    metal = rgb("#c8ccd0")
    parts = [limb([-length / 2, 1.0, 0], [length / 2, 1.0, 0], 0.02, 0.02, 6, rgb("#c49a6c"))]
    for x in np.linspace(-length / 2 + 0.2, length / 2 - 0.2, 3):
        parts.append(limb([x, 1.0, 0], [x, 1.0, -0.12], 0.012, 0.012, 4, metal))
    return merge(*parts)


def bench(rng=None, w=1.6) -> dict:
    rng = np.random.default_rng(0) if rng is None else rng
    wood = _pick(rng, WOOD)
    metal = rgb("#2a2c30")
    parts = []
    for i in range(3):
        parts.append(box((w, 0.03, 0.1), (0, 0.45, -0.15 + i * 0.12), wood))
    for i in range(2):
        parts.append(box((w, 0.1, 0.03), (0, 0.62 + i * 0.13, -0.24), wood))
    for sx in (-1, 1):
        parts.append(box((0.05, 0.45, 0.4), (sx * (w / 2 - 0.1), 0.225, -0.05), metal))
        parts.append(box((0.05, 0.4, 0.04), (sx * (w / 2 - 0.1), 0.65, -0.24), metal))
    m = merge(*parts)
    m["seat"] = [0.0, 0.47, 0.0]
    return with_colliders(m)


def trash_can() -> dict:
    return merge(cylinder(0.25, 0.85, 8, rgb("#3c4a3e"), r_top=0.27), cylinder(0.28, 0.06, 8, rgb("#2a2c30"), y0=0.85))


def book(rng=None) -> dict:
    rng = np.random.default_rng(0) if rng is None else rng
    c = _pick(rng, CLOTH_COLORS)
    return merge(box((0.16, 0.03, 0.23), (0, 0.015, 0), c), box((0.15, 0.026, 0.225), (0.006, 0.015, 0), rgb("#f2efe6")))


def bottle(rng=None) -> dict:
    rng = np.random.default_rng(0) if rng is None else rng
    c = _pick(rng, ["#2e6b3f", "#5a3a22", "#c8dce6", "#e8e6e1"])
    return merge(cylinder(0.035, 0.18, 8, c), cylinder(0.035, 0.04, 8, c, r_top=0.013, y0=0.18),
                 cylinder(0.013, 0.04, 6, c, y0=0.22), cylinder(0.015, 0.015, 6, rgb("#222"), y0=0.26))


def phone() -> dict:
    m = box((0.075, 0.009, 0.155), (0, 0.0045, 0), rgb("#151618"))
    m["extras"] = [emissive(box((0.068, 0.002, 0.145), (0, 0.0095, 0)), rgb("#2b4366"))]
    return m


def cardboard_box(rng=None, size=None) -> dict:
    rng = np.random.default_rng(0) if rng is None else rng
    s = np.array(size) if size is not None else rng.uniform(0.3, 0.6, 3)
    c = jitter_color(rgb("#b88a5a"), rng, 0.1)
    return merge(box(s, (0, s[1] / 2, 0), c), box((s[0] * 1.002, s[1] * 0.01 + 0.004, 0.06), (0, s[1], 0), shade(c, 1.2)))


# ---------------------------------------------------------------- nature

def tree(rng=None, kind="broadleaf", h=None) -> dict:
    """Trees: 'pine', 'broadleaf', 'birch', 'dead', 'palm', 'cactus'."""
    rng = np.random.default_rng(0) if rng is None else rng
    bark = jitter_color(rgb("#5a4030"), rng, 0.15)
    parts = []
    if kind == "pine":
        h = rng.uniform(6, 14) if h is None else h
        green = jitter_color(rgb(["#2c5530", "#244a2e", "#33603a", "#1f4a3a"][rng.integers(4)]), rng, 0.1)
        parts.append(cylinder(0.04 * h ** 0.8, h * 0.35, 6, bark, r_top=0.03 * h ** 0.8))
        n = 3 + int(h / 4)
        spheres = []
        for i in range(n):
            t = i / n
            r = (1 - t) * 0.28 * h + 0.3
            y0 = h * (0.2 + 0.72 * t)
            parts.append(cone(r, h * 0.3, 7, shade(green, 0.9 + 0.2 * t), y0=y0, phase=rng.uniform(0, 1)))
            spheres.append([0.0, y0 + h * 0.08, 0.0, r * 0.75])
        cols = [[[-0.2, 0, -0.2], [0.2, h * 0.4, 0.2]]]
    elif kind in ("broadleaf", "birch"):
        h = rng.uniform(5, 11) if h is None else h
        if kind == "birch":
            bark = rgb("#e8e4da")
            green = jitter_color(rgb("#7aa64a"), rng, 0.1)
        else:
            green = jitter_color(rgb(["#4a7a34", "#3f6e2f", "#5b8a3a", "#6a8f3a"][rng.integers(4)]), rng, 0.1)
        tw = 0.025 * h if kind == "broadleaf" else 0.015 * h
        trunk_top = np.array([rng.uniform(-0.2, 0.2), h * 0.6, rng.uniform(-0.2, 0.2)])
        parts.append(limb([0, 0, 0], trunk_top, tw * 1.2, tw * 0.7, 6, bark))
        if kind == "birch":
            for y in np.linspace(0.5, h * 0.55, 5):
                parts.append(box((tw * 2.2, 0.04, tw * 0.5), (trunk_top[0] * y / h / 0.6, y, tw * 0.9), rgb("#2a2a2a")))
        cr = h * (0.26 if kind == "broadleaf" else 0.18)
        spheres = []
        for i in range(rng.integers(3, 6)):
            off = rng.normal(0, 0.45, 3) * cr * np.array([1, 0.5, 1])
            c = trunk_top + [0, cr * 0.6, 0] + off
            rr = cr * rng.uniform(0.6, 0.9)
            parts.append(limb(trunk_top, c, tw * 0.5, tw * 0.25, 4, bark))
            parts.append(icosphere(rr, 0 if rng.random() < 0.5 else 1, jitter_color(green, rng, 0.12), c, rng=rng,
                                   jitter=0.18))
            spheres.append([*c, rr * 0.92])
        cols = [[[-tw * 1.5, 0, -tw * 1.5], [tw * 1.5, h * 0.6, tw * 1.5]]]
    elif kind == "dead":
        h = rng.uniform(4, 8) if h is None else h
        bark = jitter_color(rgb("#6b5a4a"), rng, 0.1)
        top = np.array([rng.uniform(-0.3, 0.3), h, rng.uniform(-0.3, 0.3)])
        parts.append(limb([0, 0, 0], top, 0.18, 0.04, 6, bark))
        for i in range(rng.integers(3, 6)):
            y = rng.uniform(0.35, 0.85)
            a = rng.uniform(0, 2 * np.pi)
            p0 = top * y
            p1 = p0 + [np.cos(a) * h * 0.3, h * rng.uniform(0.1, 0.3), np.sin(a) * h * 0.3]
            parts.append(limb(p0, p1, 0.06 * (1 - y) + 0.02, 0.015, 4, bark))
        cols = [[[-0.25, 0, -0.25], [0.25, h, 0.25]]]
    elif kind == "palm":
        h = rng.uniform(5, 9) if h is None else h
        green = rgb("#4f8a3a")
        bend = rng.uniform(0.5, 1.5)
        pts = [np.array([bend * (t ** 2), h * t, 0]) for t in np.linspace(0, 1, 5)]
        for a, b_ in zip(pts[:-1], pts[1:]):
            parts.append(limb(a, b_, 0.18, 0.16, 6, rgb("#8a7050")))
        for i in range(7):
            a = 2 * np.pi * i / 7
            tip = pts[-1] + [np.cos(a) * 2.2, -0.9, np.sin(a) * 2.2]
            mid = pts[-1] + [np.cos(a) * 1.1, 0.25, np.sin(a) * 1.1]
            parts.append(limb(pts[-1], mid, 0.16, 0.12, 3, green, rz_scale=0.15))
            parts.append(limb(mid, tip, 0.12, 0.02, 3, green, rz_scale=0.15))
        cols = [[[-0.3, 0, -0.3], [bend + 0.3, h, 0.3]]]
    elif kind == "cactus":
        h = rng.uniform(1.5, 4.5) if h is None else h
        green = jitter_color(rgb("#4d7a43"), rng, 0.1)
        r = 0.12 + 0.03 * h
        parts.append(limb([0, 0, 0], [0, h, 0], r, r * 0.85, 8, green, cap=0.5))
        for side in (-1, 1):
            if rng.random() < 0.75:
                y = h * rng.uniform(0.3, 0.6)
                elbow = np.array([side * r * 3, y, 0])
                parts.append(limb([0, y, 0], elbow, r * 0.6, r * 0.6, 6, green))
                parts.append(limb(elbow, elbow + [0, h * rng.uniform(0.25, 0.4), 0], r * 0.6, r * 0.5, 6, green, cap=0.6))
        cols = [[[-r, 0, -r], [r, h, r]]]
    else:
        raise ValueError(kind)
    m = merge(*parts)
    m["colliders"] = cols
    if kind in ("pine", "broadleaf", "birch"):
        m["spheres"] = spheres       # canopy occluders: [x, y, z, r] in mesh space
    return m


def rock(rng=None, size=None, color=None) -> dict:
    rng = np.random.default_rng(0) if rng is None else rng
    s = rng.uniform(0.3, 1.6) if size is None else size
    c = jitter_color(rgb(["#7d7b76", "#8a857c", "#6b6862", "#9a9187"][rng.integers(4)]) if color is None else color, rng, 0.1)
    m = icosphere(s, 0 if s < 0.6 else 1, c, (0, s * 0.35, 0), scale=(rng.uniform(0.8, 1.3), rng.uniform(0.5, 0.8),
                                                                       rng.uniform(0.8, 1.3)), rng=rng, jitter=0.25)
    m = facet(m, rng, 0.12)
    return with_colliders(m)


def bush(rng=None, size=None, color=None) -> dict:
    rng = np.random.default_rng(0) if rng is None else rng
    s = rng.uniform(0.4, 1.0) if size is None else size
    g = jitter_color(rgb(["#4a7a34", "#3a6a30", "#5b8a3a", "#6b8f45"][rng.integers(4)]) if color is None else color, rng, 0.1)
    parts = []
    for i in range(rng.integers(3, 6)):
        c = np.array([rng.normal(0, 0.35 * s), s * rng.uniform(0.35, 0.6), rng.normal(0, 0.35 * s)])
        parts.append(icosphere(s * rng.uniform(0.4, 0.6), 0, jitter_color(g, rng, 0.1), c, rng=rng, jitter=0.2))
    return with_colliders(merge(*parts))


def grass_tuft(rng=None, color=None, h=None) -> dict:
    rng = np.random.default_rng(0) if rng is None else rng
    g = jitter_color(rgb("#5e8c3a") if color is None else color, rng, 0.15)
    h = rng.uniform(0.15, 0.4) if h is None else h
    parts = []
    for i in range(rng.integers(3, 6)):
        a = rng.uniform(0, 2 * np.pi)
        base = np.array([np.cos(a), 0, np.sin(a)]) * rng.uniform(0, 0.06)
        tip = base + [np.cos(a) * rng.uniform(0.02, 0.1), h * rng.uniform(0.6, 1), np.sin(a) * rng.uniform(0.02, 0.1)]
        parts.append(limb(base, tip, 0.018, 0.002, 3, jitter_color(g, rng, 0.1)))
    return merge(*parts)


def flower(rng=None) -> dict:
    rng = np.random.default_rng(0) if rng is None else rng
    c = _pick(rng, ["#e63946", "#f1c40f", "#f4f1de", "#9b5de5", "#ff8fab"])
    h = rng.uniform(0.15, 0.3)
    return merge(limb([0, 0, 0], [0, h, 0], 0.005, 0.004, 3, rgb("#4a7a34")), icosphere(0.03, 0, c, (0, h, 0)))


class Heightfield:
    """Regular grid of heights over [x0, x0 + w] x [z0, z0 + d] with bilinear lookup."""

    def __init__(self, heights, x0, z0, cell):
        self.h = np.asarray(heights, np.float64)
        self.x0, self.z0, self.cell = float(x0), float(z0), float(cell)

    @property
    def extent(self):
        nz, nx = self.h.shape
        return self.x0, self.z0, self.x0 + (nx - 1) * self.cell, self.z0 + (nz - 1) * self.cell

    def __call__(self, x, z):
        x, z = np.asarray(x, float), np.asarray(z, float)
        nz, nx = self.h.shape
        fx = np.clip((x - self.x0) / self.cell, 0, nx - 1.000001)
        fz = np.clip((z - self.z0) / self.cell, 0, nz - 1.000001)
        ix, iz = np.floor(fx).astype(int), np.floor(fz).astype(int)
        tx, tz = fx - ix, fz - iz
        h = self.h
        # match the triangulation of terrain_mesh: split along the (0,0)-(1,1) diagonal
        h00, h10, h01, h11 = h[iz, ix], h[iz, ix + 1], h[iz + 1, ix], h[iz + 1, ix + 1]
        upper = tx >= tz
        a = np.where(upper, h00 + tx * (h10 - h00) + tz * (h11 - h10), h00 + tz * (h01 - h00) + tx * (h11 - h01))
        return a

    def gradient(self, x, z, eps=0.25):
        return ((self(x + eps, z) - self(x - eps, z)) / (2 * eps), (self(x, z + eps) - self(x, z - eps)) / (2 * eps))


def value_noise(shape, scale, rng, octaves=4, persistence=0.5):
    """Fractal value noise in [-1, 1] on a 2D grid (smoothly interpolated random lattices)."""
    from scipy.ndimage import zoom
    out = np.zeros(shape)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        cells = max(2, int(np.ceil(max(shape) / scale * 2 ** o)) + 2)
        lat = rng.uniform(-1, 1, (cells, cells))
        up = zoom(lat, (shape[0] / (cells - 1) * 1.0, shape[1] / (cells - 1) * 1.0), order=3)[:shape[0], :shape[1]]
        if up.shape != shape:
            up = np.pad(up, ((0, shape[0] - up.shape[0]), (0, shape[1] - up.shape[1])), mode="edge")
        out += amp * up
        tot += amp
        amp *= persistence
    return out / tot


def terrain_mesh(hf: Heightfield, color_fn, rng=None, jitter=0.05) -> dict:
    """Faceted triangle mesh of a heightfield; color_fn(x, y, z, slope) -> linear RGB per face (arrays)."""
    rng = np.random.default_rng(0) if rng is None else rng
    h = hf.h
    nz, nx = h.shape
    xs = hf.x0 + np.arange(nx) * hf.cell
    zs = hf.z0 + np.arange(nz) * hf.cell
    X, Z = np.meshgrid(xs, zs)
    P = np.stack([X, h, Z], -1)
    a, b_, c, d = P[:-1, :-1], P[:-1, 1:], P[1:, :-1], P[1:, 1:]
    # triangles (a, d, b) and (a, c, d) -- CCW seen from +Y
    t1 = np.stack([a, d, b_], 2).reshape(-1, 3, 3)
    t2 = np.stack([a, c, d], 2).reshape(-1, 3, 3)
    tris = np.concatenate([t1, t2])
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    if (n[:, 1] < 0).mean() > 0.5:
        tris = tris[:, [0, 2, 1]]
        n = -n
    cen = tris.mean(1)
    slope = 1 - n[:, 1]
    col = np.asarray(color_fn(cen[:, 0], cen[:, 1], cen[:, 2], slope), np.float32)
    col = np.clip(col * (1 + rng.uniform(-jitter, jitter, (len(col), 1))), 0, 1)
    pos = tris.reshape(-1, 3)
    return {"positions": pos.astype(np.float32), "indices": np.arange(len(pos), dtype=np.uint32).reshape(-1, 3),
            "colors": np.repeat(col, 3, 0).astype(np.float32)}


def mountain_ring(rng, radius=600.0, n=14, h=(120, 320), color=None, snow=True) -> dict:
    """Distant low-poly mountains around the origin for an outdoor backdrop."""
    rock_c = rgb("#6f6a66") if color is None else color
    parts = []
    for i in range(n):
        a = 2 * np.pi * (i + rng.uniform(-0.3, 0.3)) / n
        r = radius * rng.uniform(0.85, 1.15)
        hh = rng.uniform(*h)
        base = min(rng.uniform(1.2, 2.0) * hh, 0.5 * r)     # never reach over the playable area
        c = np.array([np.cos(a) * r, -5, np.sin(a) * r])
        m = cone(base, hh, 6, jitter_color(rock_c, rng, 0.1), phase=rng.uniform(0, 1))
        m = transform(m, c, yaw_q(rng.uniform(0, 6)), s=(1, 1, rng.uniform(0.7, 1.2)))
        parts.append(facet(m, rng, 0.1))
        if snow:
            sm = cone(base * 0.28, hh * 0.28, 6, rgb("#f2f4f7"), y0=hh * 0.725, phase=0)
            parts.append(transform(sm, c, None, s=(1, 1, 1)))
    return merge(*parts)


# ---------------------------------------------------------------- urban

FACADE_COLORS = ["#c9b79c", "#a35d4a", "#8e9aaf", "#e8e4da", "#7b6d5d", "#b5838d", "#6d7a84", "#d8cbb5", "#9c6644",
                 "#5b6c5d", "#c2a878", "#e0c9a6"]


def building(rng=None, w=12.0, d=10.0, floors=None, floor_h=3.2, color=None, storefront=True) -> dict:
    """Block building centred on x, front facade at z = +d/2, windows on all sides."""
    rng = np.random.default_rng(0) if rng is None else rng
    floors = int(rng.integers(2, 9)) if floors is None else floors
    H = floors * floor_h + 0.6
    c = _pick(rng, FACADE_COLORS) if color is None else color
    trim = shade(c, rng.choice([0.7, 1.2]))
    glass = jitter_color(rgb(["#2b3a4a", "#3d5266", "#25313d", "#4a6278"][rng.integers(4)]), rng, 0.1)
    parts = [box((w, H, d), (0, H / 2, 0), c), box((w + 0.3, 0.4, d + 0.3), (0, H + 0.2, 0), trim)]
    win_w = rng.uniform(1.0, 1.6)
    win_h = rng.uniform(1.3, 1.8)
    lit_p = rng.uniform(0.0, 0.25)
    glow = []
    for face in range(4):
        fw = w if face % 2 == 0 else d
        n_win = max(1, int((fw - 1.0) / (win_w + 1.0)))
        spacing = fw / n_win
        q = yaw_q(face * np.pi / 2)
        off = (d if face % 2 == 0 else w) / 2
        for fl in range(floors):
            if fl == 0 and storefront and face == 0:
                continue
            y = fl * floor_h + floor_h * 0.35
            for i in range(n_win):
                x = -fw / 2 + spacing * (i + 0.5)
                pane = box((win_w, win_h, 0.06), (x, y + win_h / 2, off + 0.02), glass)
                sill = box((win_w + 0.2, 0.08, 0.14), (x, y - 0.03, off + 0.05), trim)
                if rng.random() < lit_p:
                    glow.append(transform(emissive(box((win_w * 0.92, win_h * 0.92, 0.02), (x, y + win_h / 2, off + 0.055)),
                                                   rgb("#ffe2a8")), q=q))
                parts.append(transform(merge(pane, sill), q=q))
        if face == 0 and storefront:
            sc = _pick(rng, CLOTH_COLORS)
            parts.append(box((w * 0.9, 2.6, 0.08), (0, 1.5, d / 2 + 0.03), glass))
            parts.append(box((w * 0.94, 0.5, 0.4), (0, 3.0, d / 2 + 0.2), sc))
            parts.append(box((1.2, 2.3, 0.1), (rng.uniform(-w * 0.3, w * 0.3), 1.15, d / 2 + 0.06), shade(sc, 0.6)))
        for fl in range(1, floors):
            parts.append(transform(box((fw, 0.15, 0.1), (0, fl * floor_h, off + 0.05), trim), q=q))
    m = merge(*parts)
    if glow:
        m["extras"] = glow
    return with_colliders(m, [[[-w / 2, 0, -d / 2], [w / 2, H, d / 2]]])


def road_strip(length=100.0, lanes=2, lane_w=3.5, dashed=True, color=None, center_line="#e8c547") -> dict:
    """Straight road along X centred at the origin, surface at y = 0.01, with lane markings."""
    asph = rgb("#3a3c40") if color is None else color
    W = lanes * lane_w
    parts = [box((length, 0.02, W), (0, 0.0, 0), asph)]
    white = rgb("#e8e8e2")
    for side in (-1, 1):
        parts.append(box((length, 0.005, 0.15), (0, 0.0125, side * (W / 2 - 0.25)), white))
    for k in range(1, lanes):
        z = -W / 2 + k * lane_w
        colr = rgb(center_line) if (k == lanes // 2 and lanes % 2 == 0) else white
        if dashed and not (k == lanes // 2 and lanes % 2 == 0 and center_line == "#e8c547"):
            for x in np.arange(-length / 2 + 1.5, length / 2 - 1.5, 9.0):
                parts.append(box((3.0, 0.005, 0.13), (x + 1.5, 0.0125, z), colr))
        else:
            for dz in ((-0.1, 0.1) if colr is not white else (0,)):
                parts.append(box((length, 0.005, 0.11), (0, 0.0125, z + dz), colr))
    return merge(*parts)


def crosswalk(width=7.0, depth=3.0) -> dict:
    parts = []
    for z in np.arange(-width / 2 + 0.3, width / 2, 0.9):
        parts.append(box((depth, 0.006, 0.5), (0, 0.013, z + 0.25), rgb("#ececea")))
    return merge(*parts)


def sidewalk(length=100.0, w=3.5, curb_h=0.15) -> dict:
    slab = rgb("#a9a6a0")
    parts = [box((length, curb_h, w), (0, curb_h / 2, 0), slab)]
    for x in np.arange(-length / 2, length / 2, 2.0):
        parts.append(box((0.03, 0.004, w), (x, curb_h + 0.002, 0), shade(slab, 0.8)))
    return merge(*parts)


def street_lamp(h=6.0) -> dict:
    metal = rgb("#2f3438")
    parts = [cylinder(0.14, 0.4, 8, metal), cylinder(0.07, h, 6, metal, r_top=0.05),
             limb([0, h, 0], [0, h + 0.3, 1.2], 0.04, 0.04, 5, metal),
             box((0.3, 0.12, 0.55), (0, h + 0.3, 1.35), metal)]
    m = merge(*parts)
    m["extras"] = [emissive(box((0.24, 0.02, 0.45), (0, h + 0.23, 1.35)), rgb("#fff1c8"))]
    return with_colliders(m, [[[-0.15, 0, -0.15], [0.15, h, 0.15]]])


def traffic_light(state="green") -> dict:
    metal = rgb("#2a2c30")
    parts = [cylinder(0.08, 3.2, 6, metal), box((0.35, 0.95, 0.3), (0, 3.6, 0), rgb("#1c1d1f"))]
    glow = []
    for i, (name, col) in enumerate((("red", "#ff3020"), ("amber", "#ffb000"), ("green", "#30e070"))):
        y = 3.92 - i * 0.31
        c = rgb(col) if name == state else shade(rgb(col), 0.15)
        disk = transform(cylinder(0.1, 0.03, 8), (0, y, 0.15), quat_axis_angle([1, 0, 0], np.pi / 2))
        glow.append(emissive(disk, c) if name == state else recolor(disk, c))
    m = merge(*parts, *[g for g in glow if "material" not in g])
    m["extras"] = [g for g in glow if "material" in g]
    return with_colliders(m, [[[-0.18, 0, -0.18], [0.18, 4.1, 0.18]]])


def guardrail(length=10.0) -> dict:
    metal = rgb("#b8bcc2")
    parts = [box((length, 0.3, 0.06), (0, 0.6, 0), metal)]
    for x in np.arange(-length / 2, length / 2 + 0.01, 2.0):
        parts.append(box((0.1, 0.7, 0.1), (x, 0.35, -0.06), shade(metal, 0.7)))
    return merge(*parts)


def road_ribbon(points, width, color, y_off=0.0, ground=None) -> dict:
    """Flat ribbon following a polyline (N, 3) in XZ; up-facing."""
    p = np.asarray(points, float)
    t = np.gradient(p, axis=0)
    t[:, 1] = 0
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    nrm = np.stack([-t[:, 2], np.zeros(len(t)), t[:, 0]], 1)   # left of travel direction
    L = p + nrm * width / 2
    R = p - nrm * width / 2
    if ground is not None:
        L[:, 1] = ground(L[:, 0], L[:, 2])
        R[:, 1] = ground(R[:, 0], R[:, 2])
    L[:, 1] += y_off
    R[:, 1] += y_off
    pos, idx = [], []
    for i in range(len(p) - 1):
        b = len(pos)
        pos += [L[i], R[i], R[i + 1], L[i + 1]]
        idx += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    pos = np.array(pos, np.float32)
    idx = np.array(idx)
    tri = pos[idx]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = n[:, 1] < 0
    idx[flip] = idx[flip][:, [0, 2, 1]]
    return make_mesh(pos, idx, color=color)


def dashed_ribbon(points, width, color, dash=3.0, gap=6.0, y_off=0.02, offset=0.0) -> dict:
    p = np.asarray(points, float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.gradient(p, axis=0)
    t[:, 1] = 0
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    nrm = np.stack([-t[:, 2], np.zeros(len(t)), t[:, 0]], 1)
    parts = []
    st = 0.0
    while st + dash < s[-1]:
        ss = np.linspace(st, st + dash, 3)
        pts = np.stack([np.interp(ss, s, p[:, k]) for k in range(3)], 1)
        nn = np.stack([np.interp(ss, s, nrm[:, k]) for k in range(3)], 1)
        parts.append(road_ribbon(pts + nn * offset, width, color, y_off))
        st += dash + gap
    return merge(*parts)


# ---------------------------------------------------------------- room shell

def room_shell(w, d, h, floor_color, wall_color, ceiling_color=None, floor_kind="wood", rng=None,
               wall_colors=None) -> dict:
    """Inward-facing floor, four walls and ceiling of a w x d x h room centred at the origin (floor y = 0)."""
    rng = np.random.default_rng(0) if rng is None else rng
    parts = []
    if floor_kind == "tiles":
        tile = 0.5
        nx, nz = max(1, int(round(w / tile))), max(1, int(round(d / tile)))
        alt = shade(floor_color, 0.82)
        parts.append(grid_patch(w, d, nx, nz, 0.0, colors_fn=lambda i, k: floor_color if (i + k) % 2 == 0 else alt))
    elif floor_kind == "wood":
        plank = 0.18
        nz = max(1, int(round(d / plank)))
        cols = [jitter_color(floor_color, rng, 0.07) for _ in range(nz)]
        parts.append(grid_patch(w, d, 1, nz, 0.0, colors_fn=lambda i, k: cols[k]))
    else:
        parts.append(grid_patch(w, d, 1, 1, 0.0, color=floor_color))
    wc = wall_colors or [wall_color] * 4
    # walls: normals pointing inward
    parts.append(quad([[-w / 2, 0, -d / 2], [w / 2, 0, -d / 2], [w / 2, h, -d / 2], [-w / 2, h, -d / 2]], wc[0]))   # back, faces +Z
    parts.append(quad([[w / 2, 0, d / 2], [-w / 2, 0, d / 2], [-w / 2, h, d / 2], [w / 2, h, d / 2]], wc[1]))      # front, faces -Z
    parts.append(quad([[-w / 2, 0, d / 2], [-w / 2, 0, -d / 2], [-w / 2, h, -d / 2], [-w / 2, h, d / 2]], wc[2]))  # left, faces +X
    parts.append(quad([[w / 2, 0, -d / 2], [w / 2, 0, d / 2], [w / 2, h, d / 2], [w / 2, h, -d / 2]], wc[3]))      # right, faces -X
    cc = rgb("#f4f2ee") if ceiling_color is None else ceiling_color
    parts.append(quad([[-w / 2, h, -d / 2], [w / 2, h, -d / 2], [w / 2, h, d / 2], [-w / 2, h, d / 2]], cc))       # faces -Y
    # baseboards (inward-facing boxes along walls)
    bb = shade(wall_color, 0.8)
    parts.append(box((w, 0.1, 0.02), (0, 0.05, -d / 2 + 0.01), bb))
    parts.append(box((w, 0.1, 0.02), (0, 0.05, d / 2 - 0.01), bb))
    parts.append(box((0.02, 0.1, d), (-w / 2 + 0.01, 0.05, 0), bb))
    parts.append(box((0.02, 0.1, d), (w / 2 - 0.01, 0.05, 0), bb))
    return merge(*parts)


# ---------------------------------------------------------------- voxel world

BLOCKS = {  # id: (name, top, side, bottom) -- an original palette (sRGB hex)
    1: ("grass", "#6aa84f", "#7d6a4a", "#7a5c3e"),
    2: ("dirt", "#7a5c3e", "#7a5c3e", "#7a5c3e"),
    3: ("stone", "#8a8d91", "#8a8d91", "#8a8d91"),
    4: ("sand", "#e3d59a", "#dccb8c", "#d6c486"),
    5: ("water", "#3f76c4", "#3f76c4", "#3f76c4"),
    6: ("log", "#a78b5f", "#6b4b2f", "#a78b5f"),
    7: ("leaves", "#3f7f35", "#3a7531", "#356b2d"),
    8: ("planks", "#c0965c", "#b88d55", "#b08650"),
    9: ("cobble", "#70737a", "#70737a", "#70737a"),
    10: ("glass", "#bfe3ee", "#bfe3ee", "#bfe3ee"),
    11: ("snow", "#f4f6f8", "#e6e9ed", "#7a5c3e"),
    12: ("brick", "#a3523d", "#a3523d", "#a3523d"),
    13: ("roof", "#7a3b30", "#6e352b", "#6e352b"),
    14: ("flower_red", "#d23c3c", "#4a8a3a", "#4a8a3a"),
    15: ("flower_yellow", "#e8c547", "#4a8a3a", "#4a8a3a"),
    16: ("gravel", "#9a958c", "#9a958c", "#9a958c"),
    17: ("ore", "#8a8d91", "#8a8d91", "#8a8d91"),
    18: ("lamp", "#ffe08a", "#ffe08a", "#ffe08a"),
}
BLOCK_ID = {v[0]: k for k, v in BLOCKS.items()}
SOLID = np.ones(256, bool)
SOLID[[0, 5, 14, 15]] = False


def greedy_mesh(vox: np.ndarray, block: float = 1.0, origin=(0.0, 0.0, 0.0), variants: np.ndarray | None = None,
                ids=None) -> dict:
    """Greedy-meshed surface of a block grid vox[x, y, z] (0 = air).

    Faces between two filled cells are skipped; coplanar neighbouring faces of the same block
    type (and colour variant) are merged into large quads. Colours come from BLOCKS
    (top / side / bottom), darkened slightly per variant for natural patches.
    """
    vox = np.asarray(vox)
    X, Y, Z = vox.shape
    var = np.zeros_like(vox) if variants is None else variants
    code = vox.astype(np.int32) * 4 + var.astype(np.int32)
    code[vox == 0] = 0
    filled = vox > 0
    ids = set(np.unique(vox)) if ids is None else ids
    pal = {}
    for k, (_, t, s_, b) in BLOCKS.items():
        pal[k] = (rgb(t), rgb(s_), rgb(b))
    pos, idx, col = [], [], []
    origin = np.asarray(origin, float)
    for axis in range(3):
        u_ax, v_ax = (axis + 1) % 3, (axis + 2) % 3
        n_a = vox.shape[axis]
        for sgn in (1, -1):
            # visible faces: filled cell whose neighbour along +-axis is empty (or out of range)
            nb = np.zeros_like(filled)
            sl_src = [slice(None)] * 3
            sl_dst = [slice(None)] * 3
            if sgn > 0:
                sl_dst[axis] = slice(0, n_a - 1)
                sl_src[axis] = slice(1, n_a)
            else:
                sl_dst[axis] = slice(1, n_a)
                sl_src[axis] = slice(0, n_a - 1)
            nb[tuple(sl_dst)] = filled[tuple(sl_src)]
            vis = filled & ~nb
            for a in range(n_a):
                sl = [slice(None)] * 3
                sl[axis] = a
                mask = np.where(vis[tuple(sl)], code[tuple(sl)], 0)   # 2D in (u?, v?) order of remaining axes
                # remaining axes order in numpy slicing is ascending axis index
                rem = [ax for ax in range(3) if ax != axis]
                m2 = mask if rem == [u_ax, v_ax] else mask.T
                nu, nv = m2.shape
                done = np.zeros_like(m2, bool)
                for i in range(nu):
                    j = 0
                    row = m2[i]
                    while j < nv:
                        cval = row[j]
                        if cval == 0 or done[i, j]:
                            j += 1
                            continue
                        w = 1
                        while j + w < nv and m2[i, j + w] == cval and not done[i, j + w]:
                            w += 1
                        h = 1
                        while i + h < nu and np.all(m2[i + h, j:j + w] == cval) and not done[i + h, j:j + w].any():
                            h += 1
                        done[i:i + h, j:j + w] = True
                        btype, bvar = cval // 4, cval % 4
                        top, side, bot = pal[btype]
                        c = top if (axis == 1 and sgn > 0) else (bot if axis == 1 else side)
                        c = shade(c, 1 - 0.045 * bvar)
                        plane = a + (1 if sgn > 0 else 0)
                        corners = []
                        for du, dv in ((0, 0), (h, 0), (h, w), (0, w)):
                            p = np.zeros(3)
                            p[axis] = plane
                            p[u_ax] = i + du
                            p[v_ax] = j + dv
                            corners.append(origin + p * block)
                        b0 = len(pos)
                        pos += corners
                        col += [c] * 4
                        # orientation: (u, v, n) right-handed -> CCW seen from +n when sgn > 0
                        if sgn > 0:
                            idx += [[b0, b0 + 1, b0 + 2], [b0, b0 + 2, b0 + 3]]
                        else:
                            idx += [[b0, b0 + 2, b0 + 1], [b0, b0 + 3, b0 + 2]]
                        j += w
    if not pos:
        return empty_mesh()
    return make_mesh(pos, idx, col)
