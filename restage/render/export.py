"""Write the renderer's meshes into `.restage` documents (numpy; docs/SCENE_FORMAT.md).

The decoder of the model predicts parameters, builds meshes with ``shapes`` for the
loss, and uses these helpers to write *exactly the same* meshes to the replica:

* ``mesh_to_numpy``: any (positions, faces, colours) -> a format mesh dict, dropping
  vertices that no kept face uses;
* ``rounded_boxes_to_numpy``: one location's boxes, absent boxes removed;
* ``humanoid_to_numpy``: a person in the bind (rest) pose with joints / weights in
  humanoid22 order, plus its skeleton (offsets depend on the shape);
* ``pose_track`` / ``camera_track``: per-frame matrices -> quaternion tracks;
* ``ground_to_numpy`` and ``scene_doc`` for a minimal one-location document.

The posed humanoid that ``shapes.humanoid`` renders equals the format's linear blend
skinning of the exported rest mesh with these tracks (tested in tests/test_render.py).
"""
from __future__ import annotations

import numpy as np

from ..scene import HUMANOID22, mat_to_quat
from . import shapes as S

__all__ = ["region_colors", "mesh_to_numpy", "to_numpy_mesh", "rounded_boxes_to_numpy", "humanoid_to_numpy", "skeleton_doc",
           "pose_track", "camera_track", "ground_to_numpy", "scene_doc", "preview_image"]


def _np(a) -> np.ndarray:
    return np.asarray(a)


def region_colors(region: np.ndarray, colors) -> np.ndarray:
    """Per-vertex colours [V, 3] from region labels [V] and region colours [5, 3]
    (skin, top, bottom, shoes, hair; linear RGB)."""
    return np.asarray(colors, np.float32)[np.asarray(region)]


def mesh_to_numpy(positions, faces, colors=None, mesh_id: str = "mesh", face_mask=None, joints=None, weights=None,
                  material: str | None = None) -> dict:
    """Format mesh dict from numpy-convertible arrays.

    positions [V, 3]; faces [F, 3]; colors [V, 3] or [3] (optional); face_mask [F] bool keeps a
    subset of faces (vertices no kept face uses are dropped and indices remapped); joints [V]
    (rigid, weight 1) or [V, 4] with weights [V, 4].
    """
    p = _np(positions).astype(np.float32).reshape(-1, 3)
    f = _np(faces).astype(np.int64).reshape(-1, 3)
    if face_mask is not None:
        f = f[_np(face_mask).astype(bool)]
    used = np.unique(f) if len(f) else np.zeros(0, np.int64)
    remap = -np.ones(len(p), np.int64)
    remap[used] = np.arange(len(used))
    m = {"id": mesh_id, "positions": p[used], "indices": remap[f].astype(np.uint32)}
    if colors is not None:
        c = np.asarray(colors, np.float32)
        c = np.broadcast_to(c, p.shape) if c.ndim == 1 else c.reshape(-1, 3)
        m["colors"] = np.ascontiguousarray(c[used])
    if joints is not None:
        j = _np(joints)
        if j.ndim == 1:
            jj = np.zeros((len(p), 4), np.uint8)
            jj[:, 0] = j
            ww = np.zeros((len(p), 4), np.float32)
            ww[:, 0] = 1
        else:
            jj, ww = j.astype(np.uint8), _np(weights).astype(np.float32)
        m["joints"], m["weights"] = jj[used], ww[used]
    if material is not None:
        m["material"] = material
    return m


to_numpy_mesh = mesh_to_numpy     # the name used in the design notes


def rounded_boxes_to_numpy(verts, faces, meta, colors, present=None, mesh_id: str = "set_boxes") -> dict:
    """One job's boxes from ``shapes.rounded_boxes`` as one static mesh in world space.

    verts [N*Vb, 3] (one job, e.g. ``np.array(v[b])``); faces / meta from rounded_boxes;
    colors [N, 3] per box; present [N] bool (absent boxes are removed).
    """
    box = meta["box"]
    vcol = np.asarray(colors, np.float32)[box]
    fmask = None
    if present is not None:
        fmask = np.asarray(present, bool)[box[np.asarray(faces)[:, 0]]]
    return mesh_to_numpy(verts, faces, vcol, mesh_id, face_mask=fmask)


def skeleton_doc(offsets, sid: str = "humanoid22", scale: float | None = None) -> dict:
    """humanoid22 skeleton with the given offsets [22, 3] (e.g. ``shapes.humanoid_offsets(shape)``).

    Like the generator (restage/gen/FORMAT_NOTES.md item 2) it carries ``"base": "humanoid22"``
    and the height ``"scale"``; the explicit offsets are authoritative (they also include the
    shoulder width).
    """
    off = np.asarray(offsets, np.float64)
    d = {"id": sid, "base": "humanoid22",
         "joints": [{"name": n, "parent": int(p), "offset": [float(x) for x in off[j]]}
                    for j, (n, p, _) in enumerate(HUMANOID22)]}
    if scale is not None:
        d["scale"] = float(scale)
    return d


def humanoid_to_numpy(shape, region_cols, mesh_id: str = "person", skeleton_id: str | None = None):
    """A person in the bind pose for export.

    shape [4] (as given to ``shapes.humanoid``); region_cols [5, 3] linear RGB for skin, top,
    bottom, shoes, hair. Returns (mesh dict with positions / indices / colors / joints / weights,
    skeleton dict). Mesh positions are in entity space with the feet on y = 0, facing +Z.
    """
    pos, faces, offs = S.humanoid_rest_numpy(shape)
    t = S.HUMANOID_TEMPLATE
    m = mesh_to_numpy(pos, faces, region_colors(t["region"], region_cols), mesh_id, joints=t["joint"])
    return m, skeleton_doc(offs, skeleton_id or f"{mesh_id}_skeleton", scale=float(np.asarray(shape).reshape(-1)[0]))


def pose_track(joint_rot, root_rot, root_pos) -> dict:
    """Per-frame matrices -> entity track fields.

    joint_rot [T, 22, 3, 3] local rotations, root_rot [T, 3, 3], root_pos [T, 3] ->
    {"pose": [T, 22, 4], "rotation": [T, 4], "position": [T, 3]} (quaternions [x, y, z, w]).
    """
    return {"pose": mat_to_quat(_np(joint_rot)).astype(np.float32),
            "rotation": mat_to_quat(_np(root_rot)).astype(np.float32),
            "position": _np(root_pos).astype(np.float32)}


def camera_track(cam_R, cam_p, fov_deg) -> dict:
    """Camera-to-world rotations [T, 3, 3], positions [T, 3], vertical fov [T] -> camera track."""
    return {"position": _np(cam_p).astype(np.float32), "rotation": mat_to_quat(_np(cam_R)).astype(np.float32),
            "fov": _np(fov_deg).astype(np.float32).reshape(-1)}


def ground_to_numpy(extent: float = 30.0, n: int = 8, color=(0.4, 0.4, 0.38), y: float = 0.0,
                    mesh_id: str = "ground") -> dict:
    """A uniform ground grid facing +Y (same topology as ``shapes.ground_plane`` with power 1)."""
    import mlx.core as mx
    v, f, _ = S.ground_plane(mx.zeros((1, 3)), extent, n, 1.0, y)
    return mesh_to_numpy(np.array(v[0]), f, color, mesh_id)


def scene_doc(meshes, props, entities, cameras, n_frames: int, fps: int = 30, skeletons=(), sky=(0.55, 0.65, 0.8),
              title: str = "restage render export", kind: str = "other") -> dict:
    """Minimal one-location, one-shot document.

    meshes: list of mesh dicts; props: list of {"id", "mesh", ...}; entities: list of entity
    dicts (with "track"); cameras: list of one camera track dict (frames 0..n_frames).
    """
    shot = {"id": "s0", "start": 0, "end": int(n_frames), "location": "loc0", "camera": "cam0",
            "size": "wide", "move": "static", "transition": "cut"}
    return {
        "format": "restage.scene", "version": 1, "fps": fps, "n_frames": int(n_frames),
        "meta": {"title": title, "producer": "restage.render.export"},
        "materials": [], "meshes": list(meshes), "skeletons": list(skeletons),
        "locations": [{"id": "loc0", "name": "loc0", "kind": kind, "bounds": [[-10, 0, -10], [10, 4, 10]],
                       "environment": {"sky": list(map(float, sky)), "ambient": 0.45, "sun_dir": [0.4, -1, -0.3],
                                       "sun": 1.0},
                       "props": list(props)}],
        "entities": list(entities),
        "cameras": [{"id": "cam0", "shot": "s0", "track": cameras[0]}],
        "shots": [shot], "continuity": [],
    }


def preview_image(verts, faces, colors, cam_R, cam_p, fov_deg, height: int = 288, width: int = 512,
                  sky=(0.55, 0.65, 0.8), light=(0.4, 1.0, 0.3), sigma: float = 0.35) -> np.ndarray:
    """Quick flat-shaded preview of one world-space mesh through one camera, rendered with the
    soft rasteriser (useful to eyeball what the decoder writes). Returns uint8 RGB [H, W, 3].

    verts [V, 3]; faces [F, 3]; colors [V, 3] linear RGB; cam_R [3, 3]; cam_p [3]; fov_deg float.
    """
    import mlx.core as mx

    from .camera import project
    from .raster import rasterize
    v = np.asarray(verts, np.float32)
    f = np.asarray(faces, np.int64)
    c = np.asarray(colors, np.float32)
    tri = v[f]                                                     # unshare vertices: flat colour per face
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    ld = np.asarray(light, np.float32) / np.linalg.norm(light)
    shade = 0.45 + 0.55 * np.clip(n @ ld, 0, 1)
    col = c[f].mean(1) * shade[:, None]
    vv = mx.array(tri.reshape(1, -1, 3))
    ndc = project(vv, mx.array(np.asarray(cam_R, np.float32)[None]), mx.array(np.asarray(cam_p, np.float32)[None]),
                  mx.array([float(fov_deg)]), width / height)
    attrs = mx.array(np.repeat(col, 3, 0)[None])
    r = rasterize(ndc, np.arange(3 * len(f)).reshape(-1, 3), attrs, height=height, width=width, sigma=sigma,
                  gamma=0.02)
    a = np.array(r.alpha[0])[..., None]
    img = a * np.array(r.attrs[0]) + (1 - a) * np.asarray(sky, np.float32)
    srgb = np.where(img <= 0.0031308, img * 12.92, 1.055 * np.clip(img, 0, 1) ** (1 / 2.4) - 0.055)
    return (np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8)
