"""Read and write `.restage` scene files (docs/SCENE_FORMAT.md).

A Scene is the JSON document plus a dict of named numpy arrays. Accessors are
created on write: anywhere the document holds a numpy array, it is replaced by
{"offset", "dtype", "shape"} into the binary chunk; reading reverses that.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"RSTG"
VERSION = 1
DTYPES = {"f32": np.float32, "f16": np.float16, "u32": np.uint32, "u16": np.uint16, "u8": np.uint8,
          "i16": np.int16}
NAMES = {np.dtype(v): k for k, v in DTYPES.items()}


def _pack(doc, chunks: list[bytes], pos: list[int]):
    if isinstance(doc, np.ndarray):
        a = np.ascontiguousarray(doc)
        if a.dtype == np.float64:
            a = a.astype(np.float32)
        elif a.dtype == np.int64 or a.dtype == np.int32:
            a = a.astype(np.uint32)
        elif a.dtype == np.bool_:
            a = a.astype(np.uint8)
        if a.dtype not in NAMES:
            raise TypeError(f"unsupported array dtype {a.dtype}")
        b = a.tobytes()
        acc = {"offset": pos[0], "dtype": NAMES[a.dtype], "shape": list(a.shape)}
        pad = (-len(b)) % 4
        chunks.append(b + b"\0" * pad)
        pos[0] += len(b) + pad
        return acc
    if isinstance(doc, dict):
        return {k: _pack(v, chunks, pos) for k, v in doc.items()}
    if isinstance(doc, (list, tuple)):
        return [_pack(v, chunks, pos) for v in doc]
    if isinstance(doc, (np.floating, np.integer)):
        return doc.item()
    return doc


def _is_acc(d) -> bool:
    return isinstance(d, dict) and set(d) == {"offset", "dtype", "shape"}


def _unpack(doc, buf: memoryview):
    if _is_acc(doc):
        dt = np.dtype(DTYPES[doc["dtype"]])
        n = int(np.prod(doc["shape"])) if doc["shape"] else 1
        return np.frombuffer(buf, dt, n, doc["offset"]).reshape(doc["shape"]).copy()
    if isinstance(doc, dict):
        return {k: _unpack(v, buf) for k, v in doc.items()}
    if isinstance(doc, list):
        return [_unpack(v, buf) for v in doc]
    return doc


def save(path: str | Path, doc: dict) -> None:
    """Write a scene document whose arrays are numpy arrays."""
    chunks: list[bytes] = []
    packed = _pack(doc, chunks, [0])
    j = json.dumps(packed, separators=(",", ":")).encode()
    j += b" " * ((-len(j)) % 4)
    b = b"".join(chunks)
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<II", VERSION, len(j)) + j + struct.pack("<I", len(b)) + b)


def load(path: str | Path) -> dict:
    data = Path(path).read_bytes()
    if data[:4] != MAGIC:
        raise ValueError(f"{path}: not a .restage file")
    version, jl = struct.unpack_from("<II", data, 4)
    if version != VERSION:
        raise ValueError(f"{path}: unsupported version {version}")
    doc = json.loads(data[12:12 + jl])
    (bl,) = struct.unpack_from("<I", data, 12 + jl)
    return _unpack(doc, memoryview(data)[16 + jl:16 + jl + bl])


# ---------------------------------------------------------------- skeleton

HUMANOID22 = [  # name, parent, rest offset from parent (metres, +Y up, character faces +Z)
    ("pelvis", -1, (0.0, 0.95, 0.0)),
    ("spine1", 0, (0.0, 0.10, 0.0)),
    ("spine2", 1, (0.0, 0.12, 0.0)),
    ("chest", 2, (0.0, 0.12, 0.0)),
    ("neck", 3, (0.0, 0.16, 0.0)),
    ("head", 4, (0.0, 0.10, 0.0)),
    ("l_clavicle", 3, (0.07, 0.12, 0.0)),
    ("l_shoulder", 6, (0.11, 0.0, 0.0)),
    ("l_elbow", 7, (0.27, 0.0, 0.0)),
    ("l_wrist", 8, (0.25, 0.0, 0.0)),
    ("r_clavicle", 3, (-0.07, 0.12, 0.0)),
    ("r_shoulder", 10, (-0.11, 0.0, 0.0)),
    ("r_elbow", 11, (-0.27, 0.0, 0.0)),
    ("r_wrist", 12, (-0.25, 0.0, 0.0)),
    ("l_hip", 0, (0.09, -0.06, 0.0)),
    ("l_knee", 14, (0.0, -0.42, 0.0)),
    ("l_ankle", 15, (0.0, -0.41, 0.0)),
    ("l_toe", 16, (0.0, -0.05, 0.13)),
    ("r_hip", 0, (-0.09, -0.06, 0.0)),
    ("r_knee", 18, (0.0, -0.42, 0.0)),
    ("r_ankle", 19, (0.0, -0.41, 0.0)),
    ("r_toe", 20, (0.0, -0.05, 0.13)),
]
JOINTS = [j[0] for j in HUMANOID22]
PARENTS = np.array([j[1] for j in HUMANOID22])
OFFSETS = np.array([j[2] for j in HUMANOID22], np.float32)
JID = {n: i for i, n in enumerate(JOINTS)}


def skeleton_doc(scale: float = 1.0, sid: str = "humanoid22") -> dict:
    return {"id": sid, "joints": [{"name": n, "parent": int(p), "offset": [float(x) * scale for x in o]}
                                  for n, p, o in HUMANOID22]}


# ---------------------------------------------------------------- quaternion helpers (x, y, z, w)

def quat_mul(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], -1)


def quat_axis_angle(axis, angle):
    axis = np.asarray(axis, np.float64)
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    angle = np.asarray(angle, np.float64)[..., None]
    return np.concatenate([axis * np.sin(angle / 2), np.cos(angle / 2)], -1)


def quat_euler(x=0.0, y=0.0, z=0.0):
    """Intrinsic rotation applied as Y (yaw), then X (pitch), then Z (roll); angles broadcast."""
    x, y, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float))
    qy = quat_axis_angle(np.broadcast_to([0, 1, 0], y.shape + (3,)), y)
    qx = quat_axis_angle(np.broadcast_to([1, 0, 0], x.shape + (3,)), x)
    qz = quat_axis_angle(np.broadcast_to([0, 0, 1], z.shape + (3,)), z)
    return quat_mul(quat_mul(qy, qx), qz)


def quat_rotate(q, v):
    u, w = q[..., :3], q[..., 3:]
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def quat_to_mat(q):
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def look_rotation(eye, target, up=(0.0, 1.0, 0.0)):
    """Quaternion that points a camera's −Z axis from `eye` toward `target` (broadcasts over leading dims)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = target - eye
    f = f / np.maximum(np.linalg.norm(f, axis=-1, keepdims=True), 1e-9)
    z = -f
    x = np.cross(np.broadcast_to(np.asarray(up, float), z.shape), z)
    xn = np.linalg.norm(x, axis=-1, keepdims=True)
    x = np.where(xn > 1e-6, x / np.maximum(xn, 1e-9), np.broadcast_to([1.0, 0, 0], z.shape))
    y = np.cross(z, x)
    return mat_to_quat(np.stack([x, y, z], -1))


def mat_to_quat(m):
    m = np.asarray(m, float)
    tr = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = np.empty(m.shape[:-2] + (4,))
    # Shepperd's method, branch per element
    c0 = tr > 0
    s = np.sqrt(np.maximum(tr + 1.0, 1e-12)) * 2
    q0 = np.stack([(m[..., 2, 1] - m[..., 1, 2]) / s, (m[..., 0, 2] - m[..., 2, 0]) / s,
                   (m[..., 1, 0] - m[..., 0, 1]) / s, 0.25 * s], -1)
    s1 = np.sqrt(np.maximum(1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], 1e-12)) * 2
    q1 = np.stack([0.25 * s1, (m[..., 0, 1] + m[..., 1, 0]) / s1, (m[..., 0, 2] + m[..., 2, 0]) / s1,
                   (m[..., 2, 1] - m[..., 1, 2]) / s1], -1)
    s2 = np.sqrt(np.maximum(1.0 + m[..., 1, 1] - m[..., 0, 0] - m[..., 2, 2], 1e-12)) * 2
    q2 = np.stack([(m[..., 0, 1] + m[..., 1, 0]) / s2, 0.25 * s2, (m[..., 1, 2] + m[..., 2, 1]) / s2,
                   (m[..., 0, 2] - m[..., 2, 0]) / s2], -1)
    s3 = np.sqrt(np.maximum(1.0 + m[..., 2, 2] - m[..., 0, 0] - m[..., 1, 1], 1e-12)) * 2
    q3 = np.stack([(m[..., 0, 2] + m[..., 2, 0]) / s3, (m[..., 1, 2] + m[..., 2, 1]) / s3, 0.25 * s3,
                   (m[..., 1, 0] - m[..., 0, 1]) / s3], -1)
    c1 = ~c0 & (m[..., 0, 0] > m[..., 1, 1]) & (m[..., 0, 0] > m[..., 2, 2])
    c2 = ~c0 & ~c1 & (m[..., 1, 1] > m[..., 2, 2])
    c3 = ~c0 & ~c1 & ~c2
    q[:] = 0
    for c, qq in ((c0, q0), (c1, q1), (c2, q2), (c3, q3)):
        q[c] = qq[c]
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def forward_kinematics(pose, root_pos, root_rot, offsets=OFFSETS, parents=PARENTS, scale=None):
    """World joint positions and rotations.

    pose (T, J, 4) local rotations, root_pos (T, 3), root_rot (T, 4), offsets (J, 3).
    Returns (positions (T, J, 3), rotations (T, J, 4)).
    """
    T, J = pose.shape[:2]
    pos = np.zeros((T, J, 3))
    rot = np.zeros((T, J, 4))
    s = np.ones((T, 3)) if scale is None else scale
    for j in range(J):
        p = parents[j]
        if p < 0:
            rot[:, j] = quat_mul(root_rot, pose[:, j])
            pos[:, j] = root_pos + quat_rotate(root_rot, offsets[j] * s)
        else:
            rot[:, j] = quat_mul(rot[:, p], pose[:, j])
            pos[:, j] = pos[:, p] + quat_rotate(rot[:, p], offsets[j] * s)
    return pos, rot
