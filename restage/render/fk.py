"""Rotation maths and forward kinematics in MLX (differentiable).

Conventions (docs/SCENE_FORMAT.md, restage/scene.py):

* quaternions are ``[x, y, z, w]``;
* rotation matrices act on column vectors (``v' = R @ v``), so the columns of a
  camera or joint rotation are its local x, y, z axes in world coordinates;
* the 6D rotation representation (Zhou et al., CVPR 2019) is the first two
  *columns* of the matrix, ``x6 = [R[:, 0], R[:, 1]]``.

All 3x3 products are written as broadcast multiply-and-sum instead of ``@``:
MLX runs float32 matmuls as TF32 on the M5, which loses about three decimal
digits; the element-wise form is exact float32 and just as fast at this size.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from ..scene import OFFSETS, PARENTS

__all__ = [
    "matmul3", "matvec3", "transpose3", "cross3",
    "quat_mul", "quat_conj", "quat_rotate", "quat_normalize", "quat_to_mat", "mat_to_quat",
    "rot6d_to_mat", "mat_to_rot6d", "axis_angle_to_mat", "yaw_matrix",
    "forward_kinematics", "rest_joint_positions", "skin_rigid", "take_static",
]


# ---------------------------------------------------------------- small matrix helpers

def matmul3(a: mx.array, b: mx.array) -> mx.array:
    """Exact float32 ``a @ b`` for [..., 3, 3] (or [..., n, k] x [..., k, m]) arrays."""
    return (a[..., :, :, None] * b[..., None, :, :]).sum(-2)


def matvec3(m: mx.array, v: mx.array) -> mx.array:
    """Exact float32 ``m @ v`` for m [..., 3, 3] and v [..., 3]."""
    return (m * v[..., None, :]).sum(-1)


def transpose3(m: mx.array) -> mx.array:
    """Swap the last two axes."""
    return mx.swapaxes(m, -1, -2)


def cross3(a: mx.array, b: mx.array) -> mx.array:
    """Cross product of [..., 3] vectors (broadcasting). Element-wise on purpose: chained
    ``mx.linalg.cross`` calls fail under ``mx.compile`` + grad in MLX 0.32."""
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return mx.stack([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], -1)


# ---------------------------------------------------------------- quaternions [x, y, z, w]

def quat_mul(a: mx.array, b: mx.array) -> mx.array:
    """Hamilton product ``a * b`` (apply b first, then a); [..., 4] each."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return mx.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], -1)


def quat_conj(q: mx.array) -> mx.array:
    """Conjugate (the inverse of a unit quaternion)."""
    return mx.concatenate([-q[..., :3], q[..., 3:]], -1)


def quat_rotate(q: mx.array, v: mx.array) -> mx.array:
    """Rotate vectors v [..., 3] by unit quaternions q [..., 4] (broadcasting)."""
    u, w = q[..., :3], q[..., 3:]
    t = 2 * cross3(u, v)
    return v + w * t + cross3(u, t)


def quat_normalize(q: mx.array, eps: float = 1e-8) -> mx.array:
    """Unit quaternions; eps keeps the gradient finite at q = 0."""
    return q * mx.rsqrt((q * q).sum(-1, keepdims=True) + eps)


def quat_to_mat(q: mx.array) -> mx.array:
    """[..., 4] unit quaternions -> [..., 3, 3] rotation matrices (same formula as scene.quat_to_mat)."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return mx.stack([mx.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     mx.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     mx.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def mat_to_quat(m: mx.array) -> mx.array:
    """[..., 3, 3] rotation matrices -> [..., 4] unit quaternions [x, y, z, w].

    Shepperd's method with the same branch choice as ``scene.mat_to_quat``, written
    branch-free with ``mx.where`` (every branch is finite, so gradients stay finite).
    """
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    tr = m00 + m11 + m22
    s0 = mx.sqrt(mx.maximum(tr + 1.0, 1e-12)) * 2
    q0 = mx.stack([(m21 - m12) / s0, (m02 - m20) / s0, (m10 - m01) / s0, 0.25 * s0], -1)
    s1 = mx.sqrt(mx.maximum(1.0 + m00 - m11 - m22, 1e-12)) * 2
    q1 = mx.stack([0.25 * s1, (m01 + m10) / s1, (m02 + m20) / s1, (m21 - m12) / s1], -1)
    s2 = mx.sqrt(mx.maximum(1.0 + m11 - m00 - m22, 1e-12)) * 2
    q2 = mx.stack([(m01 + m10) / s2, 0.25 * s2, (m12 + m21) / s2, (m02 - m20) / s2], -1)
    s3 = mx.sqrt(mx.maximum(1.0 + m22 - m00 - m11, 1e-12)) * 2
    q3 = mx.stack([(m02 + m20) / s3, (m12 + m21) / s3, 0.25 * s3, (m10 - m01) / s3], -1)
    c0 = tr > 0
    c1 = (~c0) & (m00 > m11) & (m00 > m22)
    c2 = (~c0) & (~c1) & (m11 > m22)
    q = mx.where(c0[..., None], q0, mx.where(c1[..., None], q1, mx.where(c2[..., None], q2, q3)))
    return quat_normalize(q, 1e-12)


# ---------------------------------------------------------------- other parameterisations

def rot6d_to_mat(x: mx.array, eps: float = 1e-8) -> mx.array:
    """Zhou et al. 6D rotation [..., 6] -> [..., 3, 3].

    ``x[..., :3]`` and ``x[..., 3:]`` are (unnormalised) first and second columns;
    Gram-Schmidt gives b1, b2 and b3 = b1 x b2, returned as the matrix columns.
    """
    a1, a2 = x[..., :3], x[..., 3:6]
    b1 = a1 * mx.rsqrt((a1 * a1).sum(-1, keepdims=True) + eps)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 * mx.rsqrt((a2 * a2).sum(-1, keepdims=True) + eps)
    b3 = cross3(b1, b2)
    return mx.stack([b1, b2, b3], -1)


def mat_to_rot6d(m: mx.array) -> mx.array:
    """[..., 3, 3] -> [..., 6]: the first two columns (inverse of rot6d_to_mat for rotations)."""
    return mx.concatenate([m[..., :, 0], m[..., :, 1]], -1)


def axis_angle_to_mat(aa: mx.array, eps: float = 1e-12) -> mx.array:
    """Rodrigues: axis-angle vectors [..., 3] (direction = axis, norm = angle in radians) -> [..., 3, 3]."""
    th2 = (aa * aa).sum(-1, keepdims=True)
    th = mx.sqrt(th2 + eps)
    k = aa / th
    s, c = mx.sin(th)[..., None], mx.cos(th)[..., None]
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = mx.zeros_like(kx)
    K = mx.stack([mx.stack([zero, -kz, ky], -1), mx.stack([kz, zero, -kx], -1), mx.stack([-ky, kx, zero], -1)], -2)
    eye = mx.eye(3, dtype=aa.dtype)
    return eye + s * K + (1 - c) * matmul3(K, K)


def yaw_matrix(yaw: mx.array) -> mx.array:
    """Rotation about +Y by `yaw` radians [...] -> [..., 3, 3] (yaw 0 faces +Z, positive turns +Z toward +X)."""
    c, s = mx.cos(yaw), mx.sin(yaw)
    z, o = mx.zeros_like(yaw), mx.ones_like(yaw)
    return mx.stack([mx.stack([c, z, s], -1), mx.stack([z, o, z], -1), mx.stack([-s, z, c], -1)], -2)


# ---------------------------------------------------------------- kinematics

def rest_joint_positions(offsets, parents=PARENTS):
    """Rest-pose (all rotations identity, root at the origin) joint positions.

    offsets: [J, 3] or [B, J, 3] (numpy or MLX) -> same leading shape [.., J, 3].
    """
    xp = mx if isinstance(offsets, mx.array) else np
    out = []
    for j, p in enumerate(parents):
        o = offsets[..., j, :]
        out.append(o if p < 0 else out[p] + o)
    return xp.stack(out, -2)


def forward_kinematics(pose_rot: mx.array, root_rot: mx.array, root_pos: mx.array,
                       offsets=None, parents=PARENTS):
    """World joint rotations and positions for a batch of skeletons.

    Args:
      pose_rot: [B, J, 3, 3] local joint rotations relative to the rest pose
        (the ``pose`` track of the format, as matrices).
      root_rot: [B, 3, 3] entity rotation; root_pos: [B, 3] entity position.
      offsets: rest offsets from the parent, [J, 3] or [B, J, 3]; defaults to
        ``scene.OFFSETS`` (humanoid22). Scaling a skeleton = scaling its offsets.
      parents: static parent indices (parents precede children).

    Returns:
      (R_world [B, J, 3, 3], p_world [B, J, 3]) with, exactly as
      ``scene.forward_kinematics``::

        R[root] = root_rot @ pose[root];  p[root] = root_pos + root_rot @ offset[root]
        R[j] = R[parent] @ pose[j];       p[j] = p[parent] + R[parent] @ offset[j]
    """
    if offsets is None:
        offsets = mx.array(OFFSETS)
    elif not isinstance(offsets, mx.array):
        offsets = mx.array(np.asarray(offsets, np.float32))
    Rs, ps = [], []
    for j, par in enumerate(parents):
        off = offsets[..., j, :]
        if par < 0:
            Rs.append(matmul3(root_rot, pose_rot[:, j]))
            ps.append(root_pos + matvec3(root_rot, off))
        else:
            Rs.append(matmul3(Rs[par], pose_rot[:, j]))
            ps.append(ps[par] + matvec3(Rs[par], off))
    return mx.stack(Rs, 1), mx.stack(ps, 1)


def _runs(idx: np.ndarray):
    """Split a static index vector into runs: ("const", value, length) or ("range", start, length)."""
    out, i, n = [], 0, len(idx)
    while i < n:
        j = i + 1
        if j < n and idx[j] == idx[i]:
            while j < n and idx[j] == idx[i]:
                j += 1
            out.append(("const", int(idx[i]), j - i))
        else:
            while j < n and idx[j] == idx[j - 1] + 1:
                j += 1
            out.append(("range", int(idx[i]), j - i))
        i = j
    return out


def take_static(x: mx.array, idx, axis: int = 1) -> mx.array:
    """``mx.take(x, idx, axis)`` for a static integer index vector, built from slices and broadcasts.

    Same values, but the backward pass is a fixed-order sum instead of MLX's atomic scatter-add,
    so gradients are bitwise reproducible. Efficient when idx has few runs of equal or
    consecutive values (e.g. per-part joint indices).
    """
    idx = np.asarray(idx, np.int64).reshape(-1)
    ax = axis % x.ndim
    pieces = []
    for kind, a, n in _runs(idx):
        sl = [slice(None)] * x.ndim
        if kind == "const":
            sl[ax] = slice(a, a + 1)
            piece = x[tuple(sl)]
            shape = list(piece.shape)
            shape[ax] = n
            pieces.append(mx.broadcast_to(piece, shape))
        else:
            sl[ax] = slice(a, a + n)
            pieces.append(x[tuple(sl)])
    return pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, ax)


def skin_rigid(local: mx.array, joint_idx, R_world: mx.array, p_world: mx.array) -> mx.array:
    """Rigid skinning (one joint per vertex, weight 1).

    local: [V, 3] or [B, V, 3] vertex positions in the frame of their joint
    (``v_rest - rest_joint``); joint_idx: static int [V]; R_world [B, J, 3, 3];
    p_world [B, J, 3]. Returns world positions [B, V, 3]:
    ``v = R_world[j] @ local + p_world[j]``, which is linear blend skinning with
    the rest pose as bind pose (SCENE_FORMAT.md).
    """
    Rv = take_static(R_world, joint_idx, 1)           # [B, V, 3, 3]
    pv = take_static(p_world, joint_idx, 1)           # [B, V, 3]
    if local.ndim == 2:
        local = local[None]
    return matvec3(Rv, local) + pv
