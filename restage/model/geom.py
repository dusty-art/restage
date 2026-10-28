"""Rotation and gauge maths for the model (MLX). Conventions follow docs/SCENE_FORMAT.md:
+Y up, cameras look down their −Z axis, rotation matrices are camera/entity-to-world (columns = local axes),
quaternions are (x, y, z, w)."""
from __future__ import annotations

import mlx.core as mx


def rot6d_to_mat(x):
    """Zhou et al. 6D representation [..., 6] → rotation matrix [..., 3, 3] (columns = first two Gram–Schmidt axes)."""
    a, b = x[..., :3], x[..., 3:]
    a = a / (mx.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b - (a * b).sum(-1, keepdims=True) * a
    b = b / (mx.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    c = mx.linalg.cross(a, b)
    return mx.stack([a, b, c], axis=-1)


def mat_to_rot6d(m):
    return mx.concatenate([m[..., :, 0], m[..., :, 1]], axis=-1)


def quat_to_mat(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return mx.stack([
        mx.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        mx.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        mx.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def geodesic(r1, r2):
    """Angle (radians) between rotation matrices [..., 3, 3]."""
    tr = (r1 * r2).sum((-2, -1))                       # trace(r1ᵀ r2)
    c = mx.clip((tr - 1) / 2, -1 + 1e-6, 1 - 1e-6)
    return mx.arccos(c)


def rot_y(angle):
    """Rotation about +Y by `angle` [...] → [..., 3, 3]."""
    c, s = mx.cos(angle), mx.sin(angle)
    z, o = mx.zeros_like(angle), mx.ones_like(angle)
    return mx.stack([mx.stack([c, z, s], -1), mx.stack([z, o, z], -1), mx.stack([-s, z, c], -1)], -2)


def heading(rot):
    """Yaw of a camera's view direction (−Z axis) on the ground plane; 0 = looking toward −Z."""
    f = -rot[..., :, 2]
    return mx.arctan2(-f[..., 0], -f[..., 2])


def gauge_from_anchor(anchor_rot, anchor_pos):
    """Gauge transform (G, t) with x' = G (x + t): anchor at (0, y, 0) looking toward −Z.

    anchor_rot [..., 3, 3], anchor_pos [..., 3] → G [..., 3, 3], t [..., 3].
    """
    psi = heading(anchor_rot)
    G = rot_y(-psi)
    t = mx.stack([-anchor_pos[..., 0], mx.zeros_like(anchor_pos[..., 1]), -anchor_pos[..., 2]], -1)
    return G, t


def apply_gauge_points(G, t, x):
    """x [..., N, 3] with G [..., 3, 3], t [..., 3] broadcast over N."""
    return ((x + t[..., None, :]) @ G.swapaxes(-1, -2))


def apply_gauge_rot(G, r):
    return G @ r
