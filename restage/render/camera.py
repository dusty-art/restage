"""Pinhole cameras for a batch of views (MLX, differentiable).

Convention (docs/SCENE_FORMAT.md, three.js / glTF):

* ``cam_R`` [B, 3, 3] is the camera-to-world rotation; its columns are the
  camera's x (right), y (up) and z (backward) axes in world coordinates;
* ``cam_p`` [B, 3] is the camera position in world coordinates;
* the camera looks down its local -Z axis, +Y is up in the image;
* ``fov_deg`` [B] is the *vertical* field of view in degrees and ``aspect`` is
  width / height of the image (a float, or [B]).

Projection (identical to ``restage.gen.cinema.project``)::

    c      = cam_R^T (X - cam_p)                 camera-space point
    depth  = -c_z                                positive in front of the camera
    ndc_x  = c_x / depth / (tan(fov/2) * aspect)  in [-1, 1] across the width, +x right
    ndc_y  = c_y / depth / tan(fov/2)             in [-1, 1] across the height, +y up

``project`` returns ``[ndc_x, ndc_y, depth]`` stacked on the last axis, which is
what ``raster.rasterize`` takes. Points at or behind the camera get their NDC
computed with depth clamped to ``eps`` (so values and gradients stay finite);
their true (non-positive) depth is returned, and the rasteriser drops every
triangle with a vertex closer than its near plane.
"""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from .fk import cross3, quat_to_mat

__all__ = ["world_to_camera", "project", "look_at", "focal_from_fov", "fov_from_focal",
           "ndc_to_pixels", "camera_from_quat"]


def world_to_camera(points: mx.array, cam_R: mx.array, cam_p: mx.array) -> mx.array:
    """World points [B, N, 3] -> camera-space points [B, N, 3] (``R^T (X - p)``)."""
    rel = points - cam_p[:, None, :]
    # (R^T rel)_i = sum_j R_ji rel_j
    return (cam_R[:, None, :, :] * rel[..., :, None]).sum(-2)


def project(points: mx.array, cam_R: mx.array, cam_p: mx.array, fov_deg, aspect=16 / 9,
            eps: float = 1e-4) -> mx.array:
    """Project world points for a batch of cameras.

    Args:
      points: [B, N, 3] world points; cam_R [B, 3, 3]; cam_p [B, 3];
      fov_deg: [B] (or scalar) vertical fov in degrees; aspect: width / height.
    Returns:
      [B, N, 3]: NDC x, NDC y, view depth (metres along the viewing axis).
    """
    c = world_to_camera(points, cam_R, cam_p)
    depth = -c[..., 2]
    fov = mx.array(fov_deg, dtype=mx.float32) if not isinstance(fov_deg, mx.array) else fov_deg
    tv = mx.tan(fov * (math.pi / 360.0))
    tv = mx.reshape(tv, (-1, 1)) if tv.ndim else tv
    asp = aspect if not isinstance(aspect, mx.array) else mx.reshape(aspect, (-1, 1))
    inv = 1.0 / mx.maximum(depth, eps)
    x = c[..., 0] * inv / (tv * asp)
    y = c[..., 1] * inv / tv
    return mx.stack([x, y, depth], -1)


def ndc_to_pixels(xy: mx.array, height: int, width: int) -> mx.array:
    """NDC [..., 2] -> continuous pixel coordinates (x right, y down; pixel (i, j) has its centre at
    (j + 0.5, i + 0.5)), the frame the rasteriser's sigma is measured in."""
    return mx.stack([(xy[..., 0] + 1) * (0.5 * width), (1 - xy[..., 1]) * (0.5 * height)], -1)


def look_at(eye: mx.array, target: mx.array, up=(0.0, 1.0, 0.0), eps: float = 1e-9) -> mx.array:
    """Camera-to-world rotations [B, 3, 3] whose -Z axis points from eye [B, 3] toward target [B, 3].

    Same construction as ``scene.look_rotation`` (no roll; x = up x z).
    """
    f = target - eye
    f = f * mx.rsqrt((f * f).sum(-1, keepdims=True) + eps)
    z = -f
    upv = mx.broadcast_to(mx.array(np.asarray(up, np.float32)), z.shape)
    x = cross3(upv, z)
    x = x * mx.rsqrt((x * x).sum(-1, keepdims=True) + eps)
    y = cross3(z, x)
    return mx.stack([x, y, z], -1)


def camera_from_quat(q: mx.array) -> mx.array:
    """Camera rotation quaternion [..., 4] ([x, y, z, w], as stored in .restage camera tracks) -> [..., 3, 3]."""
    return quat_to_mat(q)


def focal_from_fov(fov_deg, height: float = 1.0):
    """Focal length in pixels for an image of `height` pixels (vertical fov in degrees)."""
    return 0.5 * height / np.tan(np.radians(fov_deg) / 2)


def fov_from_focal(f_px, height: float = 1.0):
    """Inverse of focal_from_fov."""
    return np.degrees(2 * np.arctan(0.5 * height / np.asarray(f_px, float)))
