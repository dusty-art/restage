"""Differentiable renderer for Restage (MLX + custom Metal kernels).

Modules:
  fk      rotations (quaternions, 6D, matrices) and forward kinematics on humanoid22
  camera  batched pinhole projection (three.js / glTF convention)
  shapes  fixed-topology meshes: rounded boxes, the humanoid template, ground plane
  raster  soft rasteriser (Metal forward and backward kernels) and its dense reference
  export  the same meshes as numpy for .restage documents

See README.md in this directory for conventions and measured performance.
"""
from .camera import look_at, project
from .export import humanoid_to_numpy, mesh_to_numpy, pose_track, rounded_boxes_to_numpy, to_numpy_mesh
from .fk import forward_kinematics, mat_to_quat, quat_to_mat, rot6d_to_mat
from .raster import RasterOutput, rasterize, rasterize_reference
from .shapes import ground_plane, humanoid, merge_meshes, rounded_boxes

__all__ = ["look_at", "project", "humanoid_to_numpy", "mesh_to_numpy", "pose_track", "rounded_boxes_to_numpy",
           "to_numpy_mesh", "forward_kinematics", "mat_to_quat", "quat_to_mat", "rot6d_to_mat", "RasterOutput",
           "rasterize", "rasterize_reference", "ground_plane", "humanoid", "merge_meshes", "rounded_boxes"]
