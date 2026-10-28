# restage.render: differentiable renderer (MLX + Metal)

The renderer behind the "re-render and compare" loss (docs/DESIGN.md §2.4, §4, §5): the decoder's
scene parameters become fixed-topology meshes (rounded boxes, the humanoid, a ground plane), are
projected through the predicted cameras and soft-rasterised into alpha, depth and per-vertex
attributes (colour, instance probabilities, ...) with gradients to every parameter. MLX only at
runtime; the rasteriser is a set of custom Metal kernels with a hand-written backward.

| file | contents |
|---|---|
| `fk.py` | quaternions, 6D rotations, axis-angle, yaw; `forward_kinematics` on humanoid22; rigid skinning |
| `camera.py` | batched pinhole projection, `look_at` |
| `shapes.py` | `rounded_boxes`, `humanoid`, `ground_plane`, mesh merging |
| `raster.py` | `rasterize` (Metal) and `rasterize_reference` (dense plain MLX) |
| `export.py` | the same meshes as numpy for `.restage` files; `preview_image` |
| `bench_render.py` | timing and memory on the realistic training workload |
| `../../tests/test_render.py` | 20 tests (see below) |

## Conventions

* Metres, right-handed, **+Y up**; characters and their rest pose face **+Z** (SCENE_FORMAT.md).
* Rotation matrices act on column vectors; the columns of a camera or joint rotation are its local
  x, y, z axes in world space. Quaternions are `[x, y, z, w]`. The 6D representation is the first
  two matrix *columns* `[R[:, 0], R[:, 1]]` (Zhou et al.).
* All 3x3 products are element-wise multiply-and-sum (`fk.matmul3`), exact fp32 even though MLX
  runs fp32 `@` as TF32 on the M5.
* **Cameras**: `cam_R` [B,3,3] camera-to-world, `cam_p` [B,3], vertical `fov_deg` [B],
  `aspect = W / H`. The camera looks down its local -Z, +Y is up in the image (three.js / glTF).
  `project` returns `[ndc_x, ndc_y, depth]` with NDC in [-1, 1] (+x right, +y up) and positive view
  depth; identical to `restage.gen.cinema.project` (tested).
* **Pixels**: `X = (ndc_x + 1) W / 2`, `Y = (1 - ndc_y) H / 2`; pixel (row i, column j) has its centre
  at (j + 0.5, i + 0.5). `sigma` is measured in these pixels.
* **Faces** are counter-clockwise seen from the front (three.js front faces), int32 numpy [F, 3],
  shared by the whole batch.
* **FK** (identical to `scene.forward_kinematics`): `R[root] = root_rot @ pose[root]`,
  `p[root] = root_pos + root_rot @ offset[root]`; `R[j] = R[parent] @ pose[j]`,
  `p[j] = p[parent] + R[parent] @ offset[j]`. Skinning is rigid LBS with the rest pose as bind pose:
  `v = R[j] @ (v_rest - rest_j) + p[j]`.

## API

```python
from restage.render import fk, camera, shapes, raster, export

# fk.py
fk.quat_mul(a, b); fk.quat_rotate(q, v); fk.quat_normalize(q); fk.quat_conj(q)
fk.quat_to_mat(q) -> [..., 3, 3];  fk.mat_to_quat(m) -> [..., 4]
fk.rot6d_to_mat(x6) -> [..., 3, 3];  fk.mat_to_rot6d(m) -> [..., 6]
fk.axis_angle_to_mat(aa); fk.yaw_matrix(yaw); fk.cross3(a, b); fk.matmul3(a, b); fk.matvec3(m, v)
fk.forward_kinematics(pose_rot [B,22,3,3], root_rot [B,3,3], root_pos [B,3],
                      offsets [22,3] | [B,22,3] = scene.OFFSETS, parents=scene.PARENTS)
    -> (R_world [B,22,3,3], p_world [B,22,3])
fk.rest_joint_positions(offsets) -> [.., 22, 3]
fk.skin_rigid(local [V,3]|[B,V,3], joint_idx [V], R_world, p_world) -> [B,V,3]
fk.take_static(x, idx, axis=1)            # mx.take with a reproducible (scatter-free) backward

# camera.py
camera.project(points [B,N,3], cam_R [B,3,3], cam_p [B,3], fov_deg [B], aspect=16/9) -> [B,N,3]
camera.world_to_camera(points, cam_R, cam_p); camera.look_at(eye [B,3], target [B,3], up=(0,1,0))
camera.ndc_to_pixels(xy, H, W); camera.camera_from_quat(q); camera.focal_from_fov; camera.fov_from_focal

# shapes.py: every builder returns (verts [B,V,3] MLX, faces [F,3] int32 numpy, meta dict)
shapes.rounded_boxes(center [B,N,3], size [B,N,3], yaw [B,N], roundness [B,N], n=3)
    meta: "box" [V], "face_id" [F], "n_vert_per_box", "n_face_per_box"
shapes.humanoid(shape [B,4], joint_rot [B,22,3,3], root_rot [B,3,3], root_pos [B,3])
    meta: "joint" [V], "region" [V] (0 skin, 1 top, 2 bottom, 3 shoes, 4 hair), "part" [V],
          "part_names", "joint_rot", "joint_pos" (posed joints)
shapes.ground_plane(center [B,3], extent=60, n=16, power=2.5, y=0)
shapes.merge_meshes((v1, f1), (v2, f2), ...) -> (verts, faces, vertex_offsets)
shapes.tile_faces(faces, n, n_vert); shapes.humanoid_offsets(shape); shapes.humanoid_rest_numpy(shape)

# raster.py
raster.rasterize(verts [B,V,3] (ndc_x, ndc_y, depth), faces [F,3], attrs [B,V,C] | None = None,
                 opacity [B,V] | None = None, *, height, width, sigma=0.5, gamma=0.05, near=0.05,
                 cutoff=1e-3, cull_backfaces=True, alpha_mode="sum", opacity_cull=-1.0,
                 max_entries=None, fast_math=None)
    -> RasterOutput(alpha [B,H,W], depth [B,H,W], attrs [B,H,W,C] | None, n_entries, capacity)
raster.rasterize_reference(...same arguments...) -> (alpha, depth, attrs)   # dense, small cases only
raster.default_capacity(B, F, H, W); raster.cut_distance(sigma, cutoff)

# export.py (numpy, for .restage documents; restage.scene.save writes them)
export.to_numpy_mesh(positions, faces, colors=None, mesh_id, face_mask=None, joints=None, weights=None)
export.rounded_boxes_to_numpy(verts_of_one_job, faces, meta, colors [N,3], present [N]) -> mesh
export.humanoid_to_numpy(shape [4], region_colors [5,3], mesh_id) -> (mesh, skeleton)
export.pose_track(joint_rot [T,22,3,3], root_rot [T,3,3], root_pos [T,3]) -> {"pose", "rotation", "position"}
export.camera_track(cam_R [T,3,3], cam_p [T,3], fov [T]) -> {"position", "rotation", "fov"}
export.skeleton_doc(offsets, sid, scale); export.region_colors(region, colors)
export.ground_to_numpy(extent, n, color); export.scene_doc(meshes, props, entities, cameras, n_frames)
export.preview_image(verts, faces, colors, cam_R, cam_p, fov, H, W) -> uint8 [H, W, 3]
```

Typical training use:

```python
bv, bf, bm = shapes.rounded_boxes(c, s, yaw, rnd)                       # the set
hv, hf, hm = shapes.humanoid(shape, rot6d_to_mat(j6), root_R, root_p)   # [B*P, 393, 3] people
hv = hv.reshape(B, P * 393, 3); hf = shapes.tile_faces(hf, P, 393)
gv, gf, _ = shapes.ground_plane(cam_p)
verts, faces, offs = shapes.merge_meshes((gv, gf), (bv, bf), (hv, hf))
ndc = camera.project(verts, cam_R, cam_p, fov, 64 / 36)
opacity = ...  # per vertex: box presence repeated 56x, person presence repeated 393x, 1 for ground
out = raster.rasterize(ndc, faces, attrs, opacity, height=36, width=64)
loss = bce(out.alpha, gt_mask) + (gt_mask * abs(out.depth - gt_depth)).mean() \
     + ce(out.alpha[..., None] * out.attrs, gt_instance)               # attrs = instance one-hot
```

Exporting the same meshes: `rounded_boxes_to_numpy(np.array(bv[b]), bf, bm, colors, present)` for
the set, and for each person `humanoid_to_numpy(shape[k], region_colors)` (rest-pose mesh with
`joints` / `weights` in humanoid22 order and its own skeleton; offsets include the height and
shoulder width, `"base": "humanoid22"` as the generator writes it) plus
`pose_track(joint_rot, root_rot, root_pos)` per frame. Viewer-side LBS of that file reproduces
`shapes.humanoid` to about 3e-7 m (tested with tolerance 1e-5).

## Templates

| mesh | vertices | triangles |
|---|---|---|
| rounded box, n = 3 (6 x 3 x 3 quads projected onto the superellipsoid of exponent p = 2 / roundness) | 56 | 108 |
| humanoid (21 parts: tapered octagonal prisms, box hands / feet / toes, 3x3 rounded-box head with hair faces) | 393 | 672 |
| ground plane, n = 16 | 289 | 512 |

Humanoid shape `[height, girth, head, shoulder]`, 1 = average (about 1.78 m tall): height scales the
skeleton offsets and the whole body, girth limb and torso thickness, head the head, shoulder the
chest width and the clavicle / shoulder offsets. Every part is rigid on one joint (weight 1); the
clavicles carry no geometry. Regions: skin (neck, face, forearms, hands), top (torso, upper arms),
bottom (pelvis, legs), shoes (feet, toes), hair (top / back of the head; its own vertices, so
per-vertex colours stay flat).

## Rendering model

Per pixel p and triangle f (pixel units): `sd` = exact signed distance to the triangle (+ inside);
coverage `D = sigmoid(sd / sigma)`, ignored below `cutoff` (bounded support, so binning is exact);
barycentrics are the pixel's own inside and those of the closest boundary point outside, then
perspective-corrected with the vertex depths; `zp`, opacity `op` and attributes are interpolated with
them. With `x = op * D` and `w = x * exp(-zp / gamma)`:

* `alpha = min(1, sum x)` (default; does not dip along the shared edges of a tessellated surface,
  unlike SoftRas' `1 - prod(1 - x)`, available as `alpha_mode="prob"`);
* `depth = sum w zp / sum w`, `attrs = sum w a / sum w`: soft z-buffer expectations *given a surface*
  (0 where nothing is in range). Use `alpha * attrs` (plus `1 - alpha` background) for compositing.

`gamma` is in metres (0.05 by default: surfaces 10 cm apart blend 1 : e^2). Opacity enters the
softmax weights, so a part with opacity 0 neither shows nor occludes, but gets the fade-in gradient.

Kernels: `setup` (one thread per job x face: culling, padded bbox -> rectangle of 8x4 tiles, packed
vertices) -> cumsum of per-face tile counts gives every (face, tile) entry a slot -> `bin_count` /
`bin_fill` (one SIMD group per job x tile, ballots over the job's faces; lists in face order) ->
`forward` (one SIMD group = one tile, one lane = one pixel; online log-sum-exp) -> `backward` (same
layout, recomputes each pair; the 32 lanes' gradients per entry are summed with a shuffle
reduce-scatter and written to the entry's slot) -> `gather` (per vertex, sums the slots of its
incident face corners through a static CSR table). No atomics anywhere: forward and gradients are
bitwise reproducible, and so is the whole pipeline (`fk.take_static`, one-hot channel sums).
Culling: triangles with a vertex at depth <= `near`, non-finite coordinates, zero area or (by
default) facing away are dropped in `setup`; nothing NaN reaches the outputs or the gradients.

## Performance (Apple M5, 10-core GPU, MLX 0.32.4, fp32, Metal fast math)

Workload (`bench_render.py`): B = 64 views, each with a ground plane, 24 rounded boxes, 8 humanoid
slots (some absent) and 2 object boxes: **8696 triangles, 4889 vertices per view**, C = 16 attributes
(RGB + 11-way instance one-hot + 2 spare), L1/BCE-style loss on all three outputs.

The machine was shared with other agents during these runs (an fp16 GEMM probe gave 10.8 to 12.1 of
the idle 14 TFLOPS), so medians wander by up to 2x between runs; minima are the better estimate.

| 64 x 36 | min | median | peak memory |
|---|---|---|---|
| raster forward (incl. binning) | 5.3–6.8 ms | 5.8–7.5 ms | 0.13 GB |
| raster forward + backward | **21.3–27 ms** | 21.6–30 ms | 0.64 GB |
| full step: params -> meshes -> FK -> projection -> raster -> loss, grads to all params | 31–54 ms | 31–74 ms | 0.84 GB |
| same, under `mx.compile` | 36 ms | 58 ms | |

| 128 x 72 | min | median | peak memory |
|---|---|---|---|
| raster forward | 9.4–12.3 ms | 9.5–14 ms | 0.26 GB |
| raster forward + backward | **31.4–45 ms** | 32–51 ms | 1.37 GB |
| full step | 41–71 ms | 41–88 ms | 1.54 GB |

Per-kernel minima at 64 x 36 / 128 x 72 (ms): setup 0.6 / 0.6, bin_count 0.6 / 2.1, bin_fill 0.9 /
2.8, forward 3.7–4.5 / 4.5, backward 12.4–13.3 / 15–16, gather 2.9–3.2 / 3.9–4.1
(`--stages`). The (face, tile) lists hold 1.04 M entries at 64 x 36 (57 % of the default capacity)
and 1.37 M at 128 x 72 (35 %). For comparison, the research prototype (one thread per pixel looping
over all triangles, no binning) needed 12.8 ms forward + backward for *one* 128 x 128 view of 10k
triangles; this is 64 views of 8.7k triangles in about 22–32 ms.

What the time goes into, from experiments during development: Metal fast math saves 20 % (backward)
to 40 % (forward); replacing per-value `simd_sum`s by a recursive-halving reduce-scatter with
coalesced stores saved 20 % of the backward; on average only 11.5 of the 32 lanes of an entry are
active (small triangles) and 10 % of entries have none, so tighter binning would not pay; going from
4 to 16 attribute channels adds about 9 ms to forward + backward (mostly the per-entry partial
writes, 250 MB); a simdgroup-matrix version of the attribute reduction gained only 7 % of the
backward kernel and was removed.

## Tests

`.venv/bin/python -m pytest tests/test_render.py -q` (20 tests, about 2–6 s):
quaternion / 6D / axis-angle helpers vs `restage.scene`; FK parity with `scene.forward_kinematics`
(shared and per-batch offsets, 2e-5); projection parity with `restage.gen.cinema.project` and the
-Z / +Y conventions; rounded boxes (counts, closed and outward, sharp-box volume, exact ellipsoid at
roundness 1, finite roundness gradients); ground plane; humanoid (672 triangles, 1.78 m x height,
soles on y = 0, closed parts, shape parameters); export round trip (`.restage` written and loaded,
file LBS == MLX humanoid to 1e-5, PNGs through `preview_image` and `restage.gen.render`, region
ordering head > top > bottom > shoes); rasteriser forward and gradients vs the dense reference (4
configurations + a shared-vertex scene: forward 1e-4, gradients 1e-3 relative); central finite
differences on a tiny case; no NaNs with triangles crossing / behind the near plane; bitwise
determinism; capacity overflow; `mx.compile`; zero-opacity occluders; compiled end-to-end gradients
(finite, eager == compiled, reproducible).

## Limitations

* Near-plane triangles are **dropped, not clipped**. Large triangles that cross the near plane
  vanish; `ground_plane` centred on the camera keeps the lost cells under the camera.
* The (face, tile) list has a static capacity (`max_entries`, default `default_capacity`); if
  `n_entries > capacity` the faces beyond it (last jobs first) are not rendered. Check
  `n_entries` occasionally; the backward allocates `capacity * 12 * (ceil(C/4) + 1)` floats.
* `alpha_mode="sum"` saturates at 1, so two stacked semi-transparent layers give `min(1, a+b)`
  rather than `1 - (1-a)(1-b)`; the softmax-normalised depth / attributes jump to 0 where the
  last triangle leaves the `cutoff` (alpha is below `cutoff` there).
* The fade-in gradient of a near, almost transparent surface is clamped (`exp(30)` factor) to stay
  finite.
* Up to 32 attribute channels (padded to a multiple of 4), images up to 2040 x 1020, fp32 only.
  Fast math changes results by about 1e-5 relative (`fast_math=False` for safe math).
* No per-face attributes: duplicate vertices for flat colours (as `preview_image` does).
* The 8x4 tiles leave most lanes idle on tiny triangles; for scenes with many sub-pixel triangles
  a coarser mesh (or `cutoff=1e-2`) is cheaper.
