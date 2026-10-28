# Training data contract

The data pipeline (`restage/data/`) produces packed clips; the loader turns them into fixed-shape
batches for the model (`restage/model/`). This file is the contract between the two.

## Packed clip: `data/<split>/<id>.npz`

All geometry is in the **generator's world frame of each location** (metres, +Y up). Gauge fixing
happens in the loader, because it depends on which keyframes a training window contains.

| key | dtype, shape | meaning |
|---|---|---|
| `frames` | u8 [K, 126, 224, 3] | keyframes, RGB |
| `key_frame` | i32 [K] | video frame index of each keyframe |
| `key_shot` | i32 [K] | shot index of each keyframe |
| `fps`, `n_frames` | i32 scalars | |
| `shot_range` | i32 [S, 2] | `[start, end)` frames of each shot |
| `shot_loc` | i32 [S] | location index of each shot |
| `shot_size`, `shot_move` | i32 [S] | indices into `SHOT_SIZES` / `SHOT_MOVES` below |
| `cam_rot` | f32 [K, 3, 3] | camera-to-world rotation (columns = camera x, y, z axes; camera looks down −Z) |
| `cam_pos` | f32 [K, 3] | camera position |
| `cam_fov` | f32 [K] | vertical field of view, degrees |
| `loc_points` | f32 [L, 2048, 3] | surface samples of each location's static set (props, ground included), area-weighted |
| `loc_sky`, `loc_ground` | f32 [L, 3] | sky colour and dominant ground colour (linear RGB) |
| `ent_class` | i32 [E] | 1 person, 2 animal, 3 vehicle, 4 object |
| `ent_loc` | i32 [E] | location index |
| `ent_present` | u8 [E, K] | state > 0 at the keyframe |
| `ent_visible` | u8 [E, K] | state == 2 at the keyframe |
| `ent_pos` | f32 [E, K, 3] | root position |
| `ent_rot` | f32 [E, K, 4] | root orientation quaternion (x, y, z, w) |
| `ent_pose` | f32 [E, K, 22, 4] | humanoid22 local joint quaternions (identity for non-people) |
| `ent_height` | f32 [E] | standing height in metres (people: rest-pose top of head; objects: bbox height) |
| `ent_size` | f32 [E, 3] | rest-mesh bounding-box extents |
| `ent_colors` | f32 [E, 5, 3] | people: skin, top, bottom, shoes, hair; objects: main colour in slot 0 |
| `depth` | f16 [K, 36, 64] | metric view depth at each keyframe (0 = sky / nothing) |
| `ids` | u8 [K, 36, 64] | 0 = set or background, e + 1 = entity e |
| `thumbs` | u8 [n_frames, 18, 32, 3] | every video frame, tiny, for training the cut detector |
| `meta` | JSON string | genre, seed, source paths, generator version |

Keyframes: 4 per second inside each shot plus every shot's first and last frame.

`SHOT_SIZES = ["extreme_wide", "wide", "medium", "closeup", "extreme_closeup", "insert"]`
`SHOT_MOVES = ["static", "pan", "tilt", "dolly", "truck", "orbit", "handheld", "crane", "drone", "fpv"]`

## Batch from the loader

Fixed shapes, so compiled MLX graphs are reused: `Kmax = 48` keyframes, `Smax = 12` shots,
`Lmax = 4` locations, `Emax = 10` entities, batch `B`. Every array has a matching mask.

A training window is a run of whole consecutive shots whose keyframes fit in `Kmax`. Each location
that appears in the window is **gauge-fixed on its first keyframe in the window**:

* `R0`, `p0`: that keyframe's camera rotation and position;
* yaw `ψ` = heading of the camera's −Z axis projected on the ground plane;
* gauge transform: translate by `(−p0.x, 0, −p0.z)`, then rotate about +Y by `−ψ`, so the anchor
  camera sits at `(0, p0.y, 0)` looking toward −Z, with its pitch and roll unchanged;
* applied to cameras, entity roots and orientations, and `loc_points` of that location.

| key | shape | notes |
|---|---|---|
| `frames` | u8 [B, Kmax, 126, 224, 3] | |
| `key_mask` | bool [B, Kmax] | |
| `key_shot`, `key_time` | i32 [B, Kmax], f32 [B, Kmax] | shot index within the window; seconds from window start |
| `shot_mask`, `shot_loc` | [B, Smax] | location index within the window (0..Lmax-1, in order of first appearance) |
| `loc_mask` | bool [B, Lmax] | |
| `is_anchor` | bool [B, Kmax] | the gauge keyframe of its location |
| `cam_rot`, `cam_pos`, `cam_fov` | [B, Kmax, 3, 3], [B, Kmax, 3], [B, Kmax] | gauge-fixed |
| `loc_points` | [B, Lmax, 2048, 3] | gauge-fixed |
| `loc_sky`, `loc_ground` | [B, Lmax, 3] | |
| `ent_mask` | bool [B, Emax] | entities whose location is in the window and that are present in it |
| `ent_class`, `ent_loc` | [B, Emax] | |
| `ent_present`, `ent_visible` | [B, Emax, Kmax] | |
| `ent_pos`, `ent_rot`, `ent_pose` | [B, Emax, Kmax, 3 / 4 / 22×4] | gauge-fixed |
| `ent_height`, `ent_size`, `ent_colors` | [B, Emax], [B, Emax, 3], [B, Emax, 5, 3] | |
| `depth`, `ids` | [B, Kmax, 36, 64] | after augmentation |

Augmentations (all consistent with the labels): colour jitter; **crop-cuts** (zoom a whole shot by a
random crop, updating `cam_fov`, depth and ids); horizontal flip of the whole clip (mirror x in every
gauge, swap left and right joints).
