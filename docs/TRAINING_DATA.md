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

## Clarifications and additions (from the data pipeline)

None of these change a key, shape or meaning above; they pin down details and add optional keys.

**Packed clip**

* Extra keys: `ent_ident` i32 [E], entities with the same value are the same person or object
  (the scene's `identity`: one entity per location visited, or a wardrobe copy). `meta` also holds
  `cam_jumps` (frames where the generator's camera jumps inside a shot; see
  `restage/data/GEN_REQUESTS.md`), `continuity` (number of injected errors), `locations`, `entities`.
* `ids`: the generator's prop ids (1000 + k) are set, so they map to 0. At 64×36, `ids` is the sample
  at the pixel centre of the aux render (320×180, an exact 5×5 block), and `depth` is the lower median
  of that block's pixels of the same class (sky, set, or that entity): always a real surface depth,
  never a blend across an edge.
* `loc_points` exclude the `sky` / `cloud` prop classes and distant scenery that reaches more than 20%
  beyond the location bounds (the mountain ring), and are restricted to the filmed region: x/z within
  30 m of the location's camera and entity positions (whole indoor sets fit inside it).
* `loc_ground`: area-weighted dominant colour of the up-facing ground surfaces (floor, terrain, road,
  sidewalk, stage, voxel tops, …) in that region; the environment's hemisphere `ground` if there are none.
* People: the root (`ent_pos`) is on the ground below the pelvis (for a seated person, below the
  seat). `ent_height` is the rest-pose top of the skull (skin-coloured head vertices; hair and hats
  excluded). `ent_size` is the rest (T-pose) bounding box, so x includes the arm span.
  `ent_colors`: skin = neck and hands, top = front of the torso, bottom = pelvis and thighs, shoes =
  ankles and toes, hair = the top of the head (a hat if worn, the skin colour if bald). Objects and
  vehicles: dominant colour in slot 0, the other slots 0.
* `ent_pose` of vehicles (which have their own skeleton) and animals is identity.

**Batch**

* dtypes: masks and `ent_present` / `ent_visible` are bool; `frames` and `ids` u8; indices i32;
  everything else f32 (including `depth`). Padding: zeros, except identity `cam_rot`, quaternions
  (0, 0, 0, 1) and `cam_fov` = 50 (so log-fov losses stay finite under the mask).
* A single shot with more than Kmax keyframes becomes a window of Kmax consecutive keyframes of it.
* More than Emax entities: the ones visible in the most keyframes are kept; pixels of the others
  have `ids` = 255 (ignore in losses).
* Crop-cuts are exact virtual cameras at the same position: zoom z ∈ [1.15, 2] (fov' = 2 atan(tan(fov/2)/z))
  plus a small yaw/pitch toward an off-centre crop, so `cam_rot` changes as well as `cam_fov`. Frames
  are warped by the homography, `ids` / `depth` re-sampled (depth re-expressed along the new axis), and
  an entity that leaves the frame loses `ent_visible`. Crops happen before gauge fixing, so a cropped
  anchor keyframe still sits at (0, y, 0) looking toward −Z.
* The anchor heading uses the camera's −Z; for a camera looking straight down it uses its +Y.
* Mirror: M = diag(−1, 1, 1) in every gauge; cameras R → M R M, positions → M p, quaternions
  (x, y, z, w) → (x, −y, −z, w), and `ent_pose[j]` ← reflect(`ent_pose[swap(j)]`) with left/right joints
  swapped; frames, depth and ids flip horizontally.
* Extra keys: `key_frame` [B, Kmax] (video frame), `shot_size`, `shot_move` [B, Smax], `shot_zoom`
  [B, Smax] (crop-cut zoom, 1 = none), `ent_ident` [B, Emax] (window-local identity index), `mirrored` [B],
  `clip_id` [B] (index into the loader's file list), `batch_mask` [B] (false for repeats that fill the
  last batch in epoch mode).
