# Format notes from the generator

The generator writes documents that follow `docs/SCENE_FORMAT.md` (version 1) exactly; readers
that ignore unknown keys load them unchanged. Where the format could not express something, the
generator works around it as described below. Each item ends with a proposal.

## 1. A person who visits several locations

An entity has one `location`, but a hiker walks from a forest into a mountain set, and a vlogger
can cut to b-roll outside. The generator writes one entity per (person, location). The copies
share the same mesh and an extra key `"identity": "hiker0"`, and each copy has `state` 0 while
the edit is in another location. A wardrobe change (see item 3) uses the same mechanism.

**Proposal:** make `identity` an official optional entity key. Alternatively, add a per-frame
`track.location` (u8 [T], an index into `locations`) so one entity can move between sets.

## 2. Body proportions: a skeleton per person

People have different heights. Each person has their own skeleton (`"id": "sk_e0"`) with
humanoid22's joint names and order, and offsets scaled by the person's height. The skeleton has
two extra keys: `"base": "humanoid22"` and `"scale": 0.97`. The viewer already reads offsets
from the document, so this works today.

**Proposal:** state in the format that a skeleton with `"base": "humanoid22"` has humanoid22's
joint order and names, and may scale the offsets.

## 3. Wardrobe and other appearance changes between cuts

Mesh colours are fixed per entity. A wardrobe continuity error is therefore written as a second
entity: a recoloured copy with the same `identity` and the same track. The original has `state` 0
from the cut onward and the copy has `state` 0 before it. The continuity note names the
original and carries `"replacement": "<copy id>"`.

**Proposal:** add optional `mesh_variants: [mesh ids]` on an entity, plus `track.variant`
(u8 [T]).

## 4. Time skips (ellipsis) between shots

Hiking, driving and gameplay edits skip time at some cuts. Entity tracks are in video time, so a
skip shows up as a jump in the track at the cut. That is correct, but implicit. Every shot carries
`"world_frame"`: the frame on a continuous world clock at which the shot starts. Shots after a
skip also carry `"time_skip"`: the seconds skipped since the previous shot in the same location.

**Proposal:** make `shot.world_start` (seconds or frames) an official optional key, so that tools
can tell a time skip from an error.

## 5. Emissive parts of entities

A mesh has one material. Props split their glowing parts (lamp bulbs, TV screens, windows,
traffic lights) into separate props that use emissive materials `m_glow_rrggbb`. An entity can
reference only one mesh, so car head and tail lights and phone screens on entities are merged in
as bright vertex colours and do not glow.

**Proposal:** mesh `primitives` (sub-meshes, each with its own material), or an optional per-vertex
`emissive` attribute.

## 6. Shadow casting

The viewer casts sun shadows from every mesh. Because three.js renders back faces into the shadow
map, the single-sided room shells (inward-facing walls and ceiling) would put the whole interior
into shadow. Distant backdrop mountains and clouds should not cast shadows either. The affected
props carry `"cast_shadow": false`. The generator's renderer respects this key, and the viewer
can adopt it with `mesh.castShadow = prop.cast_shadow ?? true`.

**Proposal:** add `cast_shadow` (bool, default true) to props.

## 7. Environment keys

- `ground`: the lower colour of the hemisphere light. The viewer already reads it, with a default
  of [0.35, 0.3, 0.25].
- `fog`: `{"color", "near", "far"}`, linear like `THREE.Fog`.
- `time_of_day` is informational.

**Proposal:** document `ground` and the fog fields in SCENE_FORMAT.md.

## 8. Extra keys the generator writes (all optional, safe to ignore)

| where | key | meaning |
|---|---|---|
| meta | `genre`, `music: {bpm, beat0}` | generator genre; dance tempo and the time of the first beat (s) |
| shot | `subject` | id of the entity the shot frames |
| shot | `attached_to` | first-person shot: the camera is mounted on this entity (head or dashboard) |
| shot | `focal_mm` | 35 mm-equivalent focal length matching the mean vertical fov (16:9 crop) |
| shot | `world_frame`, `time_skip` | see item 4 |
| entity | `identity` | see items 1 and 3 |
| track | `contacts` | u8 [T, 2] foot contact (left, right) for humanoids; ground truth for foot locking |
| skeleton | `base`, `scale` | see item 2 |
| prop | `cast_shadow` | see item 6 |
| continuity | `replacement` | see item 3 |

## 9. Vehicle skeletons

Cars and trucks use a 6-joint skeleton per entity:

- `chassis`: the root, at ground level.
- `body`: roll and pitch on the suspension.
- `wheel_fl`, `wheel_fr`, `wheel_rl`, `wheel_rr`: spin, and steering on the front wheels.

The mesh is rigidly skinned to it. Bikes use `frame`, `body`, `wheel_f`, `wheel_r` and `crank`.
This is valid in v1.

**Proposal:** standardise the joint names (for example `vehicle6`), so that the model can predict
wheel and steering state for any car.

## 10. Auxiliary ground truth (outside the format)

`<name>_aux.npz` is generator output, not part of `.restage`. Its instance ids are:

| id | meaning |
|---|---|
| 0 | sky, and set structure (floor, walls, terrain, roads, markings) |
| 1..N | entities in document order |
| 1000 + k | prop k of the current location |
