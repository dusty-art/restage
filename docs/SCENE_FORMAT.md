# Restage scene format (`.restage`, version 1)

A `.restage` file is the complete 3D replica of one video. It is what the model
outputs, what the procedural generator writes as ground truth, and what the
viewer plays. It follows how a film is actually made, not one continuous world:

* **locations**: one static set per place the video visits (a kitchen, a trail, a road, a Minecraft area);
* **entities**: every person, animal, vehicle or object, each with one identity
  for the whole video, its own mesh, optional skeleton, and a per-frame track;
* **shots**: contiguous frame ranges of the video, each filmed by one camera in one location;
* **cameras**: one per shot, with a per-frame position, orientation and field of view;
* **continuity**: notes on what changed across cuts that should not have.

Playing every shot through its camera, in order, reproduces the video. Any
other camera shows the same scene from a new angle.

## Conventions

| | |
|---|---|
| units | metres |
| axes | right-handed, **+Y up**; cameras look down their local **−Z** (glTF / three.js) |
| rotations | unit quaternions `[x, y, z, w]` |
| colours | linear RGB, 0–1 |
| time | integer video frames at `fps`, frame 0 = first frame; ranges are `[start, end)` |
| field of view | vertical, degrees |

## Container

Little-endian binary, so a whole replica is one file:

```
offset  size  content
0       4     magic "RSTG"
4       4     u32 version (1)
8       4     u32 J = byte length of the JSON chunk (padded with spaces to a multiple of 4)
12      J     UTF-8 JSON document (below)
12+J    4     u32 B = byte length of the binary chunk
16+J    B     binary chunk: all arrays, each starting on a 4-byte boundary
```

Arrays are referenced from the JSON by **accessors**:

```json
{"offset": 4096, "dtype": "f32", "shape": [300, 3]}
```

`dtype` is one of `f32 f16 u32 u16 u8 i16`; `offset` is relative to the start of
the binary chunk and divisible by 4; the byte length is `prod(shape) × itemsize`.

## JSON document

```jsonc
{
  "format": "restage.scene",
  "version": 1,
  "fps": 30,
  "n_frames": 900,
  "meta": {
    "title": "kitchen chat",
    "source": {"video": "clip.mp4", "width": 1280, "height": 720},   // optional
    "producer": "restage-gen 0.1 (ground truth)" ,                    // or the model + checkpoint
    "seed": 12
  },
  "materials": [ {"id": "m0", "color": [0.8, 0.7, 0.6], "roughness": 0.9, "metalness": 0.0,
                  "emissive": [0, 0, 0], "flat": true} ],
  "meshes": [ {
    "id": "mesh_counter",
    "positions": ACC,          // f32 [V, 3]
    "indices":   ACC,          // u32 or u16 [F, 3]
    "normals":   ACC,          // optional f32 [V, 3]; absent = flat shading from faces
    "colors":    ACC,          // optional f32 [V, 3] or u8 [V, 4]; multiplies the material colour
    "uvs":       ACC,          // optional f32 [V, 2]
    "joints":    ACC,          // optional u8/u16 [V, 4], indices into the entity's skeleton
    "weights":   ACC,          // optional f32 [V, 4]
    "morphs":    [ {"name": "jawOpen", "positions": ACC} ],   // optional deltas f32 [V, 3]
    "material":  "m0",
    "observed":  ACC           // optional u8 [V]: 1 = seen in the video, 0 = inferred
  } ],
  "skeletons": [ {
    "id": "humanoid22",
    "joints": [ {"name": "pelvis", "parent": -1, "offset": [0, 0.95, 0]},
                {"name": "spine1", "parent": 0,  "offset": [0, 0.10, 0]} ]   // rest offsets from the parent
  } ],
  "locations": [ {
    "id": "loc0", "name": "kitchen", "kind": "indoor",       // indoor street nature voxel stage space other
    "bounds": [[-4, 0, -4], [4, 3, 4]],
    "environment": {"sky": [0.6, 0.7, 0.9], "ambient": 0.35, "sun_dir": [0.3, -1, 0.2],
                    "sun": 1.0, "fog": null},
    "props": [ {"id": "counter", "class": "counter", "mesh": "mesh_counter",
                "position": [1.2, 0, -2], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]} ]
  } ],
  "entities": [ {
    "id": "e0", "name": "person 1", "class": "person",        // person animal vehicle object
    "location": "loc0",
    "mesh": "mesh_person0",
    "skeleton": "humanoid22",                                  // optional
    "track": {
      "state":    ACC,    // u8 [T]: 0 absent, 1 present but unseen (inferred), 2 seen in frame
      "position": ACC,    // f32 [T, 3] root position in the location
      "rotation": ACC,    // f32 [T, 4] root orientation
      "scale":    ACC,    // optional f32 [T, 3]
      "pose":     ACC,    // optional f32 [T, J, 4] joint rotations relative to the rest pose
      "morph":    ACC,    // optional f32 [T, K] morph weights (faces: ARKit-style)
      "confidence": ACC   // optional f32 [T] in 0–1
    }
  } ],
  "cameras": [ {
    "id": "cam0", "shot": "s0",
    "track": {             // frames [shot.start, shot.end)
      "position": ACC,     // f32 [n, 3]
      "rotation": ACC,     // f32 [n, 4]
      "fov":      ACC      // f32 [n]
    }
  } ],
  "shots": [ {
    "id": "s0", "start": 0, "end": 120, "location": "loc0", "camera": "cam0",
    "size": "wide",        // extreme_wide wide medium closeup extreme_closeup insert
    "move": "static",      // static pan tilt dolly truck orbit handheld crane drone fpv
    "transition": "cut"    // how this shot begins: cut dissolve fade
  } ],
  "continuity": [ {
    "kind": "moved",       // moved appeared vanished pose wardrobe lighting
    "entity": "e3", "from_shot": "s1", "to_shot": "s2", "frame": 240,
    "detail": "mug is 0.31 m left of where it was when the shot cut away", "severity": 0.6
  } ]
}
```

`T` is always `n_frames`. Entity tracks cover the whole video because a
location keeps existing while the edit is somewhere else; `state` says what was
actually seen. Tracks of entities in other locations are only meaningful while
their location is on screen or being inferred.

## Skeletons

The standard humanoid skeleton `humanoid22` (joint order is part of the format):

```
 0 pelvis      1 spine1      2 spine2      3 chest       4 neck        5 head
 6 l_clavicle  7 l_shoulder  8 l_elbow     9 l_wrist
10 r_clavicle 11 r_shoulder 12 r_elbow    13 r_wrist
14 l_hip      15 l_knee     16 l_ankle    17 l_toe
18 r_hip      19 r_knee     20 r_ankle    21 r_toe
```

The world transform of joint `j` at frame `t` is
`parent_world · T(offset_j) · R(pose[t, j])`, with the root's parent being the
entity transform `T(position[t]) · R(rotation[t]) · S(scale[t])`. Skinned
vertices use linear blend skinning with the rest pose (all `pose` identity) as
the bind pose.

## Optional keys (version 1)

Adopted from the generator (`restage/gen/FORMAT_NOTES.md`); readers that don't know them can ignore them.

| where | key | meaning |
|---|---|---|
| prop | `cast_shadow` | bool, default true; false for single-sided room shells and distant backdrops |
| entity | `identity` | entities with the same value are the same person or object (one entity per location they visit, or per wardrobe) |
| skeleton | `base`, `scale` | `"base": "humanoid22"` means humanoid22's joint names and order, with offsets scaled by `scale` |
| shot | `world_frame` | frame on a continuous world clock where the shot starts |
| shot | `time_skip` | seconds skipped since the previous shot in the same location (an ellipsis, not an error) |
| environment | `ground`, `fog` | hemisphere ground colour; fog `{color, near, far}` in linear RGB and metres |
| continuity | `replacement` | for wardrobe errors: the entity that replaces this one from the cut onward |

## Versioning

Readers must reject unknown major versions and ignore unknown keys, so later
versions can add fields (for example Gaussian detail layers or physical
properties) without breaking old files.
