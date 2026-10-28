# restage.gen: procedural scene generator

`restage.gen` generates synthetic videos together with their exact 3D ground truth, written as
`.restage` documents (`docs/SCENE_FORMAT.md`). Each video has:

- static low-poly sets, one per location;
- persistent entities with per-frame motion;
- one camera per shot;
- the edit.

The generator feeds three things: training data for the Restage model, demo scenes for the
viewer (`viewer/`), and the evaluation benchmark.

```
python -m restage.gen --genre <name|random> --n N --seconds S --res WxH --seed K --out DIR [--errors]
```

Genres: `talking`, `interview`, `dance`, `hiking`, `driving`, `voxel`. Each sample writes three
files:

| file | contents |
|---|---|
| `<name>.restage` | the full scene: sets, entities, cameras, shots, continuity notes |
| `<name>.mp4` | H.264 video at `--res` (default 640x360, 30 fps) |
| `<name>_aux.npz` | per frame at `--aux-res` (default: half of `--res`). See the keys below. |

The keys in `<name>_aux.npz`:

- `depth`: f16 [T,h,w], metric depth along the camera axis, 0 for sky.
- `ids`: u8 or u16 [T,h,w], instance ids. 0 is sky and set structure, 1..N are entities in
  document order, 1000+k is prop k of the location.
- `K`, `K_video`: intrinsics at aux and video resolution.
- `cam_to_world`: 4x4 per frame.
- `shot`: the shot index per frame.
- `frames`, `fps`, `video_size`.

Other options:

- `--dr 0..1`: image-space domain randomisation (colour, noise, vignette, motion blur, shadows,
  compression strength).
- `--normals`: also write world normals to the aux file.
- `--no-video`: write only the `.restage` file.
- `--name`: the output base name.

Generation is deterministic given the seed: sample `i` uses `seed + i`. The demo samples in
`samples/` were made with:

```
for g,s in talking:41 interview:61 dance:60 hiking:68 driving:71 voxel:26; do
  python -m restage.gen --genre $g --seed $s --seconds 16 --out samples --name $g; done
python -m restage.gen --genre talking --seed 51 --seconds 16 --errors --out samples --name continuity_errors
python -m restage.gen --genre talking --seed 48 --seconds 16 --out samples --name first --no-video
```

From Python:

```python
from restage import scene
from restage.gen import scenario, render
doc, state = scenario.generate("dance", seed=3, seconds=12)   # doc: SCENE_FORMAT document
scene.save("dance.restage", doc)
render.render_video(doc, "dance.mp4", "dance_aux.npz", 640, 360, (320, 180))
```

## Modules

| module | contents |
|---|---|
| `assets.py` | Low-poly mesh builders. Each returns `{positions, indices, colors, [joints, weights, morphs, material, colliders]}`. See [Assets](#assets). |
| `motion.py` | Procedural motion. See [Motion](#motion). |
| `sets.py` | 13 location generators. Each returns a `LocationSet` (location dict, deduplicated meshes, `spatial.World` for collisions, semantic `anchors`). See [Sets](#sets). |
| `cinema.py` | The shot planner and camera solver. See [Cinematography](#cinematography). |
| `scenario.py` | Composes a video, computes entity `state`, injects continuity errors, assembles the document. See [Scenarios](#scenarios). |
| `spatial.py` | Point-in-geometry and segment-occlusion queries over boxes, canopy spheres, room interiors, ground functions and voxel grids. |
| `render.py` | Headless moderngl renderer and video writer. See [Renderer](#renderer). |

### Assets

- **Primitives**: box, tapered box, cylinder or cone, icosphere, limb (capsule-like tapered prism),
  quad, tiled floor, and a merge helper.
- **Humanoid**: one rigid part per humanoid22 bone. It varies height (1.52–1.98 m), build,
  feminine or masculine proportions, skin, hair (short, buzz, curly, long, bob, bun, ponytail,
  bald), beard, top (t-shirt, long sleeve, open jacket, tank), bottom (pants, shorts, skirt),
  shoes, cap or beanie, and backpack. It has a face (eyes, brows, nose, ears, and a mouth with a
  `jawOpen` morph). There is also a blocky variant for voxel games.
- **Other characters and vehicles**: a dog (own skeleton); a car (sedan, hatch, SUV), a box truck
  and a bike, each with a vehicle skeleton.
- **Furniture and props**: table, chair, office chair, couch, armchair, bed, shelf with books,
  floor and desk lamps, plants, mug, ball, laptop, door, window, picture frame (original abstract
  art), microphone; kitchen counters, cabinets, fridge, TV, rug, desk, monitor, whiteboard,
  acoustic panels, speakers, stage lights, truss, mirror wall, barre, bench, bottle, book, phone.
- **Nature**: pine, broadleaf, birch, dead and palm trees, cactus, rocks, bushes, grass tufts,
  flowers, faceted heightfield terrain, a distant mountain ring.
- **Urban**: buildings with windows (some lit), storefronts, roads with lane markings,
  crosswalks, sidewalks, street lamps, traffic lights, guard rails, road ribbons along splines.
- **Voxel**: a greedy mesher over an original block palette (grass, dirt, stone, sand, water,
  log, leaves, planks, cobble, glass, roof, gravel, …) with colour-variant patches.

### Motion

- **Humanoids**, in layers:
  - A footstep planner. Stride follows speed and leg length; plants are fixed while in contact,
    with heel strike and toe-off pivots about fixed points. The pelvis is lowered wherever a leg
    cannot reach, and legs use analytic two-bone IK.
  - Walk and run along Catmull-Rom paths with exact speed profiles; turn in place; sit down and
    stand up (hip hinge); counter-movement jump; six dance styles stepping on the beat.
  - Upper body: spine lean and counter-rotation, breathing, arm swing, and IK overlays for talking
    gestures (beats, open palms, explaining, shrugs), waving, pointing, holding, reaching and
    drinking.
  - Head look-at with idle glances, nods, and `jawOpen` lip motion while speaking.
- **Vehicles**: kinematic bicycle model. Heading follows the tangent, steering is
  atan(wheelbase · curvature), wheel spin is distance / r, and body roll and pitch come from
  lateral and longitudinal acceleration.
- **Objects**: a ballistic ball with restitution, friction and rolling to rest (it spins
  consistently with its motion); objects attached to a hand (picked up without a pop, carried,
  put back); rolling objects.

### Sets

13 locations:

- Living room, kitchen, office, bedroom, podcast studio.
- Dance studio, concert stage (raised platform as ground).
- City street (two-lane street with parking, sidewalks with curbs, buildings, an intersection
  with lights and crosswalks, parked cars).
- Highway (1.8 km divided four-lane road through hilly terrain with guard rails, gantries and
  trees).
- Forest, mountain and desert trails (240 m terrains with a graded trail, vegetation, rocks,
  grass and a backdrop).
- Voxel world (block terrain, lake, trees, a house, flowers, clouds).

Sizes are real metres. The environment (sky, ground, ambient, sun direction and intensity, fog)
is randomised per location kind and time of day.

### Cinematography

- **Framing**: shot size sets the span of the subject that fills the frame height. Lens comes
  from a 35 mm-equivalent range per size. Eyes sit near the upper third with headroom, and the
  subject sits on a third with look room, or centred for direct address.
- **Camera search**: viewpoints are scored. They are rejected if the camera is inside or within
  0.3 m of geometry or people, if visibility is low, or if they break the 180° rule.
- **Moves**: static, pan, tilt, dolly, truck, orbit, follow, handheld (band-limited 1/f shake,
  breathing, walking bob), crane, drone (follow, orbit, fly-by, reveal) and first-person
  (head-mounted POV, game camera with mouse-look and bob, dashcam).
- **Genre templates**:
  - vlog: jump cuts, inserts, cutaways;
  - interview: establishing two-shot, then shot/reverse-shot, OTS and reactions on one side of
    the line;
  - dance: cuts on the beat;
  - hiking: drone, POV, follow, and wide shots with time skips across locations;
  - driving: dashcam, chase, roadside, drone, with section skips;
  - voxel: first and third person, with area changes.
- **Shot lengths**: log-normal, about 1.5–6 s.

### Scenarios

A scenario builds a world-time performance, plans the edit, maps video to world time (time
skips) and solves the cameras. It then sets entity `state` per frame:

- 2 when at least one sample point (two for distant people) is in the frustum, not occluded
  by set boxes, tree canopies, terrain, voxels or other entities, and larger than 2% of the
  frame height;
- 1 otherwise;
- 0 when the entity is absent.

With `--errors`, it injects 1–3 continuity errors at cuts and records them in `continuity`. The
kinds are: a prop moves, an object appears, an object vanishes, a top changes colour. The changed
thing is chosen among what is on screen around the cut. Vlogs can include:

- a sip from a mug: reach, pick up, drink, put back;
- a ball thrown through the room, which the edit cuts away to;
- b-roll of the host walking a street or trail. This is a second location, with an
  identity-linked copy of the host.

### Renderer

The renderer matches the viewer: three.js r170 MeshStandardMaterial with flatShading and
vertexColors, a hemisphere light at 2.2 × ambient, a sun at 2.4 × sun with shadows, ACES filmic
tone mapping, sRGB output, and THREE.Fog applied after encoding. Per fragment:

```
c = albedo / pi * (2.2 * ambient * mix(ground, sky, 0.5 n.y + 0.5) + 2.4 * sun * max(0, n · -sun_dir) * shadow) + emissive
```

Other details:

- The camera is a three.js perspective camera (vertical fov, looking down −Z).
- The depth buffer is logarithmic, so lane markings 1–2 cm above the road do not z-fight.
- RGB uses 4× MSAA. Depth and ids come from a separate exact pass at aux resolution.
- Skinning is linear blend skinning on the CPU (with morphs).
- `DomainRandomization`: colour jitter, sensor noise, vignette, motion blur (sub-frame
  accumulation with interpolated cameras and skinned vertices), shadows on or off, and CRF.

## Measured speed

Apple M5, headless OpenGL 4.1 (Metal), one process, shadows on. Mean over one 10 s scene per
genre; the range across genres is in brackets.

| | 256x144 | 640x360 |
|---|---|---|
| RGB only | 1350 fps (760–1780) | 920 fps (630–1130) |
| RGB + depth + ids | 940 fps (490–1250) | 560 fps (280–720) |
| full pipeline: RGB + H.264 encode + aux at half resolution + npz | 710 fps (440–920) | 390 fps (210–480) |

Scene generation (motion, camera solving, visibility) takes:

- 0.1–0.4 s per 16 s video for indoor and voxel scenes;
- about 2 s for trails and the city;
- about 3.5 s for the highway, which has the most occluders.

Peak memory is about 220 MB. A 16 s sample takes 2–12 MB for the `.restage` file, 0.5–2 MB
for the mp4 and 6–19 MB for the aux file.

Visibility against the rendered id buffers, measured over 6–8 seeds per genre: the analytic
`state` agrees with the rendered instance mask in 97–100% of entity-frames.

## Tests

```
.venv/bin/python -m pytest tests/test_gen.py
```

The suite has 29 tests and runs in about 5 s. It covers:

- the format round trip and the document contract (accessor alignment, shot tiling, track
  shapes, unit quaternions);
- planted feet during walks and runs, and on terrain: the lowest sole point stays on the
  ground and does not slide;
- stride growing with speed;
- cameras never inside geometry, for every genre;
- `state` matching the rendered instance ids, for every genre;
- determinism;
- recorded continuity errors;
- closed, outward-facing primitives, the greedy mesher, and humanoid rigging.

## Known limitations

- Motion is procedural, not captured. Gestures and dances are plausible but repetitive, and
  there are no finger joints (hands are rigid).
- Only one person sits per seat, and the generator never checks for people interpenetrating.
- Rigid skinning shows small gaps at joints, hidden by joint spheres.
- Visibility is coarse: boxes, spheres and terrain samples. Thin props such as lamp poles and
  table legs are boxes, and bushes are boxes. Partial occlusion is not measured, so `state` is
  binary per frame.
- Vehicles never interact: lanes run at one shared speed per lane, there is no overtaking, and
  cars do not stop for pedestrians. City traffic lights are decorative.
- Voxel avatars walk only on flat ground (they jump in place but do not climb blocks).
- Only cars, trucks and the ego vehicle are animated vehicles; bikes and the dog have meshes and
  skeletons but are not yet used in scenarios.
- Car lights on entities do not glow (one material per mesh; see FORMAT_NOTES.md).
- The viewer's shadow camera covers the whole location, while this renderer fits shadows around
  the camera outdoors. Shadow detail therefore differs slightly in large outdoor sets.
