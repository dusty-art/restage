# Restage model design

Video in → `.restage` replica out: one low-poly set per location, persistent people and objects with
per-frame motion, one camera per shot, and the edit. Trained and run on an Apple M5 (16 GB).

This design combines the first-principles notes ([FIRST_PRINCIPLES.md](FIRST_PRINCIPLES.md)) with the
five research reports in `research/`. Where the reports disagreed with my first guess, the reason is
noted.

## 1. What is new here

No existing system outputs a structured, editable scene for edited video. Reconstruction models
(research/01) output dense per-frame geometry, assume one continuous shot, and force frames from
different places into one coordinate frame. Object-centric models (research/02) don't survive cuts.
Restage treats **the edit itself as data**:

1. **Per-location gauge anchored to the edit.** Each location's coordinate frame is defined by the
   first camera that films it (origin on the ground beneath it, its view direction as −Z). Every later
   shot of that place, including reverse shots that share no background, is predicted *relative to
   that anchor*. Registration across cuts becomes a regression the network learns from synthetic
   edits, using the cues editors actually rely on: the people (their size, facing and eyelines), the
   180° rule, shot sizes and match-on-action. No feature matching between shots is needed. (research/05
   idea 1, "the edit is the rig"; no prior work found.)
2. **Video-level slots.** Location slots and entity slots live for the whole video; every keyframe is
   explained by them. Identity across a cut is slot reuse, never a separate re-identification step.
   Shots are assigned to locations by soft clustering, so the number of places is discovered, not fixed.
3. **A scene program as the output.** Sets are a ground plane plus up to 24 oriented, rounded boxes;
   people are a parametric humanoid on the format's `humanoid22` skeleton; objects are rounded boxes.
   Every output is a fixed-size regression target, so the network trains directly on synthetic ground
   truth, the result is low-poly by construction, and it is editable. (research/02 recommendation.)
4. **Compute follows the edit.** Cuts are found first; the network sees only keyframes (4 per second
   within a shot, at least 3 per shot), and motion between keyframes is interpolated. A minute of video
   is about 240 keyframes × 145 tokens.
5. **The replica is judged as a compression of the video.** Beyond pixel metrics, the benchmark scores a
   replica by how many bits a video encoder saves when the rendered replica is its reference frame
   (research/05 idea 2: correct scenes save bits, wrong locations save none). This also gives a
   principled way to choose the number of entities and locations at inference (planned, §7).

## 2. Pipeline

```
video ──► decode (30 fps, 224×126) ──► cut detector ──► shots ──► keyframes (4/s per shot)
                                                         │
                frozen DINOv2-S/14 (MLX port) ◄──────────┘   145 tokens × 384 per keyframe
                          │
                Restage compiler (MLX, ~25M trainable)
                          │
     cameras per keyframe │ location slots → sets │ entity slots → class, shape, tracks
                          ▼
              interpolate to 30 fps, decode meshes ──► .restage
                          │
              optional test-time refinement through the differentiable renderer
```

### 2.1 Front end
* **Decode** with ffmpeg to 30 fps at 224×126 (DINOv2 patch 14 → a 16×9 grid).
* **Cut detector**: a small head over per-frame DINOv2 CLS embeddings and colour-histogram distances of
  consecutive frames, trained on the generator's exact cuts plus synthetic splices of real clips
  (research/03). TransNetV2 (MIT) is the fallback teacher for real footage.
* **Keyframes**: 4 per second inside each shot, always including each shot's first and last frame.

### 2.2 Backbone
DINOv2-S/14 (Apache-2.0, ungated; DINOv3 is gated), frozen, ported to MLX and run on the fly so that
augmentations (crop-cuts, colour jitter, re-edits) are free and nothing is cached to disk. A ViT-S
forward pass is ~1 ms per keyframe on the M5 (research/04), so 128 keyframes cost ~0.15 s.

### 2.3 The compiler (MLX, d = 384)
1. **Frame adapter**: linear 384→384, learned 2D position, plus a learned *camera token* per keyframe.
2. **Shot encoder** (6 blocks): spatial self-attention within each keyframe (patches + camera token),
   then temporal self-attention among the camera tokens of the same shot (block-diagonal mask), so
   each camera token sees its whole shot. Output: camera tokens and a pooled shot token.
3. **Location slots** (max 4): slot attention over shot tokens (3 iterations, softmax over slots) gives
   each shot a soft location assignment. Each location slot then cross-attends to the patches of its
   shots to gather set evidence.
4. **Entity slots** (max 10): slot attention over all keyframe patches of the video (3 iterations),
   with a separate *identity* vector (static) and per-keyframe *state* read out by cross-attention from
   each keyframe's patches (query = slot ⊕ keyframe embedding).
5. **Edit-aware camera head**: each camera token attends to its location slot and to the entity slots
   (people are the scale bar and the cross-cut anchor), then regresses the camera in the location's
   gauge: 6D rotation, translation (metres), log field of view.

### 2.4 Decoders (all fixed-size)
| output | parameters |
|---|---|
| camera (per keyframe) | 6D rotation, translation, vertical fov |
| set (per location) | ground colour, sky colour, 24 boxes × (presence, centre, size, yaw, roundness, colour) |
| entity | class (none, person, animal, vehicle, object), location, shape, colours |
| person shape | height scale, girth, colours of skin / top / bottom / shoes / hair |
| object shape | rounded box size, roundness, colour |
| entity state (per keyframe) | present, visible, root position, yaw, 21 joint rotations (6D, people only) |

Tracks are interpolated to the video frame rate (cubic for positions, slerp for rotations) inside each
shot; cameras never interpolate across a cut.

## 3. Training

### 3.1 Data (research/03)
* **Synthetic edits** from `restage.gen`: six genres with real film grammar and optional injected
  continuity errors, rendered on the laptop with exact ground truth for cameras, cuts, locations,
  identities, poses, depth and instance ids. This carries v1.
* **Edit augmentation** of any clip: crop-cuts (a crop is an exact focal-length change), re-ordered and
  re-cut shots (the replica must not change), colour jitter, compression.
* **Real video** for self-supervision later: Wikimedia Commons, Prelinger (public domain), Blender open
  movies; no YouTube-derived sets, no Pexels/Pixabay bulk use, no share-alike data in training.

### 3.2 Gauge fixing
Ground truth is expressed in each location's anchor gauge (§1.1): rotate about +Y and translate in the
ground plane so the location's first keyframe camera sits at (0, h, 0) looking toward −Z (pitch and
roll kept). The network predicts in the same gauge, so losses need no alignment step.

### 3.3 Losses
* **Cameras**: geodesic rotation error, L1 translation (metres), L1 log-fov; for shots after the first
  in a location this is exactly the cross-cut registration problem.
* **Locations**: permutation-invariant pairwise loss: BCE on "shot i and shot j are the same place".
* **Entities**: Hungarian matching of slots to ground-truth entities (class, visibility overlap over
  keyframes, trajectory distance); then class CE (unmatched → none), presence/visibility BCE, root L1,
  yaw, joint geodesic (when visible), shape and colour L1.
* **Sets**: Chamfer distance between points sampled on the predicted boxes and on the ground-truth set,
  plus box presence regularised toward few boxes (the rate term).
* **Render consistency** (§4): depth and instance masks of the predicted scene rendered from the
  predicted cameras versus ground truth (synthetic) or teachers (real).

### 3.4 Teachers for real video (research/01, /02)
Depth Anything 3 Small or Video-Depth-Anything-Small (Apache) for depth, SAM 2.1 (Apache) for masks,
SEA-RAFT (BSD-3) for flow, MoGe-2 (MIT) for focal length. All run once per clip and are cached.
Non-commercial models (DUSt3R/MASt3R/π3/VGGT-1B/CoTracker3) and gated ones (SAM 3, DINOv3) are
excluded.

## 4. Differentiable renderer
A soft triangle rasteriser written as custom Metal kernels through `mx.fast.metal_kernel`, with a
hand-written backward (research/04 measured 1.8 ms forward+backward for 1k triangles at 128², 40–126×
faster than plain tensor ops). Primitives are tessellated in MLX (rounded boxes: 6×N² quads; people:
the parametric humanoid skinned on `humanoid22` with linear blend skinning), then rasterised at
64×36 (training) or 128×72 (refinement) into alpha, depth, instance probabilities and colour.

## 5. Optimisation for the M5 (research/04)
* MLX for everything trained; PyTorch only for offline teachers.
* Whole training step under `mx.compile`; bf16 compute with fp32 master weights; `MLX_ENABLE_TF32=0`
  for pose maths in tests.
* `mx.set_memory_limit(10 GB)`, evaluate after each micro-batch; gradient checkpointing switchable.
* Fixed shapes (padded keyframes and slots) so compiled graphs are reused.
* Curriculum: short clips with one location first, then longer edits, then several locations.
* Inference: DINOv2 + compiler at fp16; Core ML / Neural Engine export of the backbone later.

## 6. Evaluation ("CutScene" benchmark, research/03)
Held-out synthetic clips (unseen seeds and asset combinations) per genre:
* cut F1; location clustering accuracy (pairwise same-place F1);
* camera rotation and translation error, reported separately for anchor shots, same-location
  later shots (cross-cut) and moving cameras;
* entity count accuracy, identity consistency across cuts, root position error, joint angle error;
* set Chamfer distance; rendered depth error and instance IoU through the ground-truth cameras;
* bits saved by the replica as an encoder reference (§1.5).
Baselines: per-shot independent prediction (same network, no cross-shot attention) to measure what
the edit-aware parts add; and "identity across cuts by appearance only".

## 7. Planned after v1
* Inference-time model selection by bit cost (add or merge entities and locations while bits drop).
* Self-supervised fine-tuning on real video with teacher agreement and the DINO-feature re-render loss.
* Refine-then-distil: per-video refinement results fed back as training targets (research/02 idea 5).
* Voxel decoding for block worlds (research/05 idea 4); Mimetic faces for talking-head close-ups.
* Video-texture projection onto the replica for photoreal playback near the original cameras.
