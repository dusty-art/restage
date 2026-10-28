# First-principles notes (written before reading the research reports)

Goal: one model, video in → animated low-poly 3D replica out (locations, entities, cameras, the edit),
trained and run on a 16 GB M5. These are my own starting hypotheses, written down before the
literature review so the design can be checked against them rather than anchored on prior work.

## What an edited video actually contains

1. **Within a shot** the camera moves continuously and most of the frame is static set. Parallax from
   a moving camera gives geometry; a locked-off camera gives none, so priors must fill in.
2. **Across cuts** the same place is often re-filmed from a very different angle (shot/reverse-shot,
   wide then close-up). That is *wide-baseline multi-view data for free*: the reverse shot shows the
   back of the room. Classical matching fails at these baselines; a model that keeps one persistent
   representation per location can still fuse them.
3. **Editors follow rules** (180° line, eyeline match, shot sizes, cutting on action). Those rules
   constrain where the cameras of consecutive shots can be relative to each other and to the actors.
   A model that learns film grammar from synthetic edits gets a prior no reconstruction method uses.
4. **People are the scale bar.** Bodies are about 1.5–1.9 m; doors, cars and chairs are similar
   anchors. Metric scale comes from recognised classes, not geometry.
5. **Most pixels don't change between frames.** Video codecs already encode this (motion vectors,
   skip blocks, I-frames at scene changes). Computation should follow change, not frame count.

## Hypotheses for the model

* **H1 — The replica is the compressed video.** Judge a reconstruction by how well it re-renders the
  video per bit of scene description (minimum description length). This chooses the number of
  locations and entities automatically (one more entity must pay for itself in reconstruction error)
  and makes "low poly" a principled rate constraint, not a style.
* **H2 — Persistent slots that outlive cuts.** Location slots, entity slots and per-frame camera
  tokens; each frame's tokens are explained by the slots. Identity across cuts is slot reuse, never a
  separate re-identification step.
* **H3 — Amortise, then refine.** A feed-forward network gives a complete replica in one pass; a short
  test-time refinement through a differentiable renderer then fits the actual video. The network
  supplies structure (cuts, identities, film grammar); refinement supplies precision. Both run on the
  laptop because refinement only touches the scene parameters, not the network.
* **H4 — Primitives as the rendering currency.** Sets and objects as small sets of oriented boxes /
  superquadrics, people as an articulated template whose limbs are primitives. Ray–primitive
  intersection is analytic, so a soft differentiable renderer is cheap enough to run inside training
  at low resolution, and the output is low-poly by construction.
* **H5 — Synthetic edits teach structure, real video teaches appearance.** Procedural scenes with
  exact ground truth supervise everything at once (cameras, cuts, identities, depth, poses); the
  re-render loss on real footage closes the gap, with frozen teachers (monocular depth, features)
  cached once per clip.
* **H6 — Compute follows change.** Keyframes at a few fps plus cheap motion between them; tokens for
  static regions are reused; attention is local within shots and global only through the slots.

## Risks I expect

* Soft rendering losses have local minima (objects collapse or hide behind each other) → need good
  initialisation from the feed-forward pass and coarse-to-fine schedules.
* Synthetic-to-real gap in appearance → train on mixed real video with self-supervised losses only.
* Slot models historically struggle on real video → anchor slots with detection-like supervision on
  synthetic data and feature-space (not pixel) reconstruction targets.
* 16 GB memory for video → aggressive token pooling, fp16/bf16, gradient checkpointing, cached features.
