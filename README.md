# Restage

**Video in, animated 3D replica out.** Restage watches a short video, including edited video with cuts,
close-ups, odd angles and changes of place, and rebuilds it as a low-poly 3D production: one set per
location, every person and object as a persistent entity with its own motion, one camera per shot, and
the edit itself. Play it through the original cameras and it reproduces the video; move the camera and
you see the same scene from anywhere.

Everything trains and runs on a 16 GB Apple M5 laptop.

> Status: research prototype. See [docs/DESIGN.md](docs/DESIGN.md) for the model and
> [research/](research/) for the five research reports it is based on.

## Quick start

```bash
cd ~/Desktop/test2 && source .venv/bin/activate
```

The viewer (play, explore, compare against the source, export):

```bash
cd viewer && npx vite
```

Then open http://127.0.0.1:5178 and pick a sample.

Reconstruct a video from the command line:

```bash
python -m restage.infer my_clip.mp4 -o samples/my_clip.restage
```

Or start the reconstruction server and use **Reconstruct video…** in the viewer:

```bash
python -m restage.server
```

## What's here

| path | what |
|---|---|
| `docs/SCENE_FORMAT.md` | the `.restage` format: locations, entities, shots, cameras, continuity |
| `docs/DESIGN.md` | the model: edit-anchored location gauges, video-level slots, scene-program output |
| `docs/TRAINING_DATA.md` | the contract between the data pipeline and the model |
| `docs/FIRST_PRINCIPLES.md` | hypotheses written before the literature review |
| `viewer/` | web viewer: Replica / Explore / Map / Compare views, timeline, inspector, exports (GLB, Blender, OTIO, MP4, PNG) |
| `restage/gen/` | procedural generator of edited videos with exact 3D ground truth (6 genres, real film grammar) |
| `restage/data/` | MLX DINOv2-S, packer, dataset builder, gauge-fixing loader with augmentations, cut detector |
| `restage/render/` | differentiable soft rasteriser as custom Metal kernels |
| `restage/model/` | the Restage compiler: network, losses, training, decoding, evaluation |
| `restage/infer.py`, `restage/server.py` | video → `.restage`, and the local upload server |
| `research/` | the five research reports, M5 benchmarks and experiments |

## Building the data and training

```bash
python -m restage.data.make_dataset --n 2500 --split train
```

```bash
python -m restage.model.train --data data --steps 20000 --batch 2 --out runs/v1
```

```bash
python -m restage.model.evaluate runs/v1/best.safetensors --split test
```

## Licences

Code: not yet chosen. DINOv2-S weights: Apache-2.0 (Meta), downloaded from Hugging Face without a
sign-up. All training data is generated on the laptop by `restage.gen` from original assets.
