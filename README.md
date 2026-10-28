# Restage

**Video in, animated 3D replica out.** Restage watches a short video, including edited video with cuts,
close-ups, odd angles and changes of place, and rebuilds it as a low-poly 3D production: one set per
location, every person and object as a persistent entity with its own motion, one camera per shot, and
the edit itself. Play it through the original cameras and it reproduces the video; move the camera and
you see the same scene from anywhere.

Everything trains and runs on a 16 GB Apple M5 laptop.

> Status: research prototype. The viewer, scene generator, data pipeline, renderer and model code
> are built and tested; **no trained model is included yet** (training on the full synthetic dataset
> is the next step). See [docs/DESIGN.md](docs/DESIGN.md) for the model and [research/](research/)
> for the five research reports it is based on.

## Setup

Apple Silicon Mac (training and inference use MLX), Python 3.12, Node 18+, ffmpeg.

```bash
git clone https://github.com/dusty-art/restage && cd restage
```

```bash
uv venv --python 3.12 && source .venv/bin/activate && uv pip install -e ".[dev]"
```

```bash
cd viewer && npm install && cd ..
```

DINOv2-S (Apache-2.0) downloads from Hugging Face on first use, no sign-up needed. Datasets,
checkpoints and generated samples are not in the repository; the commands below build them.

## Quick start

Generate demo scenes with their videos (any genre: talking, interview, dance, hiking, driving, voxel):

```bash
python -m restage.gen --genre dance --n 1 --seconds 16 --res 640x360 --out samples
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

Build the synthetic dataset (disjoint seeds per split; about 25–30 clips a minute and 1.6 MB per clip on
an M5). `scripts/build_full_dataset.sh` does all three:

```bash
python -m restage.data.make_dataset --n 120 --split test && python -m restage.data.make_dataset --n 120 --split val && python -m restage.data.make_dataset --n 2400 --split train
```

Train the cut detector, then the compiler (about 1.2 s per step with two micro-batches; the re-render loss
switches on at step 2,000):

```bash
python -m restage.data.cuts train
```

```bash
python -m restage.model.train --data data --steps 10000 --batch 2 --accum 2 --render-from 2000 --out runs/v1
```

Benchmark against the shot-by-shot baseline, then install the checkpoint for inference:

```bash
python -m restage.model.evaluate runs/v1/best.safetensors --split test && python -m restage.model.evaluate runs/v1/best.safetensors --split test --baseline per_shot
```

```bash
cp runs/v1/best.safetensors checkpoints/restage.safetensors && cp runs/v1/best.json checkpoints/restage.json
```

Tests: `pytest tests`.

## Licences

Code: no licence chosen yet (all rights reserved until one is added). DINOv2-S weights: Apache-2.0 (Meta), downloaded from Hugging Face without a
sign-up. All training data is generated on the laptop by `restage.gen` from original assets.
