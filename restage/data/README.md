# restage.data: data pipeline and backbone

Turns generator samples into packed training clips, serves fixed-shape batches to the model, and holds
the frozen DINOv2 backbone and the cut detector. The contract with the model is
[docs/TRAINING_DATA.md](../../docs/TRAINING_DATA.md), including the clarifications at its end.

| module | contents |
|---|---|
| `dinov2.py` | DINOv2 ViT-S/14 (facebook/dinov2-small, Apache-2.0) in MLX for 224×126 keyframes; weight conversion, parity check, benchmark |
| `pack.py` | one generator sample (`.restage` + `.mp4` + `_aux.npz`) → one packed clip `data/<split>/<name>.npz` |
| `make_dataset.py` | generate and pack a split in parallel worker processes, with disjoint seeds per split |
| `loader.py` | windows, gauge fixing, augmentations (colour jitter, crop-cuts, mirror), padding, background prefetch |
| `cuts.py` | frame-difference features on 32×18 thumbnails, a tiny MLX cut classifier, `detect_cuts()` |
| `vis.py` | debug PNGs: keyframes with projected entity roots / centres, ids overlay, depth, camera info |
| `GEN_REQUESTS.md` | generator problems found while packing, with reproductions and the workarounds used here |

## Building datasets

```bash
PY=.venv/bin/python
# the smoke dataset: 40 train / 10 val / 10 test, all six genres, 8-20 s each (about 1.5 min)
$PY -m restage.data.make_dataset --smoke
# a bigger split; --seed-offset continues a split without reusing seeds
$PY -m restage.data.make_dataset --n 2000 --split train --genres all --seconds-min 8 --seconds-max 20
$PY -m restage.data.make_dataset --n 200 --split val --genres talking,interview --seed-offset 10
```

* Sample `i` of a split uses seed `SPLIT_BASE[split] + i` (train 0, val 1 000 000, test 2 000 000), so
  the splits never share a seed. Genres cycle in the order given.
* Each sample runs `python -m restage.gen` (448×252 video, 320×180 depth / ids, continuity errors in 25%
  of samples, domain randomisation of strength U(0, 0.6)), then `pack.pack`, then deletes the
  intermediates. The first `--keep` samples (default 3; 1 for val / test in `--smoke`) keep their
  `.restage` / `.mp4` / `_aux.npz` in `data/<split>/_demo/`; `--keep-all` keeps everything.
* Workers: `--workers` processes (default: cores − 2 = 8) pack; at most `--render-workers` (default 5) of
  them run the generator / GPU renderer at a time.
* Clips the packer rejects (`BadSample`: non-finite camera tracks, missing files, …) and generator
  failures are logged in `index.json` and replaced by the next seed.
* `data/<split>/index.json` lists every clip (genre, seed, length, K / S / L / E, bytes, timings) and each
  run's throughput.

Pack a single sample by hand: `python -m restage.data.pack <dir>/<name> out.npz` (prints the
camera-convention check).

## Measured (Apple M5, 16 GB, shared with other agents; load average 3-7)

**DINOv2 ViT-S/14 in MLX**, 224×126 input → CLS [384] + 9×16 patches, batch 64, compiled:

| precision | ms per frame | frames / s |
|---|---|---|
| fp32 (TF32 matmuls, MLX default on M5) | 1.44 | 693 |
| bf16 (mixed: bf16 weights / matmuls / attention, fp32 residual stream and norms) | 1.20 | 831 |

Parity against HF transformers `Dinov2Model` (torch, CPU, fp32) on the same input, max |Δ| (values reach ±24):

| input | MLX mode | CLS max abs | patches max abs | patches mean abs | patch cosine (median / 1st pct / min) |
|---|---|---|---|---|---|
| 40 real keyframes (val) | fp32, `MLX_ENABLE_TF32=0` | 1.2e-4 | 1.1e-3 | 2.3e-5 | 1.0 / 1.0 / 1.0 |
| 40 real keyframes (val) | fp32 (TF32) | 0.081 | 1.36 | 0.016 | 0.99997 / 0.9998 / 0.981 |
| 40 real keyframes (val) | bf16 mixed | 0.27 | 2.04 | 0.050 | 0.9997 / 0.998 / 0.964 |
| 4 smooth random images | fp32, `MLX_ENABLE_TF32=0` | 7.8e-5 | 1.3e-4 | 1.6e-5 | 1.0 |
| 4 smooth random images | bf16 mixed | 0.15 | 0.36 | 0.037 | 0.9998 / 0.9996 / 0.9994 |

The port is exact; the remaining differences are TF32 / bf16 rounding. Pure bf16 (bf16 residual
stream) was as fast but ~3× worse on the worst tokens (min cosine 0.91 on real frames), hence the mixed
mode. `python -m restage.data.dinov2 --parity --bench [--frames data/val/*.npz]` reproduces this.

**Dataset generation** (`--smoke`, 8 workers):

| render workers | clips / min | video seconds / min |
|---|---|---|
| 3 | 26 | 360 |
| 5 (default) | 38 | 540 |

Per clip: generator + render + H.264 + aux npz 9-11 s of wall time under that concurrency (2-6 s alone),
packing 0.3 s (ffmpeg decode, labels, point sampling, compression).

**Disk:** 1.6 MB per packed clip on average (0.7-3.0 MB for 8-20 s; about 117 kB per video second; ~62
keyframes per clip). Keyframes are ~75% of it. The intermediates are 12-14 MB per clip (5-22 MB) and are
deleted. The smoke dataset with its demo samples is 127 MB; 10 000 clips would be about 16 GB.

**Loader:** one augmented batch of 8 windows takes 0.29 s in one process; with 3 worker processes (or
threads) the loader delivers 9 batches / s.

**Cut detector:** 8 s to train (2 000 steps, 111k frames from the 40 train clips plus 300 synthetic
splices). F1 on the smoke val / test clips (35 / 37 cuts; exact frame and ±1 frame give the same numbers):

| | val | test |
|---|---|---|
| ignoring the generator's in-shot camera jumps (GEN_REQUESTS.md #3) | **0.986** (P 0.97, R 1.0) | **1.0** |
| counting those jumps as false cuts | 0.737 | 0.804 |

The jumps are real one-frame discontinuities in the rendered video (on average 1.3 per clip, in 16 of
the 40 train clips), so the detector fires on them; they are excluded from its training labels.
Retrain with `python -m restage.data.cuts train`; the weights are in `checkpoints/cuts/`.

## Conventions worth knowing

* Cameras: `cam_rot` is camera-to-world (columns = camera x, y, z), the camera looks down −Z with +Y up,
  `cam_fov` is vertical. A world point X is at pixel `u = W/2 + f x_c / (−z_c)`, `v = H/2 − f y_c / (−z_c)`
  with `(x_c, y_c, z_c) = Rᵀ (X − c)` and `f = (H/2) / tan(fov/2)` (`pack.project`). `test_camera_convention`
  checks that entity centres projected this way land on their own ids (99.8% of 3849 visible
  entity-keyframes of the smoke set; occluded centres excluded via the depth map), and that wrong
  conventions do not.
* A person's root is on the ground below the pelvis; a seated person's root is below the seat.
* `ids` 255 in a batch = an entity dropped because the window has more than Emax entities.

## Tests

```bash
.venv/bin/python -m pytest tests/test_data.py -q      # 21 tests, ~20 s
```

They generate four fresh samples (seeds outside every split) and cover: DINOv2 shapes, bf16, the bicubic
position-embedding resampling against torch, parity with transformers (TF32 on, and exact with
`MLX_ENABLE_TF32=0`); keyframe selection; label downsampling without blur across edges; the packed-clip
contract (keys, dtypes, shapes, ranges); person region colours; the camera convention (with negative
controls) and roots at the feet; gauge fixing (anchor at (0, y, 0) looking toward −Z with pitch and roll
kept, rigid per location); projections onto ids after crop-cuts and mirroring; exact crop depth and ids;
the mirrored pose equal to the mirrored skeleton; mask consistency (including Emax overflow); epoch
windows covering every keyframe; background prefetch in processes and threads; cut features, NMS and
`detect_cuts` on a video.

Visual checks from the smoke set are in `data/vis/` (`clip_<genre>.png`: keyframes with projected roots
○ and centres +, ids overlay, depth; `batch_aug_*.png`: augmented, gauge-fixed loader samples).
