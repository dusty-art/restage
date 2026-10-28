"""Generate and pack a dataset split with the procedural generator, in parallel worker processes.

    python -m restage.data.make_dataset --n 40 --split train --genres all --seconds-min 8 --seconds-max 20
    python -m restage.data.make_dataset --n 10 --split val ...
    python -m restage.data.make_dataset --smoke          # 40 train / 10 val / 10 test, mixed genres, 8-20 s

Each sample: `python -m restage.gen` renders <name>.restage / .mp4 / _aux.npz into data/<split>/_work/
(448x252 video, 320x180 depth / ids), `pack.pack` writes data/<split>/<name>.npz, and the intermediate
files are deleted (the first `--keep` samples are moved to data/<split>/_demo/ instead).

Seeds: sample i of a split uses seed SPLIT_BASE[split] + seed_offset + i, so train / val / test never share
a seed. Generator failures and clips the packer rejects are logged in index.json and replaced by the next
seed. Rendering (GPU) is limited to `--render-workers` concurrent generator processes; packing uses all
`--workers`.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PY = sys.executable
GENRES = ("talking", "interview", "dance", "hiking", "driving", "voxel")
SPLIT_BASE = {"train": 0, "val": 1_000_000, "test": 2_000_000}
SPLIT_SPAN = 1_000_000

_sem = None


def _init(sem):
    global _sem
    _sem = sem
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def sample_params(split, index, genres, smin, smax, errors_frac, dr_max):
    seed = SPLIT_BASE[split] + index
    rng = np.random.default_rng([seed, 2024])
    return {"seed": seed, "genre": genres[index % len(genres)], "seconds": round(float(rng.uniform(smin, smax)), 2),
            "errors": bool(rng.random() < errors_frac), "dr": round(float(rng.uniform(0, dr_max)), 3)}


def build_one(p, out_dir, work_dir, keep, res, aux_res, timeout=900):
    """Generate + pack one sample (runs in a worker process). Returns a stats dict."""
    from .pack import BadSample, pack
    name = f"{p['genre']}_{p['seed']:07d}"
    out_dir, work_dir = Path(out_dir), Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    st = dict(p, name=name)
    t0 = time.time()
    cmd = [PY, "-m", "restage.gen", "--genre", p["genre"], "--n", "1", "--seconds", str(p["seconds"]),
           "--res", res, "--aux-res", aux_res, "--seed", str(p["seed"]), "--out", str(work_dir), "--name", name]
    if p["errors"]:
        cmd.append("--errors")
    if p["dr"] > 0:
        cmd += ["--dr", str(p["dr"])]
    try:
        if _sem is not None:
            _sem.acquire()
        try:
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        finally:
            if _sem is not None:
                _sem.release()
        st["t_gen"] = round(time.time() - t0, 2)
        if r.returncode != 0:
            raise RuntimeError("generator failed: " + (r.stderr or r.stdout)[-800:])
        st["gen_warnings"] = [l for l in r.stderr.splitlines() if "Warning" in l][:4]
        t1 = time.time()
        clip = pack(work_dir / name, out_dir / f"{name}.npz", seed=p["seed"])
        st["t_pack"] = round(time.time() - t1, 2)
        st.update(K=int(len(clip["key_frame"])), S=int(len(clip["shot_range"])), L=int(len(clip["loc_points"])),
                  E=int(len(clip["ent_class"])), n_frames=int(clip["n_frames"]),
                  bytes=int((out_dir / f"{name}.npz").stat().st_size), ok=True)
        inter = [work_dir / f"{name}{s}" for s in (".restage", ".mp4", "_aux.npz")]
        st["bytes_intermediate"] = int(sum(f.stat().st_size for f in inter if f.exists()))
        if keep:
            demo = out_dir / "_demo"
            demo.mkdir(exist_ok=True)
            for f in inter:
                if f.exists():
                    shutil.move(str(f), str(demo / f.name))
        else:
            for f in inter:
                f.unlink(missing_ok=True)
    except (BadSample, RuntimeError, subprocess.TimeoutExpired) as e:
        st.update(ok=False, error=f"{type(e).__name__}: {e}"[:1000])
        for f in work_dir.glob(f"{name}*"):
            f.unlink(missing_ok=True)
    except Exception as e:     # unexpected: keep the traceback
        st.update(ok=False, error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
        for f in work_dir.glob(f"{name}*"):
            f.unlink(missing_ok=True)
    st["t_total"] = round(time.time() - t0, 2)
    return st


def make_split(split, n, genres=GENRES, seconds_min=8.0, seconds_max=20.0, seed_offset=0, workers=None,
               render_workers=5, keep=0, keep_all=False, errors_frac=0.25, dr_max=0.6, res="448x252",
               aux_res="320x180", out=DATA, max_failures=None, verbose=True):
    if split not in SPLIT_BASE:
        raise ValueError(f"split must be one of {list(SPLIT_BASE)}")
    out_dir = Path(out) / split
    work_dir = out_dir / "_work"
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or max(2, (os.cpu_count() or 4) - 2)
    max_failures = max_failures if max_failures is not None else max(10, n)
    ctx = mp.get_context("spawn")
    sem = ctx.Semaphore(max(1, render_workers))
    results, failures = [], []
    t0 = time.time()
    next_i = seed_offset
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init, initargs=(sem,)) as ex:
        pending = {}

        def submit():
            nonlocal next_i
            if next_i - seed_offset >= SPLIT_SPAN:
                raise RuntimeError("ran out of seeds for this split")
            p = sample_params(split, next_i, genres, seconds_min, seconds_max, errors_frac, dr_max)
            k = keep_all or (len(results) + len(pending) < keep)
            fut = ex.submit(build_one, p, str(out_dir), str(work_dir), k, res, aux_res)
            pending[fut] = p
            next_i += 1

        for _ in range(min(n, workers * 2)):
            submit()
        while pending:
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for fut in done:
                pending.pop(fut)
                st = fut.result()
                (results if st["ok"] else failures).append(st)
                if verbose:
                    msg = (f"[{split}] {len(results)}/{n} {st['name']}: K={st['K']} S={st['S']} L={st['L']} E={st['E']} "
                           f"{st['bytes'] / 1e6:.2f} MB gen {st['t_gen']}s pack {st['t_pack']}s") if st["ok"] else \
                          f"[{split}] FAILED {st['name']}: {st['error'][:200]}"
                    print(msg, flush=True)
                if len(failures) > max_failures:
                    raise RuntimeError(f"too many failures ({len(failures)})")
            while len(results) + len(pending) < n and len(pending) < workers * 2:
                submit()
    elapsed = time.time() - t0
    shutil.rmtree(work_dir, ignore_errors=True)
    # merge into index.json
    idx_path = out_dir / "index.json"
    index = json.loads(idx_path.read_text()) if idx_path.exists() else {"split": split, "clips": [], "failures": []}
    have = {c["name"] for c in index["clips"]}
    index["clips"] += [r for r in sorted(results, key=lambda r: r["seed"]) if r["name"] not in have]
    index["failures"] += failures
    b = np.array([r["bytes"] for r in results]) if results else np.zeros(1)
    secs = np.array([r["n_frames"] / 30 for r in results]) if results else np.zeros(1)
    run = {"n": len(results), "failed": len(failures), "elapsed_s": round(elapsed, 1),
           "clips_per_min": round(len(results) / max(elapsed, 1e-9) * 60, 2),
           "video_s_per_min": round(float(secs.sum()) / max(elapsed, 1e-9) * 60, 1),
           "bytes_per_clip_mean": int(b.mean()), "bytes_per_video_s": int(b.sum() / max(secs.sum(), 1e-9)),
           "workers": workers, "render_workers": render_workers, "res": res, "aux_res": aux_res,
           "genres": list(genres), "seconds": [seconds_min, seconds_max], "seed_offset": seed_offset}
    index.setdefault("runs", []).append(run)
    total = sum(c["bytes"] for c in index["clips"])
    index["summary"] = {"clips": len(index["clips"]), "bytes": total,
                        "bytes_per_clip": int(total / max(len(index["clips"]), 1)),
                        "genres": {g: sum(c["genre"] == g for c in index["clips"]) for g in GENRES}}
    idx_path.write_text(json.dumps(index, indent=1))
    if verbose:
        print(f"[{split}] {run['n']} clips in {elapsed:.0f}s: {run['clips_per_min']} clips/min, "
              f"{run['video_s_per_min']} video-s/min, {run['bytes_per_clip_mean'] / 1e6:.2f} MB/clip "
              f"({run['bytes_per_video_s'] / 1e3:.0f} kB per video second), {len(failures)} failures", flush=True)
    return index, run


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m restage.data.make_dataset", description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--split", default="train", choices=list(SPLIT_BASE))
    ap.add_argument("--genres", default="all", help="comma-separated genres or 'all'")
    ap.add_argument("--seconds-min", type=float, default=8.0)
    ap.add_argument("--seconds-max", type=float, default=20.0)
    ap.add_argument("--seed-offset", type=int, default=0, help="first sample index within the split's seed range")
    ap.add_argument("--workers", type=int, default=None, help="worker processes (default: cores - 2)")
    ap.add_argument("--render-workers", type=int, default=5, help="concurrent generator / GPU render processes")
    ap.add_argument("--keep", type=int, default=3, help="keep the mp4 / aux / restage of the first N samples in _demo/")
    ap.add_argument("--keep-all", action="store_true", help="keep every sample's intermediate files")
    ap.add_argument("--errors-frac", type=float, default=0.25, help="fraction of samples with continuity errors")
    ap.add_argument("--dr", type=float, default=0.6, help="max domain-randomisation strength (uniform 0..dr)")
    ap.add_argument("--res", default="448x252")
    ap.add_argument("--aux-res", default="320x180")
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--smoke", action="store_true", help="40 train / 10 val / 10 test, all genres, 8-20 s")
    a = ap.parse_args(argv)
    genres = GENRES if a.genres == "all" else tuple(g.strip() for g in a.genres.split(","))
    for g in genres:
        if g not in GENRES:
            ap.error(f"unknown genre {g}")
    kw = dict(genres=genres, seconds_min=a.seconds_min, seconds_max=a.seconds_max, workers=a.workers,
              render_workers=a.render_workers, keep_all=a.keep_all, errors_frac=a.errors_frac, dr_max=a.dr,
              res=a.res, aux_res=a.aux_res, out=a.out)
    if a.smoke:
        runs = {}
        for split, n in (("train", 40), ("val", 10), ("test", 10)):
            _, runs[split] = make_split(split, n, seed_offset=a.seed_offset, keep=a.keep if split == "train" else 1, **kw)
        print(json.dumps(runs, indent=1))
    else:
        make_split(a.split, a.n, seed_offset=a.seed_offset, keep=a.keep, **kw)


if __name__ == "__main__":
    main()
