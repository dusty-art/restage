"""Command-line entry point: generate synthetic Restage samples.

    python -m restage.gen --genre <name|random> --n N --seconds S --res WxH --seed K --out DIR [--errors]

For each sample it writes <name>.restage (ground-truth scene), <name>.mp4 (H.264 video) and
<name>_aux.npz (per-frame metric depth, instance ids, intrinsics, camera poses at --aux-res).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .. import scene
from . import render, scenario


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m restage.gen", description=__doc__.split("\n\n")[0])
    ap.add_argument("--genre", default="random", choices=("random",) + scenario.GENRES)
    ap.add_argument("--n", type=int, default=1, help="number of samples")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--res", default="640x360", help="video resolution WxH")
    ap.add_argument("--aux-res", default=None, help="depth / id resolution WxH (default: half the video resolution)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the first sample; sample i uses seed + i")
    ap.add_argument("--out", default="out")
    ap.add_argument("--errors", action="store_true", help="inject continuity errors between cuts")
    ap.add_argument("--dr", type=float, default=0.0, help="domain randomisation strength 0..1 (image-space)")
    ap.add_argument("--no-video", action="store_true", help="only write the .restage files")
    ap.add_argument("--normals", action="store_true", help="also store normals in the aux file")
    ap.add_argument("--name", default=None, help="base name (default: <genre>_<seed>)")
    a = ap.parse_args(argv)
    W, H = (int(x) for x in a.res.lower().split("x"))
    aw, ah = (int(x) for x in a.aux_res.lower().split("x")) if a.aux_res else (W // 2, H // 2)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(a.n):
        seed = a.seed + i
        t0 = time.time()
        genre = a.genre
        if genre == "random":
            genre = str(np.random.default_rng(seed).choice(scenario.GENRES))
        name = a.name if (a.name and a.n == 1) else f"{a.name or genre}_{seed:05d}"
        doc, state = scenario.generate(genre, seed, a.seconds, a.fps, a.errors, f"{name}.mp4", W, H)
        scene.save(out / f"{name}.restage", doc)
        t_gen = time.time() - t0
        msg = f"{name}: {genre}, {len(doc['shots'])} shots, {len(doc['entities'])} entities, " \
              f"{len(doc['continuity'])} continuity notes, generated in {t_gen:.1f}s"
        if not a.no_video:
            rng = np.random.default_rng(seed + 101)
            dr = render.DomainRandomization.random(rng, a.dr) if a.dr > 0 else render.DomainRandomization()
            stats = render.render_video(doc, out / f"{name}.mp4", out / f"{name}_aux.npz", W, H, (aw, ah), dr,
                                        normals=a.normals)
            msg += f", rendered {stats['frames']} frames at {stats['fps']:.0f} fps"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
