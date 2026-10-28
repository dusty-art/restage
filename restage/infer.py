"""Inference: video file in, `.restage` replica out.

    python -m restage.infer clip.mp4 -o samples/clip.restage

Pipeline (DESIGN.md §2): decode at 30 fps and 224×126 → cut detection → keyframes (4 per second inside each
shot, plus each shot's first and last frame) → frozen DINOv2-S → Restage compiler → decode to a scene.
`ready()` is true once a trained checkpoint exists; the server uses it to report the model's status.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "checkpoints" / "restage.safetensors"
FPS = 30
W, H = 224, 126


def ready() -> bool:
    return CHECKPOINT.exists()


def info() -> dict:
    meta = {}
    if CHECKPOINT.with_suffix(".json").exists():
        meta = json.loads(CHECKPOINT.with_suffix(".json").read_text())
    return {"checkpoint": CHECKPOINT.name if CHECKPOINT.exists() else None,
            "checkpoint_bytes": CHECKPOINT.stat().st_size if CHECKPOINT.exists() else 0,
            "trained_steps": meta.get("step")}


def probe(video: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate,nb_frames:format=duration", "-of", "json", str(video)],
                       capture_output=True, text=True, check=True)
    j = json.loads(r.stdout)
    st = j["streams"][0]
    return {"width": int(st["width"]), "height": int(st["height"]), "duration": float(j["format"]["duration"])}


def read_video(video: Path, fps: int = FPS, size=(W, H), max_seconds: float = 60.0) -> np.ndarray:
    """All frames at `fps`, letterboxed to `size` (keeps the aspect ratio), as u8 [T, H, W, 3]."""
    w, h = size
    vf = f"fps={fps},scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"
    cmd = ["ffmpeg", "-v", "error", "-t", str(max_seconds), "-i", str(video), "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def keyframes(shots: list[tuple[int, int]], per_second: float = 4.0, fps: int = FPS):
    step = fps / per_second
    kf, ks = [], []
    for si, (a, e) in enumerate(shots):
        f = set(np.round(np.arange(a, e - 1, step)).astype(int).tolist()) | {a, e - 1}
        f = sorted(x for x in f if a <= x < e)
        kf += f
        ks += [si] * len(f)
    return np.array(kf, np.int32), np.array(ks, np.int32)


def reconstruct(video: Path, out: Path, progress: Callable[[float, str], None] = lambda p, s: None,
                checkpoint: Path = CHECKPOINT) -> dict:
    import mlx.core as mx
    from restage import scene as S
    from restage.data.cuts import detect_cuts
    from restage.model.decode import decode
    from restage.model.train import Features, load

    t0 = time.time()
    progress(0.02, "Reading the video")
    meta = probe(video)
    frames = read_video(video)
    n = len(frames)
    progress(0.12, "Finding the cuts")
    shots = detect_cuts(frames)
    kf, ks = keyframes(shots)
    progress(0.25, f"{len(shots)} shots, {len(kf)} keyframes; reading them with DINOv2")
    model, cfg = load(checkpoint)
    feats_fn = Features()
    K = len(kf)
    key_mask = np.ones((1, K), bool)
    feats = feats_fn(frames[kf][None], key_mask)
    progress(0.6, "Compiling the scene")
    batch = {"key_mask": mx.array(key_mask), "key_shot": mx.array(ks[None]), "key_time": mx.array((kf / FPS)[None].astype(np.float32)),
             "shot_mask": mx.array(np.ones((1, len(shots)), bool))}
    o = model(feats, batch)
    o = {k: np.array(v.astype(mx.float32)) for k, v in o.items()}
    progress(0.85, "Building meshes and tracks")
    doc = decode(o, {"key_frame": kf, "key_shot": ks, "shot_range": np.array(shots)}, fps=FPS, n_frames=n,
                 title=video.stem, source={"video": video.name, "width": meta["width"], "height": meta["height"]})
    S.save(out, doc)
    progress(1.0, "Done")
    return {"seconds": round(time.time() - t0, 2), "frames": n, "shots": len(shots), "keyframes": K,
            "locations": len(doc["locations"]), "entities": len(doc["entities"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--checkpoint", default=str(CHECKPOINT))
    a = ap.parse_args()
    v = Path(a.video)
    out = Path(a.out) if a.out else ROOT / "samples" / f"{v.stem}.restage"
    stats = reconstruct(v, out, lambda p, s: print(f"[{p * 100:5.1f}%] {s}", flush=True), Path(a.checkpoint))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
