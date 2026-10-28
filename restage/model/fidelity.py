"""Replica fidelity: how well a `.restage` replica reproduces its source video (DESIGN.md §1.5, §6).

Two scores on sampled frames, both independent of ground-truth labels (so they also work on real video):

* **PSNR** of the replica rendered through its own shot cameras against the source frames.
* **Codec judge** (research/05, idea 2): interleave render R_t and frame F_t and encode with H.264 using one
  reference, so each frame can only be predicted from its own render (with motion compensation). The bytes
  spent on the frames, relative to a blank reference, say how much of the video the replica explains:
  `bits_saved = 1 − bytes(replica ref) / bytes(blank ref)`. A wrong location or camera saves nothing.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

FFMPEG = "ffmpeg"


def render_frames(doc: dict, frames: list[int], width: int, height: int) -> np.ndarray:
    from restage.gen.render import SceneRenderer
    r = SceneRenderer(doc, width, height)
    return np.stack([r.render(int(t), ("rgb",))["rgb"] for t in frames])


def _luma(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return np.clip(0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2], 0, 255).astype(np.uint8)


def _frame_bytes(seq: list[np.ndarray], w: int, h: int, odd_only: bool = True) -> int:
    with tempfile.TemporaryDirectory() as d:
        raw, mp4 = os.path.join(d, "s.raw"), os.path.join(d, "s.mp4")
        with open(raw, "wb") as fh:
            for p in seq:
                fh.write(np.ascontiguousarray(p).tobytes())
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}",
                        "-r", "30", "-i", raw, "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-bf", "0",
                        "-refs", "1", "-x264-params", "scenecut=0:keyint=100000", "-pix_fmt", "gray", mp4], check=True)
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=size",
                            "-of", "csv=p=0", mp4], capture_output=True, text=True, check=True)
    sizes = [int(x) for x in r.stdout.split()]
    return sum(sizes[1::2]) if odd_only else sum(sizes)


def score(doc: dict, source: np.ndarray, every: int = 3, judge_size=(320, 180)) -> dict:
    """doc: replica; source: u8 [T, H, W, 3] video frames at the doc's frame rate."""
    T, H, W = source.shape[:3]
    frames = list(range(0, min(T, doc["n_frames"]), every))
    rend = render_frames(doc, frames, W, H)
    src = source[frames]
    mse = ((rend.astype(np.float32) - src.astype(np.float32)) ** 2).mean(axis=(1, 2, 3))
    psnr = float(np.mean(10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-3))))
    jw, jh = judge_size
    import cv2
    small = lambda x: cv2.resize(x, (jw, jh), interpolation=cv2.INTER_AREA)
    sl = [_luma(small(f)) for f in src]
    rl = [_luma(small(f)) for f in rend]
    blank = np.full((jh, jw), 128, np.uint8)
    b_blank = _frame_bytes([x for f in sl for x in (blank, f)], jw, jh)
    b_rep = _frame_bytes([x for r, f in zip(rl, sl) for x in (r, f)], jw, jh)
    b_plain = _frame_bytes(sl, jw, jh, odd_only=False)
    return {"psnr_db": round(psnr, 2), "bits_saved": round(1 - b_rep / max(1, b_blank), 4),
            "bytes_blank_ref": b_blank, "bytes_replica_ref": b_rep, "bytes_plain_video": b_plain, "frames": len(frames)}
