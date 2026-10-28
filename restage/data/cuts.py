"""Cut detector: frame-difference features on 32x18 thumbnails + a tiny MLX classifier.

    python -m restage.data.cuts train            # train on data/train (+ synthetic splices), report F1 on data/val
    python -m restage.data.cuts eval data/test
    from restage.data.cuts import detect_cuts
    shots = detect_cuts("clip.mp4")              # [[0, 131], [131, 222], ...]  ([start, end) frames)

Features of frame t (the transition t-1 -> t), from `thumbs` [T, 18, 32, 3]:
  RGB-histogram L1 distance, hue/saturation-histogram chi-square distance, mean / median absolute
  grey difference, fraction of changed pixels, min and max over a 4x4 grid of block mean differences,
  gradient-magnitude difference, brightness-normalised difference, the distance t-2 -> t, and
  motion-compensated differences (best global shift of +-2 px; mean / median over a 3x4 grid of the
  best per-block shift), which separate pans, handheld and fpv motion from cuts.
The classifier sees the features of frames t-4 .. t+4 (each also divided by the median of its 17-frame
neighbourhood) and outputs P(cut at t), i.e. t is the first frame of a new shot.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "checkpoints" / "cuts"
WIN = 4
N_FEAT = 13


# ---------------------------------------------------------------- features

def _hist_rgb(th):
    q = (th.astype(np.int32) >> 5)                                 # 8 bins per channel
    T = len(th)
    h = np.zeros((T, 3, 8), np.float32)
    for c in range(3):
        for t in range(T):
            h[t, c] = np.bincount(q[t, ..., c].ravel(), minlength=8)
    return h.reshape(T, 24) / (th.shape[1] * th.shape[2])


def _hist_hs(th):
    hsv = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2HSV) for f in th])
    hq = (hsv[..., 0].astype(np.int32) * 12 // 180)
    sq = (hsv[..., 1].astype(np.int32) * 4 // 256)
    key = hq * 4 + sq
    T = len(th)
    h = np.stack([np.bincount(key[t].ravel(), minlength=48) for t in range(T)]).astype(np.float32)
    return h / (th.shape[1] * th.shape[2])


def frame_features(thumbs: np.ndarray) -> np.ndarray:
    """thumbs u8 [T, 18, 32, 3] -> features f32 [T, N_FEAT] (row t describes t-1 -> t; row 0 is zero)."""
    th = np.asarray(thumbs)
    T = len(th)
    f = np.zeros((T, N_FEAT), np.float32)
    if T < 2:
        return f
    g = th.astype(np.float32).mean(-1) / 255                      # [T, 18, 32]
    hr, hs = _hist_rgb(th), _hist_hs(th)
    d = np.abs(g[1:] - g[:-1])
    f[1:, 0] = np.abs(hr[1:] - hr[:-1]).sum(1) / 6
    f[1:, 1] = 0.5 * ((hs[1:] - hs[:-1]) ** 2 / (hs[1:] + hs[:-1] + 1e-6)).sum(1)
    f[1:, 2] = d.mean((1, 2))
    f[1:, 3] = np.median(d.reshape(T - 1, -1), 1)
    f[1:, 4] = (d > 0.1).mean((1, 2))
    blk = d[:, :16, :32].reshape(T - 1, 4, 4, 4, 8).mean((2, 4))  # 4x4 grid of block means
    f[1:, 5] = blk.reshape(T - 1, -1).min(1)
    f[1:, 6] = blk.reshape(T - 1, -1).max(1)
    gx = np.abs(np.diff(g, axis=2))[:, :-1, :]
    gy = np.abs(np.diff(g, axis=1))[:, :, :-1]
    gm = gx + gy
    f[1:, 7] = np.abs(gm[1:] - gm[:-1]).mean((1, 2))
    gn = (g - g.mean((1, 2), keepdims=True)) / (g.std((1, 2), keepdims=True) + 0.02)
    f[1:, 8] = np.abs(gn[1:] - gn[:-1]).mean((1, 2)) / 4
    if T > 2:
        f[2:, 9] = np.abs(g[2:] - g[:-2]).mean((1, 2))
    # motion-compensated differences: best global shift, and best shift per block (pans / handheld / local motion)
    a, b = g[:-1], g[1:]
    R = 2
    H, W = g.shape[1:]
    best_glob = np.full(T - 1, np.inf, np.float32)
    best_blk = np.full((T - 1, 3, 4), np.inf, np.float32)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            aa = a[:, R + dy:H - R + dy, R + dx:W - R + dx]
            dd = np.abs(b[:, R:H - R, R:W - R] - aa)                 # [T-1, 14, 28]
            best_glob = np.minimum(best_glob, dd.mean((1, 2)))
            bl = dd[:, :12, :28].reshape(T - 1, 3, 4, 4, 7).mean((2, 4))
            best_blk = np.minimum(best_blk, bl)
    f[1:, 10] = best_glob
    f[1:, 11] = best_blk.reshape(T - 1, -1).mean(1)
    f[1:, 12] = np.median(best_blk.reshape(T - 1, -1), 1)
    return f


def window_inputs(feat: np.ndarray, win=WIN) -> np.ndarray:
    """[T, F] -> [T, (2 win + 1) * 2F]: raw features (log-compressed) and features relative to the local median."""
    T, F = feat.shape
    x = np.log1p(feat * 20)
    pad = np.pad(x, ((8, 8), (0, 0)), mode="edge")
    med = np.stack([np.median(pad[t:t + 17], 0) for t in range(T)])
    rel = x - med
    z = np.concatenate([x, rel], 1)
    zp = np.pad(z, ((win, win), (0, 0)), mode="edge")
    return np.stack([zp[t:t + 2 * win + 1].reshape(-1) for t in range(T)]).astype(np.float32)


def cut_labels(shot_range, T):
    y = np.zeros(T, np.float32)
    for s, e in shot_range[1:]:
        y[int(s)] = 1
    return y


# ---------------------------------------------------------------- model (MLX)

def _mlx():
    import mlx.core as mx
    import mlx.nn as nn
    return mx, nn


def make_model(d_in, hidden=48, dropout=0.2):
    mx, nn = _mlx()

    class CutMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(d_in, hidden)
            self.l2 = nn.Linear(hidden, hidden)
            self.l3 = nn.Linear(hidden, 1)
            self.drop = nn.Dropout(dropout)

        def __call__(self, x):
            h = self.drop(nn.gelu(self.l1(self.drop(x))))
            return self.l3(self.drop(nn.gelu(self.l2(h))))[..., 0]
    return CutMLP()


class CutDetector:
    def __init__(self, model, mean, std, threshold=0.5, min_gap=4):
        self.model, self.mean, self.std = model, mean, std
        self.threshold, self.min_gap = threshold, min_gap

    def probs(self, thumbs):
        mx, _ = _mlx()
        x = (window_inputs(frame_features(thumbs)) - self.mean) / self.std
        p = mx.sigmoid(self.model(mx.array(x)))
        p = np.array(p)
        p[0] = 0
        return p

    def cuts(self, thumbs=None, probs=None):
        p = self.probs(thumbs) if probs is None else probs
        return pick_cuts(p, self.threshold, self.min_gap)

    def save(self, d=CKPT):
        mx, _ = _mlx()
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        from mlx.utils import tree_flatten
        mx.save_safetensors(str(d / "cut_mlp.safetensors"), dict(tree_flatten(self.model.parameters())))
        (d / "cut_mlp.json").write_text(json.dumps({"mean": self.mean.tolist(), "std": self.std.tolist(),
                                                    "threshold": self.threshold, "min_gap": self.min_gap,
                                                    "d_in": int(len(self.mean)), "win": WIN}))

    @staticmethod
    def load(d=CKPT):
        mx, _ = _mlx()
        d = Path(d)
        cfg = json.loads((d / "cut_mlp.json").read_text())
        m = make_model(cfg["d_in"])
        m.load_weights(str(d / "cut_mlp.safetensors"))
        m.eval()
        return CutDetector(m, np.array(cfg["mean"], np.float32), np.array(cfg["std"], np.float32), cfg["threshold"],
                           cfg["min_gap"])


def pick_cuts(p, threshold=0.5, min_gap=4):
    """Frames t with p > threshold that are the maximum within +-min_gap (non-maximum suppression)."""
    cand = np.where(p > threshold)[0]
    out = []
    for t in cand[np.argsort(-p[cand], kind="stable")]:
        if all(abs(t - u) >= min_gap for u in out):
            out.append(int(t))
    return sorted(out)


def shots_from_cuts(cuts, T):
    b = [0] + [c for c in cuts if 0 < c < T] + [T]
    return [[b[i], b[i + 1]] for i in range(len(b) - 1)]


_DET = None


def detect_cuts(video_or_frames, detector: CutDetector | None = None):
    """Shot ranges [start, end) of a video path, of frames u8 [T, H, W, 3] (any size; resized like the packer),
    or of 32x18 thumbnails."""
    global _DET
    det = detector or _DET or CutDetector.load()
    _DET = _DET or det
    if isinstance(video_or_frames, (str, Path)):
        from .pack import decode_video, thumbnails
        th = thumbnails(decode_video(video_or_frames))
    else:
        fr = np.asarray(video_or_frames)
        if fr.shape[1:3] == (18, 32):
            th = fr
        else:
            from .pack import KEY_H, KEY_W, thumbnails
            if fr.shape[1:3] != (KEY_H, KEY_W):
                fr = np.stack([cv2.resize(f, (KEY_W, KEY_H), interpolation=cv2.INTER_AREA) for f in fr])
            th = thumbnails(fr)
    return shots_from_cuts(det.cuts(th), len(th))


# ---------------------------------------------------------------- data + training

def load_thumbs(split_dir):
    """[(name, thumbs, shot_range, jumps)] where jumps are the generator's in-shot camera jumps (meta)."""
    out = []
    for f in sorted(Path(split_dir).glob("*.npz")):
        with np.load(f) as z:
            meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
            out.append((f.stem, z["thumbs"], z["shot_range"], list(meta.get("cam_jumps", []))))
    return out


def ignore_mask(jumps, T, r=1):
    m = np.zeros(T, bool)
    for j in jumps:
        m[max(0, j - r):j + r + 1] = True
    return m


def splice_sequences(clips, rng, n, seg=(8, 120)):
    """Synthetic edits: random segments of random clips (including jump cuts inside one clip) joined end to end.
    Every junction and every original cut inside a segment is a cut."""
    seqs = []
    for _ in range(n):
        parts, labels, ign = [], [], []
        for _ in range(int(rng.integers(3, 8))):
            name, th, sr, jumps = clips[int(rng.integers(len(clips)))]
            L = int(rng.integers(seg[0], seg[1]))
            if len(th) <= L + 1:
                continue
            a = int(rng.integers(0, len(th) - L))
            y = cut_labels(sr, len(th))[a:a + L].copy()
            g = ignore_mask(jumps, len(th))[a:a + L].copy()
            y[0] = 1 if parts else 0
            g[0] = False if parts else g[0]
            parts.append(th[a:a + L])
            labels.append(y)
            ign.append(g)
        if parts:
            seqs.append((np.concatenate(parts), np.concatenate(labels), np.concatenate(ign)))
    return seqs


def build_xy(seqs):
    X = np.concatenate([window_inputs(frame_features(th)) for th, _, _ in seqs])
    Y = np.concatenate([y for _, y, _ in seqs])
    G = np.concatenate([g for _, _, g in seqs])
    return X, Y, G


def f1_score(pred, true, tol=0):
    """Greedy one-to-one matching of predicted and true cut frames within +-tol."""
    true = list(true)
    tp = 0
    used = set()
    for p in pred:
        best = None
        for i, t in enumerate(true):
            if i not in used and abs(p - t) <= tol and (best is None or abs(p - t) < abs(p - true[best])):
                best = i
        if best is not None:
            used.add(best)
            tp += 1
    fp, fn = len(pred) - tp, len(true) - tp
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-9)}


def evaluate(det, clips, tol=(0, 1)):
    """Cut F1 with exact frames (tol0) and +-1 frame (tol1); `*_nojump` ignores predictions within +-1 frame of a
    generator camera jump (GEN_REQUESTS.md #3), which look like cuts but are not edits."""
    res = {}
    tot = {(t, j): {"tp": 0, "fp": 0, "fn": 0} for t in tol for j in (False, True)}
    for name, th, sr, jumps in clips:
        pred = det.cuts(th)
        true = [int(s) for s, _ in sr[1:]]
        ign = ignore_mask(jumps, len(th))
        for t in tol:
            for j in (False, True):
                pp = [p for p in pred if not (j and ign[p] and p not in true)]
                r = f1_score(pp, true, t)
                for k in ("tp", "fp", "fn"):
                    tot[t, j][k] += r[k]
    for (t, j), v in tot.items():
        tp, fp, fn = v["tp"], v["fp"], v["fn"]
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        res[f"tol{t}" + ("_nojump" if j else "")] = {**v, "precision": round(p, 4), "recall": round(r, 4),
                                                     "f1": round(2 * p * r / max(p + r, 1e-9), 4)}
    return res


def train(train_dir=ROOT / "data" / "train", val_dir=ROOT / "data" / "val", steps=2000, n_splice=300, seed=0,
          out=CKPT, verbose=True):
    mx, nn = _mlx()
    import mlx.optimizers as optim
    rng = np.random.default_rng(seed)
    t0 = time.time()
    tr = load_thumbs(train_dir)
    va = load_thumbs(val_dir)
    seqs = [(th, cut_labels(sr, len(th)), ignore_mask(j, len(th)) & (cut_labels(sr, len(th)) < 0.5))
            for _, th, sr, j in tr] + splice_sequences(tr, rng, n_splice)
    X, Y, G = build_xy(seqs)
    keep = ~G                                              # frames next to generator camera jumps: no label
    X, Y = X[keep], Y[keep]
    mean, std = X.mean(0), X.std(0) + 1e-4
    Xn = (X - mean) / std
    if verbose:
        print(f"train frames {len(Y)} (cuts {int(Y.sum())}) from {len(tr)} clips + {n_splice} splices, "
              f"features in {time.time() - t0:.1f}s")
    mx.random.seed(seed)
    model = make_model(X.shape[1])
    opt = optim.AdamW(learning_rate=optim.cosine_decay(2e-3, steps, 1e-5), weight_decay=2e-2)
    pos_w = 8.0

    def loss_fn(m, x, y):
        logit = m(x)
        w = 1 + (pos_w - 1) * y
        return (w * nn.losses.binary_cross_entropy(logit, y, with_logits=True, reduction="none")).mean()

    step = nn.value_and_grad(model, loss_fn)
    pos, neg = np.where(Y > 0.5)[0], np.where(Y < 0.5)[0]
    bs = 1024
    model.train()
    for it in range(steps):
        idx = np.concatenate([rng.choice(pos, bs // 8), rng.choice(neg, bs - bs // 8)])
        loss, g = step(model, mx.array(Xn[idx]), mx.array(Y[idx]))
        opt.update(model, g)
        mx.eval(model.parameters(), opt.state)
        if verbose and (it % 500 == 0 or it == steps - 1):
            print(f"  step {it} loss {float(loss):.4f}")
    model.eval()
    det = CutDetector(model, mean.astype(np.float32), std.astype(np.float32))
    # threshold chosen on the training clips (not on val)
    trp = [(det.probs(th), [int(s) for s, _ in sr[1:]], ignore_mask(j, len(th))) for _, th, sr, j in tr]
    grid = np.round(np.linspace(0.2, 0.95, 16), 3)
    f1s = []
    for thr in grid:
        tp = fp = fn = 0
        for p, true, ign in trp:
            r = f1_score([c for c in pick_cuts(p, thr) if not (ign[c] and c not in true)], true, 0)
            tp, fp, fn = tp + r["tp"], fp + r["fp"], fn + r["fn"]
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    f1s = np.array(f1s)
    det.threshold = float(np.median(grid[f1s >= f1s.max() - 1e-9]))     # middle of the best plateau
    res = {"train": evaluate(det, tr), "val": evaluate(det, va), "threshold": det.threshold,
           "train_frames": int(len(Y)), "train_cuts": int(Y.sum()), "ignored_frames": int(G.sum()),
           "seconds": round(time.time() - t0, 1)}
    det.save(out)
    (Path(out) / "cut_mlp_eval.json").write_text(json.dumps(res, indent=1))
    if verbose:
        print(json.dumps(res, indent=1))
    return det, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--train", default=str(ROOT / "data" / "train"))
    t.add_argument("--val", default=str(ROOT / "data" / "val"))
    t.add_argument("--steps", type=int, default=2000)
    e = sub.add_parser("eval")
    e.add_argument("split_dir")
    d = sub.add_parser("detect")
    d.add_argument("video")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.train, a.val, a.steps)
    elif a.cmd == "eval":
        print(json.dumps(evaluate(CutDetector.load(), load_thumbs(a.split_dir)), indent=1))
    else:
        print(detect_cuts(a.video))
