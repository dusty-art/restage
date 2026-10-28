"""DINOv2 ViT-S/14 (facebook/dinov2-small, Apache-2.0) in MLX, for 224x126 keyframes.

    model = DinoV2S.load()                       # fp32; DinoV2S.load(dtype=mx.bfloat16) for bf16
    cls, patches = model(images)                 # images u8 or f32 [N, 126, 224, 3] RGB
    # cls [N, 384], patches [N, 9, 16, 384]

* u8 input is scaled by 1/255; f32 input is taken as RGB in 0..1. ImageNet mean/std normalisation
  is applied inside the model (in fp32).
* bf16 (`dtype=mx.bfloat16`) is mixed precision: qkv / proj / MLP weights, matmuls and attention in bf16;
  patch embedding, layer norms and the residual stream in fp32; outputs in bf16.
* The 37x37 pretrained position embedding is resampled once at load time to the 9x16 patch grid with
  the same bicubic (A = -0.75, align_corners=False) interpolation that HF transformers' Dinov2Model
  uses, so outputs match `Dinov2Model(pixel_values).last_hidden_state` (after the final layernorm).
* The patch embedding is a reshape + matmul (equivalent to the 14x14 stride-14 conv).
* The forward pass is wrapped in `mx.compile` (shapes are static: 145 tokens).

Weights: `python -m restage.data.dinov2 --convert` downloads facebook/dinov2-small from Hugging Face
into checkpoints/dinov2-small/hf/ and writes checkpoints/dinov2-small/dinov2_small_mlx.safetensors.
`python -m restage.data.dinov2 --parity --bench` checks against HF transformers (torch, CPU fp32) and
times batch-64 inference.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "checkpoints" / "dinov2-small"
HF_DIR = CKPT_DIR / "hf"
MLX_WEIGHTS = CKPT_DIR / "dinov2_small_mlx.safetensors"
HF_REPO = "facebook/dinov2-small"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH = 14
DIM = 384
HEADS = 6
DEPTH = 12
IMG_H, IMG_W = 126, 224
GRID = (IMG_H // PATCH, IMG_W // PATCH)      # (9, 16)


# ---------------------------------------------------------------- bicubic resampling (torch-exact)

def _cubic_weights(t, A=-0.75):
    """Weights of the 4 taps at offsets -1, 0, 1, 2 for fractional position t (PyTorch's convention)."""
    def c1(x):   # |x| <= 1
        return ((A + 2) * x - (A + 3)) * x * x + 1

    def c2(x):   # 1 < |x| < 2
        return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A
    return np.array([c2(t + 1), c1(t), c1(1 - t), c2(2 - t)])


def bicubic_matrix(n_in: int, n_out: int) -> np.ndarray:
    """[n_out, n_in] matrix equal to torch.nn.functional.interpolate(mode='bicubic', align_corners=False)
    along one axis (no antialias; borders clamped)."""
    M = np.zeros((n_out, n_in), np.float64)
    scale = n_in / n_out
    for i in range(n_out):
        src = (i + 0.5) * scale - 0.5
        i0 = math.floor(src)
        t = src - i0
        w = _cubic_weights(t)
        for k in range(4):
            j = min(max(i0 - 1 + k, 0), n_in - 1)
            M[i, j] += w[k]
    return M


def resample_pos_embed(pos: np.ndarray, grid=GRID) -> np.ndarray:
    """pos [1, 1 + S*S, D] -> [1, 1 + gh*gw, D] (class token kept)."""
    pos = np.asarray(pos, np.float64)
    cls, patch = pos[:, :1], pos[:, 1:]
    S = int(round(math.sqrt(patch.shape[1])))
    D = patch.shape[-1]
    g = patch.reshape(S, S, D)
    gh, gw = grid
    if (gh, gw) == (S, S):
        out = g
    else:
        Mh, Mw = bicubic_matrix(S, gh), bicubic_matrix(S, gw)
        out = np.einsum("ai,ijd,bj->abd", Mh, g, Mw)
    return np.concatenate([cls, out.reshape(1, gh * gw, D)], 1).astype(np.float32)


# ---------------------------------------------------------------- model

class Block(nn.Module):
    def __init__(self, dim=DIM, heads=HEADS, mlp_ratio=4, eps=1e-6):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ls1 = mx.ones((dim,))
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.ls2 = mx.ones((dim,))

    def __call__(self, x, cdt=mx.float32):
        """x: fp32 residual stream [B, L, D]; matmuls and attention run in `cdt`."""
        B, L, D = x.shape
        h = self.norm1(x).astype(cdt)
        qkv = self.qkv(h).reshape(B, L, 3, self.heads, D // self.heads).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=(D // self.heads) ** -0.5)
        a = a.transpose(0, 2, 1, 3).reshape(B, L, D)
        x = x + (self.proj(a) * self.ls1).astype(mx.float32)
        h = self.fc2(nn.gelu(self.fc1(self.norm2(x).astype(cdt))))
        return x + (h * self.ls2).astype(mx.float32)


class DinoV2S(nn.Module):
    """DINOv2 ViT-S/14 encoder for [N, 126, 224, 3] RGB images."""

    def __init__(self, grid=GRID):
        super().__init__()
        self.grid = tuple(grid)
        gh, gw = self.grid
        self.patch_w = mx.zeros((PATCH * PATCH * 3, DIM))     # (ph, pw, c) -> D
        self.patch_b = mx.zeros((DIM,))
        self.cls_token = mx.zeros((1, 1, DIM))
        self.pos_embed = mx.zeros((1, 1 + gh * gw, DIM))
        self.blocks = [Block() for _ in range(DEPTH)]
        self.norm = nn.LayerNorm(DIM, eps=1e-6)
        self._mean = mx.array(np.array(IMAGENET_MEAN, np.float32))
        self._std = mx.array(np.array(IMAGENET_STD, np.float32))
        self._fwd = None
        self.dtype = mx.float32
        self.freeze()

    # -------- loading
    @staticmethod
    def load(path: str | Path | None = None, dtype=mx.float32, grid=GRID, compile: bool = True) -> "DinoV2S":
        """Load converted MLX weights (converting from the HF checkpoint on first use)."""
        path = Path(path) if path else MLX_WEIGHTS
        if not path.exists():
            if path != MLX_WEIGHTS:
                raise FileNotFoundError(path)
            convert()
        w = mx.load(str(path))
        m = DinoV2S(grid)
        pos = resample_pos_embed(np.array(w.pop("pos_embed_full")), m.grid)
        w["pos_embed"] = mx.array(pos)
        m.load_weights(list(w.items()), strict=True)
        m.set_dtype(dtype)
        m._compile = compile
        m.eval()
        return m

    def set_dtype(self, dtype):
        """Compute dtype (bf16 / fp16 / fp32): the weights of the linear layers (qkv, proj, fc1, fc2) are cast to it;
        the patch embedding, layer norms, layer scales and the residual stream stay fp32 (mixed precision: as
        fast as pure bf16 here and ~10x closer to the fp32 reference on the worst tokens). Rebuilds the compiled
        forward."""
        self.unfreeze()
        p = _tree_cast(self.parameters(), mx.float32)
        for b in p["blocks"]:
            for name in ("qkv", "proj", "fc1", "fc2"):
                b[name] = _tree_cast(b[name], dtype)
        self.update(p)
        self.freeze()
        self._mean, self._std = self._mean.astype(mx.float32), self._std.astype(mx.float32)
        self.dtype = dtype
        self._fwd = None
        return self

    # -------- forward
    def _forward(self, x):
        """x: [N, H, W, 3] float32 normalised."""
        N = x.shape[0]
        gh, gw = self.grid
        p = x.reshape(N, gh, PATCH, gw, PATCH, 3).transpose(0, 1, 3, 2, 4, 5).reshape(N, gh * gw, PATCH * PATCH * 3)
        t = p @ self.patch_w + self.patch_b
        cls = mx.broadcast_to(self.cls_token, (N, 1, DIM))
        t = mx.concatenate([cls, t], 1) + self.pos_embed
        for b in self.blocks:
            t = b(t, self.dtype)
        t = self.norm(t).astype(self.dtype)
        return t[:, 0], t[:, 1:].reshape(N, gh, gw, DIM)

    def normalize(self, images):
        x = images if isinstance(images, mx.array) else mx.array(np.asarray(images))
        if x.dtype == mx.uint8:
            x = x.astype(mx.float32) * (1.0 / 255.0)
        else:
            x = x.astype(mx.float32)
        return (x - self._mean) / self._std

    def __call__(self, images):
        """images u8 or float [N, 126, 224, 3] RGB (float in 0..1). Returns (cls [N, 384], patches [N, 9, 16, 384])
        in the model dtype."""
        x = self.normalize(images)
        gh, gw = self.grid
        if x.shape[1:] != (gh * PATCH, gw * PATCH, 3):
            raise ValueError(f"expected [N, {gh * PATCH}, {gw * PATCH}, 3] images, got {tuple(x.shape)}")
        if self._fwd is None:
            self._fwd = mx.compile(self._forward) if getattr(self, "_compile", True) else self._forward
        return self._fwd(x)

    def encode(self, frames: np.ndarray, batch: int = 64):
        """Numpy convenience: run in chunks, return numpy fp32 (cls, patches)."""
        cs, ps = [], []
        for i in range(0, len(frames), batch):
            c, p = self(frames[i:i + batch])
            mx.eval(c, p)
            cs.append(np.array(c.astype(mx.float32)))
            ps.append(np.array(p.astype(mx.float32)))
        return np.concatenate(cs), np.concatenate(ps)


def _tree_cast(tree, dtype):
    if isinstance(tree, dict):
        return {k: _tree_cast(v, dtype) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_tree_cast(v, dtype) for v in tree]
    if isinstance(tree, mx.array) and mx.issubdtype(tree.dtype, mx.floating):
        return tree.astype(dtype)
    return tree


# ---------------------------------------------------------------- conversion

def download(hf_dir: Path = HF_DIR) -> Path:
    from huggingface_hub import snapshot_download
    hf_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(HF_REPO, local_dir=str(hf_dir), allow_patterns=["*.json", "*.safetensors", "*.md"])
    return hf_dir


def convert(hf_dir: Path = HF_DIR, out: Path = MLX_WEIGHTS) -> Path:
    """HF Dinov2Model safetensors -> MLX safetensors with our parameter names (fp32)."""
    src = hf_dir / "model.safetensors"
    if not src.exists():
        download(hf_dir)
    from safetensors.numpy import load_file
    w = load_file(str(src))
    o = {}
    pw = w["embeddings.patch_embeddings.projection.weight"]          # [D, 3, 14, 14]
    o["patch_w"] = pw.transpose(2, 3, 1, 0).reshape(PATCH * PATCH * 3, DIM)   # (ph, pw, c) x D
    o["patch_b"] = w["embeddings.patch_embeddings.projection.bias"]
    o["cls_token"] = w["embeddings.cls_token"]
    o["pos_embed_full"] = w["embeddings.position_embeddings"]
    for i in range(DEPTH):
        p = f"encoder.layer.{i}."
        q = f"blocks.{i}."
        a = p + "attention.attention."
        o[q + "qkv.weight"] = np.concatenate([w[a + n + ".weight"] for n in ("query", "key", "value")], 0)
        o[q + "qkv.bias"] = np.concatenate([w[a + n + ".bias"] for n in ("query", "key", "value")], 0)
        o[q + "proj.weight"] = w[p + "attention.output.dense.weight"]
        o[q + "proj.bias"] = w[p + "attention.output.dense.bias"]
        o[q + "ls1"] = w[p + "layer_scale1.lambda1"]
        o[q + "ls2"] = w[p + "layer_scale2.lambda1"]
        for n in ("norm1", "norm2"):
            o[q + n + ".weight"] = w[p + n + ".weight"]
            o[q + n + ".bias"] = w[p + n + ".bias"]
        o[q + "fc1.weight"] = w[p + "mlp.fc1.weight"]
        o[q + "fc1.bias"] = w[p + "mlp.fc1.bias"]
        o[q + "fc2.weight"] = w[p + "mlp.fc2.weight"]
        o[q + "fc2.bias"] = w[p + "mlp.fc2.bias"]
    o["norm.weight"] = w["layernorm.weight"]
    o["norm.bias"] = w["layernorm.bias"]
    out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out), {k: mx.array(np.ascontiguousarray(v, np.float32)) for k, v in o.items()},
                        metadata={"source": HF_REPO, "license": "Apache-2.0"})
    return out


# ---------------------------------------------------------------- parity and speed

def parity(n: int = 4, seed: int = 0, images: np.ndarray | None = None) -> dict:
    """Max abs difference vs HF transformers Dinov2Model (torch CPU fp32) on the same input."""
    import torch
    from transformers import Dinov2Model
    rng = np.random.default_rng(seed)
    if images is None:
        # smooth random images (more natural statistics than white noise) plus one noise image
        base = rng.uniform(0, 255, (n, 9, 16, 3))
        import cv2
        images = np.stack([cv2.resize(b.astype(np.float32), (IMG_W, IMG_H), interpolation=cv2.INTER_CUBIC) for b in base])
        images[-1] = rng.uniform(0, 255, images[-1].shape)
        images = np.clip(images, 0, 255).astype(np.uint8)
    hf = Dinov2Model.from_pretrained(str(HF_DIR), attn_implementation="eager").eval().float()
    x = (images.astype(np.float32) / 255 - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    with torch.no_grad():
        o = hf(pixel_values=torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()).last_hidden_state.numpy()
    ref_cls, ref_patch = o[:, 0], o[:, 1:].reshape(len(images), *GRID, DIM)
    out = {"n": len(images), "ref_abs_max": float(np.abs(o).max())}
    for name, dt in (("fp32", mx.float32), ("bf16", mx.bfloat16)):
        m = DinoV2S.load(dtype=dt)
        c, p = m.encode(images)
        pc = _cos(p.reshape(-1, DIM), ref_patch.reshape(-1, DIM))
        out[name] = {"cls_max_abs": float(np.abs(c - ref_cls).max()), "patch_max_abs": float(np.abs(p - ref_patch).max()),
                     "patch_mean_abs": float(np.abs(p - ref_patch).mean()),
                     "cls_cos_min": float(_cos(c, ref_cls).min()), "patch_cos_min": float(pc.min()),
                     "patch_cos_p01": float(np.percentile(pc, 1)), "patch_cos_median": float(np.median(pc))}
    return out


def _cos(a, b):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def bench(batch: int = 64, iters: int = 20) -> dict:
    out = {}
    rng = np.random.default_rng(0)
    x = mx.array(rng.integers(0, 256, (batch, IMG_H, IMG_W, 3), dtype=np.uint8))
    for name, dt in (("fp32", mx.float32), ("bf16", mx.bfloat16)):
        m = DinoV2S.load(dtype=dt)
        for _ in range(3):
            mx.eval(*m(x))
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            mx.eval(*m(x))
            ts.append(time.perf_counter() - t0)
        med = float(np.median(ts))
        out[name] = {"batch": batch, "ms_per_batch": med * 1e3, "ms_per_frame": med * 1e3 / batch,
                     "frames_per_s": batch / med}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--frames", nargs="*", help="packed clips whose keyframes to use for --parity (default: synthetic)")
    a = ap.parse_args()
    import json
    mx.set_memory_limit(6 << 30)
    if a.convert or not MLX_WEIGHTS.exists():
        print("wrote", convert())
    if a.parity:
        imgs = None
        if a.frames:
            from .pack import load_clip
            imgs = np.concatenate([load_clip(f)["frames"][::10][:4] for f in a.frames])
        print(json.dumps(parity(images=imgs), indent=1))
    if a.bench:
        print(json.dumps(bench(a.batch), indent=1))
