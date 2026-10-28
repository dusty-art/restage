"""Minimal ViT (pre-norm, patch16, cls token) in functional MLX and in torch.

Both implementations use the same layout: patchify by reshape + Linear
(equivalent to a stride-16 conv), fused SDPA, GELU MLP, 1000-class head,
cross-entropy loss, AdamW.
"""
import math

# ----------------------------------------------------------------------- MLX


def mlx_init(depth=12, dim=384, heads=6, mlp_ratio=4, img=224, patch=16, n_cls=1000):
    import mlx.core as mx
    n_tok = (img // patch) ** 2 + 1
    pd = patch * patch * 3

    def lin(i, o):
        return {"w": mx.random.normal((i, o)) * (1 / math.sqrt(i)), "b": mx.zeros((o,))}

    def ln(d):
        return {"g": mx.ones((d,)), "b": mx.zeros((d,))}

    return {
        "embed": lin(pd, dim),
        "pos": mx.random.normal((1, n_tok, dim)) * 0.02,
        "cls": mx.zeros((1, 1, dim)),
        "blocks": [{"n1": ln(dim), "qkv": lin(dim, 3 * dim), "proj": lin(dim, dim), "n2": ln(dim),
                    "fc1": lin(dim, dim * mlp_ratio), "fc2": lin(dim * mlp_ratio, dim)} for _ in range(depth)],
        "norm": ln(dim),
        "head": lin(dim, n_cls),
    }


def mlx_forward(params, imgs, heads=6, patch=16, checkpoint=False):
    """imgs: [B, H, W, 3] (NHWC)."""
    import mlx.core as mx
    import mlx.nn as nn

    def linear(p, x):
        return x @ p["w"] + p["b"]

    def layernorm(p, x):
        return mx.fast.layer_norm(x, p["g"], p["b"], 1e-6)

    def block(p, x):
        B, N, C = x.shape
        h = linear(p["qkv"], layernorm(p["n1"], x)).reshape(B, N, 3, heads, C // heads).transpose(2, 0, 3, 1, 4)
        a = mx.fast.scaled_dot_product_attention(h[0], h[1], h[2], scale=(C // heads) ** -0.5)
        x = x + linear(p["proj"], a.transpose(0, 2, 1, 3).reshape(B, N, C))
        return x + linear(p["fc2"], nn.gelu(linear(p["fc1"], layernorm(p["n2"], x))))

    blk = mx.checkpoint(block) if checkpoint else block
    B, H, W, C = imgs.shape
    gh, gw = H // patch, W // patch
    x = imgs.reshape(B, gh, patch, gw, patch, C).transpose(0, 1, 3, 2, 4, 5).reshape(B, gh * gw, patch * patch * C)
    x = linear(params["embed"], x)
    cls = mx.broadcast_to(params["cls"], (B, 1, x.shape[-1])).astype(x.dtype)
    x = mx.concatenate([cls, x], axis=1) + params["pos"]
    for p in params["blocks"]:
        x = blk(p, x)
    x = layernorm(params["norm"], x[:, 0])
    return linear(params["head"], x)


# --------------------------------------------------------------------- torch


def torch_model(depth=12, dim=384, heads=6, mlp_ratio=4, img=224, patch=16, n_cls=1000, checkpoint=False):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint as ckpt

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = nn.LayerNorm(dim, eps=1e-6)
            self.qkv = nn.Linear(dim, 3 * dim)
            self.proj = nn.Linear(dim, dim)
            self.n2 = nn.LayerNorm(dim, eps=1e-6)
            self.fc1 = nn.Linear(dim, dim * mlp_ratio)
            self.fc2 = nn.Linear(dim * mlp_ratio, dim)

        def forward(self, x):
            B, N, C = x.shape
            h = self.qkv(self.n1(x)).reshape(B, N, 3, heads, C // heads).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(h[0], h[1], h[2])
            x = x + self.proj(a.transpose(1, 2).reshape(B, N, C))
            return x + self.fc2(F.gelu(self.fc1(self.n2(x))))

    class ViT(nn.Module):
        def __init__(self):
            super().__init__()
            n_tok = (img // patch) ** 2 + 1
            self.embed = nn.Linear(patch * patch * 3, dim)
            self.pos = nn.Parameter(torch.randn(1, n_tok, dim) * 0.02)
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            self.blocks = nn.ModuleList([Block() for _ in range(depth)])
            self.norm = nn.LayerNorm(dim, eps=1e-6)
            self.head = nn.Linear(dim, n_cls)

        def forward(self, imgs):  # [B, 3, H, W]
            B, C, H, W = imgs.shape
            gh, gw = H // patch, W // patch
            x = imgs.reshape(B, C, gh, patch, gw, patch).permute(0, 2, 4, 3, 5, 1).reshape(B, gh * gw, -1)
            x = self.embed(x)
            x = torch.cat([self.cls.expand(B, -1, -1).to(x.dtype), x], 1) + self.pos
            for b in self.blocks:
                x = ckpt(b, x, use_reentrant=False) if checkpoint else b(x)
            return self.head(self.norm(x[:, 0]))

    return ViT()


def n_params(depth, dim, mlp_ratio=4, patch=16, n_cls=1000, img=224):
    per_block = 4 * dim * dim + 2 * dim * dim * mlp_ratio + 9 * dim + dim * mlp_ratio
    return depth * per_block + patch * patch * 3 * dim + dim + ((img // patch) ** 2 + 1) * dim + dim * n_cls + n_cls + 3 * dim


def train_flops_per_image(depth, dim, mlp_ratio=4, img=224, patch=16):
    """6 * MACs (fwd 2x, bwd 4x)."""
    n = (img // patch) ** 2 + 1
    macs = depth * n * (4 * dim * dim + 2 * mlp_ratio * dim * dim + 2 * n * dim) + (n - 1) * patch * patch * 3 * dim
    return 6 * macs
