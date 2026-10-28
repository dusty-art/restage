"""Transformer building blocks for the Restage compiler (MLX).

Attention uses mx.fast.scaled_dot_product_attention (fused, memory-linear backward on the M5).
Masks are additive float arrays broadcastable to [B, heads, Q, K] (0 = keep, -inf/-1e9 = drop).
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

NEG = -1e9


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kv_dim: int | None = None):
        super().__init__()
        self.heads = heads
        self.hd = dim // heads
        kv_dim = kv_dim or dim
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(kv_dim, dim, bias=False)
        self.v = nn.Linear(kv_dim, dim, bias=False)
        self.o = nn.Linear(dim, dim)

    def __call__(self, x, ctx=None, mask=None):
        ctx = x if ctx is None else ctx
        B, Q, _ = x.shape
        K = ctx.shape[1]
        q = self.q(x).reshape(B, Q, self.heads, self.hd).transpose(0, 2, 1, 3)
        k = self.k(ctx).reshape(B, K, self.heads, self.hd).transpose(0, 2, 1, 3)
        v = self.v(ctx).reshape(B, K, self.heads, self.hd).transpose(0, 2, 1, 3)
        if mask is not None:
            mask = mask.astype(q.dtype)
        scale = 1.0 / math.sqrt(self.hd)
        if Q <= 8:
            # The fused kernel's short-query path has no backward; short queries are cheap to do by hand.
            s = (q @ k.swapaxes(-1, -2)) * scale
            if mask is not None:
                s = s + mask
            o = mx.softmax(s.astype(mx.float32), axis=-1).astype(q.dtype) @ v
        else:
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        return self.o(o.transpose(0, 2, 1, 3).reshape(B, Q, -1))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int | None = None, out: int | None = None):
        super().__init__()
        hidden = hidden or dim * 4
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, out or dim)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm self-attention block (optionally with cross-attention to a context)."""

    def __init__(self, dim: int, heads: int, cross: bool = False, ctx_dim: int | None = None):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.cross = cross
        if cross:
            self.nc = nn.LayerNorm(dim)
            self.nctx = nn.LayerNorm(ctx_dim or dim)
            self.xattn = Attention(dim, heads, ctx_dim or dim)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def __call__(self, x, mask=None, ctx=None, ctx_mask=None):
        x = x + self.attn(self.n1(x), mask=mask)
        if self.cross:
            x = x + self.xattn(self.nc(x), self.nctx(ctx), mask=ctx_mask)
        return x + self.mlp(self.n2(x))


class CrossBlock(nn.Module):
    """Queries attend to a context, then an MLP (no self-attention)."""

    def __init__(self, dim: int, heads: int, ctx_dim: int | None = None):
        super().__init__()
        self.nq = nn.LayerNorm(dim)
        self.nctx = nn.LayerNorm(ctx_dim or dim)
        self.xattn = Attention(dim, heads, ctx_dim or dim)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def __call__(self, q, ctx, mask=None):
        q = q + self.xattn(self.nq(q), self.nctx(ctx), mask=mask)
        return q + self.mlp(self.n2(q))


class SlotAttention(nn.Module):
    """Slot attention (Locatello et al. 2020) with learned initial slots and an optional key mask.

    Slots compete for inputs (softmax over slots), so each input is explained by few slots. Returns the
    slots [B, S, d] and the final attention [B, S, N] (each input's share per slot).
    """

    def __init__(self, dim: int, n_slots: int, iters: int = 3, in_dim: int | None = None):
        super().__init__()
        self.iters = iters
        self.scale = dim ** -0.5
        self.init = mx.random.normal((n_slots, dim)) * 0.02
        self.n_in = nn.LayerNorm(in_dim or dim)
        self.k = nn.Linear(in_dim or dim, dim, bias=False)
        self.v = nn.Linear(in_dim or dim, dim, bias=False)
        self.n_s = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.upd = MLP(dim * 2, dim * 2, dim)
        self.n_m = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def __call__(self, x, key_mask=None, slot_bias=None):
        """x [B, N, din]; key_mask bool [B, N]; slot_bias [B, S, d] added to the initial slots."""
        B = x.shape[0]
        x = self.n_in(x)
        k, v = self.k(x), self.v(x)
        s = mx.broadcast_to(self.init[None], (B,) + self.init.shape)
        if slot_bias is not None:
            s = s + slot_bias
        attn = None
        for _ in range(self.iters):
            q = self.q(self.n_s(s))
            logits = (q @ k.transpose(0, 2, 1)) * self.scale              # [B, S, N]
            attn = mx.softmax(logits, axis=1)                             # compete over slots
            if key_mask is not None:
                attn = attn * key_mask[:, None, :]
            w = attn / (attn.sum(-1, keepdims=True) + 1e-6)
            upd = w @ v                                                   # [B, S, d]
            s = s + self.upd(mx.concatenate([s, upd], -1))
            s = s + self.mlp(self.n_m(s))
        return s, attn


def masked_mean(x, mask, axis):
    m = mask.astype(x.dtype)
    while m.ndim < x.ndim:
        m = m[..., None]
    return (x * m).sum(axis) / (m.sum(axis) + 1e-6)
