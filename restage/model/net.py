"""The Restage compiler: frozen DINOv2 keyframe tokens → cameras, locations with sets, and entities.

Shapes follow docs/TRAINING_DATA.md: B clips, K keyframes, P = 9×16 patches, S shots, L location slots,
E entity slots, J = 22 humanoid joints, NB boxes per set. All outputs are in one frame per location chosen
by the network; losses and decoding re-anchor each location on its first keyframe (geom.gauge_from_anchor).
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from restage.model.layers import NEG, Block, CrossBlock, SlotAttention, masked_mean

N_CLASSES = 5          # none, person, animal, vehicle, object
N_JOINTS = 22
N_REGIONS = 5          # skin, top, bottom, shoes, hair (objects use region 0)
IDENT6 = mx.array([1.0, 0, 0, 0, 1.0, 0])   # 6D of the identity rotation, added so heads start near identity


@dataclass
class Config:
    dim: int = 384
    heads: int = 6
    feat_dim: int = 384
    grid: tuple[int, int] = (9, 16)
    shot_blocks: int = 6
    temporal_blocks: int = 2
    n_locs: int = 4
    n_ents: int = 10
    n_boxes: int = 24
    box_blocks: int = 2
    slot_iters: int = 3
    max_shots: int = 12


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class Restage(nn.Module):
    def __init__(self, cfg: Config = Config()):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        gh, gw = cfg.grid
        self.patch_in = nn.Linear(cfg.feat_dim, d)
        self.cls_in = nn.Linear(cfg.feat_dim, d)
        self.pos = mx.random.normal((gh * gw, d)) * 0.02
        self.cam_tok = mx.random.normal((d,)) * 0.02
        self.time_in = nn.Linear(16, d)
        self.shot_emb = mx.random.normal((cfg.max_shots, d)) * 0.02
        # shot encoder: spatial (within keyframe) then temporal (camera tokens within the shot)
        self.spatial = [Block(d, cfg.heads) for _ in range(cfg.shot_blocks)]
        self.temporal = [Block(d, cfg.heads) for _ in range(cfg.shot_blocks)]
        self.shot_norm = nn.LayerNorm(d)
        # locations
        self.loc_slots = SlotAttention(d, cfg.n_locs, cfg.slot_iters)
        self.loc_gather = CrossBlock(d, cfg.heads)
        self.loc_exist = nn.Linear(d, 1)
        # entities
        self.ent_slots = SlotAttention(d, cfg.n_ents, cfg.slot_iters)
        self.ent_read = CrossBlock(d, cfg.heads)                # (slot ⊕ keyframe) queries read that keyframe
        self.ent_temporal = [Block(d, cfg.heads) for _ in range(cfg.temporal_blocks)]
        # cameras: each camera token attends to locations and entity identities
        self.cam_ctx = [CrossBlock(d, cfg.heads) for _ in range(2)]
        # heads
        self.cam_head = _mlp(d, d, 6 + 3 + 1)
        self.loc_head = _mlp(d, d, 6)                            # sky rgb, ground rgb
        self.box_q = mx.random.normal((cfg.n_boxes, d)) * 0.02
        self.box_dec = [CrossBlock(d, cfg.heads) for _ in range(cfg.box_blocks)]
        self.box_head = _mlp(d, d, 1 + 3 + 3 + 2 + 1 + 3)         # presence, centre, log size, yaw (cos, sin), roundness, rgb
        self.ent_static = _mlp(d, d, N_CLASSES + 1 + 3 + N_REGIONS * 3)   # class, log height, log size, colours
        self.ent_loc_q = nn.Linear(d, d)
        self.ent_state = _mlp(d, d, 2 + 3 + 6 + (N_JOINTS - 1) * 6)       # present, visible, root pos, root rot, joints

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _time_features(t):
        """Sinusoidal features of time in seconds [B, K] → [B, K, 16]."""
        freqs = mx.array([0.25, 0.5, 1, 2, 4, 8, 16, 32], dtype=mx.float32)
        a = t[..., None] * freqs * 6.2831853
        return mx.concatenate([mx.sin(a), mx.cos(a)], -1)

    def __call__(self, feats, batch):
        """feats: (cls [B,K,384], patches [B,K,9,16,384]) from DINOv2; batch: loader arrays (masks, shot ids, times)."""
        cfg = self.cfg
        d = cfg.dim
        cls, patches = feats
        B, K = cls.shape[:2]
        P = cfg.grid[0] * cfg.grid[1]
        key_mask = batch["key_mask"]
        key_shot = batch["key_shot"]
        shot_mask = batch["shot_mask"]
        S = shot_mask.shape[1]

        # ---------------------------------------------------------- tokens
        x = self.patch_in(patches.reshape(B, K, P, -1)) + self.pos                       # [B,K,P,d]
        shot_onehot = (key_shot[..., None] == mx.arange(S)).astype(x.dtype)              # [B,K,S]
        tfeat = self.time_in(self._time_features(batch["key_time"]))
        cam = self.cam_tok + self.cls_in(cls) + tfeat + shot_onehot @ self.shot_emb[:S]   # [B,K,d]
        tok = mx.concatenate([cam[:, :, None], x], axis=2).reshape(B * K, P + 1, d)

        # temporal mask: camera tokens attend to camera tokens of the same shot
        same = (key_shot[:, :, None] == key_shot[:, None, :]) & key_mask[:, :, None] & key_mask[:, None, :]
        eye = mx.eye(K, dtype=mx.bool_)[None]
        tmask = mx.where(same | eye, 0.0, NEG)[:, None]                                  # [B,1,K,K]

        for sp, tp in zip(self.spatial, self.temporal):
            tok = sp(tok)
            t = tok.reshape(B, K, P + 1, d)
            c = tp(t[:, :, 0], mask=tmask)
            tok = mx.concatenate([c[:, :, None], t[:, :, 1:]], axis=2).reshape(B * K, P + 1, d)
        tok = tok.reshape(B, K, P + 1, d)
        cam = self.shot_norm(tok[:, :, 0])                                                # [B,K,d]
        pat = tok[:, :, 1:]                                                               # [B,K,P,d]

        # ---------------------------------------------------------- shots → locations
        km = key_mask.astype(cam.dtype)
        w = shot_onehot * km[..., None]                                                    # [B,K,S]
        shot_tok = (w.swapaxes(1, 2) @ cam) / (w.sum(1)[..., None] + 1e-6)                 # [B,S,d]
        loc, loc_attn = self.loc_slots(shot_tok, key_mask=shot_mask)                       # [B,L,d], [B,L,S]
        assign = loc_attn.swapaxes(1, 2)                                                   # [B,S,L] soft, sums to 1 over L
        # each location gathers evidence from the patches of its shots
        key_assign = shot_onehot @ assign                                                  # [B,K,L]
        flat = pat.reshape(B, K * P, d)
        bias = mx.log(mx.repeat(key_assign, P, axis=1) + 1e-4).swapaxes(1, 2)            # [B,L,K*P]
        kvalid = mx.repeat(key_mask, P, axis=1)[:, None, :]
        lmask = mx.where(kvalid, bias, NEG)[:, None]                                       # [B,1,L,K*P]
        loc = self.loc_gather(loc, flat, mask=lmask)

        # ---------------------------------------------------------- entities
        pmask = mx.repeat(key_mask, P, axis=1)
        ent, ent_attn = self.ent_slots(flat, key_mask=pmask)                               # [B,E,d], [B,E,K*P]
        E = ent.shape[1]
        q = (ent[:, None, :, :] + cam[:, :, None, :]).reshape(B * K, E, d)
        st = self.ent_read(q, pat.reshape(B * K, P, d)).reshape(B, K, E, d)
        st = st.transpose(0, 2, 1, 3).reshape(B * E, K, d)
        kmask_e = mx.repeat(key_mask, E, axis=0)
        emask = mx.where(kmask_e[:, None, :] & kmask_e[:, :, None], 0.0, NEG)[:, None]
        emask = mx.where(mx.eye(K, dtype=mx.bool_)[None, None], 0.0, emask)
        for blk in self.ent_temporal:
            st = blk(st, mask=emask)
        st = st.reshape(B, E, K, d)

        # ---------------------------------------------------------- cameras see locations and entities
        ctx = mx.concatenate([loc, ent], axis=1)                                           # [B,L+E,d]
        for blk in self.cam_ctx:
            cam = blk(cam, ctx)

        # ---------------------------------------------------------- heads
        c = self.cam_head(cam)
        out = {
            "cam_rot6d": c[..., :6] + IDENT6,
            "cam_pos": c[..., 6:9],
            "cam_fov": 8.0 + 112.0 * mx.sigmoid(c[..., 9] + 0.3),                          # degrees, ~55° at init
            "shot_assign": assign,                                                         # [B,S,L]
            "loc_exist": self.loc_exist(loc)[..., 0],
        }
        lh = self.loc_head(loc)
        out["loc_sky"] = mx.sigmoid(lh[..., :3])
        out["loc_ground"] = mx.sigmoid(lh[..., 3:])
        L = loc.shape[1]
        bq = mx.broadcast_to(self.box_q[None, None], (B, L) + self.box_q.shape) + loc[:, :, None]
        bq = bq.reshape(B * L, cfg.n_boxes, d)
        bctx = mx.repeat(flat, L, axis=0)
        bmask = lmask[:, 0].reshape(B * L, 1, 1, K * P)
        for blk in self.box_dec:
            bq = blk(bq, bctx, mask=bmask)
        bh = self.box_head(bq).reshape(B, L, cfg.n_boxes, -1)
        out["box_logit"] = bh[..., 0]
        out["box_center"] = bh[..., 1:4]
        out["box_size"] = mx.exp(mx.clip(bh[..., 4:7], -4, 3.5))
        out["box_yaw"] = mx.arctan2(bh[..., 8], bh[..., 7] + 1e-6)
        out["box_round"] = mx.sigmoid(bh[..., 9] - 1.0)
        out["box_color"] = mx.sigmoid(bh[..., 10:13])

        es = self.ent_static(ent)
        out["ent_class"] = es[..., :N_CLASSES]
        out["ent_height"] = mx.exp(mx.clip(es[..., N_CLASSES], -3, 2.5) + 0.4)
        out["ent_size"] = mx.exp(mx.clip(es[..., N_CLASSES + 1:N_CLASSES + 4], -4, 3) - 0.5)
        out["ent_colors"] = mx.sigmoid(es[..., N_CLASSES + 4:]).reshape(B, E, N_REGIONS, 3)
        out["ent_loc"] = (self.ent_loc_q(ent) @ loc.swapaxes(1, 2)) * (d ** -0.5)          # [B,E,L]
        sh = self.ent_state(st)                                                            # [B,E,K,...]
        out["ent_present"] = sh[..., 0]
        out["ent_visible"] = sh[..., 1]
        out["ent_pos"] = sh[..., 2:5]
        out["ent_rot6d"] = sh[..., 5:11] + IDENT6
        out["ent_joint6d"] = sh[..., 11:].reshape(B, E, K, N_JOINTS - 1, 6) + IDENT6
        out["ent_attn"] = ent_attn
        return out


def count_params(model) -> int:
    from mlx.utils import tree_flatten
    return sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
