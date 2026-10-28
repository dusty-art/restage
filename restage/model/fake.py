"""Random batches that follow docs/TRAINING_DATA.md exactly, for smoke-testing the training code path.

They are geometrically valid (unit quaternions, rotation matrices, anchors at the gauge origin) but carry
no learnable signal, so they only test plumbing, shapes and speed.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from restage import scene as S

KMAX, SMAX, LMAX, EMAX = 48, 12, 4, 10


def fake_batch(B: int, rng: np.random.Generator) -> dict:
    b = {}
    b["frames"] = rng.integers(0, 255, (B, KMAX, 126, 224, 3), np.uint8)
    nk = rng.integers(20, KMAX + 1, B)
    b["key_mask"] = np.arange(KMAX)[None] < nk[:, None]
    ns = rng.integers(2, 7, B)
    key_shot = np.zeros((B, KMAX), np.int32)
    for i in range(B):
        cuts = np.sort(rng.choice(np.arange(1, nk[i]), ns[i] - 1, replace=False))
        key_shot[i, :nk[i]] = np.searchsorted(cuts, np.arange(nk[i]), side="right")
    b["key_shot"] = key_shot
    b["key_time"] = (np.arange(KMAX)[None] / 4.0).repeat(B, 0).astype(np.float32)
    b["shot_mask"] = np.arange(SMAX)[None] < ns[:, None]
    nl = rng.integers(1, 3, B)
    b["shot_loc"] = np.stack([np.minimum(rng.integers(0, 2, SMAX), nl[i] - 1) for i in range(B)]).astype(np.int32)
    for i in range(B):
        b["shot_loc"][i, 0] = 0
    b["loc_mask"] = np.arange(LMAX)[None] < nl[:, None]
    key_loc = np.take_along_axis(b["shot_loc"], key_shot, 1)
    b["is_anchor"] = np.zeros((B, KMAX), bool)
    for i in range(B):
        for l in range(nl[i]):
            hit = np.nonzero((key_loc[i] == l) & b["key_mask"][i])[0]
            if len(hit):
                b["is_anchor"][i, hit[0]] = True
    yaw = rng.uniform(-np.pi, np.pi, (B, KMAX))
    q = S.quat_euler(x=rng.uniform(-0.3, 0.1, (B, KMAX)), y=yaw)
    b["cam_rot"] = S.quat_to_mat(q).astype(np.float32)
    b["cam_pos"] = np.concatenate([rng.normal(0, 3, (B, KMAX, 1)), rng.uniform(0.5, 2.5, (B, KMAX, 1)),
                                   rng.normal(0, 3, (B, KMAX, 1))], -1).astype(np.float32)
    b["cam_pos"][b["is_anchor"], 0] = 0
    b["cam_pos"][b["is_anchor"], 2] = 0
    b["cam_fov"] = rng.uniform(20, 70, (B, KMAX)).astype(np.float32)
    b["loc_points"] = rng.normal(0, 3, (B, LMAX, 2048, 3)).astype(np.float32)
    b["loc_sky"] = rng.uniform(0, 1, (B, LMAX, 3)).astype(np.float32)
    b["loc_ground"] = rng.uniform(0, 1, (B, LMAX, 3)).astype(np.float32)
    ne = rng.integers(1, 5, B)
    b["ent_mask"] = np.arange(EMAX)[None] < ne[:, None]
    b["ent_class"] = rng.integers(1, 5, (B, EMAX)).astype(np.int32)
    b["ent_loc"] = np.stack([np.minimum(rng.integers(0, 2, EMAX), nl[i] - 1) for i in range(B)]).astype(np.int32)
    b["ent_present"] = rng.random((B, EMAX, KMAX)) < 0.8
    b["ent_visible"] = b["ent_present"] & (rng.random((B, EMAX, KMAX)) < 0.8)
    b["ent_pos"] = rng.normal(0, 2, (B, EMAX, KMAX, 3)).astype(np.float32)
    b["ent_rot"] = S.quat_euler(y=rng.uniform(-np.pi, np.pi, (B, EMAX, KMAX))).astype(np.float32)
    pose = S.quat_euler(x=rng.normal(0, 0.3, (B, EMAX, KMAX, 22)), z=rng.normal(0, 0.2, (B, EMAX, KMAX, 22)))
    b["ent_pose"] = pose.astype(np.float32)
    b["ent_height"] = rng.uniform(0.2, 1.9, (B, EMAX)).astype(np.float32)
    b["ent_size"] = rng.uniform(0.1, 2.0, (B, EMAX, 3)).astype(np.float32)
    b["ent_colors"] = rng.uniform(0, 1, (B, EMAX, 5, 3)).astype(np.float32)
    b["depth"] = rng.uniform(0, 10, (B, KMAX, 36, 64)).astype(np.float16)
    b["ids"] = rng.integers(0, 4, (B, KMAX, 36, 64), np.uint8)
    return b


def fake_loader(B: int, seed: int):
    rng = np.random.default_rng(seed)
    while True:
        yield fake_batch(B, rng)


def fake_features(frames, key_mask):
    B, K = key_mask.shape
    return mx.random.normal((B, K, 384)), mx.random.normal((B, K, 9, 16, 384))
