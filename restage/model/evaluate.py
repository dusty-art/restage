"""CutScene benchmark v0 (DESIGN.md §6): score a checkpoint on every window of a split.

    python -m restage.model.evaluate runs/v1/best.safetensors --split test [--baseline per_shot]

Metrics per window, averaged over windows:
* cameras: rotation (deg), position (m) and fov (deg) error, split into the anchor shot of each location and
  later shots of the same location (the cross-cut registration problem);
* locations: pairwise same-place precision / recall / F1 over shots, and whether the number of places is right;
* entities: count accuracy, class accuracy, identity consistency across cuts, root position error (m), joint
  angle error (deg), visibility F1;
* sets: Chamfer distance (m) on matched locations.

`--baseline per_shot` runs the same network on one shot at a time (no cross-shot context). Each shot is then
anchored on its own first keyframe, so later shots of a location get no registration (identity transform).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from restage.model import geom
from restage.model.losses import compute_losses, prepare_targets, regauge, _take
from restage.model.train import Features, load, to_mx

ROOT = Path(__file__).resolve().parents[2]


def _np(x):
    return np.array(x.astype(mx.float32)) if isinstance(x, mx.array) else np.asarray(x)


def _rot_deg(R1, R2):
    tr = np.einsum("...ij,...ij->...", R1, R2)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def window_metrics(out, batch, tgt, b, pairs=None) -> dict:
    """Metrics for clip b of a batch, from model outputs (MLX) and the numpy batch."""
    m = {}
    km = batch["key_mask"][b]
    key_loc = tgt["key_loc"][b]
    key_shot = batch["key_shot"][b]
    anchor_shot = np.zeros(km.shape, bool)
    for l in np.unique(key_loc[km]):
        first = key_shot[np.nonzero(km & (key_loc == l))[0][0]]
        anchor_shot |= (key_shot == first) & (key_loc == l)
    G, t, Rg, pg = regauge(out, mx.array(tgt["key_loc"]), mx.array(tgt["anchor"]))
    Rg, pg = _np(Rg)[b], _np(pg)[b]
    gR, gp = batch["cam_rot"][b], batch["cam_pos"][b]
    rot = _rot_deg(Rg, gR)
    pos = np.linalg.norm(pg - gp, axis=-1)
    fov = np.abs(_np(out["cam_fov"])[b] - batch["cam_fov"][b])
    for name, sel in (("anchor_shot", km & anchor_shot), ("crosscut", km & ~anchor_shot)):
        if sel.any():
            m[f"cam_rot_deg_{name}"] = float(rot[sel].mean())
            m[f"cam_pos_m_{name}"] = float(pos[sel].mean())
    m["cam_fov_deg"] = float(fov[km].mean())

    # locations
    sm = batch["shot_mask"][b]
    assign = _np(out["shot_assign"])[b][sm].argmax(-1)
    gl = batch["shot_loc"][b][sm]
    iu = np.triu_indices(len(gl), 1)
    ps, gs = (assign[:, None] == assign[None])[iu], (gl[:, None] == gl[None])[iu]
    tp, fp, fn = (ps & gs).sum(), (ps & ~gs).sum(), (~ps & gs).sum()
    if len(gl) > 1:
        m["loc_pair_f1"] = float(2 * tp / max(1, 2 * tp + fp + fn)) if (tp + fp + fn) else 1.0
    m["loc_count_ok"] = float(len(np.unique(assign)) == len(np.unique(gl)))

    # entities
    if pairs is not None:
        ent_pairs = pairs[0][b]
        cls_p = _np(out["ent_class"])[b].argmax(-1)
        pres_p = _np(mx.sigmoid(out["ent_present"]))[b] > 0.5
        vis_p = _np(mx.sigmoid(out["ent_visible"]))[b] > 0.5
        n_pred = int(((cls_p > 0) & (pres_p[:, km].any(-1))).sum())
        n_gt = int(batch["ent_mask"][b].sum())
        m["ent_count_ok"] = float(n_pred == n_gt)
        m["ent_count_err"] = float(abs(n_pred - n_gt))
        if ent_pairs:
            cls_ok, vis_tp, vis_fp, vis_fn, pos_e, jnt_e = [], 0, 0, 0, [], []
            Gn, tn = _np(G)[b], _np(t)[b]
            for e, g in ent_pairs:
                cls_ok.append(cls_p[e] == batch["ent_class"][b, g])
                gv = batch["ent_visible"][b, g].astype(bool) & km
                vis_tp += int((vis_p[e] & gv).sum())
                vis_fp += int((vis_p[e] & ~gv & km).sum())
                vis_fn += int((~vis_p[e] & gv).sum())
                lg = batch["ent_loc"][b, g]
                pp = (_np(out["ent_pos"])[b, e] + tn[lg]) @ Gn[lg].T
                pres = batch["ent_present"][b, g].astype(bool) & km
                if pres.any():
                    pos_e.append(np.linalg.norm(pp - tgt["root_pos"][b, g], axis=-1)[pres].mean())
                if batch["ent_class"][b, g] == 1 and gv.any():
                    pJ = _np(geom.rot6d_to_mat(out["ent_joint6d"]))[b, e]
                    jnt_e.append(_rot_deg(pJ, tgt["joints"][b, g])[gv].mean())
            m["ent_class_acc"] = float(np.mean(cls_ok))
            m["ent_vis_f1"] = float(2 * vis_tp / max(1, 2 * vis_tp + vis_fp + vis_fn))
            if pos_e:
                m["ent_pos_m"] = float(np.mean(pos_e))
            if jnt_e:
                m["ent_joint_deg"] = float(np.mean(jnt_e))
            # identity across cuts: per shot, which slot best explains each GT entity's visibility
            vis_soft = _np(mx.sigmoid(out["ent_visible"]))[b]
            consistent = []
            for g in np.nonzero(batch["ent_mask"][b])[0]:
                gv = batch["ent_visible"][b, g].astype(bool) & km
                shots = np.unique(key_shot[gv])
                if len(shots) < 2:
                    continue
                best = [int(np.argmax([(vis_soft[e][gv & (key_shot == s)]).mean() for e in range(vis_soft.shape[0])])) for s in shots]
                consistent.append(len(set(best)) == 1)
            if consistent:
                m["ent_identity_consistent"] = float(np.mean(consistent))
    return m


def run(model, feats_fn, batch, per_shot=False):
    bm = to_mx(batch)
    feats = feats_fn(batch["frames"], batch["key_mask"])
    if not per_shot:
        out = model(feats, bm)
        tgt = prepare_targets(batch)
        _, _, _, pairs = compute_losses(out, batch, bm, tgt)
        return out, tgt, pairs
    # each shot alone: shot ids become 0, every shot is its own location anchored on its first keyframe
    outs = []
    B, K = batch["key_mask"].shape
    for s in range(batch["shot_mask"].shape[1]):
        sel = batch["key_mask"] & (batch["key_shot"] == s)
        if not sel.any():
            continue
        sub = {"key_mask": mx.array(sel), "key_shot": mx.zeros_like(bm["key_shot"]), "key_time": bm["key_time"],
               "shot_mask": mx.array(np.arange(batch["shot_mask"].shape[1])[None].repeat(B, 0) == 0)}
        o = model(feats, sub)
        R = geom.rot6d_to_mat(o["cam_rot6d"])
        p = o["cam_pos"]
        first = np.argmax(sel, axis=1)
        aR = _take(R, mx.array(first[:, None].astype(np.int32)), 1)[:, 0]
        ap = _take(p, mx.array(first[:, None].astype(np.int32)), 1)[:, 0]
        Gs, ts = geom.gauge_from_anchor(aR, ap)
        Rs = Gs[:, None] @ R
        ps = ((p + ts[:, None])[..., None, :] @ Gs[:, None].swapaxes(-1, -2))[..., 0, :]
        outs.append((sel, Rs, ps, o))
    # assemble: each shot keeps its own gauge (no cross-shot registration); express as 6D + pos in a fake output
    R_all = mx.zeros((B, K, 3, 3)) + mx.eye(3)
    p_all = mx.zeros((B, K, 3))
    fov_all = mx.zeros((B, K)) + 50.0
    for sel, Rs, ps, o in outs:
        m = mx.array(sel)[..., None]
        R_all = mx.where(m[..., None], Rs, R_all)
        p_all = mx.where(m, ps, p_all)
        fov_all = mx.where(mx.array(sel), o["cam_fov"], fov_all)
    base = dict(outs[0][3])
    base["cam_rot6d"] = geom.mat_to_rot6d(R_all)
    base["cam_pos"] = p_all
    base["cam_fov"] = fov_all
    S_ = batch["shot_mask"].shape[1]
    base["shot_assign"] = mx.broadcast_to(mx.eye(S_, base["shot_assign"].shape[2] if S_ <= base["shot_assign"].shape[2] else S_)[None, :S_],
                                          (B, S_, base["shot_assign"].shape[2]))
    tgt = prepare_targets(batch)
    return base, tgt, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--baseline", default="", choices=["", "per_shot"])
    ap.add_argument("--max-batches", type=int, default=10 ** 9)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from restage.data.loader import Loader
    model, cfg = load(Path(a.checkpoint))
    feats_fn = Features()
    loader = Loader(Path(a.data) / a.split, batch_size=2, augment=False, seed=0, epoch=True, workers=2)
    acc: dict[str, list[float]] = {}
    n = 0
    for batch in loader:
        out, tgt, pairs = run(model, feats_fn, batch, per_shot=a.baseline == "per_shot")
        bmask = batch.get("batch_mask", np.ones(batch["key_mask"].shape[0], bool))
        for b in range(batch["key_mask"].shape[0]):
            if not bmask[b]:
                continue
            for k, v in window_metrics(out, batch, tgt, b, pairs).items():
                acc.setdefault(k, []).append(v)
        n += 1
        if n >= a.max_batches:
            break
    res = {k: round(float(np.mean(v)), 4) for k, v in sorted(acc.items())}
    res["windows"] = len(acc.get("cam_fov_deg", []))
    res["checkpoint"] = a.checkpoint
    res["split"] = a.split
    res["baseline"] = a.baseline or "full"
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
