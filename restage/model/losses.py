"""Supervised losses for the Restage compiler on synthetic edits (DESIGN.md §3.3).

Every location of the window is re-anchored on its first keyframe, for both prediction and ground
truth (the loader already did it for ground truth), so losses compare relative geometry only. Slots are
matched to ground-truth entities and locations with the Hungarian algorithm on the host, mid-step.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
from scipy.optimize import linear_sum_assignment

from restage.model import geom
from restage.model.net import N_CLASSES, N_JOINTS
from restage import scene as S

OFF0 = mx.array(S.OFFSETS[0])                 # rest offset of the pelvis from the entity root

WEIGHTS = {
    "cam_rot": 2.0, "cam_pos": 10.0, "cam_fov": 1.0,
    "loc_pair": 1.0, "loc_exist": 0.3, "loc_color": 0.5,
    "ent_class": 1.0, "ent_present": 1.0, "ent_visible": 1.0, "ent_pos": 10.0, "ent_rot": 1.0, "ent_joint": 1.0,
    "ent_shape": 0.5, "ent_color": 0.5, "ent_loc": 0.3,
    "set_chamfer": 5.0, "set_rate": 0.02,
    "render_alpha": 0.5, "render_depth": 1.0, "render_inst": 1.0,
}


def _bce(logit, target, mask):
    l = mx.maximum(logit, 0) - logit * target + mx.log1p(mx.exp(-mx.abs(logit)))
    m = mask.astype(l.dtype)
    return (l * m).sum() / (m.sum() + 1e-6)


def _smooth_l1(d, beta=1.0):
    a = mx.abs(d)
    return mx.where(a < beta, 0.5 * a * a / beta, a - 0.5 * beta)


def _masked_mean(x, mask):
    m = mask.astype(x.dtype)
    while m.ndim < x.ndim:
        m = m[..., None]
    m = mx.broadcast_to(m, x.shape)
    return (x * m).sum() / (m.sum() + 1e-6)


def _take(x, idx, axis):
    """x[b, idx[b, ...]] along `axis` (=1) with broadcasting of trailing dims."""
    shape = idx.shape + (1,) * (x.ndim - 2)
    return mx.take_along_axis(x, idx.reshape(shape), axis=axis)


# ---------------------------------------------------------------- ground-truth preparation (numpy)

def prepare_targets(batch: dict) -> dict:
    """Host-side targets: per-keyframe location, anchor indices, root canonicalisation, set point weights."""
    key_shot = batch["key_shot"]
    shot_loc = batch["shot_loc"]
    B, K = key_shot.shape
    key_loc = np.take_along_axis(shot_loc, np.clip(key_shot, 0, shot_loc.shape[1] - 1), axis=1)
    key_loc = np.where(batch["key_mask"], key_loc, 0)
    L = batch["loc_mask"].shape[1]
    anchor = np.zeros((B, L), np.int32)
    for b in range(B):
        for l in range(L):
            hit = np.nonzero(batch["is_anchor"][b] & (key_loc[b] == l) & batch["key_mask"][b])[0]
            anchor[b, l] = hit[0] if len(hit) else 0
    # fold the pelvis' local rotation into the root, so the network predicts root + 21 joints
    pose = batch["ent_pose"]                                   # [B,E,K,22,4]
    root = batch["ent_rot"]                                    # [B,E,K,4]
    root_new = S.quat_mul(root, pose[..., 0, :])
    pos = batch["ent_pos"] + S.quat_rotate(root, S.OFFSETS[0]) - S.quat_rotate(root_new, S.OFFSETS[0])
    joints = S.quat_to_mat(pose[..., 1:, :]).astype(np.float32)        # [B,E,K,21,3,3]
    # set targets: drop the ground plane (modelled separately) and points far from every camera
    pts = batch["loc_points"]                                  # [B,L,N,3]
    w = (np.abs(pts[..., 1]) > 0.03).astype(np.float32)
    for b in range(B):
        for l in range(L):
            cams = batch["cam_pos"][b][(key_loc[b] == l) & batch["key_mask"][b]]
            if len(cams) == 0:
                w[b, l] = 0
                continue
            d = np.linalg.norm(pts[b, l][:, None, :] - cams[None], axis=-1).min(1)
            w[b, l] *= (d < 20.0)
    # spatial scale of each location in the window (metres): positional errors are measured relative to it, so
    # a 1.8 km highway and a 4 m kitchen weigh the same and re-anchoring doesn't turn rotation error into huge losses
    scale = np.full((B, L), 2.0, np.float32)
    for b in range(B):
        for l in range(L):
            sel = (key_loc[b] == l) & batch["key_mask"][b]
            if not sel.any():
                continue
            pts_l = [batch["cam_pos"][b][sel][:, [0, 2]]]
            ents = np.nonzero(batch["ent_mask"][b] & (batch["ent_loc"][b] == l))[0]
            for e in ents:
                pr = batch["ent_present"][b, e].astype(bool) & sel
                if pr.any():
                    pts_l.append(pos[b, e][pr][:, [0, 2]])
            r = np.linalg.norm(np.concatenate(pts_l), axis=-1)
            scale[b, l] = np.clip(np.percentile(r, 90), 2.0, 300.0)
    return {"key_loc": key_loc.astype(np.int32), "anchor": anchor, "root_rot": S.quat_to_mat(root_new).astype(np.float32),
            "root_pos": pos.astype(np.float32), "joints": joints, "pt_w": w, "loc_scale": scale}


# ---------------------------------------------------------------- prediction re-anchoring (MLX)

def regauge(out, key_loc, anchor):
    """Express predicted cameras in each location's anchor gauge. Returns per-location (G, t) and cameras."""
    R = geom.rot6d_to_mat(out["cam_rot6d"])                    # [B,K,3,3]
    p = out["cam_pos"]
    aR = _take(R, anchor, 1)                                   # [B,L,3,3]
    ap = _take(p, anchor, 1)                                   # [B,L,3]
    G, t = geom.gauge_from_anchor(aR, ap)                      # [B,L,3,3], [B,L,3]
    Gk = _take(G, key_loc, 1)
    tk = _take(t, key_loc, 1)
    Rg = Gk @ R
    pg = ((p + tk)[..., None, :] @ Gk.swapaxes(-1, -2))[..., 0, :]
    return G, t, Rg, pg


# ---------------------------------------------------------------- matching (host)

def match(out, batch, tgt, G, t):
    """Hungarian matching of entity slots → GT entities and location slots → GT locations."""
    B = batch["key_mask"].shape[0]
    cls_p = np.array(mx.softmax(out["ent_class"].astype(mx.float32), -1))                     # [B,E,C]
    vis_p = np.array(mx.sigmoid(out["ent_present"].astype(mx.float32)))                      # [B,E,K]
    loc_p = np.array(mx.softmax(out["ent_loc"].astype(mx.float32), -1))                      # [B,E,L]
    assign = np.array(out["shot_assign"].astype(mx.float32))                                 # [B,S,L']
    pos = np.array(out["ent_pos"].astype(mx.float32))                                        # [B,E,K,3]
    Gn, tn = np.array(G), np.array(t)
    ent_pairs, loc_pairs = [], []
    for b in range(B):
        # locations: maximise shot overlap
        S_ = batch["shot_mask"][b]
        gl = batch["shot_loc"][b]
        Lg = int(batch["loc_mask"][b].sum())
        ov = np.zeros((assign.shape[2], Lg))
        for l in range(Lg):
            ov[:, l] = assign[b][S_ & (gl == l)].sum(0)
        r, c = linear_sum_assignment(-ov)
        loc_pairs.append(list(zip(r.tolist(), c.tolist())))
        # entities
        gts = np.nonzero(batch["ent_mask"][b])[0]
        if len(gts) == 0:
            ent_pairs.append([])
            continue
        km = batch["key_mask"][b].astype(np.float32)
        cost = np.zeros((cls_p.shape[1], len(gts)))
        for j, g in enumerate(gts):
            cg = batch["ent_class"][b, g]
            lg = batch["ent_loc"][b, g]
            pres = batch["ent_present"][b, g].astype(np.float32) * km
            # predicted positions re-anchored with the GT entity's location gauge
            pp = (pos[b] + tn[b, lg]) @ Gn[b, lg].T                                         # [E,K,3]
            d = np.linalg.norm(pp - tgt["root_pos"][b, g][None], axis=-1)                   # [E,K]
            dpos = (d * pres).sum(-1) / (pres.sum() + 1e-6) / tgt["loc_scale"][b, lg] * 4.0
            inter = (vis_p[b] * pres).sum(-1)
            union = (vis_p[b] + pres - vis_p[b] * pres).sum(-1) * 1.0 + 1e-6
            cost[:, j] = (-np.log(cls_p[b, :, cg] + 1e-6) + 2.0 * (1 - inter / union)
                          + 0.5 * np.minimum(dpos, 10.0) - 0.5 * np.log(loc_p[b, :, lg] + 1e-6))
        r, c = linear_sum_assignment(cost)
        ent_pairs.append([(int(e), int(gts[j])) for e, j in zip(r, c)])
    return ent_pairs, loc_pairs


def _pairs_to_arrays(pairs, B, n_slots, fill=-1):
    """slot → gt index array [B, n_slots] (-1 = unmatched)."""
    a = np.full((B, n_slots), fill, np.int32)
    for b, ps in enumerate(pairs):
        for s, g in ps:
            a[b, s] = g
    return a


# ---------------------------------------------------------------- box surface samples

def _unit_box_samples(n_per_face=6):
    g = (np.arange(n_per_face) + 0.5) / n_per_face * 2 - 1
    u, v = np.meshgrid(g, g)
    u, v = u.ravel(), v.ravel()
    o = np.ones_like(u)
    faces = [np.stack([o, u, v], 1), np.stack([-o, u, v], 1), np.stack([u, o, v], 1), np.stack([u, -o, v], 1),
             np.stack([u, v, o], 1), np.stack([u, v, -o], 1)]
    return np.concatenate(faces).astype(np.float32)          # [6 n², 3] on the cube surface [-1, 1]³


UNIT_BOX = mx.array(_unit_box_samples(4))                     # 96 samples per box


def box_points(center, size, yaw, roundness):
    """Surface samples of rounded boxes: [..., NB, 3] params → [..., NB, M, 3]."""
    u = UNIT_BOX                                               # [M,3]
    sph = u / mx.linalg.norm(u, axis=-1, keepdims=True)
    r = roundness[..., None, None]
    shape = (1 - r) * u + r * sph                              # blend cube → sphere
    pts = shape * (size[..., None, :] * 0.5)
    c, s = mx.cos(yaw)[..., None], mx.sin(yaw)[..., None]
    x = pts[..., 0] * c + pts[..., 2] * s
    z = -pts[..., 0] * s + pts[..., 2] * c
    return mx.stack([x, pts[..., 1], z], -1) + center[..., None, :]


# ---------------------------------------------------------------- the loss

def compute_losses(out, batch: dict, bm: dict, tgt: dict, render: bool = False, rng=None):
    """out: model outputs; batch: numpy batch; bm: the same batch as MLX arrays; tgt: prepare_targets(batch)."""
    B, K = batch["key_mask"].shape
    E = out["ent_class"].shape[1]
    Lp = out["shot_assign"].shape[2]
    key_mask = bm["key_mask"]
    key_loc = mx.array(tgt["key_loc"])
    anchor = mx.array(tgt["anchor"])
    G, t, Rg, pg = regauge(out, key_loc, anchor)
    ent_pairs, loc_pairs = match(out, batch, tgt, G, t)
    L = {}

    # ---- cameras
    gR, gp = bm["cam_rot"], bm["cam_pos"]
    rot_err = geom.geodesic(Rg, gR)
    L["cam_rot"] = _masked_mean(rot_err, key_mask)
    scale = mx.array(tgt["loc_scale"])
    sk = _take(scale, key_loc, 1)[..., None]
    L["cam_pos"] = _masked_mean(_smooth_l1((pg - gp) / sk, 0.05).sum(-1), key_mask)
    L["cam_fov"] = _masked_mean(mx.abs(mx.log(out["cam_fov"]) - mx.log(bm["cam_fov"])), key_mask)

    # ---- locations: same-place pairs over shots
    a = out["shot_assign"]
    p_same = mx.clip((a[:, :, None, :] * a[:, None, :, :]).sum(-1), 1e-5, 1 - 1e-5)
    same = (bm["shot_loc"][:, :, None] == bm["shot_loc"][:, None, :]).astype(mx.float32)
    sm = bm["shot_mask"]
    pmask = sm[:, :, None] & sm[:, None, :] & ~mx.eye(sm.shape[1], dtype=mx.bool_)[None]
    lp = -(same * mx.log(p_same) + (1 - same) * mx.log(1 - p_same))
    L["loc_pair"] = _masked_mean(lp, pmask)
    loc_slot_gt = _pairs_to_arrays(loc_pairs, B, Lp)                                  # [B,Lp] → GT loc or -1
    exist_t = mx.array((loc_slot_gt >= 0).astype(np.float32))
    L["loc_exist"] = _bce(out["loc_exist"], exist_t, mx.ones_like(exist_t))
    lg = mx.array(np.maximum(loc_slot_gt, 0))
    lvalid = mx.array(loc_slot_gt >= 0)
    col_t = mx.concatenate([_take(bm["loc_sky"], lg, 1), _take(bm["loc_ground"], lg, 1)], -1)
    col_p = mx.concatenate([out["loc_sky"], out["loc_ground"]], -1)
    L["loc_color"] = _masked_mean(mx.abs(col_p - col_t).mean(-1), lvalid)

    # ---- sets: Chamfer between predicted boxes (in the matched GT location's gauge) and GT surface samples
    Gl = _take(G, lg, 1)                                                              # [B,Lp,3,3]
    tl = _take(t, lg, 1)
    psi = geom.heading(_take(geom.rot6d_to_mat(out["cam_rot6d"]), _take(anchor, lg, 1), 1))
    centers = ((out["box_center"] + tl[:, :, None, :])[..., None, :] @ Gl[:, :, None].swapaxes(-1, -2))[..., 0, :]
    pts = box_points(centers, out["box_size"], out["box_yaw"] - psi[..., None], out["box_round"])   # [B,Lp,NB,M,3]
    NB, M = pts.shape[2], pts.shape[3]
    pts = pts.reshape(B, Lp, NB * M, 3)
    w = mx.repeat(mx.sigmoid(out["box_logit"]), M, axis=2)                            # [B,Lp,NB*M]
    gpts = _take(bm["loc_points"], lg, 1)                                             # [B,Lp,N,3]
    gw = _take(mx.array(tgt["pt_w"]), lg, 1) * lvalid[..., None]
    d2 = ((pts[:, :, :, None, :] - gpts[:, :, None, :, :]) ** 2).sum(-1)              # [B,Lp,NBM,N]
    d = mx.sqrt(d2 + 1e-6) / _take(scale, lg, 1)[..., None, None]
    p2g = (w * d.min(-1)).sum(-1) / (w.sum(-1) + 1e-6)
    g2p = (gw * (d + 0.5 * (1 - w)[..., None]).min(-2)).sum(-1) / (gw.sum(-1) + 1e-6)
    has = (gw.sum(-1) > 0)
    L["set_chamfer"] = _masked_mean(p2g + g2p, has)
    L["set_rate"] = _masked_mean(mx.sigmoid(out["box_logit"]).mean(-1), lvalid)

    # ---- entities
    slot_gt = _pairs_to_arrays(ent_pairs, B, E)                                       # [B,E] GT index or -1
    matched = slot_gt >= 0
    g = np.maximum(slot_gt, 0)
    gi = mx.array(g)
    mm = mx.array(matched)
    cls_t = np.where(matched, np.take_along_axis(batch["ent_class"], g, 1), 0)
    logp = out["ent_class"] - mx.logsumexp(out["ent_class"], -1, keepdims=True)
    L["ent_class"] = -mx.take_along_axis(logp, mx.array(cls_t)[..., None], -1).mean()
    pres_t = _take(bm["ent_present"].astype(mx.float32), gi, 1) * mm[..., None]       # [B,E,K]
    vis_t = _take(bm["ent_visible"].astype(mx.float32), gi, 1) * mm[..., None]
    kmE = mx.broadcast_to(key_mask[:, None, :], pres_t.shape)
    L["ent_present"] = _bce(out["ent_present"], pres_t, kmE)
    L["ent_visible"] = _bce(out["ent_visible"], vis_t, kmE)
    eloc = np.take_along_axis(batch["ent_loc"], g, 1)                                 # [B,E] GT location
    Ge = _take(G, mx.array(eloc), 1)
    te = _take(t, mx.array(eloc), 1)
    ppos = ((out["ent_pos"] + te[:, :, None, :])[..., None, :] @ Ge[:, :, None].swapaxes(-1, -2))[..., 0, :]
    gpos = _take(mx.array(tgt["root_pos"]), gi, 1)
    pm = (pres_t > 0) & kmE
    se = _take(scale, mx.array(eloc), 1)[..., None, None]
    L["ent_pos"] = _masked_mean(_smooth_l1((ppos - gpos) / se, 0.05).sum(-1), pm)
    pR = Ge[:, :, None] @ geom.rot6d_to_mat(out["ent_rot6d"])
    gRr = _take(mx.array(tgt["root_rot"]), gi, 1)
    L["ent_rot"] = _masked_mean(geom.geodesic(pR, gRr), pm)
    is_person = mx.array(cls_t == 1)
    pJ = geom.rot6d_to_mat(out["ent_joint6d"])                                        # [B,E,K,21,3,3]
    gJ = _take(mx.array(tgt["joints"]), gi, 1)
    jw = (vis_t + 0.3 * (pres_t - vis_t)) * kmE.astype(mx.float32) * is_person[..., None]
    L["ent_joint"] = (geom.geodesic(pJ, gJ).mean(-1) * jw).sum() / (jw.sum() + 1e-6)
    h_t = _take(bm["ent_height"], gi, 1)
    s_t = _take(bm["ent_size"], gi, 1)
    L["ent_shape"] = _masked_mean(mx.abs(mx.log(out["ent_height"]) - mx.log(mx.maximum(h_t, 0.02)))
                                  + mx.abs(mx.log(out["ent_size"]) - mx.log(mx.maximum(s_t, 0.02))).mean(-1), mm)
    c_t = _take(bm["ent_colors"], gi, 1)
    L["ent_color"] = _masked_mean(mx.abs(out["ent_colors"] - c_t).mean((-2, -1)), mm)
    # entity → location slot: the slot matched to its GT location
    gt2slot = np.full((B, batch["loc_mask"].shape[1]), 0, np.int32)
    for b, ps in enumerate(loc_pairs):
        for s_, l_ in ps:
            gt2slot[b, l_] = s_
    loc_slot_t = np.take_along_axis(gt2slot, eloc, 1)
    lpe = out["ent_loc"] - mx.logsumexp(out["ent_loc"], -1, keepdims=True)
    L["ent_loc"] = _masked_mean(-mx.take_along_axis(lpe, mx.array(loc_slot_t)[..., None], -1)[..., 0], mm)

    extra = {}
    if render:
        from restage.model.render_loss import render_loss
        slot_loc = np.where(matched, eloc, -1)
        slot_of_gt = np.full((B, batch["ent_mask"].shape[1]), -1, np.int32)
        for b_, ps in enumerate(ent_pairs):
            for e_, g_ in ps:
                slot_of_gt[b_, g_] = e_
        has_slot = np.zeros((B, batch["loc_mask"].shape[1]), np.float32)
        for b_, ps in enumerate(loc_pairs):
            for s_, l_ in ps:
                has_slot[b_, l_] = 1.0
        ctx = {"Rg": Rg, "pg": pg, "key_loc": tgt["key_loc"], "gt2slot": gt2slot, "loc_has_slot": has_slot,
               "box_center_g": centers, "box_yaw_g": out["box_yaw"] - psi[..., None], "ent_pos_g": ppos, "ent_rot_g": pR,
               "slot_loc": slot_loc, "slot_of_gt": slot_of_gt}
        rl, extra = render_loss(out, batch, tgt, ctx, rng=rng)
        L.update(rl)
    total = sum(WEIGHTS[k] * v for k, v in L.items())
    # metrics (not part of the loss)
    anchor_mask = bm["is_anchor"] & key_mask
    metrics = {
        "rot_deg": _masked_mean(rot_err, key_mask) * 57.2958,
        "rot_deg_crosscut": _masked_mean(rot_err, key_mask & ~anchor_mask) * 57.2958,
        "pos_m_crosscut": _masked_mean(mx.linalg.norm(pg - gp, axis=-1), key_mask & ~anchor_mask),
        "ent_pos_m": _masked_mean(mx.linalg.norm(ppos - gpos, axis=-1), pm),
        "matched": mx.array(float(matched.sum())),
        **{k: mx.array(v) for k, v in extra.items()},
    }
    return total, L, metrics, (ent_pairs, loc_pairs)
