"""Re-render loss (DESIGN.md §3.3, §4): render the predicted scene through the predicted cameras with the soft
Metal rasteriser and compare it with depth and instance masks.

Per view (a keyframe of a clip): the ground plane under the camera, the 24 boxes of the location slot matched
to the view's location, and every entity slot posed at that keyframe, as a humanoid (people) and as a rounded
box (other classes), faded by presence × class probability. Instance probabilities are rendered as attributes:
channel 0 = set or ground, channel 1 + e = entity slot e. On synthetic clips the targets are the generator's
exact depth and ids; on real video they would come from teachers.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from restage.model import geom
from restage.render import camera, fk, raster, shapes

H, W = 36, 64
N_BOX_V = 56
N_HUM_V = 393


def _take(x, idx, axis=1):
    shape = idx.shape + (1,) * (x.ndim - idx.ndim)
    return mx.take_along_axis(x, idx.reshape(shape), axis=axis)


def render_views(out, ctx: dict, views: np.ndarray):
    """Render the predicted scene at `views` (int [V, 2] of (clip, keyframe)). Returns raster output and C."""
    B, K = out["cam_fov"].shape
    E = out["ent_class"].shape[1]
    NB = out["box_logit"].shape[2]
    vb, vk = mx.array(views[:, 0]), mx.array(views[:, 1])
    V = len(views)
    # cameras of the views, in their location's anchor gauge
    cam_R = ctx["Rg"][vb, vk]
    cam_p = ctx["pg"][vb, vk]
    fov = out["cam_fov"][vb, vk]
    vloc = mx.array(ctx["key_loc"][views[:, 0], views[:, 1]])                       # GT location of the view
    # set: boxes of the slot matched to that GT location, already in that location's gauge
    slot = mx.array(ctx["gt2slot"][views[:, 0], ctx["key_loc"][views[:, 0], views[:, 1]]])
    bc = ctx["box_center_g"][vb, slot]                                               # [V,NB,3]
    bs = out["box_size"][vb, slot]
    by = ctx["box_yaw_g"][vb, slot]
    br = out["box_round"][vb, slot]
    bp = mx.sigmoid(out["box_logit"][vb, slot]) * mx.array(ctx["loc_has_slot"][views[:, 0], ctx["key_loc"][views[:, 0], views[:, 1]]])[:, None]
    bv, bf, _ = shapes.rounded_boxes(bc, bs, by, br)
    gv, gf, _ = shapes.ground_plane(cam_p * mx.array([1.0, 0.0, 1.0]))
    # entities posed at the view's keyframe (gauge of each slot's matched GT location)
    epos = ctx["ent_pos_g"][vb, :, vk]                                               # [V,E,3]
    erot = ctx["ent_rot_g"][vb, :, vk]                                               # [V,E,3,3]
    joints = geom.rot6d_to_mat(out["ent_joint6d"][vb, :, vk])                        # [V,E,21,3,3]
    pelvis = mx.broadcast_to(mx.eye(3), (V, E, 1, 3, 3))
    jrot = mx.concatenate([pelvis, joints], axis=2).reshape(V * E, 22, 3, 3)
    hshape = mx.stack([out["ent_height"][vb] / 1.78, mx.ones((V, E)), mx.ones((V, E)), mx.ones((V, E))], -1)
    hv, hf, _ = shapes.humanoid(hshape.reshape(V * E, 4), jrot, erot.reshape(V * E, 3, 3), epos.reshape(V * E, 3))
    hv = hv.reshape(V, E * N_HUM_V, 3)
    hf = shapes.tile_faces(hf, E, N_HUM_V)
    esize = out["ent_size"][vb]                                                       # [V,E,3]
    eyaw = mx.arctan2(erot[..., 0, 2], erot[..., 2, 2])                               # yaw of the root's +Z axis
    ov, of, _ = shapes.rounded_boxes(epos + mx.stack([mx.zeros_like(esize[..., 1]), esize[..., 1] / 2,
                                                      mx.zeros_like(esize[..., 1])], -1), esize, eyaw,
                                     mx.full(esize.shape[:2], 0.3))
    verts, faces, offs = shapes.merge_meshes((gv, gf), (bv, bf), (hv, hf), (ov, of))
    # opacity per vertex
    pcls = mx.softmax(out["ent_class"][vb].astype(mx.float32), -1)                    # [V,E,5]
    pres = mx.sigmoid(out["ent_present"][vb, :, vk])                                  # [V,E]
    same_loc = (mx.array(ctx["slot_loc"])[vb] == vloc[:, None]).astype(mx.float32)    # entity's location == view's
    op_h = pres * pcls[..., 1] * same_loc
    op_o = pres * (1 - pcls[..., 0] - pcls[..., 1]) * same_loc
    op = mx.concatenate([mx.ones((V, gv.shape[1])), mx.repeat(bp, N_BOX_V, axis=1),
                         mx.repeat(op_h, N_HUM_V, axis=1), mx.repeat(op_o, N_BOX_V, axis=1)], axis=1)
    # instance attributes: channel 0 set/ground, 1 + e entity slot e
    C = 1 + E
    a_set = mx.zeros((V, gv.shape[1] + bv.shape[1], C))
    a_set = mx.concatenate([mx.ones(a_set.shape[:2] + (1,)), a_set[..., 1:]], -1)
    eye = mx.eye(C)[1:]                                                               # [E,C]
    a_h = mx.broadcast_to(mx.repeat(eye, N_HUM_V, axis=0)[None], (V, E * N_HUM_V, C))
    a_o = mx.broadcast_to(mx.repeat(eye, N_BOX_V, axis=0)[None], (V, E * N_BOX_V, C))
    attrs = mx.concatenate([a_set, a_h, a_o], axis=1)
    ndc = camera.project(verts, cam_R, cam_p, fov, W / H)
    return raster.rasterize(ndc, faces, attrs, op, height=H, width=W, sigma=0.6, gamma=0.08), C


def render_loss(out, batch: dict, tgt: dict, ctx: dict, n_views: int = 12, rng=None):
    """Depth, coverage and instance losses on `n_views` random keyframes per clip."""
    rng = rng or np.random.default_rng()
    views = []
    for b in range(batch["key_mask"].shape[0]):
        ks = np.nonzero(batch["key_mask"][b])[0]
        if len(ks):
            for k in rng.choice(ks, min(n_views, len(ks)), replace=False):
                views.append((b, k))
    views = np.array(views, np.int32)
    r, C = render_views(out, ctx, views)
    gd = mx.array(batch["depth"][views[:, 0], views[:, 1]].astype(np.float32))      # [V,H,W]
    gid = batch["ids"][views[:, 0], views[:, 1]].astype(np.int32)
    cover = (gd > 0).astype(mx.float32)
    a = mx.clip(r.alpha, 1e-4, 1 - 1e-4)
    l_alpha = -(cover * mx.log(a) + (1 - cover) * mx.log(1 - a)).mean()
    both = cover * mx.stop_gradient((r.alpha > 0.5).astype(mx.float32))
    l_depth = (both * mx.abs(mx.log(mx.maximum(r.depth, 0.05)) - mx.log(mx.maximum(gd, 0.05)))).sum() / (both.sum() + 1.0)
    # instance target: 0 for the set, matched slot + 1 for entity pixels, ignore unmatched / dropped entities
    slot_of_gt = ctx["slot_of_gt"][views[:, 0]]                                       # [V, Emax] slot or -1
    tid = np.full(gid.shape, -1, np.int32)
    tid[gid == 0] = 0
    for v in range(len(views)):
        for g in np.unique(gid[v]):
            if 0 < g < 255 and g - 1 < slot_of_gt.shape[1] and slot_of_gt[v, g - 1] >= 0:
                tid[v][gid[v] == g] = 1 + slot_of_gt[v, g - 1]
    tid_mx = mx.array(np.maximum(tid, 0))
    valid = mx.array((tid >= 0) & (batch["depth"][views[:, 0], views[:, 1]] > 0))
    p = mx.take_along_axis(r.attrs, tid_mx[..., None], -1)[..., 0]
    l_inst = (-mx.log(mx.clip(p, 1e-4, 1.0)) * valid).sum() / (valid.sum() + 1.0)
    # instance IoU for monitoring
    pred_id = np.array(mx.argmax(r.attrs, -1))
    ious = []
    for v in range(len(views)):
        for g in np.unique(tid[v]):
            if g <= 0:
                continue
            pa, ga = pred_id[v] == g, tid[v] == g
            ious.append((pa & ga).sum() / max(1, (pa | ga).sum()))
    return {"render_alpha": l_alpha, "render_depth": l_depth, "render_inst": l_inst}, \
        {"render_iou": float(np.mean(ious)) if ious else 0.0, "raster_fill": float(r.n_entries) / max(1.0, float(r.capacity))}
