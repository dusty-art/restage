"""Debug drawings of packed clips and loader batches: keyframes with projected entity roots / centres,
ids overlay and camera info (PNG)."""
from __future__ import annotations

import cv2
import numpy as np

from .pack import LAB_H, LAB_W, entity_centres, project

PALETTE = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180],
                    [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255]],
                   np.uint8)


def draw_keyframe(frame, ids, depth, cam_rot, cam_pos, fov, ent_pos, ent_centre, visible, scale=3, text="", show=None):
    """frame [126, 224, 3]; ids/depth [36, 64]; ent_* [E, 3]. Returns an RGB image: frame with projected roots
    (circles) and centres (crosses) | ids overlay | depth."""
    H, W = frame.shape[:2]
    img = cv2.resize(frame, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST).copy()
    col = np.zeros((LAB_H, LAB_W, 3), np.uint8)
    for e in range(1, int(ids.max()) + 1):
        col[ids == e] = PALETTE[(e - 1) % len(PALETTE)]
    col[ids == 255] = 128
    ov = cv2.resize(col, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    blend = (0.45 * img + 0.55 * ov).astype(np.uint8)
    blend[ov.sum(-1) == 0] = (img[ov.sum(-1) == 0] * 0.6).astype(np.uint8)
    d = depth.astype(np.float32)
    dv = np.where(d > 0, 1 - np.clip(np.log1p(d) / np.log1p(max(d.max(), 1)), 0, 1), 0)
    dimg = cv2.resize((dv * 255).astype(np.uint8), (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    dimg = cv2.applyColorMap(dimg, cv2.COLORMAP_VIRIDIS)[..., ::-1]
    for canvas in (img, blend):
        for e in range(len(ent_pos)):
            if show is not None and not show[e]:
                continue
            c = tuple(int(x) for x in PALETTE[e % len(PALETTE)])
            for pt, mark in ((ent_pos[e], "o"), (ent_centre[e], "x")):
                u, v, z = project(pt[None], cam_rot, cam_pos, fov, W * scale, H * scale)
                if z[0] <= 0.05 or not (-50 < u[0] < W * scale + 50 and -50 < v[0] < H * scale + 50):
                    continue
                p = (int(round(u[0])), int(round(v[0])))
                if mark == "o":
                    cv2.circle(canvas, p, 7, c, 2 if visible[e] else 1)
                else:
                    cv2.drawMarker(canvas, p, c, cv2.MARKER_CROSS, 14, 2)
    fwd = -cam_rot[:, 2]
    info = f"{text} pos=({cam_pos[0]:.2f},{cam_pos[1]:.2f},{cam_pos[2]:.2f}) fwd=({fwd[0]:.2f},{fwd[1]:.2f},{fwd[2]:.2f}) fov={fov:.1f}"
    out = np.concatenate([img, blend, dimg], 1)
    bar = np.full((22, out.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, info, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([bar, out], 0)


def draw_clip(clip, keys, path, scale=3):
    cen = entity_centres(clip)
    rows = []
    for k in keys:
        rows.append(draw_keyframe(clip["frames"][k], clip["ids"][k], clip["depth"][k], clip["cam_rot"][k],
                                  clip["cam_pos"][k], float(clip["cam_fov"][k]), clip["ent_pos"][:, k], cen[:, k],
                                  clip["ent_visible"][:, k], scale,
                                  f"k={k} f={clip['key_frame'][k]} shot={clip['key_shot'][k]}",
                                  show=(clip["ent_loc"] == clip["shot_loc"][clip["key_shot"][k]]) &
                                  (clip["ent_present"][:, k] > 0)))
    cv2.imwrite(str(path), np.concatenate(rows, 0)[..., ::-1])


def draw_batch(batch, i, keys, path, scale=3):
    """Keyframes of sample i of a loader batch (gauge-fixed cameras, after augmentation)."""
    from .loader import entity_centres as centres
    b = {k: v[i] for k, v in batch.items()}
    cen = centres(b["ent_pos"].astype(float), b["ent_rot"].astype(float), b["ent_class"], b["ent_height"])
    key_loc = b["shot_loc"][b["key_shot"]]
    rows = []
    for k in keys:
        sh = b["key_shot"][k]
        show = b["ent_mask"] & (b["ent_loc"] == key_loc[k]) & b["ent_present"][:, k]
        rows.append(draw_keyframe(b["frames"][k], b["ids"][k], b["depth"][k], b["cam_rot"][k].astype(float),
                                  b["cam_pos"][k].astype(float), float(b["cam_fov"][k]), b["ent_pos"][:, k],
                                  cen[:, k], b["ent_visible"][:, k], scale,
                                  f"k={k} shot={sh} loc={key_loc[k]} zoom={b['shot_zoom'][sh]:.2f} "
                                  f"mirror={bool(b['mirrored'])} anchor={bool(b['is_anchor'][k])}", show=show))
    cv2.imwrite(str(path), np.concatenate(rows, 0)[..., ::-1])
