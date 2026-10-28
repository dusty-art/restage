"""Anchor-probe trick (research note 05): interleave an anchor frame before every frame and encode
P-only with ONE reference, so every frame's motion vectors point straight at the anchor.
Compares end-of-shot rotation drift: chained frame-to-frame MVs vs. anchor-referenced MVs.
Run after codec_probe.py (reuses its synthetic video builder). No downloads."""
import os, subprocess, json
import numpy as np, cv2
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codec_probe as cp

frames, gt = cp.build_video()
idx = [i for i, g in enumerate(gt) if g["shot"] == 0]           # 90-frame pan, 25 deg total
K = cp.K_of(cp.F_TRUE, cp.W, cp.H); Ki = np.linalg.inv(K)
rec = np.dtype([("f", "<i4"), ("src", "<i4"), ("w", "<i2"), ("h", "<i2"), ("x", "<i2"), ("y", "<i2"), ("dx", "<f4"), ("dy", "<f4")])

def enc_mv(seq, tag, merange):
    raw, mp4, mvb = [os.path.join(cp.OUT, f"anchor_{tag}.{e}") for e in ("raw", "mp4", "mv")]
    cp.write_raw(seq, raw)
    cp.encode(raw, mp4, ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-bf", "0", "-refs", "1",
                         "-x264-params", f"scenecut=0:keyint=100000:me=umh:merange={merange}"])
    subprocess.run([os.path.join(cp.HERE, "mvdump"), mp4, mvb], capture_output=True, check=True)
    mv = np.fromfile(mvb, dtype=rec); [os.remove(p) for p in (raw, mp4, mvb)]
    return mv

def rot_from(mv, f):
    m = mv[(mv["f"] == f) & (mv["src"] < 0)]
    if len(m) < 30: return None
    dst = np.stack([m["x"], m["y"]], 1).astype(np.float32); src = dst + np.stack([m["dx"], m["dy"]], 1)
    Hm, _ = cv2.findHomography(dst, src, cv2.RANSAC, 2.0)
    if Hm is None: return None
    M = Ki @ Hm @ K; return cp.nearest_rot(M / np.cbrt(np.linalg.det(M)))

def gt_rel(a, b):  # R_a R_b^T
    return cp.rot(gt[a]["yaw"], gt[a]["pitch"]) @ cp.rot(gt[b]["yaw"], gt[b]["pitch"]).T

out = {}
for N in [1, 5, 10, 30, 89]:   # anchor spacing; N=1 == ordinary chaining, N=89 == single anchor for whole shot
    seq, which = [], []
    anchors = idx[::N]
    for i in idx:
        a = max(x for x in anchors if x <= i) if i not in anchors or i == idx[0] else max(x for x in anchors if x < i)
        if i == idx[0]: seq.append(frames[i]); which.append((None, i)); continue
        seq.append(frames[a]); which.append(("anchor", a)); seq.append(frames[i]); which.append((a, i))
    for mr in (16, 64):
        mv = enc_mv(seq, f"N{N}_mr{mr}", mr)
        Rabs = {idx[0]: np.eye(3)}; errs = []; fails = 0
        for k, (a, i) in enumerate(which):
            if a is None or a == "anchor": continue
            R = rot_from(mv, k)            # = R_a R_i^T (frame i relative to its anchor a)
            if R is None: fails += 1; R = np.eye(3)
            errs.append(cp.rot_angle(R @ gt_rel(a, i).T))
            Rabs[i] = Rabs[a] @ R          # R_0 R_i^T
        end = idx[-1]
        out[f"spacing={N} merange={mr}"] = dict(
            end_drift_deg=round(cp.rot_angle(Rabs[end] @ gt_rel(idx[0], end).T), 3),
            median_link_err_deg=round(float(np.median(errs)), 3), max_link_err_deg=round(float(np.max(errs)), 2),
            failed_links=fails, gt_total_deg=round(cp.rot_angle(gt_rel(idx[0], end)), 2))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(cp.OUT, "anchor_results.json"), "w"), indent=1)
