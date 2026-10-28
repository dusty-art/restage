"""Codec-as-free-computation probe (research note 05).

Builds a synthetic *edited* video with known ground truth (camera rotation per frame,
focal length, cut positions, a moving entity), encodes it with real codecs, then asks:
  1. Can H.264 motion vectors recover camera rotation and focal length?
  2. Do codec frame types / intra coverage find the cuts (with and without encoder scene-cut)?
  3. Do homography outliers among the motion vectors find the moving entity?
  4. How fast is motion-vector extraction compared with dense optical flow on this laptop?
  5. Is "bits needed to code the residual (video - render)" a usable black-box score for a
     scene hypothesis ("scene as codec")?

Nothing is downloaded. Requires ffmpeg (libx264/libx265/libsvtav1) and the mvdump binary.
Run:  python -I codec_probe.py   (from this folder)
"""
import json, os, subprocess, sys, time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
W, H, F_TRUE, FPS = 640, 360, 500.0, 30
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- synthetic world
def make_texture(seed, w=2600, h=1600):
    r = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    for s in (512, 128, 32, 8):  # multi-scale coloured noise
        n = r.random((h // s + 2, w // s + 2, 3)).astype(np.float32)
        img += cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC) * (s / 512) ** 0.3
    img = img / img.max() * 160
    for _ in range(900):  # shapes = corners/edges for motion search
        c = tuple(int(v) for v in r.integers(0, 255, 3))
        x, y = int(r.integers(0, w)), int(r.integers(0, h))
        if r.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + int(r.integers(10, 90)), y + int(r.integers(10, 90))), c, -1)
        else:
            cv2.circle(img, (x, y), int(r.integers(5, 45)), c, -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def rot(yaw, pitch, roll=0.0):
    y, p, r_ = np.radians([yaw, pitch, roll])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(r_), -np.sin(r_), 0], [np.sin(r_), np.cos(r_), 0], [0, 0, 1]])
    return Rz @ Rx @ Ry  # world -> camera


def K_of(f, w, h):
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


TEX = {0: make_texture(1), 1: make_texture(2)}
K_REF = K_of(F_TRUE, TEX[0].shape[1], TEX[0].shape[0])
K_OUT = K_of(F_TRUE, W, H)


def render(loc, yaw, pitch, f=F_TRUE):
    """Pure-rotation camera at the centre of a 'panorama' textured location."""
    Ko = K_of(f, W, H)
    Hm = K_REF @ rot(yaw, pitch).T @ np.linalg.inv(Ko)  # output pixel -> reference pixel
    return cv2.warpPerspective(TEX[loc], Hm, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)


# the edit: (location, n_frames, yaw0, yaw1, pitch0, pitch1, has_entity)
SHOTS = [(0, 90, -15, 10, 0, 3, True),   # shot 1: location A, pan right
         (1, 60, 5, 5, -5, 8, False),    # shot 2: location B, tilt up
         (0, 60, 0, -12, 2, -2, True),   # shot 3: back to location A, pan left
         (1, 45, -8, 6, 6, 0, False)]    # shot 4: location B again


def build_video():
    frames, gt = [], []
    sprite = make_texture(7, 90, 90)
    for si, (loc, n, y0, y1, p0, p1, ent) in enumerate(SHOTS):
        for k in range(n):
            a = k / (n - 1)
            yaw = y0 + (y1 - y0) * a
            pitch = p0 + (p1 - p0) * a + 0.6 * np.sin(2 * np.pi * a)
            img = render(loc, yaw, pitch).astype(np.int16)
            box = None
            if ent:  # an independently moving entity (screen-space), e.g. a person walking
                x = int(60 + 420 * a); y = int(200 - 60 * np.sin(np.pi * a))
                img[y:y + 90, x:x + 90] = sprite
                box = [x, y, x + 90, y + 90]
            img = img + rng.normal(0, 2.0, img.shape)  # sensor noise
            frames.append(np.clip(img, 0, 255).astype(np.uint8))
            gt.append(dict(shot=si, loc=loc, yaw=float(yaw), pitch=float(pitch), box=box,
                           cut=(k == 0 and si > 0)))
    return frames, gt


def write_raw(frames, path):
    with open(path, "wb") as fh:
        for fr in frames:
            fh.write(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).tobytes())


def encode(raw, out, vcodec_args):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", raw, *vcodec_args, "-pix_fmt", "yuv420p", out]
    t = time.time(); subprocess.run(cmd, check=True); return time.time() - t


def frame_types(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pict_type",
                        "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    return [l.strip().strip(",") for l in r.stdout.split() if l.strip()]


def mvdump(path, binout):
    r = subprocess.run([os.path.join(HERE, "mvdump"), path, binout], capture_output=True, text=True, check=True)
    summ = [l.split() for l in r.stdout.splitlines() if l.startswith("F ")]
    tail = r.stdout.strip().splitlines()[-1]
    rec = np.dtype([("f", "<i4"), ("src", "<i4"), ("w", "<i2"), ("h", "<i2"), ("x", "<i2"), ("y", "<i2"),
                    ("dx", "<f4"), ("dy", "<f4")])
    mv = np.fromfile(binout, dtype=rec) if binout != "-" else None
    return summ, mv, tail


def nearest_rot(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    return R * np.sign(np.linalg.det(R))


def rot_angle(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


# ---------------------------------------------------------------- analysis
def analyse_rotation(mv, gt, nframes):
    """Per-frame homography from MVs (current -> previous frame), RANSAC, rotation + focal."""
    Hs, inl_masks = {}, {}
    for f in range(1, nframes):
        if gt[f]["cut"]:
            continue
        m = mv[(mv["f"] == f) & (mv["src"] < 0)]
        if len(m) < 30:
            continue
        dst = np.stack([m["x"], m["y"]], 1).astype(np.float32)
        src = dst + np.stack([m["dx"], m["dy"]], 1)
        Hm, inl = cv2.findHomography(dst, src, cv2.RANSAC, 2.0)
        if Hm is not None:
            Hs[f] = Hm; inl_masks[f] = (m, inl.ravel().astype(bool))

    def rot_err_for(f_est):
        K = K_of(f_est, W, H); Ki = np.linalg.inv(K); errs = []
        for f, Hm in Hs.items():
            M = Ki @ Hm @ K; M = M / np.cbrt(np.linalg.det(M))
            errs.append(np.linalg.norm(M - nearest_rot(M)))
        return float(np.median(errs))

    # self-calibration: focal length that makes K^-1 H K most rotation-like (pure-rotation camera)
    grid = np.arange(250, 1001, 5.0)
    f_est = float(grid[int(np.argmin([rot_err_for(g) for g in grid]))])

    K = K_of(f_est, W, H); Ki = np.linalg.inv(K)
    per_frame_err, per_frame_gtmag = [], []
    drift = {}
    for si in range(len(SHOTS)):
        idx = [i for i, g in enumerate(gt) if g["shot"] == si]
        Racc = np.eye(3); Rgt0 = rot(gt[idx[0]]["yaw"], gt[idx[0]]["pitch"])
        for f in idx[1:]:
            if f not in Hs:
                continue
            Rrel = nearest_rot(Ki @ Hs[f] @ K / np.cbrt(np.linalg.det(Ki @ Hs[f] @ K)))  # = R_prev R_t^T
            Rp, Rt = rot(gt[f - 1]["yaw"], gt[f - 1]["pitch"]), rot(gt[f]["yaw"], gt[f]["pitch"])
            Rrel_gt = Rp @ Rt.T
            per_frame_err.append(rot_angle(Rrel @ Rrel_gt.T)); per_frame_gtmag.append(rot_angle(Rrel_gt))
            Racc = Racc @ Rrel  # R_0 R_t^T accumulated
        Rgt_end = rot(gt[idx[-1]]["yaw"], gt[idx[-1]]["pitch"])
        drift[si] = dict(frames=len(idx), gt_total_deg=rot_angle(Rgt0 @ Rgt_end.T),
                         drift_deg=rot_angle(Racc @ (Rgt0 @ Rgt_end.T).T))
    return dict(f_est=f_est, f_true=F_TRUE, n_frames_used=len(Hs),
                median_per_frame_rot_err_deg=float(np.median(per_frame_err)),
                median_per_frame_gt_rot_deg=float(np.median(per_frame_gtmag)), shot_drift=drift), inl_masks


def analyse_entity(inl_masks, gt):
    """Do RANSAC outlier blocks fall on the independently moving entity?"""
    tp = fp = fn = 0
    for f, (m, inl) in inl_masks.items():
        box = gt[f]["box"]
        cx, cy = m["x"].astype(float), m["y"].astype(float)
        on = np.zeros(len(m), bool) if box is None else (cx >= box[0]) & (cx < box[2]) & (cy >= box[1]) & (cy < box[3])
        out = ~inl
        tp += int((out & on).sum()); fp += int((out & ~on).sum()); fn += int((inl & on).sum())
    return dict(outlier_precision=tp / max(tp + fp, 1), entity_recall=tp / max(tp + fn, 1))


def analyse_cuts(summ, gt, ftypes):
    cuts = [i for i, g in enumerate(gt) if g["cut"]]
    cov = np.array([float(s[4]) for s in summ])
    by_type = [i for i, t in enumerate(ftypes) if t == "I" and i > 0]
    by_cov = [i for i in range(1, len(cov)) if cov[i] < 0.5 and ftypes[i] != "I"]
    return dict(true_cuts=cuts, I_frames_after_0=by_type, low_inter_coverage_non_I=by_cov,
                median_inter_coverage=float(np.median(cov[1:])))


def time_flow(frames, n=60):
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames[:n + 1]]
    out = {}
    t = time.time()
    for i in range(n):
        cv2.calcOpticalFlowFarneback(g[i], g[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
    out["farneback_fps"] = n / (time.time() - t)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    t = time.time()
    for i in range(n):
        dis.calc(g[i], g[i + 1], None)
    out["dis_medium_fps"] = n / (time.time() - t)
    return out


def codec_judge(frames, gt):
    """'Scene as codec': bytes needed to code (video - render(hypothesis)) for shot 1.

    Luma only, 10-bit, same quantisation scale for the raw video and every residual, so byte
    counts are comparable. Also run at a coarse level (blur + 4x downsample) to see how wide
    the basin of the score is.
    """
    idx = [i for i, g in enumerate(gt) if g["shot"] == 0]
    luma = {i: cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32) for i in idx}

    def coarse(x):
        return cv2.resize(cv2.GaussianBlur(x, (0, 0), 4), (W // 4, H // 4), interpolation=cv2.INTER_AREA)

    def nbytes(planes, tag):
        h, w = planes[0].shape
        raw = os.path.join(OUT, f"judge_{tag}.raw"); mp4 = os.path.join(OUT, f"judge_{tag}.mp4")
        with open(raw, "wb") as fh:
            for p in planes:
                fh.write((np.clip(np.round(p), 0, 1023).astype("<u2") * 64).tobytes())
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray16le", "-s", f"{w}x{h}",
                        "-r", str(FPS), "-i", raw, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                        "-pix_fmt", "gray10le", mp4], check=True)
        os.remove(raw); n = os.path.getsize(mp4); os.remove(mp4); return n

    def score(yaw_off=0.0, f=F_TRUE, loc=0, level="full"):
        tf = (lambda x: x) if level == "full" else coarse
        planes = []
        for i in idx:
            r = cv2.cvtColor(render(loc, gt[i]["yaw"] + yaw_off, gt[i]["pitch"], f), cv2.COLOR_BGR2GRAY).astype(np.float32)
            planes.append(512 + 2 * (tf(luma[i]) - tf(r)))
        return nbytes(planes, f"{level}_{yaw_off}_{f}_{loc}")

    res = {}
    for level in ("full", "coarse"):
        tf = (lambda x: x) if level == "full" else coarse
        d = {"no_scene_model(raw video)": nbytes([256 + 2 * tf(luma[i]) for i in idx], f"{level}_raw")}
        for off in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
            d[f"yaw_offset_{off}deg"] = score(yaw_off=off, level=level)
        for f in [450.0, 490.0, 510.0, 550.0]:
            d[f"focal_{int(f)}px(true 500)"] = score(f=f, level=level)
        d["wrong_location"] = score(loc=1, level=level)
        res[level] = d
    return res


def codec_judge_mc(frames, gt):
    """Motion-compensated judge (model-aided coding): interleave render R_t and frame F_t,
    encode P-only with ONE reference, so F_t may only predict from its own render R_t
    (with motion compensation). Score = total bytes of the F_t packets. Small misalignments
    now cost cheap motion vectors instead of expensive residual."""
    idx = [i for i, g in enumerate(gt) if g["shot"] == 0]
    luma = {i: cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY) for i in idx}

    def f_bytes(seq, tag, odd_only=True):
        raw = os.path.join(OUT, f"mc_{tag}.raw"); mp4 = os.path.join(OUT, f"mc_{tag}.mp4")
        with open(raw, "wb") as fh:
            for p in seq:
                fh.write(p.tobytes())
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}",
                        "-r", str(FPS), "-i", raw, "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-bf", "0",
                        "-refs", "1", "-x264-params", "scenecut=0:keyint=100000", "-pix_fmt", "gray", mp4], check=True)
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=size",
                            "-of", "csv=p=0", mp4], capture_output=True, text=True, check=True)
        sizes = [int(x) for x in r.stdout.split()]
        os.remove(raw); os.remove(mp4)
        return sum(sizes[1::2]) if odd_only else sum(sizes)

    def score(yaw_off=0.0, f=F_TRUE, loc=0):
        seq = []
        for i in idx:
            seq.append(cv2.cvtColor(render(loc, gt[i]["yaw"] + yaw_off, gt[i]["pitch"], f), cv2.COLOR_BGR2GRAY))
            seq.append(luma[i])
        return f_bytes(seq, f"{yaw_off}_{f}_{loc}")

    blank = np.full((H, W), 128, np.uint8)
    d = {"frames_given_blank_reference": f_bytes([x for i in idx for x in (blank, luma[i])], "blank"),
         "frames_given_previous_frame(ordinary video coding)": f_bytes([luma[i] for i in idx], "plain", odd_only=False)}
    for off in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        d[f"yaw_offset_{off}deg"] = score(yaw_off=off)
    for f in [450.0, 550.0]:
        d[f"focal_{int(f)}px(true 500)"] = score(f=f)
    d["wrong_location"] = score(loc=1)
    return d


def main():
    frames, gt = build_video()
    raw = os.path.join(OUT, "edit.raw"); write_raw(frames, raw)
    results = {"n_frames": len(frames), "resolution": f"{W}x{H}", "fps": FPS}
    encs = {
        # what a typical uploaded file looks like (scene-cut detection on, B-frames, 3 refs)
        "x264_default": ["-c:v", "libx264", "-preset", "medium", "-crf", "23"],
        # a platform-style fixed GOP with scene-cut disabled (cuts NOT aligned to I-frames)
        "x264_fixedgop": ["-c:v", "libx264", "-preset", "medium", "-crf", "23", "-x264-params", "keyint=60:min-keyint=60:scenecut=0"],
        # our own 'probe' re-encode: P-only, 1 reference, fast -> clean frame-to-frame motion field
        "x264_probe": ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-bf", "0", "-refs", "1"],
        "x265_default": ["-c:v", "libx265", "-preset", "medium", "-crf", "26", "-x265-params", "log-level=error"],
        "svtav1_default": ["-c:v", "libsvtav1", "-preset", "8", "-crf", "35"],
    }
    for name, args in encs.items():
        mp4 = os.path.join(OUT, f"{name}.mp4")
        t_enc = encode(raw, mp4, args)
        ft = frame_types(mp4)
        r = dict(encode_seconds=round(t_enc, 2), bytes=os.path.getsize(mp4),
                 I_frames=[i for i, t in enumerate(ft) if t == "I"])
        if name.startswith("x264"):
            summ, mv, tail = mvdump(mp4, os.path.join(OUT, f"{name}.mv"))
            r["mvdump"] = tail
            r["cuts"] = analyse_cuts(summ, gt, ft)
            rr, inl = analyse_rotation(mv, gt, len(frames))
            r["rotation"] = rr
            r["entity"] = analyse_entity(inl, gt)
            _, _, tail_nowrite = mvdump(mp4, "-")
            r["mvdump_nowrite"] = tail_nowrite
        results[name] = r
    os.remove(raw)
    results["optical_flow_cpu"] = time_flow(frames)
    results["codec_judge_shot1"] = codec_judge(frames, gt)
    results["codec_judge_mc_shot1"] = codec_judge_mc(frames, gt)
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
