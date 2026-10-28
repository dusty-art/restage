"""Turn compiler outputs for one clip into a `.restage` scene document (docs/SCENE_FORMAT.md).

Locations are re-anchored on their first keyframe (as in training), keyframe predictions are interpolated
to every video frame inside each shot (never across a cut), and meshes are built from the predicted
parameters: rounded boxes for sets and objects, the generator's humanoid for people.
"""
from __future__ import annotations

import numpy as np

from restage import scene as S
from restage.model.net import N_CLASSES

CLASS_NAMES = ["none", "person", "animal", "vehicle", "object"]
SIZES = ["extreme_wide", "wide", "medium", "closeup", "extreme_closeup", "insert"]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _rot6d(x):
    a, b = x[..., :3], x[..., 3:]
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b - (a * b).sum(-1, keepdims=True) * a
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return np.stack([a, b, np.cross(a, b)], -1)


def _heading(R):
    f = -R[..., :, 2]
    return np.arctan2(-f[..., 0], -f[..., 2])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rounded_box_mesh(size, roundness, color, n=3):
    """Subdivided cube blended toward a sphere (same formula as the training loss's box samples)."""
    g = np.linspace(-1, 1, n + 1)
    verts, faces = [], []
    for axis in range(3):
        for sign in (-1, 1):
            u, v = np.meshgrid(g, g, indexing="ij")
            p = np.zeros(u.shape + (3,))
            p[..., axis] = sign
            p[..., (axis + 1) % 3] = u
            p[..., (axis + 2) % 3] = v
            base = len(verts) and sum(len(x) for x in verts)
            pts = p.reshape(-1, 3)
            verts.append(pts)
            for i in range(n):
                for j in range(n):
                    a = base + i * (n + 1) + j
                    b, c, d = a + 1, a + n + 1, a + n + 2
                    f1, f2 = [a, c, b], [b, c, d]
                    if sign < 0:
                        f1, f2 = f1[::-1], f2[::-1]
                    faces += [f1, f2]
    v = np.concatenate(verts)
    sph = v / np.linalg.norm(v, axis=-1, keepdims=True)
    v = ((1 - roundness) * v + roundness * sph) * (np.asarray(size) * 0.5)
    f = np.array(faces, np.uint32)
    # unshare vertices so faces stay flat-coloured in the viewer
    v = v[f.reshape(-1)]
    return {"positions": v.astype(np.float32), "indices": np.arange(len(v), dtype=np.uint32).reshape(-1, 3),
            "colors": np.tile(np.asarray(color, np.float32), (len(v), 1))}


def _interp_track(key_frames, values, frames, kind="lin"):
    """Interpolate per-keyframe values to `frames` (keyframes sorted, same shot)."""
    if len(key_frames) == 1:
        return np.repeat(values[:1], len(frames), 0)
    if kind == "quat":
        out = np.empty((len(frames),) + values.shape[1:])
        i = np.clip(np.searchsorted(key_frames, frames, side="right") - 1, 0, len(key_frames) - 2)
        t = np.clip((frames - key_frames[i]) / np.maximum(1, key_frames[i + 1] - key_frames[i]), 0, 1)
        q0, q1 = values[i], values[i + 1].copy()
        dot = (q0 * q1).sum(-1, keepdims=True)
        q1 = np.where(dot < 0, -q1, q1)
        tt = t.reshape((-1,) + (1,) * (values.ndim - 1))
        q = q0 * (1 - tt) + q1 * tt                                    # nlerp (keyframes are 0.25 s apart)
        out[:] = q / np.linalg.norm(q, axis=-1, keepdims=True)
        return out
    flat = values.reshape(len(key_frames), -1)
    res = np.stack([np.interp(frames, key_frames, flat[:, j]) for j in range(flat.shape[1])], -1)
    return res.reshape((len(frames),) + values.shape[1:])


def decode(out: dict, clip: dict, fps: int, n_frames: int, title: str = "Restage replica",
           source: dict | None = None, b: int = 0) -> dict:
    """out: model outputs as numpy (batch index b); clip: key_frame [K], key_shot [K], shot_range [S, 2]."""
    o = {k: np.asarray(v[b], np.float64) if np.asarray(v).dtype != bool else np.asarray(v[b]) for k, v in out.items()}
    key_frame = np.asarray(clip["key_frame"])
    key_shot = np.asarray(clip["key_shot"])
    shot_range = np.asarray(clip["shot_range"])
    K, S_ = len(key_frame), len(shot_range)

    # ---------------------------------------------------------- locations from the shot assignment
    assign = o["shot_assign"][:S_]                                    # [S, L]
    shot_loc_slot = assign.argmax(-1)
    slots = list(dict.fromkeys(shot_loc_slot.tolist()))               # in order of first appearance
    slot_to_loc = {s: i for i, s in enumerate(slots)}
    shot_loc = np.array([slot_to_loc[s] for s in shot_loc_slot])
    key_loc = shot_loc[key_shot]

    # ---------------------------------------------------------- gauge per location (anchor = first keyframe)
    R = _rot6d(o["cam_rot6d"][:K])
    p = o["cam_pos"][:K]
    gauges = []
    for li in range(len(slots)):
        a = int(np.nonzero(key_loc == li)[0][0])
        G = _rot_y(-_heading(R[a]))
        t = np.array([-p[a, 0], 0.0, -p[a, 2]])
        gauges.append((G, t, _heading(R[a])))
    Gk = np.stack([gauges[l][0] for l in key_loc])
    tk = np.stack([gauges[l][1] for l in key_loc])
    cam_R = Gk @ R
    cam_p = np.einsum("kij,kj->ki", Gk, p + tk)
    cam_q = S.mat_to_quat(cam_R)
    fov = o["cam_fov"][:K]

    meshes, materials, entities, cameras, shots, locations = [], [], [], [], [], []

    # ---------------------------------------------------------- sets
    for li, slot in enumerate(slots):
        G, t, psi = gauges[li]
        props = []
        present = _sigmoid(o["box_logit"][slot]) > 0.5
        centers = (o["box_center"][slot] + t) @ G.T
        cam_here = cam_p[key_loc == li]
        lo = np.minimum(cam_here.min(0), centers[present].min(0) if present.any() else cam_here.min(0)) - 4
        hi = np.maximum(cam_here.max(0), centers[present].max(0) if present.any() else cam_here.max(0)) + 4
        if present.any():
            import mlx.core as mx
            from restage.render import export as RX, shapes as RS
            yaw = o["box_yaw"][slot] - psi
            bv, bf, bmeta = RS.rounded_boxes(mx.array(centers[None].astype(np.float32)), mx.array(o["box_size"][slot][None].astype(np.float32)),
                                             mx.array(yaw[None].astype(np.float32)), mx.array(o["box_round"][slot][None].astype(np.float32)))
            mid = f"loc{li}_boxes"
            meshes.append({**RX.rounded_boxes_to_numpy(np.array(bv[0]), bf, bmeta, o["box_color"][slot], present, mid), "id": mid})
            props.append({"id": "blocks", "class": "blocks", "mesh": mid})
        gid = f"loc{li}_ground"
        w = float(max(hi[0] - lo[0], hi[2] - lo[2], 8.0)) * 1.5
        gm = rounded_box_mesh((w, 0.02, w), 0.0, o["loc_ground"][slot], n=1)
        gm["positions"][:, 1] -= 0.01
        meshes.append({"id": gid, **gm})
        props.insert(0, {"id": "ground", "class": "ground", "mesh": gid,
                         "position": [float((lo[0] + hi[0]) / 2), 0.0, float((lo[2] + hi[2]) / 2)]})
        locations.append({"id": f"loc{li}", "name": f"location {li + 1}", "kind": "other",
                          "bounds": [[float(lo[0]), 0.0, float(lo[2])], [float(hi[0]), float(max(hi[1], 3)), float(hi[2])]],
                          "environment": {"sky": o["loc_sky"][slot].tolist(), "ground": o["loc_ground"][slot].tolist(),
                                          "ambient": 0.4, "sun_dir": [0.3, -1.0, 0.25], "sun": 1.0},
                          "props": props})

    # ---------------------------------------------------------- cameras and shots (per-frame inside each shot)
    for si in range(S_):
        a, e = int(shot_range[si, 0]), int(shot_range[si, 1])
        ks = np.nonzero(key_shot == si)[0]
        frames = np.arange(a, e)
        kf = key_frame[ks].astype(float)
        cameras.append({"id": f"cam{si}", "shot": f"s{si}", "track": {
            "position": _interp_track(kf, cam_p[ks], frames).astype(np.float32),
            "rotation": _interp_track(kf, cam_q[ks], frames, "quat").astype(np.float32),
            "fov": _interp_track(kf, fov[ks][:, None], frames)[:, 0].astype(np.float32)}})
        shots.append({"id": f"s{si}", "start": a, "end": e, "location": f"loc{shot_loc[si]}", "camera": f"cam{si}",
                      "size": "medium", "move": "static", "transition": "cut"})

    # ---------------------------------------------------------- entities
    skeletons = {}
    cls = o["ent_class"].argmax(-1)
    pres = _sigmoid(o["ent_present"][:, :K])
    vis = _sigmoid(o["ent_visible"][:, :K])
    for ei in range(len(cls)):
        c = int(cls[ei])
        if c == 0 or pres[ei].max() < 0.5:
            continue
        loc_slot = int(o["ent_loc"][ei][slots].argmax())               # among locations that exist
        G, t, _ = gauges[loc_slot]
        pos_k = (o["ent_pos"][ei, :K] + t) @ G.T
        rot_k = S.mat_to_quat(G @ _rot6d(o["ent_rot6d"][ei, :K]))
        state_k = np.where(vis[ei] > 0.5, 2, np.where(pres[ei] > 0.5, 1, 0))
        pos = np.zeros((n_frames, 3), np.float32)
        rot = np.tile([0, 0, 0, 1.0], (n_frames, 1)).astype(np.float32)
        state = np.zeros(n_frames, np.uint8)
        name = CLASS_NAMES[c]
        cols = o["ent_colors"][ei]
        track = {}
        if c == 1:
            jq = S.mat_to_quat(_rot6d(o["ent_joint6d"][ei, :K]))       # [K,21,4]
            pose = np.tile([0, 0, 0, 1.0], (n_frames, 22, 1)).astype(np.float32)
        for si in range(S_):
            a, e = int(shot_range[si, 0]), int(shot_range[si, 1])
            ks = np.nonzero(key_shot == si)[0]
            frames = np.arange(a, e)
            kf = key_frame[ks].astype(float)
            if shot_loc[si] != loc_slot:
                state[a:e] = 0 if pres[ei, ks].max() < 0.5 else 1
            pos[a:e] = _interp_track(kf, pos_k[ks], frames)
            rot[a:e] = _interp_track(kf, rot_k[ks], frames, "quat")
            near = np.clip(np.searchsorted(kf, frames), 0, len(ks) - 1)
            if shot_loc[si] == loc_slot:
                state[a:e] = state_k[ks][near]
            if c == 1:
                pose[a:e, 1:] = _interp_track(kf, jq[ks], frames, "quat")
        mid = f"ent{ei}_mesh"
        if c == 1:
            h = float(np.clip(o["ent_height"][ei], 1.2, 2.1))
            from restage.render import export as RX
            mesh, skel = RX.humanoid_to_numpy(np.array([h / 1.78, 1.0, 1.0, 1.0]), cols, mid, f"humanoid22_{ei}")
            sid = skel["id"]
            skeletons[sid] = skel
            meshes.append({**mesh, "id": mid})
            track["pose"] = pose
        else:
            size = o["ent_size"][ei]
            m = rounded_box_mesh(size, 0.35 if c == 4 else 0.15, cols[0])
            m["positions"][:, 1] += size[1] / 2                       # root at the bottom
            meshes.append({"id": mid, **m})
        track.update({"state": state, "position": pos, "rotation": rot})
        ent = {"id": f"e{ei}", "name": f"{name} {sum(1 for x in entities if x['class'] == name) + 1}", "class": name,
               "location": f"loc{loc_slot}", "mesh": mid, "track": track}
        if c == 1:
            ent["skeleton"] = sid
        entities.append(ent)

    return {"format": "restage.scene", "version": 1, "fps": int(fps), "n_frames": int(n_frames),
            "meta": {"title": title, "producer": "Restage compiler", "source": source or {}},
            "materials": [], "meshes": meshes, "skeletons": list(skeletons.values()), "locations": locations,
            "entities": entities, "cameras": cameras, "shots": shots, "continuity": []}
