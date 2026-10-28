"""A tiny hand-built scene for testing the viewer: two locations, a walking box-figure, a bouncing ball, three shots.

The real content comes from restage.gen; this only exercises every part of the format.
    python scripts/make_dev_fixture.py  ->  samples/dev_fixture.restage
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from restage import scene as S  # noqa: E402


def box(sx, sy, sz, c=(0, 0, 0), color=(0.8, 0.8, 0.8)):
    x, y, z = sx / 2, sy / 2, sz / 2
    v = np.array([[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z], [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]]) + c
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2], [1, 2, 6], [1, 6, 5],
                  [0, 4, 7], [0, 7, 3]])
    # unshare vertices so flat colours stay per face
    v = v[f.reshape(-1)]
    return v.astype(np.float32), np.arange(len(v)).reshape(-1, 3).astype(np.uint32), np.tile(color, (len(v), 1)).astype(np.float32)


def merge(parts, joints=None):
    vs, fs, cs, js = [], [], [], []
    n = 0
    for i, (v, f, c) in enumerate(parts):
        vs.append(v)
        fs.append(f + n)
        cs.append(c)
        if joints is not None:
            js.append(np.full(len(v), joints[i]))
        n += len(v)
    m = {"positions": np.concatenate(vs), "indices": np.concatenate(fs), "colors": np.concatenate(cs)}
    if joints is not None:
        j = np.zeros((n, 4), np.uint8)
        j[:, 0] = np.concatenate(js)
        w = np.zeros((n, 4), np.float32)
        w[:, 0] = 1
        m["joints"], m["weights"] = j, w
    return m


def figure():
    pos, _ = S.forward_kinematics(np.tile([0, 0, 0, 1.0], (1, 22, 1)), np.zeros((1, 3)), np.tile([0, 0, 0, 1.0], (1, 1)))
    p = pos[0]
    parts, joints = [], []
    skin, shirt, pants = (0.85, 0.65, 0.5), (0.2, 0.45, 0.8), (0.15, 0.15, 0.2)
    def seg(a, b, w, color):
        c = (p[S.JID[a]] + p[S.JID[b]]) / 2
        d = np.abs(p[S.JID[b]] - p[S.JID[a]])
        parts.append(box(max(d[0], w), max(d[1], w), max(d[2], w), c, color))
        joints.append(S.JID[a])
    seg("pelvis", "spine2", 0.3, shirt)
    seg("spine2", "neck", 0.34, shirt)
    parts.append(box(0.2, 0.24, 0.22, p[S.JID["head"]] + [0, 0.1, 0], skin)); joints.append(S.JID["head"])
    for s in "lr":
        seg(f"{s}_shoulder", f"{s}_elbow", 0.09, shirt)
        seg(f"{s}_elbow", f"{s}_wrist", 0.08, skin)
        seg(f"{s}_hip", f"{s}_knee", 0.13, pants)
        seg(f"{s}_knee", f"{s}_ankle", 0.11, pants)
        parts.append(box(0.1, 0.06, 0.22, p[S.JID[f"{s}_ankle"]] + [0, -0.02, 0.06], (0.1, 0.1, 0.1))); joints.append(S.JID[f"{s}_ankle"])
    return merge(parts, joints)


def main():
    fps, T = 30, 300
    t = np.arange(T) / fps
    meshes = []
    def add_mesh(mid, m):
        meshes.append({"id": mid, **m})
        return mid
    # location 0: room
    add_mesh("floor", merge([box(8, 0.1, 8, (0, -0.05, 0), (0.55, 0.45, 0.35))]))
    add_mesh("walls", merge([box(8, 3, 0.1, (0, 1.5, -4), (0.85, 0.82, 0.75)), box(0.1, 3, 8, (-4, 1.5, 0), (0.8, 0.78, 0.72))]))
    add_mesh("table", merge([box(1.4, 0.06, 0.8, (0, 0.74, 0), (0.45, 0.3, 0.2))] +
                            [box(0.06, 0.72, 0.06, (x, 0.36, z), (0.4, 0.27, 0.18)) for x in (-0.65, 0.65) for z in (-0.35, 0.35)]))
    add_mesh("mug", merge([box(0.09, 0.1, 0.09, (0, 0.05, 0), (0.9, 0.2, 0.2))]))
    # location 1: outdoors
    add_mesh("ground", merge([box(30, 0.1, 30, (0, -0.05, 0), (0.35, 0.55, 0.3))]))
    add_mesh("tree", merge([box(0.3, 2, 0.3, (0, 1, 0), (0.4, 0.28, 0.15)), box(1.6, 1.6, 1.6, (0, 2.6, 0), (0.2, 0.5, 0.22))]))
    add_mesh("person", figure())
    add_mesh("ball", merge([box(0.22, 0.22, 0.22, (0, 0.11, 0), (0.95, 0.8, 0.1))]))

    # person walks a circle in the room for frames 0..200, then appears outdoors
    ang = t * 0.6
    pos = np.stack([np.cos(ang) * 1.8, np.zeros(T), np.sin(ang) * 1.8], -1)
    heading = -ang  # tangent direction
    rot = S.quat_euler(y=heading + np.pi)
    pose = np.tile([0, 0, 0, 1.0], (T, 22, 1))
    swing = np.sin(t * 2 * np.pi * 1.6) * 0.5
    for s, k in (("l", 1), ("r", -1)):
        pose[:, S.JID[f"{s}_hip"]] = S.quat_euler(x=k * swing)
        pose[:, S.JID[f"{s}_knee"]] = S.quat_euler(x=np.maximum(0, -k * swing) * 0.9)
        pose[:, S.JID[f"{s}_shoulder"]] = S.quat_euler(x=-k * swing * 0.7, z=-k * 1.2)
    state = np.full(T, 2, np.uint8)
    # ball bounces outdoors
    bpos = np.stack([np.full(T, 1.0), np.abs(np.sin(t * 3)) * 1.2, np.full(T, -1.0)], -1)
    bstate = np.where(np.arange(T) >= 200, 2, 0).astype(np.uint8)
    mstate = np.where(np.arange(T) < 200, 2, 0).astype(np.uint8)
    shots = [(0, 90, "loc0", "wide", "static"), (90, 200, "loc0", "medium", "pan"), (200, 300, "loc1", "wide", "dolly")]
    cams = []
    for i, (a, b, loc, size, move) in enumerate(shots):
        n = b - a
        if i == 0:
            eye = np.tile([3.5, 1.6, 3.5], (n, 1))
            tgt = np.tile([0, 0.9, 0], (n, 1))
            fov = 50
        elif i == 1:
            eye = np.tile([0, 1.5, 3.8], (n, 1))
            tgt = pos[a:b] + [0, 1.2, 0]
            fov = 30
        else:
            eye = np.stack([np.linspace(-4, -2, n), np.full(n, 1.4), np.linspace(4, 3, n)], -1)
            tgt = np.tile([0.5, 0.8, -0.5], (n, 1))
            fov = 45
        cams.append({"id": f"cam{i}", "shot": f"s{i}", "track": {"position": eye.astype(np.float32),
                     "rotation": S.look_rotation(eye, tgt).astype(np.float32), "fov": np.full(n, fov, np.float32)}})
    doc = {
        "format": "restage.scene", "version": 1, "fps": fps, "n_frames": T,
        "meta": {"title": "dev fixture", "producer": "scripts/make_dev_fixture.py", "source": {"width": 640, "height": 360}},
        "materials": [], "meshes": meshes, "skeletons": [S.skeleton_doc()],
        "locations": [
            {"id": "loc0", "name": "room", "kind": "indoor", "bounds": [[-4, 0, -4], [4, 3, 4]],
             "environment": {"sky": [0.5, 0.55, 0.65], "ambient": 0.45, "sun_dir": [0.4, -1, -0.3], "sun": 0.9},
             "props": [{"id": "floor", "mesh": "floor"}, {"id": "walls", "mesh": "walls"},
                       {"id": "table", "class": "table", "mesh": "table", "position": [0, 0, 0]}]},
            {"id": "loc1", "name": "park", "kind": "nature", "bounds": [[-8, 0, -8], [8, 4, 8]],
             "environment": {"sky": [0.45, 0.65, 0.95], "ambient": 0.4, "sun_dir": [0.3, -1, 0.5], "sun": 1.1,
                             "fog": {"near": 15, "far": 60}},
             "props": [{"id": "ground", "mesh": "ground"}] + [{"id": f"tree{i}", "class": "tree", "mesh": "tree",
                       "position": [x, 0, z]} for i, (x, z) in enumerate([(-3, -3), (3.5, -2), (-2, 3), (4, 4)])]},
        ],
        "entities": [
            {"id": "e0", "name": "walker", "class": "person", "location": "loc0", "mesh": "person", "skeleton": "humanoid22",
             "track": {"state": state, "position": pos.astype(np.float32), "rotation": rot.astype(np.float32),
                       "pose": pose.astype(np.float32), "confidence": np.full(T, 0.9, np.float32)}},
            {"id": "e1", "name": "mug", "class": "object", "location": "loc0", "mesh": "mug",
             "track": {"state": mstate, "position": np.tile([0.3, 0.77, 0.1], (T, 1)).astype(np.float32),
                       "rotation": np.tile([0, 0, 0, 1.0], (T, 1)).astype(np.float32)}},
            {"id": "e2", "name": "ball", "class": "object", "location": "loc1", "mesh": "ball",
             "track": {"state": bstate, "position": bpos.astype(np.float32),
                       "rotation": np.tile([0, 0, 0, 1.0], (T, 1)).astype(np.float32)}},
        ],
        "cameras": cams,
        "shots": [{"id": f"s{i}", "start": a, "end": b, "location": loc, "camera": f"cam{i}", "size": size, "move": move,
                   "transition": "cut"} for i, (a, b, loc, size, move) in enumerate(shots)],
        "continuity": [{"kind": "moved", "entity": "e1", "from_shot": "s0", "to_shot": "s1", "frame": 90,
                        "detail": "test entry: mug shifts across the cut", "severity": 0.5}],
    }
    out = Path(__file__).resolve().parent.parent / "samples" / "dev_fixture.restage"
    S.save(out, doc)
    print("wrote", out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
