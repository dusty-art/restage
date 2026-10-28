"""Oracle replica: the ground-truth scene re-expressed in the model's output convention.

Each location is re-anchored on the camera of its first shot's first frame (origin on the ground beneath it,
looking toward −Z), exactly as the model's decoder does. Used to test the viewer's ground-truth overlay and as
the fidelity ceiling in the benchmark.

    python scripts/oracle_replica.py samples/dance.restage  ->  samples/dance_replica_oracle.restage
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from restage import scene as S  # noqa: E402


def gauge_of(doc: dict, loc_id: str):
    shot = next(s for s in doc["shots"] if s["location"] == loc_id)
    cam = next(c for c in doc["cameras"] if c["id"] == shot["camera"])
    p = cam["track"]["position"][0].astype(np.float64)
    q = cam["track"]["rotation"][0].astype(np.float64)
    f = S.quat_rotate(q, np.array([0.0, 0.0, -1.0]))
    psi = np.arctan2(-f[0], -f[2])
    gq = S.quat_euler(y=-psi)
    t = np.array([-p[0], 0.0, -p[2]])
    return gq, t


def apply(gq, t, pos, rot=None):
    p = S.quat_rotate(np.broadcast_to(gq, pos.shape[:-1] + (4,)), pos + t)
    r = None if rot is None else S.quat_mul(np.broadcast_to(gq, rot.shape), rot)
    return p.astype(np.float32), (None if r is None else r.astype(np.float32))


def oracle(doc: dict) -> dict:
    g = {l["id"]: gauge_of(doc, l["id"]) for l in doc["locations"] if any(s["location"] == l["id"] for s in doc["shots"])}
    for l in doc["locations"]:
        if l["id"] not in g:
            continue
        gq, t = g[l["id"]]
        for p in l["props"]:
            pos, rot = apply(gq, t, np.asarray(p.get("position", [0, 0, 0]), np.float64),
                             np.asarray(p.get("rotation", [0, 0, 0, 1]), np.float64))
            p["position"], p["rotation"] = pos.tolist(), rot.tolist()
        lo, hi = np.asarray(l.get("bounds", [[-5, 0, -5], [5, 3, 5]]), np.float64)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        c, _ = apply(gq, t, corners)
        l["bounds"] = [c.min(0).tolist(), c.max(0).tolist()]
        if l.get("environment", {}).get("sun_dir"):
            l["environment"]["sun_dir"] = S.quat_rotate(gq, np.asarray(l["environment"]["sun_dir"], np.float64)).tolist()
    for e in doc["entities"]:
        if e.get("location") not in g:
            continue
        gq, t = g[e["location"]]
        tr = e["track"]
        tr["position"], tr["rotation"] = apply(gq, t, tr["position"].astype(np.float64), tr["rotation"].astype(np.float64))
    for c in doc["cameras"]:
        shot = next(s for s in doc["shots"] if s["id"] == c["shot"])
        gq, t = g[shot["location"]]
        tr = c["track"]
        tr["position"], tr["rotation"] = apply(gq, t, tr["position"].astype(np.float64), tr["rotation"].astype(np.float64))
    doc.setdefault("meta", {})["producer"] = "oracle replica (ground truth re-anchored per location)"
    return doc


if __name__ == "__main__":
    src = Path(sys.argv[1])
    out = src.with_name(src.stem + "_replica.restage") if len(sys.argv) < 3 else Path(sys.argv[2])
    S.save(out, oracle(S.load(src)))
    print("wrote", out)
