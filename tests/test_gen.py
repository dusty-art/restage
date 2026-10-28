"""Tests for the procedural scene generator (restage.gen)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from restage import scene
from restage.gen import assets as A
from restage.gen import scenario
from restage.gen.motion import HumanoidAnimator
from restage.scene import JID

SECONDS = 6.0


@pytest.fixture(scope="module")
def generated():
    """One small scene per genre (generated once for the module)."""
    out = {}
    for i, g in enumerate(scenario.GENRES):
        out[g] = scenario.generate(g, seed=100 + i, seconds=SECONDS)
    return out


def _strip(doc):
    """JSON part of a document with arrays replaced by (dtype, shape, hash)."""
    def rec(x):
        if isinstance(x, np.ndarray):
            return [str(x.dtype), list(x.shape), hash(np.ascontiguousarray(x).tobytes())]
        if isinstance(x, dict):
            return {k: rec(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [rec(v) for v in x]
        return x
    return rec(doc)


# ---------------------------------------------------------------- format

def test_format_roundtrip(tmp_path, generated):
    doc, _ = generated["talking"]
    path = tmp_path / "x.restage"
    scene.save(path, doc)
    raw = path.read_bytes()
    assert raw[:4] == b"RSTG"
    version, jl = np.frombuffer(raw[4:12], "<u4")
    assert version == 1 and jl % 4 == 0
    js = json.loads(raw[12:12 + jl])

    def accessors(x):
        if isinstance(x, dict):
            if set(x) == {"offset", "dtype", "shape"}:
                yield x
            else:
                for v in x.values():
                    yield from accessors(v)
        elif isinstance(x, list):
            for v in x:
                yield from accessors(v)
    for acc in accessors(js):
        assert acc["offset"] % 4 == 0
        assert acc["dtype"] in ("f32", "f16", "u32", "u16", "u8", "i16")
    back = scene.load(path)
    assert json.dumps(_strip(back), sort_keys=True) == json.dumps(_strip(scene.load(path)), sort_keys=True)
    for e0, e1 in zip(doc["entities"], back["entities"]):
        for k, v in e0["track"].items():
            np.testing.assert_array_equal(np.asarray(v, v.dtype), e1["track"][k])


@pytest.mark.parametrize("genre", scenario.GENRES)
def test_document_contract(genre, generated):
    doc, _ = generated[genre]
    T = doc["n_frames"]
    assert doc["format"] == "restage.scene" and doc["version"] == 1
    assert T == int(round(SECONDS * doc["fps"]))
    meshes = {m["id"]: m for m in doc["meshes"]}
    mats = {m["id"] for m in doc["materials"]}
    skels = {s["id"]: s for s in doc["skeletons"]}
    locs = {l["id"] for l in doc["locations"]}
    for m in meshes.values():
        V = len(m["positions"])
        assert m["indices"].max() < V and m["indices"].shape[1] == 3
        assert m["colors"].shape == (V, 3) and m["material"] in mats
        if "joints" in m:
            assert m["joints"].shape == (V, 4) and np.allclose(m["weights"].sum(1), 1)
    for l in doc["locations"]:
        for p in l["props"]:
            assert p["mesh"] in meshes
    # shots tile the video and each has a camera of the right length
    cams = {c["id"]: c for c in doc["cameras"]}
    t = 0
    for s in doc["shots"]:
        assert s["start"] == t and s["end"] > s["start"] and s["location"] in locs
        assert s["size"] in ("extreme_wide", "wide", "medium", "closeup", "extreme_closeup", "insert")
        assert s["move"] in ("static", "pan", "tilt", "dolly", "truck", "orbit", "handheld", "crane", "drone", "fpv")
        c = cams[s["camera"]]
        n = s["end"] - s["start"]
        assert c["track"]["position"].shape == (n, 3) and c["track"]["fov"].shape == (n,)
        assert np.allclose(np.linalg.norm(c["track"]["rotation"], axis=1), 1, atol=1e-4)
        assert np.all(np.isfinite(c["track"]["position"]))
        t = s["end"]
    assert t == T
    for e in doc["entities"]:
        tr = e["track"]
        assert e["mesh"] in meshes and e["location"] in locs
        assert tr["state"].shape == (T,) and set(np.unique(tr["state"])) <= {0, 1, 2}
        assert tr["position"].shape == (T, 3) and tr["rotation"].shape == (T, 4)
        assert np.allclose(np.linalg.norm(tr["rotation"], axis=1), 1, atol=1e-4)
        if "pose" in tr:
            J = len(skels[e["skeleton"]]["joints"])
            assert tr["pose"].shape == (T, J, 4)
            assert np.allclose(np.linalg.norm(tr["pose"], axis=2), 1, atol=1e-3)


# ---------------------------------------------------------------- motion

def _contact_points(track, scale=1.0):
    """World heel and ball (sole under the toe joint) points per frame and foot: [T, 2, 2, 3]."""
    from restage.scene import quat_rotate
    jw = track["joints_world"].astype(np.float64)
    jr = track["joint_rot_world"].astype(np.float64)
    out = np.zeros((len(jw), 2, 2, 3))
    for foot, (ank, toe) in enumerate((("l_ankle", "l_toe"), ("r_ankle", "r_toe"))):
        q = jr[:, JID[ank]]
        out[:, foot, 0] = jw[:, JID[ank]] + quat_rotate(q, np.broadcast_to([0, -0.06 * scale, -0.06 * scale], q.shape[:-1] + (3,)))
        out[:, foot, 1] = jw[:, JID[toe]] + quat_rotate(jr[:, JID[toe]], np.broadcast_to([0, -0.01 * scale, 0],
                                                                                          q.shape[:-1] + (3,)))
    return out


def _stance_stats(track, ground=lambda x, z: 0.0):
    """Per stance frame: height of the lowest sole point above ground and its slide since the last frame."""
    pts = _contact_points(track)
    c = track["contacts"].astype(bool)
    heights, slides = [], []
    for foot in (0, 1):
        m = c[:, foot]
        edges = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(int), [0]])))
        for s, e in zip(edges[::2], edges[1::2]):
            if e - s < 4:
                continue
            P = pts[s + 1:e - 1, foot]                   # [n, 2 (heel, ball), 3]
            g = np.array([[ground(x, z) for x, z in pp[:, [0, 2]]] for pp in P])
            h = P[..., 1] - g
            low = np.argmin(h, axis=1)
            heights += list(h[np.arange(len(P)), low])
            # a planted contact point does not move between consecutive frames
            for k in range(1, len(P)):
                j = low[k]
                slides.append(np.linalg.norm(P[k, j, [0, 2]] - P[k - 1, j, [0, 2]]))
    return np.array(heights), np.array(slides)


@pytest.mark.parametrize("gait,speed", [("walk", 1.0), ("walk", 1.5), ("run", 3.2)])
def test_feet_planted_during_walk(gait, speed):
    an = HumanoidAnimator(30 * 12, 30, 1.0, np.random.default_rng(1), pos=(0, 0), yaw=0.0)
    an.stand(0.5).walk([(0, 4), (3, 8), (6, 9)], speed=speed, gait=gait).stand(0.5)
    tr = an.build()
    h, slide = _stance_stats(tr)
    assert len(h) > 40
    assert np.abs(h).max() < 0.015, "foot not on the ground during stance"
    assert np.percentile(slide, 95) < 0.004 and slide.max() < 0.012, "planted foot slides"
    # whole-track: the toes never go through the floor and the feet reach their targets
    jw = tr["joints_world"]
    assert jw[:, JID["l_toe"], 1].min() > -0.005 and jw[:, JID["r_toe"], 1].min() > -0.005
    err = np.linalg.norm(jw[:, JID["l_ankle"]] - tr["foot_targets"][:, 0], axis=1)
    assert err.max() < 0.01


def test_feet_planted_on_terrain():
    rng = np.random.default_rng(2)
    hf = A.Heightfield(A.value_noise((60, 60), 30, rng, 3) * 1.5, -30, -30, 1.0)
    an = HumanoidAnimator(30 * 10, 30, 1.0, rng, pos=(0, 0), yaw=0.0, ground=hf)
    an.walk([(0, 6), (4, 12)], speed=1.3)
    tr = an.build()
    h, slide = _stance_stats(tr, lambda x, z: float(hf(x, z)))
    assert np.abs(h).max() < 0.03
    assert np.percentile(slide, 95) < 0.005


def test_stride_follows_speed():
    lengths = []
    for v in (0.8, 1.4, 2.0):
        an = HumanoidAnimator(30 * 20, 30, 1.0, np.random.default_rng(0))
        an.walk([(0, 20)], speed=v)
        pl = [p.pos for p in an.plants[0]][2:-2]
        lengths.append(np.median(np.linalg.norm(np.diff(pl, axis=0), axis=1)))
    assert lengths[0] < lengths[1] < lengths[2]
    assert 1.1 < lengths[1] < 1.8      # a stride (two steps) at 1.4 m/s is about 1.5 m


# ---------------------------------------------------------------- cameras and visibility

@pytest.mark.parametrize("genre", scenario.GENRES)
def test_cameras_not_inside_geometry(genre, generated):
    doc, state = generated[genre]
    worlds = state["worlds"]
    for s, (p, q, f, info) in zip(state["shots"], state["cams"]):
        if s.move == "fpv":     # attached to a head or a dashboard by design
            continue
        inside = worlds[s.loc].inside(p.astype(np.float64), margin=0.0, ground_margin=0.05)
        assert inside.mean() < 0.02, f"{genre} shot {s.size}/{s.move} camera inside geometry"


@pytest.mark.parametrize("genre", scenario.GENRES)
def test_state_matches_rendered_visibility(genre, generated):
    from restage.gen.render import SceneRenderer
    doc, _ = generated[genre]
    T = doc["n_frames"]
    r = SceneRenderer(doc, 320, 180, aux_size=(320, 180), msaa=0)
    frames = list(range(0, T, 3))
    seen = {i + 1: [] for i in range(len(doc["entities"]))}
    for t in frames:
        ids = r.render(t, ("ids",))["ids"]
        counts = np.bincount(ids.ravel(), minlength=len(doc["entities"]) + 2)
        for k in seen:
            seen[k].append(counts[k] if k < len(counts) else 0)
    r.release()
    small = 40     # pixels at 320x180: below this an entity is not clearly visible
    agree, total = 0, 0
    for i, e in enumerate(doc["entities"]):
        st = e["track"]["state"][frames]
        px = np.array(seen[i + 1])
        assert np.all(px[st == 0] == 0), "absent entity was rendered"
        present = st > 0
        # visibility is coarse (boxes and spheres), so allow some disagreement on borderline frames
        wrong = ((st == 2) & (px == 0)) | ((st == 1) & (px >= small))
        if present.sum() >= 10:
            assert wrong[present].mean() < 0.5, f"{e['name']}: state disagrees with the image too often"
        agree += (~wrong[present]).sum()
        total += present.sum()
    assert agree / max(total, 1) > 0.88


def test_determinism():
    a, _ = scenario.generate("interview", seed=7, seconds=4.0)
    b, _ = scenario.generate("interview", seed=7, seconds=4.0)
    c, _ = scenario.generate("interview", seed=8, seconds=4.0)
    ja, jb, jc = (json.dumps(_strip(x), sort_keys=True) for x in (a, b, c))
    assert ja == jb
    assert ja != jc


def test_continuity_errors_recorded():
    doc, _ = scenario.generate("talking", seed=3, seconds=10.0, errors=True)
    notes = doc["continuity"]
    assert notes, "no continuity errors were injected"
    ents = {e["id"]: e for e in doc["entities"]}
    shots = {s["id"]: s for s in doc["shots"]}
    for n in notes:
        assert n["kind"] in ("moved", "appeared", "vanished", "pose", "wardrobe", "lighting")
        assert n["entity"] in ents and n["frame"] == shots[n["to_shot"]]["start"]
        st = ents[n["entity"]]["track"]["state"]
        f = n["frame"]
        if n["kind"] == "moved":
            p = ents[n["entity"]]["track"]["position"]
            assert np.linalg.norm(p[f] - p[f - 1]) > 0.1
        elif n["kind"] == "appeared":
            assert st[f - 1] == 0 and st[f] > 0
        elif n["kind"] in ("vanished", "wardrobe"):
            assert st[f - 1] > 0 and st[f] == 0


# ---------------------------------------------------------------- assets

def test_primitives_are_closed_and_outward():
    for m in (A.box(), A.cylinder(), A.cone(), A.icosphere(), A.limb([0, 0, 0], [0.3, 1, 0.2], 0.1, 0.05),
              A.tapered_box((1, 1), (0.4, 0.6))):
        assert A.signed_volume(m) > 0


def test_greedy_mesh_counts():
    vox = np.zeros((4, 4, 4), np.uint8)
    vox[1:3, 1:3, 1:3] = 3          # a 2x2x2 stone cube: 6 merged faces
    m = A.greedy_mesh(vox)
    assert len(m["indices"]) == 12
    assert A.signed_volume(m) == pytest.approx(8.0)


def test_humanoid_is_rigged():
    rng = np.random.default_rng(0)
    for _ in range(5):
        m, s = A.humanoid(A.humanoid_params(rng), rng)
        assert m["joints"][:, 0].max() < 22 and np.all(m["weights"][:, 0] == 1)
        # every part of the body is present
        assert set(np.unique(m["joints"][:, 0])) >= {JID[j] for j in ("pelvis", "chest", "head", "l_wrist", "r_wrist",
                                                                       "l_ankle", "r_ankle", "l_knee", "r_elbow")}
        h = m["positions"][:, 1].max() - m["positions"][:, 1].min()
        assert 1.45 < h < 2.1
