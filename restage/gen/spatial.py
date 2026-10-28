"""Coarse spatial queries over a location: is a point inside geometry, is a segment blocked.

Each location set provides a `World` made of axis-aligned boxes (furniture, buildings, tree
trunks and canopies), an optional room interior (cameras must stay inside the walls), a ground
height function (flat floors, terrain heightfields, stages) and optionally a voxel grid. The
shot planner uses it to keep cameras out of geometry and to prefer unoccluded viewpoints; the
scenario uses it to compute entity visibility (`state`).
"""
from __future__ import annotations

import numpy as np


def _flat(x, z):
    return np.zeros(np.broadcast(np.asarray(x), np.asarray(z)).shape)


class World:
    def __init__(self, ground=None, interior=None, voxels=None):
        self.ground = ground if ground is not None else _flat
        self.interior = None if interior is None else (np.asarray(interior[0], float), np.asarray(interior[1], float))
        self.voxels = voxels          # (solid bool [X, Y, Z], origin xyz, block size)
        self._lo, self._hi, self.names, self.occ = [], [], [], []
        self.lo = np.zeros((0, 3))
        self.hi = np.zeros((0, 3))
        self.occluder = np.zeros(0, bool)
        self._sph = []
        self.spheres = np.zeros((0, 4))       # round occluders (tree canopies): x, y, z, r

    def add_box(self, lo, hi, name="", occluder=True):
        self._lo.append(np.asarray(lo, float))
        self._hi.append(np.asarray(hi, float))
        self.names.append(name)
        self.occ.append(bool(occluder))
        self.lo = np.array(self._lo)
        self.hi = np.array(self._hi)
        self.occluder = np.array(self.occ)

    def add_sphere(self, center, r):
        self._sph.append([*np.asarray(center, float), float(r)])
        self.spheres = np.array(self._sph)

    # ------------------------------------------------------------ points
    def ground_at(self, x, z):
        return np.asarray(self.ground(np.asarray(x, float), np.asarray(z, float)), float)

    def voxel_solid(self, pts):
        if self.voxels is None:
            return np.zeros(len(pts), bool)
        solid, origin, bs = self.voxels
        ijk = np.floor((pts - np.asarray(origin)) / bs).astype(int)
        ok = np.all((ijk >= 0) & (ijk < np.array(solid.shape)), axis=1)
        out = np.zeros(len(pts), bool)
        out[ok] = solid[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
        return out

    def inside(self, pts, margin=0.0, ground_margin=None, skip_names=()):
        """True where a point is inside (or within `margin` of) solid geometry or outside the room."""
        pts = np.atleast_2d(np.asarray(pts, float))
        gm = margin if ground_margin is None else ground_margin
        bad = pts[:, 1] < self.ground_at(pts[:, 0], pts[:, 2]) + gm
        if self.interior is not None:
            lo, hi = self.interior
            bad |= np.any(pts < lo + margin, axis=1) | np.any(pts > hi - margin, axis=1)
        if len(self.lo):
            keep = np.array([n not in skip_names for n in self.names]) if skip_names else slice(None)
            lo, hi = self.lo[keep] - margin, self.hi[keep] + margin
            for i in range(0, len(pts), 4096):
                p = pts[i:i + 4096, None]
                bad[i:i + 4096] |= np.any(np.all((p >= lo[None]) & (p <= hi[None]), axis=2), axis=1)
        if len(self.spheres):
            c, r = self.spheres[:, :3], self.spheres[:, 3]
            for i in range(0, len(pts), 2048):
                P = pts[i:i + 2048]
                near = np.all((c >= P.min(0) - r.max() - margin) & (c <= P.max(0) + r.max() + margin), axis=1)
                if near.any():
                    d = np.linalg.norm(P[:, None] - c[near][None], axis=2)
                    bad[i:i + 2048] |= np.any(d < r[near][None] + margin, axis=1)
        if self.voxels is not None:
            bad |= self.voxel_solid(pts)
            if margin > 0:
                for d in np.eye(3):
                    bad |= self.voxel_solid(pts + d * margin) | self.voxel_solid(pts - d * margin)
        return bad

    # ------------------------------------------------------------ segments
    def blocked(self, p0, p1, shrink=0.12, occluders_only=True, n_ground=None):
        """True where the segment p0 -> p1 passes through geometry (endpoints excluded by `shrink` m)."""
        p0 = np.atleast_2d(np.asarray(p0, float))
        p1 = np.atleast_2d(np.asarray(p1, float))
        p0, p1 = np.broadcast_arrays(p0, p1)
        d = p1 - p0
        L = np.linalg.norm(d, axis=1)
        t_lo = np.minimum(shrink / np.maximum(L, 1e-6), 0.45)
        t_hi = 1 - t_lo
        out = np.zeros(len(p0), bool)
        if len(self.lo):
            sel = self.occluder if occluders_only else np.ones(len(self.lo), bool)
            lo, hi = self.lo[sel], self.hi[sel]
            # shrink boxes slightly: coarse boxes overestimate thin furniture
            c, h = (lo + hi) / 2, (hi - lo) / 2 * 0.92
            lo, hi = c - h, c + h
            inv = 1.0 / np.where(np.abs(d) < 1e-9, 1e-9, d)
            for i in range(0, len(p0), 1024):
                o = p0[i:i + 1024, None]
                iv = inv[i:i + 1024, None]
                t1 = (lo[None] - o) * iv
                t2 = (hi[None] - o) * iv
                tmin = np.minimum(t1, t2).max(2)
                tmax = np.maximum(t1, t2).min(2)
                a = np.maximum(tmin, t_lo[i:i + 1024, None])
                bmax = np.minimum(tmax, t_hi[i:i + 1024, None])
                out[i:i + 1024] |= np.any(bmax >= a, axis=1)
        if n_ground is None:      # about one terrain sample per metre (hills between camera and subject)
            n_ground = int(np.clip(np.ceil(L.max() / 1.0), 8, 96))
        tg = np.linspace(0, 1, n_ground)[None, :, None]
        steps = int(np.clip(np.ceil(L.max() / (self.voxels[2] * 0.5)), 4, 160)) if self.voxels is not None else 0
        tv = np.linspace(0, 1, max(steps, 1))[None, :, None]
        for i in range(0, len(p0), 2048):       # chunked to keep memory flat
            sl = slice(i, i + 2048)
            lo_, span = t_lo[sl, None, None], (t_hi - t_lo)[sl, None, None]
            sp = (p0[sl, None] + d[sl, None] * (lo_ + span * tg)).reshape(-1, 3)
            below = sp[:, 1] < self.ground_at(sp[:, 0], sp[:, 2]) - 0.05
            out[sl] |= below.reshape(-1, n_ground).any(1)
            if self.voxels is not None:
                sp = (p0[sl, None] + d[sl, None] * (lo_ + span * tv)).reshape(-1, 3)
                out[sl] |= self.voxel_solid(sp).reshape(-1, steps).any(1)
        return out

    def first_hit(self, origin, target, steps=48):
        """Fraction along origin -> target of the first blocked sample (1.0 if clear)."""
        origin, target = np.asarray(origin, float), np.asarray(target, float)
        ts = np.linspace(0, 1, steps)
        pts = origin[None] + (target - origin)[None] * ts[:, None]
        bad = self.inside(pts, margin=0.05)
        idx = np.where(bad)[0]
        return 1.0 if not len(idx) else float(ts[idx[0]])


def sphere_occludes(p0, p1, centers, radii):
    """For segments p0->p1 [N,3] and spheres [M,3] (radii [M]), True where any sphere blocks the segment."""
    d = p1 - p0
    L2 = np.maximum((d * d).sum(1), 1e-9)
    rel = centers[None] - p0[:, None]
    t = np.clip((rel * d[:, None]).sum(2) / L2[:, None], 0.0, 0.97)
    closest = p0[:, None] + d[:, None] * t[..., None]
    dist = np.linalg.norm(closest - centers[None], axis=2)
    return np.any((dist < radii[None]) & (t > 0.02), axis=1)
