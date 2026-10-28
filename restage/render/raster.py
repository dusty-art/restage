"""Differentiable soft rasteriser: custom Metal kernels (``mx.fast.metal_kernel``) with a
hand-written backward (``mx.custom_function``).

Model (SoftRas-style, Liu et al. 2019, with an exact signed distance)
--------------------------------------------------------------------
For pixel centre ``p`` and triangle ``f`` (vertices in pixel coordinates, see
``camera.ndc_to_pixels``):

* ``sd``: signed distance from p to the triangle in pixels (+ inside, - outside);
* coverage ``D = sigmoid(sd / sigma)``; pairs with ``D < cutoff`` are ignored
  (this bounded support is what makes tile binning exact);
* barycentrics ``lam``: the pixel's own inside the triangle, those of the closest
  boundary point outside it (continuous, never negative), then made
  perspective-correct with the vertex depths ``z``: ``b_i ~ lam_i / z_i``;
* interpolated depth ``zp = 1 / sum(lam_i / z_i)``, opacity ``op = sum b_i o_i``
  and attributes ``a = sum b_i a_i``;
* effective coverage ``x = op * D``; soft z-buffer weight
  ``w = x * exp(-zp / gamma)`` (gamma in the units of z, i.e. metres).

Outputs per pixel:

* ``alpha``: ``min(1, sum_f x_f)`` (``alpha_mode="sum"``, default) or
  ``1 - prod_f (1 - x_f)`` (``"prob"``, the SoftRas silhouette). The sum form
  does not dip along the shared edges of a tessellated surface (two triangles
  sharing an edge have coverages that sum to exactly 1 there), which matters at
  64x36 where most pixels are within a pixel of some internal edge;
* ``depth = sum_f w_f zp_f / sum_f w_f`` and ``attrs = sum_f w_f a_f / sum_f w_f``:
  expectations *given that the pixel shows a surface* (0 where nothing is within
  the cutoff). Composite over a background with ``alpha * v + (1 - alpha) * bg``.

Implementation
--------------
1. ``setup`` kernel, one thread per (job, face): near-plane / back-face /
   degenerate culling, the padded screen bounding box -> tile rectangle, and a
   packed copy of the face's three (x, y, z, opacity) vertices.
2. Exclusive cumsum of per-face tile counts (MLX): each (face, tile) entry gets
   a fixed slot (face-major); entries beyond ``max_entries`` are dropped
   (``RasterOutput.n_entries`` reports the requested total).
3. ``bin_count`` / ``bin_fill`` kernels, one SIMD group per (job, 8x4 tile):
   ballot over all faces of the job -> per-tile face lists, in face order
   (deterministic, no atomics).
4. ``forward`` kernel, one SIMD group per tile (one lane per pixel) walking the
   tile's list; online log-sum-exp for the soft z-buffer.
5. ``backward`` kernel, same layout: per (face, tile) entry the 32 lanes'
   gradients are summed with a recursive-halving reduce-scatter (shuffles) and
   written, coalesced, to the entry's slot; the ``gather`` kernel sums the slots
   of every face corner incident to each vertex (static CSR table). All sums
   have a fixed order, so results are bitwise reproducible.

Gradients flow to the vertex NDC x/y, depth, opacity and attributes. Triangles
with any vertex closer than ``near`` are dropped (not clipped): no NaNs, but a
large triangle crossing the near plane disappears, so keep ground planes finely
tessellated near the camera (``shapes.ground_plane`` does this).
"""
from __future__ import annotations

import math
from typing import NamedTuple

import mlx.core as mx
import numpy as np

__all__ = ["rasterize", "rasterize_reference", "RasterOutput", "default_capacity", "cut_distance",
           "TILE_W", "TILE_H"]

TILE_W, TILE_H = 8, 4          # 32 pixels = one SIMD group
FAST_MATH = True               # Metal fast math for the forward / backward kernels (about 25 % faster)
_TGS = 128                     # threads per threadgroup for the tile kernels

# ----------------------------------------------------------------------------- Metal sources

_HEADER = """
// MLX passes inputs with fewer than 8 elements in the constant address space: load helpers for both.
inline float4 ld4(const device float* p, size_t i) { return *((const device float4*)(p) + i); }
inline float4 ld4(const constant float* p, size_t i) { return *((const constant float4*)(p) + i); }

struct Tri {
    float2 P[3];
    float z[3];
    float o[3];
    int vi[3];
    float2 e[3];
    float il2[3];
    float invA;
};

// q0..q2: (x, y, z, opacity) of the three corners; q3.xyz: vertex indices (bit cast)
inline Tri make_tri(float4 q0, float4 q1, float4 q2, float4 q3) {
    Tri T;
    T.P[0] = q0.xy; T.z[0] = q0.z; T.o[0] = q0.w;
    T.P[1] = q1.xy; T.z[1] = q1.z; T.o[1] = q1.w;
    T.P[2] = q2.xy; T.z[2] = q2.z; T.o[2] = q2.w;
    T.vi[0] = as_type<int>(q3.x); T.vi[1] = as_type<int>(q3.y); T.vi[2] = as_type<int>(q3.z);
    for (int j = 0; j < 3; ++j) {
        T.e[j] = T.P[(j + 1) % 3] - T.P[j];
        T.il2[j] = 1.0f / max(dot(T.e[j], T.e[j]), 1e-12f);
    }
    float A = T.e[0].x * (T.P[2].y - T.P[0].y) - T.e[0].y * (T.P[2].x - T.P[0].x);
    T.invA = 1.0f / A;
    return T;
}

struct Geo {
    float2 r[3];     // p - P_j
    float t[3];      // clamped segment parameter per edge
    float lam[3];    // interpolation barycentrics (clamped to the triangle)
    float2 q;        // p - closest boundary point
    float d;         // unsigned distance to the boundary
    float sd;        // signed distance (+ inside)
    int js;          // closest edge
    bool inside;
};

inline Geo eval_geo(thread const Tri& T, float2 p) {
    Geo G;
    float E[3];
    float dmin = 3.0e38f;
    G.js = 0;
    G.q = float2(0.0f);
    for (int j = 0; j < 3; ++j) {
        G.r[j] = p - T.P[j];
        E[j] = T.e[j].x * G.r[j].y - T.e[j].y * G.r[j].x;
        float tj = clamp(dot(G.r[j], T.e[j]) * T.il2[j], 0.0f, 1.0f);
        G.t[j] = tj;
        float2 qj = G.r[j] - tj * T.e[j];
        float dj = dot(qj, qj);
        if (dj < dmin) { dmin = dj; G.js = j; G.q = qj; }
    }
    float l0 = E[1] * T.invA, l1 = E[2] * T.invA, l2 = E[0] * T.invA;
    G.inside = (l0 >= 0.0f) && (l1 >= 0.0f) && (l2 >= 0.0f);
    G.d = sqrt(max(dmin, 1e-24f));
    G.sd = G.inside ? G.d : -G.d;
    if (G.inside) {
        G.lam[0] = l0; G.lam[1] = l1; G.lam[2] = l2;
    } else {
        G.lam[0] = 0.0f; G.lam[1] = 0.0f; G.lam[2] = 0.0f;
        int j = G.js;
        G.lam[j] = 1.0f - G.t[j];
        G.lam[(j + 1) % 3] = G.t[j];
    }
    return G;
}

// Recursive-halving reduce-scatter across the 32 lanes of a SIMD group: on entry every lane holds
// NP values; afterwards lane l holds the 32-lane sums of values l * (NP / 32) + i in v[i].
#define RS_STEP(v, HALF, M)                                              \\
    {                                                                    \\
        bool upper_ = (lane & (M)) != 0;                                 \\
        for (int i_ = 0; i_ < (HALF); ++i_) {                            \\
            float lo_ = v[i_], hi_ = v[i_ + (HALF)];                     \\
            float keep_ = upper_ ? hi_ : lo_;                            \\
            float send_ = upper_ ? lo_ : hi_;                            \\
            v[i_] = keep_ + simd_shuffle_xor(send_, ushort(M));          \\
        }                                                                \\
    }
#define REDUCE_SCATTER(v, NP)                                            \\
    RS_STEP(v, (NP) / 2, 16) RS_STEP(v, (NP) / 4, 8) RS_STEP(v, (NP) / 8, 4) \\
    RS_STEP(v, (NP) / 16, 2) RS_STEP(v, (NP) / 32, 1)
"""

# params: [0] sigma (px), [1] 1/gamma, [2] cut distance (px, >0), [3] near, [4] opacity_cull
_SETUP_SRC = """
    uint gid = thread_position_in_grid.x;
    int b = int(gid) / F;
    int f = int(gid) % F;
    float cut = params[2], near = params[3], ocull = params[4];
    float4 q[3];
    int vi[3];
    bool ok = true;
    for (int k = 0; k < 3; ++k) {
        int v = faces[f * 3 + k];
        vi[k] = v;
        const device float* src = pix + ((size_t)b * V + v) * 4;
        q[k] = float4(src[0], src[1], src[2], src[3]);
        ok = ok && isfinite(q[k].x) && isfinite(q[k].y) && isfinite(q[k].z) && (q[k].z > near);
    }
    float A = (q[1].x - q[0].x) * (q[2].y - q[0].y) - (q[1].y - q[0].y) * (q[2].x - q[0].x);
    ok = ok && isfinite(A) && (fabs(A) > 1e-12f);
    if (CULL) ok = ok && (A < 0.0f);           // front faces are clockwise in pixel space (y down)
    ok = ok && (max(q[0].w, max(q[1].w, q[2].w)) > ocull);
    uint rect = 255u;
    int cnt = 0;
    if (ok) {
        float x0 = min(q[0].x, min(q[1].x, q[2].x)) - cut, x1 = max(q[0].x, max(q[1].x, q[2].x)) + cut;
        float y0 = min(q[0].y, min(q[1].y, q[2].y)) - cut, y1 = max(q[0].y, max(q[1].y, q[2].y)) + cut;
        // pixels whose centre (i + 0.5) lies in [x0, x1]
        float fx0 = clamp(ceil(x0 - 0.5f), -1.0f, float(W)), fx1 = clamp(floor(x1 - 0.5f), -1.0f, float(W));
        float fy0 = clamp(ceil(y0 - 0.5f), -1.0f, float(H)), fy1 = clamp(floor(y1 - 0.5f), -1.0f, float(H));
        int ix0 = max(int(fx0), 0), ix1 = min(int(fx1), W - 1);
        int iy0 = max(int(fy0), 0), iy1 = min(int(fy1), H - 1);
        if (ix0 <= ix1 && iy0 <= iy1) {
            uint tx0 = uint(ix0 / TW), tx1 = uint(ix1 / TW), ty0 = uint(iy0 / TH), ty1 = uint(iy1 / TH);
            rect = tx0 | (tx1 << 8) | (ty0 << 16) | (ty1 << 24);
            cnt = int((tx1 - tx0 + 1) * (ty1 - ty0 + 1));
        }
    }
    rects[gid] = rect;
    counts[gid] = cnt;
    device float4* t = (device float4*)(tri) + (size_t)gid * 4;
    t[0] = q[0]; t[1] = q[1]; t[2] = q[2];
    t[3] = float4(as_type<float>(vi[0]), as_type<float>(vi[1]), as_type<float>(vi[2]), 0.0f);
"""

_BIN_COUNT_SRC = """
    uint gid = thread_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    int tg = int(gid >> 5);
    int b = tg / NT, tile = tg % NT;
    uint tx = uint(tile % NTX), ty = uint(tile / NTX);
    auto R = rects + (size_t)b * F;
    uint total = 0;
    for (int f0 = 0; f0 < F; f0 += 32) {
        int f = f0 + int(lane);
        uint hit = 0;
        if (f < F) {
            uint r = R[f];
            hit = (tx >= (r & 255u) && tx <= ((r >> 8) & 255u) && ty >= ((r >> 16) & 255u) && ty <= (r >> 24)) ? 1u : 0u;
        }
        total += simd_sum(hit);
    }
    if (lane == 0) tcount[tg] = int(total);
"""

_BIN_FILL_SRC = """
    uint gid = thread_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    int tg = int(gid >> 5);
    int b = tg / NT, tile = tg % NT;
    uint tx = uint(tile % NTX), ty = uint(tile / NTX);
    auto R = rects + (size_t)b * F;
    int pos = tstart[tg];
    for (int f0 = 0; f0 < F; f0 += 32) {
        int f = f0 + int(lane);
        uint hit = 0;
        if (f < F) {
            uint r = R[f];
            hit = (tx >= (r & 255u) && tx <= ((r >> 8) & 255u) && ty >= ((r >> 16) & 255u) && ty <= (r >> 24)) ? 1u : 0u;
        }
        uint pre = simd_prefix_exclusive_sum(hit);
        if (hit) tlist[pos + int(pre)] = f;
        pos += int(simd_sum(hit));
    }
"""

# Shared head of the forward / backward kernels: lane `lane` shades pixel (ix, iy) of tile `tile` of
# job `b` and walks the tile's face list (all lanes read the same entry: broadcast loads).
_TILE_HEAD = """
    uint gid = thread_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    int tg = int(gid >> 5);
    int b = tg / NT, tile = tg % NT;
    int tx = tile % NTX, ty = tile / NTX;
    int ix = tx * TW + int(lane % TW);
    int iy = ty * TH + int(lane / TW);
    bool pix_ok = (ix < W) && (iy < H);
    float2 p = float2(float(ix) + 0.5f, float(iy) + 0.5f);
    float isig = 1.0f / params[0], invg = params[1], cut = params[2];
    int s0 = tstart[tg], n = tcount[tg];
    size_t tri0 = (size_t)b * F * 4;           // float4 offsets of this job's faces / vertex attributes
    size_t att0 = (size_t)b * V * C4;
"""

_FWD_SRC = _TILE_HEAD + """
    float m = -1e30f, den = 0.0f, nz = 0.0f, acc = 0.0f;
    float4 na[C4];
    for (int c = 0; c < C4; ++c) na[c] = float4(0.0f);
    for (int j = 0; j < n; ++j) {
        int f = tlist[s0 + j];
        size_t tf = tri0 + (size_t)f * 4;
        Tri T = make_tri(ld4(tri, tf), ld4(tri, tf + 1), ld4(tri, tf + 2), ld4(tri, tf + 3));
        Geo G = eval_geo(T, p);
        if (!pix_ok || G.sd < -cut) continue;
        float D = 1.0f / (1.0f + metal::exp(-G.sd * isig));
        float u0 = G.lam[0] / T.z[0], u1 = G.lam[1] / T.z[1], u2 = G.lam[2] / T.z[2];
        float zp = 1.0f / (u0 + u1 + u2);
        float b0 = u0 * zp, b1 = u1 * zp, b2 = u2 * zp;
        float x = (b0 * T.o[0] + b1 * T.o[1] + b2 * T.o[2]) * D;
        if (AMODE == 0) acc += x;
        else acc += metal::log(max(1.0f - x, 1e-6f));
        if (x > 0.0f) {
            float logit = metal::log(x) - zp * invg;
            float nm = max(m, logit);
            float sc = metal::exp(m - nm), w = metal::exp(logit - nm);
            m = nm;
            den = den * sc + w;
            nz = nz * sc + w * zp;
            size_t i0 = att0 + (size_t)T.vi[0] * C4, i1 = att0 + (size_t)T.vi[1] * C4;
            size_t i2 = att0 + (size_t)T.vi[2] * C4;
            for (int c = 0; c < C4; ++c)
                na[c] = na[c] * sc + w * (b0 * ld4(attr, i0 + c) + b1 * ld4(attr, i1 + c) + b2 * ld4(attr, i2 + c));
        }
    }
    if (!pix_ok) return;
    size_t o = ((size_t)b * H + iy) * W + ix;
    float inv = den > 0.0f ? 1.0f / den : 0.0f;
    alpha[o] = (AMODE == 0) ? min(acc, 1.0f) : 1.0f - metal::exp(acc);
    depth[o] = nz * inv;
    device float4* ao = (device float4*)(aimg) + o * C4;
    for (int c = 0; c < C4; ++c) ao[c] = na[c] * inv;
    aux[o * 2 + 0] = den > 0.0f ? m + metal::log(den) : -1e30f;
    aux[o * 2 + 1] = acc;
"""

_BWD_SRC = _TILE_HEAD + """
    constexpr int K = 4 * C4 + 4;
    // per-pixel upstream gradients and forward state
    float4 ga[C4];
    float g_al = 0.0f, g_z = 0.0f, lse = -1e30f, acc = 0.0f, G0 = 0.0f;
    if (pix_ok) {
        size_t o = ((size_t)b * H + iy) * W + ix;
        g_al = galpha[o]; g_z = gdepth[o];
        lse = aux[o * 2 + 0]; acc = aux[o * 2 + 1];
        G0 = g_z * depth[o];
        for (int c = 0; c < C4; ++c) { ga[c] = ld4(gaimg, o * C4 + c); G0 += dot(ga[c], ld4(aimg, o * C4 + c)); }
    } else {
        for (int c = 0; c < C4; ++c) ga[c] = float4(0.0f);
    }
    bool has_den = lse > -1e29f;
    float Tr = (AMODE == 0) ? 0.0f : metal::exp(acc);
    float sat = (AMODE == 0 && acc >= 1.0f) ? 0.0f : 1.0f;     // d min(acc, 1) / d acc
    for (int j = 0; j < n; ++j) {
        int f = tlist[s0 + j];
        size_t tf = tri0 + (size_t)f * 4;
        Tri T = make_tri(ld4(tri, tf), ld4(tri, tf + 1), ld4(tri, tf + 2), ld4(tri, tf + 3));
        uint r = rects[b * F + f];
        int rtx0 = int(r & 255u), rtx1 = int((r >> 8) & 255u), rty0 = int((r >> 16) & 255u);
        int slot = offs[b * F + f] + (ty - rty0) * (rtx1 - rtx0 + 1) + (tx - rtx0);
        device float* out = partial + (size_t)slot * (3 * K);
        Geo G = eval_geo(T, p);
        bool act = pix_ok && (G.sd >= -cut);
        if (!simd_any(act)) {
            for (int k = int(lane); k < 3 * K; k += 32) out[k] = 0.0f;
            continue;
        }
        float vals[NP];
        for (int k = 0; k < NP; ++k) vals[k] = 0.0f;
        if (act) {
            size_t i0 = att0 + (size_t)T.vi[0] * C4, i1 = att0 + (size_t)T.vi[1] * C4;
            size_t i2 = att0 + (size_t)T.vi[2] * C4;
            float D = 1.0f / (1.0f + metal::exp(-G.sd * isig));
            float u[3], bb[3];
            for (int k = 0; k < 3; ++k) u[k] = G.lam[k] / T.z[k];
            float zp = 1.0f / (u[0] + u[1] + u[2]);
            for (int k = 0; k < 3; ++k) bb[k] = u[k] * zp;
            float op = bb[0] * T.o[0] + bb[1] * T.o[1] + bb[2] * T.o[2];
            float x = op * D;
            float dk[3] = {0.0f, 0.0f, 0.0f};
            for (int c = 0; c < C4; ++c) {
                dk[0] += dot(ga[c], ld4(attr, i0 + c)); dk[1] += dot(ga[c], ld4(attr, i1 + c));
                dk[2] += dot(ga[c], ld4(attr, i2 + c));
            }
            float val = g_z * zp + bb[0] * dk[0] + bb[1] * dk[1] + bb[2] * dk[2] - G0;
            // d alpha / d x
            float gx = (AMODE == 0) ? g_al * sat : ((1.0f - x > 1e-6f) ? g_al * Tr / (1.0f - x) : 0.0f);
            float gzp = 0.0f, what = 0.0f;
            if (has_den) {
                gx += metal::exp(min(-zp * invg - lse, 30.0f)) * val;     // (what / x) * val
                if (x > 0.0f) {
                    what = metal::exp(metal::log(x) - zp * invg - lse);
                    gzp = what * (g_z - val * invg);
                }
            }
            float gop = gx * D, gD = gx * op;
            float gb[3], sgb = 0.0f;
            for (int k = 0; k < 3; ++k) {
                vals[k * K + 3] = gop * bb[k];                              // d / d opacity_k
                gb[k] = gop * T.o[k] + what * dk[k];
                sgb += gb[k] * bb[k];
            }
            float glam[3];
            for (int k = 0; k < 3; ++k) {
                float gu = zp * (gb[k] - sgb) - gzp * zp * zp;
                glam[k] = gu / T.z[k];
                vals[k * K + 2] = -gu * u[k] / T.z[k];                      // d / d z_k
            }
            float2 gP[3] = {float2(0.0f), float2(0.0f), float2(0.0f)};
            // coverage -> signed distance (envelope theorem on the closest point)
            float gsd = gD * D * (1.0f - D) * isig;
            int js = G.js;
            float tj = G.t[js];
            if (G.d > 1e-12f) {
                float2 nrm = G.q / G.d;
                float cc = G.inside ? -gsd : gsd;
                gP[js] += cc * (1.0f - tj) * nrm;
                gP[(js + 1) % 3] += cc * tj * nrm;
            }
            // barycentrics
            if (G.inside) {
                float gA = -(glam[0] * G.lam[0] + glam[1] * G.lam[1] + glam[2] * G.lam[2]) * T.invA;
                for (int jj = 0; jj < 3; ++jj) {
                    float gE = glam[(jj + 2) % 3] * T.invA;        // lam[(jj+2)%3] = E[jj] / A
                    float2 rn = G.r[(jj + 1) % 3], rj = G.r[jj];
                    gP[jj] += gE * float2(-rn.y, rn.x);
                    gP[(jj + 1) % 3] += gE * float2(rj.y, -rj.x);
                }
                for (int k = 0; k < 3; ++k) {
                    float2 P1 = T.P[(k + 1) % 3], P2 = T.P[(k + 2) % 3];
                    gP[k] += gA * float2(P1.y - P2.y, P2.x - P1.x);
                }
            } else if (tj > 0.0f && tj < 1.0f) {
                float gt = glam[(js + 1) % 3] - glam[js];
                float2 ee = T.e[js], rr = G.r[js];
                float il2 = T.il2[js];
                gP[js] += gt * (-ee - rr + 2.0f * tj * ee) * il2;
                gP[(js + 1) % 3] += gt * (rr - 2.0f * tj * ee) * il2;
            }
            for (int k = 0; k < 3; ++k) {
                vals[k * K + 0] = gP[k].x; vals[k * K + 1] = gP[k].y;
                for (int c = 0; c < C4; ++c) {
                    float4 g = what * bb[k] * ga[c];
                    vals[k * K + 4 + 4 * c + 0] = g.x; vals[k * K + 4 + 4 * c + 1] = g.y;
                    vals[k * K + 4 + 4 * c + 2] = g.z; vals[k * K + 4 + 4 * c + 3] = g.w;
                }
            }
        }
        // sum over the 32 pixels of the tile and write this entry's slot (coalesced)
        REDUCE_SCATTER(vals, NP)
        for (int i2 = 0; i2 < NP / 32; ++i2) {
            int idx = int(lane) * (NP / 32) + i2;
            if (idx < 3 * K) out[idx] = vals[i2];
        }
    }
"""

_GATHER_SRC = """
    uint gid = thread_position_in_grid.x;
    constexpr int K4 = C4 + 1;                 // float4 groups per face corner
    int q = int(gid) % K4;
    int bv = int(gid) / K4;
    int b = bv / V, v = bv % V;
    float4 s = float4(0.0f);
    for (int k = inc_ptr[v]; k < inc_ptr[v + 1]; ++k) {
        int corner = inc_idx[k];
        int f = corner / 3, i = corner % 3;
        int o = offs[b * F + f], n = counts[b * F + f];
        for (int e = o; e < o + n; ++e) s += ld4(partial, (size_t)e * (3 * K4) + i * K4 + q);
    }
    ((device float4*)(grad))[gid] = s;
"""

_KERNELS: dict = {}


def _kernels(fast: bool):
    """Kernel objects (JIT-compiled once per template); fast selects Metal fast math for fwd / bwd."""
    k = _KERNELS.get(fast)
    if k is None:
        opt = {"math_mode": "fast" if fast else "safe"}
        sfx = "_fast" if fast else ""
        k = {
            "setup": mx.fast.metal_kernel(name="rs_setup", input_names=["pix", "faces", "params"],
                                          output_names=["rects", "counts", "tri"], source=_SETUP_SRC),
            "count": mx.fast.metal_kernel(name="rs_bin_count", input_names=["rects"], output_names=["tcount"],
                                          source=_BIN_COUNT_SRC),
            "fill": mx.fast.metal_kernel(name="rs_bin_fill", input_names=["rects", "tstart"],
                                         output_names=["tlist"], source=_BIN_FILL_SRC),
            "fwd": mx.fast.metal_kernel(name="rs_forward" + sfx,
                                        input_names=["tri", "attr", "tlist", "tstart", "tcount", "params"],
                                        output_names=["alpha", "depth", "aimg", "aux"], source=_FWD_SRC,
                                        header=_HEADER, compile_options=opt),
            "bwd": mx.fast.metal_kernel(name="rs_backward" + sfx,
                                        input_names=["tri", "attr", "tlist", "tstart", "tcount", "rects", "offs",
                                                     "params", "galpha", "gdepth", "gaimg", "depth", "aimg", "aux"],
                                        output_names=["partial"], source=_BWD_SRC, header=_HEADER,
                                        compile_options=opt),
            "gather": mx.fast.metal_kernel(name="rs_gather",
                                           input_names=["partial", "offs", "counts", "inc_ptr", "inc_idx"],
                                           output_names=["grad"], source=_GATHER_SRC, header=_HEADER),
        }
        _KERNELS[fast] = k
    return k


# ----------------------------------------------------------------------------- static topology

class _Topology(NamedTuple):
    faces: mx.array        # [F, 3] int32
    inc_ptr: mx.array      # [V + 1] int32: CSR row pointers, vertex -> incident corners
    inc_idx: mx.array      # [3F] int32: corner ids (3 * face + k), ordered by (face, k)


_TOPO_CACHE: dict = {}


def _topology(faces_np: np.ndarray, V: int) -> _Topology:
    key = (faces_np.shape, V, hash(faces_np.tobytes()))
    t = _TOPO_CACHE.get(key)
    if t is None:
        if faces_np.size and (faces_np.min() < 0 or faces_np.max() >= V):
            raise ValueError("face index out of range")
        corner_v = faces_np.reshape(-1)
        order = np.argsort(corner_v, kind="stable")          # corners grouped by vertex, in corner order
        counts = np.bincount(corner_v, minlength=V)
        ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
        t = _Topology(mx.array(faces_np.astype(np.int32)), mx.array(ptr), mx.array(order.astype(np.int32)))
        if len(_TOPO_CACHE) > 64:
            _TOPO_CACHE.clear()
        _TOPO_CACHE[key] = t
    return t


# ----------------------------------------------------------------------------- public API

class RasterOutput(NamedTuple):
    """Rasteriser outputs.

    alpha [B, H, W]; depth [B, H, W]; attrs [B, H, W, C] (None if no attributes were given);
    n_entries: int32 scalar, number of (face, tile) entries requested (compare with capacity:
    faces beyond the capacity are dropped, in (job, face) order).
    """
    alpha: mx.array
    depth: mx.array
    attrs: mx.array | None
    n_entries: mx.array
    capacity: int


def default_capacity(B: int, F: int, height: int, width: int) -> int:
    """Default ``max_entries`` for the batch: per job, every face in ``3 * s`` tiles plus 32 faces per
    tile, with ``s = sqrt(H * W / (64 * 36))`` (triangles cover more tiles at higher resolution).

    The realistic workload (bench_render) uses about 55 % of this at 64x36 and 40 % at 128x72.
    The backward pass allocates ``capacity * 12 * (ceil(C / 4) + 1)`` floats for per-entry partials.
    """
    nt = -(-width // TILE_W) * -(-height // TILE_H)
    s = max(1.0, math.sqrt(height * width / (64 * 36)))
    return int(B * (math.ceil(3 * s * F) + 32 * nt))


def cut_distance(sigma: float, cutoff: float) -> float:
    """Distance (px) outside a triangle at which coverage sigmoid(sd / sigma) drops to `cutoff`."""
    return float(sigma * math.log((1.0 - cutoff) / cutoff))


def rasterize(verts: mx.array, faces, attrs: mx.array | None = None, opacity: mx.array | None = None, *,
              height: int, width: int, sigma: float = 0.5, gamma: float = 0.05, near: float = 0.05,
              cutoff: float = 1e-3, cull_backfaces: bool = True, alpha_mode: str = "sum",
              opacity_cull: float = -1.0, max_entries: int | None = None,
              fast_math: bool | None = None) -> RasterOutput:
    """Soft-rasterise a batch of meshes with shared topology.

    Args:
      verts: [B, V, 3] NDC x, NDC y (+y up) and positive view depth, e.g. ``camera.project``.
      faces: [F, 3] static int array (numpy or list), counter-clockwise seen from the front
        (three.js / glTF front faces).
      attrs: optional [B, V, C] per-vertex attributes (C <= 32; colour, one-hot instance ids,
        region features ...), interpolated perspective-correctly and composited with the
        depth-softmax weights. Internally padded to a multiple of 4 channels.
      opacity: optional [B, V] per-vertex opacity in [0, 1] (default 1); multiplies coverage,
        so a part with opacity 0 neither shows nor occludes but still receives gradients.
      height, width: output size in pixels.
      sigma: edge softness in pixels. gamma: depth temperature (units of z, metres).
      near: triangles with a vertex at depth <= near are dropped.
      cutoff: coverage below which a (pixel, triangle) pair is ignored (sets the bounding-box
        padding ``sigma * log((1 - cutoff) / cutoff)``).
      cull_backfaces: drop triangles facing away from the camera.
      alpha_mode: "sum" (``min(1, sum x)``) or "prob" (``1 - prod(1 - x)``).
      opacity_cull: faces whose three vertex opacities are all <= this are skipped (default -1:
        never; set 0 to skip fully transparent faces at the cost of their fade-in gradient).
      max_entries: capacity of the (face, tile) list for the whole batch (static; default
        ``default_capacity``).
      fast_math: Metal fast math in the forward / backward kernels (default ``FAST_MATH``).

    Returns:
      RasterOutput(alpha [B,H,W], depth [B,H,W], attrs [B,H,W,C] or None, n_entries, capacity).
      Differentiable w.r.t. verts, attrs and opacity.
    """
    faces_np = np.ascontiguousarray(np.asarray(faces, dtype=np.int32).reshape(-1, 3))
    B, V, _ = verts.shape
    F = int(faces_np.shape[0])
    H, W = int(height), int(width)
    ntx, nty = -(-W // TILE_W), -(-H // TILE_H)
    if ntx > 255 or nty > 255:
        raise ValueError("image too large for the 8-bit tile rectangles (max 2040 x 1020)")
    if alpha_mode not in ("sum", "prob"):
        raise ValueError("alpha_mode must be 'sum' or 'prob'")
    has_attr = attrs is not None
    C = int(attrs.shape[-1]) if has_attr else 0
    if C > 32:
        raise ValueError("at most 32 attribute channels")
    C4 = max(1, -(-C // 4))
    if has_attr:
        attr_in = attrs.astype(mx.float32)
        if 4 * C4 != C:
            attr_in = mx.concatenate([attr_in, mx.zeros((B, V, 4 * C4 - C), mx.float32)], -1)
    else:
        attr_in = mx.zeros((B, V, 4), mx.float32)
    if opacity is None:
        opacity = mx.ones((B, V), mx.float32)
    cap = int(max_entries) if max_entries is not None else default_capacity(B, F, H, W)
    if B * F == 0:
        z = mx.zeros((B, H, W))
        return RasterOutput(z, z, mx.zeros((B, H, W, C)) if has_attr else None, mx.array(0), cap)
    topo = _topology(faces_np, V)
    params = mx.array([sigma, 1.0 / gamma, cut_distance(sigma, cutoff), near, opacity_cull], mx.float32)

    X = (verts[..., 0] + 1.0) * (0.5 * W)
    Y = (1.0 - verts[..., 1]) * (0.5 * H)
    pix = mx.stack([X, Y, verts[..., 2], opacity.astype(mx.float32)], -1).astype(mx.float32)

    fast = FAST_MATH if fast_math is None else bool(fast_math)
    core = _make_core(topo, B, V, F, C4, H, W, ntx, nty, cap, params, bool(cull_backfaces),
                      0 if alpha_mode == "sum" else 1, fast)
    alpha, depth, aimg, n_entries = core(pix, attr_in)[:4]
    return RasterOutput(alpha, depth, aimg[..., :C] if has_attr else None, n_entries, cap)


def _make_core(topo: _Topology, B, V, F, C4, H, W, ntx, nty, cap, params, cull, amode, fast):
    NT = ntx * nty
    K = 4 * C4 + 4                                           # gradient values per face corner
    npow = max(32, 1 << (3 * K - 1).bit_length())            # reduce-scatter width (power of two >= 3K)
    tmpl = [("C4", C4), ("F", F), ("V", V), ("W", W), ("H", H), ("NTX", ntx), ("NT", NT), ("TW", TILE_W),
            ("TH", TILE_H), ("CULL", int(cull)), ("AMODE", int(amode)), ("NP", npow)]
    nthreads_tiles = B * NT * 32
    kern = _kernels(fast)

    def binning(pix):
        rects, counts, tri = kern["setup"](inputs=[pix, topo.faces, params], template=tmpl, grid=(B * F, 1, 1),
                                           threadgroup=(256, 1, 1), output_shapes=[(B * F,), (B * F,), (B * F * 16,)],
                                           output_dtypes=[mx.uint32, mx.int32, mx.float32])
        csum = mx.cumsum(counts)
        keep = csum <= cap
        counts = mx.where(keep, counts, 0)
        rects = mx.where(keep, rects, mx.array(255, mx.uint32))
        offs = csum - counts
        n_entries = csum[-1]
        tcount = kern["count"](inputs=[rects], template=tmpl, grid=(nthreads_tiles, 1, 1), threadgroup=(_TGS, 1, 1),
                               output_shapes=[(B * NT,)], output_dtypes=[mx.int32])[0]
        tstart = mx.cumsum(tcount) - tcount
        tlist = kern["fill"](inputs=[rects, tstart], template=tmpl, grid=(nthreads_tiles, 1, 1),
                             threadgroup=(256, 1, 1), output_shapes=[(cap,)], output_dtypes=[mx.int32])[0]
        return tri, rects, counts, offs, n_entries, tcount, tstart, tlist

    @mx.custom_function
    def core(pix, attr):
        tri, rects, counts, offs, n_entries, tcount, tstart, tlist = binning(mx.stop_gradient(pix))
        alpha, depth, aimg, aux = kern["fwd"](
            inputs=[tri, attr, tlist, tstart, tcount, params], template=tmpl,
            grid=(nthreads_tiles, 1, 1), threadgroup=(_TGS, 1, 1),
            output_shapes=[(B, H, W), (B, H, W), (B, H, W, 4 * C4), (B, H, W, 2)],
            output_dtypes=[mx.float32] * 4)
        return alpha, depth, aimg, n_entries, aux, tri, rects, counts, offs, tcount, tstart, tlist

    @core.vjp
    def core_vjp(primals, cotangents, outputs):
        _, attr = primals
        g_alpha, g_depth, g_aimg = cotangents[0], cotangents[1], cotangents[2]
        _, depth, aimg, _, aux, tri, rects, counts, offs, tcount, tstart, tlist = outputs
        partial = kern["bwd"](
            inputs=[tri, attr, tlist, tstart, tcount, rects, offs, params,
                    g_alpha.astype(mx.float32), g_depth.astype(mx.float32), g_aimg.astype(mx.float32),
                    depth, aimg, aux],
            template=tmpl, grid=(nthreads_tiles, 1, 1), threadgroup=(_TGS, 1, 1),
            output_shapes=[(cap * 3 * K,)], output_dtypes=[mx.float32])[0]
        grad = kern["gather"](inputs=[partial, offs, counts, topo.inc_ptr, topo.inc_idx], template=tmpl,
                              grid=(B * V * (K // 4), 1, 1), threadgroup=(256, 1, 1),
                              output_shapes=[(B, V, K)], output_dtypes=[mx.float32])[0]
        return grad[..., :4], grad[..., 4:]

    return core


# ----------------------------------------------------------------------------- dense reference

def rasterize_reference(verts: mx.array, faces, attrs: mx.array | None = None, opacity: mx.array | None = None, *,
                        height: int, width: int, sigma: float = 0.5, gamma: float = 0.05, near: float = 0.05,
                        cutoff: float = 1e-3, cull_backfaces: bool = True, alpha_mode: str = "sum",
                        opacity_cull: float = -1.0):
    """Plain-MLX dense version of ``rasterize`` (every pixel x every face), for tests.

    Same maths and the same culling rules; differentiable through MLX autodiff.
    Memory is O(B * H * W * F * C); use only on small cases. Returns (alpha, depth, attrs|None).
    """
    faces_np = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    B, V, _ = verts.shape
    H, W = int(height), int(width)
    has_attr = attrs is not None
    if opacity is None:
        opacity = mx.ones((B, V))
    X = (verts[..., 0] + 1.0) * (0.5 * W)
    Y = (1.0 - verts[..., 1]) * (0.5 * H)
    Z = verts[..., 2]
    fi = mx.array(faces_np)
    PX = mx.take(X, fi, axis=1)     # [B, F, 3]
    PY = mx.take(Y, fi, axis=1)
    PZ = mx.take(Z, fi, axis=1)
    PO = mx.take(opacity, fi, axis=1)
    cut = cut_distance(sigma, cutoff)

    # face validity (no gradient)
    A = (PX[..., 1] - PX[..., 0]) * (PY[..., 2] - PY[..., 0]) - (PY[..., 1] - PY[..., 0]) * (PX[..., 2] - PX[..., 0])
    ok = mx.all(PZ > near, axis=-1) & (mx.abs(A) > 1e-12)
    if cull_backfaces:
        ok = ok & (A < 0)
    ok = ok & (mx.max(PO, axis=-1) > opacity_cull)
    ok = mx.stop_gradient(ok)
    # keep invalid faces finite (they are masked out below; NaNs would leak into the masked gradients)
    PZ = mx.where(ok[..., None], PZ, 1.0)

    ys, xs = np.meshgrid(np.arange(H) + 0.5, np.arange(W) + 0.5, indexing="ij")
    px = mx.array(xs.reshape(1, -1, 1).astype(np.float32))      # [1, HW, 1]
    py = mx.array(ys.reshape(1, -1, 1).astype(np.float32))

    def ex(a):  # [B, F] -> [B, 1, F]
        return a[:, None, :]

    Pxs = [ex(PX[..., k]) for k in range(3)]
    Pys = [ex(PY[..., k]) for k in range(3)]
    rx = [px - Pxs[j] for j in range(3)]
    ry = [py - Pys[j] for j in range(3)]
    exs = [Pxs[(j + 1) % 3] - Pxs[j] for j in range(3)]
    eys = [Pys[(j + 1) % 3] - Pys[j] for j in range(3)]
    E = [exs[j] * ry[j] - eys[j] * rx[j] for j in range(3)]
    Af = ex(A)
    lam_in = [E[1] / Af, E[2] / Af, E[0] / Af]
    inside = (lam_in[0] >= 0) & (lam_in[1] >= 0) & (lam_in[2] >= 0)
    ts, d2s = [], []
    for j in range(3):
        il2 = 1.0 / mx.maximum(exs[j] * exs[j] + eys[j] * eys[j], 1e-12)
        t = mx.clip((rx[j] * exs[j] + ry[j] * eys[j]) * il2, 0.0, 1.0)
        qx, qy = rx[j] - t * exs[j], ry[j] - t * eys[j]
        ts.append(t)
        d2s.append(qx * qx + qy * qy)
    # argmin with first-wins ties, as in the kernel
    sel1 = d2s[1] < d2s[0]
    dmin01 = mx.where(sel1, d2s[1], d2s[0])
    sel2 = d2s[2] < dmin01
    js = mx.where(sel2, 2, mx.where(sel1, 1, 0))
    dmin = mx.where(sel2, d2s[2], dmin01)
    d = mx.sqrt(mx.maximum(dmin, 1e-24))
    sd = mx.where(inside, d, -d)
    tsel = mx.where(js == 0, ts[0], mx.where(js == 1, ts[1], ts[2]))
    lam_out = []
    for k in range(3):
        # vertex k gets 1 - t on edge k (start), t on edge k-1 (end), 0 on the opposite edge
        lam_out.append(mx.where(js == k, 1.0 - tsel, mx.where(js == (k - 1) % 3, tsel, 0.0)))
    lam = [mx.where(inside, lam_in[k], lam_out[k]) for k in range(3)]
    active = (sd >= -cut) & ex(ok)
    D = mx.sigmoid(sd / sigma)
    u = [lam[k] / ex(PZ[..., k]) for k in range(3)]
    zp = 1.0 / (u[0] + u[1] + u[2])
    bw = [u[k] * zp for k in range(3)]
    op = bw[0] * ex(PO[..., 0]) + bw[1] * ex(PO[..., 1]) + bw[2] * ex(PO[..., 2])
    x = mx.where(active, op * D, 0.0)                       # [B, HW, F]
    if alpha_mode == "sum":
        alpha = mx.minimum(x.sum(-1), 1.0)
    else:
        alpha = 1.0 - mx.exp(mx.log(mx.maximum(1.0 - x, 1e-6)).sum(-1))
    pos = x > 0
    logit = mx.where(pos, mx.log(mx.where(pos, x, 1.0)) - zp / gamma, -mx.inf)
    any_pos = mx.any(pos, axis=-1, keepdims=True)
    lmax = mx.stop_gradient(mx.where(any_pos, mx.max(logit, -1, keepdims=True), 0.0))
    w = mx.where(pos, mx.exp(logit - lmax), 0.0)
    den = w.sum(-1, keepdims=True)
    wn = w / mx.where(den > 0, den, 1.0)
    depth = (wn * zp).sum(-1)
    out_attr = None
    if has_attr:
        Pa = mx.take(attrs, fi, axis=1)                     # [B, F, 3, C]
        ap = sum(bw[k][..., None] * Pa[:, None, :, k, :] for k in range(3))   # [B, HW, F, C]
        out_attr = (wn[..., None] * ap).sum(-2).reshape(B, H, W, -1)
    return alpha.reshape(B, H, W), depth.reshape(B, H, W), out_attr
