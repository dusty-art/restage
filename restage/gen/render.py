"""Fast offline renderer for `.restage` scene documents (moderngl, headless).

The look matches the Restage viewer (three.js r170, MeshStandardMaterial with flatShading and
vertexColors, HemisphereLight(sky, ground, 2.2 * ambient), DirectionalLight(2.4 * sun) with
shadows, ACES filmic tone mapping, sRGB output, linear THREE.Fog): flat-shaded low-poly triangles,
colour = vertex colour x material colour, solid sky-colour background. Per fragment, in linear RGB::

    hemi = mix(ground, sky, 0.5 * n.y + 0.5)
    c = albedo / pi * (2.2 * ambient * hemi + 2.4 * sun * max(0, dot(n, -sun_dir)) * shadow) + emissive
    c = srgb(aces_filmic(c))
    c = mix(c, srgb(fog_color), smoothstep(fog_near, fog_far, depth))    # fog after encoding, as three.js

Per frame it produces RGB (4x MSAA), metric depth (distance along the camera axis, 0 for sky),
instance ids (0 = sky / set structure, 1..N = entities in entity order, 1000 + k = location prop k)
and optionally world-space normals. Skinning is linear blend skinning on the CPU.

Domain randomisation (`DomainRandomization`): colour jitter, sensor noise, vignetting, motion
blur (sub-frame accumulation with interpolated tracks), shadows, and H.264 compression strength.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field

import numpy as np

from ..scene import forward_kinematics, quat_rotate, quat_to_mat

STRUCTURAL = {"floor", "wall", "walls", "ceiling", "room", "terrain", "ground", "road", "sidewalk", "voxel_chunk",
              "backdrop", "sky", "water", "markings", "crosswalk", "stage", "rug"}
FFMPEG = "/opt/homebrew/bin/ffmpeg"

VS = """
#version 330
uniform mat4 u_view;
uniform mat4 u_proj;
in vec3 in_pos;
in vec3 in_col;
in vec3 in_emi;
in float in_id;
out vec3 v_wpos;
out vec3 v_col;
out vec3 v_emi;
out float v_depth;
flat out float v_id;
void main() {
    vec4 vp = u_view * vec4(in_pos, 1.0);
    v_wpos = in_pos;
    v_col = in_col;
    v_emi = in_emi;
    v_id = in_id;
    v_depth = -vp.z;
    gl_Position = u_proj * vp;
}
"""

FS_COMMON = """
uniform vec3 u_cam;
uniform vec3 u_sun_dir;
uniform float u_sun;
uniform float u_ambient;
uniform vec3 u_sky;
uniform vec3 u_ground;
uniform int u_fog_mode;     // 0 none, 1 linear, 2 exp2
uniform vec3 u_fog_col;
uniform vec3 u_fog;         // near, far, density
uniform float u_logc;
uniform int u_shadows;
uniform sampler2DShadow u_shadow_map;
uniform mat4 u_shadow_mat;
in vec3 v_wpos;
in vec3 v_col;
in vec3 v_emi;
in float v_depth;
flat in float v_id;
vec3 flat_normal() {
    vec3 n = normalize(cross(dFdx(v_wpos), dFdy(v_wpos)));
    if (dot(n, u_cam - v_wpos) < 0.0) n = -n;
    return n;
}
float log_depth() { return log2(max(1e-6, 1.0 + v_depth)) * u_logc; }
"""

FS_COLOR = """
#version 330
""" + FS_COMMON + """
out vec4 f_color;
vec3 to_srgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}
vec3 rrt_odt(vec3 v) {
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}
vec3 aces(vec3 c) {   // three.js ACESFilmicToneMapping, exposure 1
    const mat3 IN = mat3(vec3(0.59719, 0.07600, 0.02840), vec3(0.35458, 0.90834, 0.13383), vec3(0.04823, 0.01566, 0.83777));
    const mat3 OUT = mat3(vec3(1.60475, -0.10208, -0.00327), vec3(-0.53108, 1.10813, -0.07276), vec3(-0.07367, -0.00605, 1.07602));
    c *= 1.0 / 0.6;
    return clamp(OUT * rrt_odt(IN * c), 0.0, 1.0);
}
void main() {
    vec3 n = flat_normal();
    vec3 L = normalize(-u_sun_dir);
    float ndl = max(dot(n, L), 0.0);
    float vis = 1.0;
    if (u_shadows == 1 && ndl > 0.0) {
        vec4 sp = u_shadow_mat * vec4(v_wpos + n * 0.03, 1.0);
        vec3 sc = sp.xyz / sp.w * 0.5 + 0.5;
        if (sc.x > 0.0 && sc.x < 1.0 && sc.y > 0.0 && sc.y < 1.0 && sc.z < 1.0) {
            float s = 0.0;
            vec2 texel = 1.0 / vec2(textureSize(u_shadow_map, 0));
            for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++)
                s += texture(u_shadow_map, vec3(sc.xy + vec2(i, j) * texel, sc.z - 0.0008));
            vis = s / 9.0;
        }
    }
    vec3 hemi = mix(u_ground, u_sky, 0.5 * n.y + 0.5);
    vec3 c = v_col * (2.2 * u_ambient * hemi + vec3(2.4 * u_sun * ndl * vis)) * 0.3183099 + v_emi;
    c = to_srgb(aces(c));
    if (u_fog_mode == 1) c = mix(c, to_srgb(u_fog_col), smoothstep(u_fog.x, u_fog.y, v_depth));
    else if (u_fog_mode == 2) { float d = u_fog.z * v_depth; c = mix(c, to_srgb(u_fog_col), 1.0 - exp(-d * d)); }
    f_color = vec4(c, 1.0);
    gl_FragDepth = log_depth();
}
"""

FS_DATA = """
#version 330
""" + FS_COMMON + """
layout(location = 0) out vec4 f_data;     // depth, id, -, -
layout(location = 1) out vec4 f_normal;
void main() {
    f_data = vec4(v_depth, v_id, 0.0, 1.0);
    f_normal = vec4(flat_normal(), 1.0);
    gl_FragDepth = log_depth();
}
"""

VS_SHADOW = """
#version 330
uniform mat4 u_mvp;
in vec3 in_pos;
void main() { gl_Position = u_mvp * vec4(in_pos, 1.0); }
"""
FS_SHADOW = """
#version 330
void main() { }
"""


def perspective(fov_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fov_deg) / 2)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def ortho(l, r, b, t, n, f):
    m = np.eye(4)
    m[0, 0], m[1, 1], m[2, 2] = 2 / (r - l), 2 / (t - b), -2 / (f - n)
    m[0, 3], m[1, 3], m[2, 3] = -(r + l) / (r - l), -(t + b) / (t - b), -(f + n) / (f - n)
    return m


def view_matrix(pos, q):
    R = quat_to_mat(np.asarray(q, float))
    m = np.eye(4)
    m[:3, :3] = R.T
    m[:3, 3] = -R.T @ np.asarray(pos, float)
    return m


def intrinsics(fov_deg, w, h):
    fy = (h / 2) / np.tan(np.radians(fov_deg) / 2)
    return np.array([[fy, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], np.float32)


def _gl(m):
    return np.ascontiguousarray(m.T, dtype="f4").tobytes()


@dataclass
class DomainRandomization:
    """Image-space and render-time randomisation (all off by default)."""
    color_jitter: float = 0.0        # 0..1 strength of gain / white balance / gamma / saturation changes
    noise: float = 0.0               # sensor noise sigma (in 0..1 units at mid grey)
    vignette: float = 0.0            # 0..1
    motion_blur: int = 1             # sub-frames per frame (1 = off)
    shutter: float = 0.5             # fraction of the frame interval the shutter is open
    shadows: bool = True             # the viewer renders sun shadows
    crf: int = 20                    # H.264 constant rate factor (higher = stronger compression)
    seed: int = 0
    _params: dict = field(default_factory=dict)

    @staticmethod
    def random(rng, strength=1.0):
        return DomainRandomization(color_jitter=float(rng.uniform(0, 0.6) * strength),
                                   noise=float(rng.choice([0, 0.005, 0.01, 0.02]) * strength),
                                   vignette=float(rng.uniform(0, 0.4) * strength),
                                   motion_blur=int(rng.choice([1, 1, 3, 5])) if strength > 0 else 1,
                                   shadows=bool(rng.random() < 1 - 0.3 * strength),
                                   crf=int(rng.integers(18, 30)), seed=int(rng.integers(1 << 30)))

    def setup(self, w, h):
        rng = np.random.default_rng(self.seed)
        j = self.color_jitter
        self._params = {"gain": 1 + rng.uniform(-0.25, 0.25) * j, "wb": 1 + rng.uniform(-0.12, 0.12, 3) * j,
                        "gamma": 1 + rng.uniform(-0.2, 0.2) * j, "sat": 1 + rng.uniform(-0.35, 0.3) * j,
                        "contrast": 1 + rng.uniform(-0.2, 0.25) * j, "rng": rng}
        yy, xx = np.mgrid[0:h, 0:w]
        r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        self._params["vig"] = (1 - self.vignette * 0.35 * r2)[..., None].astype(np.float32)

    def apply(self, rgb_u8):
        if not (self.color_jitter or self.noise or self.vignette):
            return rgb_u8
        p = self._params
        x = rgb_u8.astype(np.float32) / 255
        if self.color_jitter:
            x = np.clip(x * p["gain"] * p["wb"], 0, 1) ** p["gamma"]
            g = x.mean(-1, keepdims=True)
            x = g + (x - g) * p["sat"]
            x = 0.5 + (x - 0.5) * p["contrast"]
        if self.vignette:
            x = x * p["vig"]
        if self.noise:
            rng = p["rng"]
            sigma = self.noise * np.sqrt(np.clip(x, 0.02, 1) / 0.5)
            x = x + rng.normal(0, 1, x.shape).astype(np.float32) * sigma
        return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


class SceneRenderer:
    """Renders a scene document (as returned by restage.scene.load or the generator)."""

    def __init__(self, doc, width=640, height=360, aux_size=None, msaa=4, dr: DomainRandomization | None = None,
                 near=0.1, far=None, ctx=None):
        import moderngl
        self.doc = doc
        self.W, self.H = int(width), int(height)
        self.aux = (int(aux_size[0]), int(aux_size[1])) if aux_size else (self.W, self.H)
        self.dr = dr or DomainRandomization()
        self.dr.setup(self.W, self.H)
        self.ctx = ctx or moderngl.create_standalone_context(require=330)
        ctx = self.ctx
        self.prog_c = ctx.program(vertex_shader=VS, fragment_shader=FS_COLOR)
        self.prog_d = ctx.program(vertex_shader=VS, fragment_shader=FS_DATA)
        self.prog_s = ctx.program(vertex_shader=VS_SHADOW, fragment_shader=FS_SHADOW)
        self.near = near
        self.fps = doc["fps"]
        self.T = doc["n_frames"]
        b = np.array([l["bounds"] for l in doc["locations"]], float)
        diag = np.linalg.norm(b[:, 1] - b[:, 0], axis=1).max()
        self.far = far or float(max(200.0, diag * 3))
        samples = msaa if msaa and msaa > 1 else 0
        self.fbo_ms = ctx.framebuffer(color_attachments=[ctx.renderbuffer((self.W, self.H), 4, samples=samples)],
                                      depth_attachment=ctx.depth_renderbuffer((self.W, self.H), samples=samples))
        self.fbo_c = ctx.framebuffer(color_attachments=[ctx.renderbuffer((self.W, self.H), 4)])
        aw, ah = self.aux
        self.fbo_d = ctx.framebuffer(color_attachments=[ctx.texture((aw, ah), 4, dtype="f4"),
                                                        ctx.texture((aw, ah), 4, dtype="f2")],
                                     depth_attachment=ctx.depth_renderbuffer((aw, ah)))
        self.shadow_size = 2048
        self.shadow_tex = ctx.depth_texture((self.shadow_size, self.shadow_size))
        self.shadow_tex.compare_func = "<="
        self.shadow_tex.repeat_x = self.shadow_tex.repeat_y = False
        self.fbo_s = ctx.framebuffer(depth_attachment=self.shadow_tex)
        self._mat = {m["id"]: m for m in doc["materials"]}
        self._mesh = {m["id"]: m for m in doc["meshes"]}
        self._skel = {s["id"]: s for s in doc.get("skeletons", [])}
        self._loc_vao = {}
        self._build_locations()
        self._build_entities()
        self._shot_of = np.zeros(self.T, int)
        for i, s in enumerate(doc["shots"]):
            self._shot_of[s["start"]:s["end"]] = i
        self._cams = {c["shot"]: c for c in doc["cameras"]}
        self._shadow_cache = {}

    # ------------------------------------------------------------ geometry upload
    def _mesh_arrays(self, mesh):
        mat = self._mat.get(mesh.get("material", ""), {"color": [1, 1, 1], "emissive": [0, 0, 0]})
        p = np.asarray(mesh["positions"], np.float32)
        col = np.asarray(mesh["colors"], np.float32)[:, :3] if "colors" in mesh else np.ones_like(p)
        if col.dtype != np.float32 or col.max() > 1.5:
            col = col.astype(np.float32) / 255
        col = col * np.asarray(mat.get("color", [1, 1, 1]), np.float32)
        emi = np.broadcast_to(np.asarray(mat.get("emissive", [0, 0, 0]), np.float32), p.shape)
        idx = np.asarray(mesh["indices"]).astype(np.uint32).reshape(-1)
        return p, col, emi, idx

    def _vao(self, pos, col, emi, ids, dynamic=False):
        ctx = self.ctx
        n = len(pos)
        attr = np.concatenate([col, emi, ids[:, None]], 1).astype("f4")
        vbo_p = ctx.buffer(np.ascontiguousarray(pos, dtype="f4").tobytes(), dynamic=dynamic)
        vbo_a = ctx.buffer(attr.tobytes())
        content = [(vbo_p, "3f", "in_pos"), (vbo_a, "3f 3f 1f", "in_col", "in_emi", "in_id")]
        vc = ctx.vertex_array(self.prog_c, content, skip_errors=True)
        vd = ctx.vertex_array(self.prog_d, content, skip_errors=True)
        vs = ctx.vertex_array(self.prog_s, [(vbo_p, "3f", "in_pos")], skip_errors=True)
        return {"vbo": vbo_p, "c": vc, "d": vd, "s": vs, "n": n}

    def _build_locations(self):
        for loc in self.doc["locations"]:
            groups = {True: ([], [], [], []), False: ([], [], [], [])}   # shadow casters / non-casters
            for k, pr in enumerate(loc.get("props", [])):
                mesh = self._mesh[pr["mesh"]]
                p, c, e, idx = self._mesh_arrays(mesh)
                q = np.asarray(pr.get("rotation", [0, 0, 0, 1]), float)
                sc = np.asarray(pr.get("scale", [1, 1, 1]), float)
                pw = quat_rotate(np.broadcast_to(q, (len(p), 4)), p * sc) + np.asarray(pr.get("position", [0, 0, 0]), float)
                if np.prod(np.sign(sc)) < 0:
                    idx = idx.reshape(-1, 3)[:, [0, 2, 1]].reshape(-1)
                pid = 0 if pr.get("class", "") in STRUCTURAL else 1000 + k
                P, C, E, I = groups[bool(pr.get("cast_shadow", True))]
                P.append(pw[idx])
                C.append(c[idx])
                E.append(e[idx])
                I.append(np.full(len(idx), pid, np.float32))
            vaos = []
            for cast, (P, C, E, I) in groups.items():
                if P:
                    v = self._vao(*(np.concatenate(x) for x in (P, C, E, I)))
                    v["cast"] = cast
                    vaos.append(v)
            self._loc_vao[loc["id"]] = vaos

    def _build_entities(self):
        self.ents = []
        for i, e in enumerate(self.doc["entities"]):
            mesh = self._mesh[e["mesh"]]
            p, c, emi, idx = self._mesh_arrays(mesh)
            tr = e["track"]
            ent = {"id": i + 1, "loc": e["location"], "idx": idx, "rest": p.astype(np.float64),
                   "state": np.asarray(tr["state"]), "track": tr}
            ent["morphs"] = [np.asarray(m["positions"], np.float64) for m in mesh.get("morphs", [])]
            sk = self._skel.get(e.get("skeleton")) if e.get("skeleton") else None
            if sk is not None and "pose" in tr:
                off = np.array([j["offset"] for j in sk["joints"]], float)
                par = np.array([j["parent"] for j in sk["joints"]])
                pose = np.asarray(tr["pose"], float)
                sc = np.asarray(tr["scale"], float) if "scale" in tr else None
                jp, jq = forward_kinematics(pose, np.asarray(tr["position"], float), np.asarray(tr["rotation"], float),
                                            off, par, sc)
                rest = np.zeros((len(off), 3))
                for j in range(len(off)):
                    rest[j] = (rest[par[j]] if par[j] >= 0 else 0) + off[j]
                ent["jp"], ent["jR"], ent["rest_j"] = jp, quat_to_mat(jq), rest
                ent["joints"] = np.asarray(mesh["joints"]).astype(int)
                ent["weights"] = np.asarray(mesh["weights"], np.float64)
            ent["vao"] = self._vao(np.zeros((len(idx), 3), np.float32), c[idx], emi[idx],
                                   np.full(len(idx), i + 1, np.float32), dynamic=True)
            self.ents.append(ent)

    def _skin(self, ent, t, frac=0.0):
        tr = ent["track"]
        v = ent["rest"]
        if ent["morphs"] and "morph" in tr:
            w = np.asarray(tr["morph"], float)[t]
            for k, d in enumerate(ent["morphs"]):
                if k < len(w) and w[k] != 0:
                    v = v + d * w[k]
        if "jp" in ent:
            J, W = ent["joints"], ent["weights"]
            out = np.zeros_like(v)
            for k in range(4):
                wk = W[:, k]
                if not wk.any():
                    continue
                jj = J[:, k]
                R = ent["jR"][t][jj]
                local = v - ent["rest_j"][jj]
                out += wk[:, None] * (np.einsum("nij,nj->ni", R, local) + ent["jp"][t][jj])
            v = out
        else:
            q = np.asarray(tr["rotation"], float)[t]
            sc = np.asarray(tr["scale"], float)[t] if "scale" in tr else 1.0
            v = quat_rotate(np.broadcast_to(q, (len(v), 4)), v * sc) + np.asarray(tr["position"], float)[t]
        return v[ent["idx"]]

    # ------------------------------------------------------------ frame rendering
    def _camera(self, t, frac=0.0):
        s = self.doc["shots"][self._shot_of[t]]
        cam = self._cams[s["id"]]
        i = t - s["start"]
        tr = cam["track"]
        p, q, f = (np.asarray(tr[k], float) for k in ("position", "rotation", "fov"))
        p0, q0, f0 = p[i], q[i], float(f[i])
        if frac > 0 and i + 1 < len(p):
            q1 = q[i + 1] * (1 if np.dot(q[i + 1], q0) >= 0 else -1)
            p0 = p0 * (1 - frac) + p[i + 1] * frac
            q0 = q0 * (1 - frac) + q1 * frac
            q0 /= np.linalg.norm(q0)
            f0 = f0 * (1 - frac) + float(f[i + 1]) * frac
        return s, p0, q0, f0

    def _env(self, loc):
        env = loc.get("environment", {})
        sky = np.asarray(env.get("sky", [0.6, 0.7, 0.9]), float)
        fog = env.get("fog")
        return env, sky, fog

    def _set_uniforms(self, prog, loc, cam_p, view, proj):
        env, sky, fog = self._env(loc)
        sd = np.asarray(env.get("sun_dir", [0.3, -1, 0.2]), float)
        sd = sd / np.linalg.norm(sd)
        vals = {"u_view": _gl(view), "u_proj": _gl(proj)}
        for k, v in vals.items():
            prog[k].write(v)
        for name, val in (("u_cam", tuple(map(float, cam_p))), ("u_sun_dir", tuple(map(float, sd))),
                          ("u_sun", float(env.get("sun", 1.0))), ("u_ambient", float(env.get("ambient", 0.35))),
                          ("u_sky", tuple(map(float, sky))), ("u_logc", float(1.0 / np.log2(self.far + 1.0))),
                          ("u_ground", tuple(map(float, env.get("ground", [0.35, 0.3, 0.25]))))):
            if name in prog:
                prog[name].value = val
        if "u_fog_mode" in prog:
            mode, fc, fv = 0, sky, (0.0, 1.0, 0.0)
            if fog:
                fc = np.asarray(fog.get("color", sky), float)
                if "density" in fog:
                    mode, fv = 2, (0.0, 1.0, float(fog["density"]))
                else:
                    mode, fv = 1, (float(fog.get("near", 10)), float(fog.get("far", 100)), 0.0)
            prog["u_fog_mode"].value = mode
            prog["u_fog_col"].value = tuple(map(float, fc))
            prog["u_fog"].value = fv

    def _shadow_pass(self, loc, objs, cam_p):
        env, _, _ = self._env(loc)
        sd = np.asarray(env.get("sun_dir", [0.3, -1, 0.2]), float)
        sd = sd / np.linalg.norm(sd)
        lo, hi = np.asarray(loc["bounds"][0], float), np.asarray(loc["bounds"][1], float)
        # fit the shadow frustum around the camera neighbourhood (outdoor) or the room (indoor)
        half = min(np.linalg.norm(hi - lo) / 2, 45.0)
        center = np.clip(cam_p, lo, hi) if np.linalg.norm(hi - lo) > 90 else (lo + hi) / 2
        eye = center - sd * 200
        up = np.array([0, 1.0, 0]) if abs(sd[1]) < 0.99 else np.array([1.0, 0, 0])
        z = -sd
        x = np.cross(up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z], 1)
        view = np.eye(4)
        view[:3, :3] = R.T
        view[:3, 3] = -R.T @ eye
        proj = ortho(-half, half, -half, half, 1.0, 400.0)
        mvp = proj @ view
        self.fbo_s.use()
        self.fbo_s.clear(depth=1.0)
        self.ctx.enable(self.ctx.DEPTH_TEST)
        self.ctx.disable(self.ctx.CULL_FACE)
        self.prog_s["u_mvp"].write(_gl(mvp))
        for vao in objs:
            if vao.get("cast", True):
                vao["s"].render(vertices=vao["n"])
        return mvp

    def render(self, t, outputs=("rgb", "depth", "ids")):
        """Render video frame t. Returns dict with requested outputs."""
        import moderngl
        ctx = self.ctx
        shot, cam_p, cam_q, fov = self._camera(t)
        loc = next(l for l in self.doc["locations"] if l["id"] == shot["location"])
        env, sky, _ = self._env(loc)
        view = view_matrix(cam_p, cam_q)
        out = {}
        sub = max(1, self.dr.motion_blur) if "rgb" in outputs else 1
        acc = None
        for k in range(sub):
            frac = (k / max(sub - 1, 1) - 0.5) * self.dr.shutter if sub > 1 else 0.0
            tt, fr = t, frac
            if frac < 0:
                if t - 1 >= shot["start"]:
                    tt, fr = t - 1, 1 + frac
                else:
                    fr = 0.0
            objs = self._frame_objects(loc, tt, fr)
            _, cp, cq, fv = self._camera(tt, fr)
            v = view_matrix(cp, cq)
            proj = perspective(fv, self.W / self.H, self.near, self.far)
            prog = self.prog_c
            self._set_uniforms(prog, loc, cp, v, proj)
            if self.dr.shadows:
                mvp = self._shadow_pass(loc, objs, cp)
                prog["u_shadows"].value = 1
                prog["u_shadow_mat"].write(_gl(mvp))
                self.shadow_tex.use(location=0)
                prog["u_shadow_map"].value = 0
            else:
                prog["u_shadows"].value = 0
            self.fbo_ms.use()
            sky_srgb = np.clip(np.where(sky <= 0.0031308, sky * 12.92, 1.055 * np.clip(sky, 0, 1) ** (1 / 2.4) - 0.055), 0, 1)
            self.fbo_ms.clear(*sky_srgb, 1.0, depth=1.0)
            ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
            ctx.front_face = "ccw"
            for vao in objs:
                vao["c"].render(vertices=vao["n"])
            ctx.copy_framebuffer(self.fbo_c, self.fbo_ms)
            img = np.frombuffer(self.fbo_c.read(components=3), np.uint8).reshape(self.H, self.W, 3)[::-1]
            acc = img.astype(np.float32) if acc is None else acc + img
        if "rgb" in outputs:
            rgb = (acc / sub + 0.5).astype(np.uint8) if sub > 1 else img.copy()
            out["rgb"] = self.dr.apply(rgb)
        if "depth" in outputs or "ids" in outputs or "normals" in outputs:
            objs = self._frame_objects(loc, t, 0.0)
            proj = perspective(fov, self.W / self.H, self.near, self.far)
            self._set_uniforms(self.prog_d, loc, cam_p, view, proj)
            self.fbo_d.use()
            self.fbo_d.clear(0, 0, 0, 0, depth=1.0)
            ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
            for vao in objs:
                vao["d"].render(vertices=vao["n"])
            aw, ah = self.aux
            data = np.frombuffer(self.fbo_d.read(attachment=0, components=4, dtype="f4"), np.float32).reshape(ah, aw, 4)[::-1]
            if "depth" in outputs:
                out["depth"] = data[..., 0].copy()
            if "ids" in outputs:
                out["ids"] = np.round(data[..., 1]).astype(np.uint16)
            if "normals" in outputs:
                nrm = np.frombuffer(self.fbo_d.read(attachment=1, components=4, dtype="f2"), np.float16).reshape(ah, aw, 4)[::-1]
                out["normals"] = nrm[..., :3].astype(np.float32)
        out["K"] = intrinsics(fov, *self.aux)
        out["K_video"] = intrinsics(fov, self.W, self.H)
        R = quat_to_mat(np.asarray(cam_q, float))
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = R, cam_p
        out["cam_to_world"] = c2w.astype(np.float32)
        return out

    def _frame_objects(self, loc, t, frac):
        objs = list(self._loc_vao.get(loc["id"], []))
        for ent in self.ents:
            if ent["loc"] != loc["id"] or ent["state"][t] == 0:
                continue
            v = self._skin(ent, t)
            if frac > 0 and t + 1 < self.T and ent["state"][t + 1] > 0:
                v = v * (1 - frac) + self._skin(ent, t + 1) * frac
            ent["vao"]["vbo"].write(np.ascontiguousarray(v, dtype="f4").tobytes())
            objs.append(ent["vao"])
        return objs

    def release(self):
        self.ctx.release()


def render_video(doc, mp4_path, aux_path=None, width=640, height=360, aux_size=(320, 180), dr=None,
                 frames=None, progress=False, normals=False):
    """Render a whole scene to an H.264 mp4 (+ optional aux npz with depth, ids, intrinsics)."""
    dr = dr or DomainRandomization()
    r = SceneRenderer(doc, width, height, aux_size, dr=dr)
    T = doc["n_frames"]
    frames = range(T) if frames is None else frames
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
           "-r", str(doc["fps"]), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(dr.crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    aw, ah = aux_size
    n = len(frames)
    want = ("rgb", "depth", "ids") + (("normals",) if normals else ())
    depth = np.zeros((n, ah, aw), np.float16) if aux_path else None
    ids = np.zeros((n, ah, aw), np.uint16) if aux_path else None
    nrm = np.zeros((n, ah, aw, 3), np.float16) if aux_path and normals else None
    K = np.zeros((n, 3, 3), np.float32)
    Kv = np.zeros((n, 3, 3), np.float32)
    c2w = np.zeros((n, 4, 4), np.float32)
    t0 = time.time()
    for i, t in enumerate(frames):
        o = r.render(t, want if aux_path else ("rgb",))
        proc.stdin.write(o["rgb"].tobytes())
        if aux_path:
            depth[i] = o["depth"]
            ids[i] = o["ids"]
            if normals:
                nrm[i] = o["normals"]
        K[i], Kv[i], c2w[i] = o["K"], o["K_video"], o["cam_to_world"]
        if progress and i % 60 == 0:
            print(f"  frame {i}/{n}  {i / max(time.time() - t0, 1e-6):.1f} fps", flush=True)
    proc.stdin.close()
    proc.wait()
    elapsed = time.time() - t0
    if aux_path:
        shot_idx = np.array([r._shot_of[t] for t in frames], np.int16)
        extra = {"normals": nrm} if normals else {}
        id_dtype = ids if ids.max() >= 256 else ids.astype(np.uint8)
        np.savez_compressed(aux_path, depth=depth, ids=id_dtype, K=K, K_video=Kv, cam_to_world=c2w, shot=shot_idx,
                            frames=np.asarray(list(frames), np.int32), fps=np.float32(doc["fps"]),
                            video_size=np.array([width, height], np.int32), **extra)
    r.release()
    return {"frames": n, "seconds": elapsed, "fps": n / max(elapsed, 1e-9)}


def render_png(doc, t, path, width=640, height=360, dr=None):
    from PIL import Image
    r = SceneRenderer(doc, width, height, dr=dr)
    o = r.render(t, ("rgb",))
    Image.fromarray(o["rgb"]).save(path)
    r.release()
    return o
