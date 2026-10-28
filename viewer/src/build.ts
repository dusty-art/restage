/** Builds three.js objects from a scene document and poses them for any frame. */
import * as THREE from "three";
import type { Arr, CameraDoc, EntityDoc, LocationDoc, MeshDoc, SceneDoc, ShotDoc } from "./format";

export const LOCATION_COLORS = ["#f59e0b", "#38bdf8", "#a78bfa", "#34d399", "#f472b6", "#fb7185", "#facc15", "#60a5fa"];
export const ENTITY_COLORS = ["#ef4444", "#3b82f6", "#22c55e", "#eab308", "#a855f7", "#06b6d4", "#f97316", "#ec4899",
  "#84cc16", "#14b8a6", "#6366f1", "#f43f5e"];

export type Style = "shaded" | "clay" | "wire" | "ids" | "confidence" | "normals";

export interface LocationView {
  doc: LocationDoc;
  index: number;
  color: string;
  group: THREE.Group;
  props: THREE.Group;
  helpers: THREE.Group;          // camera markers, trails, grid: hidden in replica mode
  bounds: THREE.Box3;
  sky: THREE.Color;
  ambient: number;
  sun: THREE.DirectionalLight;
  hemi: THREE.HemisphereLight;
  fog: THREE.Fog | null;
  grid: THREE.GridHelper;
}

export interface EntityView {
  doc: EntityDoc;
  index: number;
  color: THREE.Color;
  root: THREE.Group;
  mesh: THREE.Mesh | THREE.SkinnedMesh;
  bones: THREE.Bone[];
  trail: THREE.Line;
  location: LocationView | null;
  baseMaterial: THREE.Material;
  seenFrames: number;
  inferredFrames: number;
  tris: number;
}

export interface CameraView {
  doc: CameraDoc;
  shot: ShotDoc;
  shotIndex: number;
  camera: THREE.PerspectiveCamera;
  marker: THREE.Group;
  path: THREE.Line;
  location: LocationView;
}

export interface Built {
  doc: SceneDoc;
  aspect: number;
  locations: LocationView[];
  locById: Map<string, LocationView>;
  entities: EntityView[];
  cameras: CameraView[];
  camByShot: Map<string, CameraView>;
  shotAt: Int16Array;             // frame -> shot index (-1 = none)
  meshTris: number;
  dispose(): void;
}

const tmpQ = new THREE.Quaternion();

function attr(a: Arr, itemSize: number, normalized = false): THREE.BufferAttribute {
  return new THREE.BufferAttribute(a.data as THREE.TypedArray, itemSize, normalized);
}

function geometryFor(m: MeshDoc): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", attr(m.positions, 3));
  const idx = m.indices.data;
  g.setIndex(new THREE.BufferAttribute(idx instanceof Uint16Array || idx instanceof Uint32Array ? idx : new Uint32Array(idx), 1));
  if (m.normals) g.setAttribute("normal", attr(m.normals, 3));
  if (m.colors) {
    const k = m.colors.shape[1];
    g.setAttribute("color", attr(m.colors, k, m.colors.data instanceof Uint8Array));
  } else {
    const n = m.positions.shape[0];
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * 3).fill(1), 3));
  }
  if (m.uvs) g.setAttribute("uv", attr(m.uvs, 2));
  if (m.joints && m.weights) {
    g.setAttribute("skinIndex", new THREE.BufferAttribute(new Uint16Array(m.joints.data), 4));
    g.setAttribute("skinWeight", attr(m.weights, 4));
  }
  if (m.morphs?.length) {
    g.morphAttributes.position = m.morphs.map((mt) => attr(mt.positions, 3));
    g.morphTargetsRelative = true;
  }
  // Confidence channel used by the "confidence" style: observed vertices green, inferred orange.
  const n = m.positions.shape[0];
  const obs = new Float32Array(n).fill(1);
  if (m.observed) for (let i = 0; i < n; i++) obs[i] = m.observed.data[i] ? 1 : 0;
  g.setAttribute("observed", new THREE.BufferAttribute(obs, 1));
  if (!m.normals) g.computeVertexNormals();
  g.computeBoundingSphere();
  g.computeBoundingBox();
  return g;
}

function makeMaterial(doc: SceneDoc, id: string | undefined): THREE.MeshStandardMaterial {
  const md = doc.materials?.find((x) => x.id === id);
  const mat = new THREE.MeshStandardMaterial({
    color: md ? new THREE.Color().setRGB(md.color[0], md.color[1], md.color[2], THREE.LinearSRGBColorSpace) : 0xffffff,
    roughness: md?.roughness ?? 0.85,
    metalness: md?.metalness ?? 0.0,
    vertexColors: true,
    flatShading: md?.flat ?? true,
  });
  if (md?.emissive) mat.emissive.setRGB(md.emissive[0], md.emissive[1], md.emissive[2], THREE.LinearSRGBColorSpace);
  return mat;
}

function lineFrom(points: THREE.Vector3[], color: string, opacity = 1, dashed = false): THREE.Line {
  const g = new THREE.BufferGeometry().setFromPoints(points.length ? points : [new THREE.Vector3()]);
  const mat = dashed
    ? new THREE.LineDashedMaterial({ color, dashSize: 0.15, gapSize: 0.1, transparent: opacity < 1, opacity })
    : new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity });
  const l = new THREE.Line(g, mat);
  if (dashed) l.computeLineDistances();
  return l;
}

/** A small camera body + view cone, drawn in the shot's location colour. */
function cameraMarker(color: string, fov: number, aspect: number): THREE.Group {
  const g = new THREE.Group();
  const d = 0.45;
  const h = Math.tan((fov * Math.PI) / 360) * d;
  const w = h * aspect;
  const p = [new THREE.Vector3(0, 0, 0), new THREE.Vector3(-w, -h, -d), new THREE.Vector3(w, -h, -d),
    new THREE.Vector3(w, h, -d), new THREE.Vector3(-w, h, -d)];
  const seg = [0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 2, 3, 3, 4, 4, 1];
  const pts = seg.map((i) => p[i]);
  // "up" triangle above the frame, so camera roll is readable
  pts.push(new THREE.Vector3(-w * 0.35, h * 1.08, -d), new THREE.Vector3(0, h * 1.45, -d),
    new THREE.Vector3(0, h * 1.45, -d), new THREE.Vector3(w * 0.35, h * 1.08, -d));
  const lines = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 }));
  g.add(lines);
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.09, 0.16), new THREE.MeshBasicMaterial({ color }));
  body.position.z = 0.08;
  g.add(body);
  g.userData.lines = lines;
  g.userData.body = body;
  return g;
}

/** A screen-size label (shot number) drawn as a sprite, readable from any distance. */
function labelSprite(text: string, color: string): THREE.Sprite {
  const c = document.createElement("canvas");
  c.width = 64;
  c.height = 40;
  const g = c.getContext("2d")!;
  g.fillStyle = color;
  g.beginPath();
  g.roundRect(2, 2, 60, 36, 9);
  g.fill();
  g.fillStyle = "#0b0d12";
  g.font = "bold 24px Inter, system-ui, sans-serif";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText(text, 32, 21);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, sizeAttenuation: false, depthTest: false, transparent: true }));
  s.scale.set(0.045, 0.028, 1);
  s.center.set(0.5, -0.4);
  s.renderOrder = 10;
  return s;
}

function readQuat(a: Arr, i: number, out: THREE.Quaternion): THREE.Quaternion {
  const d = a.data;
  return out.set(d[i * 4], d[i * 4 + 1], d[i * 4 + 2], d[i * 4 + 3]);
}

export function buildScene(doc: SceneDoc): Built {
  const src = doc.meta?.source;
  const aspect = src?.width && src?.height ? src.width / src.height : 16 / 9;
  const geoms = new Map<string, THREE.BufferGeometry>();
  const geo = (id: string) => {
    let g = geoms.get(id);
    if (!g) {
      const m = doc.meshes.find((x) => x.id === id);
      if (!m) throw new Error(`missing mesh ${id}`);
      g = geometryFor(m);
      geoms.set(id, g);
    }
    return g;
  };
  const meshDoc = (id: string) => doc.meshes.find((x) => x.id === id)!;
  let meshTris = 0;

  // ---- locations
  const locations: LocationView[] = doc.locations.map((l, index) => {
    const group = new THREE.Group();
    group.name = l.name;
    const props = new THREE.Group();
    const helpers = new THREE.Group();
    group.add(props, helpers);
    for (const p of l.props) {
      const g = geo(p.mesh);
      const mesh = new THREE.Mesh(g, makeMaterial(doc, meshDoc(p.mesh).material));
      mesh.name = p.id;
      mesh.userData = { kind: "prop", id: p.id, cls: p.class, baseMaterial: mesh.material };
      if (p.position) mesh.position.fromArray(p.position);
      if (p.rotation) mesh.quaternion.fromArray(p.rotation);
      if (p.scale) mesh.scale.fromArray(p.scale);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      props.add(mesh);
      meshTris += (g.index?.count ?? 0) / 3;
    }
    const bounds = new THREE.Box3();
    if (l.bounds) bounds.set(new THREE.Vector3(...l.bounds[0]), new THREE.Vector3(...l.bounds[1]));
    else bounds.setFromObject(props);
    if (bounds.isEmpty()) bounds.set(new THREE.Vector3(-5, 0, -5), new THREE.Vector3(5, 3, 5));
    const env = l.environment ?? {};
    const sky = new THREE.Color().setRGB(...(env.sky ?? [0.55, 0.65, 0.8]), THREE.LinearSRGBColorSpace);
    const ground = new THREE.Color().setRGB(...(env.ground ?? [0.35, 0.3, 0.25]), THREE.LinearSRGBColorSpace);
    const ambient = env.ambient ?? 0.4;
    const hemi = new THREE.HemisphereLight(sky, ground, ambient * 2.2);
    const sun = new THREE.DirectionalLight(0xffffff, (env.sun ?? 1.0) * 2.4);
    const dir = new THREE.Vector3(...(env.sun_dir ?? [0.3, -1, 0.25])).normalize();
    const size = bounds.getSize(new THREE.Vector3());
    const center = bounds.getCenter(new THREE.Vector3());
    const reach = Math.max(size.x, size.z, 6) * 0.75;
    sun.position.copy(center).addScaledVector(dir, -reach * 2);
    sun.target.position.copy(center);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const sc = sun.shadow.camera as THREE.OrthographicCamera;
    sc.left = sc.bottom = -reach;
    sc.right = sc.top = reach;
    sc.near = 0.1;
    sc.far = reach * 5;
    sun.shadow.bias = -0.0005;
    sun.shadow.normalBias = 0.02;
    group.add(hemi, sun, sun.target);
    const fog = env.fog ? new THREE.Fog(
      env.fog.color ? new THREE.Color().setRGB(...env.fog.color, THREE.LinearSRGBColorSpace) : sky,
      env.fog.near ?? 20, env.fog.far ?? 120) : null;
    const gridSize = Math.ceil(Math.max(size.x, size.z, 4) / 2) * 2 + 4;
    const grid = new THREE.GridHelper(gridSize, gridSize, 0x666666, 0x333333);
    grid.position.set(center.x, bounds.min.y + 0.002, center.z);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.35;
    helpers.add(grid);
    return { doc: l, index, color: LOCATION_COLORS[index % LOCATION_COLORS.length], group, props, helpers,
      bounds, sky, ambient, sun, hemi, fog, grid };
  });
  const locById = new Map(locations.map((l) => [l.doc.id, l]));

  // ---- entities
  const entities: EntityView[] = doc.entities.map((e, index) => {
    const loc = e.location ? locById.get(e.location) ?? null : null;
    const root = new THREE.Group();
    root.name = e.name ?? e.id;
    const g = geo(e.mesh);
    const material = makeMaterial(doc, meshDoc(e.mesh).material);
    let mesh: THREE.Mesh | THREE.SkinnedMesh;
    const bones: THREE.Bone[] = [];
    const sk = e.skeleton ? doc.skeletons?.find((s) => s.id === e.skeleton) : undefined;
    if (sk && g.getAttribute("skinIndex")) {
      const sm = new THREE.SkinnedMesh(g, material);
      for (const j of sk.joints) {
        const b = new THREE.Bone();
        b.name = j.name;
        b.position.fromArray(j.offset);
        bones.push(b);
      }
      sk.joints.forEach((j, i) => {
        if (j.parent >= 0) bones[j.parent].add(bones[i]);
        else sm.add(bones[i]);
      });
      sm.updateMatrixWorld(true);
      sm.bind(new THREE.Skeleton(bones));
      mesh = sm;
    } else {
      mesh = new THREE.Mesh(g, material);
    }
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.frustumCulled = false;
    mesh.userData = { kind: "entity", index, baseMaterial: material };
    root.add(mesh);
    const color = new THREE.Color(ENTITY_COLORS[index % ENTITY_COLORS.length]);
    // trail of the root over the frames where the entity exists
    const tr = e.track;
    const pts: THREE.Vector3[] = [];
    let seen = 0;
    let inferred = 0;
    for (let t = 0; t < doc.n_frames; t++) {
      const s = tr.state ? tr.state.data[t] : 2;
      if (s === 2) seen++;
      else if (s === 1) inferred++;
      if (s > 0 && t % 2 === 0) pts.push(new THREE.Vector3(tr.position.data[t * 3], tr.position.data[t * 3 + 1] + 0.02, tr.position.data[t * 3 + 2]));
    }
    const trail = lineFrom(pts, `#${color.getHexString()}`, 0.8);
    if (loc) {
      loc.group.add(root);
      loc.helpers.add(trail);
    }
    meshTris += (g.index?.count ?? 0) / 3;
    return { doc: e, index, color, root, mesh, bones, trail, location: loc, baseMaterial: material,
      seenFrames: seen, inferredFrames: inferred, tris: (g.index?.count ?? 0) / 3 };
  });

  // ---- cameras
  const shotIndex = new Map(doc.shots.map((s, i) => [s.id, i]));
  const cameras: CameraView[] = [];
  for (const c of doc.cameras) {
    const shot = doc.shots.find((s) => s.id === c.shot);
    if (!shot) continue;
    const loc = locById.get(shot.location);
    if (!loc) continue;
    const cam = new THREE.PerspectiveCamera(c.track.fov.data[0], aspect, 0.03, 2000);
    const marker = cameraMarker(loc.color, c.track.fov.data[0], aspect);
    const label = labelSprite(String((shotIndex.get(shot.id) ?? 0) + 1), loc.color);
    label.userData.kind = "camera";
    label.userData.shot = shot.id;
    marker.add(label);
    const pts: THREE.Vector3[] = [];
    const n = c.track.position.shape[0];
    for (let i = 0; i < n; i++) pts.push(new THREE.Vector3(c.track.position.data[i * 3], c.track.position.data[i * 3 + 1], c.track.position.data[i * 3 + 2]));
    const path = lineFrom(pts, loc.color, 0.7, true);
    // place the marker at the shot's first frame (moved during playback)
    marker.position.copy(pts[0] ?? new THREE.Vector3());
    marker.quaternion.copy(readQuat(c.track.rotation, 0, tmpQ));
    Object.assign(marker.userData, { kind: "camera", shot: shot.id });
    loc.helpers.add(marker, path);
    cameras.push({ doc: c, shot, shotIndex: shotIndex.get(shot.id)!, camera: cam, marker, path, location: loc });
  }
  const camByShot = new Map(cameras.map((c) => [c.shot.id, c]));
  const shotAt = new Int16Array(doc.n_frames).fill(-1);
  doc.shots.forEach((s, i) => {
    for (let t = Math.max(0, s.start); t < Math.min(doc.n_frames, s.end); t++) shotAt[t] = i;
  });

  return {
    doc, aspect, locations, locById, entities, cameras, camByShot, shotAt, meshTris,
    dispose() {
      geoms.forEach((g) => g.dispose());
      const seen = new Set<THREE.Material>();
      for (const l of locations) l.group.traverse((o) => {
        const m = (o as THREE.Mesh).material;
        if (m && !Array.isArray(m) && !seen.has(m)) { seen.add(m); m.dispose(); }
        if ((o as THREE.Line).isLine) (o as THREE.Line).geometry.dispose();
      });
      for (const l of locations) l.sun.shadow.map?.dispose();
    },
  };
}

/** Pose every entity and camera for frame `t` (fractional frames interpolate). */
export function poseAt(b: Built, t: number): void {
  const n = b.doc.n_frames;
  const t0 = Math.max(0, Math.min(n - 1, Math.floor(t)));
  const t1 = Math.min(n - 1, t0 + 1);
  const a = Math.max(0, Math.min(1, t - t0));
  const q0 = new THREE.Quaternion();
  const q1 = new THREE.Quaternion();
  for (const e of b.entities) {
    const tr = e.doc.track;
    const state = tr.state ? tr.state.data[t0] : 2;
    e.root.userData.state = state;
    const p = tr.position.data;
    e.root.position.set(
      p[t0 * 3] * (1 - a) + p[t1 * 3] * a,
      p[t0 * 3 + 1] * (1 - a) + p[t1 * 3 + 1] * a,
      p[t0 * 3 + 2] * (1 - a) + p[t1 * 3 + 2] * a);
    readQuat(tr.rotation, t0, q0);
    readQuat(tr.rotation, t1, q1);
    e.root.quaternion.slerpQuaternions(q0, q1, a);
    if (tr.scale) {
      const s = tr.scale.data;
      e.root.scale.set(s[t0 * 3], s[t0 * 3 + 1], s[t0 * 3 + 2]);
    }
    if (tr.pose && e.bones.length) {
      const J = tr.pose.shape[1];
      const d = tr.pose.data;
      for (let j = 0; j < Math.min(J, e.bones.length); j++) {
        const i0 = (t0 * J + j) * 4;
        const i1 = (t1 * J + j) * 4;
        q0.set(d[i0], d[i0 + 1], d[i0 + 2], d[i0 + 3]);
        q1.set(d[i1], d[i1 + 1], d[i1 + 2], d[i1 + 3]);
        e.bones[j].quaternion.slerpQuaternions(q0, q1, a);
      }
    }
    if (tr.morph && e.mesh.morphTargetInfluences) {
      const K = tr.morph.shape[1];
      for (let k = 0; k < K; k++) e.mesh.morphTargetInfluences[k] = tr.morph.data[t0 * K + k] * (1 - a) + tr.morph.data[t1 * K + k] * a;
    }
  }
  const shot = b.shotAt[t0];
  for (const c of b.cameras) {
    const s = c.shot;
    const n = c.doc.track.position.shape[0];
    // Cameras hold their first/last pose outside the shot.
    const f = Math.max(0, Math.min(n - 1, t - s.start));
    const i0 = Math.floor(f);
    const i1 = Math.min(n - 1, i0 + 1);
    const w = f - i0;
    const p = c.doc.track.position.data;
    c.camera.position.set(p[i0 * 3] * (1 - w) + p[i1 * 3] * w, p[i0 * 3 + 1] * (1 - w) + p[i1 * 3 + 1] * w,
      p[i0 * 3 + 2] * (1 - w) + p[i1 * 3 + 2] * w);
    readQuat(c.doc.track.rotation, i0, q0);
    readQuat(c.doc.track.rotation, i1, q1);
    c.camera.quaternion.slerpQuaternions(q0, q1, w);
    const fov = c.doc.track.fov.data[i0] * (1 - w) + c.doc.track.fov.data[i1] * w;
    if (Math.abs(c.camera.fov - fov) > 1e-4) {
      c.camera.fov = fov;
      c.camera.updateProjectionMatrix();
    }
    c.camera.updateMatrixWorld();
    c.marker.position.copy(c.camera.position);
    c.marker.quaternion.copy(c.camera.quaternion);
    const active = c.shotIndex === shot;
    c.marker.scale.setScalar(active ? 1.35 : 1);
    ((c.marker.userData.lines as THREE.LineSegments).material as THREE.LineBasicMaterial).opacity = active ? 1 : 0.55;
  }
}

/** Focal length (mm) of a full-frame camera with this vertical field of view. */
export function focalMM(fovDeg: number): number {
  return 12 / Math.tan((fovDeg * Math.PI) / 360);
}
