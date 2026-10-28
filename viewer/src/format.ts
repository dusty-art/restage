/** Reader for `.restage` scene files (docs/SCENE_FORMAT.md). Accessors become typed arrays. */

export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number];

export interface Arr {
  data: Float32Array | Uint32Array | Uint16Array | Uint8Array | Int16Array;
  shape: number[];
}

export interface Material {
  id: string;
  color: Vec3;
  roughness?: number;
  metalness?: number;
  emissive?: Vec3;
  flat?: boolean;
}

export interface MeshDoc {
  id: string;
  positions: Arr;
  indices: Arr;
  normals?: Arr;
  colors?: Arr;
  uvs?: Arr;
  joints?: Arr;
  weights?: Arr;
  morphs?: { name: string; positions: Arr }[];
  material?: string;
  observed?: Arr;
}

export interface SkeletonDoc {
  id: string;
  joints: { name: string; parent: number; offset: Vec3 }[];
}

export interface Environment {
  sky?: Vec3;
  ambient?: number;
  sun_dir?: Vec3;
  sun?: number;
  fog?: { color?: Vec3; near?: number; far?: number } | null;
  ground?: Vec3;
}

export interface Prop {
  id: string;
  class?: string;
  mesh: string;
  position?: Vec3;
  rotation?: Quat;
  scale?: Vec3;
}

export interface LocationDoc {
  id: string;
  name: string;
  kind?: string;
  bounds?: [Vec3, Vec3];
  environment?: Environment;
  props: Prop[];
}

export interface Track {
  state?: Arr;
  position: Arr;
  rotation: Arr;
  scale?: Arr;
  pose?: Arr;
  morph?: Arr;
  confidence?: Arr;
}

export interface EntityDoc {
  id: string;
  name?: string;
  class?: string;
  location?: string;
  mesh: string;
  skeleton?: string;
  track: Track;
}

export interface CameraDoc {
  id: string;
  shot: string;
  track: { position: Arr; rotation: Arr; fov: Arr };
}

export interface ShotDoc {
  id: string;
  start: number;
  end: number;
  location: string;
  camera: string;
  size?: string;
  move?: string;
  transition?: string;
}

export interface ContinuityDoc {
  kind: string;
  entity?: string;
  from_shot?: string;
  to_shot?: string;
  frame?: number;
  detail?: string;
  severity?: number;
}

export interface SceneDoc {
  format: string;
  version: number;
  fps: number;
  n_frames: number;
  meta?: { title?: string; source?: { video?: string; width?: number; height?: number }; producer?: string; seed?: number; genre?: string; [k: string]: unknown };
  materials?: Material[];
  meshes: MeshDoc[];
  skeletons?: SkeletonDoc[];
  locations: LocationDoc[];
  entities: EntityDoc[];
  cameras: CameraDoc[];
  shots: ShotDoc[];
  continuity?: ContinuityDoc[];
}

const CTOR = {
  f32: Float32Array,
  u32: Uint32Array,
  u16: Uint16Array,
  u8: Uint8Array,
  i16: Int16Array,
} as const;

function f16ToF32(src: Uint16Array): Float32Array {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    const h = src[i];
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const f = h & 0x3ff;
    out[i] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
  }
  return out;
}

function isAccessor(v: unknown): v is { offset: number; dtype: string; shape: number[] } {
  if (!v || typeof v !== "object" || Array.isArray(v)) return false;
  const k = Object.keys(v);
  return k.length === 3 && "offset" in v && "dtype" in v && "shape" in v;
}

function resolve(v: unknown, bin: ArrayBuffer, base: number): unknown {
  if (isAccessor(v)) {
    const n = v.shape.reduce((a, b) => a * b, 1);
    if (v.dtype === "f16") {
      return { data: f16ToF32(new Uint16Array(bin.slice(base + v.offset, base + v.offset + n * 2))), shape: v.shape } as Arr;
    }
    const C = CTOR[v.dtype as keyof typeof CTOR];
    if (!C) throw new Error(`unknown dtype ${v.dtype}`);
    // Copy so the arrays are aligned and independent of the file buffer.
    const bytes = n * C.BYTES_PER_ELEMENT;
    return { data: new C(bin.slice(base + v.offset, base + v.offset + bytes)), shape: v.shape } as Arr;
  }
  if (Array.isArray(v)) return v.map((x) => resolve(x, bin, base));
  if (v && typeof v === "object") {
    const o: Record<string, unknown> = {};
    for (const [k, x] of Object.entries(v)) o[k] = resolve(x, bin, base);
    return o;
  }
  return v;
}

export function parseRestage(buf: ArrayBuffer): SceneDoc {
  const dv = new DataView(buf);
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== "RSTG") throw new Error("Not a .restage file");
  const version = dv.getUint32(4, true);
  if (version !== 1) throw new Error(`Unsupported .restage version ${version}`);
  const jl = dv.getUint32(8, true);
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 12, jl)));
  const binStart = 16 + jl;
  const doc = resolve(json, buf, binStart) as SceneDoc;
  doc.materials ??= [];
  doc.skeletons ??= [];
  doc.continuity ??= [];
  for (const l of doc.locations) l.props ??= [];
  return doc;
}

/** Row `i` of an [N, k] array as a plain number array view. */
export function row(a: Arr, i: number): ArrayLike<number> {
  const k = a.shape.slice(1).reduce((x, y) => x * y, 1);
  return a.data.subarray(i * k, (i + 1) * k);
}
