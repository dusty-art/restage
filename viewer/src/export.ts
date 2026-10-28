/** Exports: GLB (backlot layout with animation), Blender setup script, OpenTimelineIO edit, MP4 render, PNG still. */
import * as THREE from "three";
import { GLTFExporter } from "three/examples/jsm/exporters/GLTFExporter.js";
import { Muxer, ArrayBufferTarget } from "mp4-muxer";
import { buildScene, focalMM, poseAt, type Built } from "./build";

function download(data: BlobPart, name: string, type: string): void {
  const url = URL.createObjectURL(new Blob([data], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

const safe = (s: string) => s.replace(/[^A-Za-z0-9_]/g, "_");

/**
 * GLB with every location laid out side by side along +X ("backlot"), entities
 * and shot cameras animated per frame, and the shot list in the root's extras.
 */
export async function exportGLB(src: Built, baseName: string): Promise<{ bytes: number; layout: Record<string, number> }> {
  const doc = src.doc;
  const b = buildScene(doc);
  const root = new THREE.Group();
  root.name = "Restage";
  const layout: Record<string, number> = {};
  let x = 0;
  for (const l of b.locations) {
    const size = l.bounds.getSize(new THREE.Vector3());
    const offset = x - l.bounds.min.x;
    layout[l.doc.id] = offset;
    x += size.x + 10;
    l.group.remove(l.helpers, l.hemi, l.sun, l.sun.target);
    l.group.name = `loc_${safe(l.doc.id)}`;
    l.group.position.x = offset;
    root.add(l.group);
  }
  const fps = doc.fps;
  const times = Float32Array.from({ length: doc.n_frames }, (_, i) => i / fps);
  const tracks: THREE.KeyframeTrack[] = [];
  for (const e of b.entities) {
    const id = safe(e.doc.id);
    e.root.name = id;
    e.mesh.name = `${id}_mesh`;
    e.bones.forEach((bn) => (bn.name = `${id}_${safe(bn.name)}`));
    const tr = e.doc.track;
    tracks.push(new THREE.VectorKeyframeTrack(`${id}.position`, times, Array.from(tr.position.data)));
    tracks.push(new THREE.QuaternionKeyframeTrack(`${id}.quaternion`, times, Array.from(tr.rotation.data)));
    // Presence: scale to zero while the entity is absent (glTF has no visibility channel).
    const sc: number[] = [];
    for (let t = 0; t < doc.n_frames; t++) {
      const present = !tr.state || tr.state.data[t] > 0;
      const s = tr.scale ? [tr.scale.data[t * 3], tr.scale.data[t * 3 + 1], tr.scale.data[t * 3 + 2]] : [1, 1, 1];
      sc.push(...(present ? s : [0, 0, 0]));
    }
    tracks.push(new THREE.VectorKeyframeTrack(`${id}.scale`, times, sc, THREE.InterpolateDiscrete));
    if (tr.pose && e.bones.length) {
      const J = tr.pose.shape[1];
      for (let j = 0; j < Math.min(J, e.bones.length); j++) {
        const v = new Float32Array(doc.n_frames * 4);
        for (let t = 0; t < doc.n_frames; t++) v.set(tr.pose.data.subarray((t * J + j) * 4, (t * J + j) * 4 + 4), t * 4);
        tracks.push(new THREE.QuaternionKeyframeTrack(`${e.bones[j].name}.quaternion`, times, Array.from(v)));
      }
    }
    if (tr.morph && e.mesh.morphTargetInfluences?.length) {
      tracks.push(new THREE.NumberKeyframeTrack(`${e.mesh.name}.morphTargetInfluences`, times, Array.from(tr.morph.data)));
    }
  }
  const shots: object[] = [];
  for (const c of b.cameras) {
    const name = `shot_${String(c.shotIndex + 1).padStart(2, "0")}`;
    c.camera.name = name;
    c.camera.fov = c.doc.track.fov.data[0];
    c.camera.aspect = b.aspect;
    c.camera.updateProjectionMatrix();
    c.location.group.add(c.camera);
    const n = c.doc.track.position.shape[0];
    const ct = Float32Array.from({ length: n }, (_, i) => (c.shot.start + i) / fps);
    tracks.push(new THREE.VectorKeyframeTrack(`${name}.position`, ct, Array.from(c.doc.track.position.data)));
    tracks.push(new THREE.QuaternionKeyframeTrack(`${name}.quaternion`, ct, Array.from(c.doc.track.rotation.data)));
    shots.push({ camera: name, start: c.shot.start, end: c.shot.end, location: c.shot.location, size: c.shot.size,
      move: c.shot.move, fov: Array.from(c.doc.track.fov.data), lens_mm: focalMM(c.doc.track.fov.data[0]) });
  }
  root.userData = { restage: { fps, n_frames: doc.n_frames, aspect: b.aspect, layout_offsets_x: layout, shots,
    continuity: doc.continuity ?? [] } };
  const clip = new THREE.AnimationClip("restage", doc.n_frames / fps, tracks);
  poseAt(b, 0);
  const glb = (await new GLTFExporter().parseAsync(root, { binary: true, animations: [clip], onlyVisible: false })) as ArrayBuffer;
  download(glb, `${baseName}.glb`, "model/gltf-binary");
  b.dispose();
  return { bytes: glb.byteLength, layout };
}

/** A Blender script that imports the GLB and binds one camera per shot to timeline markers, so the edit plays back. */
export function exportBlenderScript(src: Built, baseName: string): void {
  const doc = src.doc;
  const shots = doc.shots.map((s, i) => ({ camera: `shot_${String(i + 1).padStart(2, "0")}`, start: s.start, end: s.end,
    fov: src.camByShot.get(s.id)?.doc.track.fov.data[0] ?? 50, location: s.location }));
  const w = doc.meta?.source?.width ?? 1280;
  const h = doc.meta?.source?.height ?? 720;
  const py = `# Restage → Blender. Run in Blender (Scripting tab, or: blender --python ${baseName}_blender.py).
# Imports ${baseName}.glb from the same folder, sets the frame range and resolution, and binds
# each shot camera to a timeline marker so playback and rendering cut exactly like the source video.
import bpy, json, math, os

HERE = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.path.dirname(os.path.abspath(__file__))
GLB = os.path.join(HERE, "${baseName}.glb")
SHOTS = json.loads('''${JSON.stringify(shots)}''')
FPS = ${doc.fps}
N_FRAMES = ${doc.n_frames}

bpy.ops.import_scene.gltf(filepath=GLB, merge_vertices=False)
scene = bpy.context.scene
scene.render.fps = FPS
scene.frame_start = 1
scene.frame_end = N_FRAMES
scene.render.resolution_x = ${w}
scene.render.resolution_y = ${h}
for m in list(scene.timeline_markers):
    scene.timeline_markers.remove(m)
for i, s in enumerate(SHOTS):
    cam = bpy.data.objects.get(s["camera"])
    if cam is None:
        print("missing camera", s["camera"])
        continue
    cam.data.sensor_fit = "VERTICAL"
    cam.data.sensor_height = 24.0
    cam.data.lens = 12.0 / math.tan(math.radians(s["fov"]) / 2)
    mk = scene.timeline_markers.new(f"shot {i + 1}", frame=s["start"] + 1)
    mk.camera = cam
if SHOTS:
    scene.camera = bpy.data.objects.get(SHOTS[0]["camera"])
print("Restage: imported", len(SHOTS), "shots,", N_FRAMES, "frames at", FPS, "fps")
`;
  download(py, `${baseName}_blender.py`, "text/x-python");
}

/** OpenTimelineIO edit: one clip per shot, referencing the source video, with Restage metadata. */
export function exportOTIO(src: Built, baseName: string, videoName: string | null): void {
  const doc = src.doc;
  const fps = doc.fps;
  const rt = (v: number) => ({ OTIO_SCHEMA: "RationalTime.1", rate: fps, value: v });
  const clips = doc.shots.map((s, i) => ({
    OTIO_SCHEMA: "Clip.2",
    name: `Shot ${i + 1}`,
    source_range: { OTIO_SCHEMA: "TimeRange.1", start_time: rt(s.start), duration: rt(s.end - s.start) },
    media_references: { DEFAULT_MEDIA: { OTIO_SCHEMA: "ExternalReference.1", target_url: videoName ?? doc.meta?.source?.video ?? "",
      available_range: { OTIO_SCHEMA: "TimeRange.1", start_time: rt(0), duration: rt(doc.n_frames) } } },
    active_media_reference_key: "DEFAULT_MEDIA",
    effects: [],
    markers: [],
    metadata: { restage: { location: s.location, size: s.size, move: s.move, transition: s.transition,
      camera: `shot_${String(i + 1).padStart(2, "0")}` } },
  }));
  const tl = {
    OTIO_SCHEMA: "Timeline.1",
    name: doc.meta?.title ?? baseName,
    global_start_time: null,
    metadata: { restage: { continuity: doc.continuity ?? [] } },
    tracks: { OTIO_SCHEMA: "Stack.1", name: "tracks", children: [{ OTIO_SCHEMA: "Track.1", name: "V1", kind: "Video", children: clips,
      effects: [], markers: [], metadata: {} }], effects: [], markers: [], metadata: {} },
  };
  download(JSON.stringify(tl, null, 2), `${baseName}.otio`, "application/json");
}

/**
 * Render the replica through the shot cameras into an H.264 MP4 at the source resolution.
 * Uses its own renderer so the viewport is untouched; frames get exact timestamps (no real-time pacing).
 */
export async function exportMP4(src: Built, baseName: string, onProgress: (f: number) => void, width = 0): Promise<number> {
  if (!("VideoEncoder" in window)) throw new Error("This browser has no WebCodecs VideoEncoder; use Chrome or Safari 17+.");
  const doc = src.doc;
  const W0 = doc.meta?.source?.width ?? 1280;
  const H0 = doc.meta?.source?.height ?? 720;
  const W = (width || W0) & ~1;
  const H = Math.round((W * H0) / W0) & ~1;
  const b = buildScene(doc);
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setSize(W, H, false);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene();
  for (const l of b.locations) {
    l.helpers.visible = false;
    scene.add(l.group);
  }
  const muxer = new Muxer({ target: new ArrayBufferTarget(), video: { codec: "avc", width: W, height: H, frameRate: doc.fps }, fastStart: "in-memory" });
  let failed: Error | null = null;
  const enc = new VideoEncoder({ output: (chunk, meta) => muxer.addVideoChunk(chunk, meta), error: (e) => (failed = e as Error) });
  const codec = W * H > 1280 * 720 ? "avc1.640033" : "avc1.640028";
  enc.configure({ codec, width: W, height: H, bitrate: Math.round(W * H * doc.fps * 0.12), framerate: doc.fps });
  for (let f = 0; f < doc.n_frames; f++) {
    if (failed) throw failed;
    poseAt(b, f);
    const si = b.shotAt[f];
    const shot = si >= 0 ? doc.shots[si] : null;
    for (const l of b.locations) l.group.visible = !!shot && l.doc.id === shot.location;
    for (const e of b.entities) e.root.visible = (e.doc.track.state ? e.doc.track.state.data[f] : 2) > 0;
    const cv = shot ? b.camByShot.get(shot.id) : undefined;
    const loc = shot ? b.locById.get(shot.location) : undefined;
    if (cv && loc) {
      scene.background = loc.sky;
      scene.fog = loc.fog;
      cv.camera.aspect = W / H;
      cv.camera.updateProjectionMatrix();
      renderer.render(scene, cv.camera);
    } else {
      renderer.setClearColor(0x000000);
      renderer.clear();
    }
    const frame = new VideoFrame(renderer.domElement, { timestamp: Math.round((f * 1e6) / doc.fps), duration: Math.round(1e6 / doc.fps) });
    enc.encode(frame, { keyFrame: f % (doc.fps * 2) === 0 || (shot !== null && shot.start === f) });
    frame.close();
    if (enc.encodeQueueSize > 8) await new Promise((r) => setTimeout(r, 0));
    if (f % 10 === 0) {
      onProgress(f / doc.n_frames);
      await new Promise((r) => setTimeout(r, 0));
    }
  }
  await enc.flush();
  muxer.finalize();
  const buf = (muxer.target as ArrayBufferTarget).buffer;
  download(buf, `${baseName}_replica.mp4`, "video/mp4");
  renderer.dispose();
  b.dispose();
  onProgress(1);
  return buf.byteLength;
}

export function exportPNG(canvas: HTMLCanvasElement, baseName: string, frame: number): void {
  canvas.toBlob((blob) => blob && download(blob, `${baseName}_f${frame}.png`, "image/png"), "image/png");
}
