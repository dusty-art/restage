/** The 3D viewport: replica (through the shot cameras), explore (free orbit), map (top-down), compare (with the source video). */
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { ENTITY_COLORS, poseAt, type Built, type EntityView, type LocationView, type Style } from "./build";
import type { Store } from "./state";

export interface Rect { x: number; y: number; w: number; h: number }

const GHOST = new THREE.MeshBasicMaterial({ color: 0x9ca3af, transparent: true, opacity: 0.28, depthWrite: false });

function confidenceMaterial(seen: boolean): THREE.ShaderMaterial {
  // Green where the surface was observed in the video, orange where it was inferred.
  return new THREE.ShaderMaterial({
    uniforms: { seen: { value: seen ? 1 : 0 } },
    vertexShader: /* glsl */ `
      #include <common>
      #include <skinning_pars_vertex>
      #include <morphtarget_pars_vertex>
      attribute float observed;
      varying float vObs;
      varying vec3 vN;
      void main() {
        #include <skinbase_vertex>
        #include <begin_vertex>
        #include <morphtarget_vertex>
        #include <skinning_vertex>
        #include <project_vertex>
        vObs = observed;
        vN = normalize(normalMatrix * normal);
      }`,
    fragmentShader: /* glsl */ `
      uniform float seen;
      varying float vObs;
      varying vec3 vN;
      void main() {
        float o = vObs * seen;
        vec3 c = mix(vec3(0.98, 0.55, 0.15), vec3(0.2, 0.85, 0.45), o);
        float l = 0.55 + 0.45 * abs(normalize(vN).z);
        gl_FragColor = vec4(c * l, 1.0);
      }`,
  });
}

export class Viewport {
  readonly canvas: HTMLCanvasElement;
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly orbitCam = new THREE.PerspectiveCamera(50, 1, 0.03, 3000);
  readonly mapCam = new THREE.OrthographicCamera(-10, 10, 10, -10, 0.1, 1000);
  readonly controls: OrbitControls;
  private built: Built | null = null;
  private dirty = true;
  private styleCache = new Map<string, THREE.Material>();
  private ray = new THREE.Raycaster();
  private lastLoc: LocationView | null = null;
  replicaRect: Rect = { x: 0, y: 0, w: 1, h: 1 };
  videoRect: Rect = { x: 0, y: 0, w: 0, h: 0 };
  stats = { fps: 0, ms: 0 };
  private frames = 0;
  private lastStat = performance.now();
  onRendered: (() => void) | null = null;

  constructor(private host: HTMLElement, private store: Store) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.setScissorTest(true);
    this.canvas = this.renderer.domElement;
    this.canvas.className = "viewport-canvas";
    host.appendChild(this.canvas);
    this.controls = new OrbitControls(this.orbitCam, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.addEventListener("change", () => (this.dirty = true));
    this.mapCam.up.set(0, 0, -1);
    new ResizeObserver(() => this.resize()).observe(host);
    this.canvas.addEventListener("pointerdown", (e) => this.pointerDown(e));
    store.on((keys) => {
      if (keys.includes("built")) this.setBuilt(store.s.built);
      if (keys.includes("mode") || keys.includes("exploreLocation")) this.frameLocation();
      if (keys.includes("shadows")) this.renderer.shadowMap.enabled = store.s.shadows;
      this.dirty = true;
    });
    this.resize();
    const loop = () => {
      requestAnimationFrame(loop);
      this.controls.enabled = store.s.mode === "explore" || store.s.mode === "map";
      this.controls.update();
      if (this.dirty || store.s.playing) this.render();
    };
    loop();
  }

  invalidate(): void { this.dirty = true; }

  private setBuilt(b: Built | null): void {
    if (this.built) {
      for (const l of this.built.locations) this.scene.remove(l.group);
      this.built.dispose();
    }
    this.styleCache.forEach((m) => m.dispose());
    this.styleCache.clear();
    this.built = b;
    if (!b) return;
    for (const l of b.locations) this.scene.add(l.group);
    this.lastLoc = null;
    this.frameLocation();
  }

  activeLocation(): LocationView | null {
    const b = this.built;
    if (!b) return null;
    const s = this.store.s;
    const follow = s.mode === "replica" || s.mode === "compare" || s.followEdit;
    if (follow) {
      const si = this.store.shotIndex();
      if (si >= 0) return b.locById.get(b.doc.shots[si].location) ?? b.locations[0];
    }
    return b.locations[Math.min(s.exploreLocation, b.locations.length - 1)] ?? null;
  }

  /** Point the orbit and map cameras at the active location. */
  frameLocation(): void {
    const loc = this.activeLocation();
    if (!loc) return;
    const c = loc.bounds.getCenter(new THREE.Vector3());
    const size = loc.bounds.getSize(new THREE.Vector3());
    const r = Math.max(size.x, size.z, 4);
    this.controls.target.set(c.x, Math.min(1.0, c.y), c.z);
    this.orbitCam.position.set(c.x + r * 0.55, Math.max(2.5, r * 0.45), c.z + r * 0.75);
    this.orbitCam.lookAt(this.controls.target);
    this.mapCam.position.set(c.x, loc.bounds.max.y + 50, c.z);
    this.mapCam.lookAt(c.x, 0, c.z);
    this.mapCam.userData.extent = r * 0.6;
    this.updateMapFrustum();
    this.dirty = true;
  }

  private updateMapFrustum(): void {
    const ext = (this.mapCam.userData.extent as number) ?? 10;
    const a = this.canvas.clientWidth / Math.max(1, this.canvas.clientHeight);
    this.mapCam.left = -ext * a;
    this.mapCam.right = ext * a;
    this.mapCam.top = ext;
    this.mapCam.bottom = -ext;
    this.mapCam.updateProjectionMatrix();
  }

  resize(): void {
    const w = this.host.clientWidth;
    const h = this.host.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    this.orbitCam.aspect = w / h;
    this.orbitCam.updateProjectionMatrix();
    this.updateMapFrustum();
    this.dirty = true;
  }

  private fit(aspect: number, box: Rect): Rect {
    let w = box.w;
    let h = w / aspect;
    if (h > box.h) {
      h = box.h;
      w = h * aspect;
    }
    return { x: Math.round(box.x + (box.w - w) / 2), y: Math.round(box.y + (box.h - h) / 2), w: Math.round(w), h: Math.round(h) };
  }

  private material(style: Style, e: EntityView | null, obj: THREE.Mesh): THREE.Material {
    const base = obj.userData.baseMaterial as THREE.Material;
    if (style === "shaded") return base;
    if (style === "ids") {
      const color = e ? `#${e.color.getHexString()}` : "#5b6170";
      return this.cached(`ids${color}`, () => new THREE.MeshBasicMaterial({ color }));
    }
    if (style === "clay") return this.cached("clay", () => new THREE.MeshStandardMaterial({ color: 0xd9d4cc, roughness: 0.9, flatShading: true }));
    if (style === "wire") {
      const color = e ? `#${e.color.getHexString()}` : "#8b93a7";
      return this.cached(`wire${color}`, () => new THREE.MeshBasicMaterial({ color, wireframe: true }));
    }
    if (style === "normals") return this.cached("normals", () => new THREE.MeshNormalMaterial({ flatShading: true }));
    const seen = e ? e.root.userData.state === 2 : true;
    return this.cached(`conf${seen}`, () => confidenceMaterial(seen));
  }

  private cached(key: string, make: () => THREE.Material): THREE.Material {
    let m = this.styleCache.get(key);
    if (!m) {
      m = make();
      this.styleCache.set(key, m);
    }
    return m;
  }

  render(): void {
    const t0 = performance.now();
    this.dirty = false;
    const s = this.store.s;
    const b = this.built;
    const W = this.canvas.clientWidth;
    const H = this.canvas.clientHeight;
    this.renderer.setScissor(0, 0, W, H);
    this.renderer.setViewport(0, 0, W, H);
    this.renderer.setClearColor(0x0b0d12, 1);
    this.renderer.clear();
    if (!b) return;
    poseAt(b, s.frame);
    const loc = this.activeLocation();
    if (loc !== this.lastLoc && (s.mode === "explore" || s.mode === "map") && s.followEdit) {
      this.lastLoc = loc;
      this.frameLocation();
    }
    this.lastLoc = loc;
    const replicaView = s.mode === "replica" || s.mode === "compare";
    for (const l of b.locations) {
      l.group.visible = l === loc;
      l.helpers.visible = !replicaView;
      l.grid.visible = s.showGrid;
    }
    for (const c of b.cameras) {
      c.marker.visible = s.showCameras;
      c.path.visible = s.showCameras;
    }
    const style = s.style;
    for (const e of b.entities) {
      const st = e.root.userData.state as number;
      const hidden = s.hiddenEntities.has(e.index);
      e.trail.visible = s.showTrails && !hidden;
      e.root.visible = !hidden && (st === 2 || (st === 1 && (replicaView || s.showInferred)));
      e.mesh.material = st === 1 && !replicaView && style === "shaded" ? GHOST : this.material(style, e, e.mesh);
      e.mesh.castShadow = st === 2 || replicaView;
    }
    if (loc) {
      for (const p of loc.props.children) {
        const m = p as THREE.Mesh;
        m.material = this.material(style, null, m);
      }
      const dark = style === "ids" || style === "normals" || style === "confidence" || style === "wire";
      this.scene.background = dark ? new THREE.Color(0x10131a) : loc.sky;
      // Fog belongs to the location's look through its cameras; from the map's 50 m height it only washes everything out.
      this.scene.fog = dark || s.mode === "map" ? null : loc.fog;
    }
    const aspect = b.aspect;
    if (replicaView) {
      const si = this.store.shotIndex();
      const cv = si >= 0 ? b.camByShot.get(b.doc.shots[si].id) : undefined;
      let box: Rect = { x: 0, y: 0, w: W, h: H };
      if (s.mode === "compare" && s.compareKind === "side") {
        this.videoRect = this.fit(aspect, { x: 0, y: 0, w: W / 2 - 4, h: H });
        box = { x: W / 2 + 4, y: 0, w: W / 2 - 4, h: H };
      }
      const r = this.fit(aspect, box);
      if (!(s.mode === "compare" && s.compareKind === "side")) this.videoRect = r;
      this.replicaRect = r;
      if (cv) {
        cv.camera.aspect = aspect;
        cv.camera.updateProjectionMatrix();
        this.renderer.setScissor(r.x, H - r.y - r.h, r.w, r.h);
        this.renderer.setViewport(r.x, H - r.y - r.h, r.w, r.h);
        this.renderer.render(this.scene, cv.camera);
      }
    } else {
      this.replicaRect = { x: 0, y: 0, w: W, h: H };
      if (s.followEntity !== null) {
        const e = b.entities[s.followEntity];
        if (e) {
          const p = e.root.getWorldPosition(new THREE.Vector3()).add(new THREE.Vector3(0, 0.9, 0));
          const d = p.clone().sub(this.controls.target);
          this.controls.target.add(d);
          this.orbitCam.position.add(d);
        }
      }
      const cam = s.mode === "map" ? this.mapCam : this.orbitCam;
      this.renderer.render(this.scene, cam);
    }
    this.frames++;
    const now = performance.now();
    this.stats.ms = now - t0;
    if (now - this.lastStat > 500) {
      this.stats.fps = (this.frames * 1000) / (now - this.lastStat);
      this.frames = 0;
      this.lastStat = now;
    }
    this.onRendered?.();
  }

  private pointerDown(ev: PointerEvent): void {
    const b = this.built;
    if (!b || ev.button !== 0) return;
    const start = { x: ev.clientX, y: ev.clientY };
    const up = (e2: PointerEvent) => {
      this.canvas.removeEventListener("pointerup", up);
      if (Math.hypot(e2.clientX - start.x, e2.clientY - start.y) > 4) return; // a drag, not a click
      this.pick(e2);
    };
    this.canvas.addEventListener("pointerup", up);
  }

  private pick(ev: PointerEvent): void {
    const b = this.built!;
    const s = this.store.s;
    const rect = this.canvas.getBoundingClientRect();
    const r = this.replicaRect;
    const x = ev.clientX - rect.left;
    const y = ev.clientY - rect.top;
    if (x < r.x || y < r.y || x > r.x + r.w || y > r.y + r.h) return;
    const ndc = new THREE.Vector2(((x - r.x) / r.w) * 2 - 1, -((y - r.y) / r.h) * 2 + 1);
    let cam: THREE.Camera = s.mode === "map" ? this.mapCam : this.orbitCam;
    if (s.mode === "replica" || s.mode === "compare") {
      const si = this.store.shotIndex();
      const cv = si >= 0 ? b.camByShot.get(b.doc.shots[si].id) : undefined;
      if (!cv) return;
      cam = cv.camera;
    }
    this.ray.setFromCamera(ndc, cam);
    const loc = this.activeLocation();
    const targets: THREE.Object3D[] = [];
    for (const e of b.entities) if (e.root.visible && e.location === loc) targets.push(e.mesh);
    if (s.mode !== "replica" && s.mode !== "compare" && s.showCameras) for (const c of b.cameras) if (c.location === loc) targets.push(c.marker);
    // Skinned meshes need their bounding volumes refreshed for raycasting.
    for (const e of b.entities) if ((e.mesh as THREE.SkinnedMesh).isSkinnedMesh) (e.mesh as THREE.SkinnedMesh).computeBoundingSphere();
    const hit = this.ray.intersectObjects(targets, true)[0];
    if (!hit) {
      this.store.set({ selection: null });
      return;
    }
    let o: THREE.Object3D | null = hit.object;
    while (o && !o.userData.kind) o = o.parent;
    if (!o) return;
    if (o.userData.kind === "entity") this.store.set({ selection: { kind: "entity", index: o.userData.index as number } });
    else if (o.userData.kind === "camera") {
      const i = b.doc.shots.findIndex((sh) => sh.id === o!.userData.shot);
      if (i >= 0) this.store.set({ selection: { kind: "shot", index: i } });
    }
  }

  /** Colour used for entity i in "ids" style (also used by the UI). */
  static entityColor(i: number): string {
    return ENTITY_COLORS[i % ENTITY_COLORS.length];
  }
}
