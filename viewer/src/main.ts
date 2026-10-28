/** Restage viewer: wires the store, viewport, timeline, panels, compare overlay, loading and exports. */
import "./styles.css";
import { buildScene, focalMM, type Style } from "./build";
import { parseRestage } from "./format";
import { Store, type Mode } from "./state";
import { Viewport } from "./viewport";
import { Timeline, fmtTime } from "./timeline";
import { Panels } from "./panels";
import { exportBlenderScript, exportGLB, exportMP4, exportOTIO, exportPNG } from "./export";

const store = new Store();
const app = document.getElementById("app")!;

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls = "", html = ""): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html) e.innerHTML = html;
  return e;
}

// ---------------------------------------------------------------- layout
const top = el("header", "topbar");
const stage = el("main", "stage");
const transport = el("div", "transport");
const tlWrap = el("div", "tl-wrap");
const side = el("aside", "side");
const resizer = el("div", "tl-resizer");
app.append(top, stage, transport, resizer, tlWrap, side);

const viewHost = el("div", "view-host");
stage.appendChild(viewHost);
const viewport = new Viewport(viewHost, store);
const timeline = new Timeline(tlWrap, store);
new Panels(side, store);

// HUD over the viewport
const hud = el("div", "hud");
const hudRight = el("div", "hud-right");
stage.append(hud, hudRight);

// compare overlay: the source video, aligned to the replica's letterbox
const video = el("video", "src-video") as HTMLVideoElement;
video.muted = true;
video.playsInline = true;
video.preload = "auto";
const wipeBar = el("div", "wipe-bar", `<span class="wipe-knob">⟷</span>`);
const noVideo = el("div", "no-video", "No source video loaded. Drop the .mp4 here, or use Open.");
stage.append(video, wipeBar, noVideo);

// empty state
const empty = el("div", "empty");
stage.appendChild(empty);

// ---------------------------------------------------------------- top bar
top.innerHTML = `
  <div class="brand"><svg viewBox="0 0 32 32" width="22" height="22"><rect x="3" y="10" width="26" height="17" rx="3" fill="#f59e0b"/><path d="M3 10l6-6h14l6 6z" fill="#fbbf24"/><circle cx="16" cy="18.5" r="4.2" fill="#0b0d12"/></svg><span>Restage</span></div>
  <div class="scene-pick"><button class="btn ghost" id="scene-btn"><span id="scene-name">No scene</span> ▾</button><div class="menu" id="scene-menu"></div></div>
  <button class="btn ghost" id="open-btn" title="Open a .restage scene and its source video">Open…</button>
  <input type="file" id="file-in" accept=".restage,.mp4,.webm,.mov" multiple hidden />
  <button class="btn" id="recon-btn" title="Upload a video and rebuild it in 3D with the Restage model">Reconstruct video…</button>
  <input type="file" id="recon-in" accept="video/*,.mp4,.mov,.webm,.mkv" hidden />
  <div class="seg" id="modes">
    <button data-mode="replica" title="Through the original cameras, with the original cuts (1)">Replica</button>
    <button data-mode="explore" title="Free camera around the current location (2)">Explore</button>
    <button data-mode="map" title="Top-down map with every camera and path (3)">Map</button>
    <button data-mode="compare" title="Replica against the source video (4)">Compare</button>
  </div>
  <div class="spacer"></div>
  <label class="sel-wrap" title="Render style (S)"><span>Style</span>
    <select id="style">
      <option value="shaded">Shaded</option><option value="clay">Clay</option><option value="wire">Wireframe</option>
      <option value="ids">Entities</option><option value="confidence">Seen vs. inferred</option><option value="normals">Normals</option>
    </select></label>
  <div class="dropdown"><button class="btn ghost">View ▾</button><div class="menu" id="view-menu"></div></div>
  <div class="dropdown"><button class="btn primary">Export ▾</button><div class="menu right" id="export-menu"></div></div>
`;
const $ = <T extends HTMLElement>(sel: string) => top.querySelector(sel) as T;
const sceneMenu = $<HTMLDivElement>("#scene-menu");
const fileIn = $<HTMLInputElement>("#file-in");
$<HTMLButtonElement>("#open-btn").onclick = () => fileIn.click();
fileIn.onchange = () => fileIn.files && loadFiles([...fileIn.files]);
$<HTMLButtonElement>("#scene-btn").onclick = (e) => {
  e.stopPropagation();
  refreshSamples().then(() => sceneMenu.classList.toggle("open"));
};
for (const dd of top.querySelectorAll<HTMLDivElement>(".dropdown")) {
  dd.querySelector("button")!.addEventListener("click", (e) => {
    e.stopPropagation();
    const m = dd.querySelector(".menu")!;
    const was = m.classList.contains("open");
    closeMenus();
    if (!was) m.classList.add("open");
  });
}
function closeMenus() {
  top.querySelectorAll(".menu.open").forEach((m) => m.classList.remove("open"));
}
document.addEventListener("click", closeMenus);

for (const b of top.querySelectorAll<HTMLButtonElement>("#modes button")) b.onclick = () => store.set({ mode: b.dataset.mode as Mode });
const styleSel = $<HTMLSelectElement>("#style");
styleSel.onchange = () => store.set({ style: styleSel.value as Style });

const VIEW_TOGGLES: [keyof typeof store.s, string, string][] = [
  ["showCameras", "Cameras and their paths", "C"], ["showTrails", "Entity motion trails", "T"],
  ["showInferred", "Entities while unseen (ghosted)", "I"], ["showGrid", "Ground grid", "G"],
  ["shadows", "Shadows", ""], ["followEdit", "Explore follows the edit", "E"],
];
function renderViewMenu() {
  const m = $<HTMLDivElement>("#view-menu");
  m.replaceChildren();
  for (const [k, label, key] of VIEW_TOGGLES) {
    const item = el("button", `item check${store.s[k] ? " on" : ""}`, `<span class="tick">✓</span>${label}${key ? `<kbd>${key}</kbd>` : ""}`);
    item.onclick = (e) => {
      e.stopPropagation();
      store.set({ [k]: !store.s[k] } as never);
    };
    m.appendChild(item);
  }
}

function renderExportMenu() {
  const m = $<HTMLDivElement>("#export-menu");
  m.replaceChildren();
  const items: [string, string, () => Promise<void> | void][] = [
    ["glTF binary (.glb)", "Sets, entities, cameras and animation, for any 3D tool", async () => {
      const r = await exportGLB(store.s.built!, baseName());
      toast(`Exported ${baseName()}.glb (${(r.bytes / 1e6).toFixed(1)} MB)`);
    }],
    ["Blender setup (.py)", "Imports the .glb and binds shot cameras to markers, so cuts play back", () => {
      exportBlenderScript(store.s.built!, baseName());
      toast("Exported Blender script. Put it next to the .glb and run it in Blender.");
    }],
    ["Edit (.otio)", "OpenTimelineIO shot list for editing software", () => {
      exportOTIO(store.s.built!, baseName(), store.s.videoUrl ? `${baseName()}.mp4` : null);
      toast("Exported OpenTimelineIO edit");
    }],
    ["Replica video (.mp4)", "Render through the shot cameras at the source resolution", async () => {
      const t = toast("Rendering replica… 0%", 0);
      const bytes = await exportMP4(store.s.built!, baseName(), (p) => (t.textContent = `Rendering replica… ${Math.round(p * 100)}%`));
      t.textContent = `Rendered ${baseName()}_replica.mp4 (${(bytes / 1e6).toFixed(1)} MB)`;
      setTimeout(() => t.remove(), 3500);
    }],
    ["Still image (.png)", "The current view", () => {
      viewport.render();
      exportPNG(viewport.canvas, baseName(), Math.floor(store.s.frame));
    }],
  ];
  for (const [label, sub, fn] of items) {
    const it = el("button", "item", `<div>${label}</div><small>${sub}</small>`);
    it.disabled = !store.s.built;
    it.onclick = async (e) => {
      e.stopPropagation();
      closeMenus();
      try {
        await fn();
      } catch (err) {
        console.error(err);
        toast(`Export failed: ${(err as Error).message}`, 6000, true);
      }
    };
    m.appendChild(it);
  }
}

const baseName = () => (store.s.sceneName || "scene").replace(/\.restage$/, "");

// ---------------------------------------------------------------- transport
transport.innerHTML = `
  <div class="tp-left">
    <button class="tbtn" id="tp-prev" title="Previous shot (↑)">⏮</button>
    <button class="tbtn" id="tp-back" title="Back one frame (←)">◂</button>
    <button class="tbtn play" id="tp-play" title="Play / pause (space)">▶</button>
    <button class="tbtn" id="tp-fwd" title="Forward one frame (→)">▸</button>
    <button class="tbtn" id="tp-next" title="Next shot (↓)">⏭</button>
    <button class="tbtn" id="tp-loop" title="Loop">⟳</button>
    <select id="tp-speed" title="Playback speed"><option value="0.25">¼×</option><option value="0.5">½×</option><option value="1" selected>1×</option><option value="2">2×</option></select>
    <span class="timecode" id="tp-time">0:00.00</span>
  </div>
  <div class="tp-mid" id="tp-shot"></div>
  <div class="tp-right" id="tp-stats"></div>
`;
const T = <E extends HTMLElement>(id: string) => transport.querySelector(`#${id}`) as E;
T("tp-play").onclick = () => togglePlay();
T("tp-back").onclick = () => step(-1);
T("tp-fwd").onclick = () => step(1);
T("tp-prev").onclick = () => jumpShot(-1);
T("tp-next").onclick = () => jumpShot(1);
T("tp-loop").onclick = () => store.set({ loop: !store.s.loop });
const speedSel = T<HTMLSelectElement>("tp-speed");
speedSel.onchange = () => {
  store.set({ speed: Number(speedSel.value) });
  video.playbackRate = store.s.speed;
};

function togglePlay() {
  const b = store.s.built;
  if (!b) return;
  if (!store.s.playing && Math.floor(store.s.frame) >= b.doc.n_frames - 1) store.set({ frame: 0 });
  store.set({ playing: !store.s.playing });
}
function step(d: number) {
  store.set({ playing: false });
  store.seek(Math.round(store.s.frame) + d);
}
function jumpShot(d: number) {
  const b = store.s.built;
  if (!b) return;
  const i = store.shotIndex();
  const shots = b.doc.shots;
  let j = i + d;
  // ↑ inside a shot first returns to its start
  if (d < 0 && i >= 0 && Math.floor(store.s.frame) > shots[i].start + 2) j = i;
  j = Math.max(0, Math.min(shots.length - 1, j));
  store.set({ playing: false, frame: shots[j].start, selection: { kind: "shot", index: j } });
}

// ---------------------------------------------------------------- playback clock
let last = performance.now();
function tick(now: number) {
  requestAnimationFrame(tick);
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  const s = store.s;
  const b = s.built;
  if (!b || !s.playing) return;
  const n = b.doc.n_frames;
  let f: number;
  if (s.videoUrl && video.readyState >= 2 && !video.paused) {
    f = video.currentTime * b.doc.fps;
  } else {
    f = s.frame + dt * b.doc.fps * s.speed;
  }
  if (f >= n - 1) {
    if (s.loop) {
      f = 0;
      if (s.videoUrl) video.currentTime = 0;
    } else {
      f = n - 1;
      store.set({ frame: f, playing: false });
      return;
    }
  }
  store.set({ frame: f });
}
requestAnimationFrame(tick);

store.on((keys) => {
  const s = store.s;
  if (keys.includes("playing")) {
    T("tp-play").textContent = s.playing ? "⏸" : "▶";
    if (s.videoUrl) {
      if (s.playing) {
        video.currentTime = (Math.floor(s.frame) + 0.5) / (s.built?.doc.fps ?? 30);
        video.playbackRate = s.speed;
        void video.play().catch(() => {});
      } else video.pause();
    }
  }
  if (keys.includes("frame") && !s.playing && s.videoUrl && s.built) {
    const t = (Math.floor(s.frame) + 0.5) / s.built.doc.fps;
    if (Math.abs(video.currentTime - t) > 0.5 / s.built.doc.fps) video.currentTime = t;
  }
  if (keys.includes("loop")) T("tp-loop").classList.toggle("on", s.loop);
  if (keys.some((k) => ["built", "videoUrl", "mode", "style", "showCameras", "showTrails", "showInferred", "showGrid", "shadows", "followEdit", "sceneName"].includes(k))) syncChrome();
  updateHud();
});

function syncChrome() {
  const s = store.s;
  for (const b of top.querySelectorAll<HTMLButtonElement>("#modes button")) b.classList.toggle("on", b.dataset.mode === s.mode);
  styleSel.value = s.style;
  $<HTMLSpanElement>("#scene-name").textContent = s.sceneName || "No scene";
  renderViewMenu();
  renderExportMenu();
  empty.style.display = s.built ? "none" : "flex";
  app.classList.toggle("has-scene", !!s.built);
  T("tp-loop").classList.toggle("on", s.loop);
}

// ---------------------------------------------------------------- HUD + compare overlay
function updateHud() {
  const s = store.s;
  const b = s.built;
  if (!b) {
    hud.innerHTML = "";
    hudRight.innerHTML = "";
    T("tp-shot").textContent = "";
    T("tp-time").textContent = "0:00.00";
    return;
  }
  const f = Math.floor(s.frame);
  T("tp-time").innerHTML = `${fmtTime(f / b.doc.fps)} <span class="dim">/ ${fmtTime(b.doc.n_frames / b.doc.fps)} · f ${f}</span>`;
  const si = store.shotIndex();
  const shot = si >= 0 ? b.doc.shots[si] : null;
  const loc = shot ? b.locById.get(shot.location) : null;
  const cam = shot ? b.camByShot.get(shot.id) : null;
  const shotText = shot
    ? `<span class="chip" style="--c:${loc?.color}">Shot ${si + 1}</span> ${shot.size ?? ""} · ${shot.move ?? ""} · ${cam ? focalMM(cam.camera.fov).toFixed(0) + " mm" : ""} · <b>${loc?.doc.name ?? ""}</b>`
    : "";
  T("tp-shot").innerHTML = shotText;
  const mode = s.mode;
  if (mode === "explore" || mode === "map") {
    const active = viewport.activeLocation();
    hud.innerHTML = `<div class="hud-card"><label>Location</label><select id="hud-loc">${b.locations
      .map((l, i) => `<option value="${i}"${active === l ? " selected" : ""}>${l.doc.name}</option>`).join("")}</select>
      <button class="mini${s.followEdit ? " on" : ""}" id="hud-follow" title="Switch location with the edit (E)">Follow edit</button></div>`;
    const sel = hud.querySelector<HTMLSelectElement>("#hud-loc")!;
    sel.onchange = () => store.set({ exploreLocation: Number(sel.value), followEdit: false });
    hud.querySelector<HTMLButtonElement>("#hud-follow")!.onclick = () => store.set({ followEdit: !s.followEdit });
  } else if (mode === "compare") {
    hud.innerHTML = `<div class="hud-card seg small">${(["wipe", "side", "onion"] as const)
      .map((k) => `<button data-k="${k}" class="${s.compareKind === k ? "on" : ""}">${k === "wipe" ? "Wipe" : k === "side" ? "Side by side" : "Overlay"}</button>`)
      .join("")}</div>`;
    for (const btn of hud.querySelectorAll<HTMLButtonElement>("button[data-k]")) btn.onclick = () => store.set({ compareKind: btn.dataset.k as never });
  } else hud.innerHTML = "";
  hudRight.innerHTML = mode === "compare" ? `<div class="hud-tag">Left: source video · Right: Restage replica</div>` : "";
  layoutCompare();
}

function layoutCompare() {
  const s = store.s;
  const show = s.mode === "compare" && !!s.built;
  const r = viewport.videoRect;
  const off = viewHost.offsetTop;
  if (!show) {
    video.style.display = "none";
    wipeBar.style.display = "none";
    noVideo.style.display = "none";
    return;
  }
  if (!s.videoUrl) {
    video.style.display = "none";
    wipeBar.style.display = "none";
    noVideo.style.display = "block";
    return;
  }
  noVideo.style.display = "none";
  video.style.display = "block";
  Object.assign(video.style, { left: `${r.x}px`, top: `${r.y + off}px`, width: `${r.w}px`, height: `${r.h}px` });
  if (s.compareKind === "wipe") {
    video.style.clipPath = `inset(0 ${(1 - s.wipe) * 100}% 0 0)`;
    video.style.opacity = "1";
    wipeBar.style.display = "block";
    Object.assign(wipeBar.style, { left: `${r.x + r.w * s.wipe}px`, top: `${r.y + off}px`, height: `${r.h}px` });
  } else {
    video.style.clipPath = "none";
    video.style.opacity = s.compareKind === "onion" ? "0.5" : "1";
    wipeBar.style.display = "none";
  }
}
viewport.onRendered = () => {
  if (store.s.mode === "compare") layoutCompare();
  const st = viewport.stats;
  const b = store.s.built;
  T("tp-stats").textContent = b ? `${Math.round(b.meshTris).toLocaleString()} tris · ${st.fps ? st.fps.toFixed(0) : "–"} fps` : "";
};
wipeBar.addEventListener("pointerdown", (e) => {
  wipeBar.setPointerCapture(e.pointerId);
  const move = (ev: PointerEvent) => {
    const r = viewport.videoRect;
    const x = ev.clientX - stage.getBoundingClientRect().left;
    store.set({ wipe: Math.max(0, Math.min(1, (x - r.x) / r.w)) });
  };
  const up = () => {
    wipeBar.removeEventListener("pointermove", move);
    wipeBar.removeEventListener("pointerup", up);
  };
  wipeBar.addEventListener("pointermove", move);
  wipeBar.addEventListener("pointerup", up);
});

// ---------------------------------------------------------------- timeline resizing
resizer.addEventListener("pointerdown", (e) => {
  resizer.setPointerCapture(e.pointerId);
  const startY = e.clientY;
  const startH = tlWrap.clientHeight;
  const move = (ev: PointerEvent) => {
    const h = Math.max(96, Math.min(window.innerHeight * 0.6, startH - (ev.clientY - startY)));
    app.style.setProperty("--tl-h", `${h}px`);
  };
  const up = () => {
    resizer.removeEventListener("pointermove", move);
    resizer.removeEventListener("pointerup", up);
    timeline.draw();
  };
  resizer.addEventListener("pointermove", move);
  resizer.addEventListener("pointerup", up);
});

// ---------------------------------------------------------------- loading
interface Sample { name: string; scene: string; video: string | null; bytes: number }
let samples: Sample[] = [];

async function refreshSamples(): Promise<void> {
  try {
    samples = await (await fetch("/samples/index.json", { cache: "no-store" })).json();
  } catch {
    samples = [];
  }
  sceneMenu.replaceChildren();
  if (!samples.length) sceneMenu.appendChild(el("div", "item muted", "No samples in samples/ yet"));
  for (const s of samples) {
    const it = el("button", `item${s.name === store.s.sceneName ? " on" : ""}`, `<div>${s.name}</div><small>${(s.bytes / 1e6).toFixed(1)} MB${s.video ? " · with video" : ""}</small>`);
    it.onclick = () => {
      closeMenus();
      sceneMenu.classList.remove("open");
      void loadSample(s);
    };
    sceneMenu.appendChild(it);
  }
  renderEmpty();
}

function renderEmpty() {
  empty.innerHTML = `
    <div class="empty-card">
      <h1>Rebuild any video as an animated 3D set</h1>
      <p>Open a <code>.restage</code> scene to play it through its original cameras, walk around each location, check continuity across cuts, and export to Blender, game engines or editing software.</p>
      <div class="drop">Drop a <b>.restage</b> file and its video here</div>
      <div class="samples-grid">${samples.map((s, i) => `<button class="sample" data-i="${i}"><b>${s.name}</b><small>${s.video ? "scene + video" : "scene"}</small></button>`).join("") || `<div class="muted">No sample scenes yet. Generate some with <code>python -m restage.gen</code>.</div>`}</div>
    </div>`;
  for (const btn of empty.querySelectorAll<HTMLButtonElement>(".sample")) btn.onclick = () => void loadSample(samples[Number(btn.dataset.i)]);
}

async function loadSample(s: Sample) {
  const t = toast(`Loading ${s.name}…`, 0);
  try {
    const buf = await (await fetch(s.scene, { cache: "no-store" })).arrayBuffer();
    loadScene(buf, s.name);
    setVideo(s.video);
    history.replaceState(null, "", `#${encodeURIComponent(s.name)}`);
  } catch (e) {
    toast(`Could not load ${s.name}: ${(e as Error).message}`, 6000, true);
  } finally {
    t.remove();
  }
}

function loadScene(buf: ArrayBuffer, name: string) {
  const doc = parseRestage(buf);
  const built = buildScene(doc);
  store.set({ built, sceneName: name, frame: 0, playing: false, selection: null, hiddenEntities: new Set(), followEntity: null,
    exploreLocation: 0 });
}

let objectUrl: string | null = null;
function setVideo(url: string | null) {
  if (objectUrl && url !== objectUrl) {
    URL.revokeObjectURL(objectUrl);
    objectUrl = null;
  }
  if (url) {
    video.src = url;
    video.load();
  } else video.removeAttribute("src");
  store.set({ videoUrl: url });
}

async function loadFiles(files: File[]) {
  const scene = files.find((f) => f.name.endsWith(".restage"));
  const vid = files.find((f) => /\.(mp4|webm|mov)$/i.test(f.name));
  try {
    if (scene) loadScene(await scene.arrayBuffer(), scene.name.replace(/\.restage$/, ""));
    if (vid) {
      objectUrl = URL.createObjectURL(vid);
      setVideo(objectUrl);
      if (!scene && store.s.built) store.set({ mode: "compare" });
    }
    if (!scene && !vid) toast("Drop a .restage scene and/or its video (.mp4, .webm, .mov)", 4000, true);
  } catch (e) {
    toast(`Could not open: ${(e as Error).message}`, 6000, true);
  }
}

window.addEventListener("dragover", (e) => {
  e.preventDefault();
  app.classList.add("dragging");
});
window.addEventListener("dragleave", (e) => {
  if (!e.relatedTarget) app.classList.remove("dragging");
});
window.addEventListener("drop", (e) => {
  e.preventDefault();
  app.classList.remove("dragging");
  if (e.dataTransfer?.files.length) void loadFiles([...e.dataTransfer.files]);
});

// ---------------------------------------------------------------- reconstruction jobs (restage.server)
const reconIn = $<HTMLInputElement>("#recon-in");
$<HTMLButtonElement>("#recon-btn").onclick = async () => {
  let health: { model_ready: boolean } | null = null;
  try {
    const r = await fetch("/api/health");
    if (r.ok) health = await r.json();
  } catch { /* server down */ }
  if (!health) {
    toast("The reconstruction server isn't running. Start it with: python -m restage.server", 7000, true);
    return;
  }
  if (!health.model_ready) {
    toast("The server is up, but there's no trained Restage checkpoint yet.", 6000, true);
    return;
  }
  reconIn.click();
};
reconIn.onchange = () => {
  const f = reconIn.files?.[0];
  reconIn.value = "";
  if (f) void reconstruct(f);
};

async function reconstruct(file: File) {
  const card = el("div", "job-card", `<div class="job-title">Reconstructing <b></b></div><div class="job-stage">Uploading…</div><div class="bar"><i></i></div>`);
  card.querySelector("b")!.textContent = file.name;
  document.body.appendChild(card);
  const stage = card.querySelector(".job-stage") as HTMLDivElement;
  const bar = card.querySelector(".bar i") as HTMLElement;
  const setP = (p: number, text: string) => {
    bar.style.width = `${Math.round(p * 100)}%`;
    stage.textContent = text;
  };
  try {
    const job = await new Promise<{ id: string }>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/reconstruct");
      xhr.upload.onprogress = (e) => e.lengthComputable && setP((e.loaded / e.total) * 0.1, `Uploading… ${Math.round((e.loaded / e.total) * 100)}%`);
      xhr.onload = () => (xhr.status < 300 ? resolve(JSON.parse(xhr.responseText)) : reject(new Error(JSON.parse(xhr.responseText).detail ?? xhr.statusText)));
      xhr.onerror = () => reject(new Error("upload failed"));
      const fd = new FormData();
      fd.append("video", file);
      xhr.send(fd);
    });
    for (;;) {
      await new Promise((r) => setTimeout(r, 500));
      const j = await (await fetch(`/api/jobs/${job.id}`)).json();
      if (j.state === "error") throw new Error(j.error);
      setP(0.1 + j.progress * 0.9, j.state === "queued" ? "Waiting for the model…" : j.stage);
      if (j.state === "done") {
        setP(1, `Done in ${j.elapsed.toFixed(1)} s`);
        await refreshSamples();
        const s = samples.find((x) => x.name === j.name);
        if (s) {
          await loadSample(s);
          store.set({ mode: "compare" });
        }
        setTimeout(() => card.remove(), 4000);
        return;
      }
    }
  } catch (e) {
    card.classList.add("err");
    setP(1, `Failed: ${(e as Error).message}`);
    setTimeout(() => card.remove(), 9000);
  }
}

// ---------------------------------------------------------------- toasts
const toasts = el("div", "toasts");
document.body.appendChild(toasts);
function toast(text: string, ms = 3000, error = false): HTMLDivElement {
  const t = el("div", `toast${error ? " err" : ""}`);
  t.textContent = text;
  toasts.appendChild(t);
  if (ms) setTimeout(() => t.remove(), ms);
  return t;
}

// ---------------------------------------------------------------- keyboard
window.addEventListener("keydown", (e) => {
  const tag = (e.target as HTMLElement).tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
  const s = store.s;
  const k = e.key;
  if (k === " ") { e.preventDefault(); togglePlay(); }
  else if (k === "ArrowLeft") { e.preventDefault(); step(e.shiftKey ? -10 : -1); }
  else if (k === "ArrowRight") { e.preventDefault(); step(e.shiftKey ? 10 : 1); }
  else if (k === "ArrowUp") { e.preventDefault(); jumpShot(-1); }
  else if (k === "ArrowDown") { e.preventDefault(); jumpShot(1); }
  else if (k === "Home") store.set({ frame: 0 });
  else if (k === "End" && s.built) store.set({ frame: s.built.doc.n_frames - 1 });
  else if (k >= "1" && k <= "4") store.set({ mode: (["replica", "explore", "map", "compare"] as Mode[])[Number(k) - 1] });
  else if (k === "s" || k === "S") {
    const styles: Style[] = ["shaded", "clay", "wire", "ids", "confidence", "normals"];
    store.set({ style: styles[(styles.indexOf(s.style) + 1) % styles.length] });
  } else if (k === "c") store.set({ showCameras: !s.showCameras });
  else if (k === "t") store.set({ showTrails: !s.showTrails });
  else if (k === "g") store.set({ showGrid: !s.showGrid });
  else if (k === "i") store.set({ showInferred: !s.showInferred });
  else if (k === "e") store.set({ followEdit: !s.followEdit });
  else if (k === "f") {
    const sel = s.selection;
    store.set({ followEntity: sel?.kind === "entity" && s.followEntity !== sel.index ? sel.index : null,
      mode: s.mode === "replica" || s.mode === "compare" ? "explore" : s.mode });
  } else if (k === "Escape") store.set({ selection: null, followEntity: null });
});

// ---------------------------------------------------------------- start
if (import.meta.env.DEV) {
  // Test hook for automated checks of the dev build.
  (window as unknown as Record<string, unknown>).__restage = { store, viewport, timeline, exportGLB, exportOTIO, exportBlenderScript, exportMP4 };
}
syncChrome();
void refreshSamples().then(() => {
  const want = decodeURIComponent(location.hash.slice(1));
  const s = samples.find((x) => x.name === want);
  if (s) void loadSample(s);
});
