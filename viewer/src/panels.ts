/** Side panel: outliner (locations, entities, shots, continuity) and the inspector for the selection. */
import { focalMM } from "./build";
import type { Store } from "./state";
import { fmtTime } from "./timeline";

const CLASS_ICON: Record<string, string> = {
  person: "👤", animal: "🐾", vehicle: "🚗", object: "◆", ball: "●",
};

function h<K extends keyof HTMLElementTagNameMap>(tag: K, cls = "", text = ""): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  return e;
}

export class Panels {
  readonly el: HTMLDivElement;
  private outliner: HTMLDivElement;
  private inspector: HTMLDivElement;
  private tab: "scene" | "shots" | "continuity" = "scene";
  private tabs: HTMLDivElement;

  constructor(host: HTMLElement, private store: Store) {
    this.el = h("div", "panels");
    this.tabs = h("div", "panel-tabs");
    this.outliner = h("div", "outliner");
    this.inspector = h("div", "inspector");
    this.el.append(this.tabs, this.outliner, h("div", "panel-divider"), this.inspector);
    host.appendChild(this.el);
    store.on((keys) => {
      if (keys.some((k) => ["built", "selection", "hiddenEntities", "followEntity", "exploreLocation", "followEdit"].includes(k))) this.render();
      else if (keys.includes("frame") && !store.s.playing) this.renderInspector();
    });
    this.render();
  }

  private render(): void {
    this.renderTabs();
    this.renderOutliner();
    this.renderInspector();
  }

  private renderTabs(): void {
    const b = this.store.s.built;
    this.tabs.replaceChildren();
    const n = b?.doc.continuity?.length ?? 0;
    const items: [typeof this.tab, string][] = [["scene", "Scene"], ["shots", `Shots${b ? " " + b.doc.shots.length : ""}`],
      ["continuity", `Continuity${n ? " " + n : ""}`]];
    for (const [id, label] of items) {
      const t = h("button", `tab${this.tab === id ? " on" : ""}${id === "continuity" && n ? " warn" : ""}`, label);
      t.onclick = () => {
        this.tab = id;
        this.render();
      };
      this.tabs.appendChild(t);
    }
  }

  private renderOutliner(): void {
    const s = this.store.s;
    const b = s.built;
    this.outliner.replaceChildren();
    if (!b) {
      this.outliner.appendChild(h("div", "empty-note", "Open a scene to see its locations, entities and shots."));
      return;
    }
    const sel = s.selection;
    if (this.tab === "scene") {
      this.outliner.appendChild(h("div", "section-title", `Locations · ${b.locations.length}`));
      b.locations.forEach((l, i) => {
        const r = h("div", `row${sel?.kind === "location" && sel.index === i ? " sel" : ""}`);
        const sw = h("span", "swatch");
        sw.style.background = l.color;
        const shots = b.doc.shots.filter((x) => x.location === l.doc.id).length;
        r.append(sw, h("span", "name", l.doc.name), h("span", "meta", `${l.doc.kind ?? ""} · ${shots} shot${shots === 1 ? "" : "s"}`));
        r.onclick = () => {
          const first = b.doc.shots.find((x) => x.location === l.doc.id);
          this.store.set({ selection: { kind: "location", index: i }, exploreLocation: i, followEdit: false,
            ...(s.mode === "replica" || s.mode === "compare" ? { frame: first?.start ?? s.frame } : {}) });
        };
        this.outliner.appendChild(r);
      });
      this.outliner.appendChild(h("div", "section-title", `Entities · ${b.entities.length}`));
      b.entities.forEach((e, i) => {
        const r = h("div", `row${sel?.kind === "entity" && sel.index === i ? " sel" : ""}`);
        const dot = h("span", "dot");
        dot.style.background = `#${e.color.getHexString()}`;
        const icon = h("span", "icon", CLASS_ICON[e.doc.class ?? ""] ?? "◆");
        const name = h("span", "name", e.doc.name ?? e.doc.id);
        const eye = h("button", `icon-btn${s.hiddenEntities.has(i) ? " off" : ""}`, s.hiddenEntities.has(i) ? "◌" : "◉");
        eye.title = s.hiddenEntities.has(i) ? "Show" : "Hide";
        eye.onclick = (ev) => {
          ev.stopPropagation();
          const hidden = new Set(s.hiddenEntities);
          if (hidden.has(i)) hidden.delete(i);
          else hidden.add(i);
          this.store.set({ hiddenEntities: hidden });
        };
        const follow = h("button", `icon-btn${s.followEntity === i ? " on" : ""}`, "⌖");
        follow.title = "Follow in Explore view";
        follow.onclick = (ev) => {
          ev.stopPropagation();
          this.store.set({ followEntity: s.followEntity === i ? null : i, mode: s.mode === "replica" || s.mode === "compare" ? "explore" : s.mode });
        };
        r.append(dot, icon, name, h("span", "meta", e.location?.doc.name ?? ""), follow, eye);
        r.onclick = () => this.store.set({ selection: { kind: "entity", index: i } });
        this.outliner.appendChild(r);
      });
    } else if (this.tab === "shots") {
      b.doc.shots.forEach((sh, i) => {
        const r = h("div", `row shot${sel?.kind === "shot" && sel.index === i ? " sel" : ""}${this.store.shotIndex() === i ? " live" : ""}`);
        const loc = b.locById.get(sh.location);
        const sw = h("span", "swatch");
        sw.style.background = loc?.color ?? "#888";
        r.append(sw, h("span", "num", String(i + 1)), h("span", "name", `${sh.size ?? ""} · ${sh.move ?? ""}`),
          h("span", "meta", `${((sh.end - sh.start) / b.doc.fps).toFixed(1)} s`));
        r.onclick = () => this.store.set({ selection: { kind: "shot", index: i }, frame: sh.start, playing: false });
        this.outliner.appendChild(r);
      });
    } else {
      const cs = b.doc.continuity ?? [];
      if (!cs.length) this.outliner.appendChild(h("div", "empty-note", "No continuity issues recorded for this scene."));
      cs.forEach((c, i) => {
        const r = h("div", `row cont${sel?.kind === "continuity" && sel.index === i ? " sel" : ""}`);
        const sev = c.severity ?? 0.5;
        const badge = h("span", `sev ${sev > 0.66 ? "hi" : sev > 0.33 ? "mid" : "lo"}`, c.kind);
        const ent = c.entity ? b.doc.entities.find((x) => x.id === c.entity)?.name ?? c.entity : "";
        r.append(badge, h("span", "name", ent || c.detail || ""));
        r.onclick = () => {
          const f = c.frame ?? b.doc.shots.find((x) => x.id === c.to_shot)?.start ?? 0;
          this.store.set({ selection: { kind: "continuity", index: i }, frame: f, playing: false });
        };
        this.outliner.appendChild(r);
      });
    }
  }

  renderInspector(): void {
    const s = this.store.s;
    const b = s.built;
    this.inspector.replaceChildren();
    if (!b) return;
    const sel = s.selection;
    const kv = (k: string, v: string) => {
      const r = h("div", "kv");
      r.append(h("span", "k", k), h("span", "v", v));
      this.inspector.appendChild(r);
    };
    const title = (t: string, sub = "") => {
      this.inspector.appendChild(h("div", "insp-title", t));
      if (sub) this.inspector.appendChild(h("div", "insp-sub", sub));
    };
    const doc = b.doc;
    const fps = doc.fps;
    if (!sel) {
      title(doc.meta?.title ?? s.sceneName, doc.meta?.producer ?? "");
      kv("Duration", `${fmtTime(doc.n_frames / fps)} · ${doc.n_frames} frames @ ${fps} fps`);
      kv("Locations", String(doc.locations.length));
      kv("Entities", String(doc.entities.length));
      kv("Shots", `${doc.shots.length} (${(doc.n_frames / fps / Math.max(1, doc.shots.length)).toFixed(1)} s average)`);
      kv("Triangles", Math.round(b.meshTris).toLocaleString());
      if (doc.meta?.genre) kv("Genre", String(doc.meta.genre));
      const si = this.store.shotIndex();
      if (si >= 0) kv("Now", `shot ${si + 1} · ${b.locById.get(doc.shots[si].location)?.doc.name ?? ""}`);
      this.inspector.appendChild(h("div", "hint", "Click anything in the viewport, timeline or list to inspect it."));
      return;
    }
    if (sel.kind === "entity") {
      const e = b.entities[sel.index];
      const tr = e.doc.track;
      title(e.doc.name ?? e.doc.id, `${e.doc.class ?? "object"} in ${e.location?.doc.name ?? "?"}`);
      const n = doc.n_frames;
      kv("On screen", `${((e.seenFrames / n) * 100).toFixed(0)}% of frames`);
      kv("Inferred", `${((e.inferredFrames / n) * 100).toFixed(0)}% (present, not visible)`);
      const f = Math.floor(s.frame);
      const st = tr.state ? tr.state.data[f] : 2;
      kv("This frame", st === 2 ? "visible" : st === 1 ? "present, off screen or hidden" : "absent");
      const p = tr.position.data;
      kv("Position", `${p[f * 3].toFixed(2)}, ${p[f * 3 + 1].toFixed(2)}, ${p[f * 3 + 2].toFixed(2)} m`);
      if (tr.confidence) kv("Confidence", `${(tr.confidence.data[f] * 100).toFixed(0)}%`);
      // speed over the last few frames
      const f0 = Math.max(0, f - 3);
      const d = Math.hypot(p[f * 3] - p[f0 * 3], p[f * 3 + 2] - p[f0 * 3 + 2]);
      kv("Speed", `${((d / Math.max(1, f - f0)) * fps).toFixed(2)} m/s`);
      kv("Mesh", `${Math.round(e.tris).toLocaleString()} triangles${e.bones.length ? ` · ${e.bones.length} joints` : ""}`);
      if (e.doc.identity) {
        const same = b.entities.filter((x) => x !== e && x.doc.identity === e.doc.identity);
        if (same.length) kv("Same identity", same.map((x) => `${x.doc.name ?? x.doc.id} (${x.location?.doc.name ?? "?"})`).join(", "));
      }
      const shots = doc.shots.filter((sh) => {
        if (!tr.state) return true;
        for (let t = sh.start; t < sh.end; t++) if (tr.state.data[t] === 2) return true;
        return false;
      });
      kv("Appears in", shots.length ? shots.map((sh) => doc.shots.indexOf(sh) + 1).join(", ") : "no shots");
    } else if (sel.kind === "shot") {
      const sh = doc.shots[sel.index];
      const cv = b.camByShot.get(sh.id);
      title(`Shot ${sel.index + 1}`, `${sh.size ?? ""} · ${sh.move ?? ""} · ${b.locById.get(sh.location)?.doc.name ?? ""}`);
      kv("Time", `${fmtTime(sh.start / fps)} – ${fmtTime(sh.end / fps)}`);
      kv("Length", `${((sh.end - sh.start) / fps).toFixed(2)} s · ${sh.end - sh.start} frames`);
      kv("Begins with", sh.transition ?? "cut");
      if (sh.time_skip) kv("Time skip", `${sh.time_skip.toFixed(1)} s after the previous shot here`);
      if (cv) {
        const fov = cv.doc.track.fov.data;
        kv("Lens", `${focalMM(fov[0]).toFixed(0)} mm (${fov[0].toFixed(1)}° vertical)`);
        const pos = cv.doc.track.position.data;
        const n = cv.doc.track.position.shape[0];
        let travel = 0;
        for (let i = 1; i < n; i++) travel += Math.hypot(pos[i * 3] - pos[i * 3 - 3], pos[i * 3 + 1] - pos[i * 3 - 2], pos[i * 3 + 2] - pos[i * 3 - 1]);
        kv("Camera height", `${pos[1].toFixed(2)} m`);
        kv("Camera travel", `${travel.toFixed(2)} m`);
      }
      const vis = b.entities.filter((e) => {
        const st = e.doc.track.state?.data;
        if (!st) return false;
        for (let t = sh.start; t < sh.end; t++) if (st[t] === 2) return true;
        return false;
      });
      kv("In frame", vis.map((e) => e.doc.name ?? e.doc.id).join(", ") || "nobody");
      const btn = h("button", "btn small", "Look through this camera");
      btn.onclick = () => this.store.set({ mode: "replica", frame: sh.start, playing: false });
      this.inspector.appendChild(btn);
    } else if (sel.kind === "location") {
      const l = b.locations[sel.index];
      title(l.doc.name, l.doc.kind ?? "");
      const size = l.bounds.getSize(l.bounds.min.clone());
      kv("Size", `${size.x.toFixed(1)} × ${size.z.toFixed(1)} m, ${size.y.toFixed(1)} m tall`);
      kv("Props", String(l.doc.props.length));
      const shots = doc.shots.filter((x) => x.location === l.doc.id);
      kv("Shots here", shots.map((x) => doc.shots.indexOf(x) + 1).join(", "));
      kv("Screen time", `${(shots.reduce((a, x) => a + x.end - x.start, 0) / fps).toFixed(1)} s`);
      kv("Entities", b.entities.filter((e) => e.location === l).map((e) => e.doc.name ?? e.doc.id).join(", ") || "none");
    } else if (sel.kind === "continuity") {
      const c = (doc.continuity ?? [])[sel.index];
      title(`Continuity: ${c.kind}`, c.entity ? doc.entities.find((x) => x.id === c.entity)?.name ?? c.entity : "");
      if (c.detail) this.inspector.appendChild(h("div", "detail", c.detail));
      const from = doc.shots.findIndex((x) => x.id === c.from_shot);
      const to = doc.shots.findIndex((x) => x.id === c.to_shot);
      if (from >= 0 || to >= 0) kv("Across the cut", `shot ${from + 1} → shot ${to + 1}`);
      kv("Severity", `${Math.round((c.severity ?? 0.5) * 100)}%`);
      const row = h("div", "btn-row");
      if (from >= 0) {
        const a = h("button", "btn small", "Before the cut");
        a.onclick = () => this.store.set({ frame: Math.max(doc.shots[from].start, doc.shots[from].end - 1), playing: false, mode: "replica" });
        row.appendChild(a);
      }
      if (to >= 0) {
        const z = h("button", "btn small", "After the cut");
        z.onclick = () => this.store.set({ frame: doc.shots[to].start, playing: false, mode: "replica" });
        row.appendChild(z);
      }
      this.inspector.appendChild(row);
    }
  }
}
