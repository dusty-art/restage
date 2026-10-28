/** Timeline: ruler, shot strip coloured by location, per-entity presence lanes, continuity markers, playhead. */
import type { Store } from "./state";
import { focalMM } from "./build";

const GUTTER = 132;
const RULER = 22;
const SHOTS = 34;
const LANE = 15;
const CONT = 18;

const SIZE_ABBR: Record<string, string> = {
  extreme_wide: "EWS", wide: "WS", medium: "MS", closeup: "CU", extreme_closeup: "ECU", insert: "INS",
};

export class Timeline {
  readonly el: HTMLDivElement;
  private canvas: HTMLCanvasElement;
  private tip: HTMLDivElement;
  private v0 = 0;
  private v1 = 1;
  private scrollY = 0;
  private dragging = false;

  constructor(host: HTMLElement, private store: Store) {
    this.el = document.createElement("div");
    this.el.className = "timeline";
    this.canvas = document.createElement("canvas");
    this.tip = document.createElement("div");
    this.tip.className = "tl-tip";
    this.el.append(this.canvas, this.tip);
    host.appendChild(this.el);
    new ResizeObserver(() => this.draw()).observe(this.el);
    store.on((keys) => {
      if (keys.includes("built")) {
        this.v0 = 0;
        this.v1 = Math.max(1, store.s.built?.doc.n_frames ?? 1);
        this.scrollY = 0;
      }
      if (keys.includes("frame") && store.s.playing) this.keepPlayheadVisible();
      this.draw();
    });
    this.canvas.addEventListener("pointerdown", (e) => this.down(e));
    this.canvas.addEventListener("pointermove", (e) => this.move(e));
    this.canvas.addEventListener("pointerleave", () => (this.tip.style.display = "none"));
    this.canvas.addEventListener("dblclick", (e) => this.dbl(e));
    this.canvas.addEventListener("wheel", (e) => this.wheel(e), { passive: false });
  }

  private get width() { return this.el.clientWidth; }
  private get height() { return this.el.clientHeight; }
  private x(f: number) { return GUTTER + ((f - this.v0) / (this.v1 - this.v0)) * (this.width - GUTTER - 8); }
  private f(x: number) { return this.v0 + ((x - GUTTER) / (this.width - GUTTER - 8)) * (this.v1 - this.v0); }

  private keepPlayheadVisible(): void {
    const f = this.store.s.frame;
    const span = this.v1 - this.v0;
    if (f < this.v0 || f > this.v1) {
      this.v0 = Math.max(0, f - span * 0.1);
      this.v1 = this.v0 + span;
    }
  }

  private lanesTop() { return RULER + SHOTS + CONT; }

  draw(): void {
    const dpr = window.devicePixelRatio || 1;
    const W = this.width;
    const H = this.height;
    if (!W || !H) return;
    this.canvas.width = W * dpr;
    this.canvas.height = H * dpr;
    this.canvas.style.width = `${W}px`;
    this.canvas.style.height = `${H}px`;
    const g = this.canvas.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = "#0f1218";
    g.fillRect(0, 0, W, H);
    const b = this.store.s.built;
    if (!b) {
      g.fillStyle = "#6b7280";
      g.font = "12px Inter, system-ui, sans-serif";
      g.fillText("No scene loaded", GUTTER, 20);
      return;
    }
    const doc = b.doc;
    const fps = doc.fps;
    g.font = "11px Inter, system-ui, sans-serif";
    g.textBaseline = "middle";

    // ruler
    g.fillStyle = "#151922";
    g.fillRect(0, 0, W, RULER);
    const span = (this.v1 - this.v0) / fps;
    const steps = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60];
    const step = steps.find((s) => (s * fps * (W - GUTTER)) / (this.v1 - this.v0) > 70) ?? 60;
    g.strokeStyle = "#2a3040";
    g.fillStyle = "#8b93a7";
    for (let t = Math.ceil(this.v0 / fps / step) * step; t <= this.v1 / fps; t += step) {
      const x = this.x(t * fps);
      g.beginPath();
      g.moveTo(x, RULER - 7);
      g.lineTo(x, RULER);
      g.stroke();
      g.fillText(span < 10 ? `${t.toFixed(step < 1 ? 2 : 1)}s` : fmtTime(t), x + 3, RULER / 2 - 1);
    }

    // shots
    const sy = RULER + 2;
    const sel = this.store.s.selection;
    doc.shots.forEach((s, i) => {
      const loc = b.locById.get(s.location);
      const x0 = Math.max(GUTTER, this.x(s.start));
      const x1 = Math.min(W - 8, this.x(s.end));
      if (x1 <= GUTTER || x0 >= W) return;
      g.fillStyle = loc?.color ?? "#888";
      g.globalAlpha = sel?.kind === "shot" && sel.index === i ? 1 : 0.78;
      roundRect(g, x0 + 1, sy, Math.max(1, x1 - x0 - 2), SHOTS - 6, 4);
      g.fill();
      g.globalAlpha = 1;
      if (s.transition && s.transition !== "cut") {
        g.fillStyle = "rgba(255,255,255,0.35)";
        g.beginPath();
        g.moveTo(x0 + 1, sy);
        g.lineTo(x0 + 12, sy);
        g.lineTo(x0 + 1, sy + SHOTS - 6);
        g.fill();
      }
      if (x1 - x0 > 44) {
        g.fillStyle = "#0b0d12";
        g.save();
        g.beginPath();
        g.rect(x0 + 4, sy, x1 - x0 - 8, SHOTS - 6);
        g.clip();
        g.font = "600 11px Inter, system-ui, sans-serif";
        g.fillText(`${i + 1} · ${SIZE_ABBR[s.size ?? ""] ?? s.size ?? ""}`, x0 + 6, sy + 9);
        g.font = "10px Inter, system-ui, sans-serif";
        g.fillText(`${s.move ?? ""}${loc ? " · " + loc.doc.name : ""}`, x0 + 6, sy + 21);
        g.restore();
      }
    });
    g.fillStyle = "#c9cfdc";
    g.font = "600 11px Inter, system-ui, sans-serif";
    g.fillText("Shots", 12, sy + (SHOTS - 6) / 2);

    // continuity markers
    const cy = RULER + SHOTS;
    g.fillStyle = "#c9cfdc";
    g.fillText("Continuity", 12, cy + CONT / 2);
    (doc.continuity ?? []).forEach((c, i) => {
      const f = c.frame ?? (c.to_shot ? doc.shots.find((s) => s.id === c.to_shot)?.start ?? 0 : 0);
      const x = this.x(f);
      if (x < GUTTER || x > W) return;
      const sev = c.severity ?? 0.5;
      g.fillStyle = sev > 0.66 ? "#ef4444" : sev > 0.33 ? "#f59e0b" : "#eab308";
      const selected = sel?.kind === "continuity" && sel.index === i;
      const r = selected ? 7 : 5.5;
      g.beginPath();
      g.moveTo(x, cy + CONT / 2 - r);
      g.lineTo(x + r, cy + CONT / 2);
      g.lineTo(x, cy + CONT / 2 + r);
      g.lineTo(x - r, cy + CONT / 2);
      g.closePath();
      g.fill();
    });

    // entity lanes
    const top = this.lanesTop();
    g.save();
    g.beginPath();
    g.rect(0, top, W, H - top);
    g.clip();
    b.entities.forEach((e, i) => {
      const y = top + i * LANE - this.scrollY;
      if (y + LANE < top || y > H) return;
      if (i % 2 === 0) {
        g.fillStyle = "#121620";
        g.fillRect(0, y, W, LANE);
      }
      const hidden = this.store.s.hiddenEntities.has(i);
      const color = `#${e.color.getHexString()}`;
      g.fillStyle = color;
      g.globalAlpha = hidden ? 0.3 : 1;
      g.beginPath();
      g.arc(14, y + LANE / 2, 4, 0, Math.PI * 2);
      g.fill();
      g.fillStyle = sel?.kind === "entity" && sel.index === i ? "#ffffff" : "#aeb5c4";
      g.font = "11px Inter, system-ui, sans-serif";
      g.fillText(clip(g, e.doc.name ?? e.doc.id, GUTTER - 30), 24, y + LANE / 2);
      const st = e.doc.track.state?.data;
      const f0 = Math.max(0, Math.floor(this.v0));
      const f1 = Math.min(doc.n_frames, Math.ceil(this.v1));
      let runStart = -1;
      let runState = 0;
      const flush = (end: number) => {
        if (runStart < 0 || runState === 0) return;
        const x0 = Math.max(GUTTER, this.x(runStart));
        const x1 = this.x(end);
        g.fillStyle = color;
        g.globalAlpha = (hidden ? 0.3 : 1) * (runState === 2 ? 0.9 : 0.28);
        g.fillRect(x0, y + 3, Math.max(1, x1 - x0), LANE - 6);
      };
      for (let f = f0; f < f1; f++) {
        const v = st ? st[f] : 2;
        if (v !== runState) {
          flush(f);
          runStart = f;
          runState = v;
        }
      }
      flush(f1);
      g.globalAlpha = 1;
    });
    g.restore();

    // cut lines through the lanes
    g.strokeStyle = "rgba(255,255,255,0.07)";
    for (const s of doc.shots) {
      const x = this.x(s.start);
      if (x <= GUTTER) continue;
      g.beginPath();
      g.moveTo(x, RULER);
      g.lineTo(x, H);
      g.stroke();
    }

    // gutter divider and playhead
    g.fillStyle = "#1b2030";
    g.fillRect(GUTTER - 1, 0, 1, H);
    const px = this.x(this.store.s.frame);
    if (px >= GUTTER && px <= W) {
      g.strokeStyle = "#f8fafc";
      g.lineWidth = 1.5;
      g.beginPath();
      g.moveTo(px, 0);
      g.lineTo(px, H);
      g.stroke();
      g.lineWidth = 1;
      g.fillStyle = "#f8fafc";
      g.beginPath();
      g.moveTo(px - 6, 0);
      g.lineTo(px + 6, 0);
      g.lineTo(px, 8);
      g.fill();
    }
  }

  private hit(e: PointerEvent | MouseEvent): { kind: "shot" | "cont" | "lane" | "ruler" | "none"; index: number; frame: number } {
    const r = this.canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    const frame = this.f(x);
    const b = this.store.s.built;
    if (!b) return { kind: "none", index: -1, frame };
    if (x < GUTTER) {
      if (y >= this.lanesTop()) return { kind: "lane", index: Math.floor((y - this.lanesTop() + this.scrollY) / LANE), frame };
      return { kind: "none", index: -1, frame };
    }
    if (y < RULER) return { kind: "ruler", index: -1, frame };
    if (y < RULER + SHOTS) {
      const i = b.doc.shots.findIndex((s) => frame >= s.start && frame < s.end);
      return { kind: "shot", index: i, frame };
    }
    if (y < RULER + SHOTS + CONT) {
      const cs = b.doc.continuity ?? [];
      let best = -1;
      let bd = 9;
      cs.forEach((c, i) => {
        const f = c.frame ?? b.doc.shots.find((s) => s.id === c.to_shot)?.start ?? 0;
        const d = Math.abs(this.x(f) - x);
        if (d < bd) { bd = d; best = i; }
      });
      return { kind: "cont", index: best, frame };
    }
    return { kind: "lane", index: Math.floor((y - this.lanesTop() + this.scrollY) / LANE), frame };
  }

  private down(e: PointerEvent): void {
    const h = this.hit(e);
    const b = this.store.s.built;
    if (!b) return;
    if (h.kind === "lane" && e.clientX - this.canvas.getBoundingClientRect().left < GUTTER) {
      if (h.index >= 0 && h.index < b.entities.length) this.store.set({ selection: { kind: "entity", index: h.index } });
      return;
    }
    if (h.kind === "cont" && h.index >= 0) {
      const c = b.doc.continuity![h.index];
      const f = c.frame ?? b.doc.shots.find((s) => s.id === c.to_shot)?.start ?? 0;
      this.store.set({ selection: { kind: "continuity", index: h.index }, frame: f, playing: false });
      return;
    }
    if (h.kind === "shot" && h.index >= 0) this.store.set({ selection: { kind: "shot", index: h.index } });
    this.dragging = true;
    this.canvas.setPointerCapture(e.pointerId);
    this.store.set({ playing: false });
    this.store.seek(Math.round(h.frame));
    const up = () => {
      this.dragging = false;
      this.canvas.removeEventListener("pointerup", up);
    };
    this.canvas.addEventListener("pointerup", up);
  }

  private move(e: PointerEvent): void {
    const h = this.hit(e);
    if (this.dragging) {
      this.store.seek(Math.round(h.frame));
      return;
    }
    const b = this.store.s.built;
    if (!b) return;
    let text = "";
    if (h.kind === "shot" && h.index >= 0) {
      const s = b.doc.shots[h.index];
      const cam = b.camByShot.get(s.id);
      const fov = cam ? cam.doc.track.fov.data[0] : 0;
      text = `<b>Shot ${h.index + 1}</b> · ${b.locById.get(s.location)?.doc.name ?? s.location}<br>${s.size ?? ""} · ${s.move ?? ""} · ${focalMM(fov).toFixed(0)} mm<br>${fmtTime(s.start / b.doc.fps)}–${fmtTime(s.end / b.doc.fps)} (${((s.end - s.start) / b.doc.fps).toFixed(2)} s)${s.transition && s.transition !== "cut" ? " · " + s.transition : ""}${s.time_skip ? `<br>time skip: ${s.time_skip.toFixed(1)} s later` : ""}`;
    } else if (h.kind === "cont" && h.index >= 0) {
      const c = b.doc.continuity![h.index];
      text = `<b>${c.kind}</b>${c.entity ? " · " + (b.doc.entities.find((x) => x.id === c.entity)?.name ?? c.entity) : ""}<br>${c.detail ?? ""}`;
    }
    if (text) {
      this.tip.innerHTML = text;
      this.tip.style.display = "block";
      const r = this.el.getBoundingClientRect();
      const x = Math.min(e.clientX - r.left + 12, r.width - 230);
      this.tip.style.left = `${x}px`;
      this.tip.style.top = `${Math.max(4, e.clientY - r.top - 64)}px`;
    } else this.tip.style.display = "none";
  }

  private dbl(e: MouseEvent): void {
    const h = this.hit(e);
    const b = this.store.s.built;
    if (!b) return;
    if (h.kind === "shot" && h.index >= 0) {
      const s = b.doc.shots[h.index];
      const pad = (s.end - s.start) * 0.15;
      this.v0 = Math.max(0, s.start - pad);
      this.v1 = Math.min(b.doc.n_frames, s.end + pad);
    } else {
      this.v0 = 0;
      this.v1 = b.doc.n_frames;
    }
    this.draw();
  }

  private wheel(e: WheelEvent): void {
    const b = this.store.s.built;
    if (!b) return;
    e.preventDefault();
    const r = this.canvas.getBoundingClientRect();
    const y = e.clientY - r.top;
    if (!e.ctrlKey && !e.metaKey && Math.abs(e.deltaY) > Math.abs(e.deltaX) && y > this.lanesTop() && !e.shiftKey) {
      const max = Math.max(0, b.entities.length * LANE - (this.height - this.lanesTop()));
      this.scrollY = Math.max(0, Math.min(max, this.scrollY + e.deltaY));
      this.draw();
      return;
    }
    const n = b.doc.n_frames;
    if (e.ctrlKey || e.metaKey || (!e.shiftKey && Math.abs(e.deltaY) > Math.abs(e.deltaX))) {
      const at = this.f(e.clientX - r.left);
      const k = Math.exp(e.deltaY * 0.002);
      let v0 = at - (at - this.v0) * k;
      let v1 = at + (this.v1 - at) * k;
      if (v1 - v0 < 15) return;
      v0 = Math.max(0, v0);
      v1 = Math.min(n, v1);
      this.v0 = v0;
      this.v1 = v1;
    } else {
      const d = ((e.deltaX || e.deltaY) / this.width) * (this.v1 - this.v0);
      const span = this.v1 - this.v0;
      this.v0 = Math.max(0, Math.min(n - span, this.v0 + d));
      this.v1 = this.v0 + span;
    }
    this.draw();
  }
}

function roundRect(g: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  r = Math.min(r, w / 2, h / 2);
  g.beginPath();
  g.moveTo(x + r, y);
  g.arcTo(x + w, y, x + w, y + h, r);
  g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r);
  g.arcTo(x, y, x + w, y, r);
  g.closePath();
}

function clip(g: CanvasRenderingContext2D, s: string, w: number): string {
  if (g.measureText(s).width <= w) return s;
  while (s.length > 1 && g.measureText(s + "…").width > w) s = s.slice(0, -1);
  return s + "…";
}

export function fmtTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${m}:${s.toFixed(2).padStart(5, "0")}`;
}
