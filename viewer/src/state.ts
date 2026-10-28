/** App state with change notifications. Every panel reads from and writes to one Store. */
import type { Built, Style } from "./build";

export type Mode = "replica" | "explore" | "map" | "compare";
export type CompareKind = "wipe" | "side" | "onion";
export type Selection =
  | { kind: "entity"; index: number }
  | { kind: "shot"; index: number }
  | { kind: "location"; index: number }
  | { kind: "continuity"; index: number }
  | null;

export interface State {
  built: Built | null;
  sceneName: string;
  videoUrl: string | null;
  frame: number;            // fractional while playing
  playing: boolean;
  speed: number;
  loop: boolean;
  mode: Mode;
  style: Style;
  selection: Selection;
  exploreLocation: number;  // location shown in explore/map when not following the edit
  followEdit: boolean;      // explore/map switch location with the edit
  followEntity: number | null;
  showCameras: boolean;
  showTrails: boolean;
  showInferred: boolean;
  showGrid: boolean;
  shadows: boolean;
  compareKind: CompareKind;
  wipe: number;             // 0..1, share of the frame showing the source video
  hiddenEntities: Set<number>;
}

export class Store extends EventTarget {
  s: State = {
    built: null,
    sceneName: "",
    videoUrl: null,
    frame: 0,
    playing: false,
    speed: 1,
    loop: true,
    mode: "replica",
    style: "shaded",
    selection: null,
    exploreLocation: 0,
    followEdit: true,
    followEntity: null,
    showCameras: true,
    showTrails: true,
    showInferred: true,
    showGrid: true,
    shadows: true,
    compareKind: "wipe",
    wipe: 0.5,
    hiddenEntities: new Set(),
  };

  set(patch: Partial<State>): void {
    const keys = Object.keys(patch) as (keyof State)[];
    Object.assign(this.s, patch);
    this.dispatchEvent(new CustomEvent("change", { detail: keys }));
  }

  on(fn: (keys: (keyof State)[]) => void): void {
    this.addEventListener("change", (e) => fn((e as CustomEvent).detail));
  }

  /** Index of the shot playing at the current frame, or -1. */
  shotIndex(): number {
    const b = this.s.built;
    if (!b) return -1;
    const f = Math.max(0, Math.min(b.doc.n_frames - 1, Math.floor(this.s.frame)));
    return b.shotAt[f];
  }

  seek(frame: number): void {
    const b = this.s.built;
    if (!b) return;
    const n = b.doc.n_frames;
    this.set({ frame: Math.max(0, Math.min(n - 1, frame)) });
  }
}
