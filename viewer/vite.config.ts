import { defineConfig, type Plugin } from "vite";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SAMPLES = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../samples");

/** Serves ../samples at /samples, with /samples/index.json listing the scenes and their videos. */
function samples(): Plugin {
  return {
    name: "restage-samples",
    configureServer(server) {
      server.middlewares.use("/samples", (req, res, next) => {
        const url = decodeURIComponent((req.url ?? "/").split("?")[0]);
        if (url === "/" || url === "/index.json") {
          const files = fs.existsSync(SAMPLES) ? fs.readdirSync(SAMPLES) : [];
          const scenes = files
            .filter((f) => f.endsWith(".restage"))
            .sort()
            .map((f) => {
              const base = f.slice(0, -".restage".length);
              const video = files.find((v) => v === `${base}.mp4` || v === `${base}.webm`);
              const st = fs.statSync(path.join(SAMPLES, f));
              return { name: base, scene: `/samples/${f}`, video: video ? `/samples/${video}` : null, bytes: st.size, mtime: st.mtimeMs };
            });
          res.setHeader("Content-Type", "application/json");
          res.setHeader("Cache-Control", "no-store");
          res.end(JSON.stringify(scenes));
          return;
        }
        const file = path.join(SAMPLES, path.normalize(url));
        if (!file.startsWith(SAMPLES) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) return next();
        const ext = path.extname(file);
        const type = ext === ".mp4" ? "video/mp4" : ext === ".webm" ? "video/webm" : "application/octet-stream";
        const size = fs.statSync(file).size;
        // Range support so <video> can seek.
        const range = req.headers.range;
        res.setHeader("Accept-Ranges", "bytes");
        res.setHeader("Content-Type", type);
        res.setHeader("Cache-Control", "no-store");
        if (range) {
          const [s, e] = range.replace("bytes=", "").split("-");
          const start = Number(s);
          const end = e ? Number(e) : size - 1;
          res.statusCode = 206;
          res.setHeader("Content-Range", `bytes ${start}-${end}/${size}`);
          res.setHeader("Content-Length", String(end - start + 1));
          fs.createReadStream(file, { start, end }).pipe(res);
        } else {
          res.setHeader("Content-Length", String(size));
          fs.createReadStream(file).pipe(res);
        }
      });
    },
  };
}

/** Dev-only: POST /__save?name=<file> writes the body to ../.cache/viewer_exports (used by automated export tests). */
function devSave(): Plugin {
  const dir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../.cache/viewer_exports");
  return {
    name: "restage-dev-save",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use("/__save", (req, res) => {
        const name = path.basename(new URL(req.url ?? "", "http://x").searchParams.get("name") ?? "out.bin");
        fs.mkdirSync(dir, { recursive: true });
        const chunks: Buffer[] = [];
        req.on("data", (c: Buffer) => chunks.push(c));
        req.on("end", () => {
          fs.writeFileSync(path.join(dir, name), Buffer.concat(chunks));
          res.end(JSON.stringify({ saved: path.join(dir, name), bytes: Buffer.concat(chunks).length }));
        });
      });
    },
  };
}

export default defineConfig({
  plugins: [samples(), devSave()],
  server: { port: 5178, strictPort: true, host: "127.0.0.1" },
  build: { target: "es2022", chunkSizeWarningLimit: 2000 },
});
