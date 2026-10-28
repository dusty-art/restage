"""Local reconstruction server: upload a video, get a `.restage` replica.

    python -m restage.server          ->  http://127.0.0.1:5179  (the viewer's dev server proxies /api here)

Jobs run one at a time in a worker thread (the model wants the whole GPU). Results are written to
samples/<name>.restage with the source video beside it, so the viewer lists them like any other scene.
"""
from __future__ import annotations

import re
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
UPLOADS = ROOT / ".cache" / "uploads"
MAX_BYTES = 400 * 1024 * 1024
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}


@dataclass
class Job:
    id: str
    name: str
    video: Path
    state: str = "queued"          # queued running done error
    progress: float = 0.0
    stage: str = "waiting"
    error: str | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None
    stats: dict = field(default_factory=dict)

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "state": self.state, "progress": round(self.progress, 4),
                "stage": self.stage, "error": self.error, "stats": self.stats,
                "elapsed": round((self.finished or time.time()) - self.created, 2),
                "scene": f"/samples/{self.name}.restage" if self.state == "done" else None,
                "video": f"/samples/{self.name}{self.video.suffix}" if self.state == "done" else None}


app = FastAPI(title="Restage")
JOBS: dict[str, Job] = {}
QUEUE: Queue[Job] = Queue()


def _reconstructor():
    """The model entry point, or None while no trained checkpoint exists."""
    try:
        from restage import infer
    except ImportError:
        return None
    return infer.reconstruct if infer.ready() else None


def _worker():
    while True:
        job = QUEUE.get()
        fn = _reconstructor()
        job.state = "running"
        try:
            if fn is None:
                raise RuntimeError("No trained Restage model yet (checkpoints/ is empty).")

            def progress(p: float, stage: str):
                job.progress = max(0.0, min(1.0, p))
                job.stage = stage

            SAMPLES.mkdir(exist_ok=True)
            out = SAMPLES / f"{job.name}.restage"
            job.stats = fn(job.video, out, progress) or {}
            shutil.copy(job.video, SAMPLES / f"{job.name}{job.video.suffix}")
            job.state, job.progress, job.stage = "done", 1.0, "done"
        except Exception as e:  # report every failure to the client
            traceback.print_exc()
            job.state, job.error, job.stage = "error", str(e), "failed"
        finally:
            job.finished = time.time()
            QUEUE.task_done()


threading.Thread(target=_worker, daemon=True).start()


@app.get("/api/health")
def health():
    from restage import infer
    return {"model_ready": _reconstructor() is not None, **infer.info()}


@app.post("/api/reconstruct")
async def reconstruct(video: UploadFile = File(...)):
    ext = Path(video.filename or "").suffix.lower()
    if ext not in VIDEO_EXT:
        raise HTTPException(400, f"Unsupported file type {ext or '(none)'}; send a video ({', '.join(sorted(VIDEO_EXT))}).")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    jid = uuid.uuid4().hex[:10]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(video.filename or "video").stem).strip("_")[:40] or "video"
    path = UPLOADS / f"{jid}{ext}"
    size = 0
    with open(path, "wb") as f:
        while chunk := await video.read(1 << 20):
            size += len(chunk)
            if size > MAX_BYTES:
                f.close()
                path.unlink(missing_ok=True)
                raise HTTPException(413, "Video is larger than 400 MB.")
            f.write(chunk)
    job = Job(jid, f"{stem}_{jid[:4]}", path)
    JOBS[jid] = job
    QUEUE.put(job)
    return JSONResponse(job.public(), status_code=202)


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    job = JOBS.get(jid)
    if not job:
        raise HTTPException(404, "No such job")
    return job.public()


@app.get("/api/jobs")
def jobs():
    return [j.public() for j in sorted(JOBS.values(), key=lambda j: -j.created)]


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5179, log_level="warning")


if __name__ == "__main__":
    main()
