"""Inference entry point: video file in, `.restage` replica out.

`ready()` is true once a trained checkpoint exists; until then the server reports the model as not ready
instead of producing anything. The pipeline itself is filled in by the model package.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "checkpoints" / "restage.safetensors"


def ready() -> bool:
    return CHECKPOINT.exists()


def info() -> dict:
    return {"checkpoint": CHECKPOINT.name if CHECKPOINT.exists() else None,
            "checkpoint_bytes": CHECKPOINT.stat().st_size if CHECKPOINT.exists() else 0}


def reconstruct(video: Path, out: Path, progress: Callable[[float, str], None]) -> dict:
    raise NotImplementedError("The Restage model has not been built yet.")
