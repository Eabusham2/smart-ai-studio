"""Backend-neutral training-memory cleanup and concise RAM telemetry.

Used by app Learn/RSI/awake consolidation and non-MLX trainer bridges. This never
changes model weights, sampling, KV policy, or backend selection. It only releases
transient training state/caches after an update and reports process RSS in MB.
"""
from __future__ import annotations

import gc
import os
from typing import Any, Dict


def process_rss_mb() -> float:
    try:
        import psutil
        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 ** 2)
    except Exception:
        return 0.0


def backend_label(backend: Any) -> str:
    if backend is None:
        return "none"
    for attr in ("runtime", "active_backend"):
        value = str(getattr(backend, attr, "") or "").strip()
        if value:
            return value
    return type(backend).__name__


def release_training_memory(backend: Any = None) -> Dict[str, float]:
    """Drop transient Python/framework caches without unloading the live inference model."""
    before = process_rss_mb()

    # Backend-specific optional hook first.
    hook = getattr(backend, "release_training_memory", None)
    if callable(hook):
        try:
            hook()
        except Exception:
            pass

    gc.collect(2)

    try:
        import mlx.core as mx
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
    except Exception:
        pass

    gc.collect(2)
    after = process_rss_mb()
    return {
        "before_mb": float(before),
        "after_mb": float(after),
        "released_mb": max(0.0, float(before) - float(after)),
    }
