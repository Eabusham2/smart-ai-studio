"""Backend-neutral training-memory cleanup and concise RAM telemetry.

Used by app Learn/RSI/awake consolidation and non-MLX trainer bridges. This never
changes model weights, sampling, KV policy, or backend selection. It only releases
transient training state/caches after an update and reports process RSS in MB.
"""
from __future__ import annotations

import gc
import os
import platform
from typing import Any, Dict


def process_memory_bytes(pid: int | None = None) -> int:
    """Best-effort live process footprint.

    On macOS, proc_pid_rusage/ri_phys_footprint includes unified-memory pressure
    that ordinary RSS can miss for MLX/Metal allocations. Other platforms use
    psutil RSS. Falls back to RSS everywhere if libproc is unavailable.
    """
    target_pid = int(pid or os.getpid())
    if platform.system() == "Darwin":
        try:
            import ctypes
            import ctypes.util

            class _RUsageInfoV2(ctypes.Structure):
                _fields_ = [
                    ("ri_uuid", ctypes.c_ubyte * 16),
                    ("ri_user_time", ctypes.c_uint64),
                    ("ri_system_time", ctypes.c_uint64),
                    ("ri_pkg_idle_wkups", ctypes.c_uint64),
                    ("ri_interrupt_wkups", ctypes.c_uint64),
                    ("ri_pageins", ctypes.c_uint64),
                    ("ri_wired_size", ctypes.c_uint64),
                    ("ri_resident_size", ctypes.c_uint64),
                    ("ri_phys_footprint", ctypes.c_uint64),
                    ("ri_proc_start_abstime", ctypes.c_uint64),
                    ("ri_proc_exit_abstime", ctypes.c_uint64),
                    ("ri_child_user_time", ctypes.c_uint64),
                    ("ri_child_system_time", ctypes.c_uint64),
                    ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
                    ("ri_child_interrupt_wkups", ctypes.c_uint64),
                    ("ri_child_pageins", ctypes.c_uint64),
                    ("ri_child_elapsed_abstime", ctypes.c_uint64),
                    ("ri_diskio_bytesread", ctypes.c_uint64),
                    ("ri_diskio_byteswritten", ctypes.c_uint64),
                ]

            lib_path = ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib"
            libproc = ctypes.CDLL(lib_path, use_errno=True)
            fn = libproc.proc_pid_rusage
            fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            fn.restype = ctypes.c_int
            info = _RUsageInfoV2()
            if fn(target_pid, 2, ctypes.byref(info)) == 0 and int(info.ri_phys_footprint) > 0:
                return int(info.ri_phys_footprint)
        except Exception:
            pass

    try:
        import psutil
        return int(psutil.Process(target_pid).memory_info().rss)
    except Exception:
        return 0


def process_rss_mb() -> float:
    try:
        return float(process_memory_bytes()) / (1024.0 ** 2)
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
