"""Request-time MLX headroom; independent of the optional Studio watchdog.

Host, process and Metal counters describe overlapping memory. Admission uses
the smallest remaining budget, never their sum. These are conservative batching
and optional-cache decisions, not a promise that a given model/context will fit.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect


GIB = 1024 ** 3


@dataclass(frozen=True)
class MemoryHeadroom:
    available_bytes: int | None
    process_bytes: int | None
    active_bytes: int | None
    cache_bytes: int
    device_limit_bytes: int | None
    reserve_bytes: int
    usable_bytes: int | None

    @property
    def pressure(self):
        return self.usable_bytes is None or self.usable_bytes < GIB

    def can_snapshot(self, size):
        # The active source already exists. Allow both its additional copy and
        # subsequent cache growth, leaving decode scratch beyond the OS reserve.
        return (self.usable_bytes is not None
                and self.usable_bytes >= 2 * max(0, int(size)) + GIB)

    @property
    def prefill_tokens(self):
        usable = self.usable_bytes
        if usable is None:
            return 128
        for minimum, chunk in ((6 * GIB, 2048), (3 * GIB, 1024),
                               (int(1.5 * GIB), 512), (GIB // 2, 256)):
            if usable >= minimum:
                return chunk
        return 128


def memory_headroom(*, mx=None, max_process_gb=None):
    """Read current independent bounds, reserving RAM for macOS and scratch."""
    available = total = process = active = device_limit = None
    cached = 0
    try:
        import psutil
        vm = psutil.virtual_memory()
        available, total = int(vm.available), int(vm.total)
    except (ImportError, OSError, AttributeError, ValueError):
        pass
    try:
        from core.training_memory import process_memory_bytes
        measured = int(process_memory_bytes())
        process = measured if measured > 0 else None
    except (OSError, ValueError, TypeError):
        pass
    if mx is None:
        try:
            import mlx.core as mx
        except ImportError:
            mx = None
    if mx is not None:
        try:
            active = max(0, int(mx.get_active_memory()))
            cached = max(0, int(mx.get_cache_memory()))
        except (AttributeError, RuntimeError, ValueError):
            pass
        try:
            if mx.metal.is_available():
                device_info = getattr(mx, "device_info", None) or getattr(mx.metal, "device_info", None)
                measured = int(device_info()["max_recommended_working_set_size"])
                device_limit = measured if measured > 0 else None
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            pass
    reserve = max(GIB, min(2 * GIB, int((total or 10 * GIB) * .10)))
    bounds = []
    if available is not None:
        bounds.append(available - reserve)
    if device_limit is not None and active is not None:
        bounds.append(device_limit - active - cached - reserve)
    if max_process_gb is not None and process is not None:
        try:
            limit = int(float(max_process_gb) * GIB)
            if limit > 0:
                bounds.append(limit - process - reserve)
        except (TypeError, ValueError, OverflowError):
            pass
    return MemoryHeadroom(available, process, active, cached, device_limit,
                          reserve, max(0, min(bounds)) if bounds else None)


def native_vlm_prefill_default():
    """Recognize the native prefill control before overriding its batch size."""
    try:
        generation = importlib.import_module("mlx_vlm.generate")
        stream = generation.stream_generate
        step = getattr(stream, "__globals__", {}).get("generate_step", getattr(generation, "generate_step", None))
        params = inspect.signature(stream).parameters
        value = params.get("prefill_step_size")
        if value is None:
            if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                return None
            value = inspect.signature(step).parameters["prefill_step_size"]
        default = value.default
        return default if type(default) is int and 1 <= default <= 8192 else None
    except (ImportError, AttributeError, KeyError, TypeError, ValueError):
        return None


def trim_allocator(mx, *, force=False, max_process_gb=None):
    """Bound optional allocator cache on the caller's MLX owner thread."""
    headroom = memory_headroom(mx=mx, max_process_gb=max_process_gb)
    allowance = (0 if force or headroom.pressure else
                 min(256 * 1024 ** 2, max(0, int(headroom.usable_bytes or 0) // 8)))
    setter = getattr(mx, "set_cache_limit", None)
    if callable(setter):
        try:
            setter(allowance)
        except (RuntimeError, ValueError):
            # CPU-only/older builds may expose an unsupported allocator API.
            pass
    if force or headroom.cache_bytes > allowance:
        clear = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None)
        if callable(clear):
            try:
                clear()
            except RuntimeError:
                # A poisoned Metal command buffer can raise again on cleanup;
                # preserve the original allocation error for the caller.
                pass
    return headroom
