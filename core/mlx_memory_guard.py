"""Request-time MLX headroom; independent of the optional Studio watchdog.

Host, process and Metal counters describe overlapping memory. Admission uses
the smallest remaining budget, never their sum. These are conservative batching
and optional-cache decisions, not a promise that a given model/context will fit.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
import platform
import threading
import weakref


GIB = 1024 ** 3
MAX_ALLOCATOR_CACHE_BYTES = 256 * 1024 ** 2
_CACHE_LIMIT_LOCK = threading.Lock()
_CACHE_LIMITS = weakref.WeakKeyDictionary()


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
    """Read current independent bounds, reserving host and device RAM for runtime scratch."""
    available = total = process = active = device_limit = device_free = None
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
    if mx is not None and platform.system() != "Darwin":
        try:
            if mx.default_device() == mx.gpu and not mx.metal.is_available():
                info = mx.device_info()
                if "free_memory" in info:
                    device_free = max(0, int(info["free_memory"]))
                if "total_memory" in info:
                    device_limit = int(info["total_memory"])
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            pass
    reserve = max(GIB, min(2 * GIB, int((total or 10 * GIB) * .10)))
    bounds = []
    if available is not None:
        bounds.append(available - reserve)
    if device_limit is not None and active is not None:
        bounds.append(device_limit - active - cached - reserve)
    if device_free is not None:
        bounds.append(device_free - reserve)
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


def native_last_logits_kwargs(model):
    """Request only the final vocabulary projection when the runtime supports it.

    This keeps the complete prompt and decoder forward unchanged. Older models
    with no native control retain their original call. Current mlx-vlm models
    forward the control through **kwargs to ``project_logits``. vMLX's Qwen3.5
    implementation advertises its different control on the callable itself.
    Recognize those contracts from metadata, including in pyc-only release
    bundles, without treating arbitrary **kwargs as support.
    """
    try:
        forward = model.__call__
        parameters = inspect.signature(forward).parameters
        for name in ("logits_to_keep", "num_logits_to_keep"):
            parameter = parameters.get(name)
            if parameter is not None and parameter.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                return {name: 1}
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            return {}
        implementation = getattr(forward, "__func__", forward)
        if getattr(implementation, "_vmlx_prefill_last_logits_only", False) is True:
            return {"prefill_last_logits_only": True}
        code = getattr(implementation, "__code__", None)
        projection = getattr(implementation, "__globals__", {}).get("project_logits")
        if (str(getattr(implementation, "__module__", "")).startswith("mlx_vlm.models.")
                and code is not None and "logits_to_keep" in code.co_consts
                and "project_logits" in code.co_names and callable(projection)
                and getattr(projection, "__module__", None) == "mlx_vlm.models.base"):
            return {"logits_to_keep": 1}
    except (AttributeError, TypeError, ValueError):
        pass
    return {}


def set_allocator_cache_limit(mx, allowance, *, only_reduce=False):
    """Serialize request/monitor policy so a watchdog cannot raise a live cap.

    MLX exposes the previous cap through its setter, but has no cache-limit
    getter. Remember our successful writes per runtime so normal monitor ticks
    do not disturb its allocator. An unknown cap is queried once by setting
    zero and restoring at most the prior allowance. The shared lock prevents
    an owner-thread OOM recovery from interleaving those writes.
    """
    setter = getattr(mx, "set_cache_limit", None) or getattr(getattr(mx, "metal", None), "set_cache_limit", None)
    if not callable(setter):
        return
    allowance = min(MAX_ALLOCATOR_CACHE_BYTES, max(0, int(allowance)))
    with _CACHE_LIMIT_LOCK:
        try:
            previous = _CACHE_LIMITS.get(mx)
        except TypeError:
            previous = None
        if only_reduce and previous is not None:
            allowance = min(allowance, previous)
            if allowance == previous:
                return
        try:
            if only_reduce and previous is None and allowance:
                previous = setter(0)
                # Older/custom runtimes without the documented return value
                # stay at zero instead of accidentally raising an unknown cap.
                allowance = min(allowance, previous) if type(previous) is int and previous > 0 else 0
                if allowance:
                    setter(allowance)
            else:
                setter(allowance)
            try:
                _CACHE_LIMITS[mx] = allowance
            except TypeError:
                pass
        except (RuntimeError, ValueError):
            # CPU-only/older builds may expose an unsupported allocator API.
            pass


def trim_allocator(mx, *, force=False, max_process_gb=None):
    """Bound optional allocator cache on the caller's MLX owner thread."""
    headroom = memory_headroom(mx=mx, max_process_gb=max_process_gb)
    allowance = (0 if force or headroom.pressure else
                 min(MAX_ALLOCATOR_CACHE_BYTES, max(0, int(headroom.usable_bytes or 0) // 8)))
    set_allocator_cache_limit(mx, allowance)
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
