"""Shared boundaries for native, Strata-inspired execution tuning.

Only a fixed synthetic workload enters calibration. Chat history, learned data,
model selection, sampling preferences and cache precision are not tuning inputs.
"""
from __future__ import annotations

import functools
import inspect
import math
import os
import threading


BENCHMARK_VERSION = "studio-speed-v1"
BENCHMARK_TOKENS = 24
BENCHMARK_PROMPT = (
    "Read this synthetic inventory. Continue the numbered explanation after the inventory, "
    "describing how to validate, sort, and total the records. Write at least ten steps.\n"
    + "".join(
        f"record_{i:03d}: category={('alpha', 'beta', 'gamma')[i % 3]}; "
        f"quantity={i % 11 + 1}; cost={i % 17 + 2}; state=ready.\n"
        for i in range(128)
    )
    + "\nValidation and calculation procedure:\n1. Read each inventory record and"
)


def tune_budget_seconds() -> float:
    try:
        value = float(os.getenv("SMARTAI_SPEED_TUNE_SECONDS", "90"))
        if math.isfinite(value):
            return max(15.0, min(600.0, value))
    except (TypeError, ValueError):
        pass
    return 90.0


def tuning_memory_ok(max_process_gb: float | None = None, owner_pid: int | None = None) -> bool:
    """Check live headroom; unknown memory is not permission to benchmark."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        if vm.available < max(1024 ** 3, int(vm.total * 0.05)):
            return False
        if max_process_gb is not None and float(max_process_gb) > 0:
            proc = psutil.Process(owner_pid) if owner_pid is not None else psutil.Process()
            used = proc.memory_info().rss
            # Isolated Prism/BitNet workers are part of the application's budget.
            for child in proc.children(recursive=True):
                try:
                    used += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            if used + 256 * 1024 ** 2 >= float(max_process_gb) * 1024 ** 3:
                return False
        return True
    except Exception:
        return False


def runtime_lock(backend):
    lock = getattr(backend, "_runtime_model_lock", None)
    if lock is None:
        lock = threading.RLock()
        backend._runtime_model_lock = lock
    return lock


def install_native_runtime_lock(cls) -> None:
    """Share a lock between native inference, training, unloading and tuning.

    MLX already has its runtime lock. GGUF and its isolated Prism worker need
    the same exclusion so a tuning benchmark cannot interleave with a request.
    """
    if cls.__dict__.get("_native_speed_lock_installed", False):
        return

    for name in (
        "load_model", "unload_model", "calculate_token_entropy",
        "generate_branches", "stream_generate_tokens", "review_media_input",
        "train_mini_batch", "configure_auto_speed",
    ):
        original = getattr(cls, name, None)
        if not callable(original):
            continue

        if inspect.isgeneratorfunction(original):
            def wrap_generator(method):
                @functools.wraps(method)
                def locked(self, *args, **kwargs):
                    with runtime_lock(self):
                        yield from method(self, *args, **kwargs)
                return locked
            replacement = wrap_generator(original)
        else:
            def wrap_method(method):
                @functools.wraps(method)
                def locked(self, *args, **kwargs):
                    with runtime_lock(self):
                        return method(self, *args, **kwargs)
                return locked
            replacement = wrap_method(original)
        setattr(cls, name, replacement)
    cls._native_speed_lock_installed = True
