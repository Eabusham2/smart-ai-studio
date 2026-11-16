"""Optional deterministic GPU runtime priming without executing a model.

There is deliberately no timer or background worker. Explicit load/request
boundaries may prime a tiny GPU kernel after an idle interval, and callers can
refresh the idle timestamp after useful inference. This avoids competing with
inference, retaining unloaded models, or consuming sampling RNG. It does not
claim to hold GPU clocks up indefinitely or precompile model-specific kernels.
"""
from __future__ import annotations

import math
import threading
import time


class SafeMLXWarmup:
    def __init__(self, *, enabled=False, idle_seconds=30.0, clock=time.monotonic):
        self.enabled = bool(enabled)
        interval = float(idle_seconds)
        self.idle_seconds = interval if math.isfinite(interval) and interval >= 0 else 30.0
        self._clock = clock
        self._last_activity = None
        self._closed = False
        self._lock = threading.RLock()
        self.runs = 0
        self.skipped = 0
        self.failures = 0

    def prepare(self, *, mx=None, runtime_lock=None, memory_ok=None):
        """Run at most one tiny deterministic GPU operation if idle and unlocked.

        ``runtime_lock`` should be the backend's existing reentrant model lock.
        Acquisition is nonblocking: a busy runtime is never queued behind. No
        model, tokenizer, cache, stream or random-state object is accepted/stored.
        Tests inject a fake array module; production imports MLX only when enabled.
        """
        with self._lock:
            if not self.enabled or self._closed:
                return False
            now = self._clock()
            if self._last_activity is not None and now - self._last_activity < self.idle_seconds:
                self.skipped += 1
                return False
            acquired = runtime_lock is None or runtime_lock.acquire(blocking=False)
            if not acquired:
                self.skipped += 1
                return False
            try:
                if memory_ok is not None and not memory_ok():
                    self.skipped += 1
                    return False
                if mx is None:
                    import mlx.core as mx
                # Select GPU explicitly without modifying MLX's default device,
                # random state, allocator policy or the application's live stream.
                gpu = getattr(mx, "gpu", None)
                stream_context = getattr(mx, "stream", None)
                if gpu is None or not callable(stream_context):
                    self.skipped += 1
                    return False
                metal = getattr(mx, "metal", None)
                available = getattr(metal, "is_available", None)
                if callable(available) and not available():
                    self.skipped += 1
                    return False
                with stream_context(gpu):
                    seed = mx.array([0.0, 1.0, 2.0, 3.0], dtype=mx.float32)
                    result = seed + seed
                    mx.eval(result)
                self._last_activity = self._clock()
                self.runs += 1
                return True
            except Exception:
                self.failures += 1
                # Avoid retrying a failing optional kernel on every request.
                self._last_activity = self._clock()
                return False
            finally:
                if runtime_lock is not None:
                    runtime_lock.release()

    def touch(self):
        """Record useful runtime activity; stores no inference state."""
        with self._lock:
            if not self._closed:
                self._last_activity = self._clock()

    def close(self):
        """Synchronous unload boundary: no worker, array or model survives it."""
        with self._lock:
            self._closed = True
            self.enabled = False
            self._last_activity = None

    def telemetry(self):
        return {"kernel_warmup_runs": self.runs, "kernel_warmup_skipped": self.skipped,
                "kernel_warmup_failures": self.failures, "kernel_warmup_background": False}
