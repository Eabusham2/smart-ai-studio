"""Request-scoped API memory accounting and cancellation for the loaded model."""
import logging
import math
import threading

from core.local_api import APIError
from core.memory_watchdog import SystemMemoryWatchdog

_MISSING = object()
_SAMPLE_INTERVAL = 0.25


class RequestMemoryLimit(threading.Event):
    """Keep a budget cancellation distinct from the client's disconnect event."""

    def __init__(self, engine, backend, options, client_cancel, owned_processes_provider=None):
        super().__init__()
        self.engine = engine
        self.backend = backend
        self.limit = options["limit_gb"] if options["enabled"] else None
        self.client_cancel = client_cancel
        self.owned_processes_provider = owned_processes_provider
        self.original_engine_limit = getattr(engine, "speed_memory_limit_gb", _MISSING)
        self.original_backend_limit = getattr(backend, "speed_memory_limit_gb", _MISSING)
        self.original_backend = backend
        self.error = None
        self._sample_lock = threading.Lock()
        self._stop_monitor = threading.Event()
        self._thread = None

    def is_set(self):
        return self.client_cancel.is_set() or super().is_set()

    def bind(self, backend):
        self.backend = backend
        backend.speed_memory_limit_gb = self.limit

    def start(self):
        self.engine.speed_memory_limit_gb = self.limit
        self.bind(self.backend)
        if self.limit is not None:
            self.check()
            monitor = threading.Thread(target=self._monitor, name="smartai-api-memory", daemon=True)
            monitor.start()
            self._thread = monitor

    def check(self):
        """Stop when accounting is unavailable or the observed footprint exceeds the cap."""
        if self.limit is None or self.client_cancel.is_set():
            return
        with self._sample_lock:
            if self.error is None:
                try:
                    provider = self.owned_processes_provider or getattr(self.backend, "owned_process_ids", None)
                    pids = provider() if callable(provider) else ()
                    stats = SystemMemoryWatchdog.get_detailed_memory_breakdown(owned_pids=pids)
                    usage = stats["total_allocated_bytes"] / (1024 ** 3)
                    if (not stats["process_memory_known"] or not stats["owned_memory_complete"]
                            or not math.isfinite(usage) or usage <= 0):
                        raise ValueError("Unknown Studio memory usage")
                except Exception:
                    self.error = APIError(503, "Cannot measure Studio memory for the enabled API memory limit.",
                                          "server_error", "memory_measurement_unavailable")
                else:
                    if usage > self.limit:
                        self.error = APIError(503, f"Studio and its owned workers use {usage:.2f} GiB, "
                                              f"exceeding the API memory limit of {self.limit:g} GiB.",
                                              "server_error", "memory_limit_exceeded")
            if self.error:
                self.set()
                raise self.error

    def _monitor(self):
        while not self._stop_monitor.wait(_SAMPLE_INTERVAL):
            if self.client_cancel.is_set():
                self.set()  # Wake a cooperative learning operation waiting on this event.
                return
            try:
                self.check()
            except APIError:
                backend = self.backend
                cancel = getattr(backend, "cancel_current_request", None) or getattr(backend, "cancel_generation", None)
                if callable(cancel):
                    try:
                        cancel()
                    except Exception:
                        logging.getLogger(__name__).exception("Could not interrupt the API model transport")
                return

    def checked_stream(self, stream):
        # Prefill can allocate a large cache and emit a complete tool call
        # before the next monitor tick. Check before exposing its first token.
        try:
            first = True
            for part in stream:
                if first:
                    self.check()
                    first = False
                if self.is_set():
                    return
                yield part
            self.check()
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _restore(target, value):
        if value is _MISSING:
            if "speed_memory_limit_gb" in vars(target):
                delattr(target, "speed_memory_limit_gb")
        else:
            target.speed_memory_limit_gb = value

    def close(self):
        self._stop_monitor.set()
        if self._thread is not None:
            self._thread.join()
        self._restore(self.engine, self.original_engine_limit)
        self._restore(self.original_backend, self.original_backend_limit)
        if self.backend is not self.original_backend:
            self._restore(self.backend, self.original_backend_limit)

    def restore_reloaded_backend(self, backend):
        # A learning-mode switch can replace the model twice, even when the
        # switch rolls back. Restore the desktop backend's specific cap.
        self._restore(backend, self.original_backend_limit)
