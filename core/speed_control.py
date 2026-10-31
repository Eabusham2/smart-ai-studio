"""Orchestrate optional native speed tuning at safe model lifecycle boundaries."""
from __future__ import annotations

import functools
import os
import threading

from core.native_speed import runtime_lock


def install_speed_control(cls) -> None:
    if getattr(cls, "_speed_control_installed", False):
        return
    original_init = cls.__init__
    original_load = cls.load_model
    original_unload = cls.unload_model

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # load_model calls unload_model: their lifecycle lock must be reentrant.
        self._model_lock = threading.RLock()
        self.auto_speed_enabled = os.getenv("SMARTAI_AUTO_SPEED", "1").lower() not in {"0", "false", "off", "no"}
        self.speed_tuning_active = False
        self.speed_status = {"status": "idle", "message": "Load a model to configure speed."}
        self._speed_cancel_event = threading.Event()
        self._speed_loading_thread = None

    def active_native(self):
        controller = getattr(self, "controller_backend", None)
        if getattr(self, "active_backend", None) == "controller" and getattr(controller, "runtime", None) == "strata":
            return controller
        return {
            "mlx": getattr(self, "mlx_backend", None),
            "gguf": getattr(self, "gguf_backend", None),
            "bitnet": getattr(self, "bitnet_backend", None),
        }.get(getattr(self, "active_backend", None))

    def cancel(self):
        self._speed_cancel_event.set()

    def configure_locked(self, enabled=True, force=False, progress=None, cancelled=None, *, reset_cancel=True):
        backend = active_native(self)
        if backend is None or getattr(backend, "model", None) is None:
            self.auto_speed_enabled = bool(enabled)
            status = "disabled" if not enabled else ("unsupported" if getattr(self, "active_backend", None) else "idle")
            self.speed_status = {"status": status, "message": (
                "Auto speed off." if not enabled else
                "This runtime keeps its native execution settings." if status == "unsupported" else
                "Load a model to configure speed."
            )}
            return dict(self.speed_status)
        hook = getattr(backend, "configure_auto_speed", None)
        if not callable(hook):
            self.auto_speed_enabled = bool(enabled)
            self.speed_status = {"status": "unsupported" if enabled else "disabled", "message": "This runtime keeps its native execution settings."}
            return dict(self.speed_status)
        # A cancel/unload can arrive while native weights are still loading.
        # Automatic calibration belongs to that same load operation; it must
        # not erase the request when the loader reaches its tuning phase.
        if not reset_cancel and self._speed_cancel_event.is_set():
            self.speed_status = {
                "status": "cancelled",
                "message": "Speed tuning cancelled before calibration; execution settings retained.",
            }
            return dict(self.speed_status)
        lock = runtime_lock(backend)
        if not lock.acquire(blocking=False):
            return {"status": "busy", "message": "Wait for inference or learning to finish before tuning."}
        previous_enabled = self.auto_speed_enabled
        try:
            self.auto_speed_enabled = bool(enabled)
            if reset_cancel:
                self._speed_cancel_event.clear()
            self.speed_tuning_active = True
            backend.speed_memory_limit_gb = getattr(self, "speed_memory_limit_gb", None)
            if getattr(self, "_context_budget_tokens", 0):
                backend.context_budget_tokens = int(self._context_budget_tokens)

            def is_cancelled():
                return self._speed_cancel_event.is_set() or bool(cancelled and cancelled())

            def update(message):
                self.speed_status = {"status": "tuning", "message": str(message)}
                if progress:
                    progress(str(message))

            update("Comparing native execution settings…" if enabled else "Restoring native defaults…")
            result = hook(enabled=bool(enabled), force=bool(force), progress=update, cancelled=is_cancelled)
            if result.get("status") in {"busy", "cancelled", "timeout", "memory_pressure", "error", "failed"}:
                self.auto_speed_enabled = previous_enabled
            self.speed_status = dict(result)
            return dict(result)
        except Exception as exc:
            # Tuning is optional; it must not turn a successful model load into failure.
            self.auto_speed_enabled = previous_enabled
            self.speed_status = {"status": "error", "message": f"Speed tuning unavailable; previous settings retained. {exc}"}
            return dict(self.speed_status)
        finally:
            self.speed_tuning_active = False
            lock.release()

    def configure(self, enabled=True, force=False, progress=None, cancelled=None):
        if not self._model_lock.acquire(blocking=False):
            return {"status": "busy", "message": "Wait for the current model operation to finish."}
        try:
            return configure_locked(self, enabled, force, progress, cancelled)
        finally:
            self._model_lock.release()

    @functools.wraps(original_load)
    def load(self, *args, **kwargs):
        with self._model_lock:
            self._speed_cancel_event.clear()
            previous_loading_thread = self._speed_loading_thread
            self._speed_loading_thread = threading.get_ident()
            try:
                result = original_load(self, *args, **kwargs)
                if isinstance(result, dict) and result.get("status") == "loaded" and not getattr(self.settings, "use_mock", False):
                    result = dict(result)
                    result["speed"] = configure_locked(self, self.auto_speed_enabled, reset_cancel=False)
                return result
            finally:
                self._speed_loading_thread = previous_loading_thread

    @functools.wraps(original_unload)
    def unload(self, *args, **kwargs):
        # The loader's own same-thread teardown is setup for its next model.
        # External unload/watchdog requests must still cancel before waiting.
        if self._speed_loading_thread != threading.get_ident():
            cancel(self)
        controller = getattr(self, "controller_backend", None)
        if getattr(controller, "runtime", None) == "strata":
            # A silent native prefill can hold its HTTP read for minutes. Close
            # that transport before waiting for any load/configuration lock.
            controller.cancel_current_request()
        with self._model_lock:
            backend = active_native(self)
            if backend is not None:
                with runtime_lock(backend):
                    result = original_unload(self, *args, **kwargs)
            else:
                result = original_unload(self, *args, **kwargs)
            if isinstance(result, dict) and result.get("status") == "error":
                self.speed_status = {"status": "error", "message": result.get("error", "Model unload failed.")}
            else:
                self.speed_status = {"status": "idle", "message": "Load a model to configure speed."}
            return result

    cls.__init__ = initialize
    cls.load_model = load
    cls.unload_model = unload
    cls.configure_auto_speed = configure
    cls.cancel_speed_tuning = cancel
    cls._speed_control_installed = True
