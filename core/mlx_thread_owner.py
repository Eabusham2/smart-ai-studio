"""One persistent owner thread for an Apple-Silicon MLX model's entire lifetime.

MLX >= 0.31.2 registers GPU streams per OS thread. A GUI must not load weights
on one worker and evaluate them (or close a lazy KV cache) on another. This
lane owns model loading, inference, tuning, training, generator close and
unload. Only decoded Python text crosses the lane's queue.
"""
from __future__ import annotations

import contextvars
import functools
import importlib
import inspect
import queue
import threading
from typing import Any


_STOP = object()
_END = object()


def _bind_mlx_generation_streams():
    """Bind MLX-LM/VLM import-time generation streams to this owner thread.

    Current mlx-lm and mlx-vlm allocate ``generation_stream`` at module import.
    GUI settings may import them on Tk's thread first. Merely moving the model
    load is insufficient unless those global handles are recreated here.
    """
    try:
        import mlx.core as mx
    except ImportError:
        # Model-free CI and non-Apple backends have no Metal runtime.
        return

    # mlx-lm's cache module may already have imported generate on the Tk thread.
    # Import it here if available; this is the only normal model execution lane.
    modules = []
    try:
        modules.append(importlib.import_module("mlx_lm.generate"))
    except ModuleNotFoundError as exc:
        if not str(exc.name or "").startswith("mlx_lm"):
            raise
    # mlx-vlm may have been imported by startup initialization too. Never
    # import it eagerly for a plain MLX-LM checkpoint.
    import sys
    vlm = sys.modules.get("mlx_vlm.generate")
    if vlm is not None:
        modules.append(vlm)
    factory = getattr(mx, "new_thread_local_stream", None)
    if not callable(factory):
        factory = getattr(mx, "new_stream", None)
    for module in modules:
        if hasattr(module, "generation_stream"):
            if not callable(factory):
                raise RuntimeError("MLX generation-stream factory unavailable")
            module.generation_stream = factory(mx.default_device())


class MLXThreadOwner:
    """Serialized work queue, with reentrant and cancellation-aware streaming."""

    def __init__(self):
        self._tasks: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._started_error: BaseException | None = None
        self._closing = False
        self._thread_id: int | None = None
        self._thread = threading.Thread(target=self._run, name="smartai-mlx-owner", daemon=True)
        self._thread.start()
        if not self._ready.wait(20):
            raise RuntimeError("MLX owner thread did not initialize")
        if self._started_error is not None:
            raise self._started_error

    def is_owner(self):
        return threading.get_ident() == self._thread_id

    def _run(self):
        self._thread_id = threading.get_ident()
        try:
            _bind_mlx_generation_streams()
        except BaseException as exc:
            self._started_error = exc
            self._ready.set()
            return
        self._ready.set()
        while True:
            task = self._tasks.get()
            if task is _STOP:
                break
            fn, result, done = task
            try:
                result.append((True, fn()))
            except BaseException as exc:
                result.append((False, exc))
            finally:
                done.set()
            del task, fn, result, done

    def _submit(self, fn):
        if self._closing:
            raise RuntimeError("MLX execution owner has already begun unloading")
        result: list = []
        done = threading.Event()
        # GUI and solver record exact tokens through ContextVar observers.
        # Preserve that context across the MLX lane so moving GPU work does not
        # silently break TPS, generated-token counts or Pro aggregate metrics.
        context = contextvars.copy_context()
        self._tasks.put((lambda: context.run(fn), result, done))
        return result, done

    def call(self, fn, *args, **kwargs):
        if self.is_owner():
            return fn(*args, **kwargs)
        result, done = self._submit(lambda: fn(*args, **kwargs))
        done.wait()
        success, value = result[0]
        if not success:
            raise value
        return value

    def stream(self, fn, *args, **kwargs):
        if self.is_owner():
            # Nested use on the owner must not queue behind itself.
            yield from fn(*args, **kwargs)
            return

        # No unseen token prefetch: native completion must only happen after the
        # UI has actually consumed the preceding token. Otherwise cancelling
        # after the first visible piece can incorrectly retain a finished KV.
        output: queue.Queue = queue.Queue(maxsize=1)
        stopped = threading.Event()
        finished = threading.Event()
        advance = threading.Event()

        def publish(kind, payload=None):
            while not stopped.is_set():
                try:
                    output.put((kind, payload), timeout=0.05)
                    return
                except queue.Full:
                    pass

        def execute():
            iterator = None
            try:
                iterator = iter(fn(*args, **kwargs))
                for item in iterator:
                    if stopped.is_set():
                        break
                    publish("item", item)
                    # Wait until the UI explicitly requests its next token.
                    # Cancellation wakes this wait so native generator.close()
                    # executes on the original Metal owner thread.
                    while not stopped.is_set() and not advance.wait(0.05):
                        pass
                    advance.clear()
                    if stopped.is_set():
                        break
            except BaseException as exc:
                publish("error", exc)
            finally:
                # Close the live MLX generator on its OWN thread, including
                # Metal cache operations inside generator finally blocks.
                if iterator is not None:
                    try:
                        close = getattr(iterator, "close", None)
                        if callable(close):
                            close()
                    except BaseException as exc:
                        publish("error", exc)
                finished.set()
                publish("end", _END)

        self._submit(execute)
        try:
            while True:
                kind, payload = output.get()
                if kind == "end":
                    return
                if kind == "error":
                    raise payload
                yield payload
                advance.set()
        finally:
            # A closed UI stream must never leave a blocked producer queue or
            # close its native generator from Tk's thread.
            stopped.set()
            advance.set()
            finished.wait(timeout=2.0)

    def shutdown(self):
        if self._closing:
            return
        self._closing = True
        self._tasks.put(_STOP)
        if not self.is_owner():
            self._thread.join(timeout=15)
            if self._thread.is_alive():
                raise RuntimeError("MLX model owner thread did not finish unloading")


def install_mlx_thread_ownership(cls):
    """Route every model-bound method through the same persistent OS thread."""
    if cls.__dict__.get("_mlx_thread_ownership_installed", False):
        return

    def owner_for(self):
        lock = getattr(self, "_mlx_owner_init_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._mlx_owner_init_lock = lock
        with lock:
            owner = getattr(self, "_mlx_owner", None)
            if owner is None:
                owner = MLXThreadOwner()
                self._mlx_owner = owner
            return owner

    # Include external training, media, and telemetry paths. Model internals
    # called from these methods execute inline when already on the owner.
    for name in (
        "load_model", "unload_model", "calculate_token_entropy", "generate_branches",
        "stream_generate_tokens", "configure_auto_speed", "configure_lossless_speedups",
        "invalidate_prompt_reuse", "inject_lora_adapters", "compute_mlx_fisher",
        "train_mini_batch", "review_media_input", "get_memory_breakdown",
        "prompt_reuse_telemetry", "_prepare_vlm_prompt", "get_training_model",
    ):
        original = getattr(cls, name, None)
        if not callable(original):
            continue
        if inspect.isgeneratorfunction(original):
            def wrap_stream(method):
                @functools.wraps(method)
                def owned(self, *args, **kwargs):
                    yield from owner_for(self).stream(lambda: method(self, *args, **kwargs))
                return owned
            replacement = wrap_stream(original)
        else:
            def wrap_method(method, method_name):
                @functools.wraps(method)
                def owned(self, *args, **kwargs):
                    owner = owner_for(self)
                    result = owner.call(method, self, *args, **kwargs)
                    if method_name == "unload_model":
                        owner.shutdown()
                        if getattr(self, "_mlx_owner", None) is owner:
                            self._mlx_owner = None
                    return result
                return owned
            replacement = wrap_method(original, name)
        setattr(cls, name, replacement)
    cls._mlx_thread_ownership_installed = True
