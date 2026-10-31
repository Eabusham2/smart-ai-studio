"""Measure conservative thread settings on an already loaded llama.cpp model.

The normal and isolated Prism runtimes share this adapter.  It never reloads
weights, changes their precision, edits prompts, or changes the context budget.
Only the native decode/prefill thread counts are eligible for measurement.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Optional

from core.speed_tuning import (
    SpeedProfileStore, SpeedResult, SpeedSample, SpeedTuningStopped,
    physical_cpu_count, speed_fingerprint, tune_speed,
)


def _threads(model: Any) -> dict[str, int]:
    values = {key: getattr(model, key, None) for key in ("n_threads", "n_threads_batch")}
    if any(type(value) is not int or not 1 <= value <= 1024 for value in values.values()):
        raise ValueError("The runtime does not expose its active thread counts.")
    return values


def thread_candidates(baseline: dict[str, int], cores: int,
                      limits: Optional[dict[str, int]] = None) -> list[dict[str, int]]:
    """One knob per candidate, never more threads than the loaded baseline.

    Physical-core and half-core options catch oversubscribed/hybrid CPUs without
    changing user-selected GPU placement or increasing the CPU reservation.
    """
    candidates: list[dict[str, int]] = []
    cores = max(1, int(cores))
    limits = limits or baseline
    for knob in ("n_threads", "n_threads_batch"):
        ceiling = min(limits[knob], cores)
        for value in (limits[knob], ceiling, max(1, ceiling // 2)):
            candidate = {**baseline, knob: value}
            if candidate != baseline and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _sampler_alive(sampler: Any) -> bool:
    """A sampler already closed by a runtime must never be restored."""
    if sampler is None or getattr(sampler, "closed", False) is True:
        return False
    sentinel = object()
    native = getattr(sampler, "sampler", sentinel)
    return native is sentinel or bool(native)


def _native_setter(model: Any) -> Optional[Callable[[int, int], None]]:
    # llama-cpp-python's context wrapper delegates to the documented
    # llama_set_n_threads(ctx, n_threads, n_threads_batch) native operation.
    context = getattr(model, "_ctx", None)
    setter = getattr(context, "set_n_threads", None)
    if callable(setter):
        return setter
    binding = sys.modules.get("llama_cpp.llama_cpp") or sys.modules.get("llama_cpp")
    setter = getattr(binding, "llama_set_n_threads", None)
    native_context = getattr(context, "ctx", None)
    if callable(setter) and native_context is not None:
        return lambda threads, batch: setter(native_context, threads, batch)
    return None


def _runtime_identity(model: Any) -> dict[str, Any]:
    binding = sys.modules.get("llama_cpp.llama_cpp") or sys.modules.get("llama_cpp")
    library = getattr(getattr(binding, "_lib", None), "_name", None)
    identity: dict[str, Any] = {"library": str(library) if library else None}
    if library and Path(library).is_file():
        path = Path(library)
        stat = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            digest.update(handle.read(256 * 1024))
            handle.seek(max(0, stat.st_size - 256 * 1024))
            digest.update(handle.read(256 * 1024))
        identity.update(bytes=stat.st_size, modified_ns=stat.st_mtime_ns,
                        sample_sha256=digest.hexdigest())
    params = getattr(model, "context_params", None)
    for name in ("n_batch", "n_ubatch", "type_k", "type_v", "flash_attn_type", "offload_kqv"):
        value = getattr(params, name, getattr(model, name, None))
        if isinstance(value, (str, bool, int, float)) or value is None:
            identity[name] = value
    identity["draft_model"] = type(getattr(model, "draft_model", None)).__qualname__
    identity["logits_all"] = bool(getattr(model, "_logits_all", False))
    return identity


class LlamaSpeedTuner:
    """One loaded-model identity and its original execution settings.

    Create this immediately after model construction, before any calibration.
    Cached profiles are matched to the loaded files, not to a newer adapter that
    happens to have been written to disk since the model was loaded.
    The owner serializes this object with its model's generation/training lock.
    """

    def __init__(self, model: Any, *, backend: str, model_path: str,
                 context: int, adapter_path: Optional[str] = None,
                 projector_path: Optional[str] = None, n_gpu_layers: int = -1,
                 store: Optional[SpeedProfileStore] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.model = model
        self.backend = backend
        self.clock = clock
        self.store = store
        self.initial: dict[str, int] = {}
        self.fingerprint = ""
        self.unavailable = ""
        self.setter = _native_setter(model)
        try:
            from core.native_speed import BENCHMARK_VERSION
            self.initial = _threads(model)
            actual_context = model.n_ctx() if callable(getattr(model, "n_ctx", None)) else context
            self.fingerprint = speed_fingerprint(
                backend, model_path, context=int(actual_context),
                adapter_path=adapter_path, projector_path=projector_path,
                extra={"benchmark": BENCHMARK_VERSION, "initial_threads": self.initial,
                       "n_gpu_layers": n_gpu_layers, "runtime": _runtime_identity(model)},
            )
        except Exception as exc:
            self.unavailable = str(exc)[:250]

    def _result(self, status: str, message: str,
                settings: Optional[dict[str, int]] = None) -> dict[str, Any]:
        return SpeedResult(status, dict(settings or self.initial), message,
                           fingerprint=self.fingerprint).to_dict()

    def _apply(self, settings: dict[str, int]) -> None:
        if self.setter is None:
            raise RuntimeError("This llama.cpp runtime has no live thread setter.")
        self.setter(settings["n_threads"], settings["n_threads_batch"])
        self.model.n_threads = settings["n_threads"]
        self.model.n_threads_batch = settings["n_threads_batch"]

    def _measure(self, settings: dict[str, int], *, tokens: list[int], count: int,
                 check: Callable[[], None]) -> SpeedSample:
        check()
        stream = None
        generated: list[int] = []
        first_at = None
        old_seed = self.model._seed
        old_sampler = getattr(self.model, "_sampler", None)
        absent = object()
        old_mirostat = getattr(self.model, "_mirostat_mu", absent)
        # Some binding versions/forks close the attached sampler when replacing
        # it. Detach the user's chain before reset/generate so its RNG and native
        # pointer remain untouched by the synthetic greedy requests.
        self.model._sampler = None
        try:
            self._apply(settings)
            self.model.reset()  # remove prefix reuse before every timed request
            self.model.set_seed(0)
            check()
            started = self.clock()
            stream = self.model.generate(tokens, temp=0.0, top_k=1, top_p=1.0,
                                         repeat_penalty=1.0, reset=True)
            for _ in range(count):
                check()
                token = next(stream)
                now = self.clock()
                check()
                generated.append(int(token))
                if first_at is None:
                    first_at = now
            finished = self.clock()
            if first_at is None or len(generated) < 2:
                raise RuntimeError("The benchmark did not generate enough tokens.")
            digest = hashlib.sha256(json.dumps(generated, separators=(",", ":")).encode()).hexdigest()
            return SpeedSample(len(tokens), len(generated), first_at - started,
                               finished - first_at, digest)
        finally:
            try:
                if stream is not None and callable(getattr(stream, "close", None)):
                    stream.close()
            finally:
                temporary = getattr(self.model, "_sampler", None)
                self.model._sampler = None
                try:
                    close = getattr(temporary, "close", None)
                    if temporary is not old_sampler and _sampler_alive(temporary) and callable(close):
                        close()
                finally:
                    self.model.set_seed(old_seed)
                    self.model.reset()
                    if old_mirostat is absent:
                        if hasattr(self.model, "_mirostat_mu"):
                            delattr(self.model, "_mirostat_mu")
                    else:
                        self.model._mirostat_mu = old_mirostat
                    self.model._sampler = old_sampler if _sampler_alive(old_sampler) else None

    def configure(self, enabled: bool = True, force: bool = False,
                  progress: Optional[Callable[[str], None]] = None,
                  cancelled: Optional[Callable[[], bool]] = None,
                  *, max_process_gb: Optional[float] = None,
                  memory_owner_pid: Optional[int] = None,
                  allow_cache_reset: bool = False) -> dict[str, Any]:
        from core.native_speed import BENCHMARK_PROMPT, BENCHMARK_TOKENS, tune_budget_seconds, tuning_memory_ok

        def memory_ok() -> bool:
            if memory_owner_pid is None:
                return tuning_memory_ok(max_process_gb)
            return tuning_memory_ok(max_process_gb, owner_pid=memory_owner_pid)

        if self.unavailable or not self.initial or self.setter is None:
            return self._result("unsupported", "Speed tuning unavailable: " +
                                (self.unavailable or "this runtime has no live native thread setter."))
        previous = _threads(self.model)
        if not enabled:
            try:
                self._apply(self.initial)
                return self._result("disabled", "Automatic speed tuning is off; original thread settings restored.")
            except Exception as exc:
                return self._result("error", f"Could not restore thread settings: {str(exc)[:250]}", previous)
        if cancelled is not None and cancelled():
            return self._result("cancelled", "Calibration cancelled; working settings retained.", previous)
        if not memory_ok():
            return self._result("memory_pressure", "Memory headroom is too low to tune; working settings retained.", previous)

        cores = physical_cpu_count()
        original_candidates = thread_candidates(self.initial, cores)
        permitted = {key: {settings[key] for settings in [self.initial, *original_candidates]}
                     for key in self.initial}

        def allowed(settings: Any) -> bool:
            # A previous successful retune may combine two separately measured
            # knobs. Keep that configuration cacheable, while each new trial
            # still changes just one knob of the currently working selection.
            return (isinstance(settings, dict) and set(settings) == set(permitted)
                    and all(type(value) is int and value in permitted[key]
                            for key, value in settings.items()))

        candidates = [candidate for candidate in thread_candidates(previous, cores, limits=self.initial)
                      if allowed(candidate)]
        store = self.store or SpeedProfileStore()
        if not force:
            cached = store.get(self.fingerprint)
            if cached and allowed(cached.get("settings")):
                if cancelled is not None and cancelled():
                    return self._result("cancelled", "Calibration cancelled; working settings retained.", previous)
                if not memory_ok():
                    return self._result("memory_pressure", "Saved profile deferred because memory headroom changed.", previous)
                try:
                    self._apply(cached["settings"])
                    return {**cached, "cached": True,
                            "message": "Restored the measured speed settings for this model and hardware."}
                except Exception as exc:
                    self._apply(previous)
                    return self._result("error", f"Saved speed settings could not be applied: {str(exc)[:250]}", previous)

        # A KV snapshot can be as large as the model itself. Do not duplicate it
        # or discard a user's warm conversation cache just to run a benchmark.
        if getattr(self.model, "n_tokens", None) != 0 and not allow_cache_reset:
            return self._result("deferred", "Reload the model to retune without clearing its active prompt cache.", previous)
        if not all(callable(getattr(self.model, name, None)) for name in ("generate", "tokenize", "reset", "set_seed")) or not hasattr(self.model, "_seed"):
            return self._result("unsupported", "This runtime cannot run isolated deterministic speed samples.", previous)

        deadline = self.clock() + tune_budget_seconds()

        def check() -> None:
            if cancelled is not None and cancelled():
                raise SpeedTuningStopped("cancelled", "Calibration cancelled; working settings retained.")
            if self.clock() >= deadline:
                raise SpeedTuningStopped("timeout", "Calibration budget expired; working settings retained.")
            if not memory_ok():
                raise SpeedTuningStopped("memory_pressure", "Memory headroom fell; working settings retained.")

        result: Optional[SpeedResult] = None
        try:
            check()
            tokens = list(self.model.tokenize(BENCHMARK_PROMPT.encode("utf-8")))
            capacity = int(self.model.n_ctx()) - int(BENCHMARK_TOKENS) - 1
            if capacity < 16:
                return self._result("unsupported", "The loaded context is too small for a bounded speed sample.", previous)
            tokens = tokens[:capacity]
            if not tokens:
                raise RuntimeError("The benchmark prompt produced no tokens.")
            result = tune_speed(
                dict(previous), candidates,
                lambda settings: self._measure(settings, tokens=tokens, count=BENCHMARK_TOKENS, check=check),
                cancelled=cancelled, progress=progress,
                budget_seconds=max(0.001, deadline - self.clock()), clock=self.clock,
                memory_ok=memory_ok,
            )
        except SpeedTuningStopped as exc:
            return self._result(exc.status, str(exc), previous)
        except Exception as exc:
            return self._result("error", f"Calibration failed ({type(exc).__name__}: {str(exc)[:250]}); working settings retained.", previous)
        finally:
            # Success is applied only after every temporary trial has been undone.
            self._apply(previous)

        result.fingerprint = self.fingerprint
        if result.status in {"improved", "baseline"} and (allowed(result.settings) or result.settings == previous):
            try:
                check()
                self._apply(result.settings)
            except SpeedTuningStopped as exc:
                return self._result(exc.status, str(exc), previous)
            except Exception as exc:
                self._apply(previous)
                return self._result("error", f"Measured settings could not be applied: {str(exc)[:250]}", previous)
            stored = store.put(self.fingerprint, result) if allowed(result.settings) else False
            return {**result.to_dict(), "profile_saved": bool(stored), "cached": False}
        # Stops and failed sweeps never overwrite a previously working selection.
        result.settings = dict(previous)
        return result.to_dict()
