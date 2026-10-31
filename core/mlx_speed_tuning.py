"""Measured MLX prefill tuning using the desktop's existing cache policy."""
from __future__ import annotations

import hashlib
import importlib
from importlib import metadata
import inspect
import json
from pathlib import Path
import time

from core.native_speed import (
    BENCHMARK_PROMPT, BENCHMARK_TOKENS, BENCHMARK_VERSION,
    tune_budget_seconds, tuning_memory_ok,
)
from core.speed_tuning import (
    SpeedProfileStore, SpeedResult, SpeedSample, SpeedTuningStopped,
    speed_fingerprint, tune_speed,
)


def _outcome(status, message, settings=None):
    return SpeedResult(status, dict(settings or {}), message).to_dict()


def _requested_context(backend):
    return int(getattr(backend, "context_budget_tokens", 0) or 32768)


def _source_identity(value):
    """Identify the loaded specialized implementation without reading weights."""
    module = value if inspect.ismodule(value) else inspect.getmodule(value)
    path = getattr(module, "__file__", None)
    identity = {"module": getattr(module, "__name__", None),
                "callable": getattr(value, "__qualname__", None)}
    if path and Path(path).is_file():
        source = Path(path)
        stat = source.stat()
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            digest.update(handle.read(1024 * 1024))
            if stat.st_size > 1024 * 1024:
                handle.seek(max(0, stat.st_size - 1024 * 1024))
                digest.update(handle.read(1024 * 1024))
        identity.update(path=str(source.resolve()), bytes=stat.st_size,
                        mtime_ns=stat.st_mtime_ns, sha256=digest.hexdigest())
    return identity


def _configure_vlm_speed(backend, *, force, progress, cancelled):
    """Tune the existing native mlx-vlm text stream, including Bonsai and JANG.

    Baseline calls omit prefill_step_size when the app had no measured override.
    mlx-vlm therefore owns both its original default and fresh native cache.
    Vision weights remain loaded; this path never injects MLX-LM/TurboQuant caches.
    """
    from core.mlx_engine import _adaptive_prefill_step_size, _model_context_limit
    import mlx.core as mx

    previous = getattr(backend, "_speed_prefill_step_size", None)
    previous_context = getattr(backend, "_speed_profile_context", None)
    requested_context = _requested_context(backend)
    existing_override = previous if previous_context == requested_context else None
    generation = importlib.import_module("mlx_vlm.generate")
    stream = getattr(generation, "stream_generate", None)
    # Both the older single-file API and current dispatch/ar package expose this
    # name. Prefer the actual callable referenced by the loaded stream function.
    step = getattr(stream, "__globals__", {}).get("generate_step", getattr(generation, "generate_step", None))
    try:
        stream_parameters = inspect.signature(stream).parameters
        parameter = stream_parameters.get("prefill_step_size")
        if parameter is None:
            parameter = inspect.signature(step).parameters["prefill_step_size"]
        default_chunk = parameter.default
        forwards_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in stream_parameters.values())
        if type(default_chunk) is not int or not 1 <= default_chunk <= 8192 or not callable(stream):
            raise ValueError("unknown native prefill default")
        if "prefill_step_size" not in stream_parameters and not forwards_kwargs:
            raise ValueError("stream cannot accept prefill controls")
    except (ValueError, TypeError, KeyError):
        return _outcome("unsupported", "This mlx-vlm version does not expose a verifiable native prefill default.")
    for target in (backend.model, getattr(backend.model, "language_model", None)):
        if target is not None and getattr(target, "no_chunked_prefill", False):
            return _outcome("unsupported", "This model disables chunked prefill; its native execution is retained.")

    baseline_chunk = int(existing_override or default_chunk)
    baseline = {"prefill_step_size": baseline_chunk}
    memory_ok = lambda: tuning_memory_ok(getattr(backend, "speed_memory_limit_gb", None))
    if cancelled and cancelled():
        return _outcome("cancelled", "Speed tuning cancelled; previous settings retained.", baseline)
    if not memory_ok() or baseline_chunk > _adaptive_prefill_step_size():
        return _outcome("memory_pressure", "Tuning deferred: not enough headroom to measure the model's unchanged native baseline.", baseline)

    context = requested_context
    physical_context = _model_context_limit(backend.model, backend.tokenizer)
    if physical_context:
        context = min(context, physical_context)
    prompt_budget = min(4096, context - BENCHMARK_TOKENS - 8)
    if prompt_budget < 32:
        return _outcome("unsupported", "The model context is too small for a speed comparison.", baseline)
    raw_prompt = BENCHMARK_PROMPT
    for _ in range(12):
        prompt = backend._prepare_vlm_prompt(raw_prompt, 0)
        estimated_tokens = len(backend.tokenizer.encode(prompt))
        if 0 < estimated_tokens <= prompt_budget:
            break
        raw_prompt = raw_prompt[:max(1, len(raw_prompt) // 2)]
    else:
        return _outcome("unsupported", "The native processor cannot fit a bounded synthetic speed prompt.", baseline)
    safe_chunk = min(default_chunk, _adaptive_prefill_step_size())
    candidates = [{"prefill_step_size": value} for value in sorted({safe_chunk, max(128, safe_chunk // 2)}, reverse=True)
                  if value != baseline_chunk and (value < estimated_tokens or baseline_chunk < estimated_tokens)]
    allowed = [baseline, *candidates]
    loader = getattr(backend, "_vlm_loader", None)
    versions = {}
    for package in ("mlx-vlm", "jang", "jang-tools"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unknown"
    config = getattr(loader, "config", getattr(backend.model, "config", None))
    if callable(getattr(config, "to_dict", None)):
        config = config.to_dict()
    elif hasattr(config, "__dict__"):
        config = vars(config)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, default=lambda v: str(type(v)),
                                           allow_nan=False).encode()).hexdigest()
    device_info = {}
    for owner in (mx, getattr(mx, "metal", None)):
        probe = getattr(owner, "device_info", None)
        if callable(probe):
            try:
                device_info = {key: value for key, value in probe().items()
                               if key in {"device_name", "architecture", "memory_size", "max_recommended_working_set_size"}}
                break
            except Exception:
                pass
    fingerprint = speed_fingerprint(
        str(backend._vlm_runtime), backend.model_path, context=context,
        adapter_path=getattr(backend, "adapter_path", None),
        extra={"workload": BENCHMARK_VERSION, "cache_policy": "mlx-vlm-native-fresh-default",
               "native_prefill_default": default_chunk, "prompt_tokens": estimated_tokens,
               "device": device_info,
               "model_info": getattr(backend, "model_info", {}), "versions": versions,
               "stream_source": _source_identity(stream), "step_source": _source_identity(step),
               "loader_source": _source_identity(getattr(loader, "runtime_module", type(loader))),
               "loader_config_sha256": config_hash},
    )
    store = SpeedProfileStore()
    saved = None if force else store.get(fingerprint)

    def apply(settings):
        # Keep omission/default behavior when the measured native baseline wins.
        value = settings["prefill_step_size"]
        backend._speed_prefill_step_size = None if value == default_chunk else value
        backend._speed_profile_context = requested_context if value != default_chunk else None

    if saved and saved.get("settings") in allowed:
        if cancelled and cancelled():
            return _outcome("cancelled", "Speed tuning cancelled; previous settings retained.", baseline)
        if not memory_ok() or saved["settings"]["prefill_step_size"] > _adaptive_prefill_step_size():
            return _outcome("memory_pressure", "Saved profile deferred because available memory changed.", baseline)
        apply(saved["settings"])
        return {**saved, "cached": True, "message": "Saved measured native VLM speed profile applied. " + saved["message"]}

    budget = tune_budget_seconds()
    deadline = time.monotonic() + budget
    rng = getattr(getattr(mx, "random", None), "state", None)
    saved_rng = list(rng) if isinstance(rng, list) else None
    if saved_rng:
        mx.eval(*saved_rng)
    missing = object()
    request_state = []
    for owner, names in ((backend.tokenizer, ("thinking_budget_criteria", "stopping_criteria")),
                         (backend.model, ("_position_ids", "_rope_deltas")),
                         (getattr(backend.model, "language_model", None), ("_position_ids", "_rope_deltas"))):
        if owner is not None:
            request_state.extend((owner, name, getattr(owner, name, missing)) for name in names)

    def restore_request_state():
        for owner, name, value in request_state:
            if value is missing:
                if hasattr(owner, name):
                    delattr(owner, name)
            else:
                setattr(owner, name, value)

    def check():
        if cancelled and cancelled():
            raise SpeedTuningStopped("cancelled", "Speed tuning cancelled; previous settings retained.")
        if time.monotonic() >= deadline:
            raise SpeedTuningStopped("timeout", "Tuning time budget reached; previous settings retained.")
        if not memory_ok():
            raise SpeedTuningStopped("memory_pressure", "Tuning stopped for memory headroom; previous settings retained.")

    def measure(settings):
        check()
        if settings["prefill_step_size"] > _adaptive_prefill_step_size():
            raise SpeedTuningStopped("memory_pressure", "Available memory changed; retaining previous settings.")
        if saved_rng is not None:
            rng[:] = saved_rng
        restore_request_state()
        kwargs = {"prompt": prompt, "image": [], "max_tokens": BENCHMARK_TOKENS,
                  "temperature": 0.0, "top_p": 1.0}
        if settings["prefill_step_size"] != default_chunk:
            kwargs["prefill_step_size"] = settings["prefill_step_size"]
        iterator = None
        tokens = []
        actual_prompt_tokens = None
        first = None
        mx.synchronize()
        started = time.perf_counter()
        try:
            iterator = stream(backend.model, backend.processor, **kwargs)
            for response in iterator:
                check()
                total = getattr(response, "generation_tokens", None)
                prompt_count = getattr(response, "prompt_tokens", None)
                if type(total) is not int or type(prompt_count) is not int or prompt_count < 1:
                    raise RuntimeError("The VLM runtime must expose exact prompt and generation token counts")
                if prompt_count + BENCHMARK_TOKENS > context:
                    raise RuntimeError("The native processor exceeded the bounded calibration context")
                if actual_prompt_tokens is None:
                    actual_prompt_tokens = prompt_count
                elif actual_prompt_tokens != prompt_count:
                    raise RuntimeError("The native processor changed prompt counts within one measurement")
                token = getattr(response, "token", None)
                if total == len(tokens) + 1 and token is not None:
                    tokens.append(int(token))
                    if first is None:
                        first = time.perf_counter()
                elif total != len(tokens) or (tokens and token is not None and int(token) != tokens[-1]):
                    raise RuntimeError("The VLM stream did not expose consecutive exact token IDs")
                # mlx-vlm emits a final text flush with the last token again.
                # Count each generation index once, including an EOS token first
                # reported by that final flush. Check the complete final ID list.
                complete_ids = getattr(response, "token_ids", None)
                if complete_ids is not None and [int(t) for t in complete_ids] != tokens:
                    raise RuntimeError("The VLM final token list disagrees with its streamed token IDs")
            mx.synchronize()
            ended = time.perf_counter()
            if first is None or len(tokens) > BENCHMARK_TOKENS:
                raise RuntimeError("The VLM calibration produced no bounded output")
            return SpeedSample(actual_prompt_tokens, len(tokens), first - started, ended - first,
                               hashlib.sha256(json.dumps(tokens).encode("ascii")).hexdigest())
        finally:
            try:
                if iterator is not None and callable(getattr(iterator, "close", None)):
                    iterator.close()
            finally:
                # Closing the native generator releases its own fresh KV arena.
                restore_request_state()

    completed = False
    try:
        result = tune_speed(baseline, candidates, measure, cancelled=cancelled, progress=progress,
                            budget_seconds=budget, memory_ok=memory_ok)
        result.fingerprint = fingerprint
        if result.status in {"improved", "baseline"}:
            apply(result.settings)
            completed = True
            persisted = store.put(fingerprint, result)
            return {**result.to_dict(), "persisted": persisted}
        return result.to_dict()
    finally:
        if not completed:
            backend._speed_prefill_step_size = previous
            backend._speed_profile_context = previous_context
        restore_request_state()
        if saved_rng is not None:
            rng[:] = saved_rng


def configure_mlx_speed(backend, enabled=True, force=False, progress=None, cancelled=None):
    """Calibrate prompt chunk size, with no speculative/quality setting changes.

    The caller owns the backend runtime lock. Each run builds a private prompt
    cache with the same factory used by chat; no benchmark enters chat or Learn.
    A preferred chunk is always capped by the existing live-memory policy.
    """
    from core.mlx_engine import _adaptive_prefill_step_size, _model_context_limit

    previous = getattr(backend, "_speed_prefill_step_size", None)
    previous_context = getattr(backend, "_speed_profile_context", None)
    requested_context = _requested_context(backend)
    if not enabled:
        backend._speed_prefill_step_size = None
        backend._speed_profile_context = None
        return _outcome("disabled", "Auto speed off; using the existing adaptive prefill settings.")
    if not getattr(backend, "is_mlx_available", False) or getattr(backend, "model", None) is None:
        return _outcome("idle", "Load a model to configure speed.")
    if getattr(backend, "_vlm_runtime", None) is not None:
        return _configure_vlm_speed(backend, force=force, progress=progress, cancelled=cancelled)

    safe_chunk = _adaptive_prefill_step_size()
    valid_previous = previous if previous_context == requested_context else None
    baseline_chunk = min(int(valid_previous or safe_chunk), safe_chunk)
    baseline = {"prefill_step_size": baseline_chunk}
    candidates = [
        {"prefill_step_size": n}
        for n in sorted({max(128, safe_chunk // 2), safe_chunk}, reverse=True)
        if n != baseline_chunk
    ]
    memory_ok = lambda: tuning_memory_ok(getattr(backend, "speed_memory_limit_gb", None))
    if not memory_ok():
        return _outcome("memory_pressure", "Speed tuning deferred: the system needs more free memory.", baseline)
    if cancelled and cancelled():
        return _outcome("cancelled", "Speed tuning cancelled; previous settings retained.", baseline)

    import mlx.core as mx
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler
    from core.kv_cache_manager import SmartKVCacheManager

    context = int(getattr(backend, "context_budget_tokens", 0) or 32768)
    physical_context = _model_context_limit(backend.model, backend.tokenizer)
    if physical_context:
        context = min(context, physical_context)
    prompt_tokens = list(backend.tokenizer.encode(BENCHMARK_PROMPT))
    # This shortens synthetic calibration only, never a user's context.
    prompt_tokens = prompt_tokens[:max(1, min(4096, context - BENCHMARK_TOKENS))]
    if context <= BENCHMARK_TOKENS or not prompt_tokens:
        return _outcome("unsupported", "The model context is too small for a speed comparison.", baseline)
    # A chunk at or above the synthetic prompt length has no measurable effect.
    candidates = [c for c in candidates if c["prefill_step_size"] < len(prompt_tokens)
                  or baseline_chunk < len(prompt_tokens)]

    cache = getattr(backend, "kv_cache_manager", None)
    cache_types = [type(c).__module__ + "." + type(c).__name__ for c in cache.get_cache()] if cache else []
    device_info = {}
    for owner in (mx, getattr(mx, "metal", None)):
        probe = getattr(owner, "device_info", None)
        if callable(probe):
            try:
                device_info = {key: value for key, value in probe().items()
                               if key in {"device_name", "architecture", "memory_size", "max_recommended_working_set_size"}}
                break
            except Exception:
                pass
    fingerprint = speed_fingerprint(
        "mlx", backend.model_path, context=context,
        adapter_path=getattr(backend, "adapter_path", None),
        extra={"workload": BENCHMARK_VERSION, "cache_types": cache_types,
               "cache_policy": "desktop-existing-turboquant-k8-v3-with-native-fallback",
               "prompt_tokens": len(prompt_tokens),
               "model_info": getattr(backend, "model_info", {}), "device": device_info},
    )
    store = SpeedProfileStore()
    saved = None if force else store.get(fingerprint)
    allowed = [baseline] + candidates
    if saved and saved.get("settings") in allowed:
        if cancelled and cancelled():
            return _outcome("cancelled", "Speed tuning cancelled; previous settings retained.", baseline)
        if not memory_ok() or saved["settings"]["prefill_step_size"] > _adaptive_prefill_step_size():
            return _outcome("memory_pressure", "Saved profile deferred because available memory changed.", baseline)
        backend._speed_prefill_step_size = int(saved["settings"]["prefill_step_size"])
        backend._speed_profile_context = requested_context
        return {**saved, "cached": True, "message": "Saved measured speed profile applied. " + saved["message"]}

    budget = tune_budget_seconds()
    deadline = time.monotonic() + budget
    rng = getattr(getattr(mx, "random", None), "state", None)
    saved_rng = list(rng) if isinstance(rng, list) else None
    if saved_rng:
        mx.eval(*saved_rng)

    def check():
        if cancelled and cancelled():
            raise SpeedTuningStopped("cancelled", "Speed tuning cancelled; previous settings retained.")
        if time.monotonic() >= deadline:
            raise SpeedTuningStopped("timeout", "Tuning time budget reached; previous settings retained.")
        if not memory_ok():
            raise SpeedTuningStopped("memory_pressure", "Tuning stopped for memory headroom; previous settings retained.")

    def measure(settings):
        check()
        if int(settings["prefill_step_size"]) > _adaptive_prefill_step_size():
            raise SpeedTuningStopped("memory_pressure", "Available memory changed; retaining previous settings.")
        if saved_rng is not None:
            rng[:] = saved_rng
        arena = SmartKVCacheManager(backend.model, max_tokens=backend.max_stateful_kv_tokens)
        iterator = None
        tokens = []
        first = None
        mx.synchronize()
        started = time.perf_counter()
        try:
            iterator = mlx_lm.stream_generate(
                backend.model, backend.tokenizer,
                prompt=prompt_tokens, max_tokens=BENCHMARK_TOKENS,
                sampler=make_sampler(temp=0.0, top_p=1.0),
                prefill_step_size=int(settings["prefill_step_size"]),
                prompt_cache=arena.get_cache(),
            )
            for response in iterator:
                token = getattr(response, "token", None)
                total = getattr(response, "generation_tokens", None)
                prompt_count = getattr(response, "prompt_tokens", None)
                if token is None or type(total) is not int or type(prompt_count) is not int:
                    raise RuntimeError("This MLX runtime cannot report exact calibration token IDs and counts")
                if prompt_count != len(prompt_tokens) or not 1 <= total <= BENCHMARK_TOKENS:
                    raise RuntimeError("MLX calibration token counts did not match the bounded workload")
                if total == len(tokens) + 1:
                    tokens.append(int(token))
                    if first is None:
                        first = time.perf_counter()
                elif total != len(tokens) or int(token) != tokens[-1]:
                    raise RuntimeError("The MLX stream did not expose consecutive exact token IDs")
                # MLX-LM normally yields its last token only at finalization.
                # An older/custom generator that exhausts before the requested
                # cap may repeat its last token in the final text flush.
                # Count indices handle both cases without inflating throughput.
                check()
            mx.synchronize()
            ended = time.perf_counter()
            if first is None:
                raise RuntimeError("The calibration produced no tokens")
            return SpeedSample(
                len(prompt_tokens), len(tokens), first - started, ended - first,
                hashlib.sha256(json.dumps(tokens).encode("ascii")).hexdigest(),
            )
        finally:
            if iterator is not None and callable(getattr(iterator, "close", None)):
                iterator.close()
            # Only the synthetic arena is released; the app's own cache is untouched.
            arena.cache = None
            del arena

    completed = False
    try:
        result = tune_speed(
            baseline, candidates, measure, cancelled=cancelled, progress=progress,
            budget_seconds=budget, memory_ok=memory_ok,
        )
        result.fingerprint = fingerprint
        if result.status in {"improved", "baseline"}:
            backend._speed_prefill_step_size = result.settings["prefill_step_size"]
            backend._speed_profile_context = requested_context
            completed = True
            persisted = store.put(fingerprint, result)
            record = result.to_dict()
            record["persisted"] = persisted
            return record
        return result.to_dict()
    finally:
        if not completed:
            backend._speed_prefill_step_size = previous
            backend._speed_profile_context = previous_context
        if saved_rng is not None:
            rng[:] = saved_rng
