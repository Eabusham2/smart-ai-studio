"""Model-free preferences and lifecycle configuration for the nine speedups."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from core.native_speed import runtime_lock, uses_mlx_thread_owner


DEFAULT_OPTIONS = {
    "enabled": True, "prompt_reuse": True, "branch_prefill": True,
    "tokenizer_memo": True, "persistent_cache": False, "metal_warmup": False,
    "cache_dir": None, "speculative_mode": "auto", "max_draft_tokens": None,
    "speculative_server_path": None, "draft_model_path": None,
    "speculative_runtime_receipt_path": None,
    "native_branch_count": 1,
}
_BOOL_KEYS = {key for key, value in DEFAULT_OPTIONS.items() if type(value) is bool}
_PATH_KEYS = {"cache_dir", "speculative_server_path", "draft_model_path", "speculative_runtime_receipt_path"}


def normalize_options(options=None, *, base=None):
    options = dict(options or {})
    unknown = set(options) - set(DEFAULT_OPTIONS)
    if unknown:
        raise ValueError("Unknown speed option: " + ", ".join(sorted(unknown)))
    result = {**DEFAULT_OPTIONS, **dict(base or {}), **options}
    for key in _BOOL_KEYS:
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be true or false")
    for key in _PATH_KEYS:
        value = result[key]
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{key} must be a local path")
        if value and ("\0" in value or "\n" in value or "\r" in value):
            raise ValueError(f"Invalid local path for {key}")
        result[key] = value.strip() or None if value is not None else None
    from core.speculative_runtime import normalize_speculative_mode
    result["speculative_mode"] = normalize_speculative_mode(result["speculative_mode"])
    count = result["max_draft_tokens"]
    if count is not None and (type(count) is not int or not 1 <= count <= 64):
        raise ValueError("Draft tokens must be blank for automatic, or an integer from 1 to 64")
    branches = result["native_branch_count"]
    if type(branches) is not int or not 1 <= branches <= 64:
        raise ValueError("Native branch count must be an integer from 1 to 64")
    return result


def initial_options(settings):
    values = {key: getattr(settings, "speedup_" + key, default)
              for key, default in DEFAULT_OPTIONS.items()}
    values["speculative_mode"] = getattr(settings, "speculative_mode", DEFAULT_OPTIONS["speculative_mode"])
    values["draft_model_path"] = getattr(settings, "draft_model_path", None)
    values["speculative_server_path"] = getattr(settings, "speculative_server_path", None)
    return normalize_options(values)


def active_backend(engine):
    return getattr(engine, {
        "mlx": "mlx_backend", "gguf": "gguf_backend", "bitnet": "bitnet_backend",
        "controller": "controller_backend",
    }.get(getattr(engine, "active_backend", None), "_no_speedup_backend"), None)


def cancel_before_request(engine, cancel_event):
    """Stop before prompt work and discard telemetry from a previous request."""
    if cancel_event is None or not cancel_event.is_set():
        return False
    backend = active_backend(engine)
    if getattr(backend, "runtime", None) in {"strata", "prism_speculative", "tensorfold_mlx"}:
        stop = getattr(backend, "cancel_generation", None) or getattr(backend, "cancel_current_request", None)
        if callable(stop):
            stop()
        backend.last_generation = {"status": "cancelled"}
        backend.last_branch_usage = {}
    return True


def apply_loaded_options(engine, options=None):
    selected = normalize_options(options, base=getattr(engine, "lossless_speed_options", None))
    backend = active_backend(engine)
    hook = getattr(backend, "configure_lossless_speedups", None)
    if callable(hook):
        from config.paths import get_portable_data_dir
        cache_dir = selected["cache_dir"]
        if selected["persistent_cache"] and not cache_dir:
            cache_dir = str(Path(get_portable_data_dir()) / "prompt_state")
        speculation = None
        configure_draft = getattr(backend, "configure_speculation", None)
        if (getattr(engine, "active_backend", None) == "mlx"
                and getattr(backend, "model", None) is not None and callable(configure_draft)):
            speculation = configure_draft(
                mode=selected["speculative_mode"] if selected["enabled"] else "none",
                max_draft_tokens=selected["max_draft_tokens"],
                # This preference belongs to the legacy native verifier.
                # MLX uses the selected pack's verified MTP sidecar.
                draft_model_path=None,
            )
        result = hook(**{key: selected[key] for key in _BOOL_KEYS}, cache_dir=cache_dir)
        configured = [key for key in ("prompt_reuse", "branch_prefill", "tokenizer_memo",
                                      "persistent_cache", "metal_warmup")
                      if isinstance(result, dict) and result.get(key) is True]
        if isinstance(speculation, dict) and speculation.get("mode") == "mtp":
            configured = [key for key in configured if key not in {"prompt_reuse", "branch_prefill", "persistent_cache"}]
            configured.append("native_mtp")
        return {"status": "configured", "message": "Prompt speedups configured for the loaded runtime.",
                "configured_speedups": configured, "benchmark_required": False,
                "backend": result, "speculation": speculation}
    if getattr(backend, "runtime", None) == "prism_speculative":
        telemetry = backend.get_speculative_telemetry()
        mode = telemetry.get("mode", "none")
        configured = ["native_" + mode] if mode in {"lookup", "mtp", "dflash2"} else []
        return {"status": "configured", "message": "Native target verification is configured.",
                "configured_speedups": configured, "benchmark_required": False, "backend": telemetry}
    return {"status": "pending", "message": "Speed options saved for a compatible model load."}


def describe_speed_result(engine, result):
    """Keep calibration outcome separate from optimizations enabled without it.

    Configuration is not a measured speed claim: reuse requires a matching
    prompt/cache and native drafting still depends on target acceptance.
    """
    status = getattr(engine, "lossless_speed_status", {}) or {}
    configured = list(status.get("configured_speedups", ()))
    output = dict(result)
    output["configured_speedups"] = configured
    if configured:
        labels = {"prompt_reuse": "exact prompt reuse", "branch_prefill": "shared branch prefill",
                  "tokenizer_memo": "tokenizer memo", "persistent_cache": "disk prompt cache",
                  "metal_warmup": "GPU warmup", "native_lookup": "native lookup drafting",
                  "native_mtp": "native MTP drafting", "native_dflash2": "native DFlash2 drafting"}
        output["message"] = str(output.get("message", "")) + " Speedups enabled without tuning: " + ", ".join(
            labels.get(key, key) for key in configured) + "."
    elif status.get("status") == "error":
        output["message"] = str(output.get("message", "")) + " " + str(status.get("message", ""))
    return output


def configure_speedups(engine, options):
    """Change options at an idle boundary; never launch a model or calibration."""
    selected = normalize_options(options, base=getattr(engine, "lossless_speed_options", None))
    lock = engine._model_lock
    if not lock.acquire(blocking=False):
        return {"status": "busy", "message": "Wait for the current model operation to finish."}
    backend = active_backend(engine)
    backend_lock = runtime_lock(backend) if backend is not None else None
    acquired = False
    try:
        if backend_lock is not None:
            acquired = backend_lock.acquire(blocking=False)
            if not acquired:
                return {"status": "busy", "message": "Finish the current generation or learning task first."}
            if uses_mlx_thread_owner(backend):
                # The configured MLX method is dispatched to the thread that
                # owns this same RLock; release the busy probe before dispatch.
                # The owner queue serializes the change with active inference.
                backend_lock.release()
                acquired = False
        previous = getattr(engine, "lossless_speed_options", DEFAULT_OPTIONS)
        native_changed = any(selected[key] != previous[key] for key in (
            "speculative_mode", "speculative_server_path", "draft_model_path",
            "max_draft_tokens", "speculative_runtime_receipt_path", "enabled",
        ))
        result = apply_loaded_options(engine, selected)
        if (native_changed and getattr(backend, "model", None) is not None
                and getattr(engine, "active_backend", None) == "gguf"):
            result = {"status": "reload_required", "message": "Options saved. Reload the model to apply native drafting changes."}
        engine.lossless_speed_options = selected
        engine.lossless_speed_status = result
        return result
    finally:
        if acquired:
            backend_lock.release()
        lock.release()


def load_preferences(path, *, defaults=None):
    fallback = normalize_options(defaults)
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return normalize_options(value.get("lossless_speedups", {}), base=fallback)
    except (OSError, ValueError, TypeError, AttributeError):
        return fallback


def save_preferences(path, options):
    selected = normalize_options(options)
    # Share the existing preferences lock so this cannot drop Auto speed writes.
    from core.gui_speed_tuning import _PREF_LOCK
    target = Path(path)
    with _PREF_LOCK:
        previous = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
        if not isinstance(previous, dict):
            raise ValueError("App preferences are not a JSON object")
        previous["lossless_speedups"] = selected
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                             prefix=target.name + ".", suffix=".tmp", delete=False) as handle:
                temporary = handle.name
                json.dump(previous, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
