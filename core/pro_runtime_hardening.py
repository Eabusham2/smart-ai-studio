"""Production hardening for ProReasoningEngine.

Keeps mock generation available for explicit unit-test mode only, prevents loaded
live backends from silently falling through to synthetic output, replaces hard-coded
telemetry with measured values, synchronizes loaded-state UI/cache state, and gives
live MLX learning a stable adapter checkpoint path across restarts.
"""
from __future__ import annotations

import os
import math
import time

from core.hf_downloader import register_loaded_model, unregister_loaded_model


# Kept only as the legacy migration location. New learned state is selected by
# exact model identity in core.learning_state, never by one global adapter file.
DEFAULT_MLX_ADAPTER_PATH = os.path.abspath("./consolidated_slow_lora/adapters.safetensors")


def _normalize_adapter_path(path) -> str:
    raw = str(path or "").strip()
    if not raw:
        return DEFAULT_MLX_ADAPTER_PATH
    expanded = os.path.abspath(os.path.expanduser(raw))
    if os.path.isdir(expanded) or raw.endswith(("/", "\\")):
        return os.path.join(expanded, "adapters.safetensors")
    return expanded


def _active_backend_name(engine) -> str:
    active = getattr(engine, "active_backend", None)
    controller = getattr(engine, "controller_backend", None)
    if active == "controller" and getattr(controller, "direct_cuda_loaded", False) is True:
        return controller.direct_backend_name
    if active == "controller" and getattr(getattr(engine, "controller_backend", None), "runtime", None) == "strata":
        return "strata"
    if active == "gguf" and getattr(getattr(engine, "gguf_backend", None), "runtime", None) == "prism_speculative":
        return "prism_speculative"
    if active:
        return str(active)
    mlx = getattr(engine, "mlx_backend", None)
    if mlx is not None and getattr(mlx, "is_mlx_available", False) and getattr(mlx, "model", None) is not None:
        return "mlx"
    gguf = getattr(engine, "gguf_backend", None)
    if gguf is not None and getattr(gguf, "is_gguf_available", False):
        return "gguf"
    bitnet = getattr(engine, "bitnet_backend", None)
    if bitnet is not None and getattr(bitnet, "is_loaded", False):
        return "bitnet"
    if getattr(engine, "model", None) is not None:
        return "torch"
    return "live_unloaded"


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def install_pro_runtime_hardening(cls) -> None:
    if getattr(cls, "_pro_runtime_hardening_installed", False):
        return

    original_init = cls.__init__
    original_fallback = cls._generate_fallback_branches
    original_solve = cls.solve
    original_load = cls.load_model
    original_unload = cls.unload_model

    def hardened_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not getattr(self.settings, "use_mock", False):
            selected = getattr(self, "lora_adapter_path", None)
            normalized = _normalize_adapter_path(selected) if selected else None
            self.lora_adapter_path = normalized
            self._explicit_adapter_path = bool(selected)
            mlx = getattr(self, "mlx_backend", None)
            if mlx is not None:
                mlx.adapter_path = normalized
                mlx._explicit_adapter_path = bool(selected)

    def hardened_load(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        if isinstance(result, dict) and result.get("status") == "loaded" and not getattr(self.settings, "use_mock", False):
            path = str(result.get("path") or "").strip()
            if path:
                register_loaded_model(path)
            model_name = ""
            if args:
                model_name = str(args[0] or "").strip()
            elif kwargs.get("model_name"):
                model_name = str(kwargs.get("model_name") or "").strip()
            if model_name:
                register_loaded_model(model_name)
        return result

    def hardened_unload(self, *args, **kwargs):
        owns_sidecar = (
            getattr(getattr(self, "controller_backend", None), "runtime", None) == "strata"
            or getattr(getattr(self, "gguf_backend", None), "runtime", None) == "prism_speculative"
        )
        result = None
        try:
            result = original_unload(self, *args, **kwargs)
            return result
        finally:
            # A failed owned shutdown can leave real model resources resident.
            # Keep their registration until the retained runtime is stopped.
            if not owns_sidecar or (isinstance(result, dict) and result.get("status") == "unloaded"):
                unregister_loaded_model()

    def hardened_fallback(self, prompt: str, branch_count: int):
        if getattr(self.settings, "use_mock", False):
            return original_fallback(self, prompt, branch_count)
        if not self.is_model_loaded:
            return original_fallback(self, prompt, branch_count)
        raise RuntimeError(
            "Live model weights are loaded, but every real generation backend returned no candidates. "
            "Refusing to substitute mock/synthetic output."
        )

    def hardened_solve(self, *args, **kwargs):
        from core.generation_metrics import RequestMetrics, observe_generations, valid_rate
        started = time.perf_counter()
        measured_work = RequestMetrics()
        with observe_generations(measured_work.update):
            response, metadata = original_solve(self, *args, **kwargs)
        elapsed = max(1e-9, time.perf_counter() - started)
        measurements = measured_work.snapshot()
        metadata = dict(metadata or {})
        controller = getattr(self, "controller_backend", None)
        is_strata = getattr(controller, "runtime", None) == "strata"
        native = getattr(self, "gguf_backend", None)
        is_native = (getattr(self, "active_backend", None) == "gguf"
                     and getattr(native, "runtime", None) == "prism_speculative")
        owner = controller if is_strata else native if is_native else None
        generated = measurements.get("tokens")
        metadata["token_usage_available"] = type(generated) is int
        metadata["generation_invocations"] = measurements["invocations"]
        if is_strata or is_native:
            usage = dict(getattr(owner, "last_branch_usage", {}) or {})
            if is_native and any(type(value) is not int or value < 0 for value in usage.values()):
                usage = {}
            # Whole-search server usage remains authoritative when available.
            # A deliberately stopped branch may have live sampled counters but
            # no final server usage record; do not discard those observations.
            if type(usage.get("completion_tokens")) is int:
                generated = usage["completion_tokens"]
            metadata["usage"] = usage
            metadata["token_usage_available"] = type(generated) is int
        if type(generated) is int:
            metadata["tokens_generated"] = generated
        else:
            metadata.pop("tokens_generated", None)
        # End-to-end throughput includes all candidates, never text replay.
        metadata["tok_speed"] = generated / elapsed if generated else 0.0
        metadata["measured_wall_time_s"] = elapsed
        metadata["memory_rss_mb"] = _rss_mb()
        if is_strata or is_native:
            # Count only processes whose ownership the managed runtime verified.
            from core.memory_watchdog import SystemMemoryWatchdog
            memory = SystemMemoryWatchdog.get_detailed_memory_breakdown(owned_pids=owner.owned_process_ids())
            metadata["memory_rss_mb"] = memory["total_allocated_gb"] * 1024
            if not metadata["token_usage_available"]:
                metadata.pop("tokens_generated", None)
        metadata["backend"] = _active_backend_name(self)

        # Decode throughput uses this request's observations, not a stale
        # counter from an inactive backend or a tokenizer applied to the winner.
        observed_decode = measurements.get("decode_tps")
        decode_tps = float(observed_decode) if valid_rate(observed_decode) else 0.0
        metadata["decode_tok_speed"] = decode_tps
        if is_native:
            telemetry = dict(native.get_speculative_telemetry() or {})
            metadata["speculative"] = telemetry
            if not measured_work.samples:
                native_rate = telemetry.get("decode_tokens_per_second")
                metadata["decode_tok_speed"] = float(native_rate) if valid_rate(native_rate) else 0.0
        active_owner = {"mlx": getattr(self, "mlx_backend", None),
                        "gguf": native, "controller": controller,
                        "bitnet": getattr(self, "bitnet_backend", None)}.get(
                            getattr(self, "active_backend", None), self)
        metadata["generation_cap_reason"] = str(
            getattr(active_owner, "last_generation_cap_reason", "") or ""
        )
        note = str(getattr(self, "_last_context_consolidation_note", "") or "")
        if note:
            metadata["context_consolidation"] = note
        return response, metadata

    cls.__init__ = hardened_init
    cls.load_model = hardened_load
    cls.unload_model = hardened_unload
    cls._generate_fallback_branches = hardened_fallback
    cls.solve = hardened_solve
    cls._pro_runtime_hardening_installed = True
