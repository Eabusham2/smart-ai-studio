"""Production hardening for ProReasoningEngine.

Keeps mock generation available for explicit unit-test mode only, prevents loaded
live backends from silently falling through to synthetic output, replaces hard-coded
telemetry with measured values, synchronizes loaded-state UI/cache state, and gives
live MLX learning a stable adapter checkpoint path across restarts.
"""
from __future__ import annotations

import os
import time

from core.hf_downloader import register_loaded_model, unregister_loaded_model


DEFAULT_MLX_ADAPTER_PATH = os.path.abspath("./consolidated_slow_lora/adapter.safetensors")


def _active_backend_name(engine) -> str:
    active = getattr(engine, "active_backend", None)
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


def _count_response_tokens(engine, text: str) -> int:
    tokenizers = [
        getattr(getattr(engine, "mlx_backend", None), "tokenizer", None),
        getattr(engine, "tokenizer", None),
    ]
    for tok in tokenizers:
        if tok is None or not hasattr(tok, "encode"):
            continue
        try:
            return max(0, len(tok.encode(text or "")))
        except Exception:
            pass
    return max(0, len((text or "").split()))


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
        if not getattr(self.settings, "use_mock", False) and not getattr(self, "lora_adapter_path", None):
            self.lora_adapter_path = DEFAULT_MLX_ADAPTER_PATH
            mlx = getattr(self, "mlx_backend", None)
            if mlx is not None:
                mlx.adapter_path = self.lora_adapter_path

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
        try:
            return original_unload(self, *args, **kwargs)
        finally:
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
        started = time.perf_counter()
        response, metadata = original_solve(self, *args, **kwargs)
        elapsed = max(1e-9, time.perf_counter() - started)
        metadata = dict(metadata or {})
        generated = _count_response_tokens(self, response)
        metadata["tokens_generated"] = generated
        metadata["tok_speed"] = generated / elapsed if generated else 0.0
        metadata["measured_wall_time_s"] = elapsed
        metadata["memory_rss_mb"] = _rss_mb()
        metadata["backend"] = _active_backend_name(self)
        return response, metadata

    cls.__init__ = hardened_init
    cls.load_model = hardened_load
    cls.unload_model = hardened_unload
    cls._generate_fallback_branches = hardened_fallback
    cls.solve = hardened_solve
    cls._pro_runtime_hardening_installed = True
