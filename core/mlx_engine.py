"""Merged MLX backend: rich app behavior + Metal safeguards + Dynamic SmartKV policy."""
from __future__ import annotations
import logging
import os
import platform
import core._mlx_engine_base as _base
from core._mlx_engine_base import *
from core.kv_cache_manager import SmartKVCacheManager, compute_auto_kv_budget

# Older implementation logged through a module global without defining it.
_base.logger = logging.getLogger(__name__)


class MLXReasoningBackend(_base.MLXReasoningBackend):
    """Preserves native sampling/streaming while making the SmartKV policy live.

    ProReasoningEngine sends a fully packed history prompt each request, so reusing a
    previous request's raw KV tensors would duplicate context. The manager therefore
    owns the hardware-scaled budget and a fresh prompt-cache arena per packed request;
    conversation state itself is preserved by the app's history packer.

    KV caches intentionally remain at MLX-LM's native/full precision. The model's
    native low-bit/ternary weight format is untouched; only lossy KV quantization is
    forbidden so inference does not trade attention-state precision for RAM/speed.
    """

    def __init__(self, model_path="orcarouter/Qwen3.8-27B-Uncensored-MLX", adapter_path=None):
        super().__init__(model_path=model_path, adapter_path=adapter_path)
        self.kv_cache_manager = None
        self.max_stateful_kv_tokens = compute_auto_kv_budget()

    def load_model(self) -> bool:
        """Load native model weights without forcing any lossy KV-cache quantization."""
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            self.is_mlx_available = False
            return False
        if not os.path.exists(self.model_path) and os.getenv("OFFLINE", "0") == "1":
            self.is_mlx_available = False
            return False

        try:
            import mlx_lm
            from core.memory_watchdog import SystemMemoryWatchdog

            SystemMemoryWatchdog.adjust_dynamic_metal_headroom()
            self.model, self.tokenizer = mlx_lm.load(
                self.model_path,
                adapter_path=(
                    self.adapter_path
                    if self.adapter_path and os.path.exists(self.adapter_path)
                    else None
                ),
            )
            self.is_mlx_available = self.model is not None and self.tokenizer is not None
        except Exception as exc:
            _base.logger.error("MLX full-precision load failed: %s", exc)
            self.is_mlx_available = False
            return False

        if self.is_mlx_available:
            self.kv_cache_manager = SmartKVCacheManager(
                self.model,
                max_tokens=self.max_stateful_kv_tokens,
            )
        return bool(self.is_mlx_available)

    def stream_generate_tokens(self, prompt: str, max_tokens: int = 512,
                               temperature: float = 0.75, top_p: float = 0.92):
        if not self.is_mlx_available or self.model is None:
            return

        import mlx.core as mx
        import mlx_lm

        prompt_len = 0
        try:
            prompt_len = len(self.tokenizer.encode(prompt))
        except Exception:
            pass

        # The app passes the full history-packed prompt each time. Reset the arena so
        # no previous request can contaminate it, while retaining the RAM-scaled cap.
        # This resets/recomputes cache state; it does not truncate the packed prompt.
        if self.kv_cache_manager is None:
            self.kv_cache_manager = SmartKVCacheManager(
                self.model,
                max_tokens=self.max_stateful_kv_tokens,
            )
        else:
            self.kv_cache_manager.reset()
        self.kv_cache_manager.auto_compact_if_needed(prompt_len)

        sampler = None
        try:
            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(temp=temperature, top_p=top_p)
        except Exception:
            pass

        kwargs = {"max_tokens": min(max_tokens, 4096)}
        if sampler is not None:
            kwargs["sampler"] = sampler
        else:
            kwargs["temp"] = temperature
            kwargs["top_p"] = top_p

        generated = 0
        try:
            # Newer MLX-LM versions accept an explicit native/full-precision prompt
            # cache. Fall back to the library-managed native cache on older versions.
            try:
                iterator = mlx_lm.stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    prompt_cache=self.kv_cache_manager.get_cache(),
                    **kwargs,
                )
            except TypeError:
                iterator = mlx_lm.stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    **kwargs,
                )

            for response in iterator:
                generated += 1
                if hasattr(response, "text"):
                    yield response.text
                elif isinstance(response, str):
                    yield response
                elif hasattr(response, "token"):
                    yield self.tokenizer.decode([response.token])
        except Exception as exc:
            _base.logger.error("MLX stream_generate error: %s", exc)
            try:
                answer = mlx_lm.generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=min(max_tokens, 512),
                    verbose=False,
                )
                if answer:
                    yield answer
            except Exception:
                return
        finally:
            self.kv_cache_manager.current_length = min(
                self.max_stateful_kv_tokens,
                max(0, prompt_len + generated),
            )
            try:
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
                elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                    mx.metal.clear_cache()
            except Exception:
                pass
