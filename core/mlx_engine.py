"""Merged MLX backend: rich app behavior + Metal safeguards + Dynamic SmartKV policy."""
from __future__ import annotations
import gc
import logging
import os
import platform
import psutil
import time
import core._mlx_engine_base as _base
from core._mlx_engine_base import *
from core.kv_cache_manager import SmartKVCacheManager, compute_auto_kv_budget

# Older implementation logged through a module global without defining it.
_base.logger = logging.getLogger(__name__)


_OOM_MARKERS = (
    "insufficient memory",
    "out of memory",
    "outofmemory",
    "kiogpucommandbuffercallbackerroroutofmemory",
)


def _is_metal_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold().replace(" ", "")
    return any(marker.replace(" ", "") in text for marker in _OOM_MARKERS)


def _model_context_limit(model, tokenizer):
    values = []
    for obj in (getattr(model, "args", None), getattr(model, "config", None), tokenizer):
        if obj is None:
            continue
        for name in (
            "max_position_embeddings",
            "max_seq_len",
            "max_sequence_length",
            "context_length",
            "model_max_length",
        ):
            try:
                value = int(getattr(obj, name))
            except Exception:
                continue
            if 1024 <= value <= 10_000_000:
                values.append(value)
    return min(values) if values else None


def _memory_pressure() -> bool:
    try:
        proc_gb = psutil.Process().memory_info().rss / (1024 ** 3)
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        return proc_gb >= 12.5 or avail_gb <= 0.75
    except Exception:
        return False


def _adaptive_prefill_step_size(*, aggressive: bool = False) -> int:
    """Use larger MLX prefill batches when memory headroom exists.

    This changes only prompt chunking, not weights, context, sampling, or token limits.
    MLX-LM itself defaults to 2048; we step down conservatively as memory tightens.
    """
    if aggressive:
        return 128
    try:
        proc_gb = psutil.Process().memory_info().rss / (1024 ** 3)
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        if proc_gb < 9.5 and avail_gb >= 4.0:
            return 2048
        if proc_gb < 11.0 and avail_gb >= 2.0:
            return 1024
        if avail_gb >= 1.0:
            return 512
    except Exception:
        pass
    return 256


def _reclaim_if_needed(mx, *, force: bool = False) -> None:
    if not force and not _memory_pressure():
        return
    gc.collect(2)
    try:
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass


class MLXReasoningBackend(_base.MLXReasoningBackend):
    """Preserves native sampling/streaming while making the SmartKV policy live.

    ProReasoningEngine sends a fully packed history prompt each request, so reusing a
    previous request's raw KV tensors would duplicate context. The manager therefore
    owns the hardware-scaled budget and a fresh prompt-cache arena per packed request;
    conversation state itself is preserved by the app's history packer.

    KV caches default to TurboQuant mixed K8/V3 on supported Apple-Silicon MLX
    runtimes. The model's native low-bit/ternary weight format is untouched.
    """

    def __init__(self, model_path="orcarouter/Qwen3.8-27B-Uncensored-MLX", adapter_path=None):
        super().__init__(model_path=model_path, adapter_path=adapter_path)
        self.kv_cache_manager = None
        self.max_stateful_kv_tokens = compute_auto_kv_budget()
        self.last_generation_cap_reason = ""
        self.last_effective_max_tokens = None
        self.last_model_context_limit = None
        self.last_tok_per_sec = 0.0
        self.last_generation_tokens = 0
        self.last_generation_seconds = 0.0

        # The GUI class is already defined when its ProReasoningEngine is constructed.
        # Install the optional top-row generation-cap control without rewriting app_gui.py.
        try:
            from core.gui_generation_cap import install_gui_generation_cap
            install_gui_generation_cap()
        except Exception:
            pass

    def _effective_generation_cap(self, prompt: str, requested: int) -> int:
        requested = max(1, int(requested))
        try:
            prompt_tokens = len(self.tokenizer.encode(prompt))
        except Exception:
            prompt_tokens = 0
        model_cap = _model_context_limit(self.model, self.tokenizer)
        self.last_model_context_limit = model_cap

        if model_cap is None:
            effective = requested
            self.last_generation_cap_reason = (
                f"User max {requested:,}; model context limit not exposed, so user max is authoritative."
            )
        else:
            remaining = int(model_cap) - int(prompt_tokens)
            if remaining <= 0:
                raise RuntimeError(
                    f"Packed prompt requires {prompt_tokens:,} tokens but model context is {model_cap:,}; "
                    "refusing context truncation."
                )
            effective = min(requested, remaining)
            if effective < requested:
                self.last_generation_cap_reason = (
                    f"Clamped {requested:,} → {effective:,}: packed prompt uses {prompt_tokens:,} of "
                    f"the model's {model_cap:,}-token context; context is never dropped."
                )
            else:
                self.last_generation_cap_reason = (
                    f"User max {requested:,}; {remaining:,} tokens remain in the model's "
                    f"{model_cap:,}-token context."
                )
        self.last_effective_max_tokens = int(effective)
        return int(effective)

    def load_model(self) -> bool:
        """Load native model weights; TurboQuant is applied only to runtime KV caches."""
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

    def generate_branches(
        self,
        prompt: str,
        branch_count: int = 16,
        max_tokens: int = 512,
        temperature=0.75,
        top_p: float = 0.92,
    ):
        """Generate Pro branches without silent quality-reducing fallbacks."""
        if not self.is_mlx_available or self.model is None:
            return []

        import mlx.core as mx
        import mlx_lm

        try:
            from mlx_lm.sample_utils import make_sampler
        except Exception as exc:
            raise RuntimeError("MLX sampler unavailable; refusing to change Pro sampling semantics") from exc

        count = max(1, min(int(branch_count), 16))
        temps = list(temperature) if isinstance(temperature, (list, tuple)) else [float(temperature)] * count
        branches = []
        effective_max = self._effective_generation_cap(prompt, max_tokens)

        def _one(temp_value: float, prefill_step_size: int) -> str:
            sampler = make_sampler(temp=float(temp_value), top_p=top_p)
            if self.kv_cache_manager is None:
                self.kv_cache_manager = SmartKVCacheManager(
                    self.model,
                    max_tokens=self.max_stateful_kv_tokens,
                )
            else:
                self.kv_cache_manager.reset(purge_allocator=False)
            kwargs = {
                "prompt": prompt,
                "max_tokens": effective_max,
                "sampler": sampler,
                "prefill_step_size": int(prefill_step_size),
                "prompt_cache": self.kv_cache_manager.get_cache(),
            }
            pieces = []
            response = None
            generated = 0
            started = time.perf_counter()
            try:
                iterator = mlx_lm.stream_generate(self.model, self.tokenizer, **kwargs)
            except TypeError:
                kwargs.pop("prefill_step_size", None)
                kwargs.pop("prompt_cache", None)
                iterator = mlx_lm.stream_generate(self.model, self.tokenizer, **kwargs)
            for response in iterator:
                generated += 1
                try:
                    measured = float(getattr(response, "generation_tps", 0.0) or 0.0)
                except Exception:
                    measured = 0.0
                if measured > 0.0:
                    self.last_tok_per_sec = measured
                chunk = getattr(response, "text", None)
                if chunk is None:
                    chunk = str(response)
                pieces.append(str(chunk))

            elapsed = max(0.001, time.perf_counter() - started)
            if self.last_tok_per_sec <= 0.0 and generated:
                self.last_tok_per_sec = generated / elapsed
            self.last_generation_tokens = generated
            self.last_generation_seconds = elapsed
            return "".join(pieces)

        for idx in range(count):
            _reclaim_if_needed(mx)
            temp_value = float(temps[idx % len(temps)])
            try:
                branches.append(_one(temp_value, _adaptive_prefill_step_size()))
            except RuntimeError as exc:
                if not _is_metal_oom(exc):
                    raise
                # Lossless OOM retry: same prompt, same sampler policy, same token
                # budget and the same TurboQuant cache policy. Only prefill chunking changes.
                _reclaim_if_needed(mx, force=True)
                branches.append(_one(temp_value, 64))
            finally:
                _reclaim_if_needed(mx)

        return branches

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

        effective_max = self._effective_generation_cap(prompt, max_tokens)

        # The app passes the full history-packed prompt each time. Reset only the
        # logical TurboQuant-preferred KV state. Do not purge Metal's allocator on every
        # request; the complete packed prompt is recomputed into the fresh cache.
        if self.kv_cache_manager is None:
            self.kv_cache_manager = SmartKVCacheManager(
                self.model,
                max_tokens=self.max_stateful_kv_tokens,
            )
        else:
            self.kv_cache_manager.reset(purge_allocator=False)
        self.kv_cache_manager.auto_compact_if_needed(prompt_len)

        try:
            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(temp=temperature, top_p=top_p)
        except Exception as exc:
            raise RuntimeError("MLX sampler unavailable; refusing to change sampling semantics") from exc

        generated = 0
        response = None
        started = time.perf_counter()
        self.last_tok_per_sec = 0.0

        def _run_stream(prefill_step_size: int):
            # Same prompt, sampler, token allowance and TurboQuant-preferred KV for
            # normal execution and OOM retry. Only prefill chunk size may change.
            kwargs = {
                "max_tokens": effective_max,
                "sampler": sampler,
                "prefill_step_size": int(prefill_step_size),
            }
            try:
                return mlx_lm.stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    prompt_cache=self.kv_cache_manager.get_cache(),
                    **kwargs,
                )
            except TypeError:
                # Older MLX-LM may not accept an explicit prompt cache. Library-managed
                # cache is still native/full precision and preserves the same sampler.
                kwargs.pop("prefill_step_size", None)
                return mlx_lm.stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    **kwargs,
                )

        def _yield_response(resp):
            try:
                measured = float(getattr(resp, "generation_tps", 0.0) or 0.0)
            except Exception:
                measured = 0.0
            if measured > 0.0:
                self.last_tok_per_sec = measured
            if hasattr(resp, "text"):
                return resp.text
            if isinstance(resp, str):
                return resp
            if hasattr(resp, "token"):
                return self.tokenizer.decode([resp.token])
            return str(resp)

        try:
            iterator = None
            try:
                iterator = _run_stream(_adaptive_prefill_step_size())
                for response in iterator:
                    generated += 1
                    yield _yield_response(response)
            except RuntimeError as exc:
                if not _is_metal_oom(exc) or generated:
                    raise
                # Retry only before any token has been emitted. Rebuild a fresh cache
                # and reduce transient prefill memory without changing cache policy,
                # context, sampler or requested generation length.
                self.kv_cache_manager.reset(purge_allocator=False)
                _reclaim_if_needed(mx, force=True)
                iterator = _run_stream(_adaptive_prefill_step_size(aggressive=True))
                for response in iterator:
                    generated += 1
                    yield _yield_response(response)
        except Exception as exc:
            _base.logger.error("MLX stream_generate error: %s", exc)
            raise
        finally:
            elapsed = max(0.001, time.perf_counter() - started)
            if self.last_tok_per_sec <= 0.0 and generated:
                self.last_tok_per_sec = generated / elapsed
            self.last_generation_tokens = generated
            self.last_generation_seconds = elapsed
            self.kv_cache_manager.current_length = min(
                self.max_stateful_kv_tokens,
                max(0, prompt_len + generated),
            )
            _reclaim_if_needed(mx)
