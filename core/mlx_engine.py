"""Merged MLX backend: rich app behavior + Metal safeguards + Dynamic SmartKV policy."""
from __future__ import annotations
import gc
import logging
import os
import platform
import psutil
import time
from typing import Any, Dict, Optional
import core._mlx_engine_base as _base
from core._mlx_engine_base import *
from core.kv_cache_manager import SmartKVCacheManager, compute_auto_kv_budget
from core.controller_runtime import UniversalControllerBackend, resolve_controller_runtime
from core.prompt_reuse import (
    ExactPrefixStore, TokenizerMemo, cache_nbytes, cache_policy_signature, tokenizer_signature)

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


def _local_artifact_stamp(path):
    """Guard the interval between loading local bytes and opt-in content hashing."""
    if not path:
        return ()
    try:
        from pathlib import Path
        root = Path(path).expanduser().resolve(strict=True)
        files = [root] if root.is_file() else sorted(item for item in root.rglob("*") if item.is_file())
        if not files:
            return None
        return tuple((str(item), (s := item.stat()).st_dev, s.st_ino, s.st_size,
                      s.st_mtime_ns, s.st_ctime_ns) for item in files)
    except (OSError, ValueError):
        return None


class MLXReasoningBackend(_base.MLXReasoningBackend):
    """Preserves native sampling/streaming while making the SmartKV policy live.

    Packed history is tokenized once. Supported immutable prefill snapshots are
    reused only for exact token prefixes, with isolated state for every decode.
    Unknown cache formats continue through their original complete-prefill path.

    KV caches default to TurboQuant mixed K8/V3 on supported Apple-Silicon MLX
    runtimes. The model's native low-bit/ternary weight format is untouched.
    """

    def __init__(
        self,
        model_path="orcarouter/Qwen3.8-27B-Uncensored-MLX",
        adapter_path=None,
        model_info: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_path=model_path, adapter_path=adapter_path)
        self.model_info = dict(model_info or {})
        self.kv_cache_manager = None
        self.max_stateful_kv_tokens = compute_auto_kv_budget()
        self.last_generation_cap_reason = ""
        self.last_effective_max_tokens = None
        self.last_model_context_limit = None
        self.last_prompt_token_count = None
        self.last_tok_per_sec = 0.0
        self.last_generation_tokens = 0
        self.last_generation_seconds = 0.0
        self.processor = None
        self._vlm_runtime = None
        self._vlm_loader = None
        self._speed_prefill_step_size = None
        self._speed_profile_context = None
        self.last_error = ""
        self._tokenizer_memo = TokenizerMemo()
        self._prompt_prefixes = ExactPrefixStore(max_tokens=self.max_stateful_kv_tokens)
        self._prompt_reuse_identity = None
        self._prompt_weight_revision = 0
        self._lossless_speedups = {"prompt_reuse": True, "branch_prefill": True,
                                  "tokenizer_memo": True, "persistent_cache": False,
                                  "metal_warmup": False}
        self.last_prompt_reused_tokens = 0
        self.last_prompt_reuse_reason = "idle"
        self._persistent_prompt_store = None
        self._persistent_prompt_identity = None
        self._prompt_disk_pristine = False
        self._prompt_disk_model_id = None
        self._prompt_disk_tokenizer_id = None
        self._prompt_load_stamps = None
        self._prompt_cache_directory = None
        from core.mlx_warmup import SafeMLXWarmup
        self._metal_warmup = SafeMLXWarmup()
        from core.vlm_prompt_reuse import NativeVLMCacheReuse
        self._vlm_prefixes = NativeVLMCacheReuse()

        # The GUI class is already defined when its ProReasoningEngine is constructed.
        # Install the optional top-row generation-cap control without rewriting app_gui.py.
        try:
            from core.gui_generation_cap import install_gui_generation_cap
            install_gui_generation_cap()
        except Exception:
            pass

    def configure_auto_speed(self, enabled=True, force=False, progress=None, cancelled=None):
        from core.mlx_speed_tuning import configure_mlx_speed
        from core.native_speed import runtime_lock
        with runtime_lock(self):
            return configure_mlx_speed(self, enabled, force, progress, cancelled)

    def configure_lossless_speedups(self, enabled=True, *, prompt_reuse=True,
                                   branch_prefill=True, tokenizer_memo=True,
                                   persistent_cache=False, cache_dir=None,
                                   metal_warmup=False):
        """Apply execution-only preferences; selected weights/cache policy stay fixed."""
        from core.native_speed import runtime_lock
        with runtime_lock(self):
            self.invalidate_prompt_reuse("configuration")
            self._lossless_speedups = {
                "prompt_reuse": bool(enabled and prompt_reuse),
                "branch_prefill": bool(enabled and branch_prefill),
                "tokenizer_memo": bool(enabled and tokenizer_memo),
                "persistent_cache": bool(enabled and persistent_cache),
                "metal_warmup": bool(enabled and metal_warmup),
            }
            self._prompt_cache_directory = cache_dir
            from core.mlx_warmup import SafeMLXWarmup
            self._metal_warmup = SafeMLXWarmup(enabled=self._lossless_speedups["metal_warmup"])
            return dict(self._lossless_speedups)

    def invalidate_prompt_reuse(self, reason="mutation"):
        """Call before external in-place weight/tokenizer/adapter changes, or unload."""
        from core.native_speed import runtime_lock
        with runtime_lock(self):
            self._prompt_weight_revision += 1
            self._prompt_reuse_identity = None
            self._prompt_prefixes.clear()
            self._vlm_prefixes.clear()
            self._tokenizer_memo.clear()
            self._persistent_prompt_identity = None
            if self._persistent_prompt_store is not None:
                self._persistent_prompt_store.close()
            self._persistent_prompt_store = None
            self._metal_warmup.close()
            if reason not in {"configuration", "model load"}:
                self._prompt_disk_pristine = False
            if self.kv_cache_manager is not None:
                self.kv_cache_manager.release()
            self.last_prompt_reused_tokens = 0
            self.last_prompt_reuse_reason = str(reason)

    def _encode_prompt(self, prompt):
        signature = (id(self.model), tokenizer_signature(self.tokenizer),
                     str(self.model_path), str(self.adapter_path), self._prompt_weight_revision)
        if signature != self._prompt_reuse_identity:
            if self._prompt_reuse_identity is not None:
                self._prompt_disk_pristine = False
            self._prompt_prefixes.clear()
            self._vlm_prefixes.clear()
            self._tokenizer_memo.clear()
            self._persistent_prompt_identity = None
            self._prompt_reuse_identity = signature
            if self.kv_cache_manager is not None and self.kv_cache_manager.model is not self.model:
                self.kv_cache_manager = None
        if _memory_pressure():
            self._prompt_prefixes.clear()
            self._vlm_prefixes.clear()
            self._tokenizer_memo.clear()
        return self._tokenizer_memo.encode(
            self.tokenizer, prompt, enabled=self._lossless_speedups["tokenizer_memo"])

    def prompt_reuse_telemetry(self):
        return {"reused_tokens": self.last_prompt_reused_tokens,
                "reason": self.last_prompt_reuse_reason,
                "prefix_hits": self._prompt_prefixes.hits,
                "prefix_misses": self._prompt_prefixes.misses,
                "prefix_bytes": self._prompt_prefixes.nbytes,
                "tokenizer_hits": self._tokenizer_memo.hits,
                "tokenizer_misses": self._tokenizer_memo.misses,
                **self._vlm_prefixes.telemetry(),
                **self._metal_warmup.telemetry()}

    def _vlm_reuse_kwargs(self, prepared_prompt, prefill_kwargs, *, branches=False):
        enabled = self._lossless_speedups["prompt_reuse"] or (
            branches and self._lossless_speedups["branch_prefill"])
        kwargs = self._vlm_prefixes.request_kwargs(
            self.model, self.processor, prepared_prompt,
            prefill_step_size=prefill_kwargs.get("prefill_step_size"),
            revision=[self._prompt_weight_revision, tokenizer_signature(self.tokenizer)],
            enabled=enabled and not any(getattr(owner, "training", False) for owner in
                                       (self.model, getattr(self.model, "language_model", None))),
            memory_ok=not _memory_pressure())
        self.last_prompt_reused_tokens = 0
        self.last_prompt_reuse_reason = self._vlm_prefixes.last_reason
        return kwargs

    def _finish_vlm_request(self, *, completed, branches=False):
        self._metal_warmup.touch()
        if not completed or _memory_pressure() or (
                not branches and not self._lossless_speedups["prompt_reuse"]):
            self._vlm_prefixes.clear()

    def _prepare_optional_warmup(self, mx):
        from core.native_speed import runtime_lock
        if self._metal_warmup._closed and self._lossless_speedups["metal_warmup"]:
            from core.mlx_warmup import SafeMLXWarmup
            self._metal_warmup = SafeMLXWarmup(enabled=True)
        self._metal_warmup.prepare(mx=mx, runtime_lock=runtime_lock(self),
                                   memory_ok=lambda: not _memory_pressure())

    def _disk_prompt_identity(self, prefill_step_size):
        """Opt-in disk state is allowed only for exact local, unmodified weights."""
        if (not self._lossless_speedups["persistent_cache"] or not self._prompt_cache_directory
                or not self._prompt_disk_pristine or self._vlm_runtime is not None
                or self._prompt_disk_model_id != id(self.model)
                or self._prompt_disk_tokenizer_id != id(self.tokenizer)):
            return None
        try:
            from importlib.metadata import version
            from core.persistent_prompt_cache import (
                PersistentPromptStore, local_artifact_identity, prompt_cache_identity)
            if self._persistent_prompt_identity is None:
                current_stamps = (_local_artifact_stamp(self.model_path), _local_artifact_stamp(self.adapter_path))
                if current_stamps != self._prompt_load_stamps or None in current_stamps:
                    self._prompt_disk_pristine = False
                    return None
                model = local_artifact_identity(self.model_path)
                adapter = local_artifact_identity(self.adapter_path) if self.adapter_path else None
                if model is None or (self.adapter_path and adapter is None):
                    return None
                if current_stamps != (_local_artifact_stamp(self.model_path), _local_artifact_stamp(self.adapter_path)):
                    self._prompt_disk_pristine = False
                    return None
                tokenizer_config = {
                    name: getattr(self.tokenizer, name, None) for name in
                    ("chat_template", "special_tokens_map", "bos_token_id", "eos_token_id",
                     "add_bos_token", "add_eos_token", "init_kwargs")}
                added = getattr(self.tokenizer, "get_added_vocab", None)
                if callable(added):
                    tokenizer_config["added_vocab"] = added()
                import json
                # Public configuration must be serializable, never an object repr.
                json.dumps(tokenizer_config, sort_keys=True, allow_nan=False)
                self._persistent_prompt_identity = prompt_cache_identity(
                    model=model,
                    tokenizer={"artifact": model,
                               "class": type(self.tokenizer).__module__ + "." + type(self.tokenizer).__qualname__,
                               "configuration": tokenizer_config},
                    adapter=adapter,
                    cache_policy="desktop-existing-turboquant-k8-v3-with-native-fallback",
                    versions={name: version(name) for name in ("mlx", "mlx-lm", "transformers", "tokenizers")})
                self._persistent_prompt_store = PersistentPromptStore(self._prompt_cache_directory, enabled=True)
            return {**self._persistent_prompt_identity, "prefill_step_size": int(prefill_step_size),
                    "prefill_snapshot_schema": 1}
        except Exception:
            return None

    def _prepare_text_cache(self, prompt_ids, prefill_step_size, *, branches=False):
        """Evaluate a prompt prefix once, leaving the final token for MLX-LM."""
        import inspect
        import importlib
        import mlx.core as mx
        if self.kv_cache_manager is None:
            self.kv_cache_manager = SmartKVCacheManager(
                self.model, max_tokens=self.max_stateful_kv_tokens)
        arena = self.kv_cache_manager
        arena.ensure_fresh()
        self.last_prompt_reused_tokens = 0
        self.last_prompt_reuse_reason = "full prefill"
        reuse = self._lossless_speedups["prompt_reuse"] or (
            branches and self._lossless_speedups["branch_prefill"])
        if not reuse or len(prompt_ids) < 2 or _memory_pressure() or getattr(self.model, "training", False):
            if _memory_pressure() or getattr(self.model, "training", False):
                self._prompt_prefixes.clear()
            if getattr(self.model, "training", False):
                self._prompt_disk_pristine = False
            return list(prompt_ids)
        try:
            # Check the actual generation API before evaluating any model state.
            params = inspect.signature(importlib.import_module("mlx_lm.generate").generate_step).parameters
            if not {"prompt_cache", "prefill_step_size"}.issubset(params):
                raise ValueError("runtime does not expose explicit prompt-cache support")
            cache_nbytes(arena.get_cache(), 0)
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            self.last_prompt_reuse_reason = str(exc)
            return list(prompt_ids)
        found = self._prompt_prefixes.lookup(prompt_ids, prefill_step_size)
        if found is not None and cache_policy_signature(found[0]) != cache_policy_signature(arena.get_cache()):
            self._prompt_prefixes.clear()
            found = None
        disk_identity = self._disk_prompt_identity(prefill_step_size)
        if found is None and disk_identity is not None:
            loaded = self._persistent_prompt_store.load(prompt_ids, disk_identity)
            if loaded is not None:
                tokens, cache = loaded
                n = len(tokens)
                if n == len(prompt_ids) - 1 or n % prefill_step_size == 0:
                    if (cache_policy_signature(cache) == cache_policy_signature(arena.get_cache())
                            and self._prompt_prefixes.remember(tokens, cache, prefill_step_size)):
                        found = self._prompt_prefixes.lookup(prompt_ids, prefill_step_size)
        offset = 0
        if found is not None:
            arena.cache, offset = found
            self.last_prompt_reused_tokens = offset
            self.last_prompt_reuse_reason = "exact token prefix"
        # Stop retaining prefixes above budget; the full context still reaches
        # generation. Memory pressure can release optional state at any boundary.
        if len(prompt_ids) - 1 > self._prompt_prefixes.max_tokens:
            return list(prompt_ids[offset:])
        end = len(prompt_ids) - 1
        while offset < end:
            stop = min(end, offset + int(prefill_step_size))
            self.model(mx.array([prompt_ids[offset:stop]]), cache=arena.get_cache())
            mx.eval([layer.state for layer in arena.get_cache()])
            offset = stop
            arena.current_length = offset
            if _memory_pressure():
                self._prompt_prefixes.clear()
                self.last_prompt_reuse_reason = "memory pressure; snapshot skipped"
                return list(prompt_ids[offset:])
        stored = self._prompt_prefixes.remember(prompt_ids[:end], arena.get_cache(), prefill_step_size)
        if stored and not self.last_prompt_reused_tokens:
            self.last_prompt_reuse_reason = "prefill snapshot saved"
        if stored and disk_identity is not None:
            self._persistent_prompt_store.save(prompt_ids[:end], arena.get_cache(), disk_identity)
        return list(prompt_ids[offset:])

    def _release_text_cache(self, *, branches=False):
        self._metal_warmup.touch()
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.release()
        if (not self._lossless_speedups["prompt_reuse"] and not branches) or _memory_pressure():
            self._prompt_prefixes.clear()

    def _speed_prefill_size(self) -> int:
        safe = _adaptive_prefill_step_size()
        preferred = getattr(self, "_speed_prefill_step_size", None)
        current_context = int(getattr(self, "context_budget_tokens", 0) or 32768)
        if getattr(self, "_speed_profile_context", None) != current_context:
            return safe
        return min(safe, int(preferred)) if preferred else safe

    def _speed_vlm_prefill_kwargs(self) -> Dict[str, int]:
        """Preserve the VLM library's own default until a profile is measured."""
        current_context = int(getattr(self, "context_budget_tokens", 0) or 32768)
        preferred = getattr(self, "_speed_prefill_step_size", None)
        if not preferred or getattr(self, "_speed_profile_context", None) != current_context:
            return {}
        return {"prefill_step_size": min(int(preferred), _adaptive_prefill_step_size())}

    def _effective_generation_cap(
        self,
        prompt: str,
        requested: int,
        prompt_token_count: Optional[int] = None,
    ) -> int:
        requested = max(1, int(requested))
        if prompt_token_count is None:
            try:
                prompt_tokens = len(self.tokenizer.encode(prompt))
            except Exception:
                prompt_tokens = 0
        else:
            prompt_tokens = max(0, int(prompt_token_count))
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
        """Load native model weights with the model's required Apple-Silicon runtime."""
        self.invalidate_prompt_reuse("model load")
        self.kv_cache_manager = None
        self.last_error = ""
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            self.last_error = "This MLX checkpoint requires an Apple Silicon Mac and its compatible MLX runtime."
            self.is_mlx_available = False
            return False
        if not os.path.exists(self.model_path) and os.getenv("OFFLINE", "0") == "1":
            self.last_error = "The selected MLX checkpoint is not available locally in offline mode."
            self.is_mlx_available = False
            return False

        runtime = resolve_controller_runtime(self.model_info, str(self.model_path or ""))
        specialized_vlm = runtime in {"mlx_vlm", "mlx_repo_vlm", "jang_vlm"}
        load_stamps = (_local_artifact_stamp(self.model_path), _local_artifact_stamp(self.adapter_path))

        try:
            from core.memory_watchdog import SystemMemoryWatchdog
            SystemMemoryWatchdog.adjust_dynamic_metal_headroom()

            if specialized_vlm:
                loader = UniversalControllerBackend(
                    model_path=self.model_path,
                    model_info={**self.model_info, "controller_runtime": runtime},
                )
                if not loader.load_model():
                    raise RuntimeError(f"{runtime} loader did not produce a usable model")
                self._vlm_loader = loader
                self.model = loader.model
                self.processor = loader.processor
                self.tokenizer = loader.tokenizer
                self._vlm_runtime = runtime
            else:
                import mlx_lm
                self.model, self.tokenizer = mlx_lm.load(
                    self.model_path,
                    adapter_path=(
                        self.adapter_path
                        if self.adapter_path and os.path.exists(self.adapter_path)
                        else None
                    ),
                )
                self.processor = None
                self._vlm_runtime = None
                self._vlm_loader = None

            self.is_mlx_available = self.model is not None and self.tokenizer is not None
        except Exception as exc:
            _base.logger.exception("MLX model load failed (full traceback)")
            failure = str(exc)
            try:
                self.unload_model()  # Drop partial model and cache owners on failed load.
            except Exception as cleanup_error:
                failure += f" (partial-load cleanup: {cleanup_error})"
            self.last_error = failure
            self.is_mlx_available = False
            return False

        if self.is_mlx_available and self._vlm_runtime is None:
            self.kv_cache_manager = SmartKVCacheManager(
                self.model,
                max_tokens=self.max_stateful_kv_tokens,
            )
        else:
            self.kv_cache_manager = None
        if self.is_mlx_available:
            after_stamps = (_local_artifact_stamp(self.model_path), _local_artifact_stamp(self.adapter_path))
            self._prompt_disk_pristine = load_stamps == after_stamps and None not in load_stamps
            self._prompt_load_stamps = load_stamps
            self._prompt_disk_model_id = id(self.model)
            self._prompt_disk_tokenizer_id = id(self.tokenizer)
            from core.mlx_warmup import SafeMLXWarmup
            self._metal_warmup = SafeMLXWarmup(enabled=self._lossless_speedups["metal_warmup"])
        return bool(self.is_mlx_available)

    def unload_model(self):
        """Release *all* model owners, including nested VLM and KV references.

        Clearing only ``self.model`` left ``_vlm_loader.model`` and
        ``kv_cache_manager.model`` alive in unified memory. A real failure is
        propagated to the caller rather than displaying a false unloaded state.
        """
        import gc
        from core.native_speed import runtime_lock
        with runtime_lock(self):
            self.invalidate_prompt_reuse("unload")
            loader = self._vlm_loader
            if loader is not None:
                unload = getattr(loader, "unload_model", None)
                if callable(unload):
                    unload()
            self.kv_cache_manager = None
            self._vlm_loader = None
            self._vlm_runtime = None
            self.model = None
            self.processor = None
            self.tokenizer = None
            self.is_mlx_available = False
            self._prompt_disk_model_id = None
            self._prompt_disk_tokenizer_id = None
            self._prompt_load_stamps = None
            self.last_generation_tokens = 0
            self.last_prompt_token_count = None
            self.last_tok_per_sec = 0.0
            del loader
            gc.collect()
            try:
                import mlx.core as mx
                clear = getattr(mx, "clear_cache", None)
                if not callable(clear):
                    clear = getattr(getattr(mx, "metal", None), "clear_cache", None)
                if callable(clear):
                    clear()
            except ImportError:
                pass
            return {"status": "unloaded"}

    def get_training_model(self):
        """Return the real causal language module used for text Learn/RSI updates."""
        if self._vlm_runtime is not None:
            language_model = getattr(self.model, "language_model", None)
            if language_model is not None:
                return language_model
        return self.model

    def _run_training_method(self, method_name: str, *args, **kwargs):
        # Invalidate even on failed/partial updates: weights may already differ.
        if method_name in {"inject_lora_adapters", "train_mini_batch"}:
            self.invalidate_prompt_reuse("training or adapter mutation")
        target = self.get_training_model()
        if target is None or target is self.model:
            return getattr(super(), method_name)(*args, **kwargs)
        wrapper = self.model
        self.model = target
        try:
            return getattr(super(), method_name)(*args, **kwargs)
        finally:
            self.model = wrapper

    def inject_lora_adapters(self, r: int = 8, scale: float = 2.0):
        from core.native_speed import runtime_lock
        with runtime_lock(self):
            return self._run_training_method("inject_lora_adapters", r=r, scale=scale)

    def compute_mlx_fisher(self, anchor_texts):
        return self._run_training_method("compute_mlx_fisher", anchor_texts)

    def train_mini_batch(self, adapters, data, **kwargs):
        return self._run_training_method("train_mini_batch", adapters, data, **kwargs)

    def _prepare_vlm_prompt(self, prompt: str, num_images: int = 0) -> str:
        if self._vlm_loader is not None:
            try:
                return self._vlm_loader._mlx_prompt(prompt, num_images)
            except Exception:
                pass
        return prompt

    def supports_media_input(self, kind: str) -> bool:
        if self._vlm_loader is None:
            return False
        return bool(self._vlm_loader.supports_media_input(kind))

    def review_media_input(self, path: str, kind: str, prompt: str = ""):
        self._vlm_prefixes.clear()
        if self._vlm_loader is None:
            return {
                "perception_available": False,
                "reason": f"Loaded MLX controller has no {kind} perception runtime.",
            }
        return self._vlm_loader.review_media_input(path=path, kind=kind, prompt=prompt)

    def calculate_token_entropy(self, prompt: str) -> float:
        if not self.is_mlx_available or self.model is None:
            return 0.45
        try:
            import math
            import mlx.core as mx
            tokens = self._encode_prompt(prompt)
            if not tokens:
                return 0.45
            lm = getattr(self.model, "language_model", self.model)
            logits = lm(mx.array([tokens]))
            logits = getattr(logits, "logits", logits)
            last_logits = logits[:, -1, :]
            probs = mx.softmax(last_logits, axis=-1)
            entropy_val = -mx.sum(probs * mx.log(probs + 1e-12), axis=-1).item()
            if math.isnan(entropy_val) or math.isinf(entropy_val):
                return 0.45
            return float(entropy_val)
        except Exception:
            return 0.45

    def generate_branches(
        self,
        prompt: str,
        branch_count: int = 16,
        max_tokens: int = 512,
        temperature=0.75,
        top_p: float = 0.92,
        progress_callback=None,
        stop_at_tool_boundary: bool = False,
        cancelled=None,
    ):
        """Generate Pro branches without silent quality-reducing fallbacks."""
        if not self.is_mlx_available or self.model is None or (callable(cancelled) and cancelled()):
            return []

        import mlx.core as mx
        self._prepare_optional_warmup(mx)
        from core.tool_boundary_stop import ToolBoundaryStopper
        from core.generation_metrics import TokenMeter, observe_mlx_response

        if self._vlm_runtime is not None:
            from mlx_vlm.generate import stream_generate as vlm_stream_generate
            count = max(1, min(int(branch_count), 16))
            temps = list(temperature) if isinstance(temperature, (list, tuple)) else [float(temperature)] * count
            try:
                prompt_ids = self._encode_prompt(prompt)
            except Exception:
                prompt_ids = []
            self.last_prompt_token_count = len(prompt_ids) if prompt_ids else None
            effective_max = self._effective_generation_cap(
                prompt,
                max_tokens,
                prompt_token_count=len(prompt_ids),
            )
            branches = []
            prepared_prompt = self._prepare_vlm_prompt(prompt, 0)
            for idx in range(count):
                if callable(cancelled) and cancelled():
                    return branches
                temp_value = float(temps[idx % len(temps)])
                pieces = []
                meter = TokenMeter(self, "mlx-vlm")
                started = time.perf_counter()
                generated = 0
                last = None
                iterator = None
                completed = False
                boundary = ToolBoundaryStopper() if stop_at_tool_boundary else None
                prefill_kwargs = self._speed_vlm_prefill_kwargs()
                try:
                    iterator = vlm_stream_generate(
                        self.model,
                        self.processor,
                        prompt=prepared_prompt,
                        image=[],
                        max_tokens=effective_max,
                        temperature=temp_value,
                        top_p=top_p,
                        **prefill_kwargs,
                        **self._vlm_reuse_kwargs(prepared_prompt, prefill_kwargs, branches=True),
                    )
                    for response in iterator:
                        last = response
                        generated += 1
                        observe_mlx_response(meter, response)
                        if progress_callback is not None:
                            progress_callback(idx, count, meter.tokens, time.perf_counter() - started, finished=False)
                        if callable(cancelled) and cancelled():
                            return branches
                        self.last_prompt_reused_tokens = int(getattr(response, "cached_tokens", 0) or 0)
                        piece = str(getattr(response, "text", response))
                        pieces.append(piece)
                        if boundary is not None and boundary.feed(piece):
                            pieces = [boundary.text]
                            break
                    completed = True
                finally:
                    if iterator is not None and callable(getattr(iterator, "close", None)):
                        iterator.close()
                    self._finish_vlm_request(completed=completed, branches=idx < count - 1)
                    meter.finish()
                elapsed = max(0.001, time.perf_counter() - started)
                branches.append("".join(pieces))
                if progress_callback is not None:
                    progress_callback(idx + 1, count, meter.tokens, elapsed)
                _reclaim_if_needed(mx)
            return branches

        import mlx_lm
        try:
            from mlx_lm.sample_utils import make_sampler
        except Exception as exc:
            raise RuntimeError("MLX sampler unavailable; refusing to change Pro sampling semantics") from exc

        count = max(1, min(int(branch_count), 16))
        temps = list(temperature) if isinstance(temperature, (list, tuple)) else [float(temperature)] * count
        branches = []
        prompt_ids = self._encode_prompt(prompt)
        self.last_prompt_token_count = len(prompt_ids)
        effective_max = self._effective_generation_cap(
            prompt,
            max_tokens,
            prompt_token_count=len(prompt_ids),
        )

        def _one(temp_value: float, prefill_step_size: int) -> str:
            sampler = make_sampler(temp=float(temp_value), top_p=top_p)
            branch_prompt = self._prepare_text_cache(prompt_ids, prefill_step_size, branches=True)
            kwargs = {
                "prompt": branch_prompt,
                "max_tokens": effective_max,
                "sampler": sampler,
                "prefill_step_size": int(prefill_step_size),
                "prompt_cache": self.kv_cache_manager.get_cache(),
            }
            pieces = []
            boundary = ToolBoundaryStopper() if stop_at_tool_boundary else None
            response = None
            generated = 0
            meter = TokenMeter(self, "mlx-lm")
            started = time.perf_counter()
            iterator = None
            try:
                iterator = mlx_lm.stream_generate(self.model, self.tokenizer, **kwargs)
                for response in iterator:
                    generated += 1
                    observe_mlx_response(meter, response)
                    if progress_callback is not None:
                        progress_callback(idx, count, meter.tokens, time.perf_counter() - started, finished=False)
                    if callable(cancelled) and cancelled():
                        return ""
                    try:
                        measured = float(getattr(response, "generation_tps", 0.0) or 0.0)
                    except Exception:
                        measured = 0.0
                    if measured > 0.0:
                        self.last_tok_per_sec = measured
                    chunk = getattr(response, "text", None)
                    if chunk is None:
                        chunk = str(response)
                    piece = str(chunk)
                    pieces.append(piece)
                    if boundary is not None and boundary.feed(piece):
                        pieces = [boundary.text]
                        break
            finally:
                if iterator is not None and callable(getattr(iterator, "close", None)):
                    iterator.close()
                self._release_text_cache(branches=True)
                meter.finish()

            elapsed = max(0.001, time.perf_counter() - started)
            return "".join(pieces)

        for idx in range(count):
            if callable(cancelled) and cancelled():
                return branches
            _reclaim_if_needed(mx)
            temp_value = float(temps[idx % len(temps)])
            try:
                branches.append(_one(temp_value, self._speed_prefill_size()))
            except RuntimeError as exc:
                if not _is_metal_oom(exc):
                    raise
                # Lossless OOM retry: same prompt, same sampler policy, same token
                # budget and the same TurboQuant cache policy. Only prefill chunking changes.
                self._prompt_prefixes.clear()
                self._release_text_cache(branches=True)
                _reclaim_if_needed(mx, force=True)
                branches.append(_one(temp_value, 64))
            finally:
                self._release_text_cache(branches=idx < count - 1)
                _reclaim_if_needed(mx)
            if callable(cancelled) and cancelled():
                return branches[:-1]
            if progress_callback is not None:
                progress_callback(idx + 1, count, self.last_generation_tokens, self.last_generation_seconds)

        return branches

    def stream_generate_tokens(self, prompt: str, max_tokens: int = 512,
                               temperature: float = 0.75, top_p: float = 0.92):
        if not self.is_mlx_available or self.model is None:
            return

        from core.generation_metrics import TokenMeter, observe_mlx_response
        import mlx.core as mx
        self._prepare_optional_warmup(mx)

        try:
            prompt_ids = self._encode_prompt(prompt)
        except Exception:
            prompt_ids = []
        prompt_len = len(prompt_ids)
        self.last_prompt_token_count = prompt_len if prompt_ids else None

        effective_max = self._effective_generation_cap(
            prompt,
            max_tokens,
            prompt_token_count=prompt_len,
        )

        if self._vlm_runtime is not None:
            from mlx_vlm.generate import stream_generate as vlm_stream_generate
            meter = TokenMeter(self, "mlx-vlm")
            generated = 0
            started = time.perf_counter()
            last = None
            self.last_tok_per_sec = 0.0
            prepared_prompt = self._prepare_vlm_prompt(prompt, 0)
            prefill_kwargs = self._speed_vlm_prefill_kwargs()
            iterator = None
            completed = False
            try:
                iterator = vlm_stream_generate(
                    self.model,
                    self.processor,
                    prompt=prepared_prompt,
                    image=[],
                    max_tokens=effective_max,
                    temperature=temperature,
                    top_p=top_p,
                    **prefill_kwargs,
                    **self._vlm_reuse_kwargs(prepared_prompt, prefill_kwargs),
                )
                for response in iterator:
                    last = response
                    generated += 1
                    observe_mlx_response(meter, response)
                    self.last_prompt_reused_tokens = int(getattr(response, "cached_tokens", 0) or 0)
                    chunk = getattr(response, "text", None)
                    yield str(chunk if chunk is not None else response)
                completed = True
            finally:
                if iterator is not None and callable(getattr(iterator, "close", None)):
                    iterator.close()
                self._finish_vlm_request(completed=completed)
                elapsed = max(0.001, time.perf_counter() - started)
                meter.finish()
                _reclaim_if_needed(mx)
            return

        import mlx_lm
        if not prompt_ids:
            raise RuntimeError("Text prompt tokenization produced no tokens; refusing an uncounted context")

        try:
            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(temp=temperature, top_p=top_p)
        except Exception as exc:
            raise RuntimeError("MLX sampler unavailable; refusing to change sampling semantics") from exc

        meter = TokenMeter(self, "mlx-lm")
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
            suffix = self._prepare_text_cache(prompt_ids, prefill_step_size)
            return mlx_lm.stream_generate(
                self.model,
                self.tokenizer,
                prompt=suffix,
                prompt_cache=self.kv_cache_manager.get_cache(),
                **kwargs,
            )

        def _yield_response(resp):
            observe_mlx_response(meter, resp)
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
                iterator = _run_stream(self._speed_prefill_size())
                for response in iterator:
                    generated += 1
                    yield _yield_response(response)
            except RuntimeError as exc:
                if not _is_metal_oom(exc) or generated:
                    raise
                # Retry only before any token has been emitted. Rebuild a fresh cache
                # and reduce transient prefill memory without changing cache policy,
                # context, sampler or requested generation length.
                if iterator is not None and callable(getattr(iterator, "close", None)):
                    iterator.close()
                self._release_text_cache()
                self._prompt_prefixes.clear()
                _reclaim_if_needed(mx, force=True)
                iterator = _run_stream(_adaptive_prefill_step_size(aggressive=True))
                for response in iterator:
                    generated += 1
                    yield _yield_response(response)
        except Exception as exc:
            _base.logger.error("MLX stream_generate error: %s", exc)
            raise
        finally:
            if iterator is not None and callable(getattr(iterator, "close", None)):
                iterator.close()
            elapsed = max(0.001, time.perf_counter() - started)
            meter.finish()
            self._release_text_cache()
            _reclaim_if_needed(mx)
