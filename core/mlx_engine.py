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
    from core.mlx_memory_guard import memory_headroom
    return memory_headroom().pressure


def _adaptive_prefill_step_size(*, aggressive: bool = False) -> int:
    """Use larger MLX prefill batches when memory headroom exists.

    This changes only prompt chunking, not weights, context, sampling, or token limits.
    MLX-LM itself defaults to 2048; we step down conservatively as memory tightens.
    """
    if aggressive:
        return 128
    from core.mlx_memory_guard import memory_headroom
    return memory_headroom().prefill_tokens


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
        self._prompt_prefixes = ExactPrefixStore(max_tokens=self.max_stateful_kv_tokens,
                                                memory_admission=self._prefix_copy_allowed)
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
        self._oom_prefill_step_size = None
        self.last_memory_recovery = ""

    def _prefix_copy_allowed(self, size):
        from core.mlx_memory_guard import memory_headroom
        return (not getattr(self, "_oom_prefill_step_size", None)
                and memory_headroom(max_process_gb=getattr(self, "speed_memory_limit_gb", None)).can_snapshot(size))

    def _prepare_generation_memory(self, mx):
        """Admit optional state before every turn, even when the watchdog is off."""
        from core.mlx_memory_guard import trim_allocator
        headroom = trim_allocator(mx, max_process_gb=getattr(self, "speed_memory_limit_gb", None))
        if headroom.pressure or getattr(self, "_oom_prefill_step_size", None):
            self._prompt_prefixes.clear()
            self._vlm_prefixes.clear()
            self._metal_warmup.close()
            _reclaim_if_needed(mx, force=True)
        self._request_prefill_step_size = headroom.prefill_tokens
        return headroom

    def _recover_generation_memory(self, mx):
        """Run after the failed iterator and exception scope have been released."""
        from core.mlx_memory_guard import trim_allocator
        self._oom_prefill_step_size = _adaptive_prefill_step_size(aggressive=True)
        self._prompt_prefixes.clear()
        self._vlm_prefixes.clear()
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.release()
        self._metal_warmup.close()
        self.last_prompt_reused_tokens = 0
        self.last_prompt_reuse_reason = "OOM recovery: optional snapshots released"
        self.last_memory_recovery = (
            "Temporary inference state released after Metal ran out of memory. "
            "Subsequent requests use conservative prefill and no optional cache copies; "
            "the selected model and full conversation are unchanged."
        )
        gc.collect(2)
        trim_allocator(mx, force=True, max_process_gb=getattr(self, "speed_memory_limit_gb", None))

    def _stream_with_memory_retry(self, mx, create_iterator):
        """Release a failed native request before one pre-output retry.

        Keeping the caught exception alive also keeps its model-forward frames
        (and tensors) alive. Recovery therefore runs after that exception scope,
        with the native generator closed and its traceback frames cleared.
        A response already delivered to a caller is never generated again.
        """
        import traceback

        emitted = False
        for attempt in range(2):
            iterator = response = None
            failure = None
            try:
                self._prepare_generation_memory(mx)
                iterator = create_iterator()
                for response in iterator:
                    emitted = True
                    yield response
                    response = None
            except RuntimeError as exc:
                if not _is_metal_oom(exc):
                    raise
                failure = str(exc)
                traceback.clear_frames(exc.__traceback__)
                exc.__traceback__ = None
            finally:
                try:
                    close = getattr(iterator, "close", None)
                    if callable(close):
                        close()
                except RuntimeError as exc:
                    if not _is_metal_oom(exc):
                        raise
                    failure = failure or str(exc)
                    traceback.clear_frames(exc.__traceback__)
                    exc.__traceback__ = None
                finally:
                    iterator = response = close = None
            if failure is None:
                return
            self._recover_generation_memory(mx)
            if attempt or emitted:
                # No context truncation, weight conversion, or silent restart
                # after output. A later user request can use the cleaned state.
                raise RuntimeError(failure) from None

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
            memory_ok=not _memory_pressure() and self._prefix_copy_allowed(self._vlm_prefixes.max_bytes))
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
                                   memory_ok=lambda: not _memory_pressure() and not getattr(self, "_oom_prefill_step_size", None))

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
        if (not reuse or len(prompt_ids) < 2 or _memory_pressure()
                or getattr(self, "_oom_prefill_step_size", None) or getattr(self.model, "training", False)):
            if (_memory_pressure() or getattr(self, "_oom_prefill_step_size", None)
                    or getattr(self.model, "training", False)):
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
        if getattr(self, "_request_prefill_step_size", None):
            safe = min(safe, self._request_prefill_step_size)
        if getattr(self, "_oom_prefill_step_size", None):
            safe = min(safe, self._oom_prefill_step_size)
        preferred = getattr(self, "_speed_prefill_step_size", None)
        current_context = int(getattr(self, "context_budget_tokens", 0) or 32768)
        if getattr(self, "_speed_profile_context", None) != current_context:
            return safe
        return min(safe, int(preferred)) if preferred else safe

    def _speed_vlm_prefill_kwargs(self) -> Dict[str, int]:
        """Bound a supported native prefill control even without calibration."""
        from core.mlx_memory_guard import native_vlm_prefill_default

        if any(getattr(owner, "no_chunked_prefill", False) for owner in
               (self.model, getattr(self.model, "language_model", None))):
            return {}
        default = native_vlm_prefill_default()
        if default is None:
            return {}
        current_context = int(getattr(self, "context_budget_tokens", 0) or 32768)
        preferred = getattr(self, "_speed_prefill_step_size", None)
        if not preferred or getattr(self, "_speed_profile_context", None) != current_context:
            preferred = default
        safe = _adaptive_prefill_step_size()
        if getattr(self, "_request_prefill_step_size", None):
            safe = min(safe, self._request_prefill_step_size)
        if getattr(self, "_oom_prefill_step_size", None):
            safe = min(safe, self._oom_prefill_step_size)
        selected = min(int(preferred), safe)
        return {} if selected == default else {"prefill_step_size": selected}

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

    def _resolve_mlx_source(self):
        """Bind identity and load weights from the very same local snapshot."""
        from pathlib import Path
        from core.learning_state import cached_model_source, requested_revision

        if platform.system() != "Darwin" or platform.machine() != "arm64":
            return None
        identifier = str(self.model_path)
        path = Path(identifier).expanduser()
        if path.exists():
            return str(path.absolute())
        revision = requested_revision(self.model_info)
        cached = cached_model_source(identifier, self.model_info)
        required = self.model_info.get("download_required_files") or []
        if cached.is_dir():
            if not revision and cached.parent.name == "snapshots":
                revision = cached.name
            if any(cached.rglob("*.safetensors")) and all((cached / f).is_file() for f in required):
                return str(cached)
        if os.getenv("OFFLINE", "0") == "1":
            raise RuntimeError("The selected MLX snapshot is not available locally in offline mode")
        from huggingface_hub import snapshot_download
        patterns = required or ["*.safetensors", "*.json", "*.jsonl", "*.txt", "*.model", "*.tiktoken",
                                "*.jinja", "*.py", "*.metal", "*.toml", "*.yaml", "*.yml", "runtime/*"]
        return str(snapshot_download(repo_id=identifier, revision=revision, allow_patterns=patterns))

    def load_model(self) -> bool:
        """Load native model weights with the model's required Apple-Silicon runtime."""
        self.invalidate_prompt_reuse("model load")
        self.kv_cache_manager = None
        self.last_error = ""
        self._oom_prefill_step_size = None
        self.last_memory_recovery = ""
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            self.last_error = "This MLX checkpoint requires an Apple Silicon Mac and its compatible MLX runtime."
            self.is_mlx_available = False
            return False
        runtime = resolve_controller_runtime(self.model_info, str(self.model_path or ""))
        specialized_vlm = runtime in {"mlx_vlm", "mlx_repo_vlm", "jang_vlm"}
        load_stamps = (_local_artifact_stamp(self.model_path), _local_artifact_stamp(self.adapter_path))

        try:
            from core.memory_watchdog import SystemMemoryWatchdog
            SystemMemoryWatchdog.adjust_dynamic_metal_headroom()

            source = getattr(self, "_mlx_loaded_source", None) or self._resolve_mlx_source()
            if specialized_vlm:
                loader = UniversalControllerBackend(
                    model_path=source,
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
                    source,
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
            self.adapters = {}
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

    def run_on_mlx_owner(self, operation, *args, **kwargs):
        """Perform a model-owned GPU operation without evaluating arrays elsewhere."""
        owner = getattr(self, "_mlx_owner", None)
        if owner is None:
            raise RuntimeError("MLX model owner is unavailable; load the model first")
        return owner.call(operation, *args, **kwargs)

    def switch_learning_checkpoint(self, scope):
        from core.mlx_adapter_persistence import switch_learning_checkpoint
        return self.run_on_mlx_owner(switch_learning_checkpoint, self, scope)

    def get_training_model(self):
        """Return the real causal language module used for text Learn/RSI updates."""
        if self._vlm_runtime is not None:
            language_model = getattr(self.model, "language_model", None)
            if language_model is not None:
                return language_model
        return self.model

    def _run_training_method(self, method_name: str, *args, **kwargs):
        # Invalidate even on failed/partial updates: weights may already differ.
        if method_name in {"inject_lora_adapters", "compute_mlx_fisher", "train_mini_batch"}:
            self.invalidate_prompt_reuse("training or adapter mutation")
        from core.policy_reference import private_text_positions
        target = self.get_training_model()
        if target is None or target is self.model:
            with private_text_positions(target):
                return getattr(super(), method_name)(*args, **kwargs)
        wrapper = self.model
        validation = kwargs.get("validation_callback")
        if validation is not None:
            def validate_wrapper(backend):
                backend.model = wrapper
                try:
                    return validation(backend)
                finally:
                    backend.model = target
            kwargs["validation_callback"] = validate_wrapper
        self.model = target
        try:
            with private_text_positions(target):
                return getattr(super(), method_name)(*args, **kwargs)
        finally:
            self.model = wrapper

    def inject_lora_adapters(self, r: int = 8, scale: float = 2.0, layout=None, target_paths=None, *, learning_mode=None):
        from core.native_speed import runtime_lock
        with runtime_lock(self):
            return self._run_training_method("inject_lora_adapters", r=r, scale=scale, layout=layout, target_paths=target_paths, learning_mode=learning_mode)

    def compute_mlx_fisher(self, anchor_texts, **kwargs):
        return self._run_training_method("compute_mlx_fisher", anchor_texts, **kwargs)

    def score_learning_probe(self, prompt, completion, **kwargs):
        def score():
            from core.native_speed import runtime_lock
            with runtime_lock(self):
                return self._run_training_method("score_learning_probe", prompt, completion, **kwargs)
        return self.run_on_mlx_owner(score)

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
        mx = logits = last_logits = probs = None
        memory_failed = False
        try:
            import math
            import mlx.core as mx
            from core.mlx_memory_guard import native_last_logits_kwargs
            self._prepare_generation_memory(mx)
            tokens = self._encode_prompt(prompt)
            if not tokens:
                return 0.45
            lm = getattr(self.model, "language_model", self.model)
            # Routing only consumes the last row. A supported native control
            # avoids allocating prompt-length x vocabulary logits while still
            # evaluating the complete prompt with the same decoder schedule.
            logits = lm(mx.array([tokens]), **native_last_logits_kwargs(lm))
            logits = getattr(logits, "logits", logits)
            last_logits = logits[:, -1, :]
            probs = mx.softmax(last_logits, axis=-1)
            entropy_val = -mx.sum(probs * mx.log(probs + 1e-12), axis=-1).item()
            if math.isnan(entropy_val) or math.isinf(entropy_val):
                return 0.45
            return float(entropy_val)
        except Exception as exc:
            if _is_metal_oom(exc):
                import traceback
                memory_failed = True
                traceback.clear_frames(exc.__traceback__)
                exc.__traceback__ = None
            return 0.45
        finally:
            # The route still uses the same full-prompt entropy calculation.
            # Its temporary logits must not stay live into the real response.
            logits = last_logits = probs = None
            if mx is not None:
                if memory_failed:
                    self._recover_generation_memory(mx)
                else:
                    _reclaim_if_needed(mx)

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
        self._prepare_generation_memory(mx)
        self._prepare_optional_warmup(mx)
        from core.tool_boundary_stop import ToolBoundaryStopper
        from core.generation_metrics import TokenMeter, observe_mlx_response

        if self._vlm_runtime is not None:
            from mlx_vlm.generate import stream_generate as vlm_stream_generate
            from core.vlm_prompt_tokens import prepare_vlm_inputs
            count = max(1, min(int(branch_count), 16))
            temps = list(temperature) if isinstance(temperature, (list, tuple)) else [float(temperature)] * count
            prepared_prompt = self._prepare_vlm_prompt(prompt, 0)
            prepared_inputs, prompt_count = prepare_vlm_inputs(self.model, self.processor, prepared_prompt)
            self.last_prompt_token_count = prompt_count
            effective_max = self._effective_generation_cap(
                prepared_prompt,
                max_tokens,
                prompt_token_count=prompt_count,
            )
            branches = []
            for idx in range(count):
                if callable(cancelled) and cancelled():
                    return branches
                temp_value = float(temps[idx % len(temps)])
                pieces = []
                meter = TokenMeter(self, "mlx-vlm", prompt_tokens=prompt_count)
                started = time.perf_counter()
                generated = 0
                iterator = None
                response = None
                completed = False
                boundary = ToolBoundaryStopper() if stop_at_tool_boundary else None

                def create_vlm_branch():
                    prefill_kwargs = self._speed_vlm_prefill_kwargs()
                    return vlm_stream_generate(
                        self.model,
                        self.processor,
                        prompt=prepared_prompt,
                        image=[],
                        max_tokens=effective_max,
                        temperature=temp_value,
                        top_p=top_p,
                        **prefill_kwargs,
                        **prepared_inputs,
                        **self._vlm_reuse_kwargs(prepared_prompt, prefill_kwargs, branches=True),
                    )

                try:
                    iterator = self._stream_with_memory_retry(mx, create_vlm_branch)
                    for response in iterator:
                        generated += 1
                        observe_mlx_response(meter, response)
                        if progress_callback is not None:
                            progress_callback(idx, count, meter.tokens, time.perf_counter() - started, finished=False)
                        if callable(cancelled) and cancelled():
                            return branches
                        self.last_prompt_reused_tokens = int(getattr(response, "cached_tokens", 0) or 0)
                        piece = str(getattr(response, "text", response))
                        response = None
                        pieces.append(piece)
                        if boundary is not None and boundary.feed(piece):
                            pieces = [boundary.text]
                            break
                    completed = True
                finally:
                    if iterator is not None and callable(getattr(iterator, "close", None)):
                        iterator.close()
                    iterator = response = None
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

        def _one(temp_value: float) -> str:
            sampler = make_sampler(temp=float(temp_value), top_p=top_p)

            def create_branch():
                prefill_step_size = self._speed_prefill_size()
                branch_prompt = self._prepare_text_cache(prompt_ids, prefill_step_size, branches=True)
                return mlx_lm.stream_generate(
                    self.model, self.tokenizer, prompt=branch_prompt,
                    max_tokens=effective_max, sampler=sampler,
                    prefill_step_size=int(prefill_step_size),
                    prompt_cache=self.kv_cache_manager.get_cache())
            pieces = []
            boundary = ToolBoundaryStopper() if stop_at_tool_boundary else None
            response = None
            generated = 0
            meter = TokenMeter(self, "mlx-lm", prompt_tokens=len(prompt_ids))
            started = time.perf_counter()
            iterator = None
            try:
                iterator = self._stream_with_memory_retry(mx, create_branch)
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
                    response = None
                    pieces.append(piece)
                    if boundary is not None and boundary.feed(piece):
                        pieces = [boundary.text]
                        break
            finally:
                if iterator is not None and callable(getattr(iterator, "close", None)):
                    iterator.close()
                iterator = response = None
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
                branches.append(_one(temp_value))
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
        self._prepare_generation_memory(mx)
        self._prepare_optional_warmup(mx)

        if self._vlm_runtime is not None:
            from mlx_vlm.generate import stream_generate as vlm_stream_generate
            from core.vlm_prompt_tokens import prepare_vlm_inputs
            prepared_prompt = self._prepare_vlm_prompt(prompt, 0)
            prepared_inputs, prompt_count = prepare_vlm_inputs(self.model, self.processor, prepared_prompt)
            effective_max = self._effective_generation_cap(
                prepared_prompt, max_tokens, prompt_token_count=prompt_count)
            meter = TokenMeter(self, "mlx-vlm", prompt_tokens=prompt_count)
            generated = 0
            started = time.perf_counter()
            self.last_tok_per_sec = 0.0
            iterator = None
            response = None
            completed = False

            def create_vlm_stream():
                prefill_kwargs = self._speed_vlm_prefill_kwargs()
                return vlm_stream_generate(
                    self.model,
                    self.processor,
                    prompt=prepared_prompt,
                    image=[],
                    max_tokens=effective_max,
                    temperature=temperature,
                    top_p=top_p,
                    **prefill_kwargs,
                    **prepared_inputs,
                    **self._vlm_reuse_kwargs(prepared_prompt, prefill_kwargs),
                )

            try:
                iterator = self._stream_with_memory_retry(mx, create_vlm_stream)
                for response in iterator:
                    generated += 1
                    observe_mlx_response(meter, response)
                    self.last_prompt_reused_tokens = int(getattr(response, "cached_tokens", 0) or 0)
                    chunk = getattr(response, "text", None)
                    piece = str(chunk if chunk is not None else response)
                    response = None
                    yield piece
                completed = True
            finally:
                if iterator is not None and callable(getattr(iterator, "close", None)):
                    iterator.close()
                iterator = response = None
                self._finish_vlm_request(completed=completed)
                elapsed = max(0.001, time.perf_counter() - started)
                meter.finish()
                _reclaim_if_needed(mx)
            return

        import mlx_lm
        prompt_ids = self._encode_prompt(prompt)
        if not prompt_ids:
            raise RuntimeError("Text prompt tokenization produced no tokens; refusing an uncounted context")
        effective_max = self._effective_generation_cap(
            prompt, max_tokens, prompt_token_count=len(prompt_ids))

        try:
            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(temp=temperature, top_p=top_p)
        except Exception as exc:
            raise RuntimeError("MLX sampler unavailable; refusing to change sampling semantics") from exc

        meter = TokenMeter(self, "mlx-lm", prompt_tokens=len(prompt_ids))
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
            iterator = self._stream_with_memory_retry(mx, lambda: _run_stream(self._speed_prefill_size()))
            for response in iterator:
                generated += 1
                piece = _yield_response(response)
                response = None
                yield piece
        except Exception as exc:
            _base.logger.error("MLX stream_generate error: %s", exc)
            raise
        finally:
            if iterator is not None and callable(getattr(iterator, "close", None)):
                iterator.close()
            iterator = response = None
            elapsed = max(0.001, time.perf_counter() - started)
            meter.finish()
            self._release_text_cache()
            _reclaim_if_needed(mx)
