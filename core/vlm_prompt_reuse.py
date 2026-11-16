"""Bounded native mlx-vlm checkpoints for repeated identical text prompts.

The processor, full prompt, model-specific RoPE handling and cache layout remain
owned by mlx-vlm. Only its public APC kwargs are supplied. One final prefill
checkpoint avoids introducing new chunk boundaries in hybrid/recurrent models.
A full-prompt tenant prevents cross-prompt continuation from changing those
boundaries. No model forward, tokenization, disk writer or background task runs
in this adapter.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json


class NativeVLMCacheReuse:
    def __init__(self, max_bytes=256 * 1024 * 1024):
        self.max_bytes = max(0, int(max_bytes))
        self._manager = None
        self._identity = None
        self.last_reason = "not requested"
        self.requests = 0

    def request_kwargs(self, model, processor, prepared_prompt, *, prefill_step_size=None,
                       revision=0, enabled=True, memory_ok=True):
        """Return native APC kwargs only for supported unmodified hybrid runtimes.

        The backend must call ``clear`` before mutation, cancellation/error,
        media input, unload, or speed/profile changes; callers hold its model lock.
        ``revision`` identifies any adapter/tokenizer/runtime mutation. This helper
        retains integer identities and the APC manager, never model/processor refs.
        """
        if not enabled or not memory_ok or not self.max_bytes:
            self.clear()
            self.last_reason = "disabled or memory pressure"
            return {}
        candidate = None
        try:
            if not isinstance(prepared_prompt, str) or not prepared_prompt:
                raise ValueError("requires an exact text prompt")
            generation = importlib.import_module("mlx_vlm.generate")
            stream = generation.stream_generate
            step = generation.generate_step
            stream_code = getattr(stream, "__code__", None)
            required_locals = {"apc_manager", "apc_tenant", "exact_checkpoint_lengths"}
            if stream_code is None or not required_locals.issubset(stream_code.co_varnames):
                raise ValueError("native stream lacks checkpoint APC integration")
            params = inspect.signature(step).parameters
            if not {"prompt_cache_checkpoint", "prompt_cache_checkpoint_lengths", "prefill_step_size"}.issubset(params):
                raise ValueError("native generation lacks final prefill checkpoints")
            chunk = prefill_step_size if prefill_step_size is not None else params["prefill_step_size"].default
            if type(chunk) is not int or chunk <= 0:
                raise ValueError("native chunk schedule is unknown or disabled")
            lm = getattr(model, "language_model", None)
            if lm is None or any(getattr(owner, "no_chunked_prefill", False) for owner in (model, lm)):
                raise ValueError("model does not support chunked native prefill")
            # Identity contains no strong references, and changing the full text
            # empties the one-entry store before requesting another checkpoint.
            fingerprint = hashlib.sha256(json.dumps(
                {"prompt": prepared_prompt, "chunk": chunk, "revision": revision},
                sort_keys=True, allow_nan=False, separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            identity = (id(model), id(processor), id(getattr(processor, "tokenizer", None)),
                        id(stream), id(step), fingerprint)
            if identity != self._identity:
                self.clear()
                apc = importlib.import_module("mlx_vlm.apc")
                coordinator_module = importlib.import_module("mlx_vlm.apc_coordinator")
                adapters = importlib.import_module("mlx_vlm.apc_adapters")
                native_cache = importlib.import_module("mlx_vlm.models.cache")
                if getattr(adapters, "ADAPTER_SCHEMA_VERSION", None) != 4:
                    raise ValueError("native APC snapshot schema has not been audited")
                if "overrides" not in inspect.signature(apc.APCManager).parameters:
                    raise ValueError("native APC cannot enforce bounded checkpoint settings")
                candidate = apc.APCManager(
                    num_blocks=0, disk=None,
                    overrides={"checkpoint_entries": 1, "checkpoint_interval_tokens": 0,
                               "checkpoint_guard_tokens": 1,
                               "memory_max_gb": self.max_bytes / (1024 ** 3)},
                )
                coordinator = coordinator_module.APCCoordinator(candidate, lm)
                if (not coordinator.is_checkpoint
                        or not callable(getattr(coordinator, "checkpoint_lengths", None))
                        or not callable(getattr(coordinator, "store_checkpoint", None))):
                    raise ValueError("native cache layout has no hybrid checkpoint contract")
                native_types = tuple(getattr(native_cache, name) for name in
                                     ("KVCache", "QuantizedKVCache", "ArraysCache")
                                     if isinstance(getattr(native_cache, name, None), type))
                fresh_cache = coordinator.fresh_cache()
                if (not fresh_cache or any(type(layer) not in native_types for layer in fresh_cache)
                        or not any(type(layer) is getattr(native_cache, "ArraysCache", None) for layer in fresh_cache)):
                    raise ValueError("custom/windowed cache layout has no audited snapshot contract")
                # Verify installed API obeys explicit settings rather than falling
                # through to environment defaults that introduce extra splits.
                if (candidate._exact_cache_max != 1 or candidate.checkpoint_interval_tokens != 0
                        or candidate.exact_cache_guard_tokens != 1 or candidate.disk is not None
                        or candidate.num_blocks != 0 or candidate.memory_max_bytes > self.max_bytes):
                    raise ValueError("native APC did not preserve requested bounds")
                self._manager = candidate
                candidate = None
                self._identity = identity
            self.requests += 1
            self.last_reason = "native identical-prompt hybrid checkpoint"
            return {"apc_manager": self._manager, "apc_tenant": "studio-exact-v1:" + fingerprint}
        except Exception as exc:
            if candidate is not None:
                self._release(candidate)
            self.clear()
            self.last_reason = str(exc)
            return {}

    @staticmethod
    def _release(manager):
        for method_name in ("clear", "close"):
            try:
                method = getattr(manager, method_name, None)
                if callable(method):
                    method()
            except Exception:
                pass

    def clear(self):
        manager, self._manager = self._manager, None
        self._identity = None
        if manager is not None:
            self._release(manager)

    def telemetry(self):
        return {"vlm_checkpoint_requests": self.requests,
                "vlm_checkpoint_enabled": self._manager is not None,
                "vlm_checkpoint_max_bytes": self.max_bytes,
                "vlm_checkpoint_reason": self.last_reason}
