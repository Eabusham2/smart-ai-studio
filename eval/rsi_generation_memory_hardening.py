"""Memory-safe RSI/Phase-4 branch generation without reducing reasoning length.

The corrected legacy RSI core already has the right search semantics: branches are
sequential, four temperatures are explored by RSI, answer-blind selection happens
before hidden reward, and round two self-critiques the selected prior attempt. The
later live-stream wrapper preserved those semantics but stopped owning a hard
per-branch teardown boundary and used an unbounded full-precision KV cache.

This layer keeps the current algorithm and live output while restoring the useful
legacy branch isolation and adding current MLX-LM memory controls:
- fresh sequential branch generation;
- hardware-scaled active KV residency;
- 4-bit KV cache quantization when supported;
- smaller prefill chunks to reduce peak unified-memory use;
- explicit generator teardown + Metal/Python cache cleanup after every branch;
- the shared Metal lock around each branch;
- one more-conservative retry of only the current branch on a real Metal OOM.

The generation allowance is NOT shortened. ``max_tokens`` is passed through
unchanged, so the benchmark/RSI ceiling remains 16,384 tokens.
"""
from __future__ import annotations

import contextlib
import gc
from typing import Any, Dict, List

from core.kv_cache_manager import compute_auto_kv_budget


_OOM_MARKERS = (
    "insufficient memory",
    "outofmemory",
    "out of memory",
    "kiogpucommandbuffercallbackerroroutofmemory",
)


def _is_metal_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold().replace(" ", "")
    return any(marker.replace(" ", "") in text for marker in _OOM_MARKERS)


def _clear_runtime_memory(mx: Any) -> None:
    """Release Python refs and MLX's cached Metal allocations at branch boundaries."""
    try:
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass
    gc.collect()


def _engine_kv_budget(self) -> int:
    """Use the existing app/backend hardware policy even when Pro backend is not built yet."""
    backend = getattr(self, "_phase4_pro_backend", None)
    try:
        value = int(getattr(backend, "max_stateful_kv_tokens", 0) or 0)
    except Exception:
        value = 0
    if value <= 0:
        try:
            value = int(compute_auto_kv_budget())
        except Exception:
            value = 2048
    return max(512, value)


def _memory_policy(self, *, aggressive: bool = False) -> Dict[str, int]:
    """Bound resident KV, not output length.

    On a <=16 GB machine the existing backend policy is 2048 KV tokens. The
    normal path uses that budget with 4-bit KV and 256-token prefill chunks. A
    branch that still hits a genuine Metal OOM gets one retry at half the active
    KV residency and 128-token prefill chunks. Both paths retain the caller's
    full generation token allowance.
    """
    budget = _engine_kv_budget(self)
    if aggressive:
        budget = max(512, budget // 2)
    prefill = 128 if aggressive else (256 if budget <= 2048 else 512)
    quant_start = min(512, max(128, budget // 4))
    return {
        "max_kv_size": int(budget),
        "prefill_step_size": int(prefill),
        "kv_bits": 4,
        "kv_group_size": 64,
        "quantized_kv_start": int(quant_start),
    }


def _close_iterator(iterator: Any) -> None:
    try:
        closer = getattr(iterator, "close", None)
        if callable(closer):
            closer()
    except Exception:
        pass


def install(phase4_module, live_module, legacy_generate) -> None:
    """Replace only the branch execution wrapper; leave RSI search/reward logic intact."""
    if getattr(phase4_module, "_rsi_generation_memory_hardening_installed", False):
        return

    def safe_generate_branches_same_model(
        self,
        formatted_prompt: str,
        temperatures: List[float],
        max_tokens: int,
        top_p: float = 0.92,
    ) -> List[str]:
        if (
            not phase4_module.MLX_AVAILABLE
            or self.engine.model is None
            or self.engine.tokenizer is None
        ):
            raise RuntimeError("MLX model/tokenizer unavailable for RSI/Pro branching")

        mlx_lm = phase4_module.mlx_lm
        mx = phase4_module.mx
        stream_generate = getattr(mlx_lm, "stream_generate", None)
        try:
            from mlx_lm.sample_utils import make_sampler
        except Exception:
            make_sampler = None

        # Legacy direct generation remains the compatibility path for older MLX-LM.
        # It is still run one branch at a time with the stronger post-branch teardown.
        use_stream = callable(stream_generate) and make_sampler is not None
        branches: List[str] = []
        total_branches = len(temperatures)
        lock = getattr(phase4_module, "METAL_STREAM_LOCK", None)

        def lock_context():
            return lock if hasattr(lock, "__enter__") else contextlib.nullcontext()

        def legacy_one(temp: float) -> str:
            _clear_runtime_memory(mx)
            try:
                with lock_context():
                    out = legacy_generate(
                        self,
                        formatted_prompt,
                        [temp],
                        max_tokens,
                        top_p,
                    )
                return str(out[0]) if out else ""
            finally:
                _clear_runtime_memory(mx)

        def streamed_one(temp: float, branch_idx: int, *, aggressive: bool) -> str:
            policy = _memory_policy(self, aggressive=aggressive)
            sampler = make_sampler(temp=float(temp), top_p=top_p)
            label = (
                f"live branch {branch_idx}/{total_branches} | T={float(temp):.2f} | "
                f"KV={policy['max_kv_size']} | KV4 | prefill={policy['prefill_step_size']}"
            )
            live_module._write_live_header(self, formatted_prompt, branch_label=label)

            iterator = None
            response = None
            pieces: List[str] = []
            _clear_runtime_memory(mx)
            try:
                kwargs: Dict[str, Any] = {
                    "prompt": formatted_prompt,
                    # Deliberately unchanged: this remains 16,384 when RSI passes 16,384.
                    "max_tokens": max(1, int(max_tokens)),
                    "sampler": sampler,
                    **policy,
                }
                with lock_context():
                    iterator = stream_generate(
                        self.engine.model,
                        self.engine.tokenizer,
                        **kwargs,
                    )
                    for response in iterator:
                        chunk = getattr(response, "text", None)
                        if chunk is None:
                            chunk = str(response)
                        chunk = str(chunk)
                        pieces.append(chunk)
                        live_module._append_live_text(chunk)
                return "".join(pieces)
            finally:
                # The last GenerationResponse can hold logprob/Metal-backed state.
                response = None
                _close_iterator(iterator)
                iterator = None
                sampler = None
                _clear_runtime_memory(mx)

        for branch_idx, temp in enumerate(temperatures, 1):
            if not use_stream:
                branches.append(legacy_one(float(temp)))
                continue

            try:
                branches.append(streamed_one(float(temp), branch_idx, aggressive=False))
            except TypeError:
                # Older MLX-LM may not support the modern KV/prefill kwargs. Keep
                # corrected legacy semantics rather than inventing another sampler.
                branches.append(legacy_one(float(temp)))
            except RuntimeError as exc:
                if not _is_metal_oom(exc):
                    raise
                # Recover only the failed branch. Completed branches stay valid and
                # the item/round search policy is unchanged.
                _clear_runtime_memory(mx)
                try:
                    live_module._append_live_text(
                        "\n[RSI MEMORY RECOVERY] Metal OOM: retrying this branch with "
                        "more conservative KV residency; 16,384-token allowance unchanged.\n"
                    )
                except Exception:
                    pass
                branches.append(streamed_one(float(temp), branch_idx, aggressive=True))

        return branches

    phase4_module._generate_branches_same_model = safe_generate_branches_same_model
    phase4_module._rsi_generation_memory_hardening_installed = True
