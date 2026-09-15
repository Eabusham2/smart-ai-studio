"""Memory-safe RSI/Phase-4 branch generation without reducing reasoning length.

The current RSI search/reward algorithm is left untouched. This layer restores the
strong branch-lifecycle behavior from the much older MLX implementation around the
current live-stream generator, while using modern MLX-LM KV controls.

Kept from the old implementation because it is still useful:
- branches remain strictly sequential;
- Python/Metal caches are reclaimed before AND after every branch;
- one branch owns the model/Metal lock at a time.

Modernized for the current implementation:
- 4-bit KV-cache quantization reduces long-generation unified-memory growth;
- smaller prefill chunks reduce peak prompt-prefill memory;
- a Metal OOM retries only the failed branch with earlier KV quantization;
- iterator/response references are explicitly torn down after every streamed branch.

The generation allowance is NOT shortened. ``max_tokens`` is passed through
unchanged, so the benchmark/RSI ceiling remains 16,384 tokens. The normal path also
keeps the full context (no sliding ``max_kv_size`` window). The OOM retry keeps the
full context too; it becomes more aggressive by quantizing KV earlier, not by
throwing old reasoning tokens away.
"""
from __future__ import annotations

import contextlib
import gc
from typing import Any, Dict, List


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
    """Release transient graphs and MLX's cached Metal allocations."""
    # The older implementation collected before and after every branch. Keep that
    # hard boundary; use a full collection because RSI branches are very long-lived.
    gc.collect(2)
    try:
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass


def _memory_policy(*, aggressive: bool = False) -> Dict[str, int]:
    """Return current MLX-LM controls without imposing a sliding context window.

    Normal generation leaves the first 2048 KV tokens unquantized for throughput,
    then stores the growing cache at 4 bits. On an actual Metal OOM, one retry starts
    quantizing after 128 tokens and uses smaller prefill chunks. ``max_kv_size`` is
    intentionally absent from both policies so the model can retain the full 16K
    reasoning history.
    """
    return {
        "prefill_step_size": 128 if aggressive else 256,
        "kv_bits": 4,
        "kv_group_size": 64,
        "quantized_kv_start": 128 if aggressive else 2048,
    }


def _close_iterator(iterator: Any) -> None:
    try:
        closer = getattr(iterator, "close", None)
        if callable(closer):
            closer()
    except Exception:
        pass


def install(phase4_module, live_module, legacy_generate) -> None:
    """Wrap only branch execution; never replace RSI search/reward/training logic."""
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

        # The pre-rewrite direct generator stays the compatibility path. It is
        # invoked one branch at a time so its old branch isolation is preserved.
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
                out = None
                _clear_runtime_memory(mx)

        def streamed_one(temp: float, branch_idx: int, *, aggressive: bool) -> str:
            policy = _memory_policy(aggressive=aggressive)
            sampler = make_sampler(temp=float(temp), top_p=top_p)
            label = (
                f"live branch {branch_idx}/{total_branches} | T={float(temp):.2f} | "
                f"KV4@{policy['quantized_kv_start']} | prefill={policy['prefill_step_size']}"
            )
            live_module._write_live_header(self, formatted_prompt, branch_label=label)

            iterator = None
            response = None
            pieces: List[str] = []
            _clear_runtime_memory(mx)
            try:
                kwargs: Dict[str, Any] = {
                    "prompt": formatted_prompt,
                    # Deliberately unchanged: RSI still receives 16,384 here.
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
                # GenerationResponse/logprob state and the generator can retain MLX
                # cache refs. Tear both down before the next branch starts.
                response = None
                _close_iterator(iterator)
                iterator = None
                sampler = None
                pieces.clear()
                _clear_runtime_memory(mx)

        for branch_idx, temp in enumerate(temperatures, 1):
            if not use_stream:
                branches.append(legacy_one(float(temp)))
                continue

            try:
                branches.append(streamed_one(float(temp), branch_idx, aggressive=False))
            except TypeError:
                # Older MLX-LM without modern KV/prefill kwargs: retain the old
                # direct generator rather than changing search/sampling semantics.
                branches.append(legacy_one(float(temp)))
            except RuntimeError as exc:
                if not _is_metal_oom(exc):
                    raise
                # Recover only this branch. Completed branches and RSI item progress
                # remain valid. Reasoning/output allowance remains 16,384 tokens.
                _clear_runtime_memory(mx)
                try:
                    live_module._append_live_text(
                        "\n[RSI MEMORY RECOVERY] Metal OOM: retrying only this branch "
                        "with earlier 4-bit KV quantization; 16,384-token allowance and "
                        "full context are unchanged.\n"
                    )
                except Exception:
                    pass
                branches.append(streamed_one(float(temp), branch_idx, aggressive=True))

        return branches

    phase4_module._generate_branches_same_model = safe_generate_branches_same_model
    phase4_module._rsi_generation_memory_hardening_installed = True
