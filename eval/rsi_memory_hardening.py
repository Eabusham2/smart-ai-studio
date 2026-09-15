"""Memory-safe RSI/Phase-4 branch streaming without reducing reasoning length.

The benchmark keeps its normal 16,384 generated-token ceiling. This layer only
bounds/quantizes the KV cache and restores the older sequential per-branch cleanup
behavior so four-branch RSI rounds cannot accumulate Metal memory across branches.
"""
from __future__ import annotations

import gc
from typing import Any, Dict, List

from eval import live_generation_stream as live_stream


# This is a KV-cache bound, NOT a generation bound. RSI still receives the caller's
# unchanged max_tokens value (normally the benchmark ceiling of 16,384).
RSI_MAX_KV_TOKENS = 16_384
RSI_PREFILL_STEP_SIZE = 512
RSI_KV_BITS = 4
RSI_KV_GROUP_SIZE = 64


def _clear_branch_memory(mx) -> None:
    """Release Python references and cached Metal allocations between branches."""
    gc.collect()
    try:
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass
    gc.collect()


def install(phase4_module) -> None:
    if getattr(phase4_module, "_rsi_memory_hardening_installed", False):
        return

    original = phase4_module._generate_branches_same_model

    def memory_safe_generate_branches_same_model(
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
            return original(self, formatted_prompt, temperatures, max_tokens, top_p)

        mlx_lm = phase4_module.mlx_lm
        mx = phase4_module.mx
        metal_lock = phase4_module.METAL_STREAM_LOCK
        stream_generate = getattr(mlx_lm, "stream_generate", None)
        if not callable(stream_generate):
            return original(self, formatted_prompt, temperatures, max_tokens, top_p)

        try:
            from mlx_lm.sample_utils import make_sampler
        except Exception:
            make_sampler = None
        if make_sampler is None:
            return original(self, formatted_prompt, temperatures, max_tokens, top_p)

        branches: List[str] = []
        total_branches = len(temperatures)

        # Preserve the old implementation's sequential one-branch-at-a-time
        # behavior. Never keep a live KV generator from one branch while starting
        # the next branch.
        for branch_idx, temp in enumerate(temperatures, 1):
            _clear_branch_memory(mx)
            iterator = None
            response = None
            sampler = None
            pieces: List[str] = []

            try:
                sampler = make_sampler(temp=float(temp), top_p=top_p)
                live_stream._write_live_header(
                    self,
                    formatted_prompt,
                    branch_label=(
                        f"live branch {branch_idx}/{total_branches} | "
                        f"T={float(temp):.2f} | KV=4bit/{RSI_MAX_KV_TOKENS}"
                    ),
                )

                # Keep the caller's full generation allowance unchanged. The
                # memory controls below affect KV storage/prefill only.
                with metal_lock:
                    iterator = stream_generate(
                        self.engine.model,
                        self.engine.tokenizer,
                        prompt=formatted_prompt,
                        max_tokens=max(1, int(max_tokens)),
                        sampler=sampler,
                        max_kv_size=RSI_MAX_KV_TOKENS,
                        prefill_step_size=RSI_PREFILL_STEP_SIZE,
                        kv_bits=RSI_KV_BITS,
                        kv_group_size=RSI_KV_GROUP_SIZE,
                        quantized_kv_start=0,
                    )
                    for response in iterator:
                        chunk = getattr(response, "text", None)
                        if chunk is None:
                            chunk = str(response)
                        chunk = str(chunk)
                        pieces.append(chunk)
                        live_stream._append_live_text(chunk)

                branch_text = "".join(pieces)
                branches.append(branch_text)

            except TypeError:
                # Compatibility with older mlx-lm builds: retain the established
                # one-temperature generation path, still with hard cleanup before
                # and after this branch. Current supported mlx-lm accepts the KV
                # arguments above.
                if iterator is not None and hasattr(iterator, "close"):
                    try:
                        iterator.close()
                    except Exception:
                        pass
                iterator = None
                pieces.clear()
                _clear_branch_memory(mx)
                fallback = original(self, formatted_prompt, [temp], max_tokens, top_p)
                branches.extend(fallback)

            finally:
                if iterator is not None and hasattr(iterator, "close"):
                    try:
                        iterator.close()
                    except Exception:
                        pass
                iterator = None
                response = None
                sampler = None
                pieces.clear()
                _clear_branch_memory(mx)

        return branches

    phase4_module._generate_branches_same_model = memory_safe_generate_branches_same_model
    phase4_module._rsi_memory_hardening_installed = True
