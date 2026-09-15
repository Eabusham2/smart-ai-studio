"""Memory-safe RSI/Phase-4 branch streaming without reducing reasoning length.

The benchmark keeps its normal 16,384 generated-token ceiling. This layer only
bounds/quantizes KV storage, restores the older sequential per-branch cleanup,
and retries a branch with smaller KV windows if Metal reports OOM. The generated
answer may still use the full 16,384-token allowance on every retry.
"""
from __future__ import annotations

import gc
from typing import List, Tuple

from eval import live_generation_stream as live_stream


RSI_GENERATION_TOKEN_CEILING = 16_384
RSI_KV_BITS = 4
RSI_KV_GROUP_SIZE = 64
RSI_CACHE_LIMIT_BYTES = 512 * 1024 * 1024


def _clear_branch_memory(mx) -> None:
    """Finish pending Metal work, then release cached allocations between branches."""
    gc.collect()
    try:
        if hasattr(mx, "synchronize"):
            mx.synchronize()
    except Exception:
        pass
    try:
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass
    gc.collect()


def _is_metal_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower().replace("_", "")
    return any(
        marker in text
        for marker in (
            "outofmemory",
            "insufficient memory",
            "kiogpucommandbuffercallbackerroroutofmemory",
            "metal command buffer execution failed",
        )
    )


def _kv_retry_ladder(self) -> List[Tuple[int, int]]:
    """KV windows/prefill steps only; never a generated-token limit."""
    hardware_floor = int(getattr(getattr(self.engine, "settings", None), "max_kv_tokens", 2048) or 2048)
    hardware_floor = max(1024, min(4096, hardware_floor))
    candidates = [
        (16_384, 512),
        (8_192, 512),
        (4_096, 256),
        (hardware_floor, 256),
        (1_024, 128),
        (512, 128),
    ]
    out: List[Tuple[int, int]] = []
    seen = set()
    for kv_tokens, prefill in candidates:
        key = (int(kv_tokens), int(prefill))
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


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

        # Protect active model/KV memory from being crowded out by stale allocator
        # cache. Restore the caller's previous cache limit before returning.
        old_cache_limit = None
        try:
            if hasattr(mx, "set_cache_limit"):
                old_cache_limit = mx.set_cache_limit(RSI_CACHE_LIMIT_BYTES)
        except Exception:
            old_cache_limit = None

        branches: List[str] = []
        total_branches = len(temperatures)

        try:
            # Preserve the old implementation's one-branch-at-a-time lifecycle.
            # No branch generator or prompt cache survives into the next branch.
            for branch_idx, temp in enumerate(temperatures, 1):
                branch_done = False
                last_oom: BaseException | None = None

                for kv_tokens, prefill_step in _kv_retry_ladder(self):
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
                                f"T={float(temp):.2f} | KV=4bit/{kv_tokens}"
                            ),
                        )

                        # IMPORTANT: max_tokens is passed through unchanged. KV
                        # fallback never lowers the 16,384-token reasoning ceiling.
                        with metal_lock:
                            iterator = stream_generate(
                                self.engine.model,
                                self.engine.tokenizer,
                                prompt=formatted_prompt,
                                max_tokens=max(1, int(max_tokens)),
                                sampler=sampler,
                                max_kv_size=kv_tokens,
                                prefill_step_size=prefill_step,
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

                        branches.append("".join(pieces))
                        branch_done = True
                        break

                    except TypeError as exc:
                        # Compatibility fallback for an older mlx-lm that does not
                        # accept the current KV arguments. Preserve sequential
                        # cleanup and the same max_tokens/temperature policy.
                        if "keyword" not in str(exc).lower() and "argument" not in str(exc).lower():
                            raise
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
                        branch_done = True
                        break

                    except RuntimeError as exc:
                        if not _is_metal_oom(exc):
                            raise
                        last_oom = exc
                        live_stream._append_live_text(
                            f"\n[RSI MEMORY] Metal OOM at KV={kv_tokens}; "
                            "clearing branch state and retrying smaller KV with the same generation limit.\n"
                        )

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

                if not branch_done:
                    raise RuntimeError(
                        "RSI branch exhausted KV-memory retry ladder while preserving "
                        f"the {int(max_tokens)}-token generation allowance"
                    ) from last_oom

        finally:
            _clear_branch_memory(mx)
            if old_cache_limit is not None and hasattr(mx, "set_cache_limit"):
                try:
                    mx.set_cache_limit(int(old_cache_limit))
                except Exception:
                    pass
            _clear_branch_memory(mx)

        return branches

    phase4_module._generate_branches_same_model = memory_safe_generate_branches_same_model
    phase4_module._rsi_memory_hardening_installed = True
