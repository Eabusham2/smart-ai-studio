"""Default TurboQuant KV-cache integration for Apple-Silicon MLX runtimes.

Model weights are untouched. The ordinary MLX-LM per-layer prompt cache is built first,
then compatible KV layers are converted to TurboQuant mixed K8/V3 with 128 sink tokens
kept unquantized and group size 64. Unsupported environments/cache types fail open to
MLX-LM's original prompt cache so generation remains available.
"""
from __future__ import annotations

import platform
from typing import Any, List

try:
    from mlx_lm.models.cache import make_prompt_cache as _mlx_make_prompt_cache
except Exception:
    _mlx_make_prompt_cache = None


TURBOQUANT_K_BITS = 8
TURBOQUANT_V_BITS = 3
TURBOQUANT_MIN_FP16_TOKENS = 128
TURBOQUANT_GROUP_SIZE = 64


def _apple_silicon() -> bool:
    return (
        platform.system() == "Darwin"
        and platform.machine().lower() in ("arm64", "aarch64")
    )


def make_turboquant_prompt_cache(model: Any) -> List[Any]:
    """Build the default app/eval KV cache, preferring TurboQuant on Apple Silicon."""
    if _mlx_make_prompt_cache is None:
        raise RuntimeError("MLX-LM prompt-cache support is unavailable")

    cache = _mlx_make_prompt_cache(model)
    if not _apple_silicon():
        return cache

    try:
        from turboquant_mlx.layers import convert_cache_to_turboquant

        converted = convert_cache_to_turboquant(
            cache,
            k_bits=TURBOQUANT_K_BITS,
            v_bits=TURBOQUANT_V_BITS,
            min_tokens_before_quant=TURBOQUANT_MIN_FP16_TOKENS,
            group_size=TURBOQUANT_GROUP_SIZE,
        )
        return converted if converted is not None else cache
    except Exception:
        # Compatibility fail-safe only. There is no user toggle: TurboQuant is the
        # default whenever the installed runtime and cache architecture support it.
        return cache


# Alias intentionally matches MLX-LM call sites, making the app/eval integration small.
make_prompt_cache = make_turboquant_prompt_cache
