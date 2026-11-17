"""Honest chat-footer and context labels from backend token counters.

Never infer token counts from words, UTF-8 bytes or text chunk sizes.
"""
from __future__ import annotations
import math


def measured_int(value):
    return type(value) is int and value >= 0


def measured_rate(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def response_footer(snapshot, *, pro=False):
    """Return concise, non-fabricated generated-token and decode statistics."""
    data = snapshot if isinstance(snapshot, dict) else {}
    complete = data.get("all_counts_available") is True
    tokens = data.get("tokens") if complete and measured_int(data.get("tokens")) else None
    tps = data.get("decode_tps") if not pro or complete else None
    token_text = f"{tokens:,}" if tokens is not None else "unavailable"
    rate_text = f"{float(tps):.1f} tok/s decode" if measured_rate(tps) else "TPS unavailable"
    suffix = " · across Pro branches" if pro else ""
    return f"⚡ Generated tokens: {token_text} · {rate_text}{suffix}"


def context_label(prompt_tokens, limit, *, completion_tokens=None):
    """Report only confirmed counts, and distinguish prompt-only occupancy."""
    if not measured_int(limit) or limit <= 0:
        return "📊 Context: limit unavailable"
    if not measured_int(prompt_tokens):
        return f"📊 Context: measuring / {limit:,}"
    if measured_int(completion_tokens):
        used = prompt_tokens + completion_tokens
        mode = "Last request"
    else:
        used = prompt_tokens
        mode = "Prompt"
    return f"📊 {mode}: {used:,} / {limit:,} tokens ({100 * used / limit:.1f}%)"


def context_limit(model_info, engine, configured_limit):
    """Use the loaded backend's effective context, not an unrelated UI guess.

    Model metadata gives a physical upper bound; runtime n_ctx or a selected
    context budget can be lower. The automatic startup window is a fallback
    only while no backend exposes its true supported budget.
    """
    info = model_info or {}
    published = info.get("max_context")
    selected = getattr(engine, "_context_budget_tokens", None)
    owner = None
    for key in ("controller_backend", "mlx_backend", "gguf_backend", "bitnet_backend"):
        value = getattr(engine, key, None)
        if value is not None and (
            getattr(value, "is_loaded", False)
            or getattr(value, "is_mlx_available", False)
            or getattr(value, "model", None) is not None
        ):
            owner = value
            break
    actual = []
    if owner is not None:
        n_ctx = getattr(owner, "n_ctx", None)
        if callable(n_ctx):
            try:
                n_ctx = n_ctx()
            except (TypeError, ValueError, RuntimeError):
                n_ctx = None
        for raw in (
            getattr(owner, "last_model_context_limit", None),
            getattr(owner, "context_budget_tokens", None),
            n_ctx,
        ):
            if measured_int(raw) and raw > 0:
                actual.append(raw)
    if measured_int(selected) and selected > 0:
        actual.append(selected)
    if measured_int(published) and published > 0:
        actual.append(published)
    if actual:
        return min(actual)
    for raw in (configured_limit, info.get("default_context")):
        if measured_int(raw) and raw > 0:
            return raw
    return None


def observed_prompt_tokens(engine):
    """Read exact native/backend prompt usage; never replace it with a guess."""
    active = getattr(engine, 'active_backend', None)
    name = {'mlx': 'mlx_backend', 'controller': 'controller_backend',
            'gguf': 'gguf_backend', 'bitnet': 'bitnet_backend'}.get(active)
    owner = getattr(engine, name, None) if name else None
    if owner is None:
        return None
    count = getattr(owner, 'last_prompt_token_count', None)
    if measured_int(count):
        return count
    for data in (getattr(owner, 'last_usage', None),
                 getattr(owner, 'last_generation', None)):
        if not isinstance(data, dict):
            continue
        usage = data.get('usage') if isinstance(data.get('usage'), dict) else data
        count = usage.get('prompt_tokens')
        if measured_int(count):
            return count
    return None
