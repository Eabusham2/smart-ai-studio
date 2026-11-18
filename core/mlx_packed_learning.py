"""Terminal-block LoRA for the Qwen3.5 decoders used by the two Bonsai MLX loaders."""
from __future__ import annotations

import math


def terminal_targets(model):
    """Only the terminal block is adapted. All recurrent and attention prefixes stay frozen."""
    layers = getattr(getattr(model, "model", None), "layers", ())
    if len(layers) < 2 or not callable(getattr(model, "lm_head", None)):
        return []
    last = layers[-1]
    if getattr(last, "is_linear", True):
        return []
    attn, mlp = getattr(last, "self_attn", None), getattr(last, "mlp", None)
    if not callable(getattr(attn, "q_proj", None)) or not callable(getattr(mlp, "down_proj", None)):
        return []
    return [(attn, "q_proj"), (mlp, "down_proj")]


def packed_lora(base, width, rank, scale, *, query_shape=None):
    """Infer the logical output width by running the original projection, preserving all packing."""
    import mlx.core as mx
    import mlx.nn as nn

    dtype = getattr(getattr(base, "scales", None), "dtype", mx.float16)
    output = base(mx.zeros((1, 1, width), dtype=dtype))
    if output.ndim != 3 or output.shape[:2] != (1, 1):
        raise ValueError("Packed projection returned an incompatible activation shape")
    out_width = output.shape[-1]
    gated = False
    if query_shape is not None:
        heads, dim = query_shape
        gated = out_width == 2 * heads * dim
        if not gated and out_width != heads * dim:
            raise ValueError("Packed query output does not match its declared head geometry")
    # Qwen3.5 interleaves a query and a sigmoid gate in each head. Restrict
    # the factors themselves to real query channels so Adam cannot move gates.
    factor_width = heads * dim if gated else out_width

    class ResidualLoRA(nn.Module):
        _smartai_packed_lora = True

        def __init__(self):
            super().__init__()
            self.linear = base
            self.scale = scale
            bound = 1 / math.sqrt(width)
            self.lora_a = mx.random.uniform(-bound, bound, (width, rank))
            self.lora_b = mx.zeros((rank, factor_width))

        def __call__(self, x):
            native = self.linear(x)
            delta = (x.astype(self.lora_a.dtype) @ self.lora_a) @ self.lora_b
            if gated:
                query = delta.reshape(*delta.shape[:-1], heads, dim)
                delta = mx.concatenate([query, mx.zeros_like(query)], axis=-1).reshape(native.shape)
            return native + (self.scale * delta).astype(native.dtype)

    return ResidualLoRA()


def terminal_query_loss(model, ids, spans):
    """Prefill the real hybrid decoder once, then replay only its final full-attention block."""
    import mlx.core as mx
    import mlx.nn as nn
    from core.query_adaptation import FrozenMLXSpan

    layers = model.model.layers
    if not terminal_targets(model):
        raise ValueError("Packed query replay requires a terminal full-attention block")
    cache = model.make_cache()
    count = ids.shape[-1]
    positions = mx.broadcast_to(mx.arange(count)[None, None, :], (3, 1, count))
    # Both publisher VLM loaders expose Qwen3.5's native hidden-sink API.
    # Last-block inputs include every recurrent state transition up to that point.
    out = model(ids, cache=cache, position_ids=positions, capture_layer_ids=[len(layers)-2, len(layers)-1])
    states = getattr(out, "hidden_states", None)
    if not isinstance(states, (list, tuple)) or len(states) != 2:
        raise ValueError("This packed decoder does not expose terminal block activations")
    prefix = mx.stop_gradient(states[0])
    k, v = getattr(cache[-1], "keys", None), getattr(cache[-1], "values", None)
    if (getattr(cache[-1], "offset", 0) != count or k is None or v is None
            or k.shape[-2] < count or v.shape[-2] < count or prefix.shape[:2] != ids.shape):
        raise ValueError("Terminal attention did not retain its full prefilled KV")
    pair = tuple(mx.stop_gradient(t[..., :count, :]) for t in (k, v))
    start, stop = spans[0]
    check_stop = min(stop, start + 2)
    reference = mx.stop_gradient(states[1][:, start:check_stop])
    mx.eval(prefix, reference, *pair)
    del out, cache, states

    def replay(candidate, start, stop):
        frozen = FrozenMLXSpan(pair, start, stop)
        result = candidate.model.layers[-1](prefix[:, start:stop], mask=frozen.make_mask(stop-start),
                                            cache=frozen, position_ids=positions[..., start:stop])
        if not frozen.used:
            raise RuntimeError("Terminal attention bypassed its frozen KV")
        return result

    trial = replay(model, start, check_stop)
    # BF16/FP16 prefill and replay can dispatch different native attention kernels.
    if not bool(mx.allclose(trial.astype(mx.float32), reference.astype(mx.float32),
                            rtol=.015, atol=.015).item()):
        raise ValueError("Frozen terminal-block replay disagrees with the selected packed decoder")

    def loss(candidate, start, stop):
        logits = candidate.lm_head(candidate.model.norm(replay(candidate, start, stop)))
        return nn.losses.cross_entropy(logits.astype(mx.float32), ids[:, start+1:stop+1], reduction="mean")

    return loss
