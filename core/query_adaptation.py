"""Query-projection adaptation on causally masked spans of one frozen KV prefill."""
from __future__ import annotations

import random

from core.learning_data import encode_text

TORCH_QUERY_MODELS = {"llama", "mistral", "qwen2", "qwen3"}
MLX_QUERY_MODELS = {"llama", "mistral", "qwen2", "qwen3"}


def query_schedule(tokenizer, context_text, max_context_tokens, span_tokens, steps):
    if not isinstance(context_text, str) or not context_text.strip():
        raise ValueError("Query-only adaptation requires context_text")
    if not 3 <= max_context_tokens <= 4096 or not 1 <= steps <= 8 or not 2 <= span_tokens <= 512:
        raise ValueError("Query-only budget must allow 3–4096 context tokens, 2–512 span tokens and 1–8 steps")
    tokens = encode_text(tokenizer, context_text)
    if not 3 <= len(tokens) <= max_context_tokens:
        raise ValueError("Query-only context needs at least three tokens and must fit the declared context budget")
    width = min(span_tokens, max(1, len(tokens) // 4), len(tokens) - 1)
    # A local RNG keeps probing repeatable and leaves generation sampling untouched.
    rng = random.Random(1591)
    # A length-one span at position zero can attend to only one key and
    # therefore supplies no query gradient.
    starts = [rng.randrange(int(width == 1), len(tokens) - width) for _ in range(steps)]
    return tokens, [(s, s + width) for s in starts]


def is_query_adapter(name):
    return "q_proj" in name.split(".")


def torch_query_loss(model, tokenizer, context_text, context_limit, span_limit, steps):
    """Return a span iterator holding detached, per-layer K/V from the original model."""
    import inspect
    import torch
    from transformers import DynamicCache
    from torch.nn import functional as F
    from core.transformers_placement import transformers_input_device

    if (getattr(model.config, "model_type", None) not in TORCH_QUERY_MODELS or
            getattr(model.config, "_attn_implementation", None) not in {None, "eager", "sdpa"}):
        raise ValueError("The selected Torch architecture has no verified full-attention frozen-cache path")
    tokens, spans = query_schedule(tokenizer, context_text, context_limit, span_limit, steps)
    ids = torch.tensor([tokens], device=transformers_input_device(model), dtype=torch.long)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    signature = inspect.signature(base.forward).parameters
    prefill = model
    cache_kwargs = {"config": model.config} if "config" in inspect.signature(DynamicCache).parameters else {}
    options = {"past_key_values": DynamicCache(**cache_kwargs)}
    if "logits_to_keep" in signature:
        options["logits_to_keep"] = 1
    elif "num_logits_to_keep" in signature:
        options["num_logits_to_keep"] = 1
    else:
        # Earlier Transformers decoders can populate cache without materializing
        # full sequence × vocabulary logits. Their injected adapter layers are
        # the same modules as in the complete LM.
        prefill = base.base_model
    with torch.no_grad():
        cache = prefill(input_ids=ids, use_cache=True, **options).past_key_values
    layers = getattr(cache, "layers", None)
    if layers is not None:
        pairs = [(getattr(layer, "keys", None), getattr(layer, "values", None)) for layer in layers]
    elif hasattr(cache, "key_cache"):
        pairs = list(zip(cache.key_cache, cache.value_cache))
    else:
        raise ValueError("This Transformers version/model does not expose per-layer KV tensors")
    if len(pairs) != model.config.num_hidden_layers or any(
        k is None or v is None or k.shape[-2] != len(tokens) or v.shape[-2] != len(tokens) for k, v in pairs
    ):
        raise ValueError("Query-only adaptation requires full-length prefilled K/V in every attention layer")

    class FrozenSpanCache(DynamicCache):
        def __init__(self, start, stop):
            super().__init__()
            self.start = start
            self.stop = stop
            self.visited = set()

        def get_seq_length(self, layer_idx=0):
            return self.start

        def update(self, keys, values, layer_idx, *args, **kwargs):
            self.visited.add(layer_idx)
            key, value = pairs[layer_idx]
            return key[..., :self.stop, :].detach(), value[..., :self.stop, :].detach()

    def span_loss(start, stop):
        frozen = FrozenSpanCache(start, stop)
        positions = torch.arange(start, stop, device=ids.device)
        keys = torch.arange(stop, device=ids.device)
        allowed = keys[None, :] <= positions[:, None]
        slide = getattr(model.config, "sliding_window", None)
        if slide and (getattr(model.config, "use_sliding_window", True)):
            allowed &= keys[None, :] > positions[:, None] - int(slide)
        # A 4-D additive mask works with eager attention and SDPA across the
        # supported Transformers versions. Future cached tokens cannot leak.
        dtype = model.get_input_embeddings().weight.dtype
        mask = torch.zeros(allowed.shape, device=ids.device, dtype=dtype)
        mask.masked_fill_(~allowed, torch.finfo(dtype).min)
        out = model(input_ids=ids[:, start:stop], position_ids=positions.unsqueeze(0),
                    attention_mask=mask[None, None], past_key_values=frozen, use_cache=True)
        if len(frozen.visited) != len(pairs):
            raise RuntimeError("An attention layer bypassed the frozen KV cache")
        return F.cross_entropy(out.logits[0].float(), ids[0, start + 1:stop + 1])

    return span_loss, spans, len(tokens)


def mlx_query_loss(model, tokenizer, context_text, context_limit, span_limit, steps):
    """Use the mlx-lm cache contract for full-attention text decoders."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.cache import KVCache, make_prompt_cache

    if type(model).__module__.split(".")[-1] not in MLX_QUERY_MODELS:
        raise ValueError("This MLX architecture has no verified fixed-KV query adapter")
    model_limit = getattr(getattr(model, "args", None), "max_position_embeddings", None)
    if isinstance(model_limit, int) and model_limit > 0:
        context_limit = min(context_limit, model_limit)
    tokens, spans = query_schedule(tokenizer, context_text, context_limit, span_limit, steps)
    ids = mx.array([tokens])
    cache = make_prompt_cache(model)
    if not cache or any(type(layer) is not KVCache for layer in cache):
        raise ValueError("Query-only MLX adaptation requires standard full-attention KV cache layers")
    model(ids, cache=cache)
    pairs = []
    for layer in cache:
        if layer.offset != len(tokens) or layer.keys is None or layer.values is None:
            raise ValueError("MLX decoder did not prefill the complete context")
        pair = tuple(mx.stop_gradient(v[..., :len(tokens), :]) for v in (layer.keys, layer.values))
        mx.eval(*pair)
        pairs.append(pair)

    class FrozenSpan:
        def __init__(self, pair, start, stop):
            self.offset = start
            self.stop = stop
            self.pair = pair
            self.used = False

        def make_mask(self, size, *, window_size=None, return_array=False):
            query = mx.arange(self.offset, self.offset + size)[:, None]
            key = mx.arange(self.stop)[None, :]
            mask = key <= query
            if window_size:
                mask = mask & (key > query - window_size)
            return mask

        def update_and_fetch(self, keys, values):
            self.used = True
            return tuple(v[..., :self.stop, :] for v in self.pair)

    def span_loss(candidate, start, stop):
        frozen = [FrozenSpan(pair, start, stop) for pair in pairs]
        logits = candidate(ids[:, start:stop], cache=frozen)
        if not all(layer.used for layer in frozen):
            raise RuntimeError("An MLX attention layer bypassed its frozen KV cache")
        return nn.losses.cross_entropy(logits.astype(mx.float32), ids[:, start + 1:stop + 1], reduction="mean")

    return span_loss, spans, len(tokens)
