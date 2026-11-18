"""Query-projection adaptation on causally masked spans of one frozen KV prefill."""
from __future__ import annotations

import random

from core.learning_data import encode_text

TORCH_QUERY_MODELS = {"llama", "mistral", "qwen2", "qwen3", "bitnet"}
MLX_QUERY_MODELS = {"llama", "mistral", "qwen2", "qwen3"}


def query_schedule(tokenizer, context_text, max_context_tokens, span_tokens, steps):
    if not isinstance(context_text, str) or not context_text.strip():
        raise ValueError("Query-only adaptation requires context_text")
    if (max_context_tokens is not None and (type(max_context_tokens) is not int or max_context_tokens < 3)
            or type(steps) is not int or steps < 1 or type(span_tokens) is not int or span_tokens < 2):
        raise ValueError("Query-only adaptation requires a positive step count, span of at least two and context of at least three tokens")
    tokens = encode_text(tokenizer, context_text)
    if len(tokens) < 3 or (max_context_tokens is not None and len(tokens) > max_context_tokens):
        raise ValueError("Query-only context needs at least three tokens and must fit the declared context budget")
    width = min(span_tokens, len(tokens) - 1)
    # A local RNG keeps probing repeatable and leaves generation sampling untouched.
    rng = random.Random(1591)
    starts = [rng.randrange(len(tokens) - width) for _ in range(steps)]
    return tokens, [(s, s + width) for s in starts]


def is_query_adapter(name):
    return "q_proj" in name.split(".")


def split_query_geometry(model, config):
    """A generic q_proj may combine a query and a gate. Only accept the query width."""
    heads = getattr(config, "num_attention_heads", None)
    dim = getattr(config, "head_dim", None)
    width = getattr(config, "hidden_size", None)
    if type(heads) is not int or heads < 1:
        return False
    if dim is None and type(width) is int and width % heads == 0:
        dim = width // heads
    if type(dim) is not int or dim < 1:
        return False
    projections = [module for name, module in model.named_modules() if name.split(".")[-1] == "q_proj"]
    # Check only modules targeted by the active language adapters when present.
    selected = [module for module in projections if hasattr(module, "lora_A") or hasattr(module, "lora_a")]
    for module in selected or projections:
        base = getattr(module, "base_layer", getattr(module, "linear", module))
        output_size = getattr(base, "out_features", None)
        if output_size is None:
            shape = getattr(getattr(base, "weight", None), "shape", ())
            output_size = shape[0] if len(shape) == 2 else None
        if output_size != heads * dim:
            return False
    return bool(projections)


def torch_query_config(config):
    """Advertise decoder configs whose mask can be represented by a single 4-D causal mask."""
    if getattr(config, "_attn_implementation", None) not in {None, "eager", "sdpa"}:
        return False
    if not getattr(config, "num_hidden_layers", None) or getattr(config, "is_encoder_decoder", False):
        return False
    kinds = set(getattr(config, "layer_types", ()) or ())
    return not kinds or kinds == {"full_attention"} or kinds == {"sliding_attention"}


def torch_query_loss(model, tokenizer, context_text, context_limit, span_limit, steps):
    """Return a span iterator holding detached, per-layer K/V from the original model."""
    import inspect
    import torch
    from transformers import DynamicCache
    from torch.nn import functional as F
    from core.transformers_placement import transformers_input_device

    if not torch_query_config(model.config) or not split_query_geometry(model, model.config):
        raise ValueError("The selected Torch architecture has no verified full-attention frozen-cache path")
    tokens, spans = query_schedule(tokenizer, context_text, context_limit, span_limit, steps)
    ids = torch.tensor([tokens], device=transformers_input_device(model), dtype=torch.long)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    signature = inspect.signature(base.forward).parameters
    prefill = model
    verify_replay = getattr(model.config, "model_type", None) not in TORCH_QUERY_MODELS
    cache_kwargs = {"config": model.config} if "config" in inspect.signature(DynamicCache).parameters else {}
    options = {"past_key_values": DynamicCache(**cache_kwargs)}
    if verify_replay:
        prefill = base.base_model
    elif "logits_to_keep" in signature:
        options["logits_to_keep"] = 1
    elif "num_logits_to_keep" in signature:
        options["num_logits_to_keep"] = 1
    else:
        # Earlier Transformers decoders can populate cache without materializing
        # full sequence × vocabulary logits. Their injected adapter layers are
        # the same modules as in the complete LM.
        prefill = base.base_model
    with torch.no_grad():
        output = prefill(input_ids=ids, use_cache=True, **options)
        cache = output.past_key_values
        if verify_replay:
            start = spans[0][0]
            golden = output.last_hidden_state[:, start:start+1].detach().clone()
        del output
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

    def replay(candidate, start, stop):
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
        out = candidate(input_ids=ids[:, start:stop], position_ids=positions.unsqueeze(0),
                    attention_mask=mask[None, None], past_key_values=frozen, use_cache=True)
        if len(frozen.visited) != len(pairs):
            raise RuntimeError("An attention layer bypassed the frozen KV cache")
        return out

    if verify_replay:
        with torch.no_grad():
            actual = replay(prefill, start, start+1).last_hidden_state
            if not torch.allclose(actual.float(), golden.float(), rtol=.015, atol=.015):
                raise ValueError("Frozen-cache replay disagrees with the loaded Torch decoder")

    def span_loss(start, stop):
        logits = replay(model, start, stop).logits[0].float()
        return F.cross_entropy(logits, ids[0, start + 1:stop + 1].to(logits.device))

    return span_loss, spans, len(tokens)


class FrozenMLXSpan:
    def __init__(self, pair, start, stop):
        self.offset = start
        self.stop = stop
        self.pair = pair
        self.used = False

    def make_mask(self, size, *, window_size=None, return_array=False):
        import mlx.core as mx
        query = mx.arange(self.offset, self.offset + size)[:, None]
        key = mx.arange(self.stop)[None, :]
        mask = key <= query
        if window_size:
            mask = mask & (key > query - window_size)
        return mask

    def update_and_fetch(self, keys, values):
        self.used = True
        return tuple(v[..., :self.stop, :] for v in self.pair)


def mlx_query_loss(model, tokenizer, context_text, context_limit, span_limit, steps, *, packed=False):
    """Use the mlx-lm cache contract for full-attention text decoders."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.cache import KVCache, make_prompt_cache

    model_limit = getattr(getattr(model, "args", None), "max_position_embeddings", None)
    if isinstance(model_limit, int) and model_limit > 0:
        context_limit = min(context_limit or model_limit, model_limit)
    tokens, spans = query_schedule(tokenizer, context_text, context_limit, span_limit, steps)
    ids = mx.array([tokens])
    if packed:
        from core.mlx_packed_learning import terminal_query_loss
        return terminal_query_loss(model, ids, spans), spans, len(tokens)
    if not split_query_geometry(model, getattr(model, "args", None)):
        raise ValueError("Generic MLX query adaptation requires a separate, full-width query projection")
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

    def span_loss(candidate, start, stop):
        frozen = [FrozenMLXSpan(pair, start, stop) for pair in pairs]
        result = candidate(ids[:, start:stop], cache=frozen)
        logits = getattr(result, "logits", result)
        if not all(layer.used for layer in frozen):
            raise RuntimeError("An MLX attention layer bypassed its frozen KV cache")
        return nn.losses.cross_entropy(logits.astype(mx.float32), ids[:, start + 1:stop + 1], reduction="mean")

    if type(model).__module__.split(".")[-1] not in MLX_QUERY_MODELS:
        reference = model(ids[:, :2])
        reference = getattr(reference, "logits", reference)[:, 1:2]
        expected = nn.losses.cross_entropy(reference.astype(mx.float32), ids[:, 2:3], reduction="mean")
        if not bool(mx.allclose(expected, span_loss(model, 1, 2), rtol=.001, atol=.01).item()):
            raise ValueError("Frozen-cache replay disagrees with the loaded MLX decoder")
    return span_loss, spans, len(tokens)
