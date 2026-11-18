"""Native MLX adapter selection, completion scoring and bounded minibatch primitives.

All functions touching MLX arrays run under the backend's model-owner thread.
The optional Apple dependencies are imported only when a real model is present.
"""
from __future__ import annotations

import math


def cancel_if_requested(event):
    if event is not None and (event.is_set() if hasattr(event, "is_set") else event()):
        raise InterruptedError("MLX learning cancelled")


def is_adapter_key(key):
    return key.rsplit(".", 1)[-1] in {"lora_a", "lora_b"}


def flat_trainables(model):
    from mlx.utils import tree_flatten

    weights = dict(tree_flatten(model.trainable_parameters()))
    if not weights or not all(is_adapter_key(key) for key in weights):
        raise RuntimeError("MLX learning requires exclusively LoRA adapter parameters")
    return weights


def check_finite(values, label):
    import mlx.core as mx

    checks = [mx.all(mx.isfinite(value)) for value in values]
    if not checks or not bool(mx.all(mx.stack(checks)).item()):
        raise FloatingPointError(f"MLX learning found non-finite or missing {label}")


def _config_dimension(model, name):
    for owner in (model, getattr(model, "model", None)):
        for cfg in (getattr(owner, "args", None), getattr(owner, "config", None)):
            if cfg is None:
                continue
            nested = cfg.get("text_config") if isinstance(cfg, dict) else getattr(cfg, "text_config", None)
            for config in (nested, cfg):
                value = config.get(name) if isinstance(config, dict) else getattr(config, name, None)
                if value is not None and int(value) > 0:
                    return int(value)
    raise RuntimeError(f"Packed output-head dimensions unavailable: missing {name}")


def _packed_head(model, rank, scale):
    import mlx.core as mx
    import mlx.nn as nn

    # Packed uint32 shapes encode storage, not logical feature counts. Both
    # Bonsai loaders expose these dimensions on their language configuration.
    width = _config_dimension(model, "hidden_size")
    vocab = _config_dimension(model, "vocab_size")
    base = getattr(model, "lm_head", None)
    if not callable(base):
        raise RuntimeError("Packed language model exposes no callable output projection")

    class OutputLoRA(nn.Module):
        _smartai_output_lora = True

        def __init__(self):
            super().__init__()
            self.linear = base
            limit = 1 / math.sqrt(width)
            self.lora_a = mx.random.uniform(-limit, limit, (width, rank))
            self.lora_b = mx.zeros((rank, vocab))
            self.scale = scale

        def __call__(self, inputs):
            # The base includes the publisher's Hadamard rotation, scales and
            # packing. Neither the decoder nor this opaque kernel needs a VJP.
            hidden = mx.stop_gradient(inputs)
            native = mx.stop_gradient(self.linear(hidden))
            residual = (hidden.astype(self.lora_a.dtype) @ self.lora_a) @ self.lora_b
            return native + (self.scale * residual).astype(native.dtype)

    return OutputLoRA()


def _projection_targets(model, requested):
    """Resolve checkpoint paths on the actual decoder; fresh adapters use the historical projection set."""
    if requested is not None:
        modules = dict(model.named_modules())
        for path in sorted(requested):
            parent_path, _, name = path.rpartition(".")
            parent = modules.get(parent_path)
            if parent is None or not name or getattr(parent, name, None) is None:
                raise ValueError(f"Saved LoRA target is missing from the loaded model: {path}")
            yield parent, name
        return

    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        layers = getattr(model, "layers", ())
    for layer in layers:
        for scope, name in (("mlp", "down_proj"), ("self_attn", "q_proj"),
                            ("self_attn", "v_proj"), ("linear_attn", "out_proj")):
            parent = getattr(layer, scope, None)
            if getattr(parent, name, None) is not None:
                yield parent, name


def ensure_adapters(backend, rank=8, scale=2.0, layout=None, *, target_paths=None, installed=None):
    """Reuse existing adapters and freeze every other parameter, even on fresh MLX models."""
    import mlx.nn as nn

    if rank <= 0 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("MLX LoRA rank and scale must be positive and finite")
    model = backend.model
    existing = [module for _, module in model.named_modules()
                if hasattr(module, "lora_a") and hasattr(module, "lora_b")]
    external = [m for m in existing if not getattr(m, "_smartai_output_lora", False)]
    if external:
        from mlx_lm.tuner.lora import LoRALinear
        if any(type(module) is not LoRALinear for module in external):
            raise ValueError("This external adapter type cannot be reconstructed from two LoRA factors")
    model.freeze()
    if not existing:
        packed = getattr(backend, "_vlm_runtime", None) in {"mlx_repo_vlm", "jang_vlm"}
        if layout == "head" or (packed and layout is None):
            replacement = _packed_head(model, rank, scale)
            if installed is not None:
                installed.append((model, "lm_head", model.lm_head))
            model.lm_head = replacement
            existing = [replacement]
        else:
            from mlx_lm.tuner.lora import LoRALinear

            replacements = []
            for parent, name in _projection_targets(model, target_paths):
                base = getattr(parent, name)
                if not isinstance(base, (nn.Linear, nn.QuantizedLinear)):
                    raise RuntimeError(f"Unsupported loaded projection for native LoRA: {name} ({type(base).__name__})")
                wrapped = LoRALinear.from_base(base, r=rank, scale=scale)
                replacements.append((parent, name, wrapped))
            for parent, name, wrapped in replacements:
                if installed is not None:
                    installed.append((parent, name, getattr(parent, name)))
                setattr(parent, name, wrapped)
                existing.append(wrapped)
    for module in existing:
        module.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
    backend.adapters = flat_trainables(model)
    return backend.adapters


def loss_for_tokens(model, inputs, targets, target_mask):
    import mlx.core as mx
    import mlx.nn as nn

    result = model(inputs)
    logits = getattr(result, "logits", result)
    losses = nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="none")
    return mx.sum(mx.where(target_mask, losses, 0.0)) / mx.sum(target_mask)


def window_arrays(ids, labels):
    import mlx.core as mx

    targets = mx.array([labels[1:]])
    valid = targets != -100
    return mx.array([ids[:-1]]), mx.where(valid, targets, 0), valid


def score_rows(model, tokenizer, rows, max_tokens, cancel_event=None):
    from core.learning_data import iter_probe_windows

    total = 0.0
    count = 0
    for row in rows:
        for window in iter_probe_windows(tokenizer, row["prompt"], row["completion"], max_tokens=max_tokens):
            cancel_if_requested(cancel_event)
            n = window.supervised_tokens
            if n:
                loss = float(loss_for_tokens(model, *window_arrays(window.input_ids, window.labels)).item())
                if not math.isfinite(loss):
                    raise FloatingPointError("MLX probe produced non-finite loss")
                total += n * loss
                count += n
    if not count:
        raise ValueError("MLX probe contains no supervised completion tokens")
    return total / count


def bounded_tokens(backend, requested):
    limit = int(requested)
    if not 2 <= limit <= 512:
        raise ValueError("MLX learning window must be between 2 and 512 tokens")
    context = int(getattr(backend, "context_budget_tokens", 0) or limit)
    if context < 2:
        raise ValueError("MLX context must admit a causal token pair")
    return min(limit, context)


def fisher_diagonal(backend, texts, max_tokens, cancel_event=None):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from core.learning_data import encode_text, windows_from_tokens

    trainable = ensure_adapters(backend)
    fisher = {k: mx.zeros(v.shape, dtype=mx.float32) for k, v in trainable.items()}
    get_grad = nn.value_and_grad(backend.model, loss_for_tokens)
    total = 0
    for text in texts:
        cancel_if_requested(cancel_event)
        ids = encode_text(backend.tokenizer, text)
        for window in windows_from_tokens(ids, max_tokens=max_tokens):
            cancel_if_requested(cancel_event)
            loss, grads = get_grad(backend.model, *window_arrays(window.input_ids, window.labels))
            flat = dict(tree_flatten(grads))
            if flat.keys() != trainable.keys():
                raise RuntimeError("MLX Fisher gradients mismatch active adapters")
            check_finite([loss, *flat.values()], "Fisher loss or gradients")
            n = window.supervised_tokens
            fisher = {k: mx.stop_gradient(fisher[k] + n * mx.square(g.astype(mx.float32)))
                      for k, g in flat.items()}
            mx.eval(*fisher.values())
            total += n
    if not total:
        return {}
    fisher = {k: mx.stop_gradient(v / total) for k, v in fisher.items()}
    check_finite(fisher.values(), "Fisher diagonal")
    return fisher
