"""Original full-sequence MLX LoRA training and per-anchor Fisher estimation."""
from __future__ import annotations

import gc
import math


def _targets(model):
    layers = getattr(model, "layers", None)
    if layers is None:
        layers = getattr(getattr(model, "model", None), "layers", ())
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if hasattr(mlp, "down_proj"):
            yield mlp, "down_proj"
        if hasattr(layer, "self_attn"):
            for name in ("q_proj", "v_proj"):
                if hasattr(layer.self_attn, name):
                    yield layer.self_attn, name
        elif hasattr(getattr(layer, "linear_attn", None), "out_proj"):
            yield layer.linear_attn, "out_proj"


def inject_adapters(backend, rank=8, scale=2.0, *, installed=None):
    """Use pre-PR17 projection selection. Reuse factors restored from disk."""
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.lora import LoRALinear

    if rank <= 0 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("MLX LoRA rank and scale must be positive and finite")
    model = backend.model
    if any(hasattr(module, "lora_a") and hasattr(module, "lora_b") for _, module in model.named_modules()):
        # Restored head-only and partial projection checkpoints define their own targets.
        from core.mlx_learning import ensure_adapters
        return ensure_adapters(backend, rank, scale)
    model.freeze()
    replacements = []
    for parent, name in _targets(model):
        base = getattr(parent, name)
        wrapped = LoRALinear.from_base(base, r=rank, scale=scale)
        replacements.append((parent, name, base, wrapped))
    for parent, name, base, wrapped in replacements:
        if installed is not None:
            installed.append((parent, name, base))
        setattr(parent, name, wrapped)
        wrapped.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
    backend.adapters = dict(tree_flatten(model.trainable_parameters()))
    return backend.adapters


def _causal_loss(model, ids):
    import mlx.core as mx
    import mlx.nn as nn

    result = model(ids)
    logits = getattr(result, "logits", result)
    return mx.mean(nn.losses.cross_entropy(logits[:, :-1, :], ids[:, 1:]))


def _release(generation):
    import mlx.core as mx

    gc.collect(generation)
    try:
        mx.clear_cache()
    except (AttributeError, RuntimeError):
        pass  # Cache release is best-effort across MLX devices and versions.


def fisher_diagonal(backend, texts):
    """Average the squared gradient of each whole anchor, with equal anchor weight."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from core.mlx_learning import check_finite

    params = inject_adapters(backend)
    if not params:
        return {}
    fisher = {key: mx.zeros_like(value) for key, value in params.items()}
    mx.eval(*fisher.values())
    grad = nn.value_and_grad(backend.model, _causal_loss)
    count = 0
    was_training = backend.model.training
    try:
        backend.model.train()
        for text in texts:
            tokens = backend.tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            loss, gradients = grad(backend.model, mx.array([tokens]))
            for key, value in tree_flatten(gradients):
                fisher[key] = mx.stop_gradient(fisher[key] + mx.square(value))
            mx.eval(*fisher.values())
            del loss, gradients
            count += 1
            _release(2)
        if count:
            fisher = {key: mx.stop_gradient(value / count) for key, value in fisher.items()}
        check_finite(fisher.values(), "Fisher diagonal")
        return fisher
    finally:
        backend.model.train(was_training)
        _release(2)


def optimize_rows(backend, rows, optimizer, penalty, steps):
    """One AdamW step per full ChatML row in model training mode."""
    import mlx.core as mx
    import mlx.nn as nn

    model = backend.model
    model.train()
    grad = nn.value_and_grad(model, lambda m, ids: _causal_loss(m, ids) + penalty(m))
    trained = 0
    supervised = 0
    try:
        for _ in range(max(1, int(steps))):
            for row in rows:
                text = (f"<|im_start|>user\n{row.get('prompt', '')}<|im_end|>\n"
                        f"<|im_start|>assistant\n{row.get('completion', '')}<|im_end|>")
                tokens = backend.tokenizer.encode(text)
                if len(tokens) < 2:
                    continue
                loss, gradients = grad(model, mx.array([tokens]))
                optimizer.update(model, gradients)
                mx.eval(model.parameters(), optimizer.state)
                del loss, gradients
                trained += 1
                supervised += len(tokens) - 1
                _release(1)
        if not trained:
            raise RuntimeError("MLX training found no valid prompt/completion token rows")
        return {"objective": "causal_lm", "supervise_prompt": True, "optimizer_steps": trained}, trained, supervised
    finally:
        _release(2)
