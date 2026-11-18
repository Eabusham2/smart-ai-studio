"""Original per-row AdamW updates inside the bounded, validated MLX learning transaction."""
from __future__ import annotations

from core.learning_data import IGNORE_INDEX, tokenize_training_row, windows_from_tokens
from core.mlx_learning import cancel_if_requested, check_finite
from core.mlx_legacy_learning import _causal_loss, _release, inject_adapters


def _row_gradients(model, gradient, ids, labels, max_tokens, weight, cancel_event):
    """Accumulate all supervised targets at the same weights before a single optimizer step."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    count = len(ids) - 1 if labels is None else sum(label != IGNORE_INDEX for label in labels[1:])
    if count <= 0:
        raise ValueError("MLX hybrid row has no supervised next-token targets")
    summed = {}
    windows = 0
    for window in windows_from_tokens(ids, labels, max_tokens=max_tokens or len(ids)):
        cancel_if_requested(cancel_event)
        n = window.supervised_tokens
        targets = None if n == len(window.input_ids) - 1 else mx.array([window.labels[1:]])
        loss, grads = gradient(model, mx.array([window.input_ids]), targets, weight)
        flat = dict(tree_flatten(grads))
        check_finite([loss, *flat.values()], "hybrid row loss or gradients")
        windows += 1
        if n == count:
            return flat, count, windows
        for key, value in flat.items():
            part = value.astype(mx.float32) * (n / count)
            summed[key] = summed[key] + part if key in summed else part
        mx.eval(*summed.values())
    return summed, count, windows


def _loss(model, ids, labels):
    import mlx.core as mx
    import mlx.nn as nn

    if labels is None:
        return _causal_loss(model, ids)
    result = model(ids)
    logits = getattr(result, "logits", result)[:, :-1, :]
    mask = labels != IGNORE_INDEX
    losses = nn.losses.cross_entropy(logits.astype(mx.float32), mx.where(mask, labels, 0), reduction="none")
    return mx.sum(mx.where(mask, losses, 0.0)) / mx.sum(mask)


def fisher_diagonal(backend, texts, max_tokens, cancel_event=None):
    """Weight complete anchors equally, splitting only the differentiation work into windows."""
    import mlx.core as mx
    import mlx.nn as nn

    params = inject_adapters(backend)
    if not params:
        return {}
    fisher = {key: mx.zeros_like(value) for key, value in params.items()}
    model = backend.model
    gradient = nn.value_and_grad(model, lambda m, ids, labels, _: _loss(m, ids, labels))
    was_training = model.training
    anchors = 0
    try:
        model.train()
        for text in texts:
            cancel_if_requested(cancel_event)
            ids = backend.tokenizer.encode(text)
            if len(ids) < 2:
                continue
            grads, _, _ = _row_gradients(model, gradient, ids, None, max_tokens, 1., cancel_event)
            fisher = {key: mx.stop_gradient(value + mx.square(grads[key])) for key, value in fisher.items()}
            mx.eval(*fisher.values())
            anchors += 1
            del grads
            _release(2)
        if not anchors:
            return {}
        fisher = {key: mx.stop_gradient(value / anchors) for key, value in fisher.items()}
        check_finite(fisher.values(), "hybrid Fisher diagonal")
        return fisher
    finally:
        model.train(was_training)


def optimize_rows(backend, rows, optimizer, penalty, steps, max_tokens, supervise_prompt, row_weights, cancel_event):
    """Use ChatML for ordinary examples and honor an explicitly supplied mask or conversation."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    from core.mlx_learning import flat_trainables
    from core.reward_adaptation import causal_row_weights

    weights = causal_row_weights(rows, row_weights)
    model = backend.model
    model.train()
    gradient = nn.value_and_grad(model, lambda m, ids, labels, weight: weight * _loss(m, ids, labels) + penalty(m))
    trained = windows = targets = 0
    observed_gradient = False
    try:
        for _ in range(steps):
            for row, weight in zip(rows, weights):
                cancel_if_requested(cancel_event)
                if not weight:
                    continue
                if not supervise_prompt or row.get("messages") is not None:
                    ids, labels = tokenize_training_row(backend.tokenizer, row, supervise_prompt=supervise_prompt)
                else:
                    if not str(row.get("prompt") or "").strip() or not str(row.get("completion") or "").strip():
                        raise ValueError("Learning rows require nonempty prompt and completion")
                    text = (f"<|im_start|>user\n{row['prompt']}<|im_end|>\n"
                            f"<|im_start|>assistant\n{row['completion']}<|im_end|>")
                    ids, labels = backend.tokenizer.encode(text), None
                grads, n, chunks = _row_gradients(model, gradient, ids, labels, max_tokens, weight, cancel_event)
                observed_gradient |= any(bool(mx.any(g != 0).item()) for g in grads.values())
                cancel_if_requested(cancel_event)
                optimizer.update(model, tree_unflatten(list(grads.items())))
                check_finite([*flat_trainables(model).values(), *(v for _, v in tree_flatten(optimizer.state))],
                             "hybrid adapter or optimizer weights")
                trained += 1
                targets += n
                windows += chunks
                del grads
                _release(1)
        if not observed_gradient:
            raise RuntimeError("MLX candidate received no nonzero gradients from the learning rows")
        return {"objective": "causal_lm", "supervise_prompt": supervise_prompt, "optimizer_steps": trained}, windows, targets
    finally:
        _release(2)
