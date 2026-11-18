"""Bounded causal training and evaluation on the selected model's PEFT tensors."""

from __future__ import annotations

import math
from pathlib import Path

from core.transformers_placement import transformers_input_device


def check_cancelled(event):
    if event is not None and event.is_set():
        raise RuntimeError("Parameter update cancelled")


def validate_candidate(backend, callback, cancel_event):
    check_cancelled(cancel_event)
    evidence = {}
    if callback is not None:
        result = callback(backend)
        if result is not True and not (isinstance(result, dict) and result.get("accepted") is True):
            raise RuntimeError("Candidate failed behavioral retention validation")
        if isinstance(result, dict):
            evidence = result
    check_cancelled(cancel_event)
    return evidence


def default_validation(backend, data, *, objective='causal_lm', reward_weighting="group_relative", entropic_kl_target=None):
    """Acceptance for standalone backend callers without a consolidator."""
    from memory.anchor_dataset import get_anchor_dataset
    from core.online_consolidator import RETENTION_TOLERANCE, _scores
    if not data:
        raise ValueError("No training examples were supplied")
    if objective == "reward_policy":
        from core.reward_adaptation import reward_advantages, validation_contrast_weights, validation_gain_tolerance
        _, weights = reward_advantages(data, reward_weighting=reward_weighting, entropic_kl_target=entropic_kl_target)
        weights = validation_contrast_weights(weights)
        probes = data
    else:
        first = data[0]
        weights = (1.0,)
        probes = [{"prompt": str(first["prompt"])[-180:], "completion": str(first["completion"])[:48]}]
    heldout = get_anchor_dataset()[2]
    guard = (heldout["prompt"], heldout["completion"])

    def score(values):
        return math.fsum(w * loss for w, loss in zip(weights, values)) / len(values)

    before, probe_sources = _scores(backend, probes)
    baseline, retained = score(before), backend.score_learning_probe(*guard)
    if not all(math.isfinite(value) for value in (*before, baseline, retained)):
        raise RuntimeError("Parameter-learning baseline is non-finite")

    def validate(candidate):
        after, candidate_sources = _scores(candidate, probes)
        guard_after = candidate.score_learning_probe(*guard)
        gains = [previous - candidate_loss for previous, candidate_loss in zip(before, after)]
        # Subtract the common shift first to avoid roundoff crediting an equal
        # gain on every sampled completion as a relative policy improvement.
        min_gain = 0.0
        if objective == "reward_policy":
            min_gain = validation_gain_tolerance(before, after)
            shift = math.fsum(gains) / len(gains)
            gains = [gain - shift for gain in gains]
        gain = score(gains)
        accepted = (candidate_sources == probe_sources and all(math.isfinite(value) for value in (*after, guard_after))
                    and gain > min_gain and guard_after <= retained + RETENTION_TOLERANCE)
        return {"accepted": accepted, "probe_nll_before": baseline, "probe_nll_after": score(after),
                **({"reward_validation_sources": probe_sources} if objective == "reward_policy" else {}),
                "score_kind": (reward_weighting + "_advantage_mean_nll_contrast"
                               if objective == "reward_policy" else "completion_nll"),
                "retention_nll_before": retained, "retention_nll_after": guard_after}
    return validate


def lora_parameters(model):
    named = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not named or any(not any(part in {"lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"}
                                for part in name.split(".")) for name in named):
        raise RuntimeError("Persistent learning requires frozen base weights and trainable LoRA matrices only")
    return named


def window_limit(model, requested):
    cfg = getattr(model, "config", None)
    language = getattr(cfg, "text_config", None) or cfg
    sizes = [int(requested)] if requested is not None else []
    for field in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        size = getattr(language, field, None)
        if isinstance(size, int) and size > 1:
            sizes.append(size)
    return min(sizes) if sizes else None


def window_loss(model, window):
    import torch

    device = transformers_input_device(model)
    ids = torch.tensor([window.input_ids], dtype=torch.long, device=device)
    labels = torch.tensor([window.labels], dtype=torch.long, device=device)
    output = model(input_ids=ids, labels=labels, use_cache=False)
    loss = output.loss
    if loss is None or loss.numel() != 1 or not torch.isfinite(loss).item():
        raise RuntimeError("Causal language model produced a non-finite training or probe loss")
    return loss


def score_torch_probe(model, tokenizer, prompt, completion, max_tokens=256):
    import torch
    from core.learning_data import iter_probe_windows

    total = count = 0
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            for window in iter_probe_windows(tokenizer, prompt, completion, max_tokens=window_limit(model, max_tokens)):
                total += window_loss(model, window).item() * window.supervised_tokens
                count += window.supervised_tokens
    finally:
        model.train(was_training)
    if count <= 0 or not math.isfinite(total):
        raise RuntimeError("No finite completion tokens available for the learning probe")
    return total / count


def _fisher(model, tokenizer, named, limit, cancel_event, anchor_texts=None):
    """Empirical diagonal Fisher from pre-update, unrelated language anchors."""
    import torch
    from core.learning_data import encode_text, iter_training_windows, windows_from_tokens
    from memory.anchor_dataset import get_anchor_dataset

    diagonal = {name: torch.zeros(p.shape, dtype=torch.float32, device="cpu") for name, p in named.items()}
    count = 0
    model.eval()
    try:
        if anchor_texts:
            anchors = (w for text in anchor_texts for w in windows_from_tokens(encode_text(tokenizer, text), max_tokens=limit))
        else:
            anchors = iter_training_windows(tokenizer, get_anchor_dataset()[:2], max_tokens=limit, supervise_prompt=False)
        for w in anchors:
            check_cancelled(cancel_event)
            grads = torch.autograd.grad(window_loss(model, w), tuple(named.values()), allow_unused=True)
            for (name, _), grad in zip(named.items(), grads):
                if grad is not None:
                    diagonal[name].add_(grad.detach().float().cpu().square())
            count += 1
    finally:
        model.train()
    if not count:
        raise RuntimeError("Fisher anchors supplied no supervised tokens")
    return {k: v.div_(count) for k, v in diagonal.items()}


def verify_saved_lora(model, directory):
    """Verify every persisted PEFT tensor against the finite, evaluated candidate."""
    import torch
    from peft import get_peft_model_state_dict
    from safetensors import safe_open
    root = Path(directory)
    if not (root / "adapter_config.json").is_file():
        raise RuntimeError("PEFT failed to save its adapter configuration")
    expected = get_peft_model_state_dict(model, save_embedding_layers=False)
    with safe_open(root / "adapter_model.safetensors", framework="pt", device="cpu") as saved:
        if not expected or set(saved.keys()) != set(expected):
            raise RuntimeError("Saved PEFT adapter tensor names differ from the candidate")
        for name, value in expected.items():
            tensor = saved.get_tensor(name)
            if (tensor.dtype != value.dtype or not torch.isfinite(tensor).all().item()
                    or not torch.equal(tensor, value.detach().cpu())):
                raise RuntimeError("Saved PEFT adapter tensor differs from the candidate: " + name)


_OPTIMIZER_FILE = "optimizer.safetensors"
_OPTIMIZER_SCHEMA = "torch-adamw-0.9-0.999-v1"


def restore_adamw(optimizer, named, directory):
    import torch
    from safetensors import safe_open
    if directory is None or not (Path(directory) / _OPTIMIZER_FILE).is_file():
        return False
    with safe_open(Path(directory) / _OPTIMIZER_FILE, framework="pt", device="cpu") as handle:
        if (handle.metadata() or {}).get("schema") != _OPTIMIZER_SCHEMA:
            raise RuntimeError("Incompatible saved AdamW state")
        keys = set(handle.keys())
        if not keys:
            raise RuntimeError("Saved AdamW state contains no moments")
        for name, p in named.items():
            slots = {key: name + "/" + key for key in ("step", "exp_avg", "exp_avg_sq")}
            if not keys.intersection(slots.values()):
                continue  # A previously inactive projection need not have optimizer state.
            record = {}
            for key, path in slots.items():
                if path not in keys:
                    raise RuntimeError("Incomplete AdamW state for " + name)
                keys.remove(path)
                tensor = handle.get_tensor(path)
                if (not torch.isfinite(tensor).all().item() or
                        (key == "step" and (tensor.shape != () or tensor.item() < 1)) or
                        (key != "step" and (tensor.shape != p.shape or tensor.dtype != p.dtype))):
                    raise RuntimeError("Invalid AdamW tensor for " + name)
                record[key] = tensor if key == "step" else tensor.to(p.device)
            optimizer.state[p] = record
        if keys:
            raise RuntimeError("Saved AdamW state contains unknown parameter names")
    return True


def save_adamw(optimizer, named, directory):
    import torch
    from safetensors.torch import save_file
    tensors = {name + "/" + key: tensor.detach().cpu().contiguous()
               for name, p in named.items() for key, tensor in optimizer.state.get(p, {}).items()}
    if not tensors:
        raise RuntimeError("Optimizer contains no learned moments")
    if any(not torch.isfinite(value).all().item() for value in tensors.values()):
        raise RuntimeError("Optimizer contains non-finite moments")
    save_file(tensors, Path(directory) / _OPTIMIZER_FILE, metadata={"schema": _OPTIMIZER_SCHEMA})


def update_lora(model, tokenizer, data, *, learning_rate, steps, max_tokens,
                fisher_matrix=None, lambda_ewc=0.0, cancel_event=None, before=None, anchor_texts=None,
                previous_directory=None, output_directory=None, objective='causal_lm',
                supervise_prompt=True, row_weights=None, context_text=None, max_context_tokens=2048,
                reward_weighting="group_relative", entropic_kl_target=None, reference_kl_coef=0.0):
    """Optimize each source window, measure actual drift, leave the base frozen.

    Caller owns the live-tensor rollback snapshot for an in-process model. The
    conversion bridges can discard their entire temporary model on rejection.
    """
    import torch
    from core.learning_data import iter_training_windows

    rate, strength = float(learning_rate), float(lambda_ewc)
    if not math.isfinite(rate) or rate <= 0 or not math.isfinite(strength) or strength < 0 or int(steps) < 1:
        raise ValueError("Invalid learning rate, step count or EWC strength")
    from core.query_adaptation import is_query_adapter, torch_query_loss
    from core.reward_adaptation import causal_row_weights, policy_advantages, reward_group

    rows = tuple(data)
    if objective not in {"causal_lm", "reward_policy", "query_only"}:
        raise ValueError("Unknown test-time training objective")
    if int(max_tokens) < 2:
        raise ValueError("Training window must admit a causal token pair")
    if objective != "causal_lm" and row_weights is not None:
        raise ValueError("Policy and query objectives supply their own weights")
    all_named = lora_parameters(model)
    named = {k: v for k, v in all_named.items() if objective != "query_only" or is_query_adapter(k)}
    if not named:
        raise ValueError("The selected model has no trainable query projection adapters")
    limit = window_limit(model, max_tokens)
    weights = causal_row_weights(rows, row_weights) if objective == "causal_lm" else None
    group = reward_group(tokenizer, rows, limit, reward_weighting=reward_weighting, entropic_kl_target=entropic_kl_target) if objective == "reward_policy" else None
    if before is None:
        before = {k: v.detach().cpu().clone() for k, v in named.items()}
    check_cancelled(cancel_event)
    fisher = {}
    if strength:
        diagonal = fisher_matrix or _fisher(model, tokenizer, named, limit, cancel_event, anchor_texts)
        if objective == 'query_only' and set(diagonal) == set(all_named):
            diagonal = {key: diagonal[key] for key in named}
        if set(diagonal) != set(named):
            raise ValueError("Fisher information must match the loaded LoRA parameters")
        for name, value in diagonal.items():
            tensor = torch.as_tensor(value)
            if tensor.shape != named[name].shape or not torch.isfinite(tensor).all() or (tensor < 0).any():
                raise ValueError("Invalid diagonal Fisher for " + name)
            if torch.any(tensor > 0).item():
                fisher[name] = tensor
        if not fisher:
            raise ValueError("Requested EWC has no positive diagonal Fisher importance")

    optimizer = torch.optim.AdamW(all_named.values(), lr=rate, weight_decay=0.0)
    resumed = restore_adamw(optimizer, all_named, previous_directory)
    windows = tokens = updates = 0
    method_metrics = {}
    inactive = [p for name, p in all_named.items() if name not in named]

    def penalize():
        for name, diagonal_tensor in fisher.items():
            p = named[name]
            delta = p.float() - before[name].to(device=p.device, dtype=torch.float32)
            importance = diagonal_tensor.to(device=p.device, dtype=torch.float32)
            ((strength / 2) * (importance * delta.square()).sum()).backward()

    def step():
        nonlocal updates
        penalize()
        norm = torch.nn.utils.clip_grad_norm_(named.values(), max_norm=1.0, error_if_nonfinite=True)
        if objective != "causal_lm" and norm.item() == 0:
            if objective == "reward_policy" and updates:
                return False  # The clipped surrogate can be flat after a successful first pass.
            raise RuntimeError("Test-time objective supplied no gradient to the selected parameters")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        updates += 1
        return True

    try:
        optimizer.zero_grad(set_to_none=True)
        for p in inactive:
            p.requires_grad_(False)
        model.train(objective == "causal_lm")
        if objective == "query_only":
            span_loss, spans, context_length = torch_query_loss(
                model, tokenizer, context_text, window_limit(model, max_context_tokens), limit, int(steps))
            for start, stop in spans:
                check_cancelled(cancel_event)
                loss = span_loss(start, stop)
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Non-finite cached query loss")
                loss.backward()
                step()
                windows += 1
                tokens += stop - start
            method_metrics.update(kv_prefills=1, cached_context_tokens=context_length,
                                  updated_projection="q_proj", query_spans=[list(span) for span in spans])
        elif group is not None:
            # A shared reference policy for all sampled actions. The first
            # gradient is REINFORCE. Repeated passes clip the trajectory ratio.
            with torch.no_grad():
                reference = [window_loss(model, w).item() * w.supervised_tokens for w in group.windows]
            advantages, diagnostics = policy_advantages(group, rows, reference, reference_kl_coef)
            method_metrics.update(diagnostics)
            for epoch in range(int(steps)):
                for w, advantage, old_nll in zip(group.windows, advantages, reference):
                    check_cancelled(cancel_event)
                    nll = window_loss(model, w) * w.supervised_tokens
                    ratio = (old_nll - nll).clamp(-20., 20.).exp()
                    surrogate = torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
                    (-surrogate / group.target_count).backward()
                if not step():
                    break
            tokens = group.target_count
            windows = len(group.windows)
            method_metrics.update(reward_min=min(group.rewards), reward_max=max(group.rewards),
                                  group_size=windows, supervise_prompt=False,
                                  sampled_token_ids=all("rollout_completion_ids" in row for row in rows))
        else:
            for epoch in range(int(steps)):
                for window in iter_training_windows(tokenizer, rows, max_tokens=limit, supervise_prompt=supervise_prompt):
                    check_cancelled(cancel_event)
                    weight = weights[window.row_index]
                    if not weight:
                        continue
                    (weight * window_loss(model, window)).backward()
                    step()
                    if epoch == 0:
                        windows += 1
                        tokens += window.supervised_tokens
            method_metrics["supervise_prompt"] = bool(supervise_prompt)
        check_cancelled(cancel_event)
        drift_sq = 0.0
        touched = 0
        for name, param in named.items():
            delta = param.detach().float().cpu() - before[name].float()
            drift_sq += delta.square().sum(dtype=torch.float64).item()
            touched += torch.count_nonzero(delta).item()
        if not math.isfinite(drift_sq) or drift_sq <= 0 or not tokens or not touched:
            raise RuntimeError("PEFT update produced non-finite or zero parameter change")
        if output_directory is not None:
            save_adamw(optimizer, all_named, output_directory)
        return {**method_metrics, "objective": objective, "trainable_parameters_touched": touched, "training_windows": windows,
                "optimizer": _OPTIMIZER_SCHEMA, "optimizer_resumed": resumed, "optimizer_steps": updates,
                "supervised_tokens": tokens, "ewc_lambda": strength,
                "ewc_used": bool(fisher), "fisher_tensors": len(fisher)}, math.sqrt(drift_sq)
    finally:
        optimizer.zero_grad(set_to_none=True)
        for p in inactive:
            p.requires_grad_(True)
        model.eval()
