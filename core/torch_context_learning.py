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


def default_validation(backend, data):
    """Acceptance for standalone backend callers without a consolidator."""
    from memory.anchor_dataset import get_anchor_dataset
    from core.online_consolidator import RETENTION_TOLERANCE
    if not data:
        raise ValueError("No training examples were supplied")
    first = data[0]
    witness = (str(first["prompt"])[-180:], str(first["completion"])[:48])
    heldout = get_anchor_dataset()[2]
    guard = (heldout["prompt"], heldout["completion"])
    baseline, retained = (backend.score_learning_probe(*pair) for pair in (witness, guard))
    if not all(math.isfinite(score) for score in (baseline, retained)):
        raise RuntimeError("Parameter-learning baseline is non-finite")

    def validate(candidate):
        after, guard_after = (candidate.score_learning_probe(*pair) for pair in (witness, guard))
        accepted = (math.isfinite(after) and math.isfinite(guard_after)
                    and after < baseline and guard_after <= retained + RETENTION_TOLERANCE)
        return {"accepted": accepted, "probe_nll_before": baseline, "probe_nll_after": after,
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
    sizes = [int(requested)]
    for field in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        size = getattr(language, field, None)
        if isinstance(size, int) and size > 1:
            sizes.append(size)
    return min(sizes)


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
    from safetensors.torch import save_file
    tensors = {name + "/" + key: tensor.detach().cpu().contiguous()
               for name, p in named.items() for key, tensor in optimizer.state.get(p, {}).items()}
    if not tensors:
        raise RuntimeError("Optimizer contains no learned moments")
    save_file(tensors, Path(directory) / _OPTIMIZER_FILE, metadata={"schema": _OPTIMIZER_SCHEMA})


def update_lora(model, tokenizer, data, *, learning_rate, steps, max_tokens,
                fisher_matrix=None, lambda_ewc=0.0, cancel_event=None, before=None, anchor_texts=None,
                previous_directory=None, output_directory=None):
    """Optimize each source window, measure actual drift, leave the base frozen.

    Caller owns the live-tensor rollback snapshot for an in-process model. The
    conversion bridges can discard their entire temporary model on rejection.
    """
    import torch
    from core.learning_data import iter_training_windows

    rate, strength = float(learning_rate), float(lambda_ewc)
    if not math.isfinite(rate) or rate <= 0 or not math.isfinite(strength) or strength < 0 or int(steps) < 1:
        raise ValueError("Invalid learning rate, step count or EWC strength")
    named = lora_parameters(model)
    limit = window_limit(model, max_tokens)
    if before is None:
        before = {k: v.detach().cpu().clone() for k, v in named.items()}
    check_cancelled(cancel_event)
    fisher = {}
    if strength:
        diagonal = fisher_matrix or _fisher(model, tokenizer, named, limit, cancel_event, anchor_texts)
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

    optimizer = torch.optim.AdamW(named.values(), lr=rate, weight_decay=0.0)
    resumed = restore_adamw(optimizer, named, previous_directory)
    windows = tokens = 0
    model.train()
    try:
        for epoch in range(int(steps)):
            for window in iter_training_windows(tokenizer, data, max_tokens=limit):
                check_cancelled(cancel_event)
                optimizer.zero_grad(set_to_none=True)
                loss = window_loss(model, window)
                for name, diagonal_tensor in fisher.items():
                    p = named[name]
                    delta = p.float() - before[name].to(device=p.device, dtype=torch.float32)
                    importance = diagonal_tensor.to(device=p.device, dtype=torch.float32)
                    loss = loss + (strength / 2) * (importance * delta.square()).sum().to(loss.device)
                if not torch.isfinite(loss).item():
                    raise RuntimeError("EWC loss is non-finite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(named.values(), max_norm=1.0, error_if_nonfinite=True)
                optimizer.step()
                if epoch == 0:
                    windows += 1
                    tokens += window.supervised_tokens
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
            save_adamw(optimizer, named, output_directory)
        return {"trainable_parameters_touched": touched, "training_windows": windows,
                "optimizer": _OPTIMIZER_SCHEMA, "optimizer_resumed": resumed,
                "supervised_tokens": tokens, "ewc_lambda": strength,
                "ewc_used": bool(fisher), "fisher_tensors": len(fisher)}, math.sqrt(drift_sq)
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.eval()
