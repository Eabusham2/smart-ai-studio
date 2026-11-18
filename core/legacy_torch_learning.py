"""The original single-example, completion-only AdamW trainer used by the native bridges."""
from __future__ import annotations

import math

from core.torch_context_learning import check_cancelled, save_adamw, window_limit


def set_legacy_padding_token(tokenizer):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})


class LegacyCompletionTrainer:
    """Helpers shared by the historical GGUF and BitNet standalone trainers."""

    @staticmethod
    def _snapshot_trainable(model):
        return {name: param.detach().float().cpu().clone()
                for name, param in model.named_parameters() if param.requires_grad}

    @staticmethod
    def _drift_l2(before, model):
        total = 0.0
        touched = 0
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in before:
                continue
            delta = param.detach().float().cpu() - before[name]
            total += float((delta * delta).sum().item())
            touched += int(param.numel())
        return math.sqrt(max(total, 0.0)), touched

    @staticmethod
    def _format_training_example(tokenizer, prompt, completion):
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]
        try:
            full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prefix = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
            return str(prefix), str(full)
        except Exception:
            prefix = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            return prefix, prefix + completion + "<|im_end|>"


def controller_training_example(tokenizer, prompt, completion):
    """The direct Transformers controller historically formatted only the prefix."""
    try:
        prefix = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                               tokenize=False, add_generation_prompt=True)
    except Exception:
        prefix = None
    if not prefix:
        prefix = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    full = prefix + completion
    if "<|im_start|>" in prefix:
        full += "<|im_end|>"
    return prefix, full


class LegacyControllerTrainer(LegacyCompletionTrainer):
    _format_training_example = staticmethod(controller_training_example)


def completion_rows_fit(trainer, tokenizer, rows, limit, cancel_event):
    """The original optimizer is appropriate only if every row retains an answer in one window."""
    for row in rows:
        check_cancelled(cancel_event)
        prompt, completion = str(row.get("prompt") or "").strip(), str(row.get("completion") or "").strip()
        if not prompt or not completion:
            return False
        prefix, full = trainer._format_training_example(tokenizer, prompt, completion)
        options = dict(truncation=True, max_length=limit + 1, return_tensors="pt")
        full_ids = tokenizer(full, **options)["input_ids"][0]
        if not 2 <= len(full_ids) <= limit:
            return False
        prefix_ids = tokenizer(prefix, **options)["input_ids"][0]
        if len(prefix_ids) >= len(full_ids) or not full_ids[:len(prefix_ids)].equal(prefix_ids):
            # Separate tokenization can merge the last prompt piece with the
            # first answer piece or add EOS. Use the bounded mask in that case.
            return False
    return True


def update_legacy_lora(model, tokenizer, data, *, trainer, learning_rate, steps, max_length,
                       output_directory, before=None, cancel_event=None, fisher=None, ewc_strength=0.0):
    """Apply one optimizer step per example, with a fresh optimizer on every call."""
    import torch
    from core.torch_context_learning import backward_ewc

    rows = [item for item in data or () if str(item.get("prompt") or "").strip()
            and str(item.get("completion") or "").strip()]
    if not rows:
        raise RuntimeError("Legacy trainer received no prompt/completion pairs")
    named = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not named:
        raise RuntimeError("The loaded model exposes no trainable parameters")
    if before is None:
        before = trainer._snapshot_trainable(model)
    device = next(iter(named.values())).device
    optimizer = torch.optim.AdamW(named.values(), lr=float(learning_rate))
    options = dict(return_tensors="pt", truncation=True, max_length=max(32, int(max_length)))
    context_limit = window_limit(model, options["max_length"])
    supervised = 0
    updates = 0
    model.train()
    try:
        for _ in range(max(1, int(steps))):
            for item in rows:
                check_cancelled(cancel_event)
                prefix, full = trainer._format_training_example(
                    tokenizer, str(item["prompt"]).strip(), str(item["completion"]).strip())
                encoded = {k: v.to(device) for k, v in tokenizer(full, **options).items()}
                prefix_ids = tokenizer(prefix, **options)["input_ids"]
                labels = encoded["input_ids"].clone()
                labels[:, :min(prefix_ids.shape[1], labels.shape[1])] = -100
                # Small causal models may have fewer positions than the historical
                # tokenizer cap. Keep the completion and its preceding context.
                if labels.shape[1] > context_limit:
                    encoded = {key: value[:, -context_limit:] for key, value in encoded.items()}
                    labels = labels[:, -context_limit:]
                optimizer.zero_grad(set_to_none=True)
                output = model(**encoded, labels=labels)
                loss = output.loss
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError("Legacy completion training produced a non-finite loss")
                loss.backward()
                if fisher:
                    backward_ewc(named, before, fisher, ewc_strength)
                optimizer.step()
                supervised += int((labels[:, 1:] != -100).sum().item())
                updates += 1
        drift, touched = trainer._drift_l2(before, model)
        if not math.isfinite(drift) or drift <= 0.0 or touched <= 0:
            raise RuntimeError("Legacy trainer completed but measured zero parameter change")
        # A subsequent managed update can resume these moments. Historical
        # updates themselves always start AdamW from scratch.
        if output_directory is not None:
            save_adamw(optimizer, named, output_directory)
        return {"learning_mode": "legacy", "trainable_parameters_touched": touched,
                "optimizer_resumed": False, "supervise_prompt": False,
                "supervised_tokens": supervised, "optimizer_steps": updates}, drift
    finally:
        optimizer.zero_grad(set_to_none=True)


def update_standalone_controller(backend, data, *, learning_rate, steps):
    """Support controllers supplied directly with a model and the former adapter directory."""
    import os
    from pathlib import Path
    import tempfile
    import torch

    from core.learning_state import _process_commit_lock
    from core.training_memory import release_training_memory

    if not backend.adapter_path:
        raise RuntimeError("Controller adapter persistence path is unavailable")
    destination = Path(backend.adapter_path).absolute()
    if any(parent.parent.name == "generations" and (parent / "manifest.json").exists()
           for parent in destination.parents):
        raise RuntimeError("Bind the managed controller checkpoint before continuing learning")
    destination.parent.mkdir(parents=True, exist_ok=True)
    original = backend.model
    trainability = [(p, p.requires_grad) for p in original.parameters()]
    with _process_commit_lock(destination.parent), tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        model = backend._ensure_transformers_lora(legacy=True)
        before = LegacyControllerTrainer._snapshot_trainable(model)
        stage = Path(temporary) / "peft"
        backup = Path(temporary) / "previous"
        try:
            metrics, drift = update_legacy_lora(model, backend.tokenizer, data, trainer=LegacyControllerTrainer,
                                                learning_rate=learning_rate, steps=steps, max_length=512,
                                                before=before, output_directory=None)
            model.save_pretrained(stage, safe_serialization=True)
            if destination.exists():
                os.replace(destination, backup)
            try:
                os.replace(stage, destination)
            except BaseException:
                if backup.exists():
                    os.replace(backup, destination)
                raise
            metrics["adapter_format"] = "peft-lora"
            backend.adapters = dict(metrics)
            backend.last_learning_metrics = dict(metrics)
            return metrics, drift
        except BaseException:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in before:
                        param.copy_(before[name].to(param))
            if model is not original:
                backend.model = model.unload()
                for p, was_trainable in trainability:
                    p.requires_grad_(was_trainable)
            raise
        finally:
            model.eval()
            before.clear()
            release_training_memory(backend)
