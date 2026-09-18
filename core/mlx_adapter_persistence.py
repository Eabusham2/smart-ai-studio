"""Persistence compatibility for Smart AI's directly-saved MLX LoRA tensors.

`train_mini_batch()` persists the active trainable parameters as one safetensors
file. `mlx_lm.load(adapter_path=...)` normally expects an MLX-LM adapter package
(directory/config), so passing our raw checkpoint file directly is not reliable.
This installer loads the base model first, reconstructs matching LoRA modules,
and applies the exact saved trainable tensors to the same model instance.
"""
from __future__ import annotations

import os


def _infer_rank(weights) -> int:
    candidates = []
    for tensor in (weights or {}).values():
        shape = tuple(int(x) for x in getattr(tensor, "shape", ()) or ())
        if len(shape) >= 2:
            for dim in shape:
                if 1 <= dim <= 256:
                    candidates.append(dim)
    return min(candidates) if candidates else 8


def install_mlx_adapter_persistence(cls) -> None:
    if getattr(cls, "_raw_adapter_persistence_installed", False):
        return

    original_load = cls.load_model

    def load_with_raw_adapter_restore(self):
        checkpoint = str(getattr(self, "adapter_path", "") or "").strip()
        raw_checkpoint = checkpoint if checkpoint and os.path.isfile(checkpoint) else ""

        if not raw_checkpoint:
            return original_load(self)

        # Do not pass a standalone safetensors file to mlx_lm.load as an adapter
        # package. Load/materialize the base model first.
        self.adapter_path = None
        try:
            ok = original_load(self)
        finally:
            self.adapter_path = checkpoint
        if not ok:
            return False

        try:
            import mlx.core as mx
            import mlx.utils

            weights = mx.load(raw_checkpoint)
            if not isinstance(weights, dict) or not weights:
                raise RuntimeError("saved MLX adapter checkpoint contains no tensors")

            rank = _infer_rank(weights)
            self.inject_lora_adapters(r=rank, scale=2.0)
            training_model = (
                self.get_training_model()
                if callable(getattr(self, "get_training_model", None))
                else self.model
            )
            training_model.update(mlx.utils.tree_unflatten(list(weights.items())))
            mx.eval(training_model.parameters())
            self.adapters = dict(weights)
            self.last_restored_adapter_path = raw_checkpoint
            self.last_restored_adapter_rank = rank
            return True
        except Exception as exc:
            self.is_mlx_available = False
            raise RuntimeError(
                f"Failed to restore learned MLX adapter checkpoint {raw_checkpoint}: {exc}"
            ) from exc

    cls.load_model = load_with_raw_adapter_restore
    cls._raw_adapter_persistence_installed = True
