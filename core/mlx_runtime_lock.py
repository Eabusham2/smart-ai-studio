"""Serialize live MLX inference and online parameter updates.

AwakeOnlineConsolidator runs training on a background thread. The underlying MLX
train_mini_batch() updates the active model in-place, so it must never overlap a
forward/generation pass. This installer adds one re-entrant runtime lock shared by
entropy probes, branch generation, streaming generation, and mini-batch training.
The learning thread remains asynchronous, but waits for an inference-safe boundary
before changing weights. Unloaded training fails explicitly instead of returning a
synthetic nonzero parameter drift.
"""
from __future__ import annotations

import threading


def install_mlx_runtime_lock(cls) -> None:
    if getattr(cls, "_runtime_lock_installed", False):
        return

    original_init = cls.__init__
    original_entropy = cls.calculate_token_entropy
    original_branches = cls.generate_branches
    original_stream = cls.stream_generate_tokens
    original_train = cls.train_mini_batch

    def init_with_lock(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._runtime_model_lock = threading.RLock()

    def _lock(self):
        lock = getattr(self, "_runtime_model_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._runtime_model_lock = lock
        return lock

    def entropy_locked(self, *args, **kwargs):
        with _lock(self):
            return original_entropy(self, *args, **kwargs)

    def branches_locked(self, *args, **kwargs):
        with _lock(self):
            return original_branches(self, *args, **kwargs)

    def stream_locked(self, *args, **kwargs):
        with _lock(self):
            yield from original_stream(self, *args, **kwargs)

    def train_locked(self, *args, **kwargs):
        if (
            getattr(self, "model", None) is None
            or getattr(self, "tokenizer", None) is None
            or not getattr(self, "is_mlx_available", False)
        ):
            raise RuntimeError(
                "Cannot report an MLX parameter update without a real loaded model and tokenizer."
            )
        with _lock(self):
            return original_train(self, *args, **kwargs)

    cls.__init__ = init_with_lock
    cls.calculate_token_entropy = entropy_locked
    cls.generate_branches = branches_locked
    cls.stream_generate_tokens = stream_locked
    cls.train_mini_batch = train_locked
    cls._runtime_lock_installed = True
