"""Restore the useful pre-rewrite training semantics around current Phase 3.

This does not replace the current optimizer/OGP/fail-closed pipeline. It wraps the
existing Phase-3 implementation so two older behaviors that were genuinely safer
are retained:

1. Train on the assistant completion, not on reproducing the user prompt. The old
   sleep trainer masked prompt tokens; the current benchmark trainer had regressed
   to CE over the entire prompt+completion sequence.
2. Treat the Phase-3 weight update transactionally. The old double-buffer design
   intended updates to become visible atomically. Here we snapshot the real current
   trainable MLX weights and persisted adapter; any exception/interrupt/fail-closed
   integrity error restores both weights and DB consolidation flags.

Current behavior stays authoritative for everything else: same loaded model, same
AdamW/OGP/raw-gradient fallback, same buffer refresh/swap, same real-delta proof,
same adapter persistence, and same Learn/RSI queue.
"""
from __future__ import annotations

import gc
import os
import shutil
import sqlite3
from typing import Any, Dict, List, Tuple


ASSISTANT_MARKER = "<|im_start|>assistant\n"
TRAIN_WINDOW_TOKENS = 256


class _CompletionWindowTokenizer:
    """Proxy tokenizer that keeps the completion in the current 256-token train window."""

    def __init__(self, base: Any, state: Dict[str, Any], window: int = TRAIN_WINDOW_TOKENS):
        self._base = base
        self._state = state
        self._window = max(8, int(window))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def encode(self, text: str, *args: Any, **kwargs: Any):
        raw = self._base.encode(text, *args, **kwargs)
        ids = list(raw)
        self._state["loss_start"] = 0
        self._state["completion_mask_active"] = False

        marker_pos = str(text).find(ASSISTANT_MARKER)
        if marker_pos < 0 or len(ids) < 2:
            return raw

        prefix = str(text)[: marker_pos + len(ASSISTANT_MARKER)]
        prefix_ids = list(self._base.encode(prefix, *args, **kwargs))

        # The current core slices ids[:256]. Returning the final window here keeps
        # the assistant completion present even when a long prompt would otherwise
        # consume the entire training window before the answer begins.
        start = max(0, len(ids) - self._window)
        selected = ids[start:]
        local_assistant_start = max(0, len(prefix_ids) - start)

        # CE position j predicts target token j+1. To begin training on the first
        # assistant token at index local_assistant_start, keep losses from j=start-1.
        loss_start = max(0, local_assistant_start - 1)
        if len(selected) > 1:
            loss_start = min(loss_start, len(selected) - 2)
        else:
            loss_start = 0

        self._state["loss_start"] = int(loss_start)
        self._state["completion_mask_active"] = True
        self._state["selected_tokens"] = len(selected)
        self._state["assistant_start"] = int(local_assistant_start)
        return selected


def _clear_mlx(p4: Any) -> None:
    gc.collect(2)
    if not getattr(p4, "MLX_AVAILABLE", False):
        return
    mx = getattr(p4, "mx", None)
    try:
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass


def _snapshot_trainables(p4: Any, model: Any) -> Dict[str, Any]:
    if not getattr(p4, "MLX_AVAILABLE", False) or model is None:
        return {}
    flat = dict(p4.mlx.utils.tree_flatten(model.trainable_parameters()))
    snapshot: Dict[str, Any] = {}
    for key, value in flat.items():
        try:
            clone = p4.mx.copy(value)
        except Exception:
            try:
                clone = value + p4.mx.zeros_like(value)
            except Exception:
                clone = p4.mx.array(value)
        snapshot[key] = clone
    if snapshot:
        p4.mx.eval(*snapshot.values())
    return snapshot


def _restore_trainables(p4: Any, model: Any, snapshot: Dict[str, Any]) -> None:
    if not snapshot or model is None:
        return
    model.update(p4.mlx.utils.tree_unflatten(list(snapshot.items())))
    p4.mx.eval(model.parameters())


def _queued_ids(p4: Any, self: Any) -> List[int]:
    try:
        rows = list(p4._fetch_benchmark_training_memories(self))
    except Exception:
        return []
    out: List[int] = []
    for row in rows:
        try:
            if row.get("id") is not None:
                out.append(int(row["id"]))
        except Exception:
            pass
    return out


def _reset_consolidated_flags(self: Any, ids: List[int]) -> None:
    if not ids:
        return
    db_path = str(getattr(getattr(self.engine, "kg", None), "db_path", "") or "")
    if not db_path:
        return
    try:
        placeholders = ",".join("?" for _ in ids)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                f"UPDATE episodic_interactions SET consolidated=0 WHERE id IN ({placeholders})",
                ids,
            )
    except Exception:
        pass


def install(p4: Any) -> None:
    """Wrap the final Phase-3 stack after integrity telemetry is installed."""
    if getattr(p4, "_rsi_legacy_training_hardening_installed", False):
        return

    base_phase3 = p4._run_phase3_consolidation

    def transactional_completion_only_phase3(self):
        if not getattr(p4, "MLX_AVAILABLE", False) or self.engine.model is None:
            return base_phase3(self)

        model_identity = id(self.engine.model)
        snapshot = _snapshot_trainables(p4, self.engine.model)
        queued_ids = _queued_ids(p4, self)

        adapter_path = str(getattr(p4, "RSI_ADAPTER_PATH", "") or "")
        backup_path = adapter_path + ".pre_phase3.bak" if adapter_path else ""
        had_adapter = bool(adapter_path and os.path.exists(adapter_path))
        if backup_path:
            try:
                if had_adapter:
                    shutil.copy2(adapter_path, backup_path)
                elif os.path.exists(backup_path):
                    os.remove(backup_path)
            except Exception:
                backup_path = ""

        original_tokenizer = self.engine.tokenizer
        mask_state: Dict[str, Any] = {"loss_start": 0, "completion_mask_active": False}
        proxy = _CompletionWindowTokenizer(original_tokenizer, mask_state)
        original_ce = p4.nn.losses.cross_entropy

        def completion_only_cross_entropy(logits, targets, *args, **kwargs):
            losses = original_ce(logits, targets, *args, **kwargs)
            if not mask_state.get("completion_mask_active"):
                return losses
            start = max(0, int(mask_state.get("loss_start", 0) or 0))
            shape = tuple(getattr(losses, "shape", ()) or ())
            if not shape:
                return losses
            seq = int(shape[-1])
            if seq <= 1 or start <= 0:
                return losses
            start = min(start, seq - 1)
            return losses[..., start:]

        self.engine.tokenizer = proxy
        p4.nn.losses.cross_entropy = completion_only_cross_entropy
        _clear_mlx(p4)

        try:
            result = dict(base_phase3(self) or {})
            if id(self.engine.model) != model_identity:
                raise RuntimeError("Phase 3 transaction replaced the benchmark model object")
            result["completion_only_loss"] = True
            result["transactional_update"] = True
            return result
        except BaseException:
            # Roll back the live model and persistent/DB state so a retry starts from
            # the exact pre-Phase-3 state rather than half-trained weights.
            try:
                _restore_trainables(p4, self.engine.model, snapshot)
            finally:
                if adapter_path:
                    try:
                        if had_adapter and backup_path and os.path.exists(backup_path):
                            shutil.copy2(backup_path, adapter_path)
                        elif not had_adapter and os.path.exists(adapter_path):
                            os.remove(adapter_path)
                    except Exception:
                        pass
                _reset_consolidated_flags(self, queued_ids)
            raise
        finally:
            p4.nn.losses.cross_entropy = original_ce
            self.engine.tokenizer = original_tokenizer
            if backup_path:
                try:
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                except Exception:
                    pass
            snapshot.clear()
            _clear_mlx(p4)

    p4._run_phase3_consolidation = transactional_completion_only_phase3
    p4._rsi_legacy_training_hardening_installed = True
