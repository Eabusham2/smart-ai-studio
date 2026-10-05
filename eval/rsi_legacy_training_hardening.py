"""Restore useful pre-rewrite training semantics around current Phase 3.

This does not replace the current optimizer/OGP/fail-closed pipeline. It wraps the
existing Phase-3 implementation so the real older strengths are retained:

1. Completion-only training: optimize the assistant answer, not reproduction of the
   user prompt. The old sleep trainer masked prompt tokens; benchmark Phase 3 had
   regressed to CE over the entire prompt+completion sequence.
2. Fisher/EWC protection: compute real MLX Fisher information from the existing core
   anchor dataset and add its quadratic penalty to the current Learn/RSI loss. This
   uses the same production mechanism already used by /learn; no random/noise drift.
3. Transactional updates: snapshot the real trainable weights, adapter file and MoE
   buffers. Any exception, Ctrl+C, or fail-closed integrity error restores the exact
   pre-Phase-3 state and clears consolidation flags for the queued traces.
4. Release transient MLX graphs/caches when Phase 3 exits.

Current behavior stays authoritative for everything else: same loaded model, same
AdamW/OGP/raw-gradient fallback, same real-delta proof, same adapter persistence,
and the same verified Learn/RSI queue.
"""
from __future__ import annotations

import gc
import os
import shutil
import sqlite3
from typing import Any, Dict, List

from memory.anchor_dataset import get_anchor_texts


ASSISTANT_MARKER = "<|im_start|>assistant\n"
TRAIN_WINDOW_TOKENS = 256
FISHER_ANCHOR_COUNT = 4


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

        # Current core trains only the first 256 ids. Return the final 256 instead
        # so a long prompt cannot push the assistant completion out of the window.
        start = max(0, len(ids) - self._window)
        selected = ids[start:]
        local_assistant_start = max(0, len(prefix_ids) - start)

        # CE position j predicts target token j+1. Keep losses beginning one position
        # before the first assistant token so the first completion token is trained.
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


def _clone_array(p4: Any, value: Any) -> Any:
    try:
        return p4.mx.copy(value)
    except Exception:
        try:
            return value + p4.mx.zeros_like(value)
        except Exception:
            return p4.mx.array(value)


def _snapshot_trainables(p4: Any, model: Any) -> Dict[str, Any]:
    if not getattr(p4, "MLX_AVAILABLE", False) or model is None:
        return {}
    flat = dict(p4.mlx.utils.tree_flatten(model.trainable_parameters()))
    snapshot = {key: _clone_array(p4, value) for key, value in flat.items()}
    if snapshot:
        p4.mx.eval(*snapshot.values())
    return snapshot


def _restore_trainables(p4: Any, model: Any, snapshot: Dict[str, Any]) -> None:
    if not snapshot or model is None:
        return
    model.update(p4.mlx.utils.tree_unflatten(list(snapshot.items())))
    p4.mx.eval(model.parameters())


def _snapshot_moe_buffers(p4: Any, self: Any) -> Dict[str, Dict[str, Any]]:
    manager = getattr(self.engine, "moe_manager", None)
    if manager is None:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for name in ("adapters_buffer_a", "adapters_buffer_b"):
        value = getattr(manager, name, None)
        if isinstance(value, dict):
            cloned = {k: _clone_array(p4, v) for k, v in value.items()}
            if cloned:
                try:
                    p4.mx.eval(*cloned.values())
                except Exception:
                    pass
            out[name] = cloned
    return out


def _restore_moe_buffers(self: Any, snapshot: Dict[str, Dict[str, Any]]) -> None:
    manager = getattr(self.engine, "moe_manager", None)
    if manager is None:
        return
    for name, value in snapshot.items():
        setattr(manager, name, dict(value))


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


def _compute_real_fisher(p4: Any, self: Any) -> Dict[str, Any]:
    """Compute real MLX Fisher anchors on the same model, matching current /learn."""
    try:
        backend = p4._pro_backend(self)
        fisher = backend.compute_mlx_fisher(get_anchor_texts()[:FISHER_ANCHOR_COUNT])
        if not isinstance(fisher, dict):
            return {}
        fisher = {str(k): v for k, v in fisher.items()}
        arrays = [v for v in fisher.values() if hasattr(v, "shape")]
        if arrays:
            p4.mx.eval(*arrays)
        return fisher
    except Exception:
        # OGP remains active even when Fisher cannot be computed on a model/kernel.
        return {}


def install(p4: Any) -> None:
    """Wrap the final fail-closed Phase-3 stack."""
    if getattr(p4, "_rsi_legacy_training_hardening_installed", False):
        return

    base_phase3 = p4._run_phase3_consolidation

    def transactional_completion_only_phase3(self):
        # Non-MLX app evals use the production backend's own transactional trainer.
        # Never run MLX tensor snapshot/EWC monkeypatches against GGUF/BitNet/Torch.
        if (
            str(getattr(self.engine, "backend_key", "mlx") or "mlx").lower() != "mlx"
            or not getattr(p4, "MLX_AVAILABLE", False)
            or self.engine.model is None
        ):
            return base_phase3(self)

        model_identity = id(self.engine.model)
        snapshot = _snapshot_trainables(p4, self.engine.model)
        moe_snapshot = _snapshot_moe_buffers(p4, self)
        queued_ids = _queued_ids(p4, self)
        fisher = _compute_real_fisher(p4, self)
        try:
            ewc_lambda = float(getattr(self.engine.settings, "ewc_lambda", 400.0))
        except Exception:
            ewc_lambda = 400.0
        if not fisher:
            ewc_lambda = 0.0

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

        # New bounded Phase-3 code handles completion-only target selection itself
        # and consumes this EWC context directly. Older Phase-3 implementations keep
        # the historical tokenizer/loss monkeypatch path below.
        bounded = callable(getattr(p4, "_phase3_bounded_gradients", None))
        original_tokenizer = self.engine.tokenizer
        mask_state: Dict[str, Any] = {"loss_start": 0, "completion_mask_active": False}
        proxy = _CompletionWindowTokenizer(original_tokenizer, mask_state)
        original_ce = p4.nn.losses.cross_entropy
        original_value_and_grad = p4.nn.value_and_grad

        if bounded:
            self._phase3_ewc_context = {
                "fisher": fisher,
                "reference": snapshot,
                "lambda": float(ewc_lambda),
            }
        else:
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

            def ewc_value_and_grad(model, lossfn):
                if not fisher or not snapshot or ewc_lambda <= 0.0:
                    return original_value_and_grad(model, lossfn)

                def protected_loss(current_model):
                    base_loss = lossfn(current_model)
                    current = dict(
                        p4.mlx.utils.tree_flatten(current_model.trainable_parameters())
                    )
                    penalty = p4.mx.array(0.0)
                    matched = 0
                    for key, weight in current.items():
                        f_k = fisher.get(key)
                        ref = snapshot.get(key)
                        if f_k is None or ref is None:
                            continue
                        diff = weight - ref
                        penalty = penalty + p4.mx.sum(f_k * (diff ** 2))
                        matched += 1
                    if matched == 0:
                        return base_loss
                    return base_loss + (ewc_lambda / 2.0) * penalty

                return original_value_and_grad(model, protected_loss)

            self.engine.tokenizer = proxy
            p4.nn.losses.cross_entropy = completion_only_cross_entropy
            p4.nn.value_and_grad = ewc_value_and_grad

        _clear_mlx(p4)

        try:
            result = dict(base_phase3(self) or {})
            if id(self.engine.model) != model_identity:
                raise RuntimeError("Phase 3 transaction replaced the benchmark model object")
            result["completion_only_loss"] = True
            result["transactional_update"] = True
            result["ewc_enabled"] = bool(fisher)
            result["ewc_lambda"] = float(ewc_lambda)
            result["fisher_anchors"] = FISHER_ANCHOR_COUNT if fisher else 0
            result["bounded_phase3"] = bool(bounded)
            return result
        except BaseException:
            try:
                _restore_trainables(p4, self.engine.model, snapshot)
                _restore_moe_buffers(self, moe_snapshot)
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
            if bounded:
                try:
                    delattr(self, "_phase3_ewc_context")
                except Exception:
                    pass
            else:
                p4.nn.value_and_grad = original_value_and_grad
                p4.nn.losses.cross_entropy = original_ce
                self.engine.tokenizer = original_tokenizer
            if backup_path:
                try:
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                except Exception:
                    pass
            fisher.clear()
            snapshot.clear()
            moe_snapshot.clear()
            _clear_mlx(p4)

    p4._run_phase3_consolidation = transactional_completion_only_phase3
    p4._rsi_legacy_training_hardening_installed = True
