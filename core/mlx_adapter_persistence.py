"""Persist and restore only the active MLX LoRA tensors on the model-owner thread."""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
import shutil
import tempfile


def _infer_rank(weights):
    from core.mlx_learning import is_adapter_key

    if not weights or not all(is_adapter_key(key) for key in weights):
        raise ValueError("Saved MLX checkpoint must contain only LoRA factors")
    ranks = set()
    for key, tensor in weights.items():
        prefix, _, factor = key.rpartition(".")
        a = factor == "lora_a"
        peer = f"{prefix}.{'lora_b' if a else 'lora_a'}"
        shape = tuple(tensor.shape)
        if len(shape) != 2 or peer not in weights:
            raise ValueError(f"Missing or malformed MLX adapter factor {key}")
        ranks.add(int(shape[1 if a else 0]))
    if len(ranks) != 1 or min(ranks) < 1:
        raise ValueError("Inconsistent rank in saved MLX adapter")
    return ranks.pop()


def _validate_weights(weights, expected):
    from core.mlx_learning import check_finite

    if weights.keys() != expected.keys():
        raise ValueError("Saved MLX adapter keys differ from the loaded language model")
    for key, value in weights.items():
        if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError(f"Saved MLX adapter shape or dtype mismatch at {key}")
    check_finite(weights.values(), "checkpoint weights")


def switch_learning_checkpoint(backend, scope):
    """Swap authenticated raw LoRA modules while retaining the same resident foundation."""
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from core.learning_state import model_identity
    from core.mlx_learning import _projection_targets, ensure_adapters
    from core.native_speed import runtime_lock
    from types import SimpleNamespace

    with runtime_lock(backend):
        prior = getattr(backend, "learning_state", None)
        source = getattr(backend, "_mlx_loaded_source", None)
        if prior is None or not source or not backend.is_mlx_available or backend.model is None:
            return False
        current = prior.current_checkpoint()
        if current != (backend.adapter_path or None) or getattr(backend, "_mlx_imported_source", None):
            return False
        info = dict(getattr(backend, "model_info", {}) or {})
        if not Path(backend.model_path).expanduser().exists():
            info.setdefault("repo_id", str(backend.model_path))
        if model_identity(source, info, state_root=prior.root) != prior.identity:
            raise RuntimeError("The resident MLX source differs from its learned model identity")
        store = prior.for_scope(scope)
        manifest = store.current_manifest()
        if manifest and (manifest["checkpoint"] != "adapters.safetensors" or manifest["checkpoint"] not in manifest["artifacts"]):
            return False
        selected = str(store.directory / "generations" / manifest["generation"] / manifest["checkpoint"]) if manifest else None
        language = backend.get_training_model()
        previous = {path: module for path, module in language.named_modules()
                    if hasattr(module, "lora_a") and hasattr(module, "lora_b")}
        if previous and (not current or any(getattr(module, "linear", None) is None for module in previous.values())):
            return False
        weights, rank, layout, scale = {}, None, None, None
        if selected:
            if not selected.endswith(".safetensors"):
                return False
            weights, metadata = mx.load(selected, return_metadata=True)
            if metadata.get("smartai_identity") != store.identity_digest:
                raise ValueError("The selected raw MLX adapter has no matching model identity")
            rank = _infer_rank(weights)
            layout = metadata.get("smartai_lora_layout", "legacy")
            scale = float(metadata.get("smartai_lora_scale", 2.0))
            if layout not in {"head", "legacy", "packed_terminal"} or not math.isfinite(scale) or scale <= 0:
                raise ValueError("Invalid managed MLX adapter layout or scale")
        old_modules = [(p, k, getattr(p, k)) for p, k in _projection_targets(language, previous)]
        installed = []
        proxy = SimpleNamespace(model=language, _vlm_runtime=getattr(backend, "_vlm_runtime", None))
        backend.invalidate_prompt_reuse("learning scope changed")
        try:
            for parent, key, adapter in old_modules:
                setattr(parent, key, adapter.linear)
            if selected:
                targets = {key.rsplit(".", 1)[0] for key in weights}
                expected = ensure_adapters(proxy, rank, scale, layout, target_paths=targets, installed=installed)
                _validate_weights(weights, expected)
                language.update(tree_unflatten(list(weights.items())))
                mx.eval(*weights.values())
        except BaseException:
            for parent, key, base in reversed(installed):
                setattr(parent, key, base)
            for parent, key, adapter in old_modules:
                setattr(parent, key, adapter)
            raise
        backend.learning_state = store
        backend.learning_scope = store.scope
        backend.learning_identity = store.identity
        backend.adapters = weights
        backend.last_learning_metrics = manifest["metadata"].get("metrics", {}) if manifest else {}
        backend._last_managed_checkpoint = backend.adapter_path = backend.last_restored_adapter_path = selected
        backend._mlx_parent_generation = manifest["generation"] if manifest else None
        backend.last_restored_adapter_rank = rank
        backend.last_restored_adapter_provenance = "model-bound" if selected else None
        backend._packed_learning_fallback = None
        return True


def resume_optimizer(backend, optimizer, txn, learning_rate):
    if txn is None or txn.parent is None:
        return False
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    from core.mlx_learning import check_finite

    parent = txn.store.current_manifest()
    if parent["generation"] != txn.parent:
        raise RuntimeError("MLX optimizer generation changed while preparing the candidate")
    if "adamw.safetensors" not in parent["artifacts"]:
        return False  # A prior version may have persisted only adapter weights.
    file = txn.store.directory / "generations" / txn.parent / "adamw.safetensors"
    saved = mx.load(str(file))
    optimizer.init(backend.model.trainable_parameters())
    expected = dict(tree_flatten(optimizer.state))
    if saved.keys() != expected.keys() or any(saved[k].shape != v.shape for k, v in expected.items()):
        raise ValueError("Saved MLX AdamW state does not match active adapter parameters")
    check_finite(saved.values(), "saved optimizer state")
    optimizer.state = tree_unflatten(list(saved.items()))
    optimizer.learning_rate = learning_rate
    return True


def _commit_with_export(backend, txn, checkpoint, details, save_path, cancel_event):
    """Keep the caller's standalone copy and verified managed generation consistent on failure."""
    from core.learning_state import (
        _fsync_directory, _fsync_staged_file, _process_commit_lock, write_checkpoint_manifest,
    )
    import hashlib

    target = Path(save_path).expanduser().resolve() if save_path else None
    protected_root = getattr(txn.store, "implementation_base_root", txn.store.root)
    if target is not None and target.is_relative_to(protected_root):
        parts = target.relative_to(protected_root).parts
        if parts and parts[0] == "legacy-implementation":
            parts = parts[1:]
        managed = parts and (parts[0] in {"scopes", "identity_cache", "export_locks"}
                             or len(parts[0]) == 64 and all(c in "0123456789abcdef" for c in parts[0]))
        if managed:
            # Earlier callers passed the live adapter_path back as save_path.
            # That immutable generation remains a recoverable parent.
            current = getattr(backend, "_last_managed_checkpoint", None)
            if not current or Path(current).resolve() != target:
                raise ValueError("An external MLX save path cannot overwrite managed learning state")
            target = None
    if target is None:
        return txn.commit(checkpoint=checkpoint.name, metadata=details, cancel_event=cancel_event)

    target.parent.mkdir(parents=True, exist_ok=True)
    lock = protected_root / "export_locks" / hashlib.sha256(os.fsencode(target)).hexdigest()
    lock.mkdir(parents=True, exist_ok=True)
    with _process_commit_lock(lock), tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        stage = Path(temporary) / "adapters.safetensors"
        shutil.copyfile(checkpoint, stage)
        _fsync_staged_file(stage)
        write_checkpoint_manifest(backend, stage)
        published = []
        try:
            for source, destination in ((stage, target), (Path(str(stage) + ".memory.json"), Path(str(target) + ".memory.json"))):
                backup = source.with_suffix(".previous") if destination.exists() else None
                if backup is not None:
                    shutil.copy2(destination, backup)
                os.replace(source, destination)
                published.append((destination, backup))
            _fsync_directory(target.parent)
            return txn.commit(checkpoint=checkpoint.name, metadata=details, cancel_event=cancel_event)
        except BaseException:
            if not txn.committed:
                for destination, backup in reversed(published):
                    if backup is None:
                        destination.unlink(missing_ok=True)
                    else:
                        os.replace(backup, destination)
                _fsync_directory(target.parent)
            raise


def persist_candidate(backend, candidate, save_path, txn, metadata, metrics, cancel_event, optimizer):
    import mlx.core as mx
    from core.mlx_learning import cancel_if_requested

    if txn is None and not save_path:
        return
    store = getattr(backend, "learning_state", None)
    provenance = {}
    if store is not None:
        provenance["smartai_identity"] = store.identity_digest
    scales = {float(module.scale) for _, module in backend.model.named_modules()
              if hasattr(module, "lora_a") and hasattr(module, "lora_b")}
    if len(scales) != 1:
        raise ValueError("Checkpoint requires adapters with a consistent LoRA scale")
    provenance["smartai_lora_scale"] = str(scales.pop())
    provenance["smartai_lora_layout"] = (
        "head" if getattr(getattr(backend.model, "lm_head", None), "_smartai_output_lora", False)
        else "packed_terminal" if any(getattr(m, "_smartai_packed_lora", False) for _, m in backend.model.named_modules())
        else "legacy")
    destination = txn.directory / "adapters.safetensors" if txn is not None else Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".next.safetensors", delete=False) as out:
        staging = Path(out.name)
    try:
        mx.save_safetensors(str(staging), candidate, metadata=provenance)
        saved = mx.load(str(staging))
        _validate_weights(saved, candidate)
        if not bool(mx.all(mx.stack([mx.array_equal(value, saved[key]) for key, value in candidate.items()])).item()):
            raise RuntimeError("MLX adapter serialization changed candidate weights")
        del saved
        cancel_if_requested(cancel_event)
        os.replace(staging, destination)
        if txn is not None:
            from mlx.utils import tree_flatten
            from core.mlx_learning import check_finite
            state = dict(tree_flatten(optimizer.state))
            check_finite(state.values(), "optimizer state")
            mx.save_safetensors(str(txn.directory / "adamw.safetensors"), state)
            details = {**metadata, "metrics": metrics}
            source = getattr(backend, "_mlx_imported_source", None)
            if source is not None:
                details["imported_adapter"] = source
            cancel_if_requested(cancel_event)
            _commit_with_export(backend, txn, destination, details, save_path, cancel_event)
            backend._last_managed_checkpoint = str(destination)
            backend._mlx_parent_generation = txn.generation
        backend.adapter_path = str(destination)
    finally:
        staging.unlink(missing_ok=True)


def _import_identity(path):
    from core.learning_state import file_sha256, identity_digest
    source = Path(path).resolve()
    if source.is_dir():
        files = {p.relative_to(source).as_posix(): file_sha256(p)
                 for p in sorted(source.rglob("*")) if p.is_file()}
        digest = identity_digest(files)
    else:
        digest = file_sha256(source)
    return {"path": str(source), "sha256": digest}


def install_mlx_adapter_persistence(cls):
    if getattr(cls, "_raw_adapter_persistence_installed", False):
        return

    original_load = cls.load_model

    def load_with_raw_adapter_restore(self):
        from core.learning_state import bind_learning_state, validate_checkpoint

        self._mlx_loaded_source = None
        resolve = getattr(self, "_resolve_mlx_source", None)
        if callable(resolve):
            try:
                self._mlx_loaded_source = resolve()
            except Exception as exc:
                self.is_mlx_available = False
                self.last_error = f"Could not resolve the selected MLX snapshot: {exc}"
                return False
        info = dict(getattr(self, "model_info", {}) or {})
        if self._mlx_loaded_source and not Path(self.model_path).expanduser().exists():
            if not info.get("repo_id"):
                info["repo_id"] = str(self.model_path)
        store = bind_learning_state(self, model_path=self._mlx_loaded_source, model_info=info,
                                    state_root=getattr(self, "learning_state_root", None))
        configured = str(getattr(self, "adapter_path", "") or "")
        previous = getattr(self, "_last_managed_checkpoint", None)
        explicit = bool(configured and configured != previous and os.path.exists(configured))
        from core.pro_runtime_hardening import DEFAULT_MLX_ADAPTER_PATH
        if (configured and getattr(self, "_explicit_adapter_path", None) is False
                and Path(configured).resolve() == Path(DEFAULT_MLX_ADAPTER_PATH).resolve()):
            explicit = False
        current = store.current_checkpoint()
        source = None
        if explicit and configured != current:
            if store.scope.kind != "model":
                raise RuntimeError("A scoped model must restore its own committed adapter generation")
            source = _import_identity(configured)
            prior = store.current_manifest(verify=False)
            if prior and prior["metadata"].get("imported_adapter") == source:
                explicit = False  # User-selected seed has since been trained.
        checkpoint = configured if explicit else current
        self._mlx_imported_source = source
        self._mlx_parent_generation = None
        self.adapter_path = checkpoint
        raw = bool(checkpoint and os.path.isfile(checkpoint))
        if raw:
            self.adapter_path = None
        try:
            ok = original_load(self)
        finally:
            self.adapter_path = checkpoint
        if not ok:
            return False

        try:
            if not checkpoint or not os.path.isfile(checkpoint):
                if current and explicit:
                    raise RuntimeError("Selected adapter package differs from the model's active learned generation")
                if explicit and getattr(self, "_vlm_runtime", None):
                    raise RuntimeError("The selected MLX-VLM loader cannot attach an external adapter package; select its raw language LoRA checkpoint")
                self.learning_migration_status = None
                if not explicit and current is None and store.scope.kind == "model":
                    legacy = Path(DEFAULT_MLX_ADAPTER_PATH)
                    if legacy.is_file():
                        self.learning_migration_status = {
                            "status": "legacy_adapter_available", "path": str(legacy),
                            "message": "An older shared LoRA has no recorded base model. Select its file as lora_adapter_path on the original model to migrate it.",
                        }
                        logging.getLogger(__name__).warning("%s %s", self.learning_migration_status["message"], legacy)
                return True
            self.learning_migration_status = None
            import mlx.core as mx
            from mlx.utils import tree_unflatten

            manifest = validate_checkpoint(self, checkpoint, allow_legacy=explicit)
            weights, metadata = mx.load(checkpoint, return_metadata=True)
            digest = metadata.get("smartai_identity")
            if digest and digest != store.identity_digest:
                raise ValueError("Selected MLX adapter belongs to a different model or tokenizer")
            if not isinstance(weights, dict) or not weights:
                raise ValueError("Saved MLX adapter contains no tensors")
            training_model = self.get_training_model()
            # Older media pause snapshots were saved from the enclosing VLM.
            if training_model is not self.model and all(k.startswith("language_model.") for k in weights):
                weights = {k.removeprefix("language_model."): v for k, v in weights.items()}
            matched = current == checkpoint
            if current and not matched:
                # Media resume snapshots historically carried raw factors only.
                # Recover scale/layout from the authenticated generation when
                # every tensor agrees, before reconstructing the live modules.
                ancestor, ancestor_info = mx.load(current, return_metadata=True)
                matched = weights.keys() == ancestor.keys() and all(
                    bool(mx.array_equal(value, ancestor[key]).item()) for key, value in weights.items())
                if not matched:
                    raise ValueError("Selected adapter differs from the model's active learned generation")
                for key in ("smartai_lora_scale", "smartai_lora_layout"):
                    if key in ancestor_info:
                        if key in metadata and metadata[key] != ancestor_info[key]:
                            raise ValueError(f"Selected adapter has inconsistent saved {key}")
                        metadata[key] = ancestor_info[key]
            rank = _infer_rank(weights)
            scale = float(metadata.get("smartai_lora_scale", 2.0))
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("Invalid saved LoRA scale")
            only_head = set(weights) == {"lm_head.lora_a", "lm_head.lora_b"}
            specialized = getattr(self, "_vlm_runtime", None) in {"mlx_repo_vlm", "jang_vlm"}
            layout = metadata.get("smartai_lora_layout", "head" if only_head and specialized else "legacy")
            if layout not in {"legacy", "head", "packed_terminal"} or (layout == "head" and not only_head):
                raise ValueError("Invalid MLX adapter layout in checkpoint metadata")
            target_paths = {key.rsplit(".", 1)[0] for key in weights}
            expected = self.inject_lora_adapters(r=rank, scale=scale, layout=layout, target_paths=target_paths)
            _validate_weights(weights, expected)
            training_model.update(tree_unflatten(list(weights.items())))
            mx.eval(*weights.values())
            self.adapters = dict(weights)
            self.last_restored_adapter_path = checkpoint
            self.last_restored_adapter_rank = rank
            self.last_restored_adapter_provenance = (
                "model-bound" if manifest or digest else "explicit-unverified-legacy")
            self._last_managed_checkpoint = current if matched else None
            if matched:
                self._mlx_imported_source = None  # Preserve the original seed recorded by the parent.
                self._mlx_parent_generation = Path(current).parent.name
            return True
        except Exception as exc:
            self.is_mlx_available = False
            self.last_error = f"Failed to restore learned MLX adapter checkpoint {checkpoint}: {exc}"
            self.adapters = {}
            cleanup = getattr(self, "unload_model", None)
            if callable(cleanup):
                cleanup()
            raise RuntimeError(self.last_error) from exc

    cls.load_model = load_with_raw_adapter_restore
    cls._raw_adapter_persistence_installed = True
