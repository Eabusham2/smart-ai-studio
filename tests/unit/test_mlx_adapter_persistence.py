import sys
import types

import numpy as np
import pytest

from core.mlx_adapter_persistence import _infer_rank, install_mlx_adapter_persistence
from tests.unit.test_mlx_learning_implementations import tiny_mlx


class FakeModel:
    def update(self, tree):
        self.updated = tree


class DummyMLX:
    def __init__(self, tmp_path, checkpoint):
        self.model_path = "test/mlx"
        self.learning_state_root = tmp_path / "state"
        self.model_info = {"resolved_revision": "a" * 40}
        self.adapter_path = str(checkpoint)
        self.weights = {"layer.q_proj.lora_a": np.zeros((4, 8), dtype="float32"),
                        "layer.q_proj.lora_b": np.zeros((8, 4), dtype="float32")}

    def load_model(self):
        self.original_saw_adapter_path = self.adapter_path
        self.model = FakeModel()
        self.is_mlx_available = True
        return True

    def get_training_model(self):
        return self.model

    def inject_lora_adapters(self, **kwargs):
        self.injected = kwargs
        return self.weights


install_mlx_adapter_persistence(DummyMLX)


def test_rank_uses_actual_factor_axes_even_for_a_narrower_hidden_dimension():
    assert _infer_rank({"h.lora_a": np.empty((4, 8)), "h.lora_b": np.empty((8, 4))}) == 8
    with pytest.raises(ValueError, match="only LoRA"):
        _infer_rank({"base.weight": np.empty((4, 4))})
    with pytest.raises(ValueError, match="rank"):
        _infer_rank({"h.lora_a": np.empty((4, 8)), "h.lora_b": np.empty((7, 4))})


@pytest.fixture
def restore_fixture(tmp_path, monkeypatch):
    checkpoint = tmp_path / "adapter.safetensors"
    checkpoint.write_bytes(b"legacy-fixture")
    engine = DummyMLX(tmp_path, checkpoint)
    weights = dict(engine.weights)
    metadata = {}
    mlx = types.ModuleType("mlx")
    mx = types.ModuleType("mlx.core")
    mx.load = lambda path, return_metadata=False: (weights, metadata) if return_metadata else weights
    mx.eval = lambda *args: None
    mx.isfinite, mx.all, mx.stack = np.isfinite, np.all, np.stack
    utils = types.ModuleType("mlx.utils")
    utils.tree_unflatten = dict
    for name, mod in (("mlx", mlx), ("mlx.core", mx), ("mlx.utils", utils)):
        monkeypatch.setitem(sys.modules, name, mod)
    return engine, weights, metadata


def test_raw_safetensors_checkpoint_restored_after_base_load(restore_fixture):
    engine, weights, _ = restore_fixture
    assert engine.load_model()
    assert engine.original_saw_adapter_path is None
    assert engine.injected == {"r": 8, "scale": 2.0, "layout": "legacy", "target_paths": {"layer.q_proj"}}
    assert engine.last_restored_adapter_rank == 8
    assert engine.last_restored_adapter_provenance == "explicit-unverified-legacy"
    assert engine.model.updated == weights


@pytest.mark.parametrize("issue", ["base", "missing", "shape", "nan", "identity"])
def test_invalid_checkpoint_never_updates_model(restore_fixture, issue):
    engine, weights, metadata = restore_fixture
    if issue == "base":
        weights["base.weight"] = np.ones((3, 3))
    elif issue == "missing":
        del weights["layer.q_proj.lora_a"]
    elif issue == "shape":
        weights["layer.q_proj.lora_a"] = np.zeros((1, 8), dtype="float32")
    elif issue == "nan":
        weights["layer.q_proj.lora_a"] = np.full((4, 8), np.nan, dtype="float32")
    elif issue == "identity":
        metadata["smartai_identity"] = "foreign-model"
    with pytest.raises(RuntimeError, match="Failed to restore"):
        engine.load_model()
    assert not hasattr(engine.model, "updated")
    assert not engine.is_mlx_available


def test_migrated_legacy_selection_resumes_its_updated_generation(restore_fixture):
    from pathlib import Path
    from core.mlx_adapter_persistence import _import_identity

    engine, weights, _ = restore_fixture
    legacy = engine.adapter_path
    assert engine.load_model()
    store = engine.learning_state
    with store.begin(metadata={"imported_adapter": _import_identity(legacy)}) as txn:
        (txn.directory / "adapters.safetensors").write_bytes(b"learned-generation")
        txn.commit("adapters.safetensors")
    reboot = DummyMLX(Path(engine.learning_state_root).parent, legacy)
    assert reboot.load_model()
    assert reboot.adapter_path == store.current_checkpoint()
    assert reboot._mlx_parent_generation == store.current_manifest()["generation"]
    assert reboot.last_restored_adapter_provenance == "model-bound"


def test_unbound_old_global_adapter_reports_migration_without_attaching(tmp_path, monkeypatch):
    import core.pro_runtime_hardening as hardening
    legacy = tmp_path / "consolidated_slow_lora" / "adapters.safetensors"
    legacy.parent.mkdir()
    legacy.write_bytes(b"existing-user-weights")
    monkeypatch.setattr(hardening, "DEFAULT_MLX_ADAPTER_PATH", str(legacy))
    backend = DummyMLX(tmp_path, legacy)
    backend._explicit_adapter_path = False
    assert backend.load_model() and backend.original_saw_adapter_path is None
    assert backend.learning_migration_status["status"] == "legacy_adapter_available"
    assert backend.learning_migration_status["path"] == str(legacy)
    assert legacy.read_bytes() == b"existing-user-weights"


def test_native_source_uses_requested_revision_for_first_fetch(tmp_path, monkeypatch):
    from core import learning_state
    from core import mlx_engine as engine
    exact = tmp_path / "snapshots" / ("3" * 40)
    exact.mkdir(parents=True)
    requested = {"repo_revision": "3" * 40, "download_required_files": ["config.json", "w.safetensors"]}
    monkeypatch.setattr(engine.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(engine.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(learning_state, "cached_model_source", lambda source, info: tmp_path / "absent")
    downloads = []
    def snapshot_download(**kwargs):
        downloads.append(kwargs)
        return str(exact)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    backend = types.SimpleNamespace(model_path="fixture/model", model_info=requested)
    assert engine.MLXReasoningBackend._resolve_mlx_source(backend) == str(exact)
    assert downloads == [{"repo_id": "fixture/model", "revision": "3" * 40,
                          "allow_patterns": ["config.json", "w.safetensors"]}]


def test_native_source_rejects_cached_repository_code(tmp_path, monkeypatch):
    from core import mlx_engine as engine
    source = tmp_path / "cached-model"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "modeling_custom.py").write_text("raise AssertionError('must not execute')")
    monkeypatch.setattr(engine.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(engine.platform, "machine", lambda: "arm64")
    backend = types.SimpleNamespace(model_path=str(source), model_info={})
    with pytest.raises(RuntimeError, match="Executable repository companions"):
        engine.MLXReasoningBackend._resolve_mlx_source(backend)


@pytest.mark.parametrize("package", [False, True])
def test_scoped_mlx_load_rejects_external_adapter_before_loading_base(restore_fixture, package):
    from pathlib import Path
    from core.memory_scopes import LearningScope

    engine, _, _ = restore_fixture
    if package:
        path = Path(engine.adapter_path).with_suffix("")
        path.mkdir()
        engine.adapter_path = str(path)
    engine.learning_scope = LearningScope("session", "one")
    with pytest.raises(RuntimeError, match="own committed adapter"):
        engine.load_model()
    assert not hasattr(engine, "original_saw_adapter_path")


@pytest.mark.parametrize("destination_mode", ["legacy", "managed", "hybrid"])
@pytest.mark.parametrize("bound_writer", [False, True], ids=["unbound", "different-store-root"])
def test_training_export_protects_foreign_generations_and_rolls_back(tiny_mlx, tmp_path, destination_mode, bound_writer):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from pathlib import Path
    from core.learning_state import bind_learning_state
    from core.memory_scopes import LearningScope

    rows = [{"prompt": "xyz", "completion": "abc"}]
    owner = tiny_mlx()
    owner.learning_implementation = destination_mode
    info = {"resolved_revision": "6" * 40}
    store = bind_learning_state(owner, model_info=info, state_root=tmp_path / "owner",
                                scope=LearningScope("project", "private") if destination_mode == "hybrid" else None)
    owner.train_mini_batch({}, rows, steps=1)
    target = Path(owner.adapter_path)
    owner_manifest = store.current_manifest()
    owner_bytes = target.read_bytes()

    writer = tiny_mlx()
    writer.learning_implementation = "hybrid"
    if bound_writer:
        bind_learning_state(writer, model_info=info, state_root=tmp_path / "unrelated")
    writer.train_mini_batch({}, rows, steps=1, learning_mode="hybrid", save_path=str(tmp_path / "writable.safetensors"))
    prior_path = writer.adapter_path
    prior_manifest = writer.learning_state.current_manifest() if bound_writer else None
    before = dict(tree_flatten(writer.model.parameters()))
    events = []
    def validate(candidate):
        events.append(any(not mx.array_equal(p, before[k]).item() for k, p in tree_flatten(candidate.model.parameters())))
        return True
    with pytest.raises(ValueError, match="cannot overwrite managed learning state"):
        writer.train_mini_batch({}, rows, steps=1, learning_mode="hybrid", save_path=str(target), validation_callback=validate)
    assert events == [True]
    assert writer.adapter_path == prior_path
    assert all(mx.array_equal(p, before[k]).item() for k, p in tree_flatten(writer.model.parameters()))
    assert target.read_bytes() == owner_bytes and store.current_manifest() == owner_manifest
    if bound_writer:
        assert writer.learning_state.current_manifest() == prior_manifest


@pytest.mark.parametrize("destination_mode", ["legacy", "managed", "hybrid"])
def test_own_current_generation_reference_publishes_only_a_successor(tiny_mlx, tmp_path, destination_mode):
    from pathlib import Path
    from core.learning_state import bind_learning_state

    backend = tiny_mlx()
    backend.learning_implementation = destination_mode
    store = bind_learning_state(backend, model_info={"resolved_revision": "7" * 40}, state_root=tmp_path)
    rows = [{"prompt": "xyz", "completion": "abc"}]
    backend.train_mini_batch({}, rows, steps=1)
    parent = Path(store.current_checkpoint())
    saved = parent.read_bytes()
    backend.train_mini_batch({}, rows, steps=1, save_path=str(parent))
    child = store.current_manifest()
    assert child["parent"] == parent.parent.name and store.current_checkpoint() != str(parent)
    assert parent.read_bytes() == saved and not Path(str(parent) + ".memory.json").exists()


def test_foreign_guard_protects_generation_metadata_and_future_artifacts(tmp_path):
    from types import SimpleNamespace
    from core.learning_state import LearningStateStore
    from core.mlx_adapter_persistence import _external_export_target

    identity = {"source": "fixture", "source_verified": True}
    store = LearningStateStore(identity, state_root=tmp_path / "foreign")
    with store.begin() as transaction:
        # The generation has been reserved, but another writer has not yet published its manifest.
        staged = transaction.directory / "new" / "extra.safetensors"
        with pytest.raises(ValueError, match="cannot overwrite managed learning state"):
            _external_export_target(SimpleNamespace(), None, staged)
        assert not staged.parent.exists()
        checkpoint = transaction.directory / "adapters.safetensors"
        checkpoint.write_bytes(b"verified")
        transaction.commit("adapters.safetensors")
    manifest = store.current_manifest()
    for filename in ("manifest.json", "adamw.safetensors", "new/extra.safetensors"):
        with pytest.raises(ValueError, match="cannot overwrite managed learning state"):
            _external_export_target(SimpleNamespace(), None, checkpoint.parent / filename)
    assert store.current_manifest() == manifest and not staged.parent.exists()

    # An ordinary export directory can also be named generations.
    ordinary = tmp_path / "outside" / "generations" / "my-run" / "copy.safetensors"
    assert _external_export_target(SimpleNamespace(), None, ordinary) == ordinary
