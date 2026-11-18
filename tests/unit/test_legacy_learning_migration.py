"""Exercise the upgrade from master-era model-wide artifacts through the public loader."""
import hashlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from config.settings import Settings
from core.memory_scopes import LearningScope
from core.pro_engine import ProReasoningEngine
from tests.unit.gguf_fixtures import write_gguf
from tests.unit.test_native_context_learning import tiny_controller


def engine_for(tmp_path):
    return ProReasoningEngine(settings=Settings(use_mock=False, auto_download=False,
                              database_path=str(tmp_path / "chat.db")))


@pytest.mark.parametrize("kind", ["gguf", "bitnet"])
def test_native_upgrade_requires_confirmed_original_and_prefers_committed_state(kind, tmp_path, monkeypatch):
    source = tmp_path / "base.gguf"
    write_gguf(source)
    old_root = tmp_path / "master-era"
    old_root.mkdir()
    info = {"backend_family": kind, "controller_runtime": kind, "auto_memory_placement": False}
    if kind == "gguf":
        monkeypatch.setenv("SMARTAI_GGUF_ADAPTER_ROOT", str(old_root))
        legacy = old_root / "adapter.gguf"
        factory = Mock(return_value=SimpleNamespace())
        monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=factory))
    else:
        from core.engines import bitnet_cpp_engine as bitnet
        monkeypatch.setenv("SMARTAI_BITNET_TRAINING_ROOT", str(old_root))
        legacy = old_root / "learned-i2_s.gguf"
        factory = Mock(return_value=Mock(poll=lambda: None))
        monkeypatch.setattr(bitnet, "subprocess", SimpleNamespace(Popen=factory, DEVNULL=-3))
        monkeypatch.setattr(bitnet.BitNetCppReasoningBackend, "find_server", lambda self: tmp_path / "server")
        monkeypatch.setattr(bitnet.BitNetCppReasoningBackend, "_request", lambda *args, **kwargs: {"status": "ok"})
    engine = engine_for(tmp_path)

    def load(selected=None):
        options = dict(info, legacy_adapter_path=str(selected)) if selected else info
        return engine.load_model("Local upgrade", model_path=str(source), model_info=options)

    record = {"path": str(legacy), "format": kind}
    try:
        # A file removed between the confirmation and retry cannot yield success with the base alone.
        missing = load(legacy)
        assert missing["status"] == "error" and "missing" in missing["error"]
        assert "learning_migration" not in missing
        write_gguf(legacy)
        before = legacy.read_bytes()
        for selection in (None, source):
            result = load(selection)
            assert result["status"] == "error"
            assert result["learning_migration"] == {**record, "status": "legacy_adapter_required"}
        factory.assert_not_called()
        if kind == "gguf":
            factory.side_effect = RuntimeError("Incompatible adapter")
        else:
            write_gguf(legacy, layers=2)
        incompatible = load(legacy)
        assert incompatible["status"] == "error" and "learning_migration" not in incompatible
        if kind == "gguf":
            factory.side_effect = None
        else:
            legacy.write_bytes(before)
        imported = load(legacy)
        assert imported["status"] == "loaded", imported
        assert imported["learning_migration"] == {**record, "status": "legacy_adapter_loaded"}
        if kind == "gguf":
            assert factory.call_args.kwargs["lora_path"] == str(legacy)
            backend = engine.gguf_backend
        else:
            command = factory.call_args.args[0]
            assert command[command.index("-m") + 1] == str(legacy)
            backend = engine.bitnet_backend
        store = backend.learning_state
        assert store.current_checkpoint() is None

        engine.learning_scope = LearningScope("project", "another-project")
        separate = load(legacy)
        assert separate["status"] == "loaded" and "learning_migration" not in separate
        backend = engine.gguf_backend if kind == "gguf" else engine.bitnet_backend
        assert not backend.adapter_path

        with store.begin() as transaction:
            write_gguf(transaction.directory / "managed.gguf")
            transaction.commit("managed.gguf")
        engine.learning_scope = LearningScope()
        managed = load(legacy)
        assert managed["status"] == "loaded" and "learning_migration" not in managed
        backend = engine.gguf_backend if kind == "gguf" else engine.bitnet_backend
        assert backend.adapter_path == store.current_checkpoint()
        assert legacy.read_bytes() == before
    finally:
        engine.unload_model()


def test_controller_upgrade_continues_real_peft_without_overwriting_legacy(tiny_controller, tmp_path, monkeypatch):
    backend = tiny_controller
    source = backend.model_path
    old_root = tmp_path / "controller_lora"
    key = hashlib.sha256(source.encode()).hexdigest()[:16]
    legacy = old_root / key
    backend._ensure_transformers_lora().save_pretrained(legacy, save_embedding_layers=False)
    original = {p.name: p.read_bytes() for p in legacy.iterdir() if p.is_file()}
    backend.unload_model()
    monkeypatch.setenv("SMARTAI_CONTROLLER_ADAPTER_ROOT", str(old_root))
    engine = engine_for(tmp_path)

    def load(selected=None):
        info = dict(backend.model_info)
        if selected:
            info["legacy_adapter_path"] = str(selected)
        return engine.load_model("Local controller upgrade", model_path=source, model_info=info)

    record = {"format": "peft", "path": str(legacy)}
    try:
        for selection in (None, legacy / "adapter_config.json"):
            result = load(selection)
            assert result["status"] == "error"
            assert result["learning_migration"] == {**record, "status": "legacy_adapter_required"}
        imported = load(legacy)
        assert imported["status"] == "loaded", imported
        assert imported["learning_migration"] == {**record, "status": "legacy_adapter_loaded"}
        active = engine.controller_backend
        assert active.adapter_path == str(legacy) and active.model.peft_config
        active.train_mini_batch({}, [{"prompt": "vault color", "completion": "slate"}],
                                steps=1, validation_callback=lambda candidate: True)
        committed = active.learning_state.current_checkpoint()
        assert committed and committed != str(legacy)
        restored = load()
        assert restored["status"] == "loaded" and "learning_migration" not in restored
        assert engine.controller_backend.adapter_path == committed
        assert {p.name: p.read_bytes() for p in legacy.iterdir() if p.is_file()} == original

        engine.learning_scope = LearningScope("project", "separate")
        separate = load(legacy)
        assert separate["status"] == "loaded" and "learning_migration" not in separate
        assert engine.controller_backend.adapter_path is None
    finally:
        engine.unload_model()
