"""Application and CLI selection preserve both existing learning implementations."""
from types import SimpleNamespace

import pytest

from config.settings import Settings
from core.learning_implementation import get_learning_implementation, resolve_learning_mode


def test_application_defaults_to_hybrid_but_explicit_and_direct_api_choices_stay_available(monkeypatch):
    monkeypatch.delenv("LEARNING_IMPLEMENTATION", raising=False)
    assert Settings().learning_implementation == "hybrid"
    assert Settings(learning_implementation="legacy").learning_implementation == "legacy"
    assert Settings(learning_implementation="managed").learning_implementation == "managed"
    assert get_learning_implementation(None) == "legacy"
    owner = SimpleNamespace(learning_implementation="hybrid")
    assert resolve_learning_mode(owner, None, {}) == "hybrid"
    assert resolve_learning_mode(owner, None, {"cancel_event": None}) == "hybrid"
    assert resolve_learning_mode(owner, "managed", {"cancel_event": None}) == "managed"
    assert resolve_learning_mode(owner, "legacy", {}) == "legacy"
    with pytest.raises(ValueError, match="require managed"):
        resolve_learning_mode(owner, "legacy", {"cancel_event": None})


@pytest.mark.parametrize("implementation", [None, "managed", "hybrid"])
def test_online_cli_forwards_implementation_and_restores_its_loaded_model(tmp_path, monkeypatch, implementation):
    import core
    from config import settings as settings_module
    from tools import online_learning

    monkeypatch.setattr(settings_module, "_global_settings", settings_module._global_settings)
    monkeypatch.setenv("SMARTAI_DATA_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model = tmp_path / "model"
    model.mkdir()
    seen = []

    class Engine:
        def __init__(self, settings):
            seen.append(settings.learning_implementation)

        def set_learning_scope(self, scope):
            assert scope.kind == "model"

        def load_model(self, name, **kwargs):
            assert kwargs["model_path"] == str(model)
            return {"status": "loaded", "backend": "cpu"}

        def run_online_learning(self, request, **kwargs):
            assert request == {"method": "capabilities"}
            return {"status": "completed", "capabilities": {}}

        def unload_model(self):
            seen.append("unloaded")
            return {"status": "unloaded"}

    monkeypatch.setattr(core, "ProReasoningEngine", Engine)
    args = ["--model-path", str(model), "--backend", "cpu", "--capabilities"]
    if implementation is not None:
        args += ["--learning-implementation", implementation]
    assert online_learning.main(args) == 0
    assert seen == [implementation or "managed", "unloaded"]
