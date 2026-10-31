"""The application keeps native placement and cleanup ownership across loads."""
from types import SimpleNamespace

from tests.unit.test_strata_integration import engine_for, pro_module


def test_failed_private_offload_cleanup_remains_owned_until_retry(pro_module):
    controller = SimpleNamespace(runtime="transformers_auto", _offload_directory="owned/session", model=None)
    calls = []
    def unload():
        calls.append(True)
        if len(calls) > 1:
            controller._offload_directory = None
    controller.unload_model = unload
    engine = engine_for(pro_module, controller)
    result = engine.unload_model()
    assert result["status"] == "error" and "offload files" in result["error"]
    assert engine.controller_backend is controller
    assert engine.unload_model()["status"] == "unloaded"
    assert engine.controller_backend is None


def test_native_placement_uses_requested_context_and_cap_with_speed_tuning_disabled(pro_module, tmp_path, monkeypatch):
    model = tmp_path / "chosen.gguf"
    model.write_bytes(b"selected fixture; native loader replaced in this routing test")
    captured = {}
    class Native:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.memory_placement = {"message": "verified test placement", "context_tokens": kwargs["n_ctx"]}
        def load_model(self):
            return True
        def training_ready(self):
            return False
    monkeypatch.setattr(pro_module, "GGUFReasoningBackend", Native)
    engine = engine_for(pro_module)
    engine.auto_speed_enabled = False
    result = engine.load_model("Chosen model", str(model), model_info={
        "controller_runtime": "gguf", "backend_family": "gguf", "gguf_file": model.name,
        "gguf_files": [model.name], "model_type": "text", "default_context": 32768,
    })
    assert result["status"] == "loaded", result
    assert captured["model_path"] == str(model)
    assert captured["n_ctx"] == 8192 and captured["memory_limit_gb"] == 17.5
    assert captured["auto_memory_placement"] is True
    assert result["memory_placement"]["context_tokens"] == 8192
