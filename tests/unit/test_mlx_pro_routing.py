"""Exercise MLX checkpoint routing through the desktop engine without loading weights."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.adaptation_capabilities import adaptation_capabilities
from core.awake_auto_hook import _remaining_generation_tokens, install_awake_auto_learning
from core.temperature_policy import pro_ladder
from tests.unit.test_strata_integration import engine_for, pro_module


def test_tensorfold_pro_uses_requested_budget_and_exact_runtime_input_count(pro_module):
    backend = SimpleNamespace(runtime="tensorfold_mlx", model=object(), is_loaded=True,
                              supports_entropy=False, training_ready=lambda: False,
                              n_ctx=2048, last_generation={}, last_branch_usage={},
                              api_prompt_count=Mock(return_value=19))
    engine = engine_for(pro_module, backend)
    engine.awake_consolidator = SimpleNamespace(learning_implementation="legacy", engine=backend)
    engine._context_budget_tokens = 1024
    engine.lossless_speed_options = {"native_branch_count": 3}
    engine.router.estimate_prompt_entropy_heuristic = Mock(side_effect=AssertionError("Invented entropy"))
    engine.router.route = Mock(side_effect=AssertionError("No logits were available to route"))
    calls = []

    def generate(prompt, branch_count, max_tokens, temperature, top_p):
        calls.extend((value, max_tokens) for value in temperature)
        backend.last_generation = {"status": "completed"}
        backend.last_branch_usage = {"prompt_tokens": 19 * branch_count, "completion_tokens": branch_count,
                                     "total_tokens": 20 * branch_count}
        return ["answer"] * branch_count

    backend.generate_branches = generate
    assert adaptation_capabilities(backend)["causal_lm"] is False
    assert _remaining_generation_tokens(engine, "formatted") == 1005
    with pytest.raises(RuntimeError, match="explicit"):
        engine.calculate_token_entropy("hello")
    with pytest.raises(ValueError, match="explicit"):
        engine.solve("hello")
    install_awake_auto_learning(pro_module.ProReasoningEngine)
    decisions = []
    assert "".join(engine.stream_solve("hello", cancel_event=threading.Event(),
                                       decision_callback=decisions.append)) == "answer"
    assert decisions[0]["branch_count"] == 3
    assert calls == [(t, 1005) for t in pro_ladder(3)]
    assert engine._last_stream_pro_meta["entropy"] is None
    assert engine._last_stream_pro_meta["branch_count"] == 4
    assert backend.last_branch_usage == {"prompt_tokens": 76, "completion_tokens": 4, "total_tokens": 80}


@pytest.mark.parametrize("system", ["Darwin", "Linux", "Windows"])
@pytest.mark.parametrize("implementation", ["legacy", "managed", "hybrid"])
def test_original_native_checkpoint_preserves_selected_learning_implementation(
        pro_module, monkeypatch, tmp_path, system, implementation):
    import core.cross_platform_mlx as dispatch

    engine = engine_for(pro_module)
    engine.settings.auto_download = False
    engine.settings.learning_implementation = implementation
    engine.lora_adapter_path = None
    engine.learning_scope = SimpleNamespace(kind="model")
    engine.awake_consolidator = SimpleNamespace(engine=None)
    backend = SimpleNamespace(model=object(), is_mlx_available=True, load_model=lambda: True)
    native = Mock(return_value=backend)
    monkeypatch.setattr(pro_module, "MLXReasoningBackend", native)
    monkeypatch.setattr(dispatch, "native_mlx_error", lambda: None)
    monkeypatch.setattr(dispatch, "platform", SimpleNamespace(
        system=lambda: system, machine=lambda: "arm64" if system == "Darwin" else "AMD64"))
    monkeypatch.delenv("SMARTAI_MLX_RUNTIME", raising=False)
    monkeypatch.setattr(pro_module, "UniversalControllerBackend", Mock(side_effect=AssertionError("Replaced MLX weights")))
    result = engine.load_model("Selected local MLX", str(tmp_path),
                               model_info={"controller_runtime": "mlx_lm"})
    assert result["status"] == "loaded", result
    assert native.call_args.kwargs["model_path"] == result["path"] == str(tmp_path)
    assert engine.mlx_engine is engine.awake_consolidator.engine is backend
    assert backend.learning_implementation == implementation and backend.learning_scope is engine.learning_scope


@pytest.mark.parametrize("runtime", ["mlx-vlm", "repo_mlx_vlm", "jang-vlm"])
def test_legacy_mlx_controller_alias_tracks_loaded_owner_until_unload(pro_module, monkeypatch, tmp_path, runtime):
    engine = engine_for(pro_module)
    engine.settings.auto_download = False
    engine.awake_consolidator = SimpleNamespace(engine=None)
    backend = SimpleNamespace(runtime=runtime, is_mlx_available=True, can_train=False,
                              load_model=lambda: True, unload_model=Mock())
    monkeypatch.setattr(pro_module, "UniversalControllerBackend", Mock(return_value=backend))
    receipt = engine.load_model("Saved MLX controller", str(tmp_path), model_info={"controller_runtime": runtime})
    assert receipt["status"] == "loaded" and receipt["backend"] == runtime
    assert receipt["trainable"] is False
    assert engine.mlx_engine is engine.controller_backend is engine.awake_consolidator.engine is backend
    assert engine.unload_model()["status"] == "unloaded"
    assert engine.mlx_engine is engine.controller_backend is None
    backend.unload_model.assert_called_once()
