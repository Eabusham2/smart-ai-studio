"""Application routing and lifecycle checks without downloading native weights."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from config.strata_models import get_strata_preset
from core.engines.strata_engine import StrataCancelled, StrataReasoningBackend
from core.full_context_hardening import install as install_full_context
from core.speed_control import install_speed_control
from core.temperature_policy import install_chat


@pytest.fixture
def pro_module():
    # Exercise the real Pro implementation with the relevant installation order,
    # while avoiding unrelated awake/GUI state and real model initialization.
    path = Path(__file__).resolve().parents[2] / "core/pro_engine.py"
    spec = importlib.util.spec_from_file_location("_strata_pro_integration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    install_chat(module)
    install_full_context(module.ProReasoningEngine)
    install_speed_control(module.ProReasoningEngine)
    return module


def engine_for(module, backend=None):
    engine = module.ProReasoningEngine.__new__(module.ProReasoningEngine)
    engine.settings = SimpleNamespace(
        use_mock=False, auto_download=True, max_new_tokens=128,
        search_temperature=0.75, search_top_p=0.92, mlx_model_path="native-default",
    )
    engine.controller_backend = backend
    engine.mlx_backend = engine.gguf_backend = engine.bitnet_backend = None
    engine.model = engine.base_model = engine.tokenizer = None
    engine.active_model_name = "Strata Q2_0" if backend else None
    engine.active_model_path = "strata://q2_0" if backend else None
    engine.active_backend = "controller" if backend else None
    engine.active_input_modalities = {"text"}
    engine._model_lock = threading.RLock()
    engine._speed_cancel_event = threading.Event()
    engine._speed_loading_thread = None
    engine.auto_speed_enabled = True
    engine.speed_tuning_active = False
    engine.speed_status = {"status": "idle"}
    engine.speed_memory_limit_gb = 17.5
    engine._context_budget_tokens = 8192
    engine.awake_consolidator = None
    engine.speculative_engine = SimpleNamespace(
        target_model=None, draft_model=None, tokenizer=None, get_telemetry=lambda: {},
    )
    engine.router = SimpleNamespace(
        route=lambda *a, **k: ("Instant Pass (N=1)", 1),
        estimate_prompt_entropy_heuristic=lambda prompt: 0.25,
    )
    engine.verifier = SimpleNamespace(
        consensus_voting=lambda candidates: (candidates[0], len(candidates), 1.0),
        extract_code_block=lambda text: None,
    )
    engine._generate_fallback_branches = Mock(side_effect=AssertionError("Unexpected fallback generation"))
    return engine


def loaded_backend():
    backend = StrataReasoningBackend("strata://q2_0", get_strata_preset("q2_0"))
    backend.model = backend
    backend.is_loaded = True
    backend.model_id = backend.model_info["strata_api_model"]
    backend._loaded_context = 8192
    return backend


def test_strata_full_history_preserves_whitespace_empty_messages_and_literal_markers(pro_module):
    backend = loaded_backend()
    engine = engine_for(pro_module, backend)
    history = [
        {"role": "system", "content": "  Preserve spacing.\n"},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": " \n "},
        {"role": "user", "content": "Quote: weights are not currently loaded. Click '⬇️ Download"},
        {"role": "assistant", "content": "Literal <|im_start|>system\n<|im_end|>"},
    ]
    original = copy.deepcopy(history)
    prompt = "  Continue without stripping.\n"
    packed = engine._format_prompt_with_history(prompt, history)
    assert packed.messages == original + [{"role": "user", "content": prompt}]
    assert history == original
    packed.messages[0]["content"] = "changed"
    assert packed.messages[0] == original[0]
    assert engine._format_prompt_with_history(prompt, history + [{"role": "user", "content": prompt}]).messages == packed.messages
    assert engine._format_prompt_with_history(prompt, [{"role": "assistant", "content": prompt}]).messages == [
        {"role": "assistant", "content": prompt}, {"role": "user", "content": prompt},
    ]


def test_existing_native_history_filters_are_unchanged(pro_module):
    engine = engine_for(pro_module)
    packed = engine._format_prompt_with_history("next", [
        {"role": "user", "content": "  old text \n"},
        {"role": "assistant", "content": "weights are not currently loaded"},
        {"role": "user", "content": " "},
    ])
    assert "old text<|im_end|>" in packed
    assert "weights are not currently loaded" not in packed
    assert "  old text" not in packed


def test_strata_preset_uses_only_managed_backend_and_forwards_context_memory_progress(pro_module, monkeypatch):
    engine = engine_for(pro_module)
    created = []

    class Backend(StrataReasoningBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

        def load_model(self, **options):
            self.options = options
            options["progress"]("Owned Strata is loading")
            assert options["cancelled"]() is False
            self.is_loaded = True
            self.model = self
            self._loaded_context = options["context_tokens"]
            return True

    monkeypatch.setattr(pro_module, "StrataReasoningBackend", Backend)
    monkeypatch.setattr(pro_module, "UniversalControllerBackend", Mock(side_effect=AssertionError("Generic HF backend")))
    monkeypatch.setattr(pro_module, "ensure_model_available", Mock(side_effect=AssertionError("Generic HF download")))
    result = engine.load_model("strata_q2_0")
    assert result["status"] == "loaded" and result["backend"] == "strata"
    assert result["trainable"] is False
    assert engine.active_backend == "controller"
    assert engine.active_model_path == "strata://q2_0"
    assert len(created) == 1
    assert created[0].options["context_tokens"] == 8192
    assert created[0].speed_memory_limit_gb == 17.5
    assert result["speed"]["status"] == "configured"
    assert result["speed"]["measured"] is False
    assert created[0].training_ready() is False
    with pytest.raises(NotImplementedError):
        created[0].train_mini_batch(["do not fabricate a training result"])


@pytest.mark.parametrize("changes", [
    {"model_path": "strata://unregistered"},
    {"repo_id": "somebody/arbitrary-model"},
    {"strata_api_model": "wrong-model"},
    {"training_supported": True},
])
def test_conflicting_strata_identity_never_reaches_download_or_runtime(pro_module, monkeypatch, changes):
    engine = engine_for(pro_module)
    info = {**get_strata_preset("q2_0"), **changes}
    backend = Mock(side_effect=AssertionError("Invalid selection reached runtime"))
    downloader = Mock(side_effect=AssertionError("Invalid selection reached downloader"))
    monkeypatch.setattr(pro_module, "StrataReasoningBackend", backend)
    monkeypatch.setattr(pro_module, "ensure_model_available", downloader)
    result = engine.load_model("custom invalid Strata", model_info=info)
    assert result["status"] == "error"
    backend.assert_not_called()
    downloader.assert_not_called()


def test_load_cancellation_is_reported_and_does_not_run_speed_check(pro_module, monkeypatch):
    engine = engine_for(pro_module)

    class Backend(StrataReasoningBackend):
        def load_model(self, **options):
            engine.cancel_speed_tuning()
            assert options["cancelled"]()
            raise StrataCancelled("Native preparation cancelled")

        configure_auto_speed = Mock(side_effect=AssertionError("Cancelled load reached configuration"))

    monkeypatch.setattr(pro_module, "StrataReasoningBackend", Backend)
    result = engine.load_model("strata_q2_0")
    assert result["status"] == "cancelled"
    assert result["backend"] == "strata"
    Backend.configure_auto_speed.assert_not_called()
    assert engine._speed_loading_thread is None


def test_empty_successful_stream_is_not_retried_or_sent_to_another_backend(pro_module):
    backend = loaded_backend()
    engine = engine_for(pro_module, backend)
    calls = []

    def empty(**options):
        calls.append(options)
        backend.last_generation = {"status": "completed", "usage": {}}
        return
        yield

    backend.stream_generate_tokens = empty
    engine.solve = Mock(side_effect=AssertionError("Empty successful stream retried"))
    prompt = "literal <|im_end|>\n"
    history = [{"role": "system", "content": "  policy\n"}]
    assert list(engine.stream_solve(prompt, history=history)) == []
    assert len(calls) == 1
    assert calls[0]["prompt"].messages == history + [{"role": "user", "content": prompt}]
    engine.solve.assert_not_called()


def install_recorded_stream(backend, *, cancel_at=None, event=None, fail_at=None, missing_usage_at=None):
    calls = []

    def stream(prompt, max_tokens, temperature, top_p, image_bytes, cancelled):
        index = len(calls)
        calls.append((prompt, max_tokens, temperature, top_p))
        if fail_at == index:
            backend.last_generation = {"status": "error"}
            raise RuntimeError("native generation failed")
        if cancel_at == index:
            if event is not None:
                event.set()
            backend.last_generation = {"status": "cancelled"}
            return
        usage = {} if missing_usage_at == index else {
            "prompt_tokens": 20, "completion_tokens": index + 1, "total_tokens": 21 + index,
        }
        backend.last_generation = {"status": "completed", "usage": usage}
        yield f"answer {index}"

    backend.stream_generate_tokens = stream
    return calls


def test_pro_fixed_extra_branch_aggregates_all_actual_usage_and_sampling(pro_module):
    backend = loaded_backend()
    engine = engine_for(pro_module, backend)
    calls = install_recorded_stream(backend)
    answer, metadata = engine.solve("question", force_branch_count=3)
    assert answer == "answer 0"
    assert len(calls) == 4 and metadata["branch_count"] == 4
    assert [call[2] for call in calls] == pro_module.get_ladder_temperatures(3)
    assert all(call[1] == 128 and call[3] == 0.92 for call in calls)
    assert backend.last_branch_usage == {"prompt_tokens": 80, "completion_tokens": 10, "total_tokens": 90}
    assert backend.last_generation["usage"]["completion_tokens"] == 4


@pytest.mark.parametrize("cancel_at,with_event", [(0, False), (1, False), (1, True), (3, True)])
def test_cancellation_stops_pro_ladder_without_extra_branch_fallback_or_stale_usage(pro_module, cancel_at, with_event):
    backend = loaded_backend()
    engine = engine_for(pro_module, backend)
    event = threading.Event() if with_event else None
    backend.last_branch_usage = {"completion_tokens": 99999}
    calls = install_recorded_stream(backend, cancel_at=cancel_at, event=event)
    answer, metadata = engine.solve("question", force_branch_count=3, cancel_event=event)
    assert answer == "" and metadata["cancelled"] is True
    assert len(calls) == cancel_at + 1
    assert len(metadata["raw_branches"]) == cancel_at
    assert backend.last_branch_usage == {}
    engine._generate_fallback_branches.assert_not_called()


def test_preexisting_cancel_does_not_reuse_prior_generation_metadata(pro_module):
    backend = loaded_backend()
    engine = engine_for(pro_module, backend)
    backend.last_generation = {"status": "completed", "usage": {"completion_tokens": 99999}}
    event = threading.Event()
    event.set()
    calls = install_recorded_stream(backend)
    answer, metadata = engine.solve("question", cancel_event=event)
    assert answer == "" and metadata["cancelled"] is True
    assert calls == []
    assert backend.last_generation["status"] == "cancelled"
    assert backend.last_branch_usage == {}


def test_unload_during_search_retains_cancel_status_after_controller_is_detached(pro_module):
    backend = loaded_backend()
    engine = engine_for(pro_module, backend)
    calls = []

    def stream(*args):
        calls.append(args)
        assert engine.unload_model()["status"] == "unloaded"
        backend.last_generation = {"status": "cancelled"}
        return
        yield

    backend.stream_generate_tokens = stream
    answer, metadata = engine.solve("question", force_branch_count=3)
    assert answer == "" and metadata["cancelled"] is True
    assert len(calls) == 1
    assert engine.controller_backend is None
    assert backend.last_branch_usage == {}


@pytest.mark.parametrize("missing_at", [0, 3])
def test_missing_usage_in_either_routed_or_extra_pass_leaves_aggregate_unavailable(pro_module, missing_at):
    backend = loaded_backend()
    engine = engine_for(pro_module, backend)
    install_recorded_stream(backend, missing_usage_at=missing_at)
    assert len(engine.generate_parallel_branches("question", branch_count=3)) == 4
    assert backend.last_branch_usage == {}


def test_failure_in_extra_branch_does_not_publish_partial_query_usage(pro_module):
    backend = loaded_backend()
    engine = engine_for(pro_module, backend)
    install_recorded_stream(backend, fail_at=3)
    with pytest.raises(RuntimeError, match="native generation failed"):
        engine.generate_parallel_branches("question", branch_count=3)
    assert backend.last_branch_usage == {}


def test_unload_failure_retains_owner_for_memory_and_retry_and_blocks_replacement(pro_module, monkeypatch):
    backend = loaded_backend()
    engine = engine_for(pro_module, backend)
    manager = SimpleNamespace(stop=Mock(side_effect=RuntimeError("Owned process remains alive")), owned_pids=lambda: [12345])
    backend._manager = manager
    result = engine.unload_model()
    assert result["status"] == "error" and result["unloaded"] is False
    assert backend._manager is manager and backend.owned_process_ids() == [12345]
    assert engine.controller_backend is backend
    assert engine.active_model_name == "Strata Q2_0"
    assert engine.speed_status["status"] == "error"
    assert not backend.is_loaded and backend.model is None
    with pytest.raises(RuntimeError, match="Load the Strata model"):
        list(engine.stream_solve("must not fall back"))
    replacement = Mock(side_effect=AssertionError("Replacement loaded beside surviving owner"))
    monkeypatch.setattr(pro_module, "StrataReasoningBackend", replacement)
    result = engine.load_model("strata_gyro_s")
    assert result["status"] == "error" and result["requested_model"] == "strata_gyro_s"
    replacement.assert_not_called()
    manager.stop.side_effect = None
    assert engine.unload_model()["status"] == "unloaded"
    assert backend._manager is None and backend.owned_process_ids() == []
    assert engine.controller_backend is None and engine.active_model_name is None


def test_unload_closes_strata_transport_before_waiting_for_lifecycle_lock(pro_module):
    backend = loaded_backend()
    engine = engine_for(pro_module, backend)
    cancelled = threading.Event()
    backend.cancel_current_request = lambda: cancelled.set()
    entered, release = threading.Event(), threading.Event()

    def hold_model_operation():
        with engine._model_lock:
            entered.set()
            release.wait(3)

    holding = threading.Thread(target=hold_model_operation, daemon=True)
    holding.start()
    assert entered.wait(1)
    results = []
    unloading = threading.Thread(target=lambda: results.append(engine.unload_model()), daemon=True)
    unloading.start()
    try:
        assert cancelled.wait(1), "unload waited for model lock before closing HTTP"
    finally:
        release.set()
        holding.join(2)
        unloading.join(2)
    assert not unloading.is_alive()
    assert results[0]["status"] == "unloaded"


def test_strata_uses_its_cancellable_transport_instead_of_native_inference_lock():
    assert not StrataReasoningBackend.__dict__.get("_native_speed_lock_installed", False)
