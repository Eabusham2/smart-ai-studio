"""Exercise the resident-model bridge without downloading a checkpoint."""
import threading
from types import SimpleNamespace

import pytest

from core.generation_metrics import TokenMeter
from core.local_api import APIError, GenerationRequest, normalize_request
from core.local_api_engine import FormattedPrompt, LoadedModelAdapter


class Tokenizer:
    def __init__(self):
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        return "FORMATTED full conversation"

    def encode(self, text):
        return [1, 2, 3, 4]


class Backend:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.context_budget_tokens = 128
        self.calls = []
        self.closed = False

    def stream_generate_tokens(self, prompt, max_tokens, temperature, top_p, cancelled=None,
                               stop_at_tool_boundary=False):
        self.calls.append((prompt, max_tokens, temperature, top_p, stop_at_tool_boundary))
        meter = TokenMeter(self, "fixture", prompt_tokens=4)
        try:
            for i, token in enumerate(("Hello", " world", "!"), 1):
                if cancelled and cancelled():
                    return
                meter.observe(delta=1)
                yield token
        finally:
            self.closed = True
            meter.finish()


def adapter_fixture():
    backend = Backend()
    engine = SimpleNamespace(is_model_loaded=True, active_backend="mlx", mlx_backend=backend,
                             active_model_name="Loaded Bonsai", _model_lock=threading.Lock())
    selected = {"name": "Selected", "model_type": "text", "max_context": 256}
    adapter = LoadedModelAdapter(engine, selected_model=lambda: selected)
    return adapter, engine, backend


def request(**kwargs):
    return GenerationRequest("smart-ai-studio", [{"role": "user", "content": "hello"}], **kwargs)


def test_uses_loaded_model_and_exact_native_usage_without_desktop_solver():
    adapter, engine, backend = adapter_fixture()
    chunks = list(adapter.generate(request(max_tokens=9), threading.Event()))
    assert "".join(chunk.text for chunk in chunks) == "Hello world!"
    assert chunks[-1].prompt_tokens == 4
    assert chunks[-1].completion_tokens == 3
    assert chunks[-1].finish_reason == "stop"
    assert backend.calls[0][1:] == (9, 0.65, 0.92, False)
    assert isinstance(backend.calls[0][0], FormattedPrompt)
    assert backend.calls[0][0]._smartai_api_max_tokens == 9
    assert adapter.model_info()["name"] == "Loaded Bonsai"


def test_tool_history_and_schema_reach_real_template_without_mutating_input():
    adapter, _, backend = adapter_fixture()
    body = {"messages": [
        {"role": "developer", "content": "Use the client's tools."},
        {"role": "user", "content": "Read my file"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": '{"path":"a.py"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "hello"},
    ], "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}],
        "max_tokens": 3}
    selected = normalize_request(body, "chat", "smart-ai-studio")
    chunks = list(adapter.generate(selected, threading.Event()))
    messages, kwargs = backend.tokenizer.template_calls[0]
    assert messages[0]["role"] == "system"
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == {"path": "a.py"}
    assert messages[-1]["tool_call_id"] == "call_1"
    assert kwargs["tools"] == body["tools"]
    assert kwargs["add_generation_prompt"] and not kwargs["tokenize"]
    assert isinstance(selected.messages[2]["tool_calls"][0]["function"]["arguments"], str)
    assert chunks[-1].finish_reason == "length"
    assert backend.calls[0][-1] is False


def test_no_output_limit_uses_remaining_context_without_changing_settings():
    adapter, engine, backend = adapter_fixture()
    list(adapter.generate(request(), threading.Event()))
    assert backend.calls[0][1] == 124
    assert backend.context_budget_tokens == 128
    assert "settings" not in vars(engine)


def test_harness_system_messages_and_required_tool_choice_form_one_leading_turn():
    adapter, _, backend = adapter_fixture()
    selected = GenerationRequest("smart-ai-studio", [
        {"role": "system", "content": "System policy."},
        {"role": "developer", "content": "Use client tools."},
        {"role": "user", "content": "Read a.py"},
        {"role": "developer", "content": "Updated workspace instructions."},
    ], tool_choice={"type": "function", "function": {"name": "read"}})
    original = [dict(message) for message in selected.messages]
    list(adapter.generate(selected, threading.Event()))
    messages, _ = backend.tokenizer.template_calls[0]
    assert messages == [
        {"role": "system", "content": 'System policy.\n\nUse client tools.\n\nUpdated workspace instructions.\n\nRespond by calling the supplied tool "read".'},
        {"role": "user", "content": "Read a.py"},
    ]
    assert selected.messages == original


def test_overflow_errors_before_generation_instead_of_truncating_history():
    adapter, _, backend = adapter_fixture()
    with pytest.raises(APIError, match="exceeds") as error:
        list(adapter.generate(request(max_tokens=125), threading.Event()))
    assert error.value.status == 400
    assert backend.calls == []
    assert adapter.count_tokens(request()) == 4


def test_lease_covers_generator_lifetime_and_releases_on_close():
    adapter, engine, backend = adapter_fixture()
    stream = adapter.generate(request(), threading.Event())
    next(stream)
    next(stream)
    assert engine._model_lock.locked()
    with pytest.raises(APIError, match="another request"):
        list(adapter.generate(request(), threading.Event()))
    stream.close()
    assert backend.closed
    assert not engine._model_lock.locked()
    assert list(adapter.generate(request(), threading.Event()))


def test_missing_busy_and_cancelled_model_are_explicit():
    adapter, engine, backend = adapter_fixture()
    engine.is_model_loaded = False
    assert adapter.model_info() is None
    with pytest.raises(APIError, match="Load a text"):
        list(adapter.generate(request(), threading.Event()))
    engine.is_model_loaded = True
    adapter.busy = lambda: "Learning is active"
    with pytest.raises(APIError, match="Learning is active"):
        list(adapter.generate(request(), threading.Event()))
    adapter.busy = lambda: ""
    event = threading.Event()
    stream = adapter.generate(request(), event)
    next(stream)
    next(stream)
    event.set()
    list(stream)
    assert backend.closed
    assert not engine._model_lock.locked()


def test_native_server_owns_template_and_never_gets_a_guessed_chatml_prompt():
    adapter, engine, backend = adapter_fixture()
    backend.tokenizer.apply_chat_template = None
    captured = []
    backend._request = lambda path, payload: captured.append((path, payload)) or {"prompt": "native format"}
    engine.active_backend = "gguf"
    engine.gguf_backend = backend
    list(adapter.generate(request(), threading.Event()))
    assert captured == [("/apply-template", {"messages": [{"role": "user", "content": "hello"}],
                                           "add_generation_prompt": True})]
    assert backend.calls[0][0] == "native format"
    backend._request = None
    with pytest.raises(APIError, match="chat template"):
        list(adapter.generate(request(), threading.Event()))


def test_api_cap_is_separate_from_desktop_unlimited_generation_policy():
    from core.unified_context_budget import install
    class Model:
        _encode_prompt = staticmethod(lambda prompt: [1, 2, 3, 4])
        tokenizer = SimpleNamespace(encode=_encode_prompt)
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=128))
        context_budget_tokens = 128
    install(Model)
    backend = Model()
    assert backend._effective_generation_cap("desktop", 3) == 124
    assert backend._effective_generation_cap(FormattedPrompt("api", 3), 3) == 3
    assert backend._effective_generation_cap(FormattedPrompt("api"), 3) == 124


def test_controller_preserves_preformatted_chat_prompt():
    from core.controller_runtime import UniversalControllerBackend
    prompt = FormattedPrompt("raw full history", 9)
    backend = UniversalControllerBackend.__new__(UniversalControllerBackend)
    backend._apply_chat_template = lambda *a, **k: pytest.fail("Double chat template")
    assert backend._mlx_prompt(prompt) is prompt


def test_harness_requests_do_not_receive_desktop_history_tools_or_settings():
    adapter, engine, backend = adapter_fixture()
    engine.chat_history = [{"role": "system", "content": "PRIVATE DESKTOP HISTORY"}]
    engine.tools = SimpleNamespace(list_available_tools=lambda: pytest.fail("Desktop tool registry accessed"))
    engine.solve = lambda *a, **k: pytest.fail("Desktop solver called")
    engine.settings = SimpleNamespace(search_temperature=1.99, max_new_tokens=1)
    for content in ("Harness A", "Harness B"):
        req = GenerationRequest("smart-ai-studio", [{"role": "user", "content": content}])
        list(adapter.generate(req, threading.Event()))
    assert backend.tokenizer.template_calls[0][0] == [{"role": "user", "content": "Harness A"}]
    assert backend.tokenizer.template_calls[1][0] == [{"role": "user", "content": "Harness B"}]
    assert all("tools" not in kw for _, kw in backend.tokenizer.template_calls)
    assert all(call[1:4] == (124, .65, .92) for call in backend.calls)


def test_request_context_is_optional_and_does_not_change_loaded_capacity():
    adapter, engine, backend = adapter_fixture()
    list(adapter.generate(request(context_limit=32), threading.Event()))
    assert backend.calls[-1][1] == 28
    assert backend.calls[-1][0]._smartai_api_max_tokens == 28
    assert backend.context_budget_tokens == 128
    with pytest.raises(APIError, match="exceeds"):
        list(adapter.generate(request(context_limit=129), threading.Event()))
    with pytest.raises(APIError, match="Sampling option unavailable"):
        list(adapter.generate(request(sampling={"top_k": 20}), threading.Event()))
    with pytest.raises(APIError, match="Speedup requires"):
        list(adapter.generate(request(speedups={"tokenizer_memo": False}), threading.Event()))


def test_native_sampling_and_template_controls_are_forwarded_without_mutation():
    adapter, engine, backend = adapter_fixture()
    engine.active_backend = "gguf"
    engine.gguf_backend = backend
    backend.runtime = "prism_speculative"
    backend.tokenizer.apply_chat_template = None
    backend.tokenizer.chat_template = "{% if enable_thinking %}"
    captured = []
    backend._request = lambda path, payload: captured.append((path, payload)) or {"prompt": "native format"}
    backend.sampling_defaults = {"top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0}
    kwargs_seen = []
    def stream(**kwargs):
        kwargs_seen.append(kwargs)
        yield "answer"
    backend.stream_generate_tokens = stream
    list(adapter.generate(request(max_tokens=9, sampling={"top_k": 10, "seed": 42},
                                  enable_thinking=False), threading.Event()))
    assert captured[0][1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert kwargs_seen[0]["top_k"] == 10 and kwargs_seen[0]["seed"] == 42
    assert backend.sampling_defaults["top_k"] == 20
    capabilities = adapter.model_info()["capabilities"]
    assert capabilities["context_limit"] == 128
    assert "top_k" in capabilities["sampling_options"] and "seed" in capabilities["sampling_options"]
    assert capabilities["request_speedups"] == []
    assert not capabilities["active_speedups"]["persistent_cache"]


def test_mlx_api_speedup_scope_is_restored_on_close_and_desktop_is_unchanged():
    from core.mlx_engine import MLXReasoningBackend
    from core.mlx_warmup import SafeMLXWarmup
    method = MLXReasoningBackend.stream_generate_tokens
    # Unwrap only the existing owner/lock decorators to exercise the real
    # request-scoped settings code with a deterministic decoder.
    import inspect
    method = inspect.unwrap(method)
    head = object()
    saved = {"prompt_reuse": True, "tokenizer_memo": True, "branch_prefill": True,
             "persistent_cache": True, "metal_warmup": False}
    warmup = SafeMLXWarmup()
    fake = SimpleNamespace(_lossless_speedups=saved, _metal_warmup=warmup,
                           _mtp=SimpleNamespace(head=head, max_draft_tokens=2))
    observations = []
    def decode(prompt, maximum, temperature, top_p, **kwargs):
        observations.append((dict(fake._lossless_speedups), fake._mtp.head, kwargs))
        yield "a"
        yield "b"
    fake._stream_generate_tokens_impl = decode
    stream = method(fake, FormattedPrompt("api", 4), 4, api_sampling={"top_k": 10},
                    api_speedups={"enabled": False})
    assert next(stream) == "a"
    assert observations[-1] == ({key: False for key in saved}, None, {"api_sampling": {"top_k": 10}})
    stream.close()
    assert fake._lossless_speedups is saved and fake._mtp.head is head
    assert fake._metal_warmup is warmup and fake._mtp.max_draft_tokens == 2
    list(method(fake, FormattedPrompt("desktop", api_request=False), 4))
    assert observations[-1] == (saved, head, {})
    list(method(fake, FormattedPrompt("api", 4), 4))
    assert observations[-1][0]["prompt_reuse"] is True
    assert observations[-1][0]["persistent_cache"] is False
    assert fake._lossless_speedups is saved


def test_api_uses_native_preprocessed_count_instead_of_plain_tokenizer_estimate():
    adapter, _, backend = adapter_fixture()
    backend.api_prompt_count = lambda prompt, speedups: 12
    list(adapter.generate(request(context_limit=32), threading.Event()))
    assert backend.calls[-1][1] == 20
    assert adapter.count_tokens(request()) == 12
    with pytest.raises(APIError, match="exceeds"):
        list(adapter.generate(request(context_limit=32, max_tokens=21), threading.Event()))
    backend.api_prompt_count = lambda prompt, speedups: None
    with pytest.raises(APIError, match="Cannot verify"):
        list(adapter.generate(request(max_tokens=1), threading.Event()))


def test_optional_reasoning_effort_is_boolean_and_explicit_local_setting_wins():
    base = {"messages": [{"role": "user", "content": "hello"}]}
    assert normalize_request({**base, "reasoning_effort": "high"}, "chat", "smart-ai-studio").enable_thinking is True
    assert normalize_request({"input": "hello", "reasoning": {"effort": "none"}}, "responses", "smart-ai-studio").enable_thinking is False
    assert normalize_request({**base, "reasoning_effort": "high", "smartai": {"enable_thinking": False}},
                             "chat", "smart-ai-studio").enable_thinking is False
    assert normalize_request({**base, "enable_thinking": True}, "chat", "smart-ai-studio").enable_thinking is True
    with pytest.raises(APIError, match="enable_thinking"):
        normalize_request({**base, "enable_thinking": 1}, "chat", "smart-ai-studio")
    with pytest.raises(APIError, match="Signed Anthropic"):
        normalize_request({**base, "thinking": {"type": "enabled", "budget_tokens": 2048}},
                          "anthropic", "smart-ai-studio")


def test_optional_learning_mode_validation_and_loaded_model_sampling():
    adapter, engine, backend = adapter_fixture()
    adapter.selected_model = lambda: {"model_type": "text", "bonsai25_mtp": True, "max_context": 256}
    backend.tokenizer.apply_chat_template = lambda *a, **k: "assistant<think>\n"
    list(adapter.generate(request(max_tokens=12), threading.Event()))
    assert backend.calls[-1][2:4] == (1.0, .95)
    list(adapter.generate(request(max_tokens=12, temperature=.2, top_p=.6, learning_mode="off"), threading.Event()))
    assert backend.calls[-1][2:4] == (.2, .6)
    backend.tokenizer.apply_chat_template = lambda *a, **k: "assistant<think></think>\n"
    list(adapter.generate(request(max_tokens=12), threading.Event()))
    assert backend.calls[-1][2:4] == (.7, .8)
    parsed = normalize_request({"messages": [{"role": "user", "content": "hi"}],
                                "smartai": {"learning_mode": "managed"}}, "chat", "smart-ai-studio")
    assert parsed.temperature is None and parsed.top_p is None and parsed.learning_mode == "managed"
    with pytest.raises(APIError, match="learning_mode"):
        normalize_request({"messages": [{"role": "user", "content": "hi"}],
                           "smartai": {"learning_mode": "invented"}}, "chat", "smart-ai-studio")


@pytest.mark.parametrize("mode", ["managed", "hybrid", "legacy"])
def test_api_uses_real_learning_service_without_desktop_context(mode):
    adapter, engine, backend = adapter_fixture()
    trained = []
    def train(pairs, **kwargs):
        trained.append(pairs)
        return {"status": "success", "parameter_drift_l2": .25, "trainable_parameters_touched": 42}
    def legacy(turns, **kwargs):
        assert service.db is None
        trained.append([{"prompt": turns[0]["content"], "completion": turns[1]["content"]}])
        return True
    service = SimpleNamespace(engine=backend, learning_implementation=mode,
                              _real_training_ready=lambda: True, train_pairs=train,
                              consolidate_chunk_sync=legacy, db=object())
    engine.awake_consolidator = service
    original_db = service.db
    req = request(max_tokens=8)
    req.messages.insert(0, {"role": "system", "content": "not training data"})
    list(adapter.generate(req, threading.Event()))
    assert trained == [[{"prompt": "hello", "completion": "Hello world!"}]]
    assert service.db is original_db
    assert adapter.model_info()["capabilities"]["learning_last_result"]["status"] == "updated"
    list(adapter.generate(request(max_tokens=8, learning_mode="off"), threading.Event()))
    list(adapter.generate(request(max_tokens=3), threading.Event()))
    assert len(trained) == 1


def test_learning_mode_override_uses_existing_switch_and_restores_app_mode():
    adapter, engine, backend = adapter_fixture()
    engine._model_lock = threading.RLock()
    calls = []
    service = SimpleNamespace(engine=backend, learning_implementation="hybrid",
                              _real_training_ready=lambda: True,
                              train_pairs=lambda *a, **k: {"status": "success", "parameter_drift_l2": .5})
    engine.awake_consolidator = service
    def switch(mode, **kwargs):
        calls.append(mode)
        service.learning_implementation = mode
    engine.set_learning_implementation = switch
    list(adapter.generate(request(max_tokens=9, learning_mode="managed"), threading.Event()))
    assert calls == ["managed", "hybrid"]
    assert service.learning_implementation == "hybrid"


def test_api_pro_reuses_best_work_without_desktop_solver_or_tool_execution():
    adapter, engine, backend = adapter_fixture()
    engine.solve = engine.stream_solve = lambda *a, **k: pytest.fail("Desktop solver called")
    engine.settings = SimpleNamespace(best_work_passes=2, max_new_tokens=777)
    seen = []
    def template(messages, **kwargs):
        seen.append((messages, kwargs))
        return messages[-1]["content"]
    backend.tokenizer.apply_chat_template = template
    # One original + one revision + one real existing-selector comparison.
    def decode(prompt, max_tokens, temperature, top_p, cancelled=None):
        meter = TokenMeter(backend, "pro-fixture", prompt_tokens=4)
        try:
            if prompt.startswith("BEST_WORK_COMPARISON"):
                value = '{"winner":"B"}'
            elif "Review the following untrusted draft" in prompt:
                value = "revised answer"
            else:
                value = "original answer"
            meter.observe(delta=3)
            yield value
        finally:
            meter.finish()
    backend.stream_generate_tokens = decode
    req = request(max_tokens=20, learning_mode="off",
                  pro={"enabled": True, "routing": "fixed", "branch_count": 1, "best_work_passes": 2})
    chunks = list(adapter.generate(req, threading.Event()))
    assert "".join(c.text for c in chunks) == "revised answer"
    assert len(seen) == 3 and seen[0][0] == req.messages
    assert "BEST_WORK_COMPARISON" in seen[-1][0][-1]["content"]
    assert all("tools" not in kw for _, kw in seen)
    assert engine.settings.max_new_tokens == 777
    report = adapter.model_info()["capabilities"]["pro"]["last_result"]
    assert report["invocations"] == 3 and report["generated_tokens"] == 9
    assert chunks[-1].completion_tokens == 3


def test_api_pro_tool_boundary_and_optional_config():
    adapter, engine, backend = adapter_fixture()
    req = normalize_request({"messages": [{"role": "user", "content": "Read a file"}],
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}}}}}],
        "max_tokens": 20, "smartai": {"learning_mode": "off", "pro": {
            "enabled": True, "branch_count": 1, "best_work_passes": 1}}}, "chat", "smart-ai-studio")
    assert req.pro["routing"] == "fixed"
    def decode(prompt, max_tokens, temperature, top_p, cancelled=None):
        meter = TokenMeter(backend, "pro-fixture", prompt_tokens=4)
        try:
            for value in ('<tool_call><function=read><parameter=path>a.py</parameter>',
                          '</function></tool_call>invented result'):
                meter.observe(delta=2)
                yield value
            pytest.fail("Continued decoding after the first completed tool call")
        finally:
            meter.finish()
    backend.stream_generate_tokens = decode
    chunks = list(adapter.generate(req, threading.Event()))
    assert "".join(c.text for c in chunks) == '<tool_call>{"name":"read","arguments":{"path":"a.py"}}</tool_call>'
    assert adapter.model_info()["capabilities"]["pro"]["last_result"]["invocations"] == 1
    for invalid in ({"enabled": 1}, {"routing": "auto", "branch_count": 2}, {"best_work_passes": 0}):
        with pytest.raises(APIError):
            normalize_request({"messages": [{"role": "user", "content": "hi"}],
                               "smartai": {"pro": invalid}}, "chat", "smart-ai-studio")


@pytest.mark.parametrize(("routing", "tensorfold"), [("auto", False), ("fixed", False), ("auto", True)])
def test_api_pro_routes_multiple_branches_and_selects_only_the_tournament_winner(routing, tensorfold):
    adapter, engine, backend = adapter_fixture()
    engine.settings = SimpleNamespace(best_work_passes=1, max_new_tokens=777)
    engine.router = SimpleNamespace(route=lambda entropy, **kw: ("test router", 2))
    backend.calculate_token_entropy = lambda prompt: .5
    if tensorfold:
        engine.active_backend = "controller"
        engine.controller_backend = backend
        backend.runtime = "tensorfold_mlx"
        backend.supports_entropy = False
        backend.calculate_token_entropy = lambda prompt: pytest.fail("TensorFold has no full-vocabulary logits")
        engine.lossless_speed_options = {"native_branch_count": 2}
    elif routing == "auto":
        # Existing external controllers may define their own capability attributes.
        engine.active_backend = "controller"
        engine.controller_backend = backend
        backend.runtime = "custom"
        backend.supports_entropy = backend.supports_learning_modes = False
        assert adapter.model_info()["capabilities"]["learning_modes"] == ["legacy", "managed", "hybrid", "off"]
    engine.solve = engine.stream_solve = lambda *a, **k: pytest.fail("Desktop solver called")
    backend.tokenizer.apply_chat_template = lambda messages, **kwargs: messages[-1]["content"]
    originals = []
    def decode(prompt, max_tokens, temperature, top_p, cancelled=None):
        meter = TokenMeter(backend, "pro-fixture", prompt_tokens=4)
        try:
            if prompt.startswith("BEST_WORK_COMPARISON"):
                value = '{"winner":"B"}'
            elif prompt.startswith("BEST_WORK_SYNTHESIS"):
                value = "synthesized answer"
            else:
                originals.append(temperature)
                value = f"original candidate {len(originals)}"
            meter.observe(delta=3)
            yield value
        finally:
            meter.finish()
    backend.stream_generate_tokens = decode
    pro = {"enabled": True, "routing": routing, "best_work_passes": 1}
    if routing == "fixed":
        pro["branch_count"] = 2
    chunks = list(adapter.generate(request(max_tokens=20, learning_mode="off", pro=pro), threading.Event()))
    assert "".join(chunk.text for chunk in chunks) == "synthesized answer"
    assert originals == [.2, .95, .65]
    report = adapter.model_info()["capabilities"]["pro"]["last_result"]
    if tensorfold:
        assert report["mode"] == "native_configured" and report["entropy"] is None
    elif routing == "auto":
        assert report["entropy"] == 0.5
    assert report["routed_branches"] == 2 and report["original_candidates"] == 3
    assert report["winning_branch"] == 3 and report["invocations"] == 8
    assert report["generated_tokens"] == 24 and chunks[-1].completion_tokens == 3


def test_api_pro_is_off_by_default_even_when_desktop_pro_is_enabled():
    adapter, engine, backend = adapter_fixture()
    # Simulate legacy server-start defaults captured with the desktop checkbox on.
    adapter.pro_defaults = {"enabled": True, "best_work_passes": 5}
    for pro in ({}, {"best_work_passes": 3}, {"enabled": False}):
        chunks = list(adapter.generate(request(max_tokens=8, learning_mode="off", pro=pro),
                                       threading.Event()))
        assert "".join(chunk.text for chunk in chunks) == "Hello world!"
        assert adapter.model_info()["capabilities"]["pro"]["last_result"]["status"] == "disabled"
    assert len(backend.calls) == 3
    defaults = adapter.model_info()["capabilities"]["pro"]["defaults"]
    assert defaults["enabled"] is False and defaults["best_work_passes"] == 5
    assert adapter._pro_config({"enabled": True})["enabled"] is True


def test_inference_only_tensorfold_preserves_client_sampling_and_cannot_reload_for_training():
    from unittest.mock import Mock

    adapter, engine, backend = adapter_fixture()
    engine.active_backend = "controller"
    engine.controller_backend = backend
    backend.runtime = "tensorfold_mlx"
    backend.supports_learning_modes = False
    backend.api_prompt_count = Mock(return_value=4)
    engine.awake_consolidator = SimpleNamespace(engine=backend, learning_implementation="legacy",
                                                _real_training_ready=Mock(return_value=True))
    engine.set_learning_implementation = Mock(side_effect=AssertionError("Unexpected model reload"))

    capabilities = adapter.model_info()["capabilities"]
    assert capabilities["sampling_options"] == ["temperature", "top_p"]
    assert capabilities["learning_modes"] == ["off"]
    assert capabilities["learning_mode_default"] == "off"
    assert capabilities["learning_ready"] is False
    assert list(adapter.generate(request(max_tokens=3, temperature=0.2, top_p=0.8),
                                 threading.Event()))[-1].completion_tokens == 3
    assert backend.calls[-1][1:] == (3, 0.2, 0.8, False)
    backend.api_prompt_count.assert_called_once()
    for mode in ("legacy", "managed", "hybrid"):
        with pytest.raises(APIError, match="inference-only"):
            list(adapter.generate(request(learning_mode=mode), threading.Event()))
    assert len(backend.calls) == 1
    engine.set_learning_implementation.assert_not_called()
    engine.awake_consolidator._real_training_ready.assert_not_called()


@pytest.mark.parametrize("use_pro", [False, True])
@pytest.mark.parametrize(("limit", "native_reason"), [(3, "stop"), (9, "length")])
def test_tensorfold_terminal_reason_reaches_api_even_when_token_heuristic_differs(use_pro, limit, native_reason):
    adapter, engine, backend = adapter_fixture()
    engine.active_backend = "controller"
    engine.controller_backend = backend
    backend.runtime = "tensorfold_mlx"
    backend.supports_entropy = False
    original = backend.stream_generate_tokens

    def decode(prompt, max_tokens, temperature, top_p, cancelled=None):
        yield from original(prompt, max_tokens, temperature, top_p, cancelled)
        backend.last_generation = {"status": "completed", "stop_type": native_reason}

    backend.stream_generate_tokens = decode
    pro = {"enabled": True, "best_work_passes": 1} if use_pro else {}
    result = list(adapter.generate(request(max_tokens=limit, learning_mode="off", pro=pro), threading.Event()))
    assert result[-1].completion_tokens == 3
    assert result[-1].finish_reason == native_reason


@pytest.mark.parametrize("options", [
    None, "14.5", 14.5, False, [], {"enabled": 0}, {"enabled": "off"}, {"extra": True},
    {"limit_gb": False}, {"limit_gb": 0.49}, {"limit_gb": "14.5"}, {"limit_gb": None},
    {"limit_gb": float("nan")}, {"limit_gb": float("inf")}, {"limit_gb": 10**500},
])
def test_invalid_memory_limit_fails_before_generation(options):
    with pytest.raises(APIError, match="memory_limit") as error:
        normalize_request({"messages": [{"role": "user", "content": "hi"}],
                           "smartai": {"memory_limit": options}}, "chat", "model")
    assert error.value.status == 400


def test_api_memory_defaults_overrides_and_count_are_separate_from_desktop(monkeypatch):
    from unittest.mock import Mock
    from core.memory_watchdog import SystemMemoryWatchdog
    measure = Mock(return_value={"process_memory_known": True, "owned_memory_complete": True, "total_allocated_bytes": 1024**3})
    monkeypatch.setattr(SystemMemoryWatchdog, "get_detailed_memory_breakdown", measure)
    adapter, engine, backend = adapter_fixture()
    engine.speed_memory_limit_gb, backend.speed_memory_limit_gb = 12, 10
    seen = []
    backend.owned_process_ids = lambda: [413]
    backend.prepare_request = lambda: seen.append((engine.speed_memory_limit_gb, backend.speed_memory_limit_gb))
    caps = adapter.model_info()["capabilities"]
    assert caps["memory_limit"] == {"supported": True, "defaults": {"enabled": False, "limit_gb": 14.5}}
    assert "memory_limit" not in caps["load_time_only"]
    for options in ({}, {"limit_gb": 6}):
        list(adapter.generate(request(memory_limit=options), threading.Event()))
        assert seen[-1] == (None, None)
    measure.assert_not_called()
    assert adapter.count_tokens(request(memory_limit={"enabled": True}, learning_mode="managed")) == 4
    assert len(seen) == 3 and seen[-1] == (12, 10) and not measure.called
    list(adapter.generate(request(memory_limit={"enabled": True}), threading.Event()))
    assert seen[-1] == (14.5, 14.5)
    measure.assert_called_with(owned_pids=[413])
    enabled = LoadedModelAdapter(engine, memory_limit_defaults={"enabled": True, "limit_gb": 9},
                                 owned_processes_provider=lambda: [417, 418])
    for options, expected in (({}, 9), ({"limit_gb": 3}, 3), ({"enabled": False}, None), ({}, 9)):
        list(enabled.generate(request(memory_limit=options), threading.Event()))
        assert seen[-1] == (expected, expected)
        assert (engine.speed_memory_limit_gb, backend.speed_memory_limit_gb) == (12, 10)
    measure.assert_called_with(owned_pids=[417, 418])


@pytest.mark.parametrize("known,usage,code", [(True, 17.0, "memory_limit_exceeded"),
                                            (False, 0.0, "memory_measurement_unavailable")])
def test_memory_preflight_rejects_before_inference_or_learning_switch(monkeypatch, known, usage, code):
    from core.memory_watchdog import SystemMemoryWatchdog
    monkeypatch.setattr(SystemMemoryWatchdog, "get_detailed_memory_breakdown", lambda **_: {
        "process_memory_known": known, "owned_memory_complete": True, "total_allocated_bytes": usage * 1024**3})
    adapter, engine, backend = adapter_fixture()
    engine.set_learning_implementation = lambda *a, **kw: pytest.fail("unexpected reload")
    for pro in ({}, {"enabled": True}):
        with pytest.raises(APIError) as error:
            list(adapter.generate(request(memory_limit={"enabled": True}, pro=pro, learning_mode="managed"),
                                  threading.Event()))
        assert error.value.status == 503 and error.value.code == code
        assert not backend.calls and not engine._model_lock.locked()
        assert not hasattr(engine, "speed_memory_limit_gb") and not hasattr(backend, "speed_memory_limit_gb")


def test_memory_cap_restores_after_generator_close_and_reloaded_learning_mode(monkeypatch):
    from core.memory_watchdog import SystemMemoryWatchdog
    monkeypatch.setattr(SystemMemoryWatchdog, "get_detailed_memory_breakdown", lambda **_: {
        "process_memory_known": True, "owned_memory_complete": True, "total_allocated_bytes": 1024**3})
    adapter, engine, original = adapter_fixture()
    engine._model_lock = threading.RLock()
    engine.speed_memory_limit_gb, original.speed_memory_limit_gb = 5, 4
    service = SimpleNamespace(engine=original, learning_implementation="hybrid", _real_training_ready=lambda: True)
    engine.awake_consolidator = service
    switched = []
    def switch(mode, **kwargs):
        switched.append((mode, engine.speed_memory_limit_gb))
        model = Backend()
        model.speed_memory_limit_gb = engine.speed_memory_limit_gb
        service.learning_implementation, service.engine, engine.mlx_backend = mode, model, model
    engine.set_learning_implementation = switch
    stream = adapter.generate(request(memory_limit={"enabled": True, "limit_gb": 3}, learning_mode="managed"),
                              threading.Event())
    next(stream)
    temporary = engine.mlx_backend
    assert temporary.speed_memory_limit_gb == 3
    stream.close()
    assert switched == [("managed", 3), ("hybrid", 5)]
    assert engine.mlx_backend is not original and engine.mlx_backend is not temporary
    assert (original.speed_memory_limit_gb, engine.mlx_backend.speed_memory_limit_gb) == (4, 4)
    assert engine.speed_memory_limit_gb == 5


@pytest.mark.parametrize("pro", [{}, {"enabled": True, "routing": "fixed", "branch_count": 1,
                                     "best_work_passes": 1}])
def test_api_memory_monitor_cancels_blocked_decoder_and_never_trains(monkeypatch, pro):
    from core import local_api_memory
    adapter, engine, backend = adapter_fixture()
    pressure, interrupted = threading.Event(), threading.Event()
    backend.cancel_current_request = interrupted.set
    def decode(prompt, max_tokens, temperature, top_p, cancelled=None):
        pressure.set()
        assert interrupted.wait(2) and cancelled()
        yield "should never reach a client"
    backend.stream_generate_tokens = decode
    def measure(**kwargs):
        return {"process_memory_known": True, "owned_memory_complete": True,
                "total_allocated_bytes": (20 if pressure.is_set() else 1) * 1024**3}
    monkeypatch.setattr(local_api_memory, "_SAMPLE_INTERVAL", 0.01)
    monkeypatch.setattr(local_api_memory.SystemMemoryWatchdog, "get_detailed_memory_breakdown", measure)
    engine.awake_consolidator = SimpleNamespace(engine=backend, learning_implementation="managed",
        _real_training_ready=lambda: True, train_pairs=lambda *a, **k: pytest.fail("trained an interrupted response"))
    client = threading.Event()
    chunks = []
    with pytest.raises(APIError) as error:
        chunks.extend(adapter.generate(request(memory_limit={"enabled": True}, pro=pro), client))
    assert not any(chunk.text for chunk in chunks)
    assert error.value.code == "memory_limit_exceeded" and not client.is_set()
    assert not hasattr(engine, "speed_memory_limit_gb") and not hasattr(backend, "speed_memory_limit_gb")
    assert not engine._model_lock.locked()


def test_fast_prefill_exceeding_cap_never_emits_a_tool_call(monkeypatch):
    from core import local_api_memory
    adapter, engine, backend = adapter_fixture()
    pressure = threading.Event()
    def decode(**kwargs):
        pressure.set()
        try:
            yield '<tool_call>{"name":"read","arguments":{}}</tool_call>'
        finally:
            backend.closed = True
    backend.stream_generate_tokens = decode
    monkeypatch.setattr(local_api_memory, "_SAMPLE_INTERVAL", 15)
    monkeypatch.setattr(local_api_memory.SystemMemoryWatchdog, "get_detailed_memory_breakdown",
                        lambda **_: {"process_memory_known": True, "owned_memory_complete": True,
                                     "total_allocated_bytes": (18 if pressure.is_set() else 1) * 1024**3})
    chunks = []
    with pytest.raises(APIError) as error:
        chunks.extend(adapter.generate(request(memory_limit={"enabled": True}), threading.Event()))
    assert error.value.code == "memory_limit_exceeded"
    assert not any(c.text for c in chunks) and backend.closed


def test_api_memory_cancellation_reaches_in_progress_learning_and_releases_lock(monkeypatch):
    from core import local_api_memory
    adapter, engine, backend = adapter_fixture()
    learning = threading.Event()
    def train(pairs, *, mode, cancel_event):
        learning.set()
        assert cancel_event.wait(2) and cancel_event.is_set()
        return {"status": "cancelled"}
    engine.awake_consolidator = SimpleNamespace(engine=backend, learning_implementation="managed",
        _real_training_ready=lambda: True, train_pairs=train)
    monkeypatch.setattr(local_api_memory, "_SAMPLE_INTERVAL", 0.01)
    monkeypatch.setattr(local_api_memory.SystemMemoryWatchdog, "get_detailed_memory_breakdown",
                        lambda **_: {"process_memory_known": True, "owned_memory_complete": True,
                                     "total_allocated_bytes": (17 if learning.is_set() else 1) * 1024**3})
    event = threading.Event()
    with pytest.raises(APIError) as error:
        list(adapter.generate(request(memory_limit={"enabled": True}), event))
    assert error.value.code == "memory_limit_exceeded" and learning.is_set()
    assert not event.is_set() and not engine._model_lock.locked()
    assert adapter.model_info()["capabilities"]["learning_last_result"]["status"] == "not_committed"


@pytest.mark.parametrize("exited", [False, True])
def test_memory_limit_distinguishes_exited_and_unreadable_owned_workers(monkeypatch, exited):
    import psutil
    from core import memory_watchdog
    adapter, _, backend = adapter_fixture()
    backend.owned_process_ids = lambda: [74291]
    def process(pid=None):
        if pid == 74291:
            raise (psutil.NoSuchProcess(pid) if exited else psutil.AccessDenied(pid))
        return SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=1024**3))
    monkeypatch.setattr(psutil, "Process", process)
    monkeypatch.setattr(memory_watchdog.platform, "system", lambda: "Linux")
    if exited:
        assert list(adapter.generate(request(memory_limit={"enabled": True}), threading.Event()))[-1].finish_reason == "stop"
    else:
        with pytest.raises(APIError) as error:
            list(adapter.generate(request(memory_limit={"enabled": True}), threading.Event()))
        assert error.value.code == "memory_measurement_unavailable"
        assert not backend.calls


def test_monitor_start_failure_restores_original_limits_and_releases_lease(monkeypatch):
    from core import local_api_memory
    adapter, engine, backend = adapter_fixture()
    engine.speed_memory_limit_gb, backend.speed_memory_limit_gb = 9, 8
    monkeypatch.setattr(local_api_memory.SystemMemoryWatchdog, "get_detailed_memory_breakdown",
                        lambda **_: {"process_memory_known": True, "owned_memory_complete": True,
                                     "total_allocated_bytes": 1024**3})
    def fail_start(self):
        raise RuntimeError("cannot start new thread")
    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="cannot start new thread"):
        list(adapter.generate(request(memory_limit={"enabled": True, "limit_gb": 3}), threading.Event()))
    assert (engine.speed_memory_limit_gb, backend.speed_memory_limit_gb) == (9, 8)
    assert not engine._model_lock.locked()
