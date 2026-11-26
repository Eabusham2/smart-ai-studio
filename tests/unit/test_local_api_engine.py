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
