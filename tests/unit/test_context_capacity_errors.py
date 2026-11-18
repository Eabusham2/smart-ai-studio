"""Only confirmed pre-generation capacity failures authorize recovery of a prior draft."""
from contextlib import closing
import json
from types import SimpleNamespace
from threading import Event

import pytest

from core.awake_auto_hook import _remaining_generation_tokens, install_awake_auto_learning
from core.context_memory import ContextOverflow
from core.engines.strata_engine import StrataError, StrataReasoningBackend
from core.unified_context_budget import install as install_mlx_budget
from tests.unit.test_native_speculation import http_backend
from tests.unit.test_strata_engine import KEY, MODEL, LocalService, chunk, event


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("with_history", [False, True])
def test_physical_window_checked_before_entropy_or_decoder(stream, with_history):
    class Engine:
        active_backend = "controller"
        controller_backend = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(n_positions=64)),
                                             tokenizer=SimpleNamespace(encode=list))
        _context_budget_tokens = 8192
        _best_work_route_with_history = True

        def _format_prompt_with_history(self, prompt, history):
            return "".join(turn["content"] for turn in history) + prompt

        def calculate_token_entropy(self, prompt):
            pytest.fail("this would run the model on the oversized refinement")

        def stream_solve(self, prompt, **kwargs):
            pytest.fail("unexpected decoder invocation")

        def solve(self, prompt, **kwargs):
            pytest.fail("unexpected branch invocation")

    install_awake_auto_learning(Engine)
    engine = Engine()
    history = [{"role": "system", "content": "s" * 32}] if with_history else []
    prompt = "x" * (32 if with_history else 64)
    with pytest.raises(ContextOverflow, match="64 tokens"):
        if stream:
            list(engine.stream_solve(prompt, history=history))
        else:
            engine.solve(prompt, history=history)


def test_estimate_or_tokenizer_allocation_failure_cannot_authorize_recovery():
    backend = SimpleNamespace(model=object(), tokenizer=None)
    engine = SimpleNamespace(active_backend="controller", controller_backend=backend, _context_budget_tokens=16)
    with pytest.raises(RuntimeError, match="exact token count is unavailable") as failure:
        _remaining_generation_tokens(engine, "x" * 80)
    assert not isinstance(failure.value, ContextOverflow)

    def allocation_failed(_text):
        raise MemoryError("tokenizer allocation failed")

    backend.tokenizer = SimpleNamespace(encode=allocation_failed)
    with pytest.raises(MemoryError):
        _remaining_generation_tokens(engine, "x" * 80)


def test_mlx_capacity_uses_tokenized_prompt_and_reserves_every_remaining_token():
    class Backend:
        model = SimpleNamespace(args=SimpleNamespace(max_seq_len=16))
        tokenizer = SimpleNamespace(encode=list)
        context_budget_tokens = 256

    install_mlx_budget(Backend)
    backend = Backend()
    assert backend._effective_generation_cap("input", 1, prompt_token_count=4) == 12
    with pytest.raises(ContextOverflow, match="16 tokens"):
        backend._effective_generation_cap("input", 1, prompt_token_count=16)


@pytest.mark.parametrize("context", [3, 5])
def test_prism_native_tokenizer_rejection_preserves_value_error(context, http_backend):
    backend, received, _, _ = http_backend
    backend.n_ctx = context
    with pytest.raises(ContextOverflow) as failure:
        list(backend.stream_generate_tokens("whole request", max_tokens=3))
    assert isinstance(failure.value, ValueError)
    assert [path for path, _ in received] == ["/tokenize"]


@pytest.mark.parametrize("status, error, capacity", [
    (400, {"message": "request exceeds the context window: prompt (513 tokens) leaves no room to answer in the context (512); requests are never truncated"}, True),
    (400, {"message": "request exceeds the context window: prompt (312 tokens) + max tokens (300) exceeds the context (512); requests are never truncated."}, True),
    (400, {"message": "Prompt exceeds the context window"}, True),
    (400, {"message": "Input too large", "code": "context_length_exceeded"}, True),
    (400, {"message": "Could not allocate a context buffer"}, False),
    (500, {"message": "decoder failed", "code": "context_length_exceeded"}, False),
    (200, {"message": "Prompt exceeds the context window", "code": "context_length_exceeded"}, False),
])
def test_strata_distinguishes_chat_preflight_from_service_and_stream_failures(status, error, capacity):
    with closing(LocalService()) as service:
        backend = StrataReasoningBackend("active")
        backend.is_loaded = True
        backend.model_id = MODEL
        backend._endpoint = ("127.0.0.1", service.server.server_port)
        backend._api_key = KEY
        service.status = status
        if status == 200:
            service.body = event(chunk({"content": "partial"})) + event({"error": error})
        else:
            service.content_type = "application/json"
            service.body = json.dumps({"error": error}).encode()
        stream = backend.stream_generate_tokens("send this unchanged")
        if status == 200:
            assert next(stream) == "partial"
        with pytest.raises(StrataError) as failure:
            list(stream)
        assert isinstance(failure.value, ContextOverflow) is capacity
        posts = [payload for path, _, payload in service.requests if path == "/v1/chat/completions"]
        assert len(posts) == 1 and posts[0]["messages"] == [{"role": "user", "content": "send this unchanged"}]


@pytest.mark.parametrize("use_archive", [False, True])
def test_unrecallable_oldest_keeps_its_source_while_later_verified_pair_makes_room(tmp_path, monkeypatch, use_archive):
    from core.context_memory import ChatArchive
    from core.online_consolidator import ManagedAwakeOnlineConsolidator
    from tests.unit.context_backend_fixture import TransactionalBackend
    from tests.unit.test_persistent_context_memory import add_pair

    monkeypatch.setattr("core.online_consolidator.release_training_memory",
                        lambda _: {"after_mb": 0, "released_mb": 0})

    class Engine:
        active_backend = "controller"
        _gui_force_single_stream = True
        settings = SimpleNamespace(max_new_tokens=1)

        def _format_prompt_with_history(self, prompt, history):
            return " ".join([turn["content"] for turn in history] + [prompt])

        def stream_solve(self, prompt, history=None, **kwargs):
            self.generated_with = list(history)
            yield "response"

        def solve(self, *args, **kwargs):
            pytest.fail("unexpected branch")

    install_awake_auto_learning(Engine)
    engine = Engine()
    backend = engine.controller_backend = TransactionalBackend(tmp_path)
    service = engine.awake_consolidator = ManagedAwakeOnlineConsolidator(backend, evict_ratio=0.05, session_id="one")
    rows = [{"role": "system", "content": "Always answer precisely."},
            {"role": "user", "content": "Unrecallable first message."},
            {"role": "assistant", "content": "Acknowledged."},
            {"role": "tool", "content": "Protected operation."}]
    # A system boundary closes the unrecallable pair before the protected tool.
    rows.insert(3, {"role": "system", "content": "Preserve all operations."})
    for index in range(10):
        add_pair(rows, index)
    rows.append({"role": "user", "content": "Unfinished exchange."})
    original = list(rows)
    history = rows
    if use_archive:
        history = ChatArchive(tmp_path / "archive.sqlite", "one")
        for row in rows:
            history.append(row)
    prompt = "Continue"
    packed = engine._format_prompt_with_history(prompt, original)
    engine._context_budget_tokens = len(backend.tokenizer.encode(packed))
    assert "".join(engine.stream_solve(prompt, history=history)) == "response"
    assert backend.train_calls == 2  # failed first exchange, one successful later exchange
    expected = original[:5] + original[7:]
    assert engine.generated_with == expected
    assert history == (original if use_archive else expected)
    restarted = ManagedAwakeOnlineConsolidator(TransactionalBackend(tmp_path), session_id="one")
    assert list(restarted.active_history(history if use_archive else original)) == expected
    assert not any(getattr(turn, "_capacity_recovery", None) for turn in
                   (list(history) if use_archive else original))
    # The previous recovery left the prompt above the 80% soft watermark.
    # Reusing that same working list on another request must retry the watermark.
    assert "".join(engine.stream_solve("Next", history=engine.generated_with)) == "response"
    assert backend.train_calls == 3


def test_native_packer_preserves_literal_warnings_and_equal_text_from_distinct_roles():
    from core.full_context_hardening import install

    class Engine:
        pass

    install(Engine)
    engine = Engine()
    history = [{"role": "system", "content": "When weights are not currently loaded, say so."},
               {"role": "user", "content": "Quote the label Click '⬇️ Download'."},
               {"role": "assistant", "content": "weights are not currently loaded"},
               {"role": "assistant", "content": "repeat"}]
    packed = engine._format_prompt_with_history("repeat", history)
    assert all(row["content"] in packed for row in history[:2])
    assert "assistant\nweights are not currently loaded" not in packed
    assert "assistant\nrepeat<|im_end|>\n<|im_start|>user\nrepeat" in packed
    assert engine._format_prompt_with_history("repeat", history + [{"role": "user", "content": "repeat"}]) == packed


@pytest.mark.parametrize("stop", ["cancel", "storage"])
def test_operational_failure_stops_alternate_retirement_without_hiding_source(tmp_path, monkeypatch, stop):
    from core.context_memory import ChatArchive
    from core.online_consolidator import ManagedAwakeOnlineConsolidator
    from tests.unit.context_backend_fixture import TransactionalBackend
    from tests.unit.test_persistent_context_memory import add_pair

    monkeypatch.setattr("core.online_consolidator.release_training_memory",
                        lambda _: {"after_mb": 0, "released_mb": 0})
    backend = TransactionalBackend(tmp_path)
    service = ManagedAwakeOnlineConsolidator(backend, evict_ratio=0.05)
    archive = ChatArchive(tmp_path / "archive.sqlite", "one")
    archive.append({"role": "user", "content": "Unknown earliest source"})
    archive.append({"role": "assistant", "content": "Acknowledged."})
    for i in range(6):
        add_pair(archive, i)
    original = list(archive)
    cancellation = Event()
    attempts = []
    train = backend.train_mini_batch

    def train_until_failure(**kwargs):
        attempts.append(kwargs["data"][0])
        if len(attempts) == 2:
            if stop == "storage":
                raise OSError("checkpoint write failed")
            cancellation.set()
            raise InterruptedError("request cancelled")
        return train(**kwargs)

    backend.train_mini_batch = train_until_failure
    with pytest.raises(InterruptedError if stop == "cancel" else OSError):
        service.prepare_archive(archive, lambda rows: 1 + len(rows) * 10, 140, cancellation)
    assert len(attempts) == 2
    assert list(service.active_history(archive)) == original
