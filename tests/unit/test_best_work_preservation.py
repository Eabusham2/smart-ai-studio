"""Regression cases for unchanged archive identity and N=1 sampling behavior."""
from functools import wraps
import threading

import pytest

from core.best_work import stream_best_work
from core.best_work_selection import SelectionError
from core.context_memory import ChatArchive
from core.online_consolidator import ManagedAwakeOnlineConsolidator
from tests.unit.context_backend_fixture import TransactionalBackend
from tests.unit.test_best_work_tournament import make_controller, request_kind
from tests.unit.test_persistent_context_memory import add_pair
from tests.unit.test_segmented_chat_recall import PassageBackend


class DecoderReached(Exception):
    """Stop after the production archive/awake pipeline has prepared inference."""


def prepared_archive_controller(tmp_path, monkeypatch, *, pairs, budget):
    ctrl, _, _, _ = make_controller(lambda *_: "unused", database_path=str(tmp_path / "app.db"))
    engine = ctrl.app.engine
    backend = engine.controller_backend = TransactionalBackend(tmp_path / "learned")
    engine.active_backend = "controller"
    service = engine.awake_consolidator = ManagedAwakeOnlineConsolidator(
        backend, session_id="archive-chat", evict_ratio=0.05)
    engine._context_budget_tokens = budget
    engine._format_prompt_with_history = lambda prompt, history: " ".join(
        [turn["content"] for turn in history] + [prompt])
    archive = ChatArchive(str(tmp_path / "archive.db"), "chat")
    for index in range(pairs):
        add_pair(archive, index)
    archive.append({"role": "user", "content": "Unfinished question"})
    prepared = []

    def stop_at_decoder(prompt, *, history, **kwargs):
        prepared.extend(history)
        raise DecoderReached()

    # Learning, receipts, archive projection, the installed awake wrapper and
    # Pro solve are real. No sampled answer is needed to verify prepared IDs.
    monkeypatch.setattr(engine, "generate_parallel_branches", stop_at_decoder)
    monkeypatch.setattr("core.online_consolidator.release_training_memory",
                        lambda _: {"after_mb": 0, "released_mb": 0})
    return ctrl, backend, service, archive, prepared


def test_best_work_capacity_retirement_keeps_archive_receipt_ids(tmp_path, monkeypatch):
    ctrl, backend, service, archive, prepared = prepared_archive_controller(
        tmp_path, monkeypatch, pairs=4, budget=113)
    visible = list(archive)
    original_ids = {turn.turn_id for turn in visible}

    with pytest.raises(DecoderReached):
        list(stream_best_work(ctrl, "Continue", archive, passes=2))

    # The Best Work guide crosses capacity. Actual verified consolidation
    # frees two exchanges, and its receipts must match durable archive IDs.
    assert backend.train_calls > 0
    retained = [turn for turn in prepared if turn.get("role") != "system"]
    assert len(retained) == 5
    assert {turn.turn_id for turn in retained} <= original_ids
    assert list(service.active_history(archive)) == retained
    assert list(archive) == visible
    assert [turn.turn_id for turn in archive] == [turn.turn_id for turn in visible]


def test_best_work_preserves_single_use_capacity_recovery_marker(tmp_path, monkeypatch):
    ctrl, backend, _, archive, prepared = prepared_archive_controller(
        tmp_path, monkeypatch, pairs=3, budget=120)
    history = list(archive)
    marker = history[0]._capacity_recovery = {"prepared": True}

    with pytest.raises(DecoderReached):
        list(stream_best_work(ctrl, "Continue", history, passes=2))

    # A previously recovered working list is below capacity but above the
    # soft watermark. Re-copying it must not trigger another weight update.
    assert backend.train_calls == 0
    assert marker == {}
    assert not any(getattr(turn, "_capacity_recovery", None) for turn in prepared)


@pytest.mark.parametrize("authorized", [True, False])
def test_learned_document_keeps_original_tool_authorization(tmp_path, monkeypatch, authorized):
    ctrl, _, _, actions = make_controller(lambda *_: "unused", database_path=str(tmp_path / "app.db"))
    engine = ctrl.app.engine
    backend = engine.controller_backend = PassageBackend(tmp_path / "learned")
    backend.tokenizer.model_max_length = engine._context_budget_tokens = 2048
    backend.is_loaded = True
    engine.active_backend = "controller"
    engine.gguf_backend = None
    engine.calculate_token_entropy = lambda _: 0.1
    engine.awake_consolidator = ManagedAwakeOnlineConsolidator(backend)
    ctrl.app.tools.list_available_tools = lambda: [{"name": "write_file"}]
    monkeypatch.setattr("core.online_consolidator.release_training_memory",
                        lambda _: {"after_mb": 0, "released_mb": 0})

    instruction = "Write a short note to notes.txt. " if authorized else "The label is cedar. "
    source = instruction + "cedar is the label. " * 230
    prompt = "Follow these instructions: <document>" + source + "</document>"
    archive = ChatArchive(str(tmp_path / "archive.db"), "chat")
    archive.append({"role": "user", "content": prompt})
    sampled_prompts = []
    recall_stream = backend.stream_generate_tokens

    def answer(text):
        sampled_prompts.append(text)
        if request_kind(text) == "compare":
            return '{"winner":"B"}'
        if "Actual tool result" in text:
            return "Completed."
        return '<tool_call>{"name":"write_file","arguments":{"path":"notes.txt","content":"cedar"}}</tool_call>'

    def stream_tokens(prompt, **options):
        if "<|im_start|>" in prompt:
            yield answer(prompt)
        else:
            yield from recall_stream(prompt, **options)

    # Only sampled chat text is scripted. Document learning, recall verification,
    # archive preparation, installed Pro wrappers and authorization remain real.
    backend.stream_generate_tokens = stream_tokens
    backend.generate_branches = lambda prompt, branch_count=1, **options: [
        answer(prompt) for _ in range(branch_count)]
    rendered = "".join(stream_best_work(ctrl, prompt, archive, passes=2))

    assert backend.train_calls > 0
    assert len(actions) == int(authorized)
    if authorized:
        assert actions == [("write_file", {"path": "notes.txt", "content": "cedar"})]
        assert "write_file (complete)" in rendered
    else:
        assert "This operation requires an explicit user request" in rendered
    assert rendered.endswith("Completed.")
    assert sampled_prompts and all("Learned document " in text for text in sampled_prompts)
    assert all(source not in text and instruction not in text for text in sampled_prompts)
    assert any("Actual tool result" in text for text in sampled_prompts)
    assert archive[-1]["content"] == prompt


@pytest.mark.parametrize("stream_top_p", [None, 0.71])
def test_single_branch_refinement_preserves_stream_sampling_cutoff(tmp_path, stream_top_p):
    ctrl, model, _, _ = make_controller(
        lambda model, call: '{"winner":"B"}' if call["kind"] == "compare" else "Answer",
        database_path=str(tmp_path / "chat.db"))
    engine = ctrl.app.engine
    engine.settings.search_top_p = 0.53
    if stream_top_p is not None:
        stream = engine.stream_solve

        @wraps(stream)
        def explicit_cutoff(prompt, **options):
            yield from stream(prompt, top_p=stream_top_p, **options)

        engine.stream_solve = explicit_cutoff

    assert "".join(stream_best_work(ctrl, "Explain the answer", passes=2)) == "Answer"
    expected = 0.92 if stream_top_p is None else stream_top_p
    assert [call["top_p"] for call in model.calls] == [expected] * 3
    assert [call["temperature"] for call in model.calls] == [0.65, 0.65, 0.20]
    assert engine.settings.search_top_p == 0.53


def test_multi_branch_keeps_configured_search_cutoff(tmp_path):
    ctrl, model, _, _ = make_controller(
        lambda model, call: '{"winner":"A"}' if call["kind"] == "compare" else "Answer",
        entropy=0.6, database_path=str(tmp_path / "chat.db"))
    ctrl.app.engine.settings.search_top_p = 0.53
    assert "".join(stream_best_work(ctrl, "Explain the answer", passes=2)) == "Answer"
    assert sum(call["kind"] == "original" for call in model.calls) > 1
    assert all(call["top_p"] == 0.53 for call in model.calls)
    assert ctrl.app.engine.settings.search_top_p == 0.53


@pytest.mark.parametrize("failure", ["invalid_judge", "cancel"])
def test_single_branch_sampling_setting_restored_after_failure(tmp_path, failure):
    cancelled = threading.Event()

    def respond(model, call):
        if failure == "cancel" and call["kind"] == "refine":
            cancelled.set()
        return "Not a selection" if call["kind"] == "compare" else "Answer"

    ctrl, model, _, _ = make_controller(respond, database_path=str(tmp_path / "chat.db"))
    ctrl.app.engine.settings.search_top_p = 0.53
    if failure == "invalid_judge":
        with pytest.raises(SelectionError):
            list(stream_best_work(ctrl, "Explain the answer", passes=2, cancel_event=cancelled))
    else:
        assert list(stream_best_work(ctrl, "Explain the answer", passes=2, cancel_event=cancelled)) == []
    assert model.calls and all(call["top_p"] == 0.92 for call in model.calls)
    assert ctrl.app.engine.settings.search_top_p == 0.53


@pytest.mark.parametrize("preexisting", [False, True])
def test_selection_scope_restored_when_live_stream_close_fails(tmp_path, preexisting):
    ctrl, _, _, _ = make_controller(lambda *_: ["Answer", " more"],
                                    database_path=str(tmp_path / "chat.db"))
    engine = ctrl.app.engine
    prior_selector = object()
    if preexisting:
        engine._best_work_select_candidates = prior_selector
        engine._best_work_passes = 9
    original_stream = engine.stream_solve

    @wraps(original_stream)
    def failed_close(*args, **kwargs):
        try:
            yield from original_stream(*args, **kwargs)
        finally:
            raise RuntimeError("backend close failed")

    engine.stream_solve = failed_close
    iterator = stream_best_work(ctrl, "Question", passes=1)
    assert next(iterator) == "Answer"
    with pytest.raises(RuntimeError, match="backend close failed"):
        iterator.close()
    if preexisting:
        assert engine._best_work_select_candidates is prior_selector
        assert engine._best_work_passes == 9
    else:
        assert not hasattr(engine, "_best_work_select_candidates")
        assert not hasattr(engine, "_best_work_passes")
