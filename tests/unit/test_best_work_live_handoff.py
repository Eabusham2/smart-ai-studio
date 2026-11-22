"""No neural weights: exact branch/tool/response ordering and original routing."""
import inspect
import json
from functools import wraps
from types import SimpleNamespace
import threading
import pytest

from core.best_work import stream_best_work, split_selected_action
from core.generation_metrics import TokenMeter, RequestMetrics, observe_generations
from core.mlx_engine import MLXReasoningBackend

CALL = '<tool_call>{"name":"read_file","arguments":{"path":"a"}}</tool_call>'


def controller(outputs):
    from tests.unit.test_best_work_tournament import make_controller

    events = []

    def respond(model, call):
        if call["kind"] == "compare":
            events.append(("judge", call["prompt"]))
            yield '{"winner":"B"}'
            return
        current = engine.turn
        engine.turn += 1
        prompt, history = engine._test_request
        engine.contexts.append((prompt, [turn.copy() for turn in (history or [])],
                                engine._best_work_route_with_history))
        try:
            output = outputs[current]
            for piece in output(prompt) if callable(output) else output:
                events.append(("generate", current, piece))
                yield piece
        finally:
            events.append(("closed", current))

    ctrl, model, _, _ = make_controller(respond)
    engine = ctrl.app.engine
    engine.turn = 0
    engine.contexts = []
    engine._scripted_model = model
    stream, generate = engine.stream_solve, engine.generate_parallel_branches

    @wraps(stream)
    def capture_stream(prompt, *args, **kwargs):
        engine._test_request = (prompt, kwargs.get("history"))
        yield from stream(prompt, *args, **kwargs)

    @wraps(generate)
    def capture_generate(prompt, *args, **kwargs):
        engine._test_request = (prompt, kwargs.get("history"))
        return generate(prompt, *args, **kwargs)

    engine.stream_solve = capture_stream
    engine.generate_parallel_branches = capture_generate

    class Tools:
        def list_available_tools(self):
            return [{"name": "read_file"}]
        def execute_tool(self, name, args):
            events.append(("execute", name, args))
            return True, "ACTUAL disk data"
    ctrl.app.tools = Tools()
    return ctrl, events


def test_selected_text_and_call_display_before_execution_then_model_continues():
    ctrl, events = controller([["<think>reason</think>Plan ", CALL, "forbidden imagined result"], ["Final conclusion"]])
    iterator = stream_best_work(ctrl, "Read the file and explain it", cancel_event=threading.Event(), passes=1)
    rendered = []
    for piece in iterator:
        rendered.append(piece)
        events.append(("display", piece))
    assert "forbidden imagined" not in "".join(rendered)
    assert "".join(rendered).endswith("Final conclusion")
    execution = next(i for i, e in enumerate(events) if e[0] == "execute")
    displays = [e[1] for e in events[:execution] if e[0] == "display"]
    assert "Plan " in "".join(displays) and 'Calling read_file({"path": "a"})' in "".join(displays)
    assert len([e for e in events if e[0] == "execute"]) == 1
    context = ctrl.app.engine.contexts[1]
    assert context[2] is True
    assert any("ACTUAL disk data" in turn["content"] for turn in context[1])


def test_final_response_does_not_execute_or_start_an_extra_model_pass():
    ctrl, events = controller([["<think>reason</think>", "Final answer"]])
    result = "".join(stream_best_work(ctrl, "question", passes=1))
    assert result.endswith("Final answer")
    assert ctrl.app.engine.turn == 1
    assert not any(e[0] == "execute" for e in events)


def test_cancel_between_display_and_execution_prevents_dispatch():
    ctrl, events = controller([["Plan ", CALL]])
    cancelled = threading.Event()
    stream = stream_best_work(ctrl, "Read file", cancel_event=cancelled, passes=1)
    assert next(stream) == "Plan "
    assert "Calling read_file" in next(stream)
    cancelled.set()
    assert list(stream) == []
    assert not any(e[0] == "execute" for e in events)


def test_duplicate_call_does_not_execute_again():
    ctrl, events = controller([[CALL], [CALL]])
    assert "Duplicate tool request blocked" in "".join(stream_best_work(ctrl, "read file", passes=1))
    assert sum(e[0] == "execute" for e in events) == 1


def test_single_pass_single_branch_counts_every_completed_tool_stage():
    second_call = '<tool_call>{"name":"read_file","arguments":{"path":"b"}}</tool_call>'
    ctrl, events = controller([["Inspect a. ", CALL], ["Inspect b. ", second_call], ["Finished."]])
    rendered = ''.join(stream_best_work(ctrl, 'Read both files and summarize them', passes=1))
    assert rendered.endswith('Finished.')
    assert [row[1:] for row in events if row[0] == 'execute'] == [
        ('read_file', {'path': 'a'}), ('read_file', {'path': 'b'})]
    assert not any(row[0] == 'judge' for row in events)
    report = ctrl.last_best_work_metadata
    assert report['passes_requested'] == report['passes_completed'] == 1
    assert report['stage'] == report['total_passes_completed'] == 3
    assert [(row['stage'], row['pass_index'], row['completed']) for row in report['pass_summaries']] == [
        (1, 1, True), (2, 1, True), (3, 1, True)]
    assert report['pass_usage'] == {'invocations': 3, 'tokens': 5}


def test_real_mlx_wrapper_exposes_stop_and_progress_options():
    # This tests the installed chain of real production wrappers, not __new__
    # on an unwrapped method (which hid the earlier regression).
    parameters = inspect.signature(MLXReasoningBackend.generate_branches).parameters
    assert "stop_at_tool_boundary" in parameters
    assert "progress_callback" in parameters


@pytest.mark.parametrize("text", [
    '<think>' + CALL,
    'Example:\n\t```json\n' + CALL,
    'Example:\n~~~json\n' + CALL,
    'Use `' + CALL + '` as an example',
])
def test_incomplete_thinking_and_code_examples_never_execute(text):
    assert split_selected_action(text)[1] is None


def test_json_close_tag_inside_argument_string_is_not_a_boundary():
    text = '<tool_call>{"name":"read_file","arguments":{"path":"x</tool_call>y"}}</tool_call>tail'
    before, action, tail = split_selected_action(text)
    assert action[2]["path"] == "x</tool_call>y"
    assert tail == "tail"


def test_aggregate_tokens_include_losing_branches_not_winner_replay():
    now = [0.0]
    clock = lambda: now[0]
    request = RequestMetrics(clock)
    with observe_generations(request.update):
        one = TokenMeter(clock=clock)
        now[0] = 1; one.observe(delta=1)
        now[0] = 2; one.observe(delta=4); one.finish()
        two = TokenMeter(clock=clock)
        now[0] = 3; two.observe(delta=1)
        now[0] = 4; two.observe(delta=2); two.finish()
    before = request.snapshot()
    assert before["tokens"] == 8 and before["invocations"] == 2
    assert before["decode_tps"] == 3
    now[0] = 20 # Displaying a winning answer and a slow tool is not decode.
    after = request.snapshot()
    assert after["tokens"] == 8 and after["decode_tps"] == 3
    assert after["wall_tps"] == 0.4


def test_text_events_are_never_fabricated_tokens():
    meter = TokenMeter()
    meter.observe(text=True)
    meter.observe(text=True)
    meter.finish()
    assert meter.tokens is None and meter.snapshot()["decode_tps"] is None


def _prior_draft(prompt):
    return json.loads(prompt.rsplit("\n", 1)[-1])


@pytest.mark.parametrize("configuration", ["default_settings", "engine_defaults", "missing_pass_setting"])
def test_default_best_work_executes_two_passes_and_only_displays_the_refined_answer(monkeypatch, configuration):
    from config.settings import Settings

    monkeypatch.delenv("BEST_WORK_PASSES", raising=False)

    def review(prompt):
        assert _prior_draft(prompt) == "Initial selected draft."
        return ["Refined final answer."]

    ctrl, events = controller([["Initial selected draft."], review])
    if configuration == "default_settings":
        ctrl.app.engine.settings = Settings(use_mock=True, auto_download=False)
    elif configuration == "missing_pass_setting":
        del ctrl.app.engine.settings.best_work_passes
    statuses = []
    assert list(stream_best_work(ctrl, "Question", status_callback=statuses.append)) == ["Refined final answer."]
    assert ctrl.app.engine.turn == 2
    assert [(event["pass_index"], event["passes"], event["final"])
            for event in statuses if event["kind"] == "pass"] == [(1, 2, False), (2, 2, False)]
    assert len([event for event in events if event[0] == "judge"]) == 1
    assert ctrl.last_best_work_metadata["passes_requested"] == ctrl.last_best_work_metadata["passes_completed"] == 2
    assert not any(event[0] == "execute" for event in events)


@pytest.mark.parametrize("selection", ["environment", "settings", "request"])
def test_explicit_single_pass_still_overrides_the_two_pass_default(monkeypatch, selection):
    from config.settings import Settings

    monkeypatch.delenv("BEST_WORK_PASSES", raising=False)
    if selection == "environment":
        monkeypatch.setenv("BEST_WORK_PASSES", "1")
    settings = Settings(use_mock=True, auto_download=False, **({"best_work_passes": 1} if selection == "settings" else {}))
    ctrl, _ = controller([["Original single-pass answer."]])
    ctrl.app.engine.settings = settings
    request = {"passes": 1} if selection == "request" else {}
    statuses = []
    assert list(stream_best_work(ctrl, "Question", status_callback=statuses.append, **request)) == ["Original single-pass answer."]
    assert ctrl.app.engine.turn == 1
    assert not any(event["kind"] == "pass" for event in statuses)


def test_opt_in_second_pass_reviews_draft_then_judges_before_display():
    draft = 'Choose path "a".\n' + CALL
    def review(prompt):
        assert prompt.startswith("Read the file.")
        assert _prior_draft(prompt) == draft
        assert "Imagined result" not in prompt
        yield "Revised "
        yield "answer."

    ctrl, events = controller([[draft + "Imagined result"], review])
    statuses = []
    original = [{"role": "system", "content": "Use metric units."},
                {"role": "user", "content": "Read the file."}]
    stream = stream_best_work(ctrl, "Read the file.", original, status_callback=statuses.append, passes=2)
    assert next(stream) == "Revised answer."
    assert ("generate", 1, "answer.") in events
    assert len([event for event in events if event[0] == "judge"]) == 1
    assert list(stream) == []
    assert not any(row[0] == "execute" for row in events)
    assert [(row["pass_index"], row["passes"], row["final"]) for row in statuses
            if row["kind"] == "pass"] == [(1, 2, False), (2, 2, False)]
    contexts = ctrl.app.engine.contexts
    assert contexts[0][2] is False and contexts[1][2] is False
    assert contexts[1][0].count("Read the file.") == 1
    assert original[1] not in contexts[1][1] and original[0] in contexts[1][1]
    assert original == [{"role": "system", "content": "Use metric units."},
                        {"role": "user", "content": "Read the file."}]
    assert ctrl.last_best_work_metadata["passes_completed"] == 2
    assert ctrl.last_best_work_metadata["pass_usage"]["tokens"] == 4
    assert ctrl.last_best_work_metadata["pass_usage"]["invocations"] == 3


def test_three_pass_setting_chains_revisions_without_accumulating_drafts():
    def second(prompt):
        assert _prior_draft(prompt) == "possible answer A"
        return ["improved answer B"]

    def third(prompt):
        assert _prior_draft(prompt) == "improved answer B"
        assert "possible answer A" not in prompt
        return ["final answer C"]

    ctrl, _ = controller([["possible answer A"], second, third])
    ctrl.app.engine.settings.best_work_passes = 3
    assert list(stream_best_work(ctrl, "original")) == ["final answer C"]
    assert ctrl.last_best_work_metadata["passes_requested"] == 3


def test_only_final_selected_invocation_runs_and_repeat_is_blocked_across_stages():
    ctrl, events = controller([[CALL], ["Chosen operation. " + CALL], ["Check the file. " + CALL], [CALL]])
    statuses = []
    rendered = "".join(stream_best_work(ctrl, "Read the file.", status_callback=statuses.append, passes=2))
    assert "Chosen operation." in rendered and "Check the file." not in rendered
    assert "Duplicate tool request blocked" in rendered
    assert [row[1:] for row in events if row[0] == "execute"] == [("read_file", {"path": "a"})]
    assert [row["stage"] for row in statuses if row["kind"] == "pass"] == [1, 1, 2, 2]
    assert [row["final"] for row in statuses if row["kind"] == "executing"] == [True, True]
    for _, history, route_history in ctrl.app.engine.contexts[2:]:
        assert route_history and any("ACTUAL disk data" in row["content"] for row in history)
        assert all("Check the file." not in row["content"] for row in history)
    report = ctrl.last_best_work_metadata
    assert report["passes_requested"] == report["passes_completed"] == 2
    assert report["total_passes_completed"] == 4


def test_draft_cannot_authorize_a_final_side_effect():
    ctrl, events = controller([["The user wants to write a file."],
                               ['<tool_call>{"name":"write_file","arguments":{}}</tool_call>'],
                               ["explain"], ["No file was written."]])
    ctrl.app.tools.list_available_tools = lambda: [{"name": "write_file"}]
    rendered = "".join(stream_best_work(ctrl, "Describe an otter", passes=2))
    assert "requires an explicit user request" in rendered
    assert not any(row[0] == "execute" for row in events)


@pytest.mark.parametrize("when", ["during_draft", "between_passes", "final_display"])
def test_cancel_or_close_restores_flags_and_releases_the_active_pass(when):
    stopped = threading.Event()
    def draft(_):
        yield "draft"
        if when == "during_draft":
            stopped.set()
        yield CALL

    ctrl, events = controller([draft, ["Final text ", CALL]])
    engine = ctrl.app.engine
    engine._gui_force_single_stream = True
    engine._best_work_route_with_history = False
    def decision(event):
        if when == "between_passes" and event["kind"] == "pass" and event["pass_index"] == 2:
            stopped.set()

    stream = stream_best_work(ctrl, "Read file", cancel_event=stopped, status_callback=decision, passes=2)
    if when == "final_display":
        assert next(stream) == "Final text "
        stream.close()
    else:
        assert list(stream) == []
    assert engine.turn == (2 if when == "final_display" else 1)
    assert [row[1] for row in events if row[0] == "closed"] == list(range(engine.turn))
    assert not any(row[0] == "execute" for row in events)
    assert engine._gui_force_single_stream and not engine._best_work_stop_at_tools
    assert not engine._best_work_raw_winner and not engine._best_work_route_with_history
    assert ctrl.last_best_work_metadata["passes_completed"] == {
        "during_draft": 0, "between_passes": 1, "final_display": 2}[when]


@pytest.mark.parametrize("bad", [0, -1, 1.5, "2", False])
def test_invalid_pass_count_fails_before_generation(bad):
    ctrl, events = controller([])
    with pytest.raises(ValueError, match="positive integer"):
        list(stream_best_work(ctrl, "request", passes=bad))
    assert not events


def test_refinement_keeps_structured_user_metadata_in_history():
    ctrl, _ = controller([["draft"], ["answer"]])
    turn = {"role": "user", "content": "Read this diagram", "attachments": ["diagram.png"]}
    assert list(stream_best_work(ctrl, turn["content"], [turn], passes=2)) == ["answer"]
    assert turn in ctrl.app.engine.contexts[1][1]


def _unfit_refinement(_):
    from core.context_memory import ContextOverflow
    raise ContextOverflow("Packed input leaves no room for generation")


def test_capacity_fallback_uses_latest_completed_winner_and_stops_remaining_passes():
    ctrl, events = controller([["first draft"], ["complete second draft"], _unfit_refinement])
    statuses = []
    assert list(stream_best_work(ctrl, "request", passes=5, status_callback=statuses.append)) == [
        "complete second draft"]
    assert [entry for entry in events if entry[0] == "closed"] == [("closed", 0), ("closed", 1), ("closed", 2)]
    meta = ctrl.last_best_work_metadata
    assert meta["passes_requested"] == 5 and meta["total_passes_completed"] == meta["passes_completed"] == 2
    assert [p["completed"] for p in meta["pass_summaries"]] == [True, True, False]
    skipped = [event for event in statuses if event["kind"] == "refinement_skipped"]
    assert len(skipped) == 1 and skipped[0]["selected_pass"] == 2
    assert len([event for event in events if event[0] == "judge"]) == 1


@pytest.mark.parametrize("budget", [0, 2])
def test_capacity_fallback_reuses_tool_budget_display_and_duplicate_guard(budget):
    draft = "Plan " + CALL + "guessed tool result"
    ctrl, events = controller([[draft], _unfit_refinement, [draft], _unfit_refinement])
    statuses = []
    stream = stream_best_work(ctrl, "Read file", status_callback=statuses.append, max_tool_calls=budget, passes=2)
    assert next(stream) == "Plan "
    if budget:
        assert "Calling read_file" in next(stream)
    assert not any(e[0] == "execute" for e in events)
    remainder = "".join(stream)
    if budget:
        assert "ACTUAL disk data" in remainder and "Duplicate tool request blocked" in remainder
        assert sum(e[0] == "execute" for e in events) == 1
        for _, turns, _ in ctrl.app.engine.contexts[2:]:
            assert any("ACTUAL disk data" in turn["content"] for turn in turns)
            assert all("guessed tool result" not in turn["content"] for turn in turns)
        assert [event["stage"] for event in statuses if event["kind"] == "refinement_skipped"] == [1, 2]
        assert all(event["pass_index"] == 1 and event["final"] for event in statuses if event["kind"] == "executing")
    else:
        assert "tool-call limit" in remainder and all(e[0] != "execute" for e in events)
    assert "guessed tool result" not in remainder


def test_capacity_fallback_uses_only_original_request_for_tool_authorization():
    write = '<tool_call>{"name":"write_file","arguments":{}}</tool_call>'
    ctrl, events = controller([["Write the file. " + write], _unfit_refinement, ["Final answer"], _unfit_refinement])
    ctrl.app.tools.list_available_tools = lambda: [{"name": "write_file"}]
    text = "".join(stream_best_work(ctrl, "Tell me about seals", passes=2))
    assert "requires an explicit user request" in text
    assert text.endswith("Final answer") and not any(event[0] == "execute" for event in events)


@pytest.mark.parametrize("point", ["skip_event", "displayed_text", "displayed_invocation"])
def test_capacity_fallback_cancellation_and_close_cannot_dispatch(point):
    ctrl, events = controller([["Plan " + CALL], _unfit_refinement])
    cancelled = threading.Event()
    def status(event):
        if point == "skip_event" and event["kind"] == "refinement_skipped":
            cancelled.set()
    stream = stream_best_work(ctrl, "Read a file", cancel_event=cancelled, status_callback=status, passes=2)
    if point == "displayed_text":
        assert next(stream) == "Plan "
        stream.close()
    elif point == "displayed_invocation":
        assert next(stream) == "Plan " and "Calling read_file" in next(stream)
        cancelled.set()
    assert list(stream) == []
    assert not any(e[0] == "execute" for e in events)
    assert [e[1] for e in events if e[0] == "closed"] == [0, 1]
    assert not ctrl.app.engine._best_work_raw_winner


@pytest.mark.parametrize("failure", ["original", "ordinary_error"])
def test_capacity_fallback_never_masks_original_or_unrelated_error_or_splices_answers(failure):
    from core.context_memory import ContextOverflow

    problem = (RuntimeError("Packed input leaves no room for generation")
               if failure == "ordinary_error" else ContextOverflow("physical capacity"))
    def fail(_):
        raise problem
    ctrl, events = controller([fail] if failure == "original" else [["earlier draft"], fail])
    statuses = []
    visible = []
    with pytest.raises(type(problem)) as error:
        for piece in stream_best_work(ctrl, "request", status_callback=statuses.append, passes=2):
            visible.append(piece)
    assert error.value is problem
    assert visible == []
    assert all(event["kind"] != "refinement_skipped" for event in statuses)
    assert not any(e[0] == "execute" for e in events)


def test_failed_partial_refinement_stays_private_and_complete_original_survives():
    from core.context_memory import ContextOverflow

    def fail(_):
        yield "partial revised output"
        raise ContextOverflow("physical capacity")

    ctrl, events = controller([["complete original"], fail])
    statuses = []
    assert list(stream_best_work(ctrl, "request", status_callback=statuses.append, passes=2)) == [
        "complete original"]
    assert any(event["kind"] == "refinement_skipped" for event in statuses)
    assert not any(event[0] in {"judge", "execute"} for event in events)


def test_stream_only_adapter_cannot_silently_skip_default_best_work():
    from core.best_work_selection import SelectionError

    calls = []
    def stream(prompt, history=None, cancel_event=None):
        calls.append(prompt)
        yield "Original draft"
    engine = SimpleNamespace(stream_solve=stream)
    ctrl = SimpleNamespace(app=SimpleNamespace(engine=engine))
    displayed = []
    with pytest.raises(SelectionError, match="per-branch candidate selection"):
        for piece in stream_best_work(ctrl, "request"):
            displayed.append(piece)
    assert ctrl.last_best_work_metadata["passes_requested"] == 2
    assert ctrl.last_best_work_metadata["passes_completed"] == 0
    assert displayed == [] and calls == ["request"]
    assert not hasattr(engine, "_best_work_select_candidates")
    assert list(stream_best_work(ctrl, "request", passes=1)) == ["Original draft"]
    assert ctrl.last_best_work_metadata["passes_completed"] == 1


def test_tool_result_starts_a_fresh_stage_without_a_stale_fallback_draft():
    from core.context_memory import ContextOverflow

    ctrl, events = controller([[CALL], _unfit_refinement, _unfit_refinement])
    output = []
    with pytest.raises(ContextOverflow):
        for piece in stream_best_work(ctrl, "Read file", passes=2):
            output.append(piece)
    assert "".join(output).count("Calling read_file") == 1
    assert "".join(output).count("ACTUAL disk data") == 1
    assert sum(e[0] == "execute" for e in events) == 1
    meta = ctrl.last_best_work_metadata
    assert meta["stage"] == 2 and meta["passes_completed"] == 0
    assert meta["total_passes_completed"] == 1 and "skip_reason" not in meta["pass_summaries"][-1]


@pytest.mark.parametrize("projection", [False, True])
def test_direct_best_work_projects_archive_before_copying_and_preserves_transcript(tmp_path, monkeypatch, projection):
    from core.context_memory import ActiveChatHistory, ChatArchive
    from core.online_consolidator import ManagedAwakeOnlineConsolidator
    from tests.unit.context_backend_fixture import TransactionalBackend
    from tests.unit.test_persistent_context_memory import add_pair

    archive = ChatArchive(tmp_path / "chat.db", "chat")
    add_pair(archive, 9)
    retired = {row.turn_id for row in archive}
    backend = TransactionalBackend(tmp_path)
    service = ManagedAwakeOnlineConsolidator(backend)
    monkeypatch.setattr("core.online_consolidator.release_training_memory", lambda _: {})
    assert service.sleep(archive)["turns_retired"] == len(retired)
    archive.append({"role": "system", "content": "Preserve this instruction."})
    archive.append({"role": "user", "content": "What is the label?"})
    original = list(archive)
    store = backend.learning_state
    head = store.current_manifest()["metadata"]["memory"]["retirement_head"]
    history = ActiveChatHistory(archive, service._retirement_ledger(store), head) if projection else archive

    ctrl, _ = controller([["Selected answer."], _unfit_refinement])
    engine = ctrl.app.engine
    engine.active_backend = "controller"
    engine.controller_backend = backend
    engine.awake_consolidator = service
    engine._context_budget_tokens = 512
    engine._format_prompt_with_history = lambda question, rows: " ".join(row["content"] for row in rows) + " " + question

    decode = ChatArchive._decode
    def active_rows_only(row):
        assert row[0] not in retired, "Best Work materialized a retired archive row"
        return decode(row)
    with monkeypatch.context() as mp:
        mp.setattr(ChatArchive, "_decode", staticmethod(active_rows_only))
        assert list(stream_best_work(ctrl, "What is the label?", history, passes=2)) == ["Selected answer."]
    assert [turn.turn_id for turn in archive] == [turn.turn_id for turn in original]
    assert list(archive) == original
    assert original[-2:] == engine.contexts[0][1][-2:]
    assert all("cedar9" not in str(turns) for _, turns, _ in engine.contexts)
