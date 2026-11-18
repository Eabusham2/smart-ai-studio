"""No neural weights: exact branch/tool/response ordering and original routing."""
import inspect
import json
from types import SimpleNamespace
import threading
import pytest

from core.best_work import stream_best_work, split_selected_action
from core.generation_metrics import TokenMeter, RequestMetrics, observe_generations
from core.mlx_engine import MLXReasoningBackend

CALL = '<tool_call>{"name":"read_file","arguments":{"path":"a"}}</tool_call>'


def controller(outputs):
    events = []
    class Engine:
        is_model_loaded = True
        _last_stream_pro_meta = None
        def __init__(self):
            self.turn = 0
            self.contexts = []
        def stream_solve(self, prompt, history=None, cancel_event=None, progress_callback=None, decision_callback=None):
            current = self.turn
            self.turn += 1
            self.contexts.append((prompt, history, self._best_work_route_with_history))
            if decision_callback:
                decision_callback({"kind": "route", "total": 9 if current == 0 else 1, "branch_count": 8 if current == 0 else 1})
            output = outputs[current]
            try:
                for piece in output(prompt) if callable(output) else output:
                    events.append(("generate", current, piece))
                    yield piece
            finally:
                events.append(("closed", current))
    class Tools:
        def list_available_tools(self):
            return [{"name": "read_file"}]
        def execute_tool(self, name, args):
            events.append(("execute", name, args))
            return True, "ACTUAL disk data"
    engine = Engine()
    return SimpleNamespace(app=SimpleNamespace(engine=engine, tools=Tools())), events


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


def test_default_second_pass_reviews_only_selected_draft_and_streams_final_live():
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
    stream = stream_best_work(ctrl, "Read the file.", original, status_callback=statuses.append)
    assert next(stream) == "Revised "
    assert ("generate", 1, "answer.") not in events
    assert list(stream) == ["answer."]
    assert not any(row[0] == "execute" for row in events)
    assert [(row["pass_index"], row["passes"], row["final"]) for row in statuses
            if row["kind"] == "pass"] == [(1, 2, False), (2, 2, True)]
    contexts = ctrl.app.engine.contexts
    assert contexts[0][2] is False and contexts[1][2] is True
    assert contexts[1][0].count("Read the file.") == 1
    assert original[1] not in contexts[1][1] and original[0] in contexts[1][1]
    assert original == [{"role": "system", "content": "Use metric units."},
                        {"role": "user", "content": "Read the file."}]
    assert ctrl.last_best_work_metadata["passes_completed"] == 2
    assert ctrl.last_best_work_metadata["pass_usage"]["tokens"] is None


def test_three_pass_setting_chains_immediate_winners_without_accumulating_drafts():
    def second(prompt):
        assert _prior_draft(prompt) == "possible answer A"
        return ["improved answer B"]

    def third(prompt):
        assert _prior_draft(prompt) == "improved answer B"
        assert "possible answer A" not in prompt
        return ["final answer C"]

    ctrl, _ = controller([["possible answer A"], second, third])
    ctrl.app.engine.settings = SimpleNamespace(best_work_passes=3)
    assert list(stream_best_work(ctrl, "original")) == ["final answer C"]
    assert ctrl.last_best_work_metadata["passes_requested"] == 3


def test_only_final_selected_invocation_runs_and_repeat_is_blocked_across_stages():
    ctrl, events = controller([[CALL], ["Chosen operation. " + CALL], ["Check the file. " + CALL], [CALL]])
    statuses = []
    rendered = "".join(stream_best_work(ctrl, "Read the file.", status_callback=statuses.append))
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
    rendered = "".join(stream_best_work(ctrl, "Describe an otter"))
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

    stream = stream_best_work(ctrl, "Read file", cancel_event=stopped, status_callback=decision)
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
    assert ctrl.last_best_work_metadata["passes_completed"] == (0 if when == "during_draft" else 1)


@pytest.mark.parametrize("bad", [0, -1, 1.5, "2", False])
def test_invalid_pass_count_fails_before_generation(bad):
    ctrl, events = controller([])
    with pytest.raises(ValueError, match="positive integer"):
        list(stream_best_work(ctrl, "request", passes=bad))
    assert not events


def test_refinement_keeps_structured_user_metadata_in_history():
    ctrl, _ = controller([["draft"], ["answer"]])
    turn = {"role": "user", "content": "Read this diagram", "attachments": ["diagram.png"]}
    assert list(stream_best_work(ctrl, turn["content"], [turn])) == ["answer"]
    assert turn in ctrl.app.engine.contexts[1][1]
