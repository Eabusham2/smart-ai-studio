"""No neural weights: exact branch/tool/response ordering and original routing."""
import inspect
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
            for piece in outputs[current]:
                events.append(("generate", current, piece))
                yield piece
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
    iterator = stream_best_work(ctrl, "Read the file and explain it", cancel_event=threading.Event())
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
    result = "".join(stream_best_work(ctrl, "question"))
    assert result.endswith("Final answer")
    assert ctrl.app.engine.turn == 1
    assert not any(e[0] == "execute" for e in events)


def test_cancel_between_display_and_execution_prevents_dispatch():
    ctrl, events = controller([["Plan ", CALL]])
    cancelled = threading.Event()
    stream = stream_best_work(ctrl, "Read file", cancel_event=cancelled)
    assert next(stream) == "Plan "
    assert "Calling read_file" in next(stream)
    cancelled.set()
    assert list(stream) == []
    assert not any(e[0] == "execute" for e in events)


def test_duplicate_call_does_not_execute_again():
    ctrl, events = controller([[CALL], [CALL]])
    assert "Duplicate tool request blocked" in "".join(stream_best_work(ctrl, "read file"))
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
