"""Model-free regression tests for Best Work execution boundaries."""
import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace

source = Path(__file__).resolve().parents[2] / "core/best_work.py"
spec = importlib.util.spec_from_file_location("best_work_test", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Engine:
    is_model_loaded = True
    def __init__(self, results=()):
        self.results = iter(results)
        self.calls = []
    def solve(self, prompt, *, history, cancel_event, force_branch_count, progress_callback):
        self.calls.append((prompt, history, force_branch_count))
        return next(self.results)


class Registry:
    def __init__(self):
        self.calls = []
    def list_available_tools(self):
        return [{"name": name} for name in ("read_file", "write_file", "web_search")]
    def execute_tool(self, name, args):
        self.calls.append((name, args))
        return True, "actual result"


class Controller:
    def __init__(self, results=()):
        self.app = SimpleNamespace(engine=Engine(results), tools=Registry())
        self.calls = []
    def call(self, name, args, allow_update=False):
        self.calls.append((name, args, allow_update))
        return {"status": "success", "artifact_id": "real"}


def meta(*candidates, winner=0):
    return {"raw_branches": candidates, "winning_branch": winner, "branch_count": len(candidates)}


def run(ctrl, answer="", data=None, prompt="Research this"):
    return "".join(module.continue_selected_best_work(
        ctrl, prompt=prompt, history=[{"role": "user", "content": prompt}],
        answer=answer, metadata=data or meta(answer), branch_count=4,
        cancel_event=threading.Event(),
    ))


def test_extract_first_tool_boundary_after_thinking_and_text():
    text = (
        "<think>Reason.</think>\nPlan.\n"
        '<tool_call>{"name":"read_file","arguments":{"path":"x"}}</tool_call>'
        "untrusted future answer"
    )
    before, action, tail = module.split_selected_action(text)
    assert before == "<think>Reason.</think>\nPlan.\n"
    assert action == ("tool_call", "read_file", {"path": "x"})
    assert tail == "untrusted future answer"


def test_code_fence_and_thinking_are_not_executable():
    fence = chr(96) * 3
    text = (
        '<think><tool_call>{"name":"write_file","arguments":{}}</tool_call></think>'
        + "\n" + fence + "\n"
        + '<tool_call>{"name":"write_file","arguments":{}}</tool_call>'
        + "\n" + fence + "\nAnswer"
    )
    before, action, _ = module.split_selected_action(text)
    assert before == text and action is None


def test_plain_best_answer_no_extra_inference():
    ctrl = Controller()
    result = run(ctrl, data=meta("loser", "<think>Thought.</think>\nBest answer", winner=1))
    assert result == "<think>Thought.</think>\nBest answer"
    assert not ctrl.app.engine.calls and not ctrl.app.tools.calls
    assert ctrl.last_best_work_metadata["winning_branch"] == 1


def test_only_selected_tool_executes_and_reasons_using_real_result():
    losing = '<tool_call>{"name":"write_file","arguments":{"path":"bad"}}</tool_call>'
    winning = (
        "Plan first.\n"
        '<tool_call>{"name":"read_file","arguments":{"path":"right"}}</tool_call>'
        "invented output which must be ignored"
    )
    ctrl = Controller([("Final response.", meta("Final response."))])
    result = run(ctrl, data=meta(losing, winning, winner=1))
    assert "Plan first" in result and "invented output" not in result
    assert result.count("actual result") == 1 and result.endswith("Final response.")
    assert ctrl.app.tools.calls == [("read_file", {"path": "right"})]
    assert len(ctrl.app.engine.calls) == 1
    assert "actual result" in ctrl.app.engine.calls[0][1][-1]["content"]


def test_repeated_tool_action_blocked():
    call = '<tool_call>{"name":"read_file","arguments":{"path":"a"}}</tool_call>'
    ctrl = Controller([(call, meta(call))])
    result = run(ctrl, answer=call)
    assert "blocked a duplicate" in result
    assert len(ctrl.app.tools.calls) == 1


def test_nonexistent_or_unrequested_side_effect_is_not_executed():
    call = '<tool_call>{"name":"write_file","arguments":{"path":"a"}}</tool_call>'
    ctrl = Controller([("Finished.", meta("Finished."))])
    result = run(ctrl, answer=call, prompt="Compare different approaches")
    assert "explicit user request" in result
    assert not ctrl.app.tools.calls


def test_selected_media_action_only_and_single_continuation():
    call = '<media_call>{"name":"media_generate","arguments":{"kind":"image","prompt":"lake"}}</media_call>'
    ctrl = Controller([("Finished.", meta("Finished."))])
    result = run(ctrl, answer=call)
    assert ctrl.calls == [("media_generate", {"kind": "image", "prompt": "lake"}, False)]
    assert "real" in result and result.endswith("Finished.")
