"""Model-free contracts for Best-of-N branch selection and one-tool handoff."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.entropy_router import EntropyRouter
from core.media_orchestrator import MediaController


def make_controller():
    media = MediaController.__new__(MediaController)
    media.lock = threading.RLock()
    media.app = SimpleNamespace(cancel_event=threading.Event())
    media.call = Mock(return_value={"status": "success", "artifact_id": "one"})
    return media


def test_gui_has_only_one_pro_checkbox_and_no_heuristic_picker():
    source = (Path(__file__).resolve().parents[2] / "app_gui.py").read_text()
    assert 'self.pro_branches_var = tk.BooleanVar(value=False)' in source
    assert "self.auto_pro_var" not in source
    assert "self.pro_branch_count_select" not in source
    assert "select_branch_budget" not in source


def test_original_model_entropy_router_keeps_one_eight_sixteen_policy():
    router = EntropyRouter()
    assert router.route(0.10)[1] == 1
    assert router.route(0.45)[1] == 8
    assert router.route(0.85)[1] == 16


def test_selected_branch_executes_one_terminal_tool_and_returns_once():
    media = make_controller()
    answer = '<think>Work out the approach first.</think><media_call>{"name":"media_generate","arguments":{"kind":"image","prompt":"one result"}}</media_call>'
    event = media.execute_selected_branch_tool(answer, "Generate an image")
    assert event == {"name": "media_generate", "result": {"status": "success", "artifact_id": "one"}}
    media.call.assert_called_once_with("media_generate", {"kind": "image", "prompt": "one result"}, allow_update=False)


@pytest.mark.parametrize("answer", [
    "Plain answer.",
    "Before <media_call>{\"name\":\"media_generate\",\"arguments\":{}}</media_call> after",
    "<media_call>{\"name\":\"media_generate\",\"arguments\":{}}</media_call><media_call>{\"name\":\"media_generate\",\"arguments\":{}}</media_call>",
])
def test_nonterminal_or_multiple_tool_calls_are_not_executed(answer):
    media = make_controller()
    assert media.execute_selected_branch_tool(answer, "hello") is None
    media.call.assert_not_called()


def test_learning_tool_requires_explicit_user_authorization():
    media = make_controller()
    call = '<media_call>{"name":"media_learn","arguments":{"model_id":"example","source":"file"}}</media_call>'
    media.execute_selected_branch_tool(call, "create an image")
    media.call.assert_called_once_with("media_learn", {"model_id": "example", "source": "file"}, allow_update=False)
    media.call.reset_mock()
    media.execute_selected_branch_tool(call, "learn from my image")
    media.call.assert_called_once_with("media_learn", {"model_id": "example", "source": "file"}, allow_update=True)


def test_cancellation_prevents_selected_tool_execution():
    media = make_controller()
    media.app.cancel_event.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        media.execute_selected_branch_tool('<media_call>{"name":"media_generate","arguments":{}}</media_call>', "generate image")
    media.call.assert_not_called()
