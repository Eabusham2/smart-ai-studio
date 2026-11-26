"""The pre-PR17 implementations remain executable and retain their original contracts."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app_gui import SmartAIChatbotApp
from config.settings import Settings
from core.best_work import stream_best_work


# Body hashes captured from d3cead010685a04c95e97b2469c07fa3de627156.
# Renaming a function for explicit selection does not alter its algorithm.
BASELINE_BODIES = [
    ('core/best_work.py', 'stream_best_work_legacy', '0f38b3efe74b56afd1a7c9205a869615a04bfd26b889dd03064a2ba7d6aebbd4'),
    ('core/awake_auto_hook.py', '_apply_legacy_awake_learning', '9cad1b566c13f65ddd8353b4ca38f59df7060abb9001120e773e57a22cef879d'),
    ('app_gui.py', '_on_export_chat_button', '5ff33aaebc47ab67a141c8e310ab8efacf00811dcf69f4cd31780ba4010b248f'),
]


def _canonical_ast(value):
    # Python versions add empty AST fields and change ast.dump's defaults.
    # Compare the original algorithm without depending on those presentation
    # details, source locations, or the public name of the containing function.
    if isinstance(value, ast.AST):
        return [type(value).__name__, {
            name: _canonical_ast(field) for name, field in ast.iter_fields(value)
            if field is not None and field != []
        }]
    if isinstance(value, list):
        return [_canonical_ast(item) for item in value]
    return value


@pytest.mark.parametrize("path,name,expected", BASELINE_BODIES)
def test_original_algorithm_bodies_match_pre_merge_version(path, name, expected):
    tree = ast.parse((Path(__file__).resolve().parents[2] / path).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    if name == 'stream_best_work_legacy':
        # Approved one-line bugfix: preserve TranscriptTurn IDs and the shared
        # recovery marker. Normalize only that exact copy expression so every
        # other legacy handoff statement still matches the original hash.
        assignment = next(stmt for stmt in node.body if isinstance(stmt, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'turns'
                                  for target in stmt.targets))
        corrected = ast.parse('turns = [turn.copy() for turn in (history or [])]').body[0]
        assert _canonical_ast(assignment) == _canonical_ast(corrected)
        assignment.value.elt = ast.parse('dict(turn)', mode='eval').body
        # Approved authorization fix: the unprepared user request is used only
        # for dispatch; legacy generation and continuation keep their prompt.
        dispatch = next(call for call in ast.walk(node) if isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Name) and call.func.id == '_execute_action')
        corrected = ast.parse(
            '_execute_action(controller, action, original_prompt if authorization_prompt is None '
            'else str(authorization_prompt), executed)', mode='eval').body
        assert _canonical_ast(dispatch) == _canonical_ast(corrected)
        dispatch.args[2] = ast.parse('original_prompt', mode='eval').body
        # Media cards may present a real tool result instead of the two text
        # messages. Verify the complete presentation hook before comparing
        # the original dispatch and continuation algorithm.
        loop = next(stmt for stmt in ast.walk(node) if isinstance(stmt, ast.While))
        start = next(i for i, stmt in enumerate(loop.body) if isinstance(stmt, ast.Assign)
                     and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == "media_hook")
        presentation = ast.parse(r'''media_hook = getattr(controller, "media_result_presenter", None)
presenter = media_hook(action[1]) if callable(media_hook) else None
invocation = "\n⚙️ Calling " + action[1] + "(" + json.dumps(action[2], ensure_ascii=False) + ")\n"
if presenter is None:
    yield invocation''').body
        assert _canonical_ast(loop.body[start:start + 4]) == _canonical_ast(presentation)
        loop.body[start:start + 4] = [ast.Expr(value=ast.Yield(value=presentation[2].value))]
        dispatch_index = next(i for i, stmt in enumerate(loop.body)
                              if isinstance(stmt, ast.Assign) and stmt.value is dispatch)
        reported = loop.body[dispatch_index + 1]
        expected_report = ast.parse(r'''if presenter is None or not presenter(result):
    if presenter is not None:
        yield invocation
    yield f"\n⚙️ {action[1]} ({'complete' if ok else 'failed'}): {result}\n"''').body[0]
        assert _canonical_ast(reported) == _canonical_ast(expected_report)
        loop.body[dispatch_index + 1] = reported.body[-1]
    serialized = json.dumps(_canonical_ast(node.body), sort_keys=True, separators=(',', ':'))
    actual = hashlib.sha256(serialized.encode()).hexdigest()
    assert actual == expected


def test_best_work_default_uses_two_passes_and_keeps_legacy_selectable(monkeypatch):
    monkeypatch.delenv("BEST_WORK_PASSES", raising=False)
    monkeypatch.delenv("LEARNING_IMPLEMENTATION", raising=False)
    settings = Settings(learning_implementation="legacy")
    assert settings.best_work_passes == 2
    assert settings.learning_implementation == "legacy"


def test_explicit_single_pass_really_executes_original_implementation():
    from tests.unit.test_best_work_live_handoff import controller
    ctrl, events = controller([["Original complete answer"]])
    seen = []
    assert "".join(stream_best_work(ctrl, "Question", status_callback=seen.append, passes=1)) == "Original complete answer"
    assert ctrl.app.engine.turn == 1
    assert not any(event["kind"] == "pass" for event in seen)
    assert not any(event[0] == "execute" for event in events)


def test_original_export_still_uses_original_tool_and_filename(monkeypatch):
    import app_gui
    calls, messages = [], []
    app = SimpleNamespace(tools=SimpleNamespace(execute_tool=lambda name, args: (calls.append((name, args)) or (True, "saved"))),
                          _append_ai_message=messages.append)
    monkeypatch.setattr(app_gui.messagebox, "showinfo", lambda *a, **k: None)
    monkeypatch.setattr(app_gui.filedialog, "asksaveasfilename", lambda **k: pytest.fail("Original export never opens Save As"))
    SmartAIChatbotApp._on_export_chat_button(app)
    assert calls == [("export_chat_history", {"filename": "chat_history_export.md"})]
    assert "Chat History Exported" in messages[0]


def test_full_transcript_export_remains_available_alongside_original(tmp_path, monkeypatch):
    import app_gui
    destination = tmp_path / "all.md"
    turns = [{"role": "user", "content": "First message"}, {"role": "assistant", "content": "Complete reply"}]
    app = SimpleNamespace(chat_history={"tab": turns}, active_tab_id="tab")
    monkeypatch.setattr(app_gui.filedialog, "asksaveasfilename", lambda **k: str(destination))
    monkeypatch.setattr(app_gui.messagebox, "showinfo", lambda *a, **k: None)
    SmartAIChatbotApp._on_export_full_transcript(app)
    exported = destination.read_text()
    assert "First message" in exported and "Complete reply" in exported
    assert len(turns) == 2


@pytest.mark.parametrize("implementation, expected_count", [("legacy", 140), ("managed", 100)])
def test_original_full_transcript_and_new_paged_view_are_both_available(implementation, expected_count):
    rendered = []
    history = [{"role": "user", "content": str(index)} for index in range(140)]
    app = SimpleNamespace(settings=SimpleNamespace(learning_implementation=implementation),
        active_tab_id="tab", chat_history={"tab": history}, chat_stream=SimpleNamespace(configure=lambda **k: None),
        _append_rendered_user_turn=rendered.append, _append_rendered_ai_turn=rendered.append,
        _scroll_chat_to_bottom=lambda: None)
    SmartAIChatbotApp._render_tab_transcript(app)
    assert len(rendered) == expected_count
    assert rendered[-1] == "139"
    assert ("0" in rendered) == (implementation == "legacy")
    assert len(history) == 140

@pytest.mark.parametrize("presented", [False, True])
def test_legacy_media_result_presented_once_with_text_fallback(presented):
    from unittest.mock import Mock
    from tests.unit.test_best_work_live_handoff import controller

    call = '<media_call>{"name":"media_generate","arguments":{"prompt":"a landscape"}}</media_call>'
    ctrl, _ = controller([[call], ["Finished."]])
    ctrl.call = Mock(return_value={"status": "success", "path": "image.png"})
    card = Mock(return_value=presented)
    ctrl.media_result_presenter = Mock(return_value=card)
    answer = "".join(stream_best_work(ctrl, "Generate a landscape", passes=1))
    ctrl.call.assert_called_once()
    ctrl.media_result_presenter.assert_called_once_with("media_generate")
    assert json.loads(card.call_args.args[0])["path"] == "image.png"
    card.assert_called_once()
    assert answer.endswith("Finished.")
    assert ("Calling media_generate" in answer) is (not presented)
    assert ("image.png" in answer) is (not presented)
    assert "image.png" in ctrl.app.engine.contexts[1][1][-1]["content"]
