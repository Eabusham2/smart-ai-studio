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
    serialized = json.dumps(_canonical_ast(node.body), sort_keys=True, separators=(',', ':'))
    actual = hashlib.sha256(serialized.encode()).hexdigest()
    assert actual == expected


def test_settings_restore_original_defaults(monkeypatch):
    monkeypatch.delenv("BEST_WORK_PASSES", raising=False)
    monkeypatch.delenv("LEARNING_IMPLEMENTATION", raising=False)
    settings = Settings()
    assert settings.best_work_passes == 1
    assert settings.learning_implementation == "legacy"


def test_default_best_work_really_executes_original_single_pass():
    from tests.unit.test_best_work_live_handoff import controller
    ctrl, events = controller([["Original complete answer"]])
    seen = []
    assert "".join(stream_best_work(ctrl, "Question", status_callback=seen.append)) == "Original complete answer"
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
