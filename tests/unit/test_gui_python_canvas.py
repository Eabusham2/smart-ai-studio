"""The real Canvas editor passes unchanged source to the real Python tool."""
from core.tools import AgentToolRegistry
from tests.unit.test_best_work_chat_ui import work_app
from tests.unit.test_python_canvas_execution import line_fourteen_program


def test_canvas_preserves_editor_line_number_and_shows_exception(work_app, tmp_path):
    app = work_app
    app.tools = AgentToolRegistry(workspace_dir=str(tmp_path))
    recorded = []
    app._append_tool_call = lambda *args: recorded.append(args)
    code = line_fourteen_program()
    app.txt_canvas.delete("1.0", "end")
    app.txt_canvas.insert("1.0", code)
    app._on_canvas_run_code()
    assert len(recorded) == 1
    tool, label, result = recorded[0]
    assert tool == "python_sandbox" and label == "Canvas Code Execution"
    assert "started before failure" in result
    assert 'File "<string>", line 14' in result
    assert "ValueError: line fourteen diagnostic" in result
    assert app.txt_canvas.get("1.0", "end-1c") == code


def test_canvas_does_not_strip_indentation_or_execute_blank_input(work_app, tmp_path):
    app = work_app
    app.tools = AgentToolRegistry(workspace_dir=str(tmp_path))
    recorded = []
    app._append_tool_call = lambda *args: recorded.append(args)
    app.txt_canvas.delete("1.0", "end")
    app.txt_canvas.insert("1.0", "    print('indented')\n")
    app._on_canvas_run_code()
    assert len(recorded) == 1 and "IndentationError" in recorded[0][2]
    app.txt_canvas.delete("1.0", "end")
    app.txt_canvas.insert("1.0", " \n\t\n")
    app._on_canvas_run_code()
    assert len(recorded) == 1


def test_canvas_long_stdout_keeps_final_traceback_visible_in_bounded_real_widget(work_app, tmp_path):
    app = work_app
    app.tools = AgentToolRegistry(workspace_dir=str(tmp_path))
    code = line_fourteen_program().replace(
        "print('started before failure')", "print('started before failure ' + 'x' * 5000)")
    app.txt_canvas.delete("1.0", "end")
    app.txt_canvas.insert("1.0", code)
    app.chat_stream.configure(state="normal")
    app.chat_stream.delete("1.0", "end")
    app.chat_stream.configure(state="disabled")

    # Use the actual renderer, not a capture callback: its old 1,200-character
    # prefix preview hid the traceback even after the tool preserved stderr.
    app._on_canvas_run_code()
    app.root.update()
    shown = app.chat_stream.get("1.0", "end")
    assert "started before failure" in shown
    assert "Output truncated; showing start and final diagnostics" in shown
    assert 'File "<string>", line 14' in shown
    assert "ValueError: line fourteen diagnostic" in shown
    assert "Error (Exit 1)" in shown
    assert len(shown) < 1400  # Tool header plus the bounded 1,200-character preview.
    assert app.txt_canvas.get("1.0", "end-1c") == code


def test_canvas_run_routes_html_to_browser_without_python_or_workspace(work_app, tmp_path, monkeypatch):
    import app_gui
    from pathlib import Path
    from unittest.mock import Mock
    app = work_app
    app.workspace_dir = None
    app._canvas_preview_directory = str(tmp_path)
    app.tools = Mock()
    opened = []
    monkeypatch.setattr(app_gui.webbrowser, "open", lambda uri: opened.append(uri) or True)
    app._append_ai_message = Mock()
    code = '<!doctype HTML>\n<html><style>body { background: #1a1a2e; }</style><script>function flap() {}</script></html>'
    app.txt_canvas.delete("1.0", "end")
    app.txt_canvas.insert("1.0", code)
    app._on_canvas_run_code()
    assert opened == [(tmp_path / "canvas_interactive_preview.html").resolve().as_uri()]
    assert (tmp_path / "canvas_interactive_preview.html").read_text() == code
    app.tools.execute_tool.assert_not_called()
    assert app.txt_canvas.get("1.0", "end-1c") == code
