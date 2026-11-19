"""Exercise the Canvas tool's real child interpreter and failure diagnostics."""
import errno
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import venv

import psutil
import pytest

from core.tools import AgentToolRegistry


@pytest.fixture
def registry(tmp_path):
    return AgentToolRegistry(workspace_dir=str(tmp_path))


def line_fourteen_program():
    # Leading blank lines must survive the editor/tool boundary.
    return ("\n\nprint('started before failure')\n" + "# source line\n" * 10
            + "raise ValueError('line fourteen diagnostic')\n")


def test_stdout_does_not_hide_traceback_or_editor_line_number(registry):
    code = line_fourteen_program()
    assert code.splitlines()[13].startswith("raise ValueError")
    ok, result = registry.execute_tool("python_sandbox", {"code": code})
    assert not ok
    assert "started before failure" in result
    assert 'File "<string>", line 14' in result
    assert "ValueError: line fourteen diagnostic" in result


def test_leading_indentation_is_not_silently_changed(registry):
    ok, result = registry.execute_tool("python_sandbox", {"code": "    print('indented')\n"})
    assert not ok
    assert "line 1" in result and "IndentationError" in result


@pytest.mark.parametrize("language", ["python", "Python3", "py", ""])
def test_complete_python_fences_preserve_source_lines(registry, language):
    code = f"```{language}\n\nprint('fenced output')\nraise ValueError('fenced failure')\n```"
    ok, result = registry.execute_tool("python_sandbox", {"code": code})
    assert not ok
    assert "fenced output" in result
    assert 'File "<string>", line 4' in result
    assert "ValueError: fenced failure" in result


@pytest.mark.parametrize("code", ["", " \n\t\n", None, "```python\n```"])
def test_empty_code_does_not_launch_a_child(registry, monkeypatch, code):
    def unexpected_launch(*args, **kwargs):
        pytest.fail("Empty Canvas input must not launch Python")
    monkeypatch.setattr("core.tools.subprocess.run", unexpected_launch)
    assert registry.execute_tool("python_sandbox", {"code": code}) == (False, "Python code required.")


def test_utf8_stdout_and_stderr_are_both_reported(registry):
    code = "import sys\nprint('مرحبا ☃')\nprint('diagnostic: café ⚠', file=sys.stderr)\n"
    ok, result = registry.execute_tool("python_sandbox", {"code": code})
    assert ok
    assert "مرحبا ☃" in result and "diagnostic: café ⚠" in result


def test_silent_nonzero_exit_is_not_reported_as_success(registry):
    ok, result = registry.execute_tool("python_sandbox", {"code": "raise SystemExit(7)"})
    assert not ok
    assert "Exit 7" in result and "exited with code 7 and no output" in result
    assert "executed successfully" not in result


def test_large_canvas_source_executes_once_without_temporary_scripts(registry, tmp_path):
    # This exceeds Linux's per-argument limit and Windows' command-line limit.
    # Other platforms may accept -c directly; both paths must run the same code.
    code = "#" + "x" * 200_000 + "\nprint('large canvas complete')\n"
    before = set(tmp_path.iterdir())
    ok, result = registry.execute_tool("python_sandbox", {"code": code})
    assert ok, result
    assert result.count("large canvas complete") == 1
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("windows", [False, True])
def test_argument_limit_fallback_keeps_interpreter_workspace_timeout_and_source(registry, monkeypatch, windows):
    real_run = subprocess.run
    calls = []
    error = OSError(errno.ENAMETOOLONG if windows else errno.E2BIG, "argument limit")
    if windows:
        error.winerror = 206

    def launch(argv, **kwargs):
        calls.append((argv, kwargs))
        if len(calls) == 1:
            raise error
        return real_run(argv, **kwargs)

    monkeypatch.setattr("core.tools.subprocess.run", launch)
    code = line_fourteen_program()
    ok, result = registry.execute_tool("python_sandbox", {"code": code})
    assert not ok
    assert "started before failure" in result
    assert 'File "<stdin>", line 14' in result
    assert "ValueError: line fourteen diagnostic" in result
    assert calls[0][0] == [sys.executable, "-c", code]
    assert calls[1][0] == [sys.executable, "-"]
    assert calls[1][1]["input"] == code
    for _argv, options in calls:
        assert options["cwd"] == registry.workspace_dir
        assert options["timeout"] == 6.0
        assert options.get("shell", False) is False
        assert options["env"]["PYTHONIOENCODING"] == "utf-8"


def test_unrelated_launch_failure_does_not_retry(registry, monkeypatch):
    calls = []
    def denied(*args, **kwargs):
        calls.append(args)
        raise PermissionError(errno.EACCES, "permission denied")
    monkeypatch.setattr("core.tools.subprocess.run", denied)
    ok, result = registry.execute_tool("python_sandbox", {"code": "print('never runs')"})
    assert not ok and "permission denied" in result
    assert len(calls) == 1


def test_runtime_error_does_not_retry_or_repeat_side_effects(registry, tmp_path):
    code = ("from pathlib import Path\n"
            "with Path('executions.txt').open('a') as log: log.write('once\\n')\n"
            "print('before error')\nraise RuntimeError('no retry')\n")
    ok, result = registry.execute_tool("python_sandbox", {"code": code})
    assert not ok and "RuntimeError: no retry" in result
    assert (tmp_path / "executions.txt").read_text() == "once\n"


def test_timeout_keeps_partial_output_and_reaps_child(registry, tmp_path):
    code = "import os, time\nprint('CHILD_PID=' + str(os.getpid()), flush=True)\ntime.sleep(60)\n"
    before = set(tmp_path.iterdir())
    ok, result = registry.execute_tool("python_sandbox", {"code": code})
    assert not ok and "timed out (6.0s limit)" in result
    pid = re.search(r"CHILD_PID=(\d+)", result)
    assert pid, result
    assert not psutil.pid_exists(int(pid.group(1))), "Timed-out Python must be killed and reaped"
    assert set(tmp_path.iterdir()) == before


def test_packaged_interpreter_and_workspace_paths_with_spaces(registry, tmp_path, monkeypatch):
    # A separate real interpreter path models the bundle layout; no PATH lookup
    # or host Python fallback is needed, even with spaces and non-ASCII names.
    bundle = tmp_path / "Smart AI Studio é" / "python"
    venv.EnvBuilder(with_pip=False, symlinks=False).create(bundle)
    executable = bundle / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    workspace = tmp_path / "Canvas workspace é"
    workspace.mkdir()
    registry.set_workspace_dir(str(workspace))
    monkeypatch.setattr("core.tools.sys.executable", str(executable))
    monkeypatch.setenv("PATH", "")
    code = ("import json, math, os, sys\n"
            "print(json.dumps({'python': sys.executable, 'cwd': os.getcwd(), "
            "'result': math.sqrt(144)}))\n")
    ok, result = registry.execute_tool("python_sandbox", {"code": code})
    assert ok, result
    report = json.loads(result.split("```\n", 1)[1].rsplit("\n```", 1)[0])
    assert Path(report["python"]).resolve() == executable.resolve()
    assert Path(report["cwd"]).resolve() == workspace.resolve()
    assert report["result"] == 12.0
