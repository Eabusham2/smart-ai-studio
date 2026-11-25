"""Static-policy regressions for generated-code execution boundaries."""

from types import SimpleNamespace

from core.container_sandbox import ContainerExecution, build_docker_command
from core.best_work import _execute_selected_tool
from core.swe_sandbox import HardenedSWESandbox
from core.verifier import GroundTruthVerifier


def _execution(returncode=0, *, error=None, timed_out=False, output_limited=False):
    return ContainerExecution(
        returncode,
        "expected output",
        "",
        2.0,
        timed_out=timed_out,
        output_limited=output_limited,
        error=error,
    )


def test_container_policy_revokes_host_authority():
    command = build_docker_command(
        "approved-python@sha256:test", "/tmp/candidate.cid", 512
    )

    assert command[:2] == ["docker", "run"]
    assert ["--network", "none"] == command[command.index("--network"):command.index("--network") + 2]
    assert "--read-only" in command
    assert ["--cap-drop", "ALL"] == command[command.index("--cap-drop"):command.index("--cap-drop") + 2]
    assert ["--user", "65534:65534"] == command[command.index("--user"):command.index("--user") + 2]
    assert ["--pids-limit", "64"] == command[command.index("--pids-limit"):command.index("--pids-limit") + 2]
    assert "--volume" not in command and "-v" not in command
    assert command[command.index("--pull") + 1] == "never"


def test_verifier_cannot_select_host_fallback(monkeypatch):
    calls = []

    def run(source, **options):
        calls.append((source, options))
        return _execution()

    monkeypatch.setattr("core.verifier.run_python_in_container", run)
    verifier = GroundTruthVerifier(use_docker=False, docker_image="approved-image")
    result = verifier.verify_in_sandbox(
        "def add(left, right): return left + right",
        "assert add(2, 3) == 5",
    )

    assert result.passed is True
    assert result.verifier_type == "container_sandbox"
    assert "def add" in calls[0][0] and "assert add" in calls[0][0]
    assert calls[0][1]["image"] == "approved-image"


def test_unavailable_container_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "core.verifier.run_python_in_container",
        lambda *args, **kwargs: _execution(None, error="docker unavailable"),
    )

    result = GroundTruthVerifier(use_docker=False).verify_in_sandbox(
        "print('must not reach a host interpreter')", "assert True"
    )

    assert result.passed is False
    assert result.verifier_type == "container_sandbox"
    assert "Sandbox unavailable" in result.details


def test_benchmark_python_preserves_results_in_container(monkeypatch):
    calls = []

    def run(source, **options):
        calls.append((source, options))
        return _execution()

    monkeypatch.setattr("core.swe_sandbox.run_python_in_container", run)
    sandbox = HardenedSWESandbox(
        timeout_sec=3.0, max_memory_mb=256, docker_image="approved-image"
    )
    result = sandbox.execute_python_code(
        "def multiply(left, right): return left * right",
        "assert multiply(3, 4) == 12",
    )

    assert result.passed is True and result.reward == 1.0
    assert "def multiply" in calls[0][0] and "assert multiply" in calls[0][0]
    assert calls[0][1] == {
        "timeout": 3.0,
        "max_memory_mb": 256,
        "image": "approved-image",
    }


def test_patched_benchmark_tests_use_container(monkeypatch):
    git_calls = []
    container_calls = []

    def git_run(command, **options):
        git_calls.append((command, options))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    def run_workspace(root, command, **options):
        container_calls.append((root, command, options))
        return _execution()

    monkeypatch.setattr("core.swe_sandbox.shutil.which", lambda name: "/usr/bin/git")
    monkeypatch.setattr("core.swe_sandbox.subprocess.run", git_run)
    monkeypatch.setattr("core.swe_sandbox.run_workspace_in_container", run_workspace)
    sandbox = HardenedSWESandbox(docker_image="approved-image")
    result = sandbox.verify_git_diff_patch(
        {"app.py": "value = 1\n"},
        "--- app.py\n+++ app.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n",
        "python -c \"import app; assert app.value == 2\"",
    )

    assert result.passed is True
    assert all(options.get("shell") is not True for _command, options in git_calls)
    assert container_calls[0][1] == "python -c \"import app; assert app.value == 2\""
    assert container_calls[0][2]["image"] == "approved-image"


def test_selected_python_tool_uses_verifier_boundary():
    class Tools:
        def __init__(self):
            self.calls = []

        def execute_tool(self, name, arguments):
            self.calls.append((name, arguments))
            return True, "checked"

    tools = Tools()
    result = _execute_selected_tool(
        tools,
        "python_sandbox",
        {"code": "print(42)", "assertions": "assert True"},
    )

    assert result == (True, "checked")
    assert tools.calls == [("dsl_evaluator", {
        "dsl_type": "python",
        "expression_or_code": "print(42)",
        "assertions": "assert True",
    })]
