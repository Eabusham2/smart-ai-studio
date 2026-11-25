"""Large checked programs keep the same execution semantics in the container."""
import subprocess
import time

import pytest

from core.container_sandbox import ContainerExecution
from core.verifier import GroundTruthVerifier


def test_large_program_reaches_container_unchanged(monkeypatch):
    calls = []
    program = "assert 2 + 2 == 4\n" * 10000

    def run(source, **options):
        calls.append((source, options))
        return ContainerExecution(0, "", "", 1.0)

    monkeypatch.setattr("core.verifier.run_python_in_container", run)
    verifier = GroundTruthVerifier()
    result = verifier._verify_in_subprocess(program, time.perf_counter())
    assert result.passed and calls == [
        (program, {"timeout": 4.0, "max_memory_mb": 512, "image": "python:3.10-slim"})
    ]


def test_required_memory_mode_cannot_restore_host_execution(monkeypatch):
    calls = []

    def run(source, **options):
        calls.append((source, options))
        return ContainerExecution(0, "", "", 1.0)

    monkeypatch.setattr("core.verifier.run_python_in_container", run)
    verifier = GroundTruthVerifier(max_memory_mb=96, required_memory_resource="resident")
    result = verifier.verify_in_sandbox("def answer(): return 42", "assert answer() == 42")

    assert result.passed is True
    assert calls[0][1]["max_memory_mb"] == 96


def test_checked_python_rejects_excess_memory_and_timeout():
    pytest.importorskip("resource")
    from core.learning_code_verifier import PythonTaskVerifier
    cases = [{"args": [1], "expected": 1}]
    memory = PythonTaskVerifier({"function": "f", "cases": cases, "memory_mb": 96})
    assert memory("def f(n):\n    return len('a' * (384 * 1024 * 1024)) // (384 * 1024 * 1024)") == 0, memory.results
    assert any(reason in memory.results[0]["reason"] for reason in ("MemoryError", "resident memory", "Container exited with code")), memory.results
    time_limited = PythonTaskVerifier({"function": "f", "cases": cases, "timeout_seconds": .25})
    assert time_limited("def f(n):\n    while n:\n        pass\n    return 1") == 0
    assert "timed out" in time_limited.results[0]["reason"]


def test_resident_budget_checks_freed_peak_and_reaps_timed_out_worker(tmp_path):
    pytest.importorskip("resource")
    from core.resident_verifier import run_resident_task

    # Complete and free the allocation between polls; wait4 must still catch it.
    allocate = "a = bytearray(80 * 1048576); a[::4096] = b'x' * (len(a) // 4096)"
    code = allocate + "; del a; print('correct')"
    outcome = run_resident_task(code, 48 * 1048576, 6, poll_seconds=1)
    assert outcome.returncode != 0 and "peak resident memory" in outcome.stderr, outcome
    assert run_resident_task("print(1)", 48 * 1048576, 5).returncode == 0

    # Keep an oversize allocation alive. The supervisor must kill it promptly,
    # without waiting until the task finishes or the wall-time budget expires.
    outcome = run_resident_task(allocate + "; import time; time.sleep(4)", 48 * 1048576, 2)
    assert outcome.returncode != 0 and "resident memory" in outcome.stderr, outcome

    pidfile = tmp_path / "worker.pid"
    with pytest.raises(subprocess.TimeoutExpired):
        run_resident_task(f"import os, time; open({str(pidfile)!r}, 'w').write(str(os.getpid())); time.sleep(10)",
                          48 * 1048576, 1)
    pid = int(pidfile.read_text())
    import psutil
    deadline = time.monotonic() + 2
    while psutil.pid_exists(pid):
        assert time.monotonic() < deadline, "timed-out checked worker survived"
        time.sleep(.01)
