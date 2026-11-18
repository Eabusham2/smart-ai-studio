"""Large checked programs keep the same execution semantics on every platform."""
import errno
import subprocess
import time
from types import SimpleNamespace

import pytest

from core.verifier import GroundTruthVerifier


@pytest.mark.parametrize("windows", [False, True])
def test_argument_limit_retries_unchanged_program_over_stdin(monkeypatch, windows):
    calls = []
    program = "assert 2 + 2 == 4\n" * 10000

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if len(calls) == 1:
            error = OSError(errno.ENAMETOOLONG if windows else errno.E2BIG, "argument limit")
            if windows:
                error.winerror = 206
            raise error
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    verifier = GroundTruthVerifier()
    result = verifier._verify_in_subprocess(program, time.perf_counter())
    assert result.passed and len(calls) == 2
    assert calls[0][0][-2:] == ["-c", program]
    assert calls[1][0][-1] == "-" and calls[1][1]["input"] == program
    assert {k: v for k, v in calls[1][1].items() if k != "input"} == calls[0][1]


def test_checked_worker_preserves_lower_inherited_limit_and_bounds_candidate_input():
    resource = pytest.importorskip("resource")
    import sys

    name = "RLIMIT_DATA" if sys.platform == "darwin" else "RLIMIT_AS"
    # On Darwin even a clean Python interpreter can reserve much more virtual
    # DATA than it has resident. Choose a valid inherited native limit there.
    inherited = 1 << 47 if sys.platform == "darwin" else 128 * 1048576
    requested = inherited * 2
    outer = """import resource, runpy, subprocess, sys
kind = getattr(resource, sys.argv[1])
hard = resource.getrlimit(kind)[1]
cap = int(sys.argv[2]) if hard == resource.RLIM_INFINITY else min(int(sys.argv[2]), hard)
resource.setrlimit(kind, (cap, cap))
applied = resource.getrlimit(kind)
assert all(value != resource.RLIM_INFINITY and value <= cap for value in applied)
program = f"import resource; print(resource.getrlimit(resource.{sys.argv[1]}))"
if sys.platform == "darwin":
    run = runpy.run_path(sys.argv[4])["run_resident_task"]
    outcome = run(program, int(sys.argv[3]), 5)
else:
    worker = runpy.run_path(sys.argv[4])["_LIMITED_PYTHON"]
    outcome = subprocess.run([sys.executable, "-I", "-S", "-c", worker, sys.argv[1], sys.argv[3]],
                             input=program, capture_output=True, text=True)
assert outcome.returncode == 0, outcome.stderr
assert outcome.stdout.strip() == str(applied), (outcome.stdout, applied)
"""
    from pathlib import Path
    module = "resident_verifier.py" if sys.platform == "darwin" else "verifier.py"
    worker_file = Path(__file__).resolve().parents[2] / "core" / module
    result = subprocess.run([sys.executable, "-c", outer, name, str(inherited), str(requested), str(worker_file)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_checked_python_rejects_excess_memory_and_timeout():
    pytest.importorskip("resource")
    from core.learning_code_verifier import PythonTaskVerifier
    cases = [{"args": [1], "expected": 1}]
    memory = PythonTaskVerifier({"function": "f", "cases": cases, "memory_mb": 96})
    assert memory("def f(n):\n    return len('a' * (384 * 1024 * 1024)) // (384 * 1024 * 1024)") == 0, memory.results
    assert any(reason in memory.results[0]["reason"] for reason in ("MemoryError", "resident memory", "exited with code -")), memory.results
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
