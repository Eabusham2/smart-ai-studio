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
    from core.verifier import _LIMITED_PYTHON

    name = "RLIMIT_DATA" if sys.platform == "darwin" else "RLIMIT_AS"
    requested = 256 * 1048576
    inherited = 128 * 1048576
    # Use a separate interpreter to impose a tighter hard limit without changing
    # pytest's process. Keep that interpreter and the actual worker in isolation.
    outer = """import resource, subprocess, sys
kind = getattr(resource, sys.argv[1])
hard = resource.getrlimit(kind)[1]
cap = int(sys.argv[2]) if hard == resource.RLIM_INFINITY else min(int(sys.argv[2]), hard)
resource.setrlimit(kind, (cap, cap))
child = subprocess.run([sys.executable, "-I", "-S", "-c", sys.argv[4], sys.argv[1], sys.argv[3]],
                       input=f"import resource; print(resource.getrlimit(resource.{sys.argv[1]}))",
                       capture_output=True, text=True)
assert child.returncode == 0, child.stderr
assert child.stdout.strip() == str((cap, cap)), (child.stdout, cap)
"""
    import os
    env = {**os.environ, "MallocNanoZone": "0"} if sys.platform == "darwin" else None
    result = subprocess.run([sys.executable, "-I", "-S", "-c", outer, name, str(inherited),
                             str(requested), _LIMITED_PYTHON], capture_output=True, text=True, env=env, timeout=10)
    assert result.returncode == 0, result.stderr


def test_checked_python_rejects_excess_memory_and_timeout():
    pytest.importorskip("resource")
    from core.learning_code_verifier import PythonTaskVerifier
    cases = [{"args": [1], "expected": 1}]
    memory = PythonTaskVerifier({"function": "f", "cases": cases, "memory_mb": 96})
    assert memory("def f(n):\n    return len('a' * (384 * 1024 * 1024)) // (384 * 1024 * 1024)") == 0, memory.results
    assert "MemoryError" in memory.results[0]["reason"] or "exited with code -" in memory.results[0]["reason"]
    time_limited = PythonTaskVerifier({"function": "f", "cases": cases, "timeout_seconds": .25})
    assert time_limited("def f(n):\n    while n:\n        pass\n    return 1") == 0
    assert "timed out" in time_limited.results[0]["reason"]
