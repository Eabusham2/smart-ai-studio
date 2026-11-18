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
