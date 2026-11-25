"""Regression tests for the benchmark sandbox compatibility boundary."""

from dataclasses import dataclass

import eval.swe_verifier_hardening as hardening
import master_4000_eval_suite  # installs the compatibility layer
import run_studio_complete as compat
from core.swe_sandbox import HardenedSWESandbox


def test_canonical_benchmark_inherits_mandatory_sandbox():
    assert issubclass(compat.POSIXHardenedSandbox, HardenedSWESandbox)
    assert compat.POSIXHardenedSandbox._smartai_patch_timeout_hardened is True


def test_legacy_adapter_delegates_to_secure_runner(monkeypatch):
    calls = []

    @dataclass
    class Result:
        passed: bool
        execution_time_ms: float
        output: str
        error: str | None = None
        reward: float = 0.0

    class LegacySandbox:
        timeout = 7.0
        max_memory_mb = 384
        max_files = 12
        docker_image = "approved-python@sha256:test"

    class SecureRunner:
        def __init__(self, **settings):
            calls.append(("settings", settings))

        def verify_git_diff_patch(self, repository, patch, command):
            calls.append(("verify", repository, patch, command))
            return Result(True, 3.0, "passed", None, 1.0)

    monkeypatch.setattr(hardening, "HardenedSWESandbox", SecureRunner)
    hardening._install_one(LegacySandbox, Result)
    result = LegacySandbox().verify_git_diff_patch(
        {"app.py": "value = 1\n"},
        "--- app.py\n+++ app.py\n",
        "python app.py",
    )

    assert result.passed is True
    assert calls[0][1] == {
        "timeout_sec": 7.0,
        "max_memory_mb": 384,
        "max_files": 12,
        "docker_image": "approved-python@sha256:test",
    }
    assert calls[1][0] == "verify"


def test_global_system_prompt_did_not_gain_unknown_instruction():
    import eval.master_4000_runtime as runtime

    assert "unknown" not in runtime.SYSTEM_PROMPT.lower()
