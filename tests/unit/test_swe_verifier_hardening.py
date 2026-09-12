"""Regression tests for DeepSWE patch verifier timeout hardening."""

from types import SimpleNamespace

import eval.swe_verifier_hardening as hardening
import master_4000_eval_suite  # installs the hardening layer
import run_studio_complete as compat


def test_active_benchmark_patch_step_uses_configured_timeout(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs.get("timeout")))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(hardening.subprocess, "run", fake_run)

    sandbox = compat.POSIXHardenedSandbox(timeout_sec=4.0, max_memory_mb=512)
    result = sandbox.verify_git_diff_patch(
        {"app/calc.py": "def compute():\n    return 46\n"},
        """```diff
--- a/app/calc.py
+++ b/app/calc.py
@@ -1,2 +1,2 @@
 def compute():
-    return 46
+    return 92
```""",
        "python -c \"from app.calc import compute; assert compute() == 92\"",
    )

    assert result.passed is True
    patch_calls = [(cmd, timeout) for cmd, timeout in calls if isinstance(cmd, list) and cmd and cmd[0] == "patch"]
    assert patch_calls, calls
    assert patch_calls[0][1] >= 4.0
    assert patch_calls[0][1] != 2

    test_calls = [(cmd, timeout) for cmd, timeout in calls if isinstance(cmd, str)]
    assert test_calls, calls
    assert test_calls[-1][1] == 4.0


def test_global_system_prompt_did_not_gain_unknown_instruction():
    import eval.master_4000_runtime as runtime

    assert "unknown" not in runtime.SYSTEM_PROMPT.lower()
