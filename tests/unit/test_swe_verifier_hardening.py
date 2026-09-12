"""Regression tests for DeepSWE patch verifier hardening."""

from types import SimpleNamespace

import eval.swe_verifier_hardening as hardening
import master_4000_eval_suite  # installs the hardening layer
import run_studio_complete as compat


def _run_with_fake_subprocess(monkeypatch, patch_text):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs.get("timeout")))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(hardening.subprocess, "run", fake_run)
    sandbox = compat.POSIXHardenedSandbox(timeout_sec=4.0, max_memory_mb=512)
    result = sandbox.verify_git_diff_patch(
        {"app/calc.py": "def compute():\n    return 10\n"},
        patch_text,
        "python -c \"from app.calc import compute; assert compute() == 20\"",
    )
    return result, calls


def test_plain_model_diff_uses_p0_and_configured_timeout(monkeypatch):
    result, calls = _run_with_fake_subprocess(
        monkeypatch,
        """```diff
--- app/calc.py
+++ app/calc.py
@@ -1,2 +1,2 @@
 def compute():
-    return 10
+    return 20
```""",
    )
    assert result.passed is True
    patch_calls = [(cmd, timeout) for cmd, timeout in calls if isinstance(cmd, list) and cmd and cmd[0] == "patch"]
    assert patch_calls
    assert any("-p0" in cmd for cmd, _ in patch_calls)
    assert all(timeout >= 4.0 for _, timeout in patch_calls)
    assert any("--dry-run" in cmd for cmd, _ in patch_calls)


def test_git_style_diff_uses_p1(monkeypatch):
    result, calls = _run_with_fake_subprocess(
        monkeypatch,
        """```diff
--- a/app/calc.py
+++ b/app/calc.py
@@ -1,2 +1,2 @@
 def compute():
-    return 10
+    return 20
```""",
    )
    assert result.passed is True
    patch_calls = [(cmd, timeout) for cmd, timeout in calls if isinstance(cmd, list) and cmd and cmd[0] == "patch"]
    assert patch_calls
    assert any("-p1" in cmd for cmd, _ in patch_calls)
    assert all(timeout >= 4.0 for _, timeout in patch_calls)


def test_global_system_prompt_did_not_gain_unknown_instruction():
    import eval.master_4000_runtime as runtime

    assert "unknown" not in runtime.SYSTEM_PROMPT.lower()
