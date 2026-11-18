"""Execute the publication condition against branch, test and release outcomes."""
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import yaml


WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/ci-build-release.yml"


def release_allowed(event, ref, publish=False, *, reuse=False, failure=None):
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    required = jobs["build-and-release"]["needs"]
    results = {name: SimpleNamespace(result="failure" if name == failure else "success") for name in required}
    results["preflight"].outputs = SimpleNamespace(reuse_green="true" if reuse else "false")
    expression = jobs["build-and-release"]["if"].replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"needs\.([\w-]+)", lambda m: f'needs["{m[1]}"]', expression)
    return eval("(" + expression + ")", {"__builtins__": {}}, {
        "always": lambda: True, "startsWith": lambda value, prefix: value.startswith(prefix),
        "needs": results, "github": SimpleNamespace(event_name=event, ref=ref),
        "inputs": SimpleNamespace(publish_release=publish),
    })


@pytest.mark.parametrize("publish", [False, True])
def test_feature_ci_and_prs_cannot_publish_a_release(publish):
    assert not release_allowed("workflow_dispatch", "refs/heads/feature/persistent-context-learning", publish)
    assert not release_allowed("pull_request", "refs/pull/100/merge", publish)


def test_default_branch_auto_release_survives_and_manual_release_is_explicit():
    assert release_allowed("push", "refs/heads/master")
    assert release_allowed("push", "refs/tags/v1.2.3")
    assert not release_allowed("workflow_dispatch", "refs/heads/master")
    assert release_allowed("workflow_dispatch", "refs/heads/master", True)


@pytest.mark.parametrize("failure", ["test", "mlx-learning", "preflight", "package"])
def test_failed_required_learning_or_packaging_gate_prevents_release(failure):
    assert not release_allowed("workflow_dispatch", "refs/heads/master", True, failure=failure)


def test_exact_green_merge_reuse_is_preserved():
    assert release_allowed("push", "refs/heads/master", reuse=True, failure="test")
    assert not release_allowed("push", "refs/heads/master", reuse=True, failure="package")


def test_native_learning_lanes_fail_on_errors_and_require_both_devices():
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    mlx = jobs["mlx-learning"]
    assert {lane["device"] for lane in mlx["strategy"]["matrix"]["include"]} == {"cpu", "metal"}
    assert not mlx.get("continue-on-error")
    assert all(not step.get("continue-on-error") for step in mlx["steps"])
    commands = "\n".join(step.get("run", "") for step in mlx["steps"])
    assert "assert mx.metal.is_available()" in commands
    assert "test_mlx_packed_terminal.py" in commands
