"""The fast release gate is bounded, offline, and leaves full tests available."""
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import yaml

from tools.run_ci_tests import ROOT, selected_tests, verify_desktop_report


def test_default_gate_selects_regressions_without_collecting_the_entire_suite():
    selected = selected_tests("regression")
    assert 20 <= len(selected) <= 46
    assert all((ROOT / path).is_file() for path in selected)
    assert "tests/unit/test_desktop_app.py" in selected
    assert "tests/unit/test_best_work_tournament.py" in selected
    assert "tests/unit/test_native_speculation.py" in selected
    assert "tests/unit/test_original_user_behaviors.py" in selected
    assert "tests/unit/test_persistent_context_memory.py" in selected
    assert "tests/unit/test_dependency_bootstrap_contract.py" in selected
    assert all(f"tests/unit/test_{name}_runtime.py" in selected
               for name in ("gguf_image", "wan_gguf", "mlx_video"))
    for pattern in ("test_local_api*.py", "test_bonsai25*.py"):
        assert all(str(path.relative_to(ROOT)) in selected for path in (ROOT / "tests/unit").glob(pattern))
    assert selected_tests("regression", full=True) == ["tests/unit", "tests/live"]
    assert set(selected_tests("native")) < set(selected_tests("native", full=True))


def test_ci_keeps_full_matrix_opt_in_and_all_native_package_release_gates():
    workflow = (ROOT / ".github/workflows/ci-build-release.yml").read_text()
    data = yaml.safe_load(workflow)
    matrix = data["jobs"]["test"]["strategy"]["matrix"]
    assert "inputs.full_tests" in matrix
    assert '"os":["ubuntu-latest"]' in matrix
    assert '"python-version":["3.11"]' in matrix
    assert '"os":["ubuntu-latest","macos-latest","windows-latest"]' in matrix
    assert '"python-version":["3.10","3.11"]' in matrix
    assert '"exclude":[{"os":"macos-latest","python-version":"3.10"}]' in matrix
    assert '"include":[{"os":"macos-latest","python-version":"3.12"}]' in matrix
    assert 'HF_HUB_OFFLINE: "1"' in workflow
    assert 'TRANSFORMERS_OFFLINE: "1"' in workflow
    assert "build_bonsai2_bundled_zip.py" not in workflow
    assert "HF_TOKEN:" not in workflow
    assert "needs.package.result == 'success'" in workflow
    assert "needs.mlx-learning.result == 'success'" in workflow
    lanes = data["jobs"]["mlx-learning"]["strategy"]["matrix"]["include"]
    assert {(lane["os"], lane["device"]) for lane in lanes} == {
        ("ubuntu-latest", "cpu"), ("macos-latest", "metal"), ("windows-latest", "cpu"),
    }
    assert not (ROOT / ".github/workflows/cross-platform-mlx-windows.yml").exists()
    verification = data["jobs"]["verify-packages"]
    assert verification["permissions"] == {"contents": "read", "actions": "read"}
    assert verification["if"] == "github.event_name == 'workflow_dispatch' && inputs.verify_packages"
    commands = "\n".join(step.get("run", "") for step in verification["steps"])
    assert "tools/verify_release_packages.py" in commands
    assert "tools.build_macos_full_installer" in commands
    assert "--include-installer --manifest" in commands
    assert "gh release" not in commands


@pytest.mark.parametrize("fault", ["missing", "skipped"])
def test_headless_or_skipped_desktop_smoke_cannot_pass_the_small_gate(tmp_path, fault):
    root = ET.Element("testsuite")
    if fault == "skipped":
        case = ET.SubElement(root, "testcase", name="test_chatbot_ui_initialization")
        ET.SubElement(case, "skipped")
    report = tmp_path / "tests.xml"
    ET.ElementTree(root).write(report)
    with pytest.raises(RuntimeError, match="Desktop smoke did not execute"):
        verify_desktop_report(report)
