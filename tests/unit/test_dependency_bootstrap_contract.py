"""Source-only packaging checks; no network, pip installation, or model inference."""
import subprocess
import os
import shutil
from pathlib import Path
import sys

import pytest

import build_app


def _bash():
    if os.name == 'nt':
        # System32/bash.exe starts WSL; it is not Git Bash and may have no Linux
        # distribution. Use the native Bash shipped with the runner's Git.
        for base in (os.environ.get('ProgramFiles'), os.environ.get('ProgramFiles(x86)')):
            if base:
                candidate = Path(base) / 'Git/bin/bash.exe'
                if candidate.is_file():
                    return str(candidate)
        git = shutil.which('git')
        if git:
            candidate = Path(git).parent.parent / 'bin/bash.exe'
            if candidate.is_file():
                return str(candidate)
        raise AssertionError('A native Git Bash is required for shell syntax checks')
    executable = shutil.which('bash')
    assert executable
    return executable


def test_first_launch_installers_generated_for_all_formats(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("app_gui.py", "requirements.txt", "pyproject.toml"):
        (tmp_path / name).write_text("# fixture\n", encoding="utf-8")
    dist = tmp_path / "dist"
    dist.mkdir()
    build_app.create_windows_bundle(str(dist), "SmartAI")
    build_app.create_linux_bundle(str(dist), "SmartAI")
    build_app.create_macos_bundle(str(dist), "SmartAI")

    bat = (dist / "SmartAI-Windows" / "SmartAI.bat").read_text(encoding='utf-8')
    vbs = (dist / "SmartAI-Windows" / "SmartAI.vbs").read_text(encoding='utf-8')
    assert "Python.Python.3.12" in bat
    assert '-m venv "%SMARTAI_ENV%"' in bat
    assert "-m pip install -r" in bat
    assert ".smartai-deps-ready" in bat
    assert "SMARTAI_REQ_HASH" in bat
    assert "WindowStyle = 1" in vbs and "FileExists(Ready)" in vbs

    linux = dist / "SmartAI-Linux" / "SmartAI.AppDir" / "AppRun"
    source = linux.read_text(encoding='utf-8')
    assert "python3 -m venv" in source
    assert "python3-tk" in source
    assert '"$SMARTAI_REQ_HASH" != "$SMARTAI_READY_HASH"' in source
    subprocess.run([_bash(), "-n"], input=linux.read_bytes(), check=True)

    mac = dist / "SmartAI.app" / "Contents" / "Resources" / "app"
    launch = (mac / "launch_studio.sh").read_text(encoding='utf-8')
    setup = (mac / "Install_Dependencies.command").read_text(encoding='utf-8')
    assert "vendor/python3.10" in launch
    assert 'SMARTAI_READY_HASH' in launch
    assert "SMARTAI_REQ_HASH" in setup
    assert "-m pip install" in setup
    subprocess.run([_bash(), "-n"], input=(mac / "launch_studio.sh").read_bytes(), check=True)
    subprocess.run([_bash(), "-n"], input=(mac / "Install_Dependencies.command").read_bytes(), check=True)


@pytest.mark.parametrize(("builder", "layout"), [
    (build_app.create_macos_bundle, "SmartAI.app/Contents/Resources/app"),
    (build_app.create_windows_bundle, "SmartAI-Windows"),
    (build_app.create_linux_bundle, "SmartAI-Linux/SmartAI.AppDir/usr/bin"),
])
def test_relocated_bundle_can_import_mlx_helpers_and_run_setup_cli(tmp_path, monkeypatch, builder, layout):
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    monkeypatch.setenv("SMARTAI_PACKAGE_ASSEMBLY_ONLY", "1")
    monkeypatch.delenv("SMARTAI_EMBEDDED_RUNTIME", raising=False)
    dist = tmp_path / "build"
    dist.mkdir()
    builder(str(dist), "SmartAI")
    code = dist.rename(tmp_path / "relocated package") / layout

    # Ignore the checkout, PYTHONPATH and installed packages. These two
    # bootstrap helpers must import even before pip has installed MLX.
    isolated = [sys.executable, "-I", "-S"]
    script = """
import sys
sys.path.insert(0, sys.argv[1])
from mlx_import_compat import load_mlx_lm
from mlx_setup import install
assert callable(load_mlx_lm) and callable(install)
"""
    subprocess.run([*isolated, "-c", script, str(code)], cwd=tmp_path, check=True)
    result = subprocess.run([*isolated, str(code / "tools/setup_mlx.py"), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert "--check" in result.stdout


def test_release_workflow_packages_python_and_checks_native_launch_before_publication():
    workflow = (Path(__file__).resolve().parents[2] /
                ".github/workflows/ci-build-release.yml").read_text(encoding="utf-8")
    assert "Build bundled runtime and test relocated first launch" in workflow
    assert "python tools/build_release_package.py" in workflow
    assert "needs.package.result == 'success'" in workflow
    assert workflow.index("Verify the canonical portable and setup packages") < workflow.index("Build one Mac installer using the portable app archive")
