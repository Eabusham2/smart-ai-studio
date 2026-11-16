"""Source-only packaging checks; no network, pip installation, or model inference."""
import subprocess
from pathlib import Path

import build_app


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
    subprocess.run(["sh", "-n"], input=linux.read_bytes(), check=True)

    mac = dist / "SmartAI.app" / "Contents" / "Resources" / "app"
    launch = (mac / "launch_studio.sh").read_text(encoding='utf-8')
    setup = (mac / "Install_Dependencies.command").read_text(encoding='utf-8')
    assert "vendor/python3.10" in launch
    assert 'SMARTAI_READY_HASH' in launch
    assert "SMARTAI_REQ_HASH" in setup
    assert "-m pip install" in setup
    subprocess.run(["bash", "-n"], input=(mac / "launch_studio.sh").read_bytes(), check=True)
    subprocess.run(["bash", "-n"], input=(mac / "Install_Dependencies.command").read_bytes(), check=True)


def test_release_workflow_packages_python_and_checks_native_launch_before_publication():
    workflow = (Path(__file__).resolve().parents[2] /
                ".github/workflows/ci-build-release.yml").read_text(encoding="utf-8")
    assert "Build bundled runtime and test relocated first launch" in workflow
    assert "python tools/build_release_package.py" in workflow
    assert "needs.package.result == 'success'" in workflow
    assert workflow.index("Verify package receipts") < workflow.index("Stream Both Bonsai 2 MLX Models into macOS Release ZIP")
