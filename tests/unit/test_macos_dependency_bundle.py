"""The Mac package carries usable, explicit setup and launch entry points."""
import builtins
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
from types import SimpleNamespace
import zipfile

import pytest

import build_app


ROOT = Path(__file__).resolve().parents[2]


def test_stock_gguf_dependency_is_available_to_apple_source_installs():
    requirements = (ROOT / "requirements.txt").read_text()
    project = (ROOT / "pyproject.toml").read_text()
    assert "llama-cpp-python>=0.3.0\n" in requirements
    assert '"llama-cpp-python>=0.3.0",' in project


@pytest.mark.parametrize("default_encoding", ["utf-8", "cp1252"])
def test_macos_bundle_carries_manifests_explicit_installer_and_correct_launch_path(tmp_path, monkeypatch, default_encoding):
    source = tmp_path / "source with spaces"
    source.mkdir()
    for name in ("app_gui.py", "requirements.txt", "pyproject.toml"):
        (source / name).write_text("# fixture " + name)
    monkeypatch.chdir(source)
    monkeypatch.delenv("SMARTAI_MODEL_BUNDLE_RELEASE", raising=False)
    def host_open(path, mode="r", *args, **kwargs):
        # Reproduce Windows' legacy default code page on every CI platform.
        # Explicit encodings must win over the host's default.
        if "b" not in mode and kwargs.get("encoding") is None:
            kwargs["encoding"] = default_encoding
        return builtins.open(path, mode, *args, **kwargs)
    monkeypatch.setattr(build_app, "open", host_open, raising=False)
    monkeypatch.setattr(build_app.shutil, "which", lambda name: name if name in {"osacompile", "hdiutil"} else None)
    native_commands = []

    def native(command, **kwargs):
        native_commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(build_app.subprocess, "run", native)
    dist = tmp_path / "dist with spaces"
    build_app.create_macos_bundle(str(dist), "SmartAI")
    app = dist / "SmartAI.app"
    code = app / "Contents/Resources/app"
    assert (code / "requirements.txt").read_text() == "# fixture requirements.txt"
    assert (code / "pyproject.toml").read_text() == "# fixture pyproject.toml"
    installer = (code / "Install_Dependencies.command").read_text()
    launcher = (code / "launch_studio.sh").read_text()
    assert "-m venv" in installer and "-r \"$SMARTAI_CODE_DIR/requirements.txt\"" in installer
    assert "--no-binary=llama-cpp-python" in installer and "-DGGML_METAL=on" in installer
    assert "pip install" not in launcher and "-m venv" not in launcher
    assert '"$SMARTAI_VENV/bin/python"' in launcher
    assert '"$SMARTAI_CODE_DIR/app_gui.py" "$@"' in launcher
    applescript = next(command[-1] for command in native_commands if command[0] == "osacompile")
    assert 'quoted form of (resDir & "/launch_studio.sh")' in applescript
    with (app / "Contents/Info.plist").open("rb") as handle:
        plist = plistlib.load(handle)
    assert plist["CFBundleExecutable"] == "applet"
    assert plist["NSHumanReadableCopyright"].startswith("Copyright © 2026")
    with zipfile.ZipFile(dist / "SmartAI-macOS-arm64.zip") as archive:
        for name, target in (("Launch_SmartAI_macOS.command", "launch_studio.sh"),
                             ("Install_Dependencies.command", "Install_Dependencies.command")):
            wrapper = archive.read(name).decode()
            assert f'SmartAI.app/Contents/Resources/app/{target}" "$@"' in wrapper


@pytest.mark.skipif(os.name == "nt", reason="Executes the POSIX shell entry point shipped only in the Mac package")
def test_mac_launcher_uses_installed_environment_and_preserves_paths_and_arguments(tmp_path):
    code = tmp_path / "app with spaces"
    code.mkdir()
    build_app._write_macos_runtime_scripts(str(code))
    home = tmp_path / "user with spaces"
    python = home / "Library/Application Support/SmartAI/python/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/usr/bin/env bash\nif [ "$1" = -c ]; then exit 0; fi\nprintf "%s\\n" "$PWD" "$@"\n')
    python.chmod(0o755)
    bash = shutil.which("bash")
    assert bash
    for name in ("launch_studio.sh", "Install_Dependencies.command"):
        subprocess.run([bash, "-n", str(code / name)], check=True, capture_output=True, text=True)
    result = subprocess.run([bash, str(code / "launch_studio.sh"), "an argument with spaces", "literal;$text"],
                            env={**os.environ, "HOME": str(home)}, check=True, capture_output=True, text=True)
    assert result.stdout.splitlines() == [str(code), str(code / "app_gui.py"), "an argument with spaces", "literal;$text"]
