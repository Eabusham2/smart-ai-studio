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

import bootstrap_runtime
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
    monkeypatch.delenv("SMARTAI_EMBEDDED_RUNTIME", raising=False)
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
    assert "vendor/python3.10" in launcher and "Install_Dependencies.command" in launcher
    assert '"$SMARTAI_VENV/bin/python"' in launcher
    assert '"$SMARTAI_CODE_DIR/app_gui.py" "$@"' in launcher
    applescript = next(command[-1] for command in native_commands if command[0] == "osacompile")
    assert 'quoted form of (resDir & "/launch_studio.sh")' in applescript
    with (app / "Contents/Info.plist").open("rb") as handle:
        plist = plistlib.load(handle)
    assert plist["CFBundleExecutable"] == "applet"
    assert plist["CFBundleName"] == plist["CFBundleDisplayName"] == "Smart AI Studio"
    assert "SmartAINativeLauncher" not in plist  # This fixture is a source-only bundle.
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
    (python.parent.parent / ".smartai-deps-ready").touch()  # Already-installed environment
    bash = shutil.which("bash")
    assert bash
    for name in ("launch_studio.sh", "Install_Dependencies.command"):
        subprocess.run([bash, "-n", str(code / name)], check=True, capture_output=True, text=True)
    result = subprocess.run([bash, str(code / "launch_studio.sh"), "an argument with spaces", "literal;$text"],
                            env={**os.environ, "HOME": str(home)}, check=True, capture_output=True, text=True)
    assert result.stdout.splitlines() == [str(code), str(code / "app_gui.py"), "an argument with spaces", "literal;$text"]


def test_embedded_mac_bundle_builds_native_launcher_with_the_pinned_runtime_headers(tmp_path, monkeypatch):
    source = tmp_path / "source with spaces"
    source.mkdir()
    for name in ("app_gui.py", "requirements.txt", "bootstrap_runtime.py", "package_smoke.py"):
        (source / name).write_text("# fixture " + name)
    runtime = tmp_path / "pinned runtime"
    header = runtime / "include/python3.11/Python.h"
    header.parent.mkdir(parents=True); header.write_text("/* fixture */")
    library = runtime / "lib/libpython3.11.dylib"
    library.parent.mkdir(); library.write_bytes(b"fixture CPython")
    python = runtime / "bin/python3"
    python.parent.mkdir(); python.write_bytes(b"fixture Python")
    monkeypatch.chdir(source)
    monkeypatch.setenv("SMARTAI_EMBEDDED_RUNTIME", str(runtime))
    monkeypatch.setenv("SMARTAI_PACKAGE_ASSEMBLY_ONLY", "1")
    monkeypatch.setattr(build_app.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(build_app.shutil, "which", lambda name: name)
    calls = []

    def native(command, **options):
        calls.append((command, options))
        if command[0] == "xcrun":
            Path(command[command.index("-o") + 1]).write_bytes(b"\xcf\xfa\xed\xfe" + b"fixture launcher")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(build_app.subprocess, "run", native)
    dist = tmp_path / "release with spaces"
    build_app.create_macos_bundle(str(dist), "SmartAI")
    app = dist / "SmartAI.app"
    info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    assert info["CFBundleExecutable"] == "applet"
    assert info["CFBundleName"] == info["CFBundleDisplayName"] == "Smart AI Studio"
    assert info["SmartAINativeLauncher"] == 1
    assert info["CFBundleIdentifier"] == "com.smartai.studio"
    assert os.access(app / "Contents/MacOS/applet", os.X_OK)
    command, options = next((command, options) for command, options in calls if command[0] == "xcrun")
    assert options["check"] is True
    assert command[command.index("-I") + 1] == str(app / "Contents/Resources/app/runtime/python/include/python3.11")
    assert '-DSMARTAI_PYTHON_LIBRARY="libpython3.11.dylib"' in command
    assert command[command.index("-o") + 1] == str(app / "Contents/MacOS/applet")
    assert not any(command[0] == "osacompile" for command, _ in calls)


@pytest.mark.parametrize("private_runtime", [False, True])
def test_mac_bootstrap_keeps_native_identity_on_first_second_and_update_launch(tmp_path, monkeypatch, private_runtime):
    app = tmp_path / "installed somewhere else" / "Renamed Studio.app"
    code = app / "Contents/Resources/app"
    code.mkdir(parents=True)
    launcher = app / "Contents/MacOS/applet"
    launcher.parent.mkdir(); launcher.write_bytes(b"fixture native executable"); launcher.chmod(0o755)
    (app / "Contents/Info.plist").write_bytes(plistlib.dumps({"SmartAINativeLauncher": 1}))
    selected = tmp_path / "private setup home/runtime" if private_runtime else code / "runtime/python"
    selected.mkdir(parents=True)
    monkeypatch.setattr(bootstrap_runtime, "__file__", str(code / "bootstrap_runtime.py"))
    monkeypatch.setattr(bootstrap_runtime.sys, "platform", "darwin")
    monkeypatch.setattr(bootstrap_runtime, "ensure_runtime", lambda root: selected)
    monkeypatch.setenv("PYTHONHOME", "/wrong python")
    monkeypatch.setenv("PYTHONPATH", "/wrong packages")
    monkeypatch.setenv("SMARTAI_BOOTSTRAP_PID", "stale-parent")
    calls = []
    monkeypatch.setattr(bootstrap_runtime.os, "execve", lambda executable, arguments, env: calls.append((executable, arguments, env)))
    monkeypatch.setattr(bootstrap_runtime.subprocess, "call", lambda *_a, **_k: pytest.fail("Mac GUI must not become a bare Python child"))
    # The updater invokes this same bootstrap directly with no LaunchServices
    # environment, so the handoff must derive the native path from the app.
    for args in (["first launch", "literal;$text"], ["second launch"], []):
        assert bootstrap_runtime.main(args) == 0
        executable, arguments, env = calls[-1]
        assert executable == str(launcher)
        assert arguments == [str(launcher), "--smartai-run-python", str(selected.resolve()), str(code / "app_gui.py"), *args]
        assert env["SMARTAI_BOOTSTRAP_PID"] == str(os.getpid())
        assert "PYTHONHOME" not in env and "PYTHONPATH" not in env
    assert bootstrap_runtime.main(["--smoke-test"]) == 0
    assert calls[-1][1][-1] == str(code / "package_smoke.py")


def test_mac_source_bundle_fallback_does_not_reenter_an_applescript_applet(tmp_path, monkeypatch):
    code = tmp_path / "Source.app/Contents/Resources/app"
    code.mkdir(parents=True)
    (code.parent.parent / "Info.plist").write_bytes(plistlib.dumps({"CFBundleExecutable": "applet"}))
    monkeypatch.setattr(bootstrap_runtime.sys, "platform", "darwin")
    assert bootstrap_runtime.macos_native_launcher(code) is None
