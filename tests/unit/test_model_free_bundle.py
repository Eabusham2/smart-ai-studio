"""Exercise model-free app assembly without allowing any weight downloads."""
import io
import json
import os
import subprocess
import sys
import zipfile

import pytest

import build_bonsai2_bundled_zip as bundler
from tools.build_macos_full_installer import build as build_installer
from tools.build_macos_noadmin_portable import (
    ARCHIVE, ASSEMBLER, LAUNCHER, MANIFEST, build as build_kit, verify_manifest,
)


def test_model_free_archive_and_both_installers_need_no_hub_access(tmp_path, monkeypatch):
    app = tmp_path / "SmartAI.app"
    code = app / "Contents/Resources/app"
    python = code / "runtime/python/bin/python3"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    (code / "runtime/python/.smartai-deps-ready").write_text("test-runtime\n")
    (code / "app_gui.py").write_text("# synthetic app\n")
    output = tmp_path / "release"
    monkeypatch.setattr(bundler, "_repo_files", lambda *a, **k: pytest.fail("No model metadata in CI"))
    monkeypatch.setattr(bundler, "_stream_remote_file", lambda *a, **k: pytest.fail("No model weights in CI"))
    monkeypatch.setattr(sys, "argv", ["bundle", "--app-bundle", str(app),
                                    "--output-dir", str(output), "--part-bytes", "400", "--no-models"])
    bundler.main()
    manifest = json.loads((output / MANIFEST).read_text())
    assert manifest["models"] == []
    assert manifest["model_downloads"] == "on-demand"
    verify_manifest(manifest)
    payload = b"".join((output / part["name"]).read_bytes() for part in manifest["parts"])
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert archive.testzip() is None
        assert not any(name.endswith((".safetensors", ".gguf")) for name in archive.namelist())
        policy = json.loads(archive.read("SmartAI.app/Contents/Resources/app/preloaded_models/manifest.json"))
        assert policy["model_downloads"] == "on-demand" and policy["models"] == []
    values = {"repo": "Eabusham2/smart-ai-studio", "tag": "latest", "commit": "2" * 40}
    kit = build_kit(output / MANIFEST, output / ASSEMBLER, output, **values)
    installer = build_installer(kit, output, **values)
    with zipfile.ZipFile(kit) as archive:
        launcher = archive.read(LAUNCHER).decode()
        readme = archive.read("README-NoAdmin-Portable.txt").decode()
    assert 'BUNDLED_MODELS="0"' in launcher
    assert 'BUNDLED_MODELS="0"' in installer.read_text()
    assert "No model weights are included" in readme
    if os.name == "posix":
        subprocess.run(["/bin/bash", "-n"], input=launcher.encode(), check=True)
        subprocess.run(["/bin/bash", "-n", str(installer)], check=True)


def test_empty_or_conflicting_model_manifest_must_be_explicit(tmp_path):
    data = {"archive_name": ARCHIVE, "models": []}
    with pytest.raises(ValueError, match="both original Bonsai"):
        verify_manifest(data)
    data.update(model_downloads="on-demand", models=[{"repo_id": "unexpected/model"}])
    with pytest.raises(ValueError, match="must not contain model snapshots"):
        verify_manifest(data)
