"""Offline checks for the separately installed packed image runtime."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from config.media_models import media_download_metadata
from core import bonsai_image_runtime as runtime

pytestmark = pytest.mark.skipif(os.name == "nt", reason="The Mac image adapter uses POSIX runtime/process isolation.")


@pytest.fixture
def publisher_host(monkeypatch):
    monkeypatch.setattr(runtime, "_require_platform", lambda: None)
    monkeypatch.setattr(runtime, "sys", SimpleNamespace(executable=sys.executable, version_info=(3, 11, 0)))


def test_runtime_install_uses_its_private_python_and_exact_publisher_sources(tmp_path, monkeypatch, publisher_host):
    directory = tmp_path / "app data with spaces" / "native image"
    monkeypatch.setattr(runtime, "_runtime_directory", lambda: directory)
    monkeypatch.setenv("PYTHONHOME", "/parent/python")
    monkeypatch.setenv("PYTHONPATH", "/parent/site-packages")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "shared hub"))
    calls = []

    def command(argv, **options):
        calls.append((argv, options))
        assert "PYTHONHOME" not in options["env"]
        assert "PYTHONPATH" not in options["env"]
        assert options["env"]["HF_HUB_CACHE"] == str(tmp_path / "shared hub")
        if "venv" in argv:
            python = directory / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("fixture private interpreter")

    monkeypatch.setattr(runtime, "_run_command", command)
    python = runtime.ensure_bonsai_image_runtime()
    assert python == directory / "bin" / "python"
    assert calls[0][0] == [sys.executable, "-I", "-m", "venv", str(directory)]
    install = calls[1][0]
    assert install[:4] == [str(python), "-I", "-m", "pip"]
    assert "--isolated" in install and "--user" not in install
    assert "--system-site-packages" not in str(calls)
    assert f"mlx=={runtime.MLX_VERSION}" in install
    for name, url, digest in runtime.PUBLISHER_SOURCES:
        assert f"{name} @ {url}#sha256={digest}" in install
    assert "--native-probe" in calls[2][0]
    assert calls[2][1]["env"]["HF_HUB_OFFLINE"] == "1"
    assert calls[2][1]["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert json.loads((directory / "smartai-runtime-ready.json").read_text()) == runtime._runtime_contract()
    calls.clear()
    assert runtime.ensure_bonsai_image_runtime() == python
    assert calls == []


def test_runtime_never_replaces_an_unrecognized_directory(tmp_path, monkeypatch, publisher_host):
    directory = tmp_path / "existing files"
    directory.mkdir()
    original = directory / "keep.txt"
    original.write_text("user data")
    monkeypatch.setattr(runtime, "_runtime_directory", lambda: directory)
    with pytest.raises(RuntimeError, match="unrecognized files"):
        runtime.ensure_bonsai_image_runtime()
    assert original.read_text() == "user data"


def test_failed_native_probe_never_marks_runtime_ready(tmp_path, monkeypatch, publisher_host):
    monkeypatch.setattr(runtime, "_runtime_directory", lambda: tmp_path / "private")

    def command(argv, **_options):
        if "venv" in argv:
            python = Path(argv[-1]) / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("fixture")
        if "--native-probe" in argv:
            raise RuntimeError("Publisher native API mismatch")

    monkeypatch.setattr(runtime, "_run_command", command)
    with pytest.raises(RuntimeError, match="API mismatch"):
        runtime.ensure_bonsai_image_runtime()
    assert not (tmp_path / "private" / "smartai-runtime-ready.json").exists()


def test_cancel_stops_the_owned_child_process(tmp_path, monkeypatch):
    event = threading.Event()
    spawned = []
    original = subprocess.Popen

    def popen(*args, **kwargs):
        process = original(*args, **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(runtime.subprocess, "Popen", popen)
    timer = threading.Timer(0.15, event.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(runtime.BonsaiImageCancelled):
            runtime._run_command(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                env=runtime._child_env(), label="Fixture generation", cancel_event=event,
            )
    finally:
        timer.cancel()
    assert time.monotonic() - started < 5
    assert len(spawned) == 1 and spawned[0].poll() is not None
    assert runtime.owned_process_ids() == []


def test_wrong_model_or_platform_cannot_start_downloads_or_setup(tmp_path, monkeypatch):
    from core import hf_downloader

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Invalid selection reached model download")

    monkeypatch.setattr(hf_downloader, "download_model_from_hf", forbidden)
    monkeypatch.setattr(runtime, "ensure_bonsai_image_runtime", forbidden)
    info = {"repo_id": runtime.MODEL_REPO, **media_download_metadata(runtime.MODEL_REPO)}
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    result = runtime.generate_bonsai_image(info, "a tree", tmp_path / "out.png")
    assert result["status"] == "error" and "Apple Silicon" in result["error"]
    monkeypatch.setattr(runtime, "_require_platform", lambda: None)
    result = runtime.generate_bonsai_image({**info, "repo_revision": "0" * 40}, "a tree", tmp_path / "out.png")
    assert result["status"] == "error" and "exact selected" in result["error"]
    result = runtime.generate_bonsai_image({**info, "width": 255}, "a tree", tmp_path / "out.png")
    assert result["status"] == "error" and "dimensions" in result["error"]
