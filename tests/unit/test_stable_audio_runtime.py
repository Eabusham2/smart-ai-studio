"""Private native audio setup and real WAV boundaries, without model weights."""
import json
import os
from pathlib import Path
import struct
import sys
import threading
import time
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from config.media_models import media_download_metadata
from core import stable_audio_runtime as runtime


def _write_wave(path, seconds=1.0, value=8192):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(struct.pack("<hh", value, value) * int(8000 * seconds))


def _snapshot(root):
    root.mkdir(parents=True)
    (root / "t5gemma-b-b-ul2").mkdir()
    config = {"sample_rate": 8000, "model": {"conditioning": {"configs": [
        {"type": "t5gemma", "config": {"repo_id": "stabilityai/original", "subfolder": "t5gemma-b-b-ul2"}},
    ]}}}
    (root / "model_config.json").write_text(json.dumps(config))
    (root / "model.safetensors").write_bytes(b"fixture weights")
    (root / "t5gemma-b-b-ul2/model.safetensors").write_bytes(b"fixture conditioner")
    return config


def test_cpu_probe_has_separate_identity_and_private_dependency_install(tmp_path, monkeypatch):
    monkeypatch.setenv("SMARTAI_DATA_DIR", str(tmp_path))
    normal, cpu = runtime._runtime_directory(), runtime._runtime_directory(True)
    assert normal != cpu
    monkeypatch.setenv("PYTHONHOME", "/app/python")
    monkeypatch.setenv("PYTHONPATH", "/app/site-packages")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "shared-model-cache"))
    calls = []
    def command(argv, **options):
        calls.append((argv, options))
        assert "PYTHONHOME" not in options["env"] and "PYTHONPATH" not in options["env"]
        assert options["env"]["HF_HUB_CACHE"] == str(tmp_path / "shared-model-cache")
        if "venv" in argv:
            python = runtime._python_path(Path(argv[-1]))
            python.parent.mkdir(parents=True)
            python.write_text("private test interpreter")
    monkeypatch.setattr(runtime, "_run_command", command)
    python = runtime.ensure_stable_audio_runtime(cpu_probe=True)
    assert python == runtime._python_path(cpu)
    assert calls[0][0] == [sys.executable, "-I", "-m", "venv", str(cpu)]
    cpu_install = calls[1][0]
    assert cpu_install[:4] == [str(python), "-I", "-m", "pip"]
    assert "https://download.pytorch.org/whl/cpu" in cpu_install
    assert "--system-site-packages" not in str(calls) and "--user" not in str(calls)
    install = calls[2][0]
    assert f"transformers=={runtime.TRANSFORMERS_VERSION}" in install
    assert f"stable-audio-3 @ {runtime.PUBLISHER_URL}#sha256={runtime.PUBLISHER_SHA256}" in install
    assert "--native-probe" in calls[3][0] and "--cpu-probe" in calls[3][0]
    assert calls[3][1]["env"]["HF_HUB_OFFLINE"] == "1"
    assert json.loads((cpu / "smartai-runtime-ready.json").read_text()) == runtime._runtime_contract(True)
    calls.clear()
    assert runtime.ensure_stable_audio_runtime(cpu_probe=True) == python and calls == []
    assert os.environ["PYTHONHOME"] == "/app/python" and os.environ["PYTHONPATH"] == "/app/site-packages"


@pytest.mark.parametrize("available,cuda_version,requested,expected", [
    (True, "12.8", "auto", "cu128"), (True, "12.6", "auto", "cu126"),
    (True, "12.1", "auto", "cu118"), (False, None, "auto", "cpu"),
    (True, "12.8", "cpu", "cpu"), (False, None, "cuda", "cu126"),
])
def test_private_windows_linux_torch_wheels_follow_the_selected_device(monkeypatch, available, cuda_version, requested, expected):
    monkeypatch.setattr(runtime.sys, "platform", "win32")
    monkeypatch.setenv("SMARTAI_AUDIO_DEVICE", requested)
    monkeypatch.setattr(runtime, "_nvidia_cuda_version", lambda: "")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: available), version=SimpleNamespace(cuda=cuda_version)))
    assert runtime._torch_index() == expected
    assert runtime._runtime_contract()["torch_index"] == expected
    assert "torch_index" not in runtime._runtime_contract(True)


def test_nvidia_driver_selects_cuda_even_when_the_app_torch_is_cpu_only(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.sys, "platform", "win32")
    monkeypatch.setenv("SMARTAI_AUDIO_DEVICE", "auto")
    monkeypatch.setenv("SMARTAI_DATA_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False), version=SimpleNamespace(cuda=None)))
    monkeypatch.setattr(runtime, "_nvidia_cuda_version", lambda: "12.8")
    assert runtime._torch_index() == "cu128"
    calls = []
    def command(argv, **options):
        calls.append(argv)
        if "venv" in argv:
            python = runtime._python_path(Path(argv[-1]))
            python.parent.mkdir(parents=True)
            python.write_text("private fixture interpreter")
    monkeypatch.setattr(runtime, "_run_command", command)
    runtime.ensure_stable_audio_runtime()
    assert "https://download.pytorch.org/whl/cu128" in calls[1]
    assert "--torch-index" in calls[-1] and calls[-1][-1] == "cu128"


def test_cuda_runtime_cannot_silently_use_a_cpu_wheel_or_fall_back_to_cpu():
    torch = SimpleNamespace(version=SimpleNamespace(cuda=None), cuda=SimpleNamespace(is_available=lambda: False),
                            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))
    with pytest.raises(RuntimeError, match="does not match"):
        runtime._check_torch_build(torch, "cu126")
    with pytest.raises(RuntimeError, match="unavailable"):
        runtime._device(torch, "auto", "cu126")


def test_failed_probe_never_marks_runtime_ready_or_changes_other_files(tmp_path, monkeypatch):
    monkeypatch.setenv("SMARTAI_DATA_DIR", str(tmp_path))
    directory = runtime._runtime_directory()
    def command(argv, **options):
        if "venv" in argv:
            python = runtime._python_path(Path(argv[-1]))
            python.parent.mkdir(parents=True)
            python.write_text("private test interpreter")
        if "--native-probe" in argv:
            raise RuntimeError("Publisher API mismatch")
    monkeypatch.setattr(runtime, "_run_command", command)
    with pytest.raises(RuntimeError, match="API mismatch"):
        runtime.ensure_stable_audio_runtime()
    assert not (directory / "smartai-runtime-ready.json").exists()
    (directory / "smartai-runtime-owner.json").unlink()
    keep = directory / "keep.txt"
    keep.write_text("user data")
    with pytest.raises(RuntimeError, match="unrecognized files"):
        runtime.ensure_stable_audio_runtime()
    assert keep.read_text() == "user data"


def test_cancel_stops_the_owned_child_cross_platform(monkeypatch):
    event = threading.Event()
    seen = []
    original = runtime.subprocess.Popen
    def popen(*args, **kwargs):
        process = original(*args, **kwargs)
        seen.append(process)
        return process
    monkeypatch.setattr(runtime.subprocess, "Popen", popen)
    timer = threading.Timer(0.2, event.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(runtime.StableAudioCancelled):
            runtime._run_command([sys.executable, "-c", "import time; time.sleep(30)"],
                                 env=runtime._child_env(), label="Fixture audio process", cancel_event=event)
    finally:
        timer.cancel()
    assert time.monotonic() - started < 8
    assert seen and seen[0].poll() is not None
    assert runtime.owned_process_ids() == []


def test_prepare_requires_original_pin_and_preserves_local_lora(tmp_path, monkeypatch):
    from core import hf_transfer
    repo = "stabilityai/stable-audio-3-small-music"
    info = {"repo_id": repo, **media_download_metadata(repo)}
    source = tmp_path / "selected-snapshot"
    for name in info["download_required_files"]:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"selected fixture")
    adapter = tmp_path / "learned.safetensors"
    adapter.write_bytes(b"learned fixture")
    setups, probes = [], []
    monkeypatch.setattr(runtime, "ensure_stable_audio_runtime",
                        lambda **kwargs: setups.append(kwargs) or tmp_path / "private-python")
    def snapshot(**kwargs):
        probes.append(kwargs)
        assert kwargs["repo_id"] == repo and kwargs["revision"] == info["repo_revision"]
        assert kwargs["local_files_only"] is True
        return str(source)
    monkeypatch.setattr(hf_transfer, "snapshot_download", snapshot)
    with pytest.raises(RuntimeError, match="exact selected"):
        runtime.prepare_stable_audio_model({**info, "repo_revision": "0" * 40})
    assert setups == [] and probes == []
    prepared = runtime.prepare_stable_audio_model(info, lora_files=[adapter])
    assert prepared["model_path"] == str(source.resolve())
    assert prepared["lora_files"] == [str(adapter.resolve())]
    assert prepared["revision"] == info["repo_revision"]
    assert probes[0]["allow_patterns"] == info["download_required_files"]
    assert not setups[0].get("cpu_probe")


def test_worker_uses_local_publisher_constructor_lora_and_real_sampler_steps(tmp_path, monkeypatch):
    root = tmp_path / "snapshot"
    original_config = _snapshot(root)
    calls, events = [], []
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))
    def load(config, checkpoint, **options):
        assert checkpoint == str(root / "model.safetensors")
        conditioner = config["model"]["conditioning"]["configs"][0]["config"]
        assert conditioner["model_path"] == str(root.resolve())
        assert conditioner["repo_id"] == str(root.resolve())
        calls.append(("load", options))
        return SimpleNamespace()
    class Tensor:
        def detach(self): return self
        def float(self): return self
        def cpu(self): return self
        def numpy(self): return np.full((1, 2, 8000), 0.25, dtype=np.float32)
    class Model:
        def __init__(self, *args): pass
        def load_lora(self, paths): calls.append(("lora", paths))
        def generate(self, **kwargs):
            calls.append(("generate", kwargs))
            kwargs["callback"]({"i": 0})
            kwargs["callback"]({"i": 1})
            return Tensor()
    def write(path, samples, rate, **kwargs):
        np.testing.assert_array_equal(samples, np.full((8000, 2), 0.25))
        assert rate == 8000 and kwargs == {"format": "WAV", "subtype": "PCM_16"}
        _write_wave(path)
    monkeypatch.setattr(runtime, "_native_components", lambda **kwargs: (torch, Model, load))
    monkeypatch.setattr(runtime, "_event", lambda value, message: events.append((value, message)))
    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(write=write))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    request = {"model_path": str(root), "required_files": ["model_config.json", "model.safetensors"],
               "device": "cpu", "lora_files": ["/local/learned.safetensors"], "steps": 2,
               "prompt": "music", "duration": 1.0, "cfg_scale": 1.0, "seed": 42,
               "output_path": str(tmp_path / "actual.wav")}
    runtime._native_worker(request)
    assert calls[0] == ("load", {"device": "cpu", "model_half": False})
    assert calls[1] == ("lora", request["lora_files"])
    assert [msg for _, msg in events if "denoising step" in msg] == [
        "Generating audio: denoising step 1/2", "Generating audio: denoising step 2/2"]
    assert runtime.inspect_waveform(request["output_path"])["samples"] == [0.25] * 2048
    assert json.loads((root / "model_config.json").read_text()) == original_config


@pytest.mark.parametrize("failure", [None, "truncated", "cancelled"])
def test_generation_publishes_only_valid_completed_audio(tmp_path, monkeypatch, failure):
    output = tmp_path / "result.wav"
    output.write_bytes(b"existing output")
    previews, event = [], threading.Event()
    def run(command, **options):
        assert options["env"]["HF_HUB_OFFLINE"] == "1"
        assert options["env"]["TRANSFORMERS_OFFLINE"] == "1"
        request = json.loads(Path(command[-1]).read_text())
        assert request["lora_files"] == ["/saved/adapter.safetensors"]
        _write_wave(request["output_path"])
        if failure == "truncated":
            path = Path(request["output_path"])
            path.write_bytes(path.read_bytes()[:-8])
        if failure == "cancelled":
            event.set()
    monkeypatch.setattr(runtime, "_run_command", run)
    prepared = {"python": sys.executable, "lora_files": ["/saved/adapter.safetensors"]}
    result = runtime.generate_stable_audio(prepared, "a tone", output, duration=1,
                                           preview_callback=previews.append, cancel_event=event)
    if failure:
        assert result["status"] == ("cancelled" if failure == "cancelled" else "error")
        assert output.read_bytes() == b"existing output" and previews == []
    else:
        assert result["status"] == "success" and result["path"] == str(output)
        assert previews[0]["samples"] == [0.25] * 2048
        assert previews[0]["duration_seconds"] == 1.0
        assert runtime.inspect_waveform(output)["sample_rate"] == 8000


def test_audio_engine_selects_private_runtime_without_importing_incompatible_dependencies(monkeypatch):
    from core.audio_engine import AudioGenerationEngine
    repo = "stabilityai/stable-audio-3-small-sfx"
    info = {"repo_id": repo, **media_download_metadata(repo)}
    event, calls = threading.Event(), []
    prepared = {"selected": "original native snapshot"}
    def prepare(selection, **kwargs):
        calls.append((selection, kwargs))
        return prepared
    monkeypatch.setattr(runtime, "prepare_stable_audio_model", prepare)
    monkeypatch.setattr(runtime, "generate_stable_audio", lambda source, *args, **kwargs:
                        {"status": "success", "source": source, "cancel": kwargs["cancel_event"]})
    engine = AudioGenerationEngine()
    monkeypatch.setattr(engine, "_saved_lora_files", lambda _: ["/saved/adapter.safetensors"])
    assert engine.load_model(info, cancel_event=event)["backend"] == "stable_audio_3_isolated"
    assert calls[0][1]["lora_files"] == ["/saved/adapter.safetensors"]
    result = engine.generate("a sound", cancel_event=event)
    assert result["source"] is prepared and result["cancel"] is event
