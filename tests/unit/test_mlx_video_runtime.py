"""Offline runtime/route regressions; no pretrained video weights are fetched."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
import wave

import imageio.v2 as imageio
import numpy as np
import pytest

from config.media_models import media_download_metadata
from core import mlx_video_runtime as runtime


@pytest.fixture
def selection(tmp_path, monkeypatch):
    import psutil
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=128 * 1024 ** 3))
    monkeypatch.setattr(runtime, "_require_platform", lambda: None)

    def create(kind):
        repo, _ = runtime.MODEL_CONTRACTS[kind]
        info = {"repo_id": repo, **media_download_metadata(repo)}
        info.update(width=128, height=128, frames=9 if kind == "ltx_mlx" else 22, fps=24)
        selections = [{"role": "model", "revision": info["repo_revision"],
                       "required_files": info["download_required_files"]}, *info.get("download_components", [])]
        roots = {}
        for entry in selections:
            root = tmp_path / "cached model files with spaces" / entry["role"] / entry["revision"]
            roots[entry["role"]] = str(root)
            for filename in entry["required_files"]:
                path = root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                if filename.endswith(".index.json"):
                    shards = [Path(name).name for name in entry["required_files"]
                              if name.endswith(".safetensors") and Path(name).parent == Path(filename).parent]
                    path.write_text(json.dumps({"weight_map": {f"tensor_{i}": name for i, name in enumerate(shards)}}))
                else:
                    path.write_bytes(b"{}" if filename.endswith(".json") else b"tiny selected fixture")
        info["path"] = info["model_path"] = roots.pop("model")
        info["component_paths"] = roots
        return info
    return create


@pytest.mark.skipif(os.name == "nt", reason="Apple Silicon uses POSIX private runtimes.")
@pytest.mark.parametrize("kind", ["ltx_mlx", "minimax_mlx"])
def test_private_setup_is_pinned_reusable_and_probe_failure_is_not_ready(tmp_path, monkeypatch, kind):
    directory = tmp_path / "private runtime with spaces"
    monkeypatch.setattr(runtime, "_require_platform", lambda: None)
    monkeypatch.setattr(runtime, "sys", SimpleNamespace(executable=sys.executable, version_info=(3, 11, 0)))
    monkeypatch.setattr(runtime, "_runtime_directory", lambda value: directory)
    monkeypatch.setenv("PYTHONPATH", "/unrelated/parent/site-packages")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "shared cache"))
    calls = []

    def source(path, selected, cancel):
        assert path == directory and selected == kind
        return path / "publisher"

    def run(argv, **options):
        calls.append((argv, options))
        assert "PYTHONPATH" not in options["env"]
        assert options["env"]["HF_HUB_CACHE"] == str(tmp_path / "shared cache")
        if "venv" in argv:
            python = directory / "bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("private Python fixture")
        if "--native-probe" in argv:
            assert options["env"]["HF_HUB_OFFLINE"] == options["env"]["TRANSFORMERS_OFFLINE"] == "1"

    monkeypatch.setattr(runtime, "_prepare_source", source)
    monkeypatch.setattr(runtime, "_run_command", run)
    python = runtime.ensure_mlx_video_runtime(kind)
    assert python == directory / "bin/python"
    assert set(runtime.DEPENDENCIES[kind]) <= set(calls[1][0])
    assert calls[1][0][:4] == [str(python), "-I", "-m", "pip"]
    assert calls[-1][0][-2:] == ["--runtime-dir", str(directory)]
    calls.clear()
    assert runtime.ensure_mlx_video_runtime(kind) == python and not calls

    def fail_probe(*args, **kwargs):
        raise RuntimeError("native dependency mismatch")
    monkeypatch.setattr(runtime, "_run_command", fail_probe)
    with pytest.raises(RuntimeError, match="dependency mismatch"):
        runtime.ensure_mlx_video_runtime(kind, force_probe=True)
    assert not (directory / "smartai-runtime-ready.json").exists()


def _video(path, *, frames=9, width=128, height=128, fps=24):
    audio = Path(path).with_suffix(".wav")
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(32000)
        stream.writeframes(np.zeros((round(frames / fps * 32000), 2), dtype="<i2").tobytes())
    imageio.mimwrite(path, [np.full((height, width, 3), index * 7 % 255, np.uint8)
                           for index in range(frames)], fps=fps, codec="libx264",
                    audio_path=str(audio), audio_codec="aac")
    audio.unlink()


def test_ltx_worker_uses_local_distilled_pipeline_audio_and_real_mp4(selection, tmp_path, monkeypatch):
    info = selection("ltx_mlx")
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in Path(info["path"]).rglob("*") if path.is_file()}
    python = tmp_path / "private runtime/bin/python"
    python.parent.mkdir(parents=True)
    calls, progress, previews = [], [], []

    class Pipeline:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def generate_and_save(self, **kwargs):
            assert self.generate_audio is True
            assert kwargs["stage1_steps"] == 8 and kwargs["stage2_steps"] == 3
            assert kwargs["num_frames"] == 9 and kwargs["frame_rate"] == 24
            # Exercise the real progress callback without fabricating a latent image.
            request = self.stepwise.request
            self.stepwise.request = {**request, "previews": False}
            self.stepwise.bind(stage=1)(0, 8, None, 1.0)
            _video(kwargs["output_path"])

    monkeypatch.setitem(sys.modules, "ltx_pipelines_mlx", SimpleNamespace(DistilledPipeline=Pipeline))
    monkeypatch.setitem(sys.modules, "ltx_pipelines_mlx.utils.generation",
                        SimpleNamespace(is_ltx25_pack=lambda path: path == info["path"]))
    monkeypatch.setattr(runtime, "ensure_mlx_video_runtime", lambda *args, **kwargs: python)
    monkeypatch.setattr(runtime, "_activate_runtime", lambda *args: None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.delenv("SMARTAI_MLX_VIDEO_OWNER_PID", raising=False)
    events = []
    monkeypatch.setattr(runtime, "_event", lambda value, message, preview=None: events.append((value, message)))

    def worker(command, **kwargs):
        request = json.loads(Path(command[command.index("--worker") + 1]).read_text())
        runtime._native_worker(request, python.parent.parent)
    monkeypatch.setattr(runtime, "_run_command", worker)
    output = tmp_path / "finished movie.mp4"
    result = runtime.generate_mlx_video(info, "a camera moves over a lake", output,
                                       progress_callback=lambda *args: progress.append(args),
                                       preview_callback=previews.append)
    assert result["status"] == "success", result
    assert calls == [{"model_dir": info["path"], "low_memory": True, "low_ram_streaming": True}]
    assert result["frames"] == 9 and result["fps"] == 24
    assert any("Denoising stage 1" in message for _, message in events)
    assert previews[-1]["kind"] == "video" and previews[-1]["image"].size == (128, 128)
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}


def test_minimax_routes_original_companions_lora_beta_schedules_and_audio(selection, tmp_path, monkeypatch):
    info = selection("minimax_mlx")
    request = runtime._request(info, tmp_path / "out.mp4", steps=3)
    request["prompt"] = "a fox walks along a river"
    loads, encoded, applied, saved, schedules = [], [], [], [], []
    payload = object()
    mx = SimpleNamespace(
        eval=lambda value: encoded.append(value), clear_cache=lambda: None,
        load=lambda path: loads.append(path) or {"selected_lora": payload})
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)

    class Encoder:
        def __init__(self, path, **kwargs):
            loads.append(str(path))
        def encode(self, prompt, images):
            assert prompt == request["prompt"] and images is None
            return payload, [1]

    class Scheduler:
        def __init__(self, shift):
            self.shift = shift
        def set_timesteps(self, *, sigmas):
            schedules.append((self.shift, sigmas))

    class Pipeline:
        def __init__(self, dit, encoder, video_vae, audio_vae, config):
            self.encoder, self.config = encoder, config
        def __call__(self, prompt, **kwargs):
            assert kwargs["num_inference_steps"] == 4 and kwargs["drop_adaln"] is True
            assert self.encoder.encode(prompt, kwargs["images"])[0] is payload
            self._build_schedules(kwargs["num_inference_steps"])
            return SimpleNamespace(video=np.zeros((22, 128, 128, 3), np.uint8),
                                   audio=np.ones((2, round(22 / 24 * 32000)), np.float32), fps=24, sample_rate=32000)

    def component(path):
        loads.append(str(path))
        if Path(path).name == "audio_vae":
            # The publisher's loader opens this before constructing the VAE.
            json.loads((Path(path) / "metadata.json").read_text())
        return payload
    monkeypatch.setitem(sys.modules, "minimax_h3_mlx.config", SimpleNamespace(PipelineConfig=SimpleNamespace(
        from_model_index=lambda path: SimpleNamespace(sigma_shift_video=12, sigma_shift_audio=3))))
    monkeypatch.setitem(sys.modules, "minimax_h3_mlx.load", SimpleNamespace(
        load_dit=component, load_video_vae=component, load_audio_vae=component))
    mux = runtime._save_minimax_mp4
    def save(*args):
        saved.append(args)
        mux(*args)
    monkeypatch.setattr(runtime, "_save_minimax_mp4", save)
    monkeypatch.setitem(sys.modules, "minimax_h3_mlx.pipeline", SimpleNamespace(MiniMaxH3Pipeline=Pipeline))
    monkeypatch.setitem(sys.modules, "minimax_h3_mlx.scheduler", SimpleNamespace(MiniMaxH3Scheduler=Scheduler))
    monkeypatch.setitem(sys.modules, "minimax_h3_mlx.text_encoder", SimpleNamespace(MiniMaxH3TextEncoder=Encoder))
    monkeypatch.setattr(runtime, "_attach_aftermidnight", lambda *args: applied.append(args) or 200)
    runtime._minimax_generate(request)
    root = Path(info["component_paths"]["upstream"]) / "Ref2VA"
    assert loads[0] == str(root / "text_encoder") and encoded == [payload]
    assert loads[1] == info["path"]
    assert loads[2] == str(Path(info["component_paths"]["lora"]) / runtime.AFTERMIDNIGHT_FILE)
    assert loads[3:] == [str(root / "video_vae"), str(root / "audio_vae")]
    assert applied == [(payload, {"selected_lora": payload}, 1.0)]
    assert [shift for shift, _ in schedules] == [12, 3]
    assert all(len(grid) == 4 and grid[0] == 1 and grid[-1] == 0 for _, grid in schedules)
    assert all(all(a > b for a, b in zip(grid, grid[1:])) for _, grid in schedules)
    assert saved[0][0] == request["output_path"] and saved[0][3].shape == (2, round(22 / 24 * 32000))
    # The upstream -shortest writer drops the final frame at this audio length.
    # Our actual codec path must retain all generated frames and its AAC stream.
    runtime._validate_video(request["output_path"], request)


def test_invalid_or_low_memory_request_never_downloads_or_replaces_output(selection, tmp_path, monkeypatch):
    import psutil
    from core import media_assets
    info = selection("minimax_mlx")
    calls = []
    monkeypatch.setattr(media_assets, "ensure_media_model_assets", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=12 * 1024 ** 3))
    result = runtime.generate_mlx_video(info, "a scene", tmp_path / "new.mp4")
    assert result["status"] == "error" and "available memory" in result["error"]
    result = runtime.generate_mlx_video({**info, "repo_revision": "0" * 40}, "a scene", tmp_path / "new.mp4")
    assert "exact selected" in result["error"]
    result = runtime.generate_mlx_video(info, "a scene", tmp_path / "new.mp4", height=432)
    assert "multiples of 32" in result["error"]
    original = tmp_path / "existing.mp4"
    original.write_bytes(b"existing user file")
    result = runtime.generate_mlx_video(info, "a scene", original)
    assert result["status"] == "error" and original.read_bytes() == b"existing user file"
    assert not calls


def test_lora_rejects_unknown_and_partial_targets_and_video_requires_all_frames(tmp_path):
    with pytest.raises(ValueError, match="Unsupported AfterMidnight"):
        runtime._lora_groups({"different_model.weight": object()})
    with pytest.raises(ValueError, match="incomplete tensor pair"):
        runtime._lora_groups({"lora_unet_blocks_0_attn_qkv_proj.alpha": np.array(64)})
    output = tmp_path / "incomplete.mp4"
    _video(output, frames=2)
    with pytest.raises(RuntimeError, match="complete requested frame sequence"):
        runtime._validate_video(output, {"width": 128, "height": 128, "fps": 24, "frames": 9})


@pytest.mark.skipif(os.name == "nt", reason="Apple Silicon uses POSIX process groups.")
def test_cancel_stops_owned_worker_and_clears_tracking(tmp_path):
    cancelled = threading.Event()
    timer = threading.Timer(0.2, cancelled.set)
    timer.start()
    try:
        with pytest.raises(runtime.MLXVideoCancelled):
            runtime._run_command([sys.executable, "-c", "import time; time.sleep(30)"],
                                 env=runtime._child_env(), label="fixture video", cancel_event=cancelled)
    finally:
        timer.cancel()
    assert runtime.owned_process_ids() == []


def test_native_quantized_lora_residual_when_mlx_is_available():
    pytest.importorskip("mlx.core")
    runtime._probe_quant_lora()
