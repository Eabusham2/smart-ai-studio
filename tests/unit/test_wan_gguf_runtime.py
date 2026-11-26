"""Exact Wan selection and real video I/O, without downloading model weights."""
import copy
from pathlib import Path
import subprocess
import sys
import threading
import types

import pytest

from core import wan_gguf_runtime as runtime


@pytest.fixture
def selection(tmp_path, monkeypatch):
    main = tmp_path / "model cache with spaces" / runtime.MODEL_REVISION
    paths = {role: tmp_path / role / spec["revision"] for role, spec in runtime.COMPONENTS.items()}
    data = {
        runtime.HIGH_NOISE_FILE: b"GGUF\x03\0\0\0high-noise test header",
        runtime.LOW_NOISE_FILE: b"GGUF\x03\0\0\0low-noise test header",
        runtime.COMPONENTS["t5xxl"]["required_files"][0]: b"GGUF\x03\0\0\0text encoder test header",
    }
    header = b'{"fixture":{}}'
    data[runtime.COMPONENTS["vae"]["required_files"][0]] = len(header).to_bytes(8, "little") + header + b"x"
    # These are intentionally tiny format fixtures, never pretend model weights.
    monkeypatch.setattr(runtime, "_FILE_SIZES", {name: len(content) for name, content in data.items()})
    for name in (runtime.HIGH_NOISE_FILE, runtime.LOW_NOISE_FILE):
        path = main / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data[name])
    component_files = {}
    for role, spec in runtime.COMPONENTS.items():
        name = spec["required_files"][0]
        path = paths[role] / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data[name])
        component_files[role] = [str(path)]
    return {
        "repo_id": runtime.MODEL_REPO, "repo_revision": runtime.MODEL_REVISION,
        "download_required_files": [runtime.HIGH_NOISE_FILE, runtime.LOW_NOISE_FILE],
        "download_components": [{"role": role, **copy.deepcopy(spec)} for role, spec in runtime.COMPONENTS.items()],
        "model_path": str(main), "component_paths": {role: str(path) for role, path in paths.items()},
        "component_files": component_files, "width": 32, "height": 32,
        "fps": 16, "inference_steps": 18, "high_noise_steps": -1, "seed": 17,
    }


def _stub_native(monkeypatch, run):
    calls = []

    def prepare(**kwargs):
        calls.append(kwargs)
        return Path("/fixture runtime/sd-cli")

    monkeypatch.setitem(sys.modules, "core.gguf_image_runtime", types.SimpleNamespace(
        ensure_gguf_media_runtime=prepare,
        run_gguf_media_command=run,
        owned_process_ids=lambda: [123],
    ))
    return calls


def _real_fixture_video(path, *, frames=5, fps=16):
    """Encode five tiny actual test-pattern frames to an AVI input fixture."""
    import imageio_ffmpeg
    result = subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc2=size=32x32:rate={fps}",
        "-frames:v", str(frames), "-c:v", "mjpeg", str(path),
    ], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_wan_command_uses_both_exact_t2v_q4_experts_and_full_vae(selection, tmp_path):
    files = runtime._component_files(selection)
    options = runtime._generation_options(selection, 49)
    argv = runtime._generation_command("/native/sd-cli", files, "a river at sunset", selection,
                                       options, tmp_path / "frames.avi", tmp_path / "preview.avi")
    assert argv[1:3] == ["-M", "vid_gen"]
    model_path = Path(selection["model_path"])
    assert Path(argv[argv.index("--diffusion-model") + 1]) == model_path / runtime.LOW_NOISE_FILE
    assert Path(argv[argv.index("--high-noise-diffusion-model") + 1]) == model_path / runtime.HIGH_NOISE_FILE
    assert argv[argv.index("--t5xxl") + 1] == selection["component_files"]["t5xxl"][0]
    assert argv[argv.index("--vae") + 1] == selection["component_files"]["vae"][0]
    assert argv[argv.index("--high-noise-steps") + 1] == "-1"
    assert argv[argv.index("--video-frames") + 1] == "49"
    assert argv[argv.index("--fps") + 1] == "16"
    assert argv[argv.index("--seed") + 1] == "17"
    assert argv[argv.index("--preview") + 1] == "proj"
    assert "Wan2.1-T2V-1.3B" not in str(argv)


@pytest.mark.parametrize("problem", ["expert", "vae", "component_revision", "model_revision", "pointer", "wrong_file"])
def test_missing_or_mismatched_components_cannot_start_a_runtime(selection, tmp_path, monkeypatch, problem):
    def forbidden(*_args, **_kwargs):
        pytest.fail("An invalid model selection started a native process")

    preparations = _stub_native(monkeypatch, forbidden)
    if problem == "expert":
        (Path(selection["model_path"]) / runtime.LOW_NOISE_FILE).unlink()
    elif problem == "vae":
        Path(selection["component_files"]["vae"][0]).write_bytes(b"")
    elif problem == "component_revision":
        selection["download_components"][0]["revision"] = "0" * 40
    elif problem == "model_revision":
        selection["repo_revision"] = "0" * 40
    elif problem == "pointer":
        source = Path(selection["component_files"]["t5xxl"][0])
        source.write_bytes(b"x" * source.stat().st_size)
    else:
        selection["component_files"]["vae"] = [str(tmp_path / "another-vae.safetensors")]
    result = runtime.generate_wan_gguf_video(selection, "a river", tmp_path / "out.mp4", frames=5)
    assert result["status"] == "error"
    assert preparations == []
    assert not (tmp_path / "out.mp4").exists()


@pytest.mark.parametrize("updates,frames", [({"width": 33}, 5), ({"height": 0}, 5), ({}, 6),
                                           ({"high_noise_steps": 0}, 5), ({"guidance_scale": float("nan")}, 5)])
def test_invalid_generation_options_fail_before_loading_weights(selection, tmp_path, monkeypatch, updates, frames):
    preparations = _stub_native(monkeypatch, lambda *_args, **_kwargs: pytest.fail("Unexpected generation"))
    result = runtime.generate_wan_gguf_video({**selection, **updates}, "a river", tmp_path / "out.mp4", frames=frames)
    assert result["status"] == "error" and preparations == []


def test_actual_generated_frames_become_a_checked_mp4_and_real_previews(selection, tmp_path, monkeypatch):
    calls, previews, progress = [], [], []

    def command(argv, **kwargs):
        calls.append((argv, kwargs))
        if "vid_gen" in argv:
            _real_fixture_video(argv[argv.index("-o") + 1])
            _real_fixture_video(argv[argv.index("--preview-path") + 1])
            kwargs["progress_callback"](0.5, "Sampling 9/18")
            return "Sampling 18/18"
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        assert completed.returncode == 0, completed.stderr
        return completed.stdout + completed.stderr

    _stub_native(monkeypatch, command)
    output = tmp_path / "video output with spaces.mp4"
    result = runtime.generate_wan_gguf_video(selection, "a river", output, frames=5,
                                            progress_callback=lambda *args: progress.append(args),
                                            preview_callback=previews.append)
    assert result["status"] == "success", result
    assert result["backend"] == "wan_gguf" and result["frames"] == 5
    assert result["format"] == "MP4" and result["width"] == result["height"] == 32
    assert result["fps"] == 16 and output.stat().st_size > 0
    assert len(calls) == 2 and calls[0][1]["timeout"] is None
    assert calls[1][1]["timeout"] == 300
    assert "-xerror" in calls[1][0] and "libx264" in calls[1][0]
    assert [preview["label"] for preview in previews] == [
        "Wan sampling preview (latent projection)", "Generated Wan video frame",
    ]
    assert all(preview["image"].size == (32, 32) for preview in previews)
    assert progress[-1] == (1.0, "Video complete")
    assert not list(tmp_path.glob(".smartai-wan-*"))
    assert runtime.owned_process_ids() == [123]


@pytest.mark.parametrize("output_kind", ["missing", "corrupt", "short"])
def test_exit_zero_with_missing_corrupt_or_short_video_is_not_success(selection, tmp_path, monkeypatch, output_kind):
    def command(argv, **_kwargs):
        if "vid_gen" in argv:
            output = Path(argv[argv.index("-o") + 1])
            if output_kind == "corrupt":
                output.write_bytes(b"not a video")
            elif output_kind == "short":
                _real_fixture_video(output, frames=2)
            return "native exit zero"
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        return completed.stdout

    _stub_native(monkeypatch, command)
    output = tmp_path / "keep original.mp4"
    output.write_bytes(b"existing output")
    result = runtime.generate_wan_gguf_video(selection, "a river", output, frames=5)
    assert result["status"] == "error"
    assert output.read_bytes() == b"existing output"
    assert not list(tmp_path.glob(".smartai-wan-*"))


def test_cancel_after_native_generation_prevents_encoding_or_replacing_output(selection, tmp_path, monkeypatch):
    event = threading.Event()
    calls = []

    def command(argv, **kwargs):
        assert kwargs["cancel_event"] is event
        calls.append(argv)
        event.set()
        return "interrupted"

    _stub_native(monkeypatch, command)
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"original")
    result = runtime.generate_wan_gguf_video(selection, "a river", output, frames=5, cancel_event=event)
    assert result["status"] == "error" and "cancelled" in result["error"]
    assert len(calls) == 1 and output.read_bytes() == b"original"
    assert not list(tmp_path.glob(".smartai-wan-*"))
