"""Native media routing and process contracts without pretrained weights."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import zipfile

import pytest
from PIL import Image

from core import gguf_image_runtime as runtime
from tools import build_gguf_media_runtime as builder


@pytest.mark.parametrize("script", ["build_gguf_media_runtime.py", "build_release_package.py"])
def test_packaging_entry_points_need_only_the_standard_library(script):
    entry = Path(__file__).resolve().parents[2] / "tools" / script
    result = subprocess.run([sys.executable, "-I", "-S", str(entry), "--help"],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.fixture
def selected(tmp_path):
    primary = tmp_path / "primary model with spaces"
    primary.mkdir()
    (primary / "selected.gguf").write_bytes(b"tiny selected diffusion fixture")
    info = {"repo_id": "fixture/selected", "repo_revision": "a" * 40,
            "path": str(primary), "model_path": str(primary),
            "download_required_files": ["selected.gguf"], "gguf_architecture": "z_image",
            "download_components": [], "component_paths": {}, "component_files": {}}
    for role, filename in (("llm", "encoder.gguf"), ("vae", "decoder.safetensors"),
                           ("llm_vision", "encoder.mmproj-Q8_0.gguf")):
        folder = tmp_path / (role + " component")
        folder.mkdir()
        path = folder / filename
        path.write_bytes(b"tiny companion fixture")
        info["download_components"].append({"role": role, "repo_id": "fixture/" + role,
                                             "revision": "b" * 40, "required_files": [filename]})
        info["component_paths"][role] = str(folder)
        info["component_files"][role] = [str(path)]
    return info


@pytest.fixture
def native_fixture(tmp_path, monkeypatch):
    # Replace only heavyweight model allocation. The real subprocess runner,
    # argv, local artifact checks, PNG decoder and atomic publication remain.
    script = tmp_path / "native fixture.py"
    script.write_text(r'''
import argparse, os
from pathlib import Path
from PIL import Image
p = argparse.ArgumentParser()
p.add_argument('--diffusion-model'); p.add_argument('--llm'); p.add_argument('--vae')
p.add_argument('--llm_vision'); p.add_argument('-r', action='append', default=[])
p.add_argument('-W', type=int); p.add_argument('-H', type=int); p.add_argument('-o')
p.add_argument('-p'); p.add_argument('-n'); p.add_argument('--steps'); p.add_argument('--seed')
p.add_argument('--cfg-scale'); p.add_argument('--sampling-method'); p.add_argument('--scheduler')
p.add_argument('--flow-shift'); p.add_argument('--auto-fit')
a = p.parse_args()
assert os.environ['HF_HUB_OFFLINE'] == os.environ['TRANSFORMERS_OFFLINE'] == '1'
for file in (a.diffusion_model, a.llm, a.vae):
    assert Path(file).is_file() and Path(file).stat().st_size
for file in a.r:
    with Image.open(file) as img:
        assert img.format == 'PNG' and img.mode == 'RGB'
print('  |################| 9/9 - 55.00MB/s', flush=True)
print('  |====>           | 1/4 - 1.00it/s', flush=True)
print('  |================| 4/4 - 1.00it/s', flush=True)
behavior = os.environ.get('SMARTAI_TEST_NATIVE_OUTPUT', 'valid')
if behavior == 'missing':
    raise SystemExit(0)
if behavior == 'corrupt':
    Path(a.o).write_bytes(b'not a decoded image')
else:
    size = (a.W, a.H) if behavior == 'valid' else (32, 32)
    image = Image.new('RGB', size, (220, 30, 10))
    image.putpixel((0, 0), (20, 30, 200))
    image.save(a.o, format='PNG')
''', encoding="utf-8")
    real_run = runtime.run_gguf_media_command
    commands = []

    def execute(command, **options):
        commands.append([str(arg) for arg in command])
        return real_run([sys.executable, str(script), *command[1:]], **options)

    monkeypatch.setattr(runtime, "ensure_sd_cpp_runtime", lambda **_kwargs: Path("sd-cli"))
    monkeypatch.setattr(runtime, "run_gguf_media_command", execute)
    return commands


@pytest.mark.parametrize("architecture,edit", [("z_image", False), ("qwen_image", False), ("qwen_image", True)])
def test_exact_components_generate_real_decoded_output(selected, native_fixture, tmp_path, architecture, edit):
    selected["gguf_architecture"] = architecture
    if architecture == "z_image":
        selected["download_components"] = [item for item in selected["download_components"] if item["role"] != "llm_vision"]
        for field in ("component_paths", "component_files"):
            selected[field].pop("llm_vision")
    prompt = 'a paper boat; $(not a shell) "quoted"'
    source = tmp_path / "original reference.webp"
    Image.new("RGB", (96, 80), (5, 60, 80)).save(source)
    before = source.read_bytes()
    destination = tmp_path / "output with spaces.png"
    updates = []
    result = runtime.generate_gguf_image(selected, prompt, destination, width=128, height=64,
                                         steps=4, seed=0, reference_images=[source] if edit else None,
                                         progress_callback=lambda value, text: updates.append((value, text)))
    assert result["status"] == "success"
    with Image.open(destination) as image:
        assert image.format == "PNG" and image.size == (128, 64)
        assert image.getpixel((0, 0)) == (20, 30, 200)
    command = native_fixture[0]
    assert command[command.index("-p") + 1] == prompt
    assert command[command.index("--seed") + 1] == "0"
    assert command[command.index("--diffusion-model") + 1] == str(Path(selected["path"]) / "selected.gguf")
    assert command[command.index("--llm") + 1] == selected["component_files"]["llm"][0]
    if architecture == "qwen_image":
        assert command[command.index("--llm_vision") + 1] == selected["component_files"]["llm_vision"][0]
    else:
        assert "--llm_vision" not in command
    assert ("-r" in command) == edit
    assert command[command.index("--scheduler") + 1] == ("discrete" if architecture == "z_image" else "beta")
    assert source.read_bytes() == before
    assert [text for _, text in updates if text.startswith("Sampling")] == ["Sampling step 1/4", "Sampling step 4/4"]
    assert not list(tmp_path.glob(".smartai-gguf-*"))


@pytest.mark.parametrize("behavior", ["missing", "corrupt", "wrong_size"])
def test_invalid_native_output_never_replaces_user_file(selected, native_fixture, tmp_path, monkeypatch, behavior):
    destination = tmp_path / "keep.png"
    destination.write_bytes(b"existing user output")
    monkeypatch.setenv("SMARTAI_TEST_NATIVE_OUTPUT", behavior)
    with pytest.raises((RuntimeError, OSError)):
        runtime.generate_gguf_image(selected, "a paper boat", destination, width=64, height=64)
    assert destination.read_bytes() == b"existing user output"


def test_missing_companion_fails_before_runtime_execution(selected, monkeypatch, tmp_path):
    Path(selected["component_files"]["vae"][0]).unlink()
    monkeypatch.setattr(runtime, "ensure_sd_cpp_runtime", lambda **_kwargs: pytest.fail("Incomplete model reached native runtime"))
    with pytest.raises(RuntimeError, match="missing or empty"):
        runtime.generate_gguf_image(selected, "boat", tmp_path / "output.png")


def test_mismatched_component_mapping_cannot_select_another_file(selected, monkeypatch, tmp_path):
    selected["component_files"]["llm"] = selected["component_files"]["vae"]
    monkeypatch.setattr(runtime, "ensure_sd_cpp_runtime", lambda **_kwargs: pytest.fail("Mismatched selection reached native runtime"))
    with pytest.raises(ValueError, match="installed selection"):
        runtime.generate_gguf_image(selected, "boat", tmp_path / "output.png")


def test_runner_drains_real_steps_retains_ffmpeg_tail_and_never_claims_completion():
    updates = []
    output = runtime.run_gguf_media_command(
        [sys.executable, "-c", "print('|###| 9/9 - 3.00MB/s'); print('|==>| 1/2 - 2.00it/s'); "
         "print('|===| 2/2 - 2.00it/s'); print('|==>| 1/2 - 2.00it/s'); print('frame=17'); print('progress=end')"],
        progress_callback=lambda value, text: updates.append((value, text)), timeout=10)
    assert "frame=17" in output and "progress=end" in output
    assert [text for _, text in updates] == ["Sampling step 1/2", "Sampling step 2/2", "Sampling step 1/2"]
    values = [value for value, _ in updates]
    assert values == sorted(values) and max(values) < 1
    assert runtime.owned_process_ids() == []


def test_runner_cancellation_terminates_its_actual_owned_child(monkeypatch):
    event = threading.Event()
    seen = []
    popen = runtime.subprocess.Popen

    def capture(*args, **kwargs):
        child = popen(*args, **kwargs)
        seen.append(child)
        return child

    monkeypatch.setattr(runtime.subprocess, "Popen", capture)
    timer = threading.Timer(0.2, event.set)
    timer.start()
    try:
        with pytest.raises(runtime.GGUFMediaCancelled):
            runtime.run_gguf_media_command([sys.executable, "-c", "import time; time.sleep(30)"],
                                           cancel_event=event, timeout=10)
    finally:
        timer.cancel()
    assert seen[0].poll() is not None and runtime.owned_process_ids() == []


def test_runner_preserves_native_failure_detail():
    with pytest.raises(RuntimeError, match="unsupported fixture tensor"):
        runtime.run_gguf_media_command([sys.executable, "-c", "print('unsupported fixture tensor'); raise SystemExit(3)"], timeout=10)


def test_bundle_validates_every_native_dependency(tmp_path):
    executable = "sd-cli.exe" if os.name == "nt" else "sd-cli"
    for name in (executable, "native-library.dll"):
        (tmp_path / name).write_bytes(b"small native bundle fixture")
    receipt = {"schema": 1, "source_commit": runtime.SD_CPP_COMMIT, "ggml_commit": runtime.GGML_COMMIT,
               "source_archives": [list(item) for item in runtime.SOURCE_ARCHIVES],
               "platform": runtime.platform.system(), "machine": runtime._machine(), "native_probe": "passed",
               "files": {name: runtime._sha256(tmp_path / name) for name in (executable, "native-library.dll")}}
    (tmp_path / runtime.RECEIPT_NAME).write_text(json.dumps(receipt))
    assert runtime.validate_runtime_bundle(tmp_path)[0] == tmp_path / executable
    (tmp_path / "native-library.dll").write_bytes(b"changed dependency")
    with pytest.raises(RuntimeError, match="failed verification"):
        runtime.validate_runtime_bundle(tmp_path)


def test_source_archive_rejects_traversal_and_skips_example_images(tmp_path):
    safe = tmp_path / "safe.zip"
    with zipfile.ZipFile(safe, "w") as archive:
        archive.writestr("publisher/src/model.cpp", "fixture")
        archive.writestr("publisher/assets/example.png", "not a build input")
    builder._extract_source(safe, tmp_path / "source")
    assert (tmp_path / "source/src/model.cpp").read_text() == "fixture"
    assert not (tmp_path / "source/assets").exists()
    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("publisher/../escape", "bad")
    with pytest.raises(RuntimeError, match="Unsafe path"):
        builder._extract_source(unsafe, tmp_path / "untrusted")
    assert not (tmp_path / "escape").exists()


def test_mac_build_keeps_deployment_floor_and_source_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(builder.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(builder, "_machine", lambda: "arm64")
    command = builder._configure_command("cmake", tmp_path / "source", tmp_path / "build", "metal")
    assert "-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0" in command
    assert "-DGGML_METAL_EMBED_LIBRARY=ON" in command
    assert "-DSD_METAL=ON" in command
    assert "-DGIT_EXE=OFF" in command
    assert "-DSDCPP_BUILD_COMMIT=" + runtime.SD_CPP_COMMIT in command


def test_auto_backend_requires_a_vulkan_sdk_and_keeps_cpu_fallback(monkeypatch, capsys):
    monkeypatch.setattr(builder.platform, "system", lambda: "Linux")
    monkeypatch.setattr(builder, "_vulkan_sdk_available", lambda: False)
    assert builder.select_backend() == "cpu"
    assert "CPU backend" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="Vulkan SDK"):
        builder.select_backend("vulkan")
    monkeypatch.setattr(builder, "_vulkan_sdk_available", lambda: True)
    assert builder.select_backend() == "vulkan"


def test_auto_backend_rejects_old_or_incomplete_sdk(tmp_path, monkeypatch):
    monkeypatch.setenv("VULKAN_SDK", str(tmp_path))
    monkeypatch.setattr(builder.shutil, "which", lambda _name: None)
    for name, value in {
        "Bin/glslc.exe": "fixture compiler", "Include/vulkan/vulkan.hpp": "fixture C++ headers",
        "Include/vulkan/vulkan_core.h": "old Vulkan 1.3.204 headers",
        "Lib/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake": "fixture CMake package",
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    assert not builder._vulkan_sdk_available()
    (tmp_path / "Include/vulkan/vulkan_core.h").write_text(
        "VkPhysicalDevicePipelineRobustnessFeaturesEXT VkPhysicalDeviceFaultFeaturesEXT")
    assert builder._vulkan_sdk_available()
    (tmp_path / "Lib/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake").unlink()
    assert not builder._vulkan_sdk_available()
