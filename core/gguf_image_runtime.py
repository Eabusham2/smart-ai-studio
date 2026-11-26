"""Run selected GGUF image components with Studio's bundled native media CLI.

The executable is built from the pinned publisher sources during packaging.
Users need neither a compiler nor a second Python environment. Generation uses
only the local files selected by the catalog; this module downloads no weights.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import queue
import re
import signal
import subprocess
import tempfile
import threading
import time


SD_CPP_COMMIT = "52491bd9f1fc15281791723bdff13baab7fd6819"
GGML_COMMIT = "eeda9ec48fc84a3df0bb9dcc19ae9053075a2539"
SOURCE_ARCHIVES = (
    ("stable-diffusion", f"https://codeload.github.com/leejet/stable-diffusion.cpp/zip/{SD_CPP_COMMIT}",
     "843968d07d759acbec66f4a3dccb10c9cc62b8868b7ee866a4edcd1afe51192c"),
    ("ggml", f"https://codeload.github.com/leejet/ggml/zip/{GGML_COMMIT}",
     "f2fd29587016cd3525ca25e3f5233ba03c504b723558d89b12913e7956d14f68"),
)
RECEIPT_NAME = "smartai-gguf-media.json"
_PROCESSES = {}
_PROCESS_LOCK = threading.Lock()
_PROBED = set()
_STEP = re.compile(r"(?:step\s*[:=]?\s*|\|\s*)(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class GGUFMediaCancelled(RuntimeError):
    pass


def _check_cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise GGUFMediaCancelled("Native media generation cancelled; installed model files were kept.")


def _notify(callback, fraction, message):
    if callback is not None:
        callback(float(fraction), str(message))


def owned_process_ids():
    with _PROCESS_LOCK:
        return sorted(pid for pid, process in _PROCESSES.items() if process.poll() is None)


def _stop_process(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.TimeoutExpired):
            process.terminate()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=3)


def run_gguf_media_command(command, *, cancel_event=None, progress_callback=None,
                           cwd=None, label="Native media generation", timeout=None):
    """Run an owned native process, including cancellable video encoding.

    Progress is read from the actual CLI step counter. Stage resets cannot move
    the progress bar backwards, and success here never claims a valid output.
    The image/video caller must decode the resulting file before reporting it.
    """
    _check_cancelled(cancel_event)
    command = [str(value) for value in command]
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if Path(command[0]).name in {"sd-cli", "sd-cli.exe"}:
        # This pin expects a single library in this override, not a directory.
        # Its normal discovery already searches beside the bundled executable.
        env.pop("GGML_BACKEND_PATH", None)
    options = ({"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0) |
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)} if os.name == "nt"
               else {"start_new_session": True})
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                               errors="replace", bufsize=1, **options)
    with _PROCESS_LOCK:
        _PROCESSES[process.pid] = process
    messages = queue.Queue()
    tail = deque(maxlen=500)

    def read_output():
        try:
            for line in process.stdout:
                messages.put(_ANSI.sub("", line.rstrip())[-3000:])
        finally:
            process.stdout.close()

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout if timeout is not None else None
    progress = 0.0
    try:
        while process.poll() is None or reader.is_alive() or not messages.empty():
            _check_cancelled(cancel_event)
            if deadline is not None and time.monotonic() > deadline:
                raise RuntimeError(f"{label} exceeded its time limit.")
            try:
                line = messages.get(timeout=0.1)
            except queue.Empty:
                continue
            tail.append(line)
            # The loader prints a similar bar measured in MB/s. Only actual
            # sampler counters measured in iterations are denoising progress.
            matched = _STEP.search(line) if ("it/s" in line or "s/it" in line or
                                              re.search(r"\bstep\s*[:=]?\s*\d+\s*/", line, re.IGNORECASE)) else None
            if matched:
                step, total = map(int, matched.groups())
                if total > 0 and 0 <= step <= total:
                    progress = max(progress, min(0.95, step / total * 0.95))
                    _notify(progress_callback, progress, f"Sampling step {step}/{total}")
            elif line and any(word in line.lower() for word in ("loading", "encoding", "decoding", "backend", "offload")):
                _notify(progress_callback, progress, line)
        code = process.wait()
        _check_cancelled(cancel_event)
        output = "\n".join(tail)
        if code:
            raise RuntimeError(f"{label} failed (exit {code}):\n{output[-12000:]}")
        return output
    except BaseException:
        _stop_process(process)
        raise
    finally:
        reader.join(timeout=1)
        with _PROCESS_LOCK:
            _PROCESSES.pop(process.pid, None)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _machine():
    value = platform.machine().lower()
    return {"amd64": "x86_64", "aarch64": "arm64"}.get(value, value)


def _relative_file(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Invalid selected native media file.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError("Invalid selected native media file.")
    return path


def validate_runtime_bundle(directory):
    """Check package provenance and every native dependency before execution."""
    directory = Path(directory)
    try:
        receipt = json.loads((directory / RECEIPT_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError("The bundled GGUF media runtime is missing or incomplete. Reinstall the current Studio package.") from error
    if (receipt.get("schema") != 1 or receipt.get("source_commit") != SD_CPP_COMMIT or
            receipt.get("ggml_commit") != GGML_COMMIT or
            receipt.get("source_archives") != [list(item) for item in SOURCE_ARCHIVES] or
            receipt.get("platform") != platform.system() or receipt.get("machine") != _machine() or
            receipt.get("native_probe") != "passed"):
        raise RuntimeError("The bundled GGUF media runtime does not match this Studio build or platform.")
    files = receipt.get("files")
    executable_name = "sd-cli.exe" if os.name == "nt" else "sd-cli"
    if not isinstance(files, dict) or executable_name not in files:
        raise RuntimeError("The bundled GGUF media runtime receipt has no executable.")
    for name, digest in files.items():
        relative = _relative_file(name)
        path = directory.joinpath(*relative.parts)
        if (path.is_symlink() or not path.is_file() or path.stat().st_size == 0 or
                not isinstance(digest, str) or _sha256(path) != digest):
            raise RuntimeError(f"The bundled GGUF media runtime file failed verification: {name}")
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
    if actual != set(files) | {RECEIPT_NAME}:
        raise RuntimeError("The reserved GGUF media runtime directory contains unexpected files.")
    return directory / executable_name, receipt


def probe_sd_cpp_runtime(executable, *, cancel_event=None):
    """Exercise the real native CLI/API without loading any model weights."""
    version = run_gguf_media_command([executable, "--version"], cancel_event=cancel_event,
                                     label="Native media version probe", timeout=60)
    if SD_CPP_COMMIT[:7] not in version:
        raise RuntimeError("The native media executable reports a different publisher revision.")
    help_text = run_gguf_media_command([executable, "--help"], cancel_event=cancel_event,
                                       label="Native media CLI probe", timeout=60)
    for flag in ("--diffusion-model", "--llm", "--llm_vision", "--vae", "--t5xxl",
                 "--high-noise-diffusion-model", "--auto-fit", "--list-devices",
                 "--sampling-method", "--scheduler"):
        if flag not in help_text:
            raise RuntimeError(f"The native media executable is missing required support: {flag}")
    devices = run_gguf_media_command([executable, "--list-devices"], cancel_event=cancel_event,
                                     label="Native media device probe", timeout=60)
    if not re.search(r"^(?:CPU|Metal|Vulkan|CUDA)[^\r\n]*\t", devices, re.IGNORECASE | re.MULTILINE):
        raise RuntimeError("The native media runtime could not initialize a usable compute backend.")
    return devices.strip()


def ensure_sd_cpp_runtime(*, progress_callback=None, cancel_event=None, force_probe=False):
    _check_cancelled(cancel_event)
    directory = Path(os.getenv("SMARTAI_GGUF_MEDIA_DIR") or
                     Path(__file__).resolve().parents[1] / "runtime" / "gguf-media")
    executable, receipt = validate_runtime_bundle(directory)
    key = (str(executable.absolute()), receipt["files"][executable.name])
    if force_probe or key not in _PROBED:
        _notify(progress_callback, 0.01, "Checking the bundled native image/video runtime")
        devices = probe_sd_cpp_runtime(executable, cancel_event=cancel_event)
        _PROBED.add(key)
        _notify(progress_callback, 0.02, "Native media devices: " + devices)
    return executable


ensure_gguf_media_runtime = ensure_sd_cpp_runtime


def _selected_files(directory, names):
    if not directory or not isinstance(names, (list, tuple)) or not names:
        raise ValueError("The selected image model has no complete local artifact list.")
    root = Path(directory).expanduser().absolute()
    paths = [root.joinpath(*_relative_file(name).parts) for name in names]
    if root.is_file() and len(names) == 1 and root.name == PurePosixPath(names[0]).name:
        paths = [root]
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        raise RuntimeError("A selected native image component is missing or empty. Finish Install before generating.")
    return paths


def _image_components(info):
    primary = _selected_files(info.get("model_path") or info.get("path"), info.get("download_required_files"))
    diffusion = [path for path in primary if path.suffix.lower() == ".gguf"]
    if len(diffusion) != 1:
        raise ValueError("Select exactly one diffusion-model GGUF; alternate quants are not interchangeable.")
    components = {"diffusion": diffusion[0]}
    for component in info.get("download_components") or []:
        role = component.get("role")
        if role not in {"llm", "llm_vision", "vae"} or role in components:
            raise ValueError(f"Invalid or duplicated image component role: {role}")
        paths = _selected_files((info.get("component_paths") or {}).get(role), component.get("required_files"))
        if len(paths) != 1:
            raise ValueError(f"Native image component {role} must select one exact file.")
        declared = (info.get("component_files") or {}).get(role)
        if declared is not None and [str(Path(path).absolute()) for path in declared] != [str(path) for path in paths]:
            raise ValueError(f"Native image component {role} does not match its installed selection.")
        components[role] = paths[0]
    required = {"llm", "vae"}
    if info.get("gguf_architecture") == "qwen_image":
        required.add("llm_vision")
    if not required.issubset(components):
        raise ValueError("The native image model is missing its text encoder, VAE, or vision component selection.")
    return components


def generate_gguf_image(model_info, prompt, output_path, *, steps=None, width=1024, height=1024,
                        seed=0, reference_images=None, progress_callback=None,
                        preview_callback=None, cancel_event=None):
    """Generate and decode real pixels, then atomically publish the result."""
    from PIL import Image

    _check_cancelled(cancel_event)
    architecture = model_info.get("gguf_architecture")
    if architecture not in {"z_image", "qwen_image"}:
        raise ValueError("This GGUF image architecture is not configured.")
    if not str(prompt).strip():
        raise ValueError("Enter a prompt for image generation.")
    width, height = int(width), int(height)
    if any(value < 64 or value > 8192 or value % 16 for value in (width, height)):
        raise ValueError("Image width and height must be multiples of 16 between 64 and 8192.")
    steps = int(steps if steps is not None else model_info.get("inference_steps", 8 if architecture == "z_image" else 4))
    if not 1 <= steps <= 1000:
        raise ValueError("Image generation requires between 1 and 1000 sampling steps.")
    references = list(reference_images or [])
    if references and architecture != "qwen_image":
        raise ValueError("Use the Qwen Image Edit preset to edit a reference image.")
    if len(references) > 3:
        raise ValueError("Qwen Image Edit accepts up to three reference images here.")
    components = _image_components(model_info)
    executable = ensure_sd_cpp_runtime(progress_callback=progress_callback, cancel_event=cancel_event)
    destination = Path(output_path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    formats = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP"}
    if destination.suffix.lower() not in formats:
        raise ValueError("Save generated images as PNG, JPEG, or WebP.")
    with tempfile.TemporaryDirectory(prefix=".smartai-gguf-", dir=destination.parent) as temporary:
        work = Path(temporary)
        generated = work / "generated.png"
        command = [executable, "--diffusion-model", components["diffusion"],
                   "--llm", components["llm"], "--vae", components["vae"],
                   "-p", str(prompt), "-W", str(width), "-H", str(height),
                   "--steps", str(steps), "--seed", str(int(seed)), "--cfg-scale", "1",
                   "--sampling-method", "euler" if architecture == "z_image" else "euler_a",
                   "--scheduler", "discrete" if architecture == "z_image" else "beta",
                   "--flow-shift", "3", "--auto-fit", "on",
                   "-o", generated]
        if "llm_vision" in components:
            command.extend(["--llm_vision", components["llm_vision"]])
        negative = str(model_info.get("negative_prompt") or "")
        if negative:
            command.extend(["-n", negative])
        for index, source in enumerate(references):
            converted = work / f"reference-{index}.png"
            with Image.open(source) as image:
                image.convert("RGB").save(converted, format="PNG")
            command.extend(["-r", converted])
        _notify(progress_callback, 0.03, "Loading the selected GGUF image components")
        run_gguf_media_command(command, cancel_event=cancel_event,
                               progress_callback=lambda value, text: _notify(progress_callback, 0.04 + value * 0.91, text),
                               label="GGUF image generation", timeout=None)
        _check_cancelled(cancel_event)
        if not generated.is_file() or generated.stat().st_size == 0:
            raise RuntimeError("The native image runtime exited without producing an image.")
        with Image.open(generated) as image:
            image.verify()
        with Image.open(generated) as image:
            image.load()
            if image.size != (width, height):
                raise RuntimeError("The native image runtime returned an unexpected image size.")
            prepared = work / ("validated" + destination.suffix.lower())
            image.convert("RGB").save(prepared, format=formats[destination.suffix.lower()])
        _check_cancelled(cancel_event)
        prepared.replace(destination)
    _notify(progress_callback, 0.99, "Generated image decoded successfully")
    return {"status": "success", "path": str(destination), "backend": "gguf",
            "repo_id": model_info.get("repo_id"), "runtime_commit": SD_CPP_COMMIT}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check the bundled native GGUF media runtime without model weights")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    executable = ensure_sd_cpp_runtime(force_probe=args.probe,
                                        progress_callback=lambda _value, text: print(text, flush=True))
    print("GGUF_MEDIA_RUNTIME_VERIFIED", executable, SD_CPP_COMMIT, flush=True)


if __name__ == "__main__":
    main()
