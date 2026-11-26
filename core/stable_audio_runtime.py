"""Run the original Stable Audio 3 checkpoints in their own Python environment.

The publisher requires newer Transformers than Studio's chat runtime. Only this
private environment receives those dependencies; model files stay in the shared,
revision-pinned Hub cache. The generation worker is always offline.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import queue
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave


PUBLISHER_COMMIT = "4007b6f2987bcb760626c2b4ae3e3541753984e9"
PUBLISHER_URL = f"https://codeload.github.com/Stability-AI/stable-audio-3/zip/{PUBLISHER_COMMIT}"
PUBLISHER_SHA256 = "e49f190ab2cfc06e77bc395b8b793a2e9285046fab688ecb3d16b9fed696d54c"
TORCH_VERSION = "2.7.1"
TRANSFORMERS_VERSION = "5.8.0"
MODEL_REPOS = frozenset({"stabilityai/stable-audio-3-small-music", "stabilityai/stable-audio-3-small-sfx"})
_EVENT_PREFIX = "SMARTAI_STABLE_AUDIO_PROGRESS "
_PROCESSES = {}
_PROCESS_LOCK = threading.Lock()


class StableAudioCancelled(RuntimeError):
    pass


def _check_cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise StableAudioCancelled("Audio generation cancelled; completed model downloads were kept.")


def _notify(callback, value, message):
    if callback is not None:
        callback(float(value), str(message))


def _nvidia_cuda_version():
    """Read the driver when Studio itself has a CPU-only Torch installation."""
    if os.getenv("CUDA_VISIBLE_DEVICES", "0").strip() in {"", "-1"}:
        return ""
    executable = shutil.which("nvidia-smi")
    if not executable:
        return ""
    try:
        import xml.etree.ElementTree as ET
        options = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}
        result = subprocess.run([executable, "-q", "-x"], capture_output=True, text=True,
                                timeout=3, check=True, **options)
        if len(result.stdout) > 1024 * 1024:
            return ""
        report = ET.fromstring(result.stdout)
        return str(report.findtext("cuda_version") or "") if report.findall("gpu") else ""
    except (OSError, ValueError, ET.ParseError, subprocess.SubprocessError):
        return ""


def _torch_index():
    """Choose an official wheel family using the requested/working app device."""
    requested = os.getenv("SMARTAI_AUDIO_DEVICE", "auto").lower()
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("SMARTAI_AUDIO_DEVICE must be auto, cpu, cuda, or mps.")
    if sys.platform == "darwin":
        if requested == "cuda":
            raise RuntimeError("CUDA audio requires Windows or Linux; it is unavailable on macOS.")
        return "pypi"  # The official macOS wheel contains CPU and MPS support.
    if requested == "mps":
        raise RuntimeError("MPS audio requires macOS.")
    if requested == "cpu":
        return "cpu"
    available, cuda_version = False, ""
    try:
        import torch
        available = torch.cuda.is_available()
        cuda_version = str(getattr(getattr(torch, "version", None), "cuda", "") or "")
    except ImportError:
        pass
    if not available:
        reported = _nvidia_cuda_version()
        if reported:
            available, cuda_version = True, reported
    if requested == "auto" and not available:
        return "cpu"
    # These are the publisher-documented Torch 2.7.1 CUDA distributions. Reuse
    # a channel compatible with the app's working driver instead of taking a
    # CPU-only PyPI wheel on Windows. Explicit CUDA without an app CUDA build
    # uses the publisher's default CUDA 12.6 and is checked by the native probe.
    try:
        version = tuple(int(part) for part in cuda_version.split(".")[:2])
    except ValueError:
        version = ()
    return "cu128" if version >= (12, 8) else "cu126" if not version or version >= (12, 6) else "cu118"


def _runtime_contract(cpu_probe=False):
    contract = {"schema": 1, "publisher_commit": PUBLISHER_COMMIT,
            "publisher_url": PUBLISHER_URL, "publisher_sha256": PUBLISHER_SHA256,
            "torch": TORCH_VERSION, "transformers": TRANSFORMERS_VERSION,
            "purpose": "cpu-probe" if cpu_probe else "generation",
            "base_python": os.path.realpath(sys.executable),
            "python_version": list(sys.version_info[:3])}
    if not cpu_probe:
        contract["torch_index"] = _torch_index()
    return contract


def _runtime_directory(cpu_probe=False):
    from config.paths import get_portable_data_dir
    digest = hashlib.sha256(json.dumps(_runtime_contract(cpu_probe), sort_keys=True).encode()).hexdigest()[:20]
    return Path(get_portable_data_dir()) / "runtimes" / "stable-audio-3" / digest


def _python_path(directory):
    return Path(directory) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _child_env(offline=False):
    env = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE", "VIRTUAL_ENV",
                 "PIP_TARGET", "PIP_PREFIX", "PIP_ROOT", "PIP_USER"):
        env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


def owned_process_ids():
    with _PROCESS_LOCK:
        return sorted(pid for pid, process in _PROCESSES.items() if process.poll() is None)


def _stop_process(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        # taskkill /T stops pip's build children too; no shell command is used.
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


def _run_command(command, *, env, label, cancel_event=None, progress_callback=None, timeout=1800):
    _check_cancelled(cancel_event)
    options = ({"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) |
                getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt"
               else {"start_new_session": True})
    process = subprocess.Popen([str(value) for value in command], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1, **options)
    with _PROCESS_LOCK:
        _PROCESSES[process.pid] = process
    messages, tail = queue.Queue(), deque(maxlen=30)
    def read_output():
        try:
            for line in process.stdout:
                messages.put(line.rstrip()[-2000:])
        finally:
            process.stdout.close()
    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout
    try:
        while process.poll() is None or reader.is_alive() or not messages.empty():
            _check_cancelled(cancel_event)
            if time.monotonic() > deadline:
                raise RuntimeError(f"{label} exceeded its time limit.")
            try:
                line = messages.get(timeout=0.1)
            except queue.Empty:
                continue
            if line.startswith(_EVENT_PREFIX):
                try:
                    event = json.loads(line[len(_EVENT_PREFIX):])
                    _notify(progress_callback, event["value"], event["message"])
                except (ValueError, TypeError, KeyError):
                    tail.append(line)
            else:
                tail.append(line)
        code = process.wait()
        _check_cancelled(cancel_event)
        if code:
            raise RuntimeError(f"{label} failed:\n" + ("\n".join(tail) or f"Process exited with status {code}."))
    except BaseException:
        _stop_process(process)
        raise
    finally:
        reader.join(timeout=1)
        with _PROCESS_LOCK:
            _PROCESSES.pop(process.pid, None)


@contextmanager
def _setup_lock(path, cancel_event):
    from filelock import FileLock, Timeout
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path))
    while True:
        _check_cancelled(cancel_event)
        try:
            lock.acquire(timeout=0.1)
            break
        except Timeout:
            continue
    try:
        yield
    finally:
        lock.release()


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def ensure_stable_audio_runtime(*, cancel_event=None, progress_callback=None, force_probe=False, cpu_probe=False):
    """Install publisher code/wheels only. CPU CI probes use a separate identity."""
    if sys.version_info < (3, 10):
        raise RuntimeError("Stable Audio 3 requires Python 3.10 or newer.")
    from core.hf_transfer import hf_subprocess_command
    directory = _runtime_directory(cpu_probe)
    contract = _runtime_contract(cpu_probe)
    python = _python_path(directory)
    owner, receipt = directory / "smartai-runtime-owner.json", directory / "smartai-runtime-ready.json"
    with _setup_lock(directory.with_suffix(".lock"), cancel_event):
        if directory.is_symlink():
            raise RuntimeError("The private audio runtime directory must not be a symlink.")
        if directory.exists() and any(directory.iterdir()) and _read_json(owner) != contract:
            raise RuntimeError(f"The reserved audio runtime folder contains unrecognized files: {directory}. Move it aside and retry.")
        directory.mkdir(parents=True, exist_ok=True)
        if not owner.exists():
            _write_json(owner, contract)
        ready = python.is_file() and _read_json(receipt) == contract
        if not ready:
            _notify(progress_callback, 0.03, "Preparing the private Stable Audio runtime (first use)")
            # Re-running venv also repairs an interrupted ensurepip stage. It
            # keeps existing private packages and never touches the app's Python.
            _run_command([sys.executable, "-I", "-m", "venv", str(directory)], env=_child_env(),
                         label="Private audio Python setup", cancel_event=cancel_event)
            pip = [str(python), "-I", "-m", "pip", "--isolated", "install", "--disable-pip-version-check",
                   "--no-input", "--retries", "2", "--timeout", "45", "--only-binary=:all:"]
            torch_index = "cpu" if cpu_probe else contract["torch_index"]
            if torch_index != "pypi":
                _notify(progress_callback, 0.05, f"Installing the private PyTorch {torch_index} audio runtime")
                _run_command([*pip, "--index-url", f"https://download.pytorch.org/whl/{torch_index}",
                              f"torch=={TORCH_VERSION}", f"torchaudio=={TORCH_VERSION}"],
                             env=_child_env(), label="Private audio tensor dependencies", cancel_event=cancel_event)
            _notify(progress_callback, 0.08, "Installing the publisher's isolated Stable Audio dependencies")
            _run_command([*pip, "--no-binary=stable-audio-3", f"transformers=={TRANSFORMERS_VERSION}",
                          f"stable-audio-3 @ {PUBLISHER_URL}#sha256={PUBLISHER_SHA256}"],
                         env=_child_env(), label="Publisher audio runtime installation", cancel_event=cancel_event)
        if not ready or force_probe:
            _notify(progress_callback, 0.12, "Checking the publisher's audio runtime and native tensor operations")
            command = [str(python), "-I", str(Path(__file__).resolve()), "--native-probe"]
            if cpu_probe:
                command.append("--cpu-probe")
            else:
                command.extend(["--torch-index", contract["torch_index"]])
            _run_command(hf_subprocess_command(command), env=_child_env(offline=True),
                         label="Native audio runtime validation", cancel_event=cancel_event, timeout=180)
            _write_json(receipt, contract)
    return python


def _check_torch_build(torch, torch_index):
    expected = {"cpu": None, "cu118": "11.8", "cu126": "12.6", "cu128": "12.8"}
    if torch_index in expected and torch.version.cuda != expected[torch_index]:
        raise RuntimeError(f"The private audio Torch wheel does not match the selected {torch_index} backend.")


def _native_components(cpu_probe=False, torch_index=None):
    from importlib import metadata
    for name, expected in (("torch", TORCH_VERSION), ("torchaudio", TORCH_VERSION),
                           ("transformers", TRANSFORMERS_VERSION)):
        if metadata.version(name).split("+")[0] != expected:
            raise RuntimeError(f"The private audio runtime has an unexpected {name} version.")
    direct = json.loads(metadata.distribution("stable-audio-3").read_text("direct_url.json") or "{}")
    archive = direct.get("archive_info", {})
    digest = (archive.get("hashes") or {}).get("sha256") or str(archive.get("hash", "")).removeprefix("sha256=")
    if direct.get("url") != PUBLISHER_URL or digest != PUBLISHER_SHA256:
        raise RuntimeError("The private audio runtime does not match the pinned publisher source.")
    import torch
    import soundfile  # noqa: F401
    import torchaudio  # noqa: F401
    from transformers import T5GemmaEncoderModel  # noqa: F401
    from stable_audio_3 import StableAudioModel
    from stable_audio_3.loading_utils import load_diffusion_cond
    from stable_audio_3.inference.sampling import sample_diffusion
    if "callback" not in inspect.signature(sample_diffusion).parameters:
        raise RuntimeError("The publisher audio sampler does not expose the expected progress callback.")
    _check_torch_build(torch, "cpu" if cpu_probe else torch_index)
    return torch, StableAudioModel, load_diffusion_cond


def _device(torch, requested="auto", torch_index=None):
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("SMARTAI_AUDIO_DEVICE must be auto, cpu, cuda, or mps.")
    if requested == "auto" and str(torch_index).startswith("cu"):
        requested = "cuda"
    cuda = torch.cuda.is_available()
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if requested == "cuda" and not cuda or requested == "mps" and not mps:
        raise RuntimeError(f"The explicitly selected audio device {requested} is unavailable.")
    return ("cuda" if cuda else "mps" if mps else "cpu") if requested == "auto" else requested


def _native_probe(cpu_probe=False, torch_index=None):
    torch, _, _ = _native_components(cpu_probe, torch_index)
    device = "cpu" if cpu_probe else _device(torch, os.getenv("SMARTAI_AUDIO_DEVICE", "auto").lower(), torch_index)
    layer = torch.nn.Linear(8, 4).to(device)
    result = layer(torch.ones((1, 8), device=device))
    if tuple(result.shape) != (1, 4) or not bool(torch.isfinite(result).all()):
        raise RuntimeError("The native audio tensor probe failed.")
    print(f"Stable Audio publisher imports, T5Gemma, audio I/O and {device} tensor probe passed; no model weights loaded.", flush=True)


def _local_audio_config(root):
    config = json.loads((root / "model_config.json").read_text(encoding="utf-8"))
    for conditioner in config.get("model", {}).get("conditioning", {}).get("configs", []):
        options = conditioner.get("config") or {}
        if conditioner.get("type") == "t5gemma":
            if options.get("subfolder") != "t5gemma-b-b-ul2" or not (root / "t5gemma-b-b-ul2/model.safetensors").is_file():
                raise RuntimeError("The pinned audio text conditioner is incomplete.")
            options["repo_id"] = options["model_path"] = str(root)
    return config


def _event(value, message):
    print(_EVENT_PREFIX + json.dumps({"value": value, "message": message}), flush=True)


def _native_worker(request):
    if not all(os.getenv(name) == "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise RuntimeError("Audio generation must use the already downloaded local snapshot offline.")
    torch, StableAudioModel, load_diffusion_cond = _native_components(torch_index=request.get("torch_index"))
    root = Path(request["model_path"]).resolve()
    for name in request["required_files"]:
        if not (root / name).is_file() or (root / name).stat().st_size == 0:
            raise RuntimeError(f"The selected audio artifact is missing: {name}")
    config = _local_audio_config(root)
    device = _device(torch, request["device"], request.get("torch_index"))
    half = device == "cuda"
    _event(0.2, f"Loading the selected Stable Audio model and local text conditioner on {device.upper()}")
    model = load_diffusion_cond(config, str(root / "model.safetensors"), device=device, model_half=half)
    model.use_lora, model.lora_names = False, []
    engine = StableAudioModel(model, config, device, half)
    if request["lora_files"]:
        engine.load_lora(request["lora_files"])
    count = request["steps"]
    def step_progress(state):
        step = min(count, int(state["i"]) + 1)
        _event(0.35 + 0.5 * step / count, f"Generating audio: denoising step {step}/{count}")
    _event(0.35, "Generating the waveform from the selected model")
    audio = engine.generate(prompt=request["prompt"], duration=request["duration"], steps=count,
                            cfg_scale=request["cfg_scale"], seed=request["seed"], callback=step_progress)
    import numpy as np
    import soundfile as sf
    samples = audio.detach().float().cpu().numpy()
    if samples.ndim != 3 or samples.shape[0] != 1 or samples.shape[1] not in {1, 2}:
        raise RuntimeError("Stable Audio returned an unexpected waveform shape.")
    samples = samples[0].T
    if samples.size == 0 or not np.isfinite(samples).all():
        raise RuntimeError("Stable Audio returned an empty or non-finite waveform.")
    _event(0.92, "Saving the completed audio waveform")
    sf.write(request["output_path"], samples, int(config["sample_rate"]), format="WAV", subtype="PCM_16")


def prepare_stable_audio_model(model_info, *, lora_files=(), cancel_event=None, progress_callback=None):
    """Resolve the unchanged catalog selection and prepare its private runtime."""
    from config.media_models import media_download_metadata
    from core.hf_downloader import download_model_from_hf
    from core.hf_transfer import snapshot_download
    _check_cancelled(cancel_event)
    repo = str(model_info.get("repo_id") or "")
    manifest = media_download_metadata(repo)
    if repo not in MODEL_REPOS or model_info.get("repo_revision") != manifest.get("repo_revision"):
        raise RuntimeError("The native audio adapter requires the exact selected publisher repository and revision.")
    required = manifest["download_required_files"]
    loras = [str(Path(path).resolve()) for path in lora_files]
    if any(not Path(path).is_file() or Path(path).stat().st_size == 0 for path in loras):
        raise RuntimeError("A selected learned audio adapter is missing or empty.")
    python = ensure_stable_audio_runtime(cancel_event=cancel_event, progress_callback=progress_callback)
    root = None
    try:
        cached = Path(snapshot_download(repo_id=repo, revision=manifest["repo_revision"],
                                       allow_patterns=required, local_files_only=True, cancel_event=cancel_event))
        if all((cached / name).is_file() and (cached / name).stat().st_size > 0 for name in required):
            root = cached
    except Exception:
        _check_cancelled(cancel_event)
    if root is None:
        result = download_model_from_hf(repo, revision=manifest["repo_revision"],
                                       allow_patterns=manifest["download_allow_patterns"], required_files=required,
                                       cancel_event=cancel_event,
                                       progress_callback=lambda message, percent: _notify(progress_callback, 0.12 + percent * 0.0007, message))
        if result.get("status") == "cancelled":
            raise StableAudioCancelled(result.get("error") or "Audio download cancelled.")
        if result.get("status") != "success":
            raise RuntimeError(result.get("error") or "Audio model download failed.")
        root = Path(result["local_dir"])
    _check_cancelled(cancel_event)
    return {"python": str(python), "model_path": str(root.resolve()), "required_files": required,
            "repo_id": repo, "revision": manifest["repo_revision"], "lora_files": loras,
            "seed": int(model_info.get("seed", -1)), "torch_index": _torch_index()}


def inspect_waveform(path):
    """Validate actual PCM frames and return a bounded preview of those samples."""
    with wave.open(str(path), "rb") as handle:
        channels, sample_width, rate, frames = handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getnframes()
        if channels not in {1, 2} or sample_width != 2 or not 8000 <= rate <= 192000 or frames <= 0:
            raise RuntimeError("The audio runtime did not produce a valid PCM waveform.")
        remaining = frames
        while remaining:
            count = min(65536, remaining)
            if len(handle.readframes(count)) != count * channels * sample_width:
                raise RuntimeError("The audio runtime produced a truncated WAV file.")
            remaining -= count
        count = min(2048, frames)
        samples = []
        for i in range(count):
            handle.setpos(i * (frames - 1) // max(1, count - 1))
            values = struct.unpack("<" + "h" * channels, handle.readframes(1))
            samples.append(sum(values) / channels / 32768.0)
    return {"kind": "audio", "samples": samples, "sample_rate": rate,
            "duration_seconds": frames / rate, "label": "Completed audio waveform"}


def generate_stable_audio(prepared, prompt, output_path, *, duration=10.0, steps=50, cfg_scale=5.0,
                          progress_callback=None, preview_callback=None, cancel_event=None):
    try:
        from core.hf_transfer import hf_subprocess_command
        _check_cancelled(cancel_event)
        duration, steps, cfg_scale = float(duration), int(steps), float(cfg_scale)
        if not math.isfinite(duration) or not 0 < duration <= 120:
            raise ValueError("Stable Audio Small supports a duration above zero and at most 120 seconds.")
        if not 1 <= steps <= 500 or not math.isfinite(cfg_scale) or cfg_scale < 0:
            raise ValueError("Audio generation requires 1–500 steps and a finite, nonnegative guidance scale.")
        if not str(prompt or "").strip():
            raise ValueError("Enter an audio prompt before generation.")
        destination = Path(output_path).absolute()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".smartai-audio-", dir=destination.parent) as job:
            temporary = Path(job) / "generated.wav"
            request = {**prepared, "prompt": str(prompt), "duration": duration, "steps": steps,
                       "cfg_scale": cfg_scale, "device": os.getenv("SMARTAI_AUDIO_DEVICE", "auto").lower(),
                       "output_path": str(temporary)}
            request_path = Path(job) / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            command = [prepared["python"], "-I", str(Path(__file__).resolve()), "--worker", str(request_path)]
            _run_command(hf_subprocess_command(command), env=_child_env(offline=True), label="Stable Audio generation",
                         progress_callback=progress_callback, cancel_event=cancel_event, timeout=6 * 60 * 60)
            _check_cancelled(cancel_event)
            preview = inspect_waveform(temporary)
            if abs(preview["duration_seconds"] - duration) > max(0.05, duration * 0.01):
                raise RuntimeError("The generated WAV does not have the requested audio duration.")
            if preview_callback is not None:
                try:
                    preview_callback(preview)
                except Exception:
                    _check_cancelled(cancel_event)
            _check_cancelled(cancel_event)
            os.replace(temporary, destination)
        _notify(progress_callback, 1.0, "Audio complete")
        return {"status": "success", "path": str(destination), "backend": "stable_audio_3_isolated"}
    except Exception as error:
        cancelled = isinstance(error, StableAudioCancelled) or (cancel_event is not None and cancel_event.is_set())
        return {"status": "cancelled" if cancelled else "error", "error": str(error), "backend": "stable_audio_3_isolated"}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Install the private publisher runtime without model weights")
    parser.add_argument("--probe", action="store_true", help="Repeat the native dependency/API check")
    parser.add_argument("--cpu-probe", action="store_true", help="Use a separate CPU-only verification environment")
    parser.add_argument("--native-probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--torch-index", choices=("cpu", "cu118", "cu126", "cu128", "pypi"), help=argparse.SUPPRESS)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.native_probe:
        _native_probe(args.cpu_probe, args.torch_index)
    elif args.worker:
        if args.cpu_probe:
            parser.error("The CPU verification environment is not a generation runtime.")
        _native_worker(json.loads(Path(args.worker).read_text(encoding="utf-8")))
    elif args.prepare:
        print(ensure_stable_audio_runtime(force_probe=args.probe, cpu_probe=args.cpu_probe,
                                         progress_callback=lambda value, message: print(message, flush=True)), flush=True)
    else:
        parser.error("Use --prepare --probe to verify the publisher runtime without loading model weights.")


if __name__ == "__main__":
    main()
