"""Pinned publisher MLX video pipelines in private Apple Silicon runtimes.

The selected model files stay in Studio's shared Hub cache. Only code and wheels
are installed in the private environments; generation workers run offline. LTX
uses its own distilled transformer, Gemma tower, VAEs and upscaler. MiniMax uses
the selected Q4 transformer, original Ref2VA companions and AfterMidnight LoRA.
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
import platform
import queue
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile


MODEL_CONTRACTS = {
    "ltx_mlx": ("dgrauet/ltx-2.5-mlx-q4", "e9a20add8c5937fcab635fb8e7a52926487a5776"),
    "minimax_mlx": ("pipenetwork/MiniMax-H3-MLX-4bit", "95a0e7d97f23f8b0029afea9411f13535262d935"),
}
MINIMAX_COMPANIONS = {
    "upstream": ("MiniMaxAI/MiniMax-H3", "12253298adb81e36757eabebb81d2ba1c0cc70cc"),
    "lora": ("SexGod1979/AfterMidnight-MiniMax-H3-NSFW", "4b325f60229c136b97501f5830bb277aace738b6"),
}
AFTERMIDNIGHT_FILE = "AfterMidnight_ref2va_h3_sexytime_rank64-v1.2.safetensors"
SOURCES = {
    "ltx_mlx": {
        "repo": "dgrauet/ltx-2-mlx", "commit": "e5db9dd6935d3a2ff00108853d3ccc0f919154f6",
        "sha256": "6444c7e258eebd274d542b3ed2f9e400ef36037c2462e391bb03436680d83e3d",
    },
    "minimax_mlx": {
        "repo": "PipeNetwork/minimax-h3-mlx", "commit": "b2f7e4d2b7861cefe68b75e4b59ab81cc4e7c318",
        "sha256": "071479d92621c79f204cbd7b4bb1783344e1117ffd942d5a7285771c0f299e63",
    },
}
# LTX pins follow the publisher's uv.lock. MiniMax's Qwen conditioner uses the
# same verified mlx-vlm/Transformers interface as Studio, in its own environment.
DEPENDENCIES = {
    "ltx_mlx": (
        "mlx==0.32.2", "mlx-lm==0.31.1", "mlx-arsenal==0.2.4", "transformers==5.3.0",
        "huggingface-hub==1.7.1", "numpy==2.4.3", "safetensors==0.7.0",
        "pillow==12.1.1", "tqdm==4.67.3", "imageio==2.37.0", "imageio-ffmpeg==0.6.0", "psutil==7.0.0",
    ),
    "minimax_mlx": (
        "mlx==0.32.2", "mlx-lm==0.31.3", "mlx-vlm==0.6.3", "mlx-audio==0.4.6", "transformers==5.5.0",
        "huggingface-hub==1.7.1", "numpy==2.4.3", "safetensors==0.7.0",
        "pillow==12.1.1", "scipy==1.16.3", "imageio==2.37.0", "imageio-ffmpeg==0.6.0", "psutil==7.0.0",
    ),
}
_EVENT_PREFIX = "SMARTAI_MLX_VIDEO_EVENT "
_PROCESSES = {}
_PROCESS_LOCK = threading.Lock()


class MLXVideoCancelled(RuntimeError):
    pass


def _check_cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise MLXVideoCancelled("Video generation cancelled; completed model files were kept.")


def _notify(callback, value, message):
    if callback is not None:
        callback(float(value), str(message))


def _backend(value):
    if isinstance(value, dict):
        kind = value.get("video_backend")
        if kind not in MODEL_CONTRACTS:
            kind = next((name for name, pair in MODEL_CONTRACTS.items() if pair[0] == value.get("repo_id")), None)
        if kind not in MODEL_CONTRACTS or (value.get("repo_id"), value.get("repo_revision")) != MODEL_CONTRACTS[kind]:
            raise ValueError("The MLX video adapter requires the exact selected publisher repository and revision.")
        if kind == "minimax_mlx":
            selected = {item.get("role"): item for item in value.get("download_components", [])}
            for role, pair in MINIMAX_COMPANIONS.items():
                item = selected.get(role, {})
                if (item.get("repo_id"), item.get("revision")) != pair:
                    raise ValueError(f"The MiniMax {role} component does not match the selected pinned release.")
            if selected["lora"].get("required_files") != [AFTERMIDNIGHT_FILE]:
                raise ValueError("The MiniMax preset requires its selected AfterMidnight v1.2 LoRA.")
        return kind
    if value not in MODEL_CONTRACTS:
        raise ValueError(f"Unknown MLX video backend: {value}")
    return value


def _require_platform():
    if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        raise RuntimeError("These selected MLX video models require an Apple Silicon Mac.")


def _memory_estimate(kind, info, *, frames=None, width=None, height=None):
    defaults = (33, 768, 512, 20.0, 4.0) if kind == "ltx_mlx" else (124, 1344, 768, 64.0, 9.3)
    count, w, h, baseline, activation = defaults
    volume = (int(frames or info.get("frames", count)) * int(width or info.get("width", w))
              * int(height or info.get("height", h))) / (count * w * h)
    # A conservative admission estimate, not a measured full-checkpoint peak.
    # MiniMax's 50.3 GB original text tower is never resident with its DiT/VAEs.
    return max(baseline, float(info.get("estimated_peak_memory_gb") or baseline)) + activation * max(0.0, volume - 1.0)


def validate_mlx_video_memory(model_info, *, frames=None, width=None, height=None):
    """Reject a clearly insufficient memory budget before a large download/load."""
    import psutil
    kind = _backend(model_info)
    required = _memory_estimate(kind, model_info, frames=frames, width=width, height=height)
    available = psutil.virtual_memory().available / 1024 ** 3
    if available < required:
        label = "LTX-2.5" if kind == "ltx_mlx" else "MiniMax-H3 with its original text encoder"
        raise RuntimeError(f"{label} needs an estimated {required:.1f} GiB of available memory for this request; "
                           f"{available:.1f} GiB is available. Completed model files were kept. "
                           "Use a Mac with more memory or close other models and applications.")
    return required


def _runtime_contract(kind):
    return {"schema": 1, "backend": kind, "source": SOURCES[kind],
            "dependencies": list(DEPENDENCIES[kind]), "base_python": os.path.realpath(sys.executable),
            "python_version": list(sys.version_info[:3])}


def _runtime_directory(kind):
    from config.paths import get_portable_data_dir
    digest = hashlib.sha256(json.dumps(_runtime_contract(kind), sort_keys=True).encode()).hexdigest()[:20]
    return Path(get_portable_data_dir()) / "runtimes" / kind / digest


def _child_env(offline=False):
    env = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE", "VIRTUAL_ENV",
                 "PIP_TARGET", "PIP_PREFIX", "PIP_ROOT", "PIP_USER"):
        env.pop(name, None)
    env.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", SMARTAI_MLX_VIDEO_OWNER_PID=str(os.getpid()))
    if offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


def owned_process_ids():
    with _PROCESS_LOCK:
        return sorted(pid for pid, process in _PROCESSES.items() if process.poll() is None)


def _stop_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=3)


def _read_preview(path, label, callback, root):
    if callback is None:
        return
    from PIL import Image
    path = Path(path).resolve()
    if root is None or not path.is_relative_to(Path(root).resolve()):
        raise ValueError("A video preview must belong to its active generation job.")
    with Image.open(path) as source:
        picture = source.convert("RGB")
        picture.thumbnail((768, 512))
    callback({"kind": "video", "image": picture, "label": str(label)})


def _run_command(command, *, env, label, cancel_event=None, progress_callback=None,
                 preview_callback=None, preview_root=None, timeout=1800):
    _check_cancelled(cancel_event)
    process = subprocess.Popen([str(arg) for arg in command], env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                               bufsize=1, start_new_session=True)
    with _PROCESS_LOCK:
        _PROCESSES[process.pid] = process
    lines, tail = queue.Queue(), deque(maxlen=35)

    def read_output():
        try:
            for line in process.stdout:
                lines.put(line.rstrip()[-2500:])
        finally:
            process.stdout.close()

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout
    try:
        while process.poll() is None or reader.is_alive() or not lines.empty():
            _check_cancelled(cancel_event)
            if time.monotonic() > deadline:
                raise RuntimeError(f"{label} exceeded its time limit.")
            try:
                line = lines.get(timeout=0.1)
            except queue.Empty:
                continue
            if line.startswith(_EVENT_PREFIX):
                try:
                    event = json.loads(line[len(_EVENT_PREFIX):])
                    if "preview" in event:
                        _read_preview(event["preview"], event["message"], preview_callback, preview_root)
                    else:
                        _notify(progress_callback, event["value"], event["message"])
                except (OSError, ValueError, KeyError, TypeError) as error:
                    tail.append(f"Preview/progress event unavailable: {error}")
            else:
                tail.append(line)
                # MiniMax prints this only after mx.eval completes each real step.
                match = re.search(r"\bstep (\d+)/(\d+)\s+.*?eta", line)
                if match:
                    done, total = map(int, match.groups())
                    _notify(progress_callback, 0.30 + 0.58 * done / max(1, total),
                            f"Denoising {done}/{total}")
        if process.wait():
            raise RuntimeError(f"{label} failed:\n" + "\n".join(tail))
        _check_cancelled(cancel_event)
    except BaseException:
        _stop_process(process)
        raise
    finally:
        reader.join(timeout=1)
        with _PROCESS_LOCK:
            _PROCESSES.pop(process.pid, None)


@contextmanager
def _setup_lock(path, cancel_event):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        while True:
            _check_cancelled(cancel_event)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _prepare_source(directory, kind, cancel_event):
    """Download a small pinned code archive, verify it, then expose it atomically."""
    target = directory / "publisher"
    spec = SOURCES[kind]
    if target.exists():
        if _read_json(target / "smartai-source.json") != spec:
            raise RuntimeError("The private publisher-code folder contains unrecognized files.")
        return target
    url = f"https://codeload.github.com/{spec['repo']}/zip/{spec['commit']}"
    with tempfile.TemporaryDirectory(prefix="source-", dir=directory) as staging:
        staging = Path(staging)
        archive = staging / "source.zip"
        digest, size = hashlib.sha256(), 0
        with urllib.request.urlopen(url, timeout=30) as response, archive.open("wb") as output:
            while chunk := response.read(256 * 1024):
                _check_cancelled(cancel_event)
                size += len(chunk)
                if size > 64 * 1024 * 1024:
                    raise RuntimeError("The publisher code archive exceeded its expected size.")
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != spec["sha256"]:
            raise RuntimeError("The publisher source archive failed SHA-256 verification.")
        with zipfile.ZipFile(archive) as bundle:
            for entry in bundle.infolist():
                path = (staging / entry.filename).resolve()
                if not path.is_relative_to(staging.resolve()) or stat.S_ISLNK(entry.external_attr >> 16):
                    raise RuntimeError("The publisher code archive contains an invalid path.")
            bundle.extractall(staging)
        source = staging / f"{spec['repo'].split('/')[-1]}-{spec['commit']}"
        if not source.is_dir():
            raise RuntimeError("The publisher source archive has an unexpected layout.")
        _write_json(source / "smartai-source.json", spec)
        _check_cancelled(cancel_event)
        source.rename(target)
    return target


def ensure_mlx_video_runtime(model_info_or_backend, *, progress_callback=None, cancel_event=None, force_probe=False):
    """Prepare private code/dependencies automatically, without model downloads."""
    kind = _backend(model_info_or_backend)
    _require_platform()
    if isinstance(model_info_or_backend, dict):
        validate_mlx_video_memory(model_info_or_backend)
    if sys.version_info < (3, 11):
        raise RuntimeError("The video runtime requires Studio's bundled Python 3.11 or newer.")
    directory = _runtime_directory(kind)
    python = directory / "bin/python"
    owner, ready = directory / "smartai-runtime-owner.json", directory / "smartai-runtime-ready.json"
    contract = _runtime_contract(kind)
    with _setup_lock(directory.with_suffix(".lock"), cancel_event):
        if directory.is_symlink():
            raise RuntimeError("The private video runtime folder must not be a symlink.")
        if directory.exists() and any(directory.iterdir()) and _read_json(owner) != contract:
            raise RuntimeError(f"The reserved video runtime folder contains unrecognized files: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        if not owner.exists():
            _write_json(owner, contract)
        usable = python.is_file() and _read_json(ready) == contract
        if not usable:
            _notify(progress_callback, 0.03, "Preparing the selected publisher's private video runtime")
            if not python.is_file():
                _run_command([sys.executable, "-I", "-m", "venv", str(directory)], env=_child_env(),
                             label="Private video Python setup", cancel_event=cancel_event)
            source = _prepare_source(directory, kind, cancel_event)
            packages = list(DEPENDENCIES[kind])
            if kind == "ltx_mlx":
                packages += [str(source / "packages/ltx-core-mlx"), str(source / "packages/ltx-pipelines-mlx")]
            _notify(progress_callback, 0.07, "Installing the pinned video dependencies in their private environment")
            _run_command([str(python), "-I", "-m", "pip", "--isolated", "install", "--no-input",
                          "--disable-pip-version-check", "--retries", "2", "--timeout", "45",
                          "--only-binary=:all:", "--no-binary=ltx-core-mlx,ltx-pipelines-mlx", *packages],
                         env=_child_env(), label="Publisher video runtime installation", cancel_event=cancel_event)
        if not usable or force_probe:
            _notify(progress_callback, 0.12, "Checking publisher APIs, native computation and the bundled video encoder")
            try:
                _run_command([str(python), "-I", str(Path(__file__).resolve()), "--native-probe", kind,
                              "--runtime-dir", str(directory)], env=_child_env(offline=True),
                             label="Native video runtime validation", cancel_event=cancel_event, timeout=180)
            except BaseException:
                ready.unlink(missing_ok=True)
                raise
            _write_json(ready, contract)
    return python


def _activate_runtime(kind, directory):
    from importlib import metadata
    source = Path(directory) / "publisher"
    if _read_json(source / "smartai-source.json") != SOURCES[kind]:
        raise RuntimeError("The selected video runtime does not match its verified publisher source.")
    for requirement in DEPENDENCIES[kind]:
        name, version = requirement.split("==")
        if metadata.version(name) != version:
            raise RuntimeError(f"The private video runtime needs {requirement}.")
    if kind == "minimax_mlx":
        sys.path.insert(0, str(source))
    else:
        for package in ("ltx-core-mlx", "ltx-pipelines-mlx"):
            if metadata.version(package) != "0.18.0":
                raise RuntimeError("The private LTX pipeline version does not match the pinned source.")
    import imageio_ffmpeg
    tools = Path(directory) / "tools"
    tools.mkdir(exist_ok=True)
    ffmpeg = tools / "ffmpeg"
    target = imageio_ffmpeg.get_ffmpeg_exe()
    if ffmpeg.is_symlink():
        if os.readlink(ffmpeg) != target:
            ffmpeg.unlink()
    elif ffmpeg.exists():
        raise RuntimeError("The private ffmpeg path contains an unrecognized file.")
    if not ffmpeg.exists():
        ffmpeg.symlink_to(target)
    os.environ["PATH"] = str(tools) + os.pathsep + os.environ.get("PATH", "")
    return source


def _event(value, message, preview=None):
    data = {"value": float(value), "message": str(message)}
    if preview is not None:
        data["preview"] = str(preview)
    print(_EVENT_PREFIX + json.dumps(data), flush=True)


def _request(info, output_path, *, steps=None, frames=None, fps=None, width=None, height=None, seed=None,
             previews=False, validate_files=True):
    kind = _backend(info)
    def setting(explicit, name, fallback):
        return explicit if explicit is not None else info.get(name, fallback)
    width = int(setting(width, "width", 768 if kind == "ltx_mlx" else 1344))
    height = int(setting(height, "height", 512 if kind == "ltx_mlx" else 768))
    frames = int(setting(frames, "frames", 33 if kind == "ltx_mlx" else 124))
    fps = float(setting(fps, "fps", 24))
    steps = int(setting(steps, "inference_steps", info.get("steps", 8 if kind == "ltx_mlx" else 16)))
    multiple = 64 if kind == "ltx_mlx" else 32
    if any(value < 128 or value > 2048 or value % multiple for value in (width, height)):
        raise ValueError(f"{kind} video dimensions must be multiples of {multiple} from 128 to 2048.")
    if not math.isfinite(fps) or not 1 <= fps <= 60 or (kind == "minimax_mlx" and fps != 24):
        raise ValueError("MiniMax-H3 uses 24 fps; LTX supports frame rates from 1 to 60.")
    if not 2 <= frames <= 721:
        raise ValueError("Video generation needs between 2 and 721 frames.")
    if kind == "ltx_mlx":
        frames = 1 + math.ceil((frames - 1) / 8) * 8
        if not 1 <= steps <= 8:
            raise ValueError("The selected distilled LTX checkpoint uses 1–8 first-stage steps and 3 refinement steps.")
    else:
        frames += (5 - frames) % 17
        if not 1 <= steps <= 100:
            raise ValueError("MiniMax generation needs between 1 and 100 denoising steps.")
    selected = [{"role": "model", "required_files": info["download_required_files"]},
                *info.get("download_components", [])]
    roots = {"model": str(Path(info.get("path") or info.get("model_path") or "").resolve()),
             **dict(info.get("component_paths") or {})}
    for component in selected if validate_files else ():
        root = Path(roots.get(component["role"], ""))
        if not root.is_dir() or any(not (root / name).is_file() or (root / name).stat().st_size <= 0
                                    for name in component["required_files"]):
            raise RuntimeError(f"The selected {component['role']} video component is incomplete.")
    return {"backend": kind, "repo_id": info["repo_id"], "revision": info["repo_revision"],
            "roots": roots, "selected": selected, "output_path": str(Path(output_path).resolve()),
            "frames": frames, "fps": fps, "steps": steps, "width": width, "height": height,
            "estimated_peak_memory_gb": _memory_estimate(kind, info, frames=frames, width=width, height=height),
            "seed": int(setting(seed, "seed", 42)), "previews": bool(previews),
            "image_path": info.get("image_path"), "lora_strength": float(info.get("lora_strength", 1.0))}


class _LTXProgress:
    def __init__(self, request):
        self.request, self.last_preview = request, 0.0

    def bind(self, **geometry):
        request = self.request
        stage = geometry.get("stage") or 1
        decode = None
        directory = Path(request["output_path"]).parent / "previews"
        if request["previews"]:
            from ltx_pipelines_mlx.utils.stepwise import StepwiseConfig, StepwisePreview
            directory.mkdir(exist_ok=True)
            decode = StepwisePreview(StepwiseConfig(output_dir=directory, interval=4, frames=1,
                                     frame_rate=request["fps"], seed=request["seed"])).bind(**geometry)

        def on_step(index, total, prediction, sigma):
            value = (0.28 if stage == 1 else 0.66) + (0.38 if stage == 1 else 0.23) * (index + 1) / total
            label = f"Denoising stage {stage}: {index + 1}/{total}"
            _event(value, label)
            if decode is not None and time.monotonic() - self.last_preview >= 1.0:
                decode(index, total, prediction, sigma)
                tag = "" if geometry.get("stage") is None else f"_s{stage}"
                preview = directory / f"seed_{request['seed']}{tag}_step{index + 1:03d}of{total:03d}.webp"
                if preview.is_file():
                    self.last_preview = time.monotonic()
                    _event(value, label + " — decoded intermediate frame", preview)
        return on_step


def _ltx_generate(request):
    from ltx_pipelines_mlx import DistilledPipeline
    from ltx_pipelines_mlx.utils.generation import is_ltx25_pack
    root = request["roots"]["model"]
    if not is_ltx25_pack(root):
        raise RuntimeError("The selected local LTX pack does not declare the LTX-2.5 architecture.")
    _event(0.18, "Loading LTX-2.5 with its local Gemma text encoder and streamed Q4 transformer")
    pipeline = DistilledPipeline(model_dir=root, low_memory=True, low_ram_streaming=True)
    pipeline.stepwise = _LTXProgress(request)
    pipeline.generate_audio = True
    pipeline.generate_and_save(
        prompt=request["prompt"], output_path=request["output_path"],
        width=request["width"], height=request["height"], num_frames=request["frames"],
        frame_rate=request["fps"], seed=request["seed"], stage1_steps=request["steps"], stage2_steps=3,
        image=request.get("image_path"),
    )


def _lora_groups(weights):
    """Parse the original publisher's Kohya-style keys without guessing targets."""
    groups = {}
    pattern = re.compile(r"^lora_unet_blocks_(\d+)_(attn_(?:qkv_proj|out_proj)|mlp_(?:fc1|fc2))\.(alpha|lora_down\.weight|lora_up\.weight)$")
    for name, value in weights.items():
        match = pattern.fullmatch(name)
        if match is None:
            raise ValueError(f"Unsupported AfterMidnight LoRA tensor: {name}")
        index, target, part = match.groups()
        groups.setdefault((int(index), target.replace("attn_", "attn.").replace("mlp_", "mlp.")), {})[part] = value
    if not groups or any(set(parts) != {"alpha", "lora_down.weight", "lora_up.weight"} for parts in groups.values()):
        raise ValueError("The selected AfterMidnight LoRA has an incomplete tensor pair.")
    return groups


def _attach_aftermidnight(model, weights, strength=1.0):
    import mlx.core as mx
    import mlx.nn as nn
    groups = _lora_groups(weights)
    expected = {(index, target) for index in range(len(model.blocks))
                for target in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")}
    if set(groups) != expected:
        raise ValueError("The AfterMidnight LoRA does not cover exactly the selected transformer's blocks.")
    if not math.isfinite(strength) or not 0 < strength <= 2:
        raise ValueError("AfterMidnight strength must be greater than 0 and at most 2.")

    class LoRALinear(nn.Module):
        def __init__(self, base, down, up, scale):
            super().__init__()
            self.base, self.down, self.up, self.scale = base, down, up, scale

        def __call__(self, inputs):
            original = self.base(inputs)
            delta = (inputs.astype(mx.float32) @ self.down.T) @ self.up.T
            return (original.astype(mx.float32) + self.scale * delta).astype(original.dtype)

    replacements = []
    for (index, target), parts in groups.items():
        parent_name, name = target.split(".")
        parent = getattr(model.blocks[index], parent_name)
        base = getattr(parent, name)
        down, up = parts["lora_down.weight"], parts["lora_up.weight"]
        input_dim = (base.scales.shape[-1] * base.group_size if hasattr(base, "scales") else base.weight.shape[1])
        if down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1] or down.shape[1] != input_dim or up.shape[0] != base.weight.shape[0]:
            raise ValueError(f"AfterMidnight tensor dimensions do not match block {index} {target}.")
        alpha = float(parts["alpha"].item())
        if not math.isfinite(alpha):
            raise ValueError("The AfterMidnight LoRA has an invalid alpha.")
        replacements.append((parent, name, LoRALinear(base, down.astype(mx.float32), up.astype(mx.float32),
                                                    strength * alpha / down.shape[0])))
    for parent, name, replacement in replacements:
        setattr(parent, name, replacement)
    return len(replacements)


def _beta_sigmas(steps, shift):
    """Euler beta schedule, retaining MiniMax's separate video/audio flow shifts.

    Beta(0.6, 0.6) quantiles follow the beta sampling schedule used by the LoRA
    publisher's recommended workflow; the terminal zero is explicit.
    """
    import numpy as np
    from scipy.stats import beta
    grid = beta.ppf(1.0 - np.linspace(0.0, 1.0, int(steps), endpoint=False), 0.6, 0.6)
    sigmas = shift * grid / (1.0 + (shift - 1.0) * grid)
    return [float(value) for value in sigmas] + [0.0]


def _save_minimax_mp4(path, video, fps, audio, sample_rate):
    """Mux the actual generated streams without truncating a final video frame.

    The publisher's writer uses ffmpeg ``-shortest``. At 24 fps its rounded
    32 kHz audio duration can end just before the last frame, dropping that
    frame. Feeding every generated frame without that switch retains both
    complete streams; the normal final MP4 validation checks the frame count.
    """
    import imageio.v2 as imageio
    import numpy as np
    import wave
    video, audio = np.asarray(video), np.asarray(audio)
    if video.ndim != 4 or video.shape[-1] != 3 or len(video) < 2 or video.dtype != np.uint8:
        raise RuntimeError("MiniMax did not return a complete RGB video sequence.")
    if (audio.ndim != 2 or audio.shape[0] != 2 or audio.shape[1] == 0
            or not np.isfinite(audio).all() or int(sample_rate) != 32000):
        raise RuntimeError("MiniMax did not return valid generated stereo audio at 32 kHz.")
    audio_path = Path(path).with_suffix(".wav")
    with wave.open(str(audio_path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(int(sample_rate))
        stream.writeframes((np.clip(audio.T, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
    with imageio.get_writer(str(path), format="ffmpeg", fps=fps, codec="libx264", quality=None,
                            audio_path=str(audio_path), audio_codec="aac",
                            output_params=["-crf", "18", "-movflags", "+faststart"]) as writer:
        for frame in video:
            writer.append_data(frame)


def _minimax_generate(request):
    import gc
    import mlx.core as mx
    from minimax_h3_mlx.config import PipelineConfig
    from minimax_h3_mlx.load import load_audio_vae, load_dit, load_video_vae
    from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline
    from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler
    from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder

    upstream = Path(request["roots"]["upstream"]) / "Ref2VA"
    images = None
    if request.get("image_path"):
        from PIL import Image, ImageOps
        with Image.open(request["image_path"]) as image:
            images = [ImageOps.exif_transpose(image).convert("RGB")]
    _event(0.17, "Encoding the prompt with the original MiniMax Qwen3-VL conditioner")
    encoder = MiniMaxH3TextEncoder(upstream / "text_encoder", load_vision=bool(images))
    encoded = encoder.encode(request["prompt"], images)
    mx.eval(encoded[0])
    del encoder
    gc.collect()
    mx.clear_cache()

    class EncodedPrompt:
        def encode(self, prompt, requested_images=None):
            if prompt != request["prompt"] or requested_images is not images:
                raise ValueError("The MiniMax prompt changed after local conditioning.")
            return encoded

    class AfterMidnightPipeline(MiniMaxH3Pipeline):
        def _build_schedules(self, num_inference_steps):
            schedules = []
            for shift in (self.config.sigma_shift_video, self.config.sigma_shift_audio):
                scheduler = MiniMaxH3Scheduler(shift=shift)
                scheduler.set_timesteps(sigmas=_beta_sigmas(num_inference_steps - 1, shift))
                schedules.append(scheduler)
            return tuple(schedules)

        def _decode_video(self, *args):
            _event(0.90, "Decoding generated video latents with the original MiniMax video VAE")
            return super()._decode_video(*args)

        def _decode_audio(self, *args):
            _event(0.93, "Decoding the synchronized generated audio")
            return super()._decode_audio(*args)

    _event(0.22, "Loading the selected MiniMax Q4 transformer and AfterMidnight v1.2 LoRA")
    dit = load_dit(request["roots"]["model"])
    weights = mx.load(str(Path(request["roots"]["lora"]) / AFTERMIDNIGHT_FILE))
    count = _attach_aftermidnight(dit, weights, request["lora_strength"])
    del weights
    _event(0.26, f"Applied AfterMidnight to {count} transformer layers; loading original audio/video VAEs")
    pipeline = AfterMidnightPipeline(dit, EncodedPrompt(), load_video_vae(upstream / "video_vae"),
                                    load_audio_vae(upstream / "audio_vae"), PipelineConfig.from_model_index(upstream / "model_index.json"))
    result = pipeline(request["prompt"], duration_seconds=request["frames"] / request["fps"],
                      num_inference_steps=request["steps"] + 1, seed=request["seed"],
                      height=request["height"], width=request["width"], images=images,
                      keyframe_anchors=("first",) if images else (), drop_adaln=True, verbose=True)
    _event(0.96, "Encoding the generated video and audio into MP4")
    _save_minimax_mp4(request["output_path"], result.video, result.fps, result.audio, result.sample_rate)


def _validate_video(path, request=None):
    import imageio.v2 as imageio
    import numpy as np
    path = Path(path)
    if path.suffix.lower() != ".mp4" or not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError("The publisher pipeline did not produce an MP4 file.")
    reader = imageio.get_reader(str(path), format="ffmpeg")
    try:
        metadata = reader.get_meta_data()
        first, second = reader.get_data(0), reader.get_data(1)
        if (first.ndim != 3 or first.shape[-1] != 3 or second.shape != first.shape
                or not np.isfinite(first).all() or not np.isfinite(second).all()
                or not math.isfinite(float(metadata.get("fps", 0))) or float(metadata.get("fps", 0)) <= 0):
            raise RuntimeError("The generated MP4 has invalid video frames.")
        if request and (tuple(first.shape[:2]) != (request["height"], request["width"])
                        or abs(float(metadata["fps"]) - request["fps"]) > 0.1):
            raise RuntimeError("The generated MP4 does not match the requested dimensions or frame rate.")
        if request and reader.count_frames() != request["frames"]:
            raise RuntimeError("The generated MP4 does not contain the complete requested frame sequence.")
        if request and request.get("backend") in SOURCES and not metadata.get("audio_codec"):
            raise RuntimeError("The generated MP4 is missing the selected pipeline's synchronized audio.")
        return first
    finally:
        reader.close()


def _native_worker(request, directory):
    if not all(os.environ.get(name) == "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise RuntimeError("Video workers must use the already downloaded local model components.")
    if (request["repo_id"], request["revision"]) != MODEL_CONTRACTS[request["backend"]]:
        raise RuntimeError("The worker received a different video-model selection.")
    import psutil
    available = psutil.virtual_memory().available / 1024 ** 3
    if available < request["estimated_peak_memory_gb"]:
        raise RuntimeError(f"The video worker needs an estimated {request['estimated_peak_memory_gb']:.1f} GiB "
                           f"available before loading weights; only {available:.1f} GiB is available.")
    owner = int(os.environ.get("SMARTAI_MLX_VIDEO_OWNER_PID") or 0)
    if owner:
        def watch_owner():
            while os.getppid() == owner:
                time.sleep(0.5)
            if os.getpgrp() == os.getpid():
                os.killpg(os.getpgrp(), signal.SIGTERM)
            os._exit(130)
        threading.Thread(target=watch_owner, daemon=True).start()
    _activate_runtime(request["backend"], directory)
    for selection in request["selected"]:
        root = Path(request["roots"][selection["role"]])
        if any(not (root / name).is_file() or (root / name).stat().st_size <= 0 for name in selection["required_files"]):
            raise RuntimeError(f"The selected {selection['role']} component changed before generation.")
    (_ltx_generate if request["backend"] == "ltx_mlx" else _minimax_generate)(request)
    _validate_video(request["output_path"], request)
    _event(0.99, "Generated MP4 verified")


def generate_mlx_video(model_info, prompt, output_path, *, steps=None, frames=None, fps=None,
                       width=None, height=None, seed=None, progress_callback=None, cancel_event=None,
                       preview_callback=None):
    """Generate, decode-check and publish an MP4 without changing cached weights."""
    kind = model_info.get("video_backend", "mlx_video")
    try:
        _check_cancelled(cancel_event)
        kind = _backend(model_info)
        _require_platform()
        if not str(prompt or "").strip():
            raise ValueError("Enter a video prompt before starting generation.")
        output = Path(output_path).absolute()
        if output.suffix.lower() != ".mp4":
            raise ValueError("Generated videos must be saved as MP4.")
        if output.exists():
            raise FileExistsError(f"The output file already exists: {output}")
        preflight = _request(model_info, output, steps=steps, frames=frames, fps=fps, width=width,
                             height=height, seed=seed, validate_files=False)
        validate_mlx_video_memory(model_info, frames=preflight["frames"], width=preflight["width"], height=preflight["height"])
        from core.media_assets import ensure_media_model_assets
        info = ensure_media_model_assets(model_info, progress_callback=progress_callback, cancel_event=cancel_event)
        python = ensure_mlx_video_runtime(info, progress_callback=progress_callback, cancel_event=cancel_event)
        jobs = python.parent.parent / "jobs"
        jobs.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="generation-", dir=jobs) as job:
            job = Path(job)
            request = _request(info, job / "generated.mp4", steps=steps, frames=frames, fps=fps,
                               width=width, height=height, seed=seed, previews=preview_callback is not None)
            request["prompt"] = str(prompt)
            request_file = job / "request.json"
            request_file.write_text(json.dumps(request), encoding="utf-8")
            _run_command([str(python), "-I", str(Path(__file__).resolve()), "--worker", str(request_file),
                          "--runtime-dir", str(python.parent.parent)], env=_child_env(offline=True),
                         label="MLX video generation", cancel_event=cancel_event,
                         progress_callback=progress_callback, preview_callback=preview_callback,
                         preview_root=job, timeout=48 * 60 * 60)
            picture = _validate_video(request["output_path"], request)
            _check_cancelled(cancel_event)
            output.parent.mkdir(parents=True, exist_ok=True)
            # The temporary copy is on the destination filesystem. Linking it
            # publishes atomically and refuses to replace any existing file.
            with tempfile.NamedTemporaryFile(prefix=".smartai-video-", suffix=".mp4", dir=output.parent) as staged:
                with open(request["output_path"], "rb") as source:
                    shutil.copyfileobj(source, staged)
                staged.flush()
                _check_cancelled(cancel_event)
                os.link(staged.name, output)
            if preview_callback is not None:
                from PIL import Image
                preview_callback({"kind": "video", "image": Image.fromarray(picture), "label": "Generated video"})
        return {"status": "success", "path": str(output), "backend": kind,
                "frames": request["frames"], "fps": request["fps"]}
    except MLXVideoCancelled as error:
        return {"status": "cancelled", "error": str(error), "backend": kind}
    except Exception as error:
        if cancel_event is not None and cancel_event.is_set():
            return {"status": "cancelled", "error": "Video generation cancelled.", "backend": kind}
        return {"status": "error", "error": str(error), "backend": kind}


def _probe_quant_lora():
    """Verify real residual LoRA arithmetic without dequantizing the Q4 base."""
    import mlx.core as mx
    import mlx.nn as nn
    model = nn.Module()
    block = nn.Module()
    block.attn, block.mlp = nn.Module(), nn.Module()
    model.blocks = [block]
    weights, expected, packed = {}, {}, {}
    inputs = mx.ones((1, 64))
    for target in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
        parent, name = target.split(".")
        layer = nn.Linear(64, 64, bias=False).to_quantized(group_size=64, bits=4)
        setattr(getattr(block, parent), name, layer)
        down, up = mx.ones((2, 64)) / 64, mx.ones((64, 2)) / 16
        prefix = "lora_unet_blocks_0_" + target.replace(".", "_")
        weights.update({prefix + ".alpha": mx.array(2.0), prefix + ".lora_down.weight": down,
                        prefix + ".lora_up.weight": up})
        expected[target] = layer(inputs) + 0.5 * ((inputs @ down.T) @ up.T)
        packed[target] = layer.weight
    if _attach_aftermidnight(model, weights, 0.5) != 4:
        raise RuntimeError("The quantized LoRA probe missed a transformer target.")
    for target, reference in expected.items():
        parent, name = target.split(".")
        layer = getattr(getattr(block, parent), name)
        actual = layer(inputs)
        mx.eval(actual, reference)
        if (not bool(mx.all(mx.isfinite(actual)).item())
                or not bool(mx.allclose(actual, reference, atol=1e-5).item())
                or layer.base.weight is not packed[target]):
            raise RuntimeError("The Q4 plus AfterMidnight residual probe failed.")


def _native_probe(kind, directory):
    source = _activate_runtime(kind, directory)
    import psutil
    if psutil.virtual_memory().total <= 0:
        raise RuntimeError("The private video runtime could not read the memory budget.")
    if kind == "ltx_mlx":
        from ltx_pipelines_mlx import DistilledPipeline
        from ltx_pipelines_mlx.utils.stepwise import StepwisePreview
        if not {"model_dir", "low_memory", "low_ram_streaming"} <= set(inspect.signature(DistilledPipeline).parameters):
            raise RuntimeError("The publisher LTX constructor does not match the adapter.")
        if not callable(StepwisePreview.bind):
            raise RuntimeError("The publisher LTX preview API is unavailable.")
    else:
        import runpy
        from mlx_vlm.models.qwen3_vl.language import Qwen3VLModel  # noqa: F401
        from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline
        if "transformer_dir" not in inspect.signature(MiniMaxH3Pipeline.from_pretrained).parameters:
            raise RuntimeError("The publisher MiniMax loader does not match the adapter.")
        # The publisher's tiny real DiT forward tests both audio and video
        # outputs without pretrained weights or a model download.
        runpy.run_path(str(source / "tests/test_dit_smoke.py"))["test_forward_shapes"]()
        if len(_beta_sigmas(4, 12.0)) != 5:
            raise RuntimeError("The AfterMidnight beta sampler is unavailable.")
    _probe_quant_lora()
    import numpy as np
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        path = Path(temporary) / "encoder-probe.mp4"
        video = np.stack([np.full((32, 32, 3), index * 10, np.uint8) for index in range(22)])
        audio = np.zeros((2, round(22 / 24 * 32000)), np.float32)
        _save_minimax_mp4(path, video, 24, audio, 32000)
        _validate_video(path, {"backend": kind, "width": 32, "height": 32, "frames": 22, "fps": 24})
    print(f"MLX_VIDEO_NATIVE_SMOKE {kind} passed; publisher APIs, native Q4 and MP4, no model weights", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", choices=tuple(SOURCES))
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--native-probe", choices=tuple(SOURCES), help=argparse.SUPPRESS)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--runtime-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.native_probe:
        _native_probe(args.native_probe, args.runtime_dir)
    elif args.worker:
        _native_worker(json.loads(Path(args.worker).read_text(encoding="utf-8")), args.runtime_dir)
    elif args.prepare:
        print(ensure_mlx_video_runtime(args.prepare, force_probe=args.probe), flush=True)
    else:
        parser.error("Use --prepare BACKEND --probe to validate the runtime without pretrained model weights.")


if __name__ == "__main__":
    main()
