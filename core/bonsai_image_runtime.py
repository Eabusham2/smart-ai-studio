"""Run the publisher's packed Bonsai image model in an isolated Mac runtime.

The publisher's mflux fork requires MLX < 0.32; the chat engine uses MLX 0.32.
Only the private environment below receives the older image dependency. Model
files remain in the application's shared, revision-pinned Hugging Face cache.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time


MODEL_REPO = "prism-ml/bonsai-image-ternary-4B-mlx-2bit"
MODEL_REVISION = "2c24c81b934a658ba5590cf39088ba929985b4a8"
MLX_VERSION = "0.31.0"
PUBLISHER_SOURCES = (
    (
        "mflux",
        "https://codeload.github.com/PrismML-Eng/mflux-prism/zip/bcd13e83b7fcfd76186c98ef322dd9cf28e996c1",
        "ef6a624c8399504537a499e0b85aaad06991e69ed44f4cf8e8b17506ad2cab57",
    ),
    (
        "prism-image-studio",
        "https://codeload.github.com/PrismML-Eng/image-studio/zip/31b02171634c16b5da0eec6aea075e7489d5fb39",
        "c200c5c4581739630628d1ee82555e3764b81ac0b5dd4af7a7f914c2c629b747",
    ),
)
_EVENT_PREFIX = "SMARTAI_BONSAI_IMAGE_PROGRESS "
_PROCESSES = {}
_PROCESS_LOCK = threading.Lock()


class BonsaiImageCancelled(RuntimeError):
    pass


def _check_cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise BonsaiImageCancelled("Bonsai image generation cancelled.")


def _notify(callback, value, message):
    if callback is not None:
        callback(float(value), str(message))


def _runtime_contract():
    return {
        "schema": 1, "mlx": MLX_VERSION,
        "publisher_sources": [list(source) for source in PUBLISHER_SOURCES],
        "base_python": os.path.realpath(sys.executable),
        "python_version": list(sys.version_info[:3]),
    }


def _require_platform():
    if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        raise RuntimeError("The selected packed Bonsai image runtime requires an Apple Silicon Mac.")


def _runtime_directory():
    from config.paths import get_portable_data_dir

    contract = json.dumps(_runtime_contract(), sort_keys=True).encode()
    identity = hashlib.sha256(contract).hexdigest()[:20]
    return Path(get_portable_data_dir()) / "runtimes" / "bonsai-image" / identity


def _child_env(*, offline=False):
    env = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE", "VIRTUAL_ENV",
                 "PIP_TARGET", "PIP_PREFIX", "PIP_ROOT", "PIP_USER"):
        env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["SMARTAI_BONSAI_IMAGE_OWNER_PID"] = str(os.getpid())
    # HF_HOME/HF_HUB_CACHE and the user's existing authentication are preserved.
    if offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


def owned_process_ids():
    """Expose only this adapter's live children to the app's memory monitor."""
    with _PROCESS_LOCK:
        return sorted(pid for pid, process in _PROCESSES.items() if process.poll() is None)


def _stop_process(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=3)


def _run_command(command, *, env, label, cancel_event=None, progress_callback=None, timeout=1800):
    """Drain logs while allowing cancellation of the owned process group."""
    _check_cancelled(cancel_event)
    process = subprocess.Popen(
        [str(value) for value in command], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        start_new_session=True,
    )
    with _PROCESS_LOCK:
        _PROCESSES[process.pid] = process
    messages = queue.Queue()
    tail = deque(maxlen=35)

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
                raise RuntimeError(f"{label} exceeded its time limit. Retry or cancel setup.")
            try:
                line = messages.get(timeout=0.1)
            except queue.Empty:
                continue
            if line.startswith(_EVENT_PREFIX):
                try:
                    event = json.loads(line[len(_EVENT_PREFIX):])
                    _notify(progress_callback, event["value"], event["message"])
                except (ValueError, KeyError, TypeError):
                    tail.append(line)
            else:
                tail.append(line)
        code = process.wait()
        _check_cancelled(cancel_event)
        if code:
            detail = "\n".join(tail).strip() or f"Process exited with status {code}."
            raise RuntimeError(f"{label} failed:\n{detail}")
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
    with path.open("a+") as lock:
        while True:
            _check_cancelled(cancel_event)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if cancel_event is not None:
                    cancel_event.wait(0.1)
                else:
                    time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def ensure_bonsai_image_runtime(*, cancel_event=None, progress_callback=None, force_probe=False):
    """Prepare code and wheels only; never download model weights here."""
    _require_platform()
    if sys.version_info < (3, 11):
        raise RuntimeError("The publisher's image runtime requires the app's Python 3.11 or newer.")
    from core.hf_transfer import hf_subprocess_command

    directory = _runtime_directory()
    contract = _runtime_contract()
    python = directory / "bin" / "python"
    owner = directory / "smartai-runtime-owner.json"
    receipt = directory / "smartai-runtime-ready.json"
    with _setup_lock(directory.with_suffix(".lock"), cancel_event):
        if directory.is_symlink():
            raise RuntimeError("The private Bonsai image runtime directory must not be a symlink.")
        if directory.exists() and any(directory.iterdir()) and _read_json(owner) != contract:
            raise RuntimeError(f"The reserved runtime folder contains unrecognized files: {directory}. Move it aside and retry.")
        directory.mkdir(parents=True, exist_ok=True)
        if not owner.exists():
            _write_json(owner, contract)
        ready = python.is_file() and _read_json(receipt) == contract
        if not ready:
            _notify(progress_callback, 0.03, "Preparing a private Bonsai image runtime (first use)")
            if not python.is_file():
                _run_command(
                    [sys.executable, "-I", "-m", "venv", str(directory)],
                    env=_child_env(), label="Private image Python setup", cancel_event=cancel_event,
                )
            _notify(progress_callback, 0.08, "Installing the publisher's pinned image runtime and its private MLX dependency")
            sources = [f"{name} @ {url}#sha256={digest}" for name, url, digest in PUBLISHER_SOURCES]
            _run_command(
                [str(python), "-I", "-m", "pip", "--isolated", "install", "--disable-pip-version-check",
                 "--no-input", "--force-reinstall", "--retries", "2", "--timeout", "45",
                 "--only-binary=:all:", "--no-binary=mflux,prism-image-studio",
                 f"mlx=={MLX_VERSION}", *sources],
                env=_child_env(), label="Publisher image runtime installation", cancel_event=cancel_event,
            )
        if not ready or force_probe:
            _notify(progress_callback, 0.12, "Checking the native image runtime and its 2-bit Metal operation")
            try:
                _run_command(
                    hf_subprocess_command([str(python), "-I", str(Path(__file__).resolve()), "--native-probe"]),
                    env=_child_env(offline=True), label="Native image runtime validation",
                    cancel_event=cancel_event, timeout=180,
                )
            except BaseException:
                receipt.unlink(missing_ok=True)
                raise
            _write_json(receipt, contract)
    return python


def _verify_native_dependencies():
    from importlib import metadata

    if metadata.version("mlx") != MLX_VERSION:
        raise RuntimeError("The private image runtime has an unexpected MLX version.")
    for name, url, digest in PUBLISHER_SOURCES:
        direct = json.loads(metadata.distribution(name).read_text("direct_url.json") or "{}")
        archive = direct.get("archive_info", {})
        actual = (archive.get("hashes") or {}).get("sha256") or str(archive.get("hash", "")).removeprefix("sha256=")
        if direct.get("url") != url or actual != digest:
            raise RuntimeError(f"The private {name} package does not match the verified publisher source.")


def _native_components():
    _verify_native_dependencies()
    # These two publisher modules install the slim packed-transformer, local
    # 4-bit text encoder, and local full-VAE component loaders.
    import backend.eviction  # noqa: F401
    import backend.text_encoder_4bit  # noqa: F401
    from mflux.models.common.vae.tiling_config import TilingConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    expected = {"model_path", "use_klein_fast_transformer", "klein_fast_precision", "vae_variant"}
    if not expected <= set(inspect.signature(Flux2Klein.__init__).parameters):
        raise RuntimeError("The publisher's packed-image constructor is incompatible with this adapter.")
    return Flux2Klein, TilingConfig


def _native_probe():
    _native_components()
    import mlx.core as mx
    import mlx.nn as nn

    layer = nn.Linear(128, 16, bias=False).to_quantized(group_size=128, bits=2)
    result = layer(mx.ones((1, 128)))
    mx.eval(result)
    if result.shape != (1, 16) or not bool(mx.all(mx.isfinite(result)).item()):
        raise RuntimeError("The native 2-bit image-runtime probe failed.")
    print("Bonsai image publisher imports and native 2-bit probe passed.", flush=True)


def _event(value, message):
    print(_EVENT_PREFIX + json.dumps({"value": value, "message": message}), flush=True)


def _native_worker(request):
    if not all(os.getenv(name) == "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise RuntimeError("Native image generation must use the already downloaded local snapshot.")
    owner = int(os.getenv("SMARTAI_BONSAI_IMAGE_OWNER_PID") or 0)
    if owner:
        def watch_owner():
            while os.getppid() == owner:
                time.sleep(0.5)
            os._exit(130)

        threading.Thread(target=watch_owner, daemon=True).start()
    Flux2Klein, TilingConfig = _native_components()
    root = Path(request["model_path"])
    for name in request["required_files"]:
        if not (root / name).is_file():
            raise RuntimeError(f"The selected Bonsai image component is missing: {name}")
    _event(0.2, "Loading the selected packed transformer and local 4-bit text encoder")
    model = Flux2Klein(
        model_path=str(root), use_klein_fast_transformer=True,
        klein_fast_precision="2bit", vae_variant="full",
        evict_text_encoder=True, lazy_components=False, bucketed_seq_len=False,
    )
    model._studio_te_4bit = True
    model.tiling_config = TilingConfig()

    class StepProgress:
        def call_in_loop(self, t, latents, config, **_kwargs):
            import mlx.core as mx

            mx.eval(latents)
            count = int(config.num_inference_steps)
            _event(0.35 + 0.55 * (int(t) + 1) / count, f"Denoising {int(t) + 1}/{count}")

        def call_after_loop(self, **_kwargs):
            _event(0.92, "Decoding the image with the selected local VAE")

    register = getattr(getattr(model, "callbacks", None), "register", None)
    if callable(register):
        register(StepProgress())
    _event(0.35, "Generating with the selected model and its local VAE")
    generated = model.generate_image(
        seed=request["seed"], prompt=request["prompt"],
        num_inference_steps=request["steps"], height=request["height"], width=request["width"],
        guidance=1.0, max_sequence_length=512,
    )
    generated.image.save(request["output_path"], format="PNG")
    _event(0.96, "Native image generation complete")


def generate_bonsai_image(model_info, prompt, output_path, *, steps=4, progress_callback=None, cancel_event=None):
    """Generate a real PNG for the caller to validate and publish atomically."""
    try:
        _check_cancelled(cancel_event)
        _require_platform()
        if model_info.get("repo_id") != MODEL_REPO or model_info.get("repo_revision") != MODEL_REVISION:
            raise RuntimeError("The packed Bonsai image adapter requires the exact selected publisher repository and revision.")
        width = int(model_info.get("width", 1024))
        height = int(model_info.get("height", 1024))
        if any(value < 256 or value > 2048 or value % 32 for value in (width, height)):
            raise ValueError("Bonsai image dimensions must be multiples of 32 between 256 and 2048.")
        if not 1 <= int(steps) <= 100:
            raise ValueError("Bonsai image generation requires between 1 and 100 steps.")
        if not str(prompt or "").strip():
            raise ValueError("Enter an image prompt before starting generation.")
        from config.media_models import media_download_metadata
        from core.hf_downloader import download_model_from_hf
        from core.hf_transfer import hf_subprocess_command

        metadata = media_download_metadata(MODEL_REPO)
        required = metadata.get("download_required_files")
        if not required:
            raise RuntimeError("The packed Bonsai image artifact manifest is unavailable.")
        root = Path(model_info.get("path") or model_info.get("model_path") or "")
        if not root.is_dir() or not all((root / name).is_file() for name in required):
            result = download_model_from_hf(
                MODEL_REPO, revision=MODEL_REVISION, required_files=required,
                allow_patterns=metadata["download_allow_patterns"], cancel_event=cancel_event,
                progress_callback=lambda message, percent: _notify(progress_callback, percent / 100 * 0.15, message),
            )
            if result.get("status") == "cancelled":
                raise BonsaiImageCancelled("Bonsai image download cancelled.")
            if result.get("status") != "success":
                raise RuntimeError(result.get("error") or "Bonsai image download failed.")
            root = Path(result["local_dir"])
        root = root.resolve()
        python = ensure_bonsai_image_runtime(cancel_event=cancel_event, progress_callback=progress_callback)
        request = {
            "model_path": str(root), "required_files": required,
            "output_path": os.path.abspath(output_path), "prompt": str(prompt),
            "seed": int(model_info.get("seed", 42)), "steps": int(steps),
            "width": width, "height": height,
        }
        jobs = python.parent.parent / "jobs"
        jobs.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="generation-", dir=jobs) as job:
            request_path = Path(job) / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            _run_command(
                hf_subprocess_command([str(python), "-I", str(Path(__file__).resolve()), "--worker", str(request_path)]),
                env=_child_env(offline=True), label="Bonsai image generation",
                cancel_event=cancel_event, progress_callback=progress_callback, timeout=6 * 60 * 60,
            )
        return {"status": "success", "path": str(output_path), "backend": "bonsai_mlx"}
    except BonsaiImageCancelled as error:
        return {"status": "cancelled", "error": str(error)}
    except Exception as error:
        return {"status": "error", "error": str(error), "backend": "bonsai_mlx"}


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Install the isolated publisher runtime without model weights")
    parser.add_argument("--probe", action="store_true", help="Repeat native validation even when the runtime is already ready")
    parser.add_argument("--native-probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.native_probe:
        _native_probe()
    elif args.worker:
        _native_worker(json.loads(Path(args.worker).read_text(encoding="utf-8")))
    elif args.prepare:
        print(ensure_bonsai_image_runtime(force_probe=args.probe), flush=True)
    else:
        parser.error("Use --prepare --probe to validate the native runtime without downloading model weights.")


if __name__ == "__main__":
    main()
