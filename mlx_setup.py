"""Install and diagnose the native MLX stack without downloading model weights.

The frontend extras currently select backend wheels only on Linux. Select the
official backend distribution explicitly so Windows installs are complete too.
This module uses only the standard library and is safe to import during Setup.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

MLX_VERSION = "0.32.5"
MLX_LM_VERSION = "0.31.3"
MLX_VLM_VERSION = "0.6.3"
MLX_AUDIO_VERSION = "0.4.6"
TRANSFORMERS_VERSION = "5.5.0"
BACKENDS = {"cpu": "mlx-cpu", "cuda12": "mlx-cuda-12", "cuda13": "mlx-cuda-13"}


def _version_tuple(value):
    parts = str(value).strip().split(".")
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError("Invalid NVIDIA driver version")
    return tuple(int(part) for part in (parts + ["0", "0"])[:3])


def _nvidia_devices():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode:
            return []
        rows = []
        for line in result.stdout.splitlines():
            capability, driver = (part.strip() for part in line.split(","))
            rows.append((float(capability), _version_tuple(driver)))
        return rows
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return []


def select_backend(requested="auto", *, system=None, machine=None, devices=None):
    system = platform.system() if system is None else system
    machine = platform.machine().lower() if machine is None else machine.lower()
    if system not in {"Linux", "Windows"}:
        raise RuntimeError("This setup command targets Windows and Linux; macOS uses Studio's existing MLX installation.")
    if machine not in {"x86_64", "amd64", "aarch64", "arm64"}:
        raise RuntimeError(f"No official native MLX wheel is configured for {system} {machine}.")
    requested = str(requested).strip().lower()
    if requested not in {"auto", *BACKENDS}:
        raise ValueError("MLX backend must be auto, cpu, cuda12, or cuda13.")
    if system == "Windows" and machine in {"arm64", "aarch64"}:
        if requested not in {"auto", "cpu"}:
            raise RuntimeError("Windows ARM64 has official CPU wheels; native CUDA wheels require Windows x64.")
        return "cpu"
    if requested != "auto":
        return requested
    devices = _nvidia_devices() if devices is None else devices
    # MLX uses device zero unless visibility is configured before launch.
    # Do not choose CUDA because a later card is capable while device zero is not.
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = visibility.split(",")[0].strip() if visibility is not None else "0"
    # An empty value hides every GPU. UUID/MIG visibility cannot be matched to
    # nvidia-smi row indices, so leave those configurations to an explicit choice.
    index = int(visible) if visible.isdigit() else None
    selected = devices[index:index + 1] if index is not None else []
    usable = [(cap, driver) for cap, driver in selected if cap >= 7.5]
    if any(driver >= (580, 0, 0) for _, driver in usable):
        return "cuda13"
    if any(driver >= (550, 54, 14) for _, driver in usable):
        return "cuda12"
    return "cpu"


def _pinned_versions(backend):
    if backend not in BACKENDS:
        raise ValueError("Unknown MLX backend")
    return {
        "mlx": MLX_VERSION,
        BACKENDS[backend]: MLX_VERSION,
        "mlx-lm": MLX_LM_VERSION,
        "mlx-vlm": MLX_VLM_VERSION,
        # Newer MLX Audio releases pull a Transformers version outside Studio's pin.
        "mlx-audio": MLX_AUDIO_VERSION,
        "transformers": TRANSFORMERS_VERSION,
    }


def packages(backend):
    return [f"{name}=={version}" for name, version in _pinned_versions(backend).items()]


def installed_backends(python):
    script = (
        "import importlib.metadata as m,json\n"
        "r={}\n"
        "for name in ('mlx-cpu','mlx-cuda-12','mlx-cuda-13'):\n"
        " try: r[name]=m.version(name)\n"
        " except m.PackageNotFoundError: pass\n"
        "print(json.dumps(r))\n"
    )
    result = subprocess.run([str(python), "-c", script], check=True,
                            capture_output=True, text=True, timeout=30)
    return json.loads(result.stdout.strip())


def probe(python, backend):
    """Execute packed arithmetic and autodiff in a fresh process, without a model."""
    expected = _pinned_versions(backend)
    script = """
import importlib.metadata as metadata, json, sys
backend, expected_json, root = sys.argv[1:]
versions = {}
for name, wanted in json.loads(expected_json).items():
    try:
        actual = metadata.version(name)
    except metadata.PackageNotFoundError:
        actual = None
    if actual != wanted:
        raise RuntimeError(f"Studio requires {name}=={wanted} (found {actual or 'no installed version'}).")
    versions[name] = actual
import mlx.core as mx
import mlx.nn
import mlx.optimizers
sys.path.insert(0, root)
from mlx_import_compat import load_mlx_lm
load_mlx_lm()
import mlx_vlm
if backend.startswith("cuda"):
    if not mx.cuda.is_available():
        raise RuntimeError("The selected CUDA backend cannot access a supported NVIDIA GPU.")
    mx.set_default_device(mx.gpu)
else:
    mx.set_default_device(mx.cpu)
x = mx.ones((1, 128), dtype=mx.float32)
w, scales, biases = mx.quantize(mx.ones((4, 128), dtype=mx.float32), group_size=128, bits=2)
y = mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=128, bits=2)
h = mx.hadamard_transform(x)
g = mx.grad(lambda value: mx.sum(value * value))(x)
mx.eval(y, h, g)
if not bool(mx.all(mx.isfinite(y)).item()) or not bool(mx.all(g == 2).item()):
    raise RuntimeError("MLX arithmetic/gradient check failed.")
print(json.dumps({"status": "ready", "backend": backend, "device": str(mx.default_device()),
                  "versions": versions,
                  "packed_matmul": True, "hadamard": True, "autodiff": True,
                  "model_loaded": False}))
"""
    root = str(Path(__file__).resolve().parent)
    result = subprocess.run([str(python), "-c", script, backend, json.dumps(expected), root],
                            capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(f"Native MLX {backend} probe failed: {result.stderr.strip() or result.stdout.strip()}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def install(python=sys.executable, requested="auto", *, check_only=False):
    # A venv's interpreter can be a symlink to the base Python. Keep the
    # venv path so that dependency checks and pip use its site-packages.
    python = str(Path(python).expanduser().absolute())
    installed = installed_backends(python)
    if requested == "auto" and len(installed) == 1:
        # A portable release's lock already chose a backend. Reuse that backend;
        # --backend cuda12/cuda13 is an explicit, repeatable switch.
        selected = next(key for key, value in BACKENDS.items() if value in installed)
        select_backend(selected)
    else:
        selected = select_backend(requested)
    if check_only:
        return probe(python, selected)
    wanted = packages(selected)
    conflicts = sorted(set(installed) - {BACKENDS[selected]})
    pip = [python, "-m", "pip"]
    if conflicts:
        # Download the replacement before removing a working backend. All MLX
        # backends install the same native-library paths, so they cannot coexist.
        with tempfile.TemporaryDirectory(prefix="smartai-mlx-") as temporary:
            subprocess.run([*pip, "download", "--only-binary=:all:", "--no-deps",
                            "--dest", temporary, f"{BACKENDS[selected]}=={MLX_VERSION}"], check=True)
            wheels = list(Path(temporary).glob("*.whl"))
            if len(wheels) != 1:
                raise RuntimeError("Expected exactly one official MLX backend wheel.")
            subprocess.run([*pip, "uninstall", "-y", *conflicts], check=True)
            subprocess.run([*pip, "install", "--force-reinstall", "--no-deps", str(wheels[0])], check=True)
    subprocess.run([*pip, "install", "--prefer-binary",
                    "--only-binary=mlx,mlx-cpu,mlx-cuda-12,mlx-cuda-13",
                    *wanted], check=True)
    return probe(python, selected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", *BACKENDS),
                        default=os.environ.get("SMARTAI_MLX_BACKEND", "auto"))
    parser.add_argument("--python", default=sys.executable, help="Studio's Python interpreter")
    parser.add_argument("--check", action="store_true", help="Verify the installed runtime without installing")
    parser.add_argument("--dry-run", action="store_true", help="Print the package plan without installing")
    args = parser.parse_args(argv)
    if args.dry_run:
        selected = select_backend(args.backend)
        result = {"backend": selected, "packages": packages(selected), "model_loaded": False}
    else:
        result = install(args.python, args.backend, check_only=args.check)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print("Native MLX setup failed: " + str(exc), file=sys.stderr)
        raise SystemExit(1)

