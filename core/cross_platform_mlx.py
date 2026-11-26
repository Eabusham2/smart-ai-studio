"""Route the selected MLX artifact without converting or replacing its weights."""
from __future__ import annotations

import os
import platform
import shutil


def native_mlx_platform():
    system, machine = platform.system(), platform.machine().lower()
    if system == "Darwin":
        return machine in {"arm64", "aarch64"}
    return system in {"Linux", "Windows"} and machine in {"x86_64", "amd64", "arm64", "aarch64"}


def native_mlx_error():
    """Probe the active device and MLX-LM, including missing shared libraries."""
    if not native_mlx_platform():
        return "Native MLX needs Apple Silicon or a supported Windows/Linux processor."
    try:
        import mlx.core as mx
        from mlx_import_compat import load_mlx_lm
        device = mx.default_device()
        if not mx.is_available(device):
            return f"MLX reports an unavailable default device: {device}"
        # Import success alone does not prove a CUDA driver or CPU runtime can execute work.
        if (mx.array(1) + 1).item() != 2:
            return f"MLX failed its arithmetic check on {device}"
        load_mlx_lm()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def native_mlx_installed():
    return native_mlx_error() is None


def tensorfold_executable():
    explicit = os.environ.get("SMARTAI_TENSORFOLD_EXECUTABLE", "").strip()
    if explicit:
        path = os.path.abspath(os.path.expanduser(explicit))
        return path if os.path.isfile(path) and os.access(path, os.X_OK) else None
    return shutil.which("tensorfold-native") or shutil.which("tensorfold-native.exe")


def select_mlx_runtime(controller_runtime, info):
    system = platform.system()
    choice = str(info.get("mlx_runtime") or os.getenv("SMARTAI_MLX_RUNTIME", "auto")).strip().lower()
    if choice not in {"auto", "native", "tensorfold"}:
        raise ValueError("MLX runtime must be auto, native or tensorfold")

    text_pack = (controller_runtime in {"mlx_lm", "auto", ""}
                 and set(info.get("input_modalities") or ["text"]) == {"text"}
                 and not info.get("bonsai25_mtp"))
    if choice in {"auto", "native"}:
        if not native_mlx_platform():
            raise RuntimeError("Native MLX needs Apple Silicon or a supported Windows/Linux processor.")
        if system == "Darwin":
            return "native"
        error = native_mlx_error()
        if error is None:
            if controller_runtime in {"jang_vlm", "jang-vlm"}:
                raise RuntimeError("This older JANG archive requires the separate Apple Silicon vMLX runtime. "
                                   "The Bonsai 2.5 and CRACK 2.5 MLX presets use the publisher loader.")
            return "native"
        raise RuntimeError("Install the native MLX runtime using tools/setup_mlx.py in Studio's Python "
                           f"environment. MLX probe failed: {error}")
    if system not in {"Windows", "Linux"}:
        raise RuntimeError("The TensorFold adapter is currently enabled on Windows and Linux only.")
    if not text_pack:
        raise RuntimeError("TensorFold admits standard text MLX checkpoints only. Select the native MLX "
                           "runtime for the Bonsai 2.5 publisher loader and multimodal checkpoints.")
    if not tensorfold_executable():
        raise RuntimeError("Install a compatible native TensorFold executable on PATH, "
                           "or set SMARTAI_TENSORFOLD_EXECUTABLE to its full path.")
    return "tensorfold"
