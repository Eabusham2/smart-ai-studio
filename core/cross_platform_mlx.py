"""Route the selected MLX artifact without converting or replacing its weights."""
from __future__ import annotations
import importlib.util
import os
import platform
import shutil


def native_mlx_platform():
    return platform.system() == "Linux" or (
        platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}
    )


def native_mlx_installed():
    if not native_mlx_platform():
        return False
    try:
        return all(importlib.util.find_spec(name) is not None for name in ("mlx", "mlx_lm"))
    except (ImportError, ValueError):
        return False


def tensorfold_executable():
    explicit = os.environ.get("SMARTAI_TENSORFOLD_EXECUTABLE", "").strip()
    if explicit:
        path = os.path.abspath(os.path.expanduser(explicit))
        return path if os.path.isfile(path) else None
    return shutil.which("tensorfold-native") or shutil.which("tensorfold-native.exe")


def select_mlx_runtime(controller_runtime, info):
    system = platform.system()
    choice = str(info.get("mlx_runtime") or os.getenv("SMARTAI_MLX_RUNTIME", "auto")).strip().lower()
    if choice not in {"auto", "native", "tensorfold"}:
        raise ValueError("MLX runtime must be auto, native or tensorfold")
    if choice == "auto":
        choice = "tensorfold" if system == "Windows" or (
            system == "Linux" and not native_mlx_installed() and tensorfold_executable()
        ) else "native"
    if choice == "native":
        if not native_mlx_platform():
            raise RuntimeError("Native MLX needs Apple Silicon or Linux. Windows uses TensorFold native.")
        if system == "Linux":
            if controller_runtime in {"mlx_repo_vlm", "jang_vlm"} or info.get("bonsai25_mtp"):
                raise RuntimeError("This publisher-specific MLX pack needs unported Metal/Hadamard kernels. "
                                   "Its original weights have not been replaced.")
            if not native_mlx_installed():
                raise RuntimeError("Install MLX in Studio's Python environment: "
                                   "mlx[cuda12]==0.32.0 or mlx[cuda13]==0.32.0 or mlx[cpu]==0.32.0, "
                                   "plus mlx-lm==0.31.3.")
        return choice
    if system not in {"Windows", "Linux"}:
        raise RuntimeError("The TensorFold adapter is currently enabled on Windows and Linux only.")
    if (controller_runtime not in {"mlx_lm", "auto", ""}
            or set(info.get("input_modalities") or ["text"]) != {"text"} or info.get("bonsai25_mtp")):
        raise RuntimeError("TensorFold adapter currently admits standard text MLX checkpoints only; "
                           "publisher-specific ternary and multimodal kernels require qualification.")
    if not tensorfold_executable():
        raise RuntimeError("Install a compatible native TensorFold executable on PATH, "
                           "or set SMARTAI_TENSORFOLD_EXECUTABLE to its full path.")
    return choice
