"""
Cross-Platform Hardware & Inference Backend Router.
Provides seamless detection and routing between:
1. Native MLX (Apple Silicon, Linux CPU/CUDA, Windows CPU/CUDA)
2. PyTorch CUDA / ROCm / Metal MPS / CPU (Windows & Linux)
3. GGUF / llama.cpp & CPU fallbacks
Guarantees zero unhandled import errors across platforms.
"""

import os
import platform
import sys
from typing import Any, Dict, List, Optional, Tuple
from core.hf_transfer import run_hf_operation


class PlatformRouter:
    """Intelligent platform and compute backend router."""

    def __init__(self, override_backend: Optional[str] = None):
        self.override_backend = override_backend
        self.last_error = ""
        self._cached_platform_info: Optional[Dict[str, Any]] = None

    def get_platform_info(self) -> Dict[str, Any]:
        """Returns comprehensive host hardware and software telemetry."""
        if self._cached_platform_info is not None:
            return self._cached_platform_info

        os_name = platform.system().lower()
        machine = platform.machine().lower()

        # OS identification
        if "darwin" in os_name:
            os_family = "macos"
        elif "windows" in os_name:
            os_family = "windows"
        else:
            os_family = "linux"

        # Probe the native binary backend as well as the Python frontend.
        from core.cross_platform_mlx import native_mlx_installed
        mlx_available = native_mlx_installed()
        mlx_device = "cpu"
        if mlx_available:
            import mlx.core as mx
            if mx.default_device() == mx.gpu:
                mlx_device = "mps" if os_family == "macos" else "cuda"

        # Check PyTorch & Accelerators
        torch_available = False
        cuda_available = False
        mps_available = False
        gpu_count = 0
        device_name = "CPU"

        try:
            import torch
            torch_available = True
            if torch.cuda.is_available():
                cuda_available = True
                gpu_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                mps_available = True
                device_name = "Apple Silicon Metal (MPS)"
        except Exception:
            torch_available = False

        # Determine optimal backend
        if self.override_backend:
            backend = self.override_backend.lower()
        elif mlx_available:
            backend = "mlx"
        elif torch_available:
            backend = "torch"
        else:
            backend = "torch"

        # Determine primary compute device
        if backend == "mlx":
            device = mlx_device
            if device == "cuda":
                device_name = "NVIDIA CUDA (MLX)"
            elif device == "mps":
                device_name = "Apple Silicon Metal (MLX)"
            else:
                device_name = "CPU (MLX)"
        elif cuda_available:
            device = "cuda"
        elif mps_available:
            device = "mps"
        else:
            device = "cpu"

        self._cached_platform_info = {
            "os_family": os_family,
            "architecture": machine,
            "backend": backend,
            "device": device,
            "device_name": device_name,
            "mlx_available": mlx_available,
            "mlx_device": mlx_device if mlx_available else None,
            "torch_available": torch_available,
            "cuda_available": cuda_available,
            "mps_available": mps_available,
            "gpu_count": gpu_count,
            "python_version": sys.version.split()[0],
            "executable": sys.executable
        }
        return self._cached_platform_info

    def route_model_loader(
        self,
        model_path: str,
        kv_bits: Optional[int] = None,
    ) -> Tuple[str, Any, Any]:
        """
        Loads the reasoning model and tokenizer using the optimal engine for this OS:
        - Native MLX when installed, on its current CPU/CUDA/Metal device
        - PyTorch / Transformers when that backend is selected
        ``kv_bits`` is retained only as a backwards-compatible argument and ignored.
        Returns (backend_name, model_instance, tokenizer_instance)
        """
        info = self.get_platform_info()
        self.last_error = ""
        backend = info["backend"]
        device = info["device"]
        source_options = {}

        if backend == "mlx":
            try:
                from mlx_import_compat import load_mlx_lm
                load = load_mlx_lm().load
                print(f"[*] Loading native MLX model from {model_path} (full-precision KV)...")
                model, tokenizer = run_hf_operation(lambda: load(model_path))
                return "mlx", model, tokenizer
            except Exception as e:
                self.last_error = str(e)
                print(f"[!] Native MLX load failed: {e}.")
                if not info["torch_available"]:
                    return "none", None, None
                # Preserve the original Transformers fallback only for its own checkpoints.
                from core.controller_runtime import resolve_local_snapshot
                from core.model_policy import admit_hf_model, admit_local_model
                source = resolve_local_snapshot(model_path)
                if os.path.exists(source):
                    allowed, _, metadata = admit_local_model(source)
                    model_path = source
                else:
                    allowed, _, metadata = admit_hf_model(model_path)
                    if allowed:
                        source_options["revision"] = metadata["repo_revision"]
                if not allowed or metadata.get("backend_family") != "transformers":
                    return "none", None, None
                backend = "torch"
                device = "cuda" if info["cuda_available"] else "mps" if info["mps_available"] else "cpu"

        if backend == "torch":
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                dtype = torch.bfloat16 if (device == "cuda" or (device == "mps" and info["os_family"] == "macos")) else torch.float32

                print(f"[*] Loading PyTorch Model from {model_path} on {device} ({dtype})...")
                tokenizer = run_hf_operation(lambda: AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, **source_options))
                model = run_hf_operation(lambda: AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True, **source_options
                ))
                if device == "mps":
                    model = model.to("mps")
                elif device == "cpu":
                    model = model.to("cpu")
                self.last_error = ""
                return "torch", model, tokenizer
            except Exception as e:
                self.last_error = str(e)
                print(f"[!] PyTorch model load failed: {e}.")

        return "none", None, None


_global_router: Optional[PlatformRouter] = None


def get_platform_router() -> PlatformRouter:
    """Returns singleton instance of PlatformRouter."""
    global _global_router
    if _global_router is None:
        _global_router = PlatformRouter()
    return _global_router


def detect_hardware() -> Dict[str, Any]:
    """Convenience function returning host hardware profile."""
    return get_platform_router().get_platform_info()


def get_auto_context_window_size(total_ram_gb: Optional[float] = None) -> int:
    """
    Dynamically autosets the optimal context window token size based on host physical RAM:
    - >= 64 GB RAM: 262,144 tokens (256K)
    - >= 32 GB RAM: 131,072 tokens (128K)
    - >= 15 GB RAM: 65,536 tokens (64K)
    - < 15 GB RAM:  32,768 tokens (32K)
    """
    if total_ram_gb is None:
        try:
            import psutil
            total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            try:
                import subprocess
                out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()
                total_ram_gb = int(out) / (1024 ** 3)
            except Exception:
                total_ram_gb = 16.0

    if total_ram_gb >= 64.0:
        return 262144
    elif total_ram_gb >= 32.0:
        return 131072
    elif total_ram_gb >= 15.0:
        return 65536
    else:
        return 32768
