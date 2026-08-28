"""
System Hardware Profiler & Intelligent Engine Backend Resolver.
Detects Apple Silicon Metal, NVIDIA CUDA VRAM, AMD ROCm, SIMD CPU extensions (NEON/AVX2/AVX512),
and dynamically maps AI models to the highest-performance native execution backend (MLX, GGUF, BitNet, PyTorch).
"""

from dataclasses import dataclass, field
import os
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple

def _get_os_name() -> str:
    if sys.platform == "darwin":
        return "Darwin"
    elif sys.platform == "win32":
        return "Windows"
    return "Linux"


def _get_arch_name() -> str:
    try:
        return os.uname().machine
    except Exception:
        import struct
        return "x86_64" if struct.calcsize("P") == 8 else "x86"


@dataclass
class SystemHardwareProfile:
    os_name: str = field(default_factory=_get_os_name)
    arch: str = field(default_factory=_get_arch_name)
    cpu_count: int = field(default_factory=lambda: os.cpu_count() or 4)
    total_ram_gb: float = 16.0
    available_ram_gb: float = 8.0
    has_apple_silicon: bool = False
    has_cuda: bool = False
    cuda_vram_gb: float = 0.0
    cuda_device_name: str = ""
    has_rocm: bool = False
    cpu_features: Dict[str, bool] = field(default_factory=dict)
    recommended_backend: str = "torch"
    recommended_device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "os_name": self.os_name,
            "arch": self.arch,
            "cpu_count": self.cpu_count,
            "total_ram_gb": round(self.total_ram_gb, 2),
            "available_ram_gb": round(self.available_ram_gb, 2),
            "has_apple_silicon": self.has_apple_silicon,
            "has_cuda": self.has_cuda,
            "cuda_vram_gb": round(self.cuda_vram_gb, 2),
            "cuda_device_name": self.cuda_device_name,
            "has_rocm": self.has_rocm,
            "cpu_features": self.cpu_features,
            "recommended_backend": self.recommended_backend,
            "recommended_device": self.recommended_device
        }


def _detect_cpu_features() -> Dict[str, bool]:
    """Detects SIMD vector instruction sets for hardware acceleration."""
    features = {
        "avx2": False,
        "avx512": False,
        "neon": False,
        "fma": False
    }

    system = _get_os_name()
    machine = _get_arch_name().lower()

    if "arm" in machine or "aarch64" in machine:
        features["neon"] = True

    try:
        if system == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.features"]).decode().lower()
            features["avx2"] = "avx2" in out
            features["avx512"] = "avx512" in out
            features["fma"] = "fma" in out
            if "arm64" in machine:
                features["neon"] = True
        elif system == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                flags = f.read().lower()
            features["avx2"] = "avx2" in flags
            features["avx512"] = "avx512" in flags
            features["neon"] = "neon" in flags or "asimd" in flags
            features["fma"] = "fma" in flags
        elif system == "Windows":
            # Safe assumptions on modern 64-bit Windows
            features["avx2"] = True
    except Exception:
        pass

    return features


def detect_system_hardware() -> SystemHardwareProfile:
    """Profiles host hardware accelerators and memory topology."""
    os_name = _get_os_name()
    arch = _get_arch_name().lower()
    cpu_count = os.cpu_count() or 4

    total_ram_gb = 16.0
    avail_ram_gb = 8.0

    # 1. RAM Detection (psutil with native fallbacks)
    try:
        import psutil
        vm = psutil.virtual_memory()
        total_ram_gb = vm.total / (1024 ** 3)
        avail_ram_gb = vm.available / (1024 ** 3)
    except Exception:
        try:
            if os_name == "Darwin":
                out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()
                total_ram_gb = int(out) / (1024 ** 3)
                avail_ram_gb = max(2.0, total_ram_gb * 0.5)
            elif os_name == "Linux":
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            total_ram_gb = int(line.split()[1]) / (1024 ** 2)
                        elif line.startswith("MemAvailable:"):
                            avail_ram_gb = int(line.split()[1]) / (1024 ** 2)
        except Exception:
            pass

    # 2. Apple Silicon Metal Detection
    has_apple_silicon = (os_name == "Darwin" and ("arm64" in arch or "aarch64" in arch))

    # 3. NVIDIA CUDA Detection
    has_cuda = False
    cuda_vram_gb = 0.0
    cuda_device_name = ""

    try:
        import torch
        if torch.cuda.is_available():
            has_cuda = True
            cuda_device_name = torch.cuda.get_device_name(0)
            cuda_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass

    # 4. AMD ROCm Detection
    has_rocm = False
    try:
        import torch
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            has_rocm = True
    except Exception:
        pass

    cpu_features = _detect_cpu_features()

    # 5. Determine Recommended Backend & Device
    if has_apple_silicon:
        recommended_backend = "mlx"
        recommended_device = "mps"
    elif has_cuda and cuda_vram_gb >= 4.0:
        recommended_backend = "gguf"
        recommended_device = "cuda"
    elif cpu_features.get("neon") or cpu_features.get("avx2"):
        recommended_backend = "bitnet"
        recommended_device = "cpu"
    else:
        recommended_backend = "gguf"
        recommended_device = "cpu"

    return SystemHardwareProfile(
        os_name=os_name,
        arch=arch,
        cpu_count=cpu_count,
        total_ram_gb=total_ram_gb,
        available_ram_gb=avail_ram_gb,
        has_apple_silicon=has_apple_silicon,
        has_cuda=has_cuda,
        cuda_vram_gb=cuda_vram_gb,
        cuda_device_name=cuda_device_name,
        has_rocm=has_rocm,
        cpu_features=cpu_features,
        recommended_backend=recommended_backend,
        recommended_device=recommended_device
    )


def resolve_optimal_backend(model_type: str = "ternary") -> Tuple[str, str]:
    """
    Dynamically maps a model type to the optimal engine backend and compute device.
    Returns: (backend_name, device_name)
    Examples: ('mlx', 'mps'), ('gguf', 'cuda'), ('bitnet', 'cpu'), ('gguf', 'cpu')
    """
    hw = detect_system_hardware()

    # 1. Apple Silicon always utilizes Metal-native MLX
    if hw.has_apple_silicon:
        return "mlx", "mps"

    # 2. Discrete NVIDIA CUDA GPU
    if hw.has_cuda and hw.cuda_vram_gb >= 3.5:
        return "gguf", "cuda"

    # 3. CPU execution: BitNet 1.58-bit kernel for ternary models, GGUF for general models
    if "ternary" in model_type.lower() or "bitnet" in model_type.lower() or "1.58" in model_type.lower():
        return "bitnet", "cpu"

    return "gguf", "cpu"


if __name__ == "__main__":
    profile = detect_system_hardware()
    print("=== System Hardware Profile ===")
    for k, v in profile.to_dict().items():
        print(f"  {k}: {v}")
    print(f"\nResolved Backend for Ternary Model: {resolve_optimal_backend('ternary')}")
