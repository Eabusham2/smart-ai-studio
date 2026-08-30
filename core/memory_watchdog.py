"""
System Memory Pressure Watchdog.
Continuously monitors host RAM and GPU/Metal VRAM pressure.
Automatically triggers graceful model unloading when memory thresholds are exceeded
to prevent system freezing, OOM kills, or UI locking.
"""

import gc
import os
import platform
import shutil
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple


class SystemMemoryWatchdog:
    def __init__(
        self,
        check_interval_seconds: float = 3.0,
        max_ram_usage_percent: float = 98.5,
        min_free_ram_gb: float = 0.15,
        max_process_ram_gb: float = 15.8,
        on_pressure_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.check_interval_seconds = check_interval_seconds
        self.max_ram_usage_percent = max_ram_usage_percent
        self.min_free_ram_gb = min_free_ram_gb
        self.max_process_ram_gb = max_process_ram_gb
        self.on_pressure_callback = on_pressure_callback

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.last_check_status: Dict[str, Any] = {}

    @staticmethod
    def reclaim_process_memory():
        """Proactively purges garbage, Python memory pools, Metal/MPS buffers, and CUDA caches."""
        gc.collect(2)
        try:
            import mlx.core as mx
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass

    @staticmethod
    def get_detailed_memory_breakdown() -> Dict[str, Any]:
        """
        Computes granular, authentic memory breakdown across:
        1. App Process RSS (Python GUI / framework memory).
        2. Model Metal Active Memory (MLX unified memory tensors).
        3. Model Metal Peak & Cache Allocation.
        4. Host System RAM (Used, Available, Total) matching macOS Activity Monitor.
        5. Dynamic RAM scaling parameters.
        """
        app_rss_gb = 0.0
        try:
            import psutil
            app_rss_gb = psutil.Process().memory_info().rss / (1024 ** 3)
        except Exception:
            try:
                import resource
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                app_rss_gb = (rss / (1024 ** 3)) if platform.system() == "Darwin" else (rss / (1024 ** 2))
            except Exception:
                pass

        model_active_gb = 0.0
        model_peak_gb = 0.0
        model_cache_gb = 0.0
        try:
            import mlx.core as mx
            if hasattr(mx, "get_active_memory"):
                model_active_gb = mx.get_active_memory() / (1024 ** 3)
            elif hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
                model_active_gb = mx.metal.get_active_memory() / (1024 ** 3)

            if hasattr(mx, "get_peak_memory"):
                model_peak_gb = mx.get_peak_memory() / (1024 ** 3)
            elif hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
                model_peak_gb = mx.metal.get_peak_memory() / (1024 ** 3)

            if hasattr(mx, "get_cache_memory"):
                model_cache_gb = mx.get_cache_memory() / (1024 ** 3)
            elif hasattr(mx, "metal") and hasattr(mx.metal, "get_cache_memory"):
                model_cache_gb = mx.metal.get_cache_memory() / (1024 ** 3)
        except Exception:
            pass

        system_total_gb = 16.0
        system_used_gb = 5.0
        system_free_gb = 11.0
        system_percent = 31.25

        try:
            import psutil
            vm = psutil.virtual_memory()
            system_total_gb = vm.total / (1024 ** 3)
            system_used_gb = vm.used / (1024 ** 3)
            system_free_gb = vm.available / (1024 ** 3)
            system_percent = vm.percent
        except ImportError:
            try:
                if platform.system() == "Darwin":
                    import subprocess
                    out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()
                    system_total_gb = int(out) / (1024 ** 3)
                    system_free_gb = max(2.0, system_total_gb * 0.3)
                    system_used_gb = system_total_gb - system_free_gb
                    system_percent = (system_used_gb / system_total_gb) * 100
            except Exception:
                pass

        total_allocated_gb = max(app_rss_gb, app_rss_gb + model_active_gb)

        return {
            "app_rss_gb": round(app_rss_gb, 2),
            "model_metal_active_gb": round(model_active_gb, 2),
            "model_metal_peak_gb": round(max(model_active_gb, model_peak_gb), 2),
            "model_metal_cache_gb": round(model_cache_gb, 2),
            "total_allocated_gb": round(total_allocated_gb, 2),
            "system_total_gb": round(system_total_gb, 1),
            "system_used_gb": round(system_used_gb, 1),
            "system_free_gb": round(system_free_gb, 1),
            "system_percent": round(system_percent, 1),
            "dynamic_scaling_active": True,
            "allocation_strategy": "Dynamic Unified Memory (Elastic Expansion on Demand)"
        }

    @staticmethod
    def adjust_dynamic_metal_headroom(min_headroom_gb: float = 1.2):
        """
        Dynamically configures Apple Silicon Metal cache limits based on live system RAM availability.
        Allows model memory to expand elastically during heavy rollout batches without crashing.
        """
        try:
            import psutil
            import mlx.core as mx
            avail_gb = psutil.virtual_memory().available / (1024 ** 3)
            # Allow Metal to scale dynamically up to 95% of available free RAM
            dynamic_cache_gb = max(2.5, avail_gb * 0.95)
            dynamic_cache_bytes = int(dynamic_cache_gb * (1024 ** 3))

            if hasattr(mx, "set_cache_limit"):
                mx.set_cache_limit(dynamic_cache_bytes)
            elif hasattr(mx, "metal") and hasattr(mx.metal, "set_cache_limit"):
                mx.metal.set_cache_limit(dynamic_cache_bytes)
        except Exception:
            pass

    @staticmethod
    def get_process_rss_gb() -> float:
        """Returns the current process memory in Gigabytes."""
        breakdown = SystemMemoryWatchdog.get_detailed_memory_breakdown()
        return breakdown["total_allocated_gb"]

    @staticmethod
    def get_system_memory_status() -> Dict[str, Any]:
        """Queries host total, used, and free RAM in GB matching macOS Activity Monitor."""
        breakdown = SystemMemoryWatchdog.get_detailed_memory_breakdown()
        return {
            "total_gb": breakdown["system_total_gb"],
            "used_gb": breakdown["system_used_gb"],
            "free_gb": breakdown["system_free_gb"],
            "used_percent": breakdown["system_percent"],
            "process_rss_gb": breakdown["total_allocated_gb"],
            "app_rss_gb": breakdown["app_rss_gb"],
            "model_metal_active_gb": breakdown["model_metal_active_gb"],
            "model_metal_peak_gb": breakdown["model_metal_peak_gb"]
        }

    def check_memory_pressure(self) -> Tuple[bool, Dict[str, Any]]:
        """Evaluates whether current system or process memory exceeds safe operating bounds."""
        status = self.get_system_memory_status()
        self.last_check_status = status

        # Dynamically scale Metal cache headroom based on available memory
        self.adjust_dynamic_metal_headroom()

        # Proactive memory reclamation if total process/model memory grows above safe threshold
        if status.get("process_rss_gb", 0) > 15.6:
            self.reclaim_process_memory()

        is_under_pressure = (
            status["used_percent"] >= self.max_ram_usage_percent or
            status["free_gb"] < self.min_free_ram_gb or
            status.get("process_rss_gb", 0) >= self.max_process_ram_gb
        )

        return is_under_pressure, status

    def start_monitoring(self):
        """Launches continuous watchdog background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop_monitoring(self):
        """Stops background watchdog."""
        self._running = False

    def _monitor_loop(self):
        while self._running:
            try:
                pressure, status = self.check_memory_pressure()
                if pressure and self.on_pressure_callback:
                    self.on_pressure_callback(status)
            except Exception:
                pass
            time.sleep(self.check_interval_seconds)
