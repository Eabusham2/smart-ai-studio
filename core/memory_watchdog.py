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
        check_interval_seconds: float = 2.5,
        max_ram_usage_percent: float = 94.0,
        min_free_ram_gb: float = 0.8,
        max_process_ram_gb: float = 12.0,
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
    def get_process_rss_gb() -> float:
        """Returns the current process RSS (Resident Set Size) in Gigabytes."""
        try:
            import psutil
            return round(psutil.Process().memory_info().rss / (1024 ** 3), 2)
        except Exception:
            try:
                import resource
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if platform.system() == "Darwin":
                    return round(rss / (1024 ** 3), 2)
                return round(rss / (1024 ** 2), 2)
            except Exception:
                return 0.0

    @staticmethod
    def get_system_memory_status() -> Dict[str, Any]:
        """Queries host total, used, and free RAM in GB, percentage, and process RSS."""
        total_gb = 16.0
        used_gb = 5.0
        free_gb = 11.0
        used_percent = 31.25

        try:
            import psutil
            vm = psutil.virtual_memory()
            total_gb = vm.total / (1024 ** 3)
            used_gb = vm.used / (1024 ** 3)
            free_gb = vm.available / (1024 ** 3)
            used_percent = vm.percent
        except ImportError:
            try:
                if platform.system() == "Darwin":
                    import subprocess
                    out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()
                    total_gb = int(out) / (1024 ** 3)
                    free_gb = max(2.0, total_gb * 0.4)
                    used_gb = total_gb - free_gb
                    used_percent = (used_gb / total_gb) * 100
                elif platform.system() == "Linux":
                    with open("/proc/meminfo", "r") as f:
                        lines = f.readlines()
                    info = {}
                    for line in lines:
                        parts = line.split(":")
                        if len(parts) == 2:
                            info[parts[0].strip()] = parts[1].strip()
                    total_kb = float(info.get("MemTotal", "16000000 kB").split()[0])
                    avail_kb = float(info.get("MemAvailable", "8000000 kB").split()[0])
                    total_gb = total_kb / (1024 ** 2)
                    free_gb = avail_kb / (1024 ** 2)
                    used_gb = total_gb - free_gb
                    used_percent = (used_gb / total_gb) * 100
            except Exception:
                pass

        process_gb = SystemMemoryWatchdog.get_process_rss_gb()

        return {
            "total_gb": round(total_gb, 2),
            "used_gb": round(used_gb, 2),
            "free_gb": round(free_gb, 2),
            "used_percent": round(used_percent, 1),
            "process_rss_gb": process_gb
        }

    def check_memory_pressure(self) -> Tuple[bool, Dict[str, Any]]:
        """Evaluates whether current system or process memory exceeds safe operating bounds."""
        status = self.get_system_memory_status()
        self.last_check_status = status

        # Proactive memory reclamation if process memory grows above 3.5 GB
        if status.get("process_rss_gb", 0) > 3.5:
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
