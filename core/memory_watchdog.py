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
        max_process_ram_gb: float = 13.0,
        on_pressure_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        additional_processes_provider: Optional[Callable[[], Any]] = None,
    ):
        self.check_interval_seconds = check_interval_seconds
        self.max_ram_usage_percent = max_ram_usage_percent
        self.min_free_ram_gb = min_free_ram_gb
        self.max_process_ram_gb = max_process_ram_gb
        self.on_pressure_callback = on_pressure_callback
        # The runtime owner verifies identity/lifetime before returning these PIDs.
        # The watchdog never discovers workers by names or unrelated child scans.
        self.additional_processes_provider = additional_processes_provider

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
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
            from core.training_memory import clear_available_mps_cache
            clear_available_mps_cache(torch)
        except Exception:
            pass

    @staticmethod
    def get_detailed_memory_breakdown(owned_pids=None) -> Dict[str, Any]:
        """
        Computes granular, authentic memory breakdown across:
        1. App Process RSS and optional, caller-verified owned worker RSS.
        2. Model Metal Active Memory (MLX unified memory tensors).
        3. Model Metal Peak & Cache Allocation.
        4. Host System RAM (Used, Available, Total).
        5. Dynamic RAM scaling parameters.
        """
        app_rss_gb = 0.0
        process_memory_known = False
        try:
            import psutil
            app_rss_gb = psutil.Process().memory_info().rss / (1024 ** 3)
            process_memory_known = True
        except Exception:
            # ru_maxrss is a historical peak, not live RAM; displaying it as
            # current allocation makes a completed consolidation look leaked.
            pass

        # Metal's active/cache counters overlap process memory. Adding them to
        # RSS counts the same unified-memory pages twice. macOS physical
        # footprint also covers compressed/Metal memory which RSS can omit.
        app_footprint_gb = app_rss_gb
        accounting_method = "rss"
        if platform.system() == "Darwin":
            from core.training_memory import process_memory_bytes
            footprint = process_memory_bytes()
            if footprint > 0:
                app_footprint_gb = footprint / (1024 ** 3)
                accounting_method = "physical_footprint_or_rss"
                process_memory_known = True

        owned_rss_gb = 0.0
        owned_footprint_gb = 0.0
        owned_process_count = 0
        owned_memory_complete = True
        if owned_pids:
            try:
                import psutil
                exited = (ProcessLookupError, getattr(psutil, "NoSuchProcess", ProcessLookupError))
                seen = {os.getpid()}
                for pid in owned_pids:
                    if type(pid) is not int or pid <= 0 or pid in seen:
                        continue
                    seen.add(pid)
                    try:
                        rss = int(psutil.Process(pid).memory_info().rss)
                        if rss < 0:
                            owned_memory_complete = False
                            continue
                        owned_rss_gb += rss / (1024 ** 3)
                        footprint = process_memory_bytes(pid) if platform.system() == "Darwin" else rss
                        owned_footprint_gb += (footprint if footprint > 0 else rss) / (1024 ** 3)
                        owned_process_count += 1
                    except exited:
                        # A verified worker can exit between discovery and this read.
                        continue
                    except Exception:
                        owned_memory_complete = False
            except (ImportError, TypeError):
                owned_memory_complete = False

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

        system_total_gb = system_used_gb = system_free_gb = system_percent = 0.0
        system_memory_known = False

        try:
            import psutil
            vm = psutil.virtual_memory()
            system_total_gb = vm.total / (1024 ** 3)
            system_used_gb = vm.used / (1024 ** 3)
            system_free_gb = vm.available / (1024 ** 3)
            system_percent = vm.percent
            system_memory_known = True
        except Exception:
            # Unknown is not a fabricated 16 GB machine or a measured zero-free
            # event. The watchdog can still enforce the measured process cap.
            pass

        total_allocated_gb = app_footprint_gb + owned_footprint_gb

        return {
            "app_rss_gb": round(app_rss_gb, 2),
            "app_footprint_gb": round(app_footprint_gb, 2),
            "owned_process_rss_gb": round(owned_rss_gb, 2),
            "owned_process_footprint_gb": round(owned_footprint_gb, 2),
            "owned_process_count": owned_process_count,
            "owned_memory_complete": owned_memory_complete,
            "model_metal_active_gb": round(model_active_gb, 2),
            "model_metal_peak_gb": round(max(model_active_gb, model_peak_gb), 2),
            "model_metal_cache_gb": round(model_cache_gb, 2),
            "total_allocated_gb": round(total_allocated_gb, 2),
            "total_allocated_bytes": round(total_allocated_gb * (1024 ** 3)),
            "system_total_gb": round(system_total_gb, 1),
            "system_used_gb": round(system_used_gb, 1),
            "system_free_gb": round(system_free_gb, 1),
            "system_percent": round(system_percent, 1),
            "system_memory_known": system_memory_known,
            "accounting_method": accounting_method,
            "process_memory_known": process_memory_known,
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
            # The cache is optional. Never impose a 2.5 GB minimum when the
            # machine is already short of memory, or spend the OS reserve.
            dynamic_cache_gb = max(0.0, min(avail_gb * 0.25, avail_gb - min_headroom_gb))
            dynamic_cache_bytes = int(dynamic_cache_gb * (1024 ** 3))

            # Requests can enforce a smaller cap using device/process headroom
            # or OOM recovery. The monitor runs concurrently and must never
            # replace that cap with a larger host-RAM-derived allowance.
            from core.mlx_memory_guard import set_allocator_cache_limit
            set_allocator_cache_limit(mx, dynamic_cache_bytes, only_reduce=True)
        except Exception:
            pass

    @staticmethod
    def get_process_rss_gb(owned_pids=None) -> float:
        """Return Studio's accounted memory plus any verified owned workers, in GB."""
        breakdown = (SystemMemoryWatchdog.get_detailed_memory_breakdown()
                     if owned_pids is None else SystemMemoryWatchdog.get_detailed_memory_breakdown(owned_pids=owned_pids))
        return breakdown["total_allocated_gb"]

    @staticmethod
    def get_system_memory_status(owned_pids=None) -> Dict[str, Any]:
        """Queries host total, used, and free RAM in GB matching macOS Activity Monitor."""
        breakdown = (SystemMemoryWatchdog.get_detailed_memory_breakdown()
                     if owned_pids is None else SystemMemoryWatchdog.get_detailed_memory_breakdown(owned_pids=owned_pids))
        return {
            "total_gb": breakdown["system_total_gb"],
            "used_gb": breakdown["system_used_gb"],
            "free_gb": breakdown["system_free_gb"],
            "used_percent": breakdown["system_percent"],
            "process_rss_gb": breakdown["total_allocated_gb"],
            "app_rss_gb": breakdown["app_rss_gb"],
            "app_footprint_gb": breakdown.get("app_footprint_gb", breakdown["app_rss_gb"]),
            "owned_process_rss_gb": breakdown.get("owned_process_rss_gb", 0.0),
            "owned_process_count": breakdown.get("owned_process_count", 0),
            "model_metal_active_gb": breakdown["model_metal_active_gb"],
            "model_metal_peak_gb": breakdown["model_metal_peak_gb"],
            "system_memory_known": breakdown.get("system_memory_known", True),
            "accounting_method": breakdown.get("accounting_method", "rss"),
        }

    def check_memory_pressure(self) -> Tuple[bool, Dict[str, Any]]:
        """Evaluates whether current system or process memory exceeds safe operating bounds."""
        owned_pids = None
        provider = self.additional_processes_provider
        if callable(provider):
            try:
                owned_pids = list(provider() or ())
            except Exception:
                # Preserve host/Studio monitoring if a runtime is being replaced.
                owned_pids = None
        status = (self.get_system_memory_status(owned_pids=owned_pids)
                  if owned_pids else self.get_system_memory_status())
        self.last_check_status = status

        # Dynamically scale Metal cache headroom based on available memory
        self.adjust_dynamic_metal_headroom()

        # Reclaim slightly before the user-selected hard process/model ceiling.
        proactive_limit = max(0.5, float(self.max_process_ram_gb) - 0.5)
        if status.get("process_rss_gb", 0) >= proactive_limit:
            self.reclaim_process_memory()

        is_under_pressure = (
            (status.get("system_memory_known", True) and (
                status["used_percent"] >= self.max_ram_usage_percent or
                status["free_gb"] < self.min_free_ram_gb)) or
            status.get("process_rss_gb", 0) >= self.max_process_ram_gb
        )

        return is_under_pressure, status

    def set_memory_limit_gb(self, limit_gb: float):
        """Updates the live hard ceiling without restarting the monitor."""
        self.max_process_ram_gb = max(0.5, float(limit_gb))

    def start_monitoring(self):
        """Launches one continuous watchdog background thread."""
        if self._running and self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        stop_event = threading.Event()
        self._stop_event = stop_event
        self._thread = threading.Thread(
            target=self._monitor_loop,
            args=(stop_event,),
            daemon=True,
        )
        self._thread.start()

    def stop_monitoring(self):
        """Stops the current watchdog run without affecting a later restart."""
        self._running = False
        stop_event = self._stop_event
        if stop_event is not None:
            stop_event.set()
        self._stop_event = None
        self._thread = None

    def _monitor_loop(self, stop_event: threading.Event):
        while not stop_event.is_set():
            try:
                pressure, status = self.check_memory_pressure()
                if pressure and self.on_pressure_callback:
                    self.on_pressure_callback(status)
            except Exception:
                pass
            stop_event.wait(self.check_interval_seconds)
