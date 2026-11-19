"""Owned native workers share Studio's user-selected memory ceiling."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("_strata_memory_watchdog_tests", ROOT / "core/memory_watchdog.py")
memory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(memory)
GB = 1024 ** 3


class TestOwnedWorkerMemory(unittest.TestCase):
    def setUp(self):
        self.reads = []
        self.rss = {None: 2 * GB, 111: 4 * GB, 222: 6 * GB}
        owner = self

        def process(pid=None):
            owner.reads.append(pid)
            if pid not in owner.rss:
                raise ProcessLookupError("worker already exited")
            return types.SimpleNamespace(memory_info=lambda: types.SimpleNamespace(rss=owner.rss[pid]))

        psutil = types.ModuleType("psutil")
        psutil.Process = process
        psutil.virtual_memory = lambda: types.SimpleNamespace(
            total=32 * GB, used=10 * GB, available=22 * GB, percent=31.25)
        self.modules = patch.dict(sys.modules, {"psutil": psutil, "mlx": None, "mlx.core": None})
        self.modules.start()

    def tearDown(self):
        self.modules.stop()

    def test_hud_deduplicates_verified_workers_excludes_studio_and_skips_exited_pids(self):
        own = os.getpid()
        result = memory.SystemMemoryWatchdog.get_detailed_memory_breakdown(
            owned_pids=[111, 111, own, 222, 333, True, -1, 0, "111", None, {}])
        self.assertEqual(result["app_rss_gb"], 2)
        self.assertEqual(result["owned_process_rss_gb"], 10)
        self.assertEqual(result["owned_process_count"], 2)
        self.assertEqual(result["total_allocated_gb"], 12)
        self.assertEqual(self.reads, [None, 111, 222, 333])
        self.assertEqual(result["system_free_gb"], 22)

    def test_owned_memory_alone_can_cross_process_cap_with_ample_system_ram(self):
        watchdog = memory.SystemMemoryWatchdog(
            max_process_ram_gb=10, additional_processes_provider=lambda: [111, 222, 111])
        with patch.object(watchdog, "adjust_dynamic_metal_headroom"), \
             patch.object(watchdog, "reclaim_process_memory") as reclaim:
            pressure, status = watchdog.check_memory_pressure()
        self.assertTrue(pressure)
        self.assertEqual(status["process_rss_gb"], 12)
        self.assertEqual(status["app_rss_gb"], 2)
        self.assertEqual(status["owned_process_rss_gb"], 10)
        self.assertEqual(status["free_gb"], 22)
        reclaim.assert_called_once_with()
        self.assertIs(watchdog.last_check_status, status)

    def test_provider_is_refreshed_after_worker_replacement_and_unload(self):
        selected = [111]
        watchdog = memory.SystemMemoryWatchdog(
            max_process_ram_gb=7, additional_processes_provider=lambda: selected)
        with patch.object(watchdog, "adjust_dynamic_metal_headroom"), \
             patch.object(watchdog, "reclaim_process_memory"):
            self.assertFalse(watchdog.check_memory_pressure()[0])
            selected[:] = [222]
            pressure, status = watchdog.check_memory_pressure()
            self.assertTrue(pressure)
            self.assertEqual(status["process_rss_gb"], 8)
            selected.clear()
            pressure, status = watchdog.check_memory_pressure()
            self.assertFalse(pressure)
            self.assertEqual(status["process_rss_gb"], 2)
            self.assertEqual(status["owned_process_count"], 0)

    def test_default_empty_and_failing_provider_preserve_zero_argument_status_mocks(self):
        status = {"used_percent": 20, "free_gb": 20, "process_rss_gb": 2}

        def failing():
            raise RuntimeError("runtime is unloading")

        for provider in (None, lambda: [], failing):
            with self.subTest(provider=provider):
                watchdog = memory.SystemMemoryWatchdog(additional_processes_provider=provider)
                with patch.object(watchdog, "get_system_memory_status", side_effect=lambda: status) as get, \
                     patch.object(watchdog, "adjust_dynamic_metal_headroom"), \
                     patch.object(watchdog, "reclaim_process_memory"):
                    self.assertEqual(watchdog.check_memory_pressure(), (False, status))
                get.assert_called_once_with()

    def test_static_status_and_process_helpers_accept_explicit_owned_workers(self):
        status = memory.SystemMemoryWatchdog.get_system_memory_status(owned_pids=[222])
        self.assertEqual(status["process_rss_gb"], 8)
        self.assertEqual(status["owned_process_count"], 1)
        self.assertEqual(memory.SystemMemoryWatchdog.get_process_rss_gb(owned_pids=[111]), 6)
        self.assertEqual(memory.SystemMemoryWatchdog.get_process_rss_gb(), 2)

    def test_metal_is_telemetry_and_not_added_to_process_total_twice(self):
        mlx = types.ModuleType("mlx")
        mlx.__path__ = []
        mx = types.ModuleType("mlx.core")
        mx.get_active_memory = lambda: 3 * GB
        mx.get_peak_memory = lambda: 5 * GB
        mx.get_cache_memory = lambda: GB
        mlx.core = mx
        with patch.dict(sys.modules, {"mlx": mlx, "mlx.core": mx}):
            result = memory.SystemMemoryWatchdog.get_detailed_memory_breakdown(owned_pids=[111, 111])
        self.assertEqual(result["app_rss_gb"], 2)
        self.assertEqual(result["model_metal_active_gb"], 3)
        self.assertEqual(result["model_metal_peak_gb"], 5)
        self.assertEqual(result["owned_process_rss_gb"], 4)
        self.assertEqual(result["total_allocated_gb"], 6)

    def test_mac_physical_footprint_includes_metal_and_workers_once(self):
        footprints = {None: 5 * GB, 111: 7 * GB}
        with patch.object(memory.platform, "system", return_value="Darwin"), \
             patch("core.training_memory.process_memory_bytes", side_effect=lambda pid=None: footprints[pid]):
            result = memory.SystemMemoryWatchdog.get_detailed_memory_breakdown(owned_pids=[111, 111])
        self.assertEqual(result["app_rss_gb"], 2)
        self.assertEqual(result["app_footprint_gb"], 5)
        self.assertEqual(result["owned_process_footprint_gb"], 7)
        self.assertEqual(result["total_allocated_gb"], 12)

    def test_unknown_system_memory_does_not_fabricate_a_machine_or_unload(self):
        with patch.object(sys.modules["psutil"], "virtual_memory", side_effect=OSError("unavailable")):
            result = memory.SystemMemoryWatchdog.get_detailed_memory_breakdown()
            watchdog = memory.SystemMemoryWatchdog(max_process_ram_gb=10)
            with patch.object(watchdog, "adjust_dynamic_metal_headroom"):
                pressure, _ = watchdog.check_memory_pressure()
        self.assertFalse(result["system_memory_known"])
        self.assertEqual(result["system_total_gb"], 0)
        self.assertFalse(pressure)

    def test_optional_metal_cache_respects_os_reserve(self):
        mlx = types.ModuleType("mlx")
        mlx.__path__ = []
        mx = types.ModuleType("mlx.core")
        limits = []
        mx.set_cache_limit = limits.append
        mlx.core = mx
        with patch.dict(sys.modules, {"mlx": mlx, "mlx.core": mx}), \
             patch.object(sys.modules["psutil"], "virtual_memory", return_value=types.SimpleNamespace(available=GB)):
            memory.SystemMemoryWatchdog.adjust_dynamic_metal_headroom()
        self.assertEqual(limits, [0])


if __name__ == "__main__":
    unittest.main()
