"""Never enter PyTorch's native MPS cache without an available MPS device."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


training = load("_safe_mps_training_tests", "core/training_memory.py")
watchdog = load("_safe_mps_watchdog_tests", "core/memory_watchdog.py")
audio = load("_safe_mps_audio_tests", "core/audio_engine.py")
media = load("_safe_mps_media_tests", "core/media_engine.py")


class TorchWithoutMPS:
    def __init__(self, *, available=False, probe=True):
        self.cuda = types.SimpleNamespace(is_available=lambda: False)
        self.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: available)) if probe else types.SimpleNamespace()
        self.mps_accesses = 0

    @property
    def mps(self):
        self.mps_accesses += 1
        raise AssertionError("Entering this native MPS allocator would abort on this host")


class TestMPSAvailability(unittest.TestCase):
    @staticmethod
    def media_engines():
        audio_engine = audio.AudioGenerationEngine()
        audio_engine.model = object()
        audio_engine.backend = "test-audio"
        audio_engine.repo_id = "test/audio"
        media_engine = media.MediaGenerationEngine()
        media_engine.active_pipeline = object()
        media_engine.active_repo_id = "test/media"
        return audio_engine, media_engine

    def test_unavailable_or_missing_backend_never_touches_native_mps_namespace(self):
        for torch in (TorchWithoutMPS(), TorchWithoutMPS(probe=False)):
            with self.subTest(probe=hasattr(torch.backends, "mps")):
                training.clear_available_mps_cache(torch)
                self.assertEqual(torch.mps_accesses, 0)

    def test_available_mps_releases_the_cache_once(self):
        empty = Mock()
        torch = types.SimpleNamespace(
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
            mps=types.SimpleNamespace(empty_cache=empty))
        training.clear_available_mps_cache(torch)
        empty.assert_called_once_with()

    def test_real_cleanup_entrypoints_skip_unavailable_mps_and_still_report_memory(self):
        torch = TorchWithoutMPS()
        core = types.ModuleType("core")
        core.__path__ = [str(ROOT / "core")]
        with patch.dict(sys.modules, {"torch": torch, "mlx": None, "mlx.core": None,
                                     "core": core, "core.training_memory": training}), \
             patch.object(training, "process_rss_mb", side_effect=[100.0, 80.0]), \
             patch.object(training.gc, "collect"):
            result = training.release_training_memory()
            watchdog.SystemMemoryWatchdog.reclaim_process_memory()
        self.assertEqual(torch.mps_accesses, 0)
        self.assertEqual(result, {"before_mb": 100.0, "after_mb": 80.0, "released_mb": 20.0})

    def test_probe_failure_does_not_abort_cleanup_or_reach_native_cache(self):
        torch = TorchWithoutMPS()
        torch.backends.mps.is_available = Mock(side_effect=RuntimeError("Metal probe unavailable"))
        core = types.ModuleType("core")
        core.__path__ = [str(ROOT / "core")]
        with patch.dict(sys.modules, {"torch": torch, "mlx": None, "mlx.core": None,
                                     "core": core, "core.training_memory": training}), \
             patch.object(training, "process_rss_mb", return_value=100.0), \
             patch.object(training.gc, "collect"):
            result = training.release_training_memory()
            watchdog.SystemMemoryWatchdog.reclaim_process_memory()
        self.assertEqual(torch.mps_accesses, 0)
        self.assertEqual(result["released_mb"], 0.0)

    def test_audio_and_media_unload_skip_unavailable_or_missing_native_mps(self):
        for probe in (True, False):
            for engine in self.media_engines():
                with self.subTest(engine=type(engine).__name__, probe=probe):
                    torch = TorchWithoutMPS(probe=probe)
                    core = types.ModuleType("core")
                    core.__path__ = [str(ROOT / "core")]
                    with patch.dict(sys.modules, {"torch": torch, "mlx": None, "mlx.core": None,
                                                 "core": core, "core.training_memory": training}), \
                         patch.object(training.gc, "collect"):
                        engine.unload_model()
                    self.assertEqual(torch.mps_accesses, 0)
                    self.assertIsNone(getattr(engine, "model", getattr(engine, "active_pipeline", None)))
                    self.assertEqual(getattr(engine, "repo_id", getattr(engine, "active_repo_id", None)), "")

    def test_audio_and_media_unload_preserve_available_device_cleanup(self):
        for cuda_available in (False, True):
            for engine in self.media_engines():
                with self.subTest(engine=type(engine).__name__, cuda=cuda_available):
                    empty_mps, empty_cuda = Mock(), Mock()
                    torch = types.SimpleNamespace(
                        cuda=types.SimpleNamespace(is_available=lambda: cuda_available, empty_cache=empty_cuda),
                        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
                        mps=types.SimpleNamespace(empty_cache=empty_mps),
                    )
                    core = types.ModuleType("core")
                    core.__path__ = [str(ROOT / "core")]
                    with patch.dict(sys.modules, {"torch": torch, "mlx": None, "mlx.core": None,
                                                 "core": core, "core.training_memory": training}), \
                         patch.object(training.gc, "collect"):
                        engine.unload_model()
                    empty_mps.assert_called_once_with()
                    self.assertEqual(empty_cuda.call_count, int(cuda_available))

    def test_audio_and_media_unload_survive_a_failing_mps_availability_probe(self):
        for engine in self.media_engines():
            with self.subTest(engine=type(engine).__name__):
                torch = TorchWithoutMPS()
                torch.backends.mps.is_available = Mock(side_effect=RuntimeError("Metal probe unavailable"))
                core = types.ModuleType("core")
                core.__path__ = [str(ROOT / "core")]
                with patch.dict(sys.modules, {"torch": torch, "mlx": None, "mlx.core": None,
                                             "core": core, "core.training_memory": training}), \
                     patch.object(training.gc, "collect"):
                    engine.unload_model()
                self.assertEqual(torch.mps_accesses, 0)
                self.assertIsNone(getattr(engine, "model", getattr(engine, "active_pipeline", None)))


if __name__ == "__main__":
    unittest.main()
