"""Exercise application memory accounting and media/tuning lock boundaries."""
import sys
import threading
from types import SimpleNamespace

from core.media_orchestrator import MediaController
from core.native_speed import tuning_memory_ok


def test_worker_accounts_for_parent_and_siblings_in_user_limit(monkeypatch):
    gb = 1024 ** 3
    processes = {}

    class Process:
        def __init__(self, pid=None):
            self.pid = 300 if pid is None else pid

        def memory_info(self):
            return SimpleNamespace(rss=processes[self.pid][0])

        def children(self, recursive=False):
            assert recursive
            return [Process(pid) for pid in processes[self.pid][1]]

    processes.update({100: (3 * gb, [200, 300]), 200: (2 * gb, []), 300: (4 * gb, [])})
    fake = SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(total=32 * gb, available=12 * gb),
        Process=Process, NoSuchProcess=ProcessLookupError, AccessDenied=PermissionError,
    )
    monkeypatch.setitem(sys.modules, "psutil", fake)
    assert tuning_memory_ok(8)  # The worker alone fits.
    assert not tuning_memory_ok(8, owner_pid=100)  # The complete app does not.
    assert tuning_memory_ok(10, owner_pid=100)


def test_media_pause_takes_lifecycle_before_runtime_and_reenters_for_reload():
    held = []

    class CheckedLock:
        def __init__(self, name):
            self.name = name
            self.lock = threading.RLock()

        def __enter__(self):
            if self.name == "runtime":
                assert "lifecycle" in held
            self.lock.acquire()
            held.append(self.name)
            return self

        def __exit__(self, *args):
            assert held.pop() == self.name
            self.lock.release()

    lifecycle, runtime = CheckedLock("lifecycle"), CheckedLock("runtime")
    engine = SimpleNamespace(_model_lock=lifecycle, awake_consolidator=None)
    backend = SimpleNamespace(_runtime_model_lock=runtime)
    controller = MediaController.__new__(MediaController)
    with controller._pause_guard(engine, backend):
        # unload/load reenters the lifecycle lock while the lease holds runtime.
        with engine._model_lock:
            assert held == ["lifecycle", "runtime", "lifecycle"]
    assert held == []
