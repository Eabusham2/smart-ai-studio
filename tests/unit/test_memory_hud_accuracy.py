"""HUD totals use the same owned-process measurement as the memory ceiling."""
from types import SimpleNamespace
import os
import sys

import pytest

import app_gui


def test_hud_reports_measured_total_even_when_unloaded_and_uses_binary_units(monkeypatch):
    seen, shown = [], {}

    def measure(**kwargs):
        seen.append(kwargs)
        return {"total_allocated_gb": 6.25, "system_used_gb": 8.75, "system_total_gb": 16,
                "process_memory_known": True, "system_memory_known": True,
                "model_metal_active_gb": 4.0}

    monkeypatch.setattr(app_gui.SystemMemoryWatchdog, "get_detailed_memory_breakdown", measure)
    app = SimpleNamespace(lbl_vram=SimpleNamespace(configure=lambda **kw: shown.update(kw)),
                          _owned_runtime_pids=lambda: [42], is_model_loaded=False)
    app_gui.SmartAIChatbotApp._update_memory_hud_badge(app)
    assert seen == [{"owned_pids": [42]}]
    assert shown["text"] == "💾 AI RAM: 6.2 GiB | Total RAM: 8.8/16 GiB"


def test_hud_unknown_is_not_reported_as_zero_or_ready(monkeypatch):
    shown = {}
    monkeypatch.setattr(app_gui.SystemMemoryWatchdog, "get_detailed_memory_breakdown",
                        lambda: {"process_memory_known": False, "system_memory_known": False})
    app = SimpleNamespace(lbl_vram=SimpleNamespace(configure=lambda **kw: shown.update(kw)),
                          _owned_runtime_pids=lambda: [], is_model_loaded=True)
    app_gui.SmartAIChatbotApp._update_memory_hud_badge(app)
    assert shown["text"] == "💾 AI RAM: unavailable | Total RAM: unavailable"


def test_system_monitor_uses_owned_worker_total_and_labels_gib(monkeypatch, tmp_path):
    from core.tools import AgentToolRegistry

    seen = []

    def measure(**kwargs):
        seen.append(kwargs)
        return {"total_allocated_gb": 6.25, "app_rss_gb": 1., "app_footprint_gb": 2.,
                "owned_process_footprint_gb": 4.25,
                "system_used_gb": 8.75, "system_total_gb": 16, "system_free_gb": 7.25,
                "system_percent": 54.7, "model_metal_active_gb": 1., "model_metal_peak_gb": 2.,
                "process_memory_known": True, "system_memory_known": True}

    monkeypatch.setattr(app_gui.SystemMemoryWatchdog, "get_detailed_memory_breakdown", measure)
    tools = AgentToolRegistry(workspace_dir=str(tmp_path), owned_processes_provider=lambda: [42])
    ok, output = tools.execute_tool("system_monitor", {})
    assert ok and seen == [{"owned_pids": [42]}]
    assert "6.25 GiB" in output and "4.25 GiB" in output
    assert "7.2 GiB" in output and "not added again" in output
    assert " GB" not in output and "1.58-Bit" not in output


class _EvalProcess:
    def __init__(self, pid, *, ancestors=(), started=10., alive=True):
        self.pid = pid
        self.ancestors = list(ancestors)
        self.started = started
        self.alive = alive
        self.descendants = []

    def ppid(self):
        return self.ancestors[0].pid

    def create_time(self):
        return self.started

    def is_running(self):
        return self.alive

    def parents(self):
        return self.ancestors

    def children(self, recursive):
        assert recursive is True
        return self.descendants


def _eval_fixture(monkeypatch, *, poll=None):
    app_process = _EvalProcess(os.getpid())
    parent = _EvalProcess(1001, ancestors=[app_process])
    child = _EvalProcess(1002, ancestors=[parent, app_process])
    grandchild = _EvalProcess(1003, ancestors=[child, parent, app_process])
    parent.descendants = [child, grandchild]
    looked_up = []

    def get_process(pid):
        looked_up.append(pid)
        assert pid == parent.pid  # Do not scan or look up unrelated app children.
        return parent

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=get_process, Error=ProcessLookupError))
    app = SimpleNamespace(
        engine=SimpleNamespace(active_backend="gguf",
                               gguf_backend=SimpleNamespace(owned_process_ids=lambda: [42, child.pid]),
                               controller_backend=SimpleNamespace(owned_process_ids=lambda: [43])),
        _strata_unload_pending=True, _native_unload_pending=True,
        _eval_proc=SimpleNamespace(pid=parent.pid, poll=poll or (lambda: None)),
    )
    return app, parent, child, grandchild, looked_up


def test_live_eval_parent_descendants_and_backend_owners_are_deduplicated(monkeypatch):
    app, parent, child, grandchild, looked_up = _eval_fixture(monkeypatch)
    parent.descendants.append(child)  # Duplicate discovery must not inflate RAM.
    assert app_gui.SmartAIChatbotApp._owned_runtime_pids(app) == [42, 43, 1001, 1002, 1003]
    assert looked_up == [1001]


def test_ended_eval_is_not_discovered_or_counted(monkeypatch):
    app, _, _, _, looked_up = _eval_fixture(monkeypatch, poll=lambda: 0)
    assert app_gui.SmartAIChatbotApp._owned_runtime_pids(app) == [42, 43, 1002]
    assert looked_up == []


@pytest.mark.parametrize("stale", ["wrong_parent", "reused_identity", "exited_during_read", "replaced_owner"])
def test_stale_eval_owner_is_excluded_without_losing_backend_owners(monkeypatch, stale):
    app, parent, _, _, _ = _eval_fixture(monkeypatch)
    if stale == "wrong_parent":
        parent.ancestors = [_EvalProcess(9999)]
    elif stale == "reused_identity":
        parent.alive = False  # psutil.is_running rejects an old PID/create_time pair.
    elif stale == "exited_during_read":
        statuses = iter([None, 0])
        app._eval_proc.poll = lambda: next(statuses)
    else:
        original_children = parent.children

        def children(recursive):
            app._eval_proc = SimpleNamespace(pid=7777, poll=lambda: None)
            return original_children(recursive)

        parent.children = children
    assert app_gui.SmartAIChatbotApp._owned_runtime_pids(app) == [42, 43, 1002]


def test_eval_excludes_unrelated_dead_reparented_and_reused_descendants(monkeypatch):
    app, parent, child, grandchild, _ = _eval_fixture(monkeypatch)
    unrelated = _EvalProcess(2001, ancestors=[_EvalProcess(os.getpid())])
    dead = _EvalProcess(2002, ancestors=[parent], alive=False)
    reused = _EvalProcess(2003, ancestors=[_EvalProcess(parent.pid, started=99.)])
    departed = _EvalProcess(2004, ancestors=[parent])
    departed.parents = lambda: (_ for _ in ()).throw(ProcessLookupError("finished"))
    parent.descendants.extend([unrelated, dead, reused, departed])
    assert app_gui.SmartAIChatbotApp._owned_runtime_pids(app) == [42, 43, parent.pid, child.pid, grandchild.pid]
