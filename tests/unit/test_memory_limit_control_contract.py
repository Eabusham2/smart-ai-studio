from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_top_bar_memory_limit_control_is_persistent_and_visible():
    src = _src("app_gui.py")
    assert 'APP_PREFS_FILE = os.path.join(get_portable_data_dir(), "app_preferences.json")' in src
    assert 'text="Mem Limit"' in src
    assert "self.entry_memory_limit.pack(" in src
    assert 'text="GB"' in src
    assert 'data["memory_limit"]' in src
    assert '"enabled": bool(self._memory_limit_enabled)' in src
    assert '"limit_gb": round(float(self._memory_limit_gb), 2)' in src


def test_default_limit_is_12_5_or_1_2x_model_size():
    src = _src("app_gui.py")
    assert "return round(max(12.5, model_gb * 1.2), 1)" in src


def test_watcher_lifecycle_follows_loaded_model_and_tick():
    src = _src("app_gui.py")
    # Watchdog is constructed but not unconditionally started at app startup.
    init_anchor = src.index("self.watchdog = SystemMemoryWatchdog(")
    build_anchor = src.index("self._init_window()", init_anchor)
    assert "self.watchdog.start_monitoring()" not in src[init_anchor:build_anchor]

    assert "not self._memory_limit_enabled" in src
    assert 'not getattr(self, "is_model_loaded", False)' in src
    assert "watchdog.stop_monitoring()" in src
    assert "watchdog.start_monitoring()" in src
    assert "self._sync_memory_watchdog(target_info)" in src
    assert "self._sync_memory_watchdog(force_stop=True)" in src


def test_watcher_restart_is_generation_safe():
    src = _src("core/memory_watchdog.py")
    assert "self._stop_event: Optional[threading.Event] = None" in src
    assert "stop_event = threading.Event()" in src
    assert "args=(stop_event,)" in src
    assert "stop_event.wait(self.check_interval_seconds)" in src
    assert "def set_memory_limit_gb" in src


def test_stale_pressure_events_cannot_hit_new_model():
    src = _src("app_gui.py")
    assert "self._memory_watchdog_epoch = 0" in src
    assert '("memory_pressure", (self._memory_watchdog_epoch, status))' in src
    assert "event_epoch == self._memory_watchdog_epoch" in src
