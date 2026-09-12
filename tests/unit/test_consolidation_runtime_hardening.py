from types import SimpleNamespace

import pytest

from consolidation.runtime_hardening import install_sleep_daemon_hardening


class DummyDaemon:
    def __init__(self, use_mock=False, init_success=True):
        self.settings = SimpleNamespace(use_mock=use_mock)
        self.model = None
        self.init_success = init_success
        self.original_runs = 0

    def _init_model_and_lora(self):
        if self.init_success:
            self.model = object()

    def run_consolidation_cycle(self):
        self.original_runs += 1
        if self.model is None:
            return {"status": "success", "simulated": True}
        return {"status": "success", "simulated": False}


install_sleep_daemon_hardening(DummyDaemon)


def test_live_mode_initializes_real_model_before_cycle():
    daemon = DummyDaemon(use_mock=False, init_success=True)
    result = daemon.run_consolidation_cycle()
    assert result == {"status": "success", "simulated": False}
    assert daemon.original_runs == 1


def test_live_mode_never_reaches_mock_simulation_if_init_fails():
    daemon = DummyDaemon(use_mock=False, init_success=False)
    with pytest.raises(RuntimeError, match="refusing mock-simulation success"):
        daemon.run_consolidation_cycle()
    assert daemon.original_runs == 0


def test_explicit_mock_mode_preserves_unit_test_simulation():
    daemon = DummyDaemon(use_mock=True, init_success=False)
    result = daemon.run_consolidation_cycle()
    assert result == {"status": "success", "simulated": True}
