"""Runtime integrity guard for SleepConsolidationDaemon.

The legacy daemon could enter its `self.model is None` mock-simulation branch even
when `settings.use_mock` was false, because the live model was never initialized
before the branch check. In live mode we now initialize the real trainable model
first and fail honestly if that cannot be done. Explicit mock tests remain intact.
"""
from __future__ import annotations


def install_sleep_daemon_hardening(cls) -> None:
    if getattr(cls, "_live_integrity_hardening_installed", False):
        return

    original_run = cls.run_consolidation_cycle

    def hardened_run(self, *args, **kwargs):
        if not getattr(self.settings, "use_mock", False) and self.model is None:
            self._init_model_and_lora()
        if not getattr(self.settings, "use_mock", False) and self.model is None:
            raise RuntimeError(
                "Live consolidation has no real trainable model; refusing mock-simulation success."
            )
        return original_run(self, *args, **kwargs)

    cls.run_consolidation_cycle = hardened_run
    cls._live_integrity_hardening_installed = True
