"""Runtime integrity guard for SleepConsolidationDaemon.

Repairs two recovered compatibility/integrity issues:
- app/daemon callers still pass `use_mock=...`; accept that legacy keyword safely;
- live mode must initialize a real trainable model before consolidation and must
  never fall through to the daemon's mock-simulation branch.
"""
from __future__ import annotations


def install_sleep_daemon_hardening(cls) -> None:
    if getattr(cls, "_live_integrity_hardening_installed", False):
        return

    original_init = cls.__init__
    original_run = cls.run_consolidation_cycle

    def hardened_init(self, *args, **kwargs):
        explicit_mock = kwargs.pop("use_mock", None)
        original_init(self, *args, **kwargs)
        if explicit_mock is not None:
            try:
                self.settings.use_mock = bool(explicit_mock)
            except Exception:
                object.__setattr__(self.settings, "use_mock", bool(explicit_mock))

    def hardened_run(self, *args, **kwargs):
        if not getattr(self.settings, "use_mock", False) and self.model is None:
            self._init_model_and_lora()
        if not getattr(self.settings, "use_mock", False) and self.model is None:
            raise RuntimeError(
                "Live consolidation has no real trainable model; refusing mock-simulation success."
            )
        return original_run(self, *args, **kwargs)

    cls.__init__ = hardened_init
    cls.run_consolidation_cycle = hardened_run
    cls._live_integrity_hardening_installed = True
