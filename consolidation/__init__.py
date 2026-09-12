from consolidation.fisher import FisherEstimator
from consolidation.ewc_loss import EWCLossCalculator
from consolidation.daemon import SleepConsolidationDaemon
from consolidation.runtime_hardening import install_sleep_daemon_hardening

install_sleep_daemon_hardening(SleepConsolidationDaemon)

__all__ = [
    "FisherEstimator",
    "EWCLossCalculator",
    "SleepConsolidationDaemon",
]
