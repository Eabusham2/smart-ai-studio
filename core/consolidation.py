"""
Consolidation Module Interface.
Re-exports consolidation engine components for core module consistency.
"""

from consolidation.daemon import SleepConsolidationDaemon
from consolidation.ewc_loss import EWCLossCalculator
from consolidation.fisher import FisherEstimator

__all__ = [
    "SleepConsolidationDaemon",
    "EWCLossCalculator",
    "FisherEstimator"
]
