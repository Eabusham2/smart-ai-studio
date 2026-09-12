"""Canonical entry point for the merged 4,014-item evaluation suite."""
from eval._master_4000_base import *
import eval.master_4000_runtime as master_runtime
import eval.phase4_pro_rsi as phase4_pro_rsi
from eval.dataset_hardening import install as install_dataset_hardening
from eval.historical_good_merge import install as install_historical_good_merge
from eval.scoring_hardening import install as install_scoring_hardening

master_runtime.install(Master4000EvaluationEngine)
install_dataset_hardening(master_runtime, phase4_pro_rsi)
phase4_pro_rsi.install(Master4000EvaluationEngine)
install_historical_good_merge(phase4_pro_rsi)
install_scoring_hardening(Master4000EvaluationEngine, phase4_pro_rsi)

if __name__ == "__main__":
    Master4000EvaluationEngine(max_duration_hours=72.0).run_full_suite()
