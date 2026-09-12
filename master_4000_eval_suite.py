"""Canonical entry point for the merged 4,014-item evaluation suite."""
from eval._master_4000_base import *
from eval.master_4000_runtime import install
import eval.phase4_pro_rsi as phase4_pro_rsi
from eval.historical_good_merge import install as install_historical_good_merge
from eval.scoring_hardening import install as install_scoring_hardening

install(Master4000EvaluationEngine)
phase4_pro_rsi.install(Master4000EvaluationEngine)
install_historical_good_merge(phase4_pro_rsi)
install_scoring_hardening(Master4000EvaluationEngine, phase4_pro_rsi)

if __name__ == "__main__":
    Master4000EvaluationEngine(max_duration_hours=72.0).run_full_suite()
