"""Canonical entry point for the merged 4,014-item evaluation suite."""
from eval._master_4000_base import *
from eval.master_4000_runtime import install
from eval.phase4_pro_rsi import install as install_phase4_pro_rsi

install(Master4000EvaluationEngine)
install_phase4_pro_rsi(Master4000EvaluationEngine)

if __name__ == "__main__":
    Master4000EvaluationEngine(max_duration_hours=72.0).run_full_suite()
