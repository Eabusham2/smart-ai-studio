"""Canonical entry point for the merged 4,014-item evaluation suite."""
from eval._master_4000_base import *
from eval.master_4000_runtime import install
from eval.master_4000_runtime import SYSTEM_PROMPT
from eval.master_4000_runtime_fix import install as install_runtime_fix

install(Master4000EvaluationEngine)
install_runtime_fix(Master4000EvaluationEngine, SYSTEM_PROMPT)

if __name__ == "__main__":
    Master4000EvaluationEngine(max_duration_hours=72.0).run_full_suite()
