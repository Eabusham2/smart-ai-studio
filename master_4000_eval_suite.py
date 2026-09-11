"""Canonical entry point for the merged 4,014-item evaluation suite."""
from eval._master_4000_base import *
from eval.master_4000_runtime import install
from eval.recent20_telemetry import install as install_recent20_telemetry

install(Master4000EvaluationEngine)
install_recent20_telemetry(Master4000EvaluationEngine)

if __name__ == "__main__":
    Master4000EvaluationEngine(max_duration_hours=72.0).run_full_suite()
