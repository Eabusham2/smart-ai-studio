"""Canonical entry point for the merged 4,014-item evaluation suite."""
from eval._master_4000_base import *
import eval.live_generation_stream as live_generation_stream
import eval.master_4000_runtime as master_runtime
import eval.phase4_pro_rsi as phase4_pro_rsi
import eval.scoring_hardening as scoring_hardening
import eval.stage_integrity_telemetry as stage_integrity_telemetry
from eval.checkpoint_hardening import install as install_checkpoint_hardening
from eval.code_prompt_hardening import install as install_code_prompt_hardening
from eval.dataset_hardening import install as install_dataset_hardening
from eval.historical_good_merge import install as install_historical_good_merge
from eval.live_generation_stream import install_baseline_stream, install_phase4_stream
from eval.metal_branch_memory_hardening import install as install_metal_branch_memory_hardening
from eval.reader_hardening import install as install_reader_hardening
from eval.rsi_miss_only_hardening import capture_before_historical_merge, enforce_after_historical_merge
from eval.rsi_prompt_hardening import install as install_rsi_prompt_hardening
from eval.rsi_resume_hardening import install as install_rsi_resume_hardening
from eval.rsi_telemetry_compact import install as install_rsi_telemetry_compact
from eval.scoring_hardening import install as install_scoring_hardening
from eval.stage_integrity_telemetry import install as install_stage_integrity_telemetry
from eval.swe_verifier_hardening import install as install_swe_verifier_hardening

install_swe_verifier_hardening(master_runtime)
install_checkpoint_hardening(master_runtime)
install_reader_hardening(scoring_hardening)

master_runtime.install(Master4000EvaluationEngine)
install_baseline_stream(master_runtime, Master4000EvaluationEngine)
phase4_pro_rsi._append_raw_generation_log = master_runtime._append_raw_generation_log
install_dataset_hardening(master_runtime, phase4_pro_rsi)
install_code_prompt_hardening(master_runtime, phase4_pro_rsi, Master4000EvaluationEngine)
install_rsi_prompt_hardening(phase4_pro_rsi)
phase4_pro_rsi.install(Master4000EvaluationEngine)
install_phase4_stream(phase4_pro_rsi)
install_metal_branch_memory_hardening(phase4_pro_rsi, live_generation_stream)

capture_before_historical_merge(phase4_pro_rsi)
install_historical_good_merge(phase4_pro_rsi)
enforce_after_historical_merge(phase4_pro_rsi)
install_rsi_resume_hardening(phase4_pro_rsi)
install_stage_integrity_telemetry(phase4_pro_rsi, Master4000EvaluationEngine)
install_rsi_telemetry_compact(stage_integrity_telemetry)
install_scoring_hardening(Master4000EvaluationEngine, phase4_pro_rsi)

if __name__ == "__main__":
    try:
        Master4000EvaluationEngine(max_duration_hours=72.0).run_full_suite()
    except KeyboardInterrupt:
        print("\n[*] Ctrl+C: clean shutdown.", flush=True)
        raise SystemExit(130)
