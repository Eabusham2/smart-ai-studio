"""Canonical entry point for the merged 4,014-item evaluation suite."""
from eval._master_4000_base import *
import eval.code_prompt_hardening as code_prompt_hardening
import eval.deepswe_dataset_override as deepswe_dataset_override
import eval.deepswe_optional_flagship as deepswe_optional_flagship
import eval.deepswe_phase4_eta_overlay as deepswe_phase4_eta_overlay
import eval.deepswe_rsi_counter_fix as deepswe_rsi_counter_fix
import eval.eta_progress_hardening as eta_progress_hardening
import eval.flagship_dataset_overrides as flagship_dataset_overrides
import eval.learn_progress_overlay as learn_progress_overlay
import eval.live_generation_stream as live_generation_stream
import eval.master_4000_runtime as master_runtime
import eval.phase4_pro_rsi as phase4_pro_rsi
import eval.real_benchmark_runtime as real_benchmark_runtime
import eval.real_choice_scoring as real_choice_scoring
import eval.real_dataset_fetch_fixes as real_dataset_fetch_fixes
import eval.real_phase4_context as real_phase4_context
import eval.real_prompt_overrides as real_prompt_overrides
import eval.real_split_labels as real_split_labels
import eval.scoring_hardening as scoring_hardening
import eval.stage_integrity_telemetry as stage_integrity_telemetry
from eval.checkpoint_hardening import install as install_checkpoint_hardening
from eval.code_prompt_hardening import install as install_code_prompt_hardening
from eval.conversation_teach_hardening import install as install_conversation_teach_hardening
from eval.dataset_hardening import install as install_dataset_hardening
from eval.historical_good_merge import install as install_historical_good_merge
from eval.live_generation_stream import install_baseline_stream, install_phase4_stream
from eval.reader_hardening import install as install_reader_hardening
from eval.rsi_generation_memory_hardening import install as install_rsi_generation_memory_hardening
from eval.rsi_legacy_training_hardening import install as install_rsi_legacy_training_hardening
from eval.rsi_miss_only_hardening import capture_before_historical_merge, enforce_after_historical_merge
from eval.rsi_prompt_hardening import install as install_rsi_prompt_hardening
from eval.rsi_resume_hardening import install as install_rsi_resume_hardening
from eval.rsi_telemetry_compact import install as install_rsi_telemetry_compact
from eval.scoring_hardening import install as install_scoring_hardening
from eval.stage2_future_hardening import install as install_stage2_future_hardening
from eval.stage_integrity_telemetry import install as install_stage_integrity_telemetry
from eval.stage_tps_hardening import install as install_stage_tps_hardening
from eval.swe_verifier_hardening import install as install_swe_verifier_hardening

install_swe_verifier_hardening(master_runtime)
install_checkpoint_hardening(master_runtime)
install_reader_hardening(scoring_hardening)
real_choice_scoring.install(scoring_hardening)

master_runtime.install(Master4000EvaluationEngine)
install_baseline_stream(master_runtime, Master4000EvaluationEngine)
phase4_pro_rsi._append_raw_generation_log = master_runtime._append_raw_generation_log
install_dataset_hardening(master_runtime, phase4_pro_rsi)
install_code_prompt_hardening(master_runtime, phase4_pro_rsi, Master4000EvaluationEngine)
install_rsi_prompt_hardening(phase4_pro_rsi)
phase4_pro_rsi.install(Master4000EvaluationEngine)

_legacy_rsi_branch_generate = phase4_pro_rsi._generate_branches_same_model
install_phase4_stream(phase4_pro_rsi)
install_rsi_generation_memory_hardening(
    phase4_pro_rsi,
    live_generation_stream,
    _legacy_rsi_branch_generate,
)

capture_before_historical_merge(phase4_pro_rsi)
install_historical_good_merge(phase4_pro_rsi)
enforce_after_historical_merge(phase4_pro_rsi)
install_rsi_resume_hardening(phase4_pro_rsi)
install_stage_integrity_telemetry(phase4_pro_rsi, Master4000EvaluationEngine)
install_stage2_future_hardening(phase4_pro_rsi, stage_integrity_telemetry)
install_rsi_legacy_training_hardening(phase4_pro_rsi)
install_conversation_teach_hardening(phase4_pro_rsi, Master4000EvaluationEngine)
install_rsi_telemetry_compact(stage_integrity_telemetry)
install_stage_tps_hardening(stage_integrity_telemetry, phase4_pro_rsi, Master4000EvaluationEngine)
install_scoring_hardening(Master4000EvaluationEngine, phase4_pro_rsi)

flagship_dataset_overrides.install(real_benchmark_runtime)
real_dataset_fetch_fixes.install(real_benchmark_runtime)
real_split_labels.install(real_benchmark_runtime)
real_prompt_overrides.install(real_benchmark_runtime)
real_benchmark_runtime.install(BenchmarkDatasetProvider, master_runtime, phase4_pro_rsi, Master4000EvaluationEngine)
deepswe_dataset_override.install(
    real_benchmark_runtime,
    master_runtime,
    phase4_pro_rsi,
    Master4000EvaluationEngine,
)
real_phase4_context.install(phase4_pro_rsi, Master4000EvaluationEngine)

_real_repair_suite = master_runtime._repair_suite
def _stable_real_repair_suite(splits):
    splits = _real_repair_suite(splits)
    for split_name in real_benchmark_runtime.REAL_SPLIT_COUNTS:
        for idx, item in enumerate(splits.get(split_name, [])):
            item["id"] = f"REAL-{split_name}-{idx:04d}"
    return splits
master_runtime._repair_suite = _stable_real_repair_suite
phase4_pro_rsi._repair_suite = _stable_real_repair_suite

# Optional true flagship DeepSWE v1.1. Default N means zero DeepSWE work/cost.
deepswe_optional_flagship.install(
    master_runtime,
    phase4_pro_rsi,
    code_prompt_hardening,
    Master4000EvaluationEngine,
)
# Count DeepSWE verified RSI traces in the already-existing consolidation budget.
deepswe_rsi_counter_fix.install(phase4_pro_rsi)
# Suite-wide smooth ETA / total progress, including optional DeepSWE when enabled.
eta_progress_hardening.install(master_runtime, phase4_pro_rsi, Master4000EvaluationEngine)
# While normal Phase-4 runs, include still-pending DeepSWE Pro retakes in total ETA.
deepswe_phase4_eta_overlay.install(master_runtime, eta_progress_hardening)
# Detailed Phase-2 MCTS progress only; MCTS behavior itself is unchanged.
learn_progress_overlay.install(Master4000EvaluationEngine)

if __name__ == "__main__":
    try:
        Master4000EvaluationEngine(max_duration_hours=72.0).run_full_suite()
    except KeyboardInterrupt:
        print("\n[*] Ctrl+C: clean shutdown.", flush=True)
        raise SystemExit(130)
