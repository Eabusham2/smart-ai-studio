"""Canonical entry point for the merged 4,014-item evaluation suite."""
from eval._master_4000_base import *
import eval.master_4000_runtime as master_runtime
import eval.phase4_pro_rsi as phase4_pro_rsi
import eval.scoring_hardening as scoring_hardening
from eval.checkpoint_hardening import install as install_checkpoint_hardening
from eval.code_prompt_hardening import install as install_code_prompt_hardening
from eval.dataset_hardening import install as install_dataset_hardening
from eval.historical_good_merge import install as install_historical_good_merge
from eval.live_generation_stream import install_baseline_stream, install_phase4_stream
from eval.reader_hardening import install as install_reader_hardening
from eval.rsi_miss_only_hardening import capture_before_historical_merge, enforce_after_historical_merge
from eval.rsi_prompt_hardening import install as install_rsi_prompt_hardening
from eval.scoring_hardening import install as install_scoring_hardening
from eval.stage_integrity_telemetry import install as install_stage_integrity_telemetry
from eval.swe_verifier_hardening import install as install_swe_verifier_hardening

# Fix active SWE patch application before any engine exists. Actual tests remain
# under the configured sandbox timeout/resource limits.
install_swe_verifier_hardening(master_runtime)

# A legacy done marker without a surviving boolean result is not enough to score
# or skip an item. Marker-only entries are rerun honestly.
install_checkpoint_hardening(master_runtime)

# Narrow reader fix: accept an explicit final `(D) Description` choice without
# restoring the old unsafe substring grading.
install_reader_hardening(scoring_hardening)

master_runtime.install(Master4000EvaluationEngine)
# Install the live tap before Phase-4 wraps _fast_generate so baseline/Learn/RSI
# keep the recovered fused decoder while exposing raw <think> tokens in real time.
install_baseline_stream(master_runtime, Master4000EvaluationEngine)
# phase4_pro_rsi imported this function by value during module import. Point its
# local reference at the wrapped logger too so RSI/LearningFacts cannot bypass the
# live current-item file.
phase4_pro_rsi._append_raw_generation_log = master_runtime._append_raw_generation_log
install_dataset_hardening(master_runtime, phase4_pro_rsi)
# Keep the exact Gemini-tested global system prompt for normal benchmark generation.
# Only task families that showed a concrete smoke-test failure get small clarifications.
install_code_prompt_hardening(master_runtime, phase4_pro_rsi, Master4000EvaluationEngine)
# RSI has no system role. Any family-specific guidance selected for the split is
# preserved by moving it into the RSI user message instead.
install_rsi_prompt_hardening(phase4_pro_rsi)
phase4_pro_rsi.install(Master4000EvaluationEngine)
# Phase-4/RSI multi-branch generation uses mlx_lm.stream_generate when available.
install_phase4_stream(phase4_pro_rsi)

# Capture the true Phase-1-miss-only RSI before historical recovery installs its
# optional extra autonomous RLVR tasks. Historical Learn/retention/parameter-delta
# behavior is still restored, then RSI is put back to Phase-1 misses only.
capture_before_historical_merge(phase4_pro_rsi)
install_historical_good_merge(phase4_pro_rsi)
enforce_after_historical_merge(phase4_pro_rsi)

# Add stage telemetry and fail-closed proof that every supplied LearningFact is
# queued, trained, consolidated, moves real trainable weights, and is persisted.
install_stage_integrity_telemetry(phase4_pro_rsi, Master4000EvaluationEngine)
install_scoring_hardening(Master4000EvaluationEngine, phase4_pro_rsi)

if __name__ == "__main__":
    try:
        Master4000EvaluationEngine(max_duration_hours=72.0).run_full_suite()
    except KeyboardInterrupt:
        # Inner evaluation loops save the active checkpoint before re-raising.
        # If Ctrl+C occurs during model loading, there is no new checkpoint state
        # to save. Either way, exit cleanly without an alarming traceback.
        print("\n[*] Ctrl+C: clean shutdown.", flush=True)
        raise SystemExit(130)
