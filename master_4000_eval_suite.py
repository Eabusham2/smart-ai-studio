"""Canonical entry point for the merged 4,014-item evaluation suite."""
from eval._master_4000_base import *
import eval.live_generation_stream as live_generation_stream
import eval.master_4000_runtime as master_runtime
import eval.phase4_pro_rsi as phase4_pro_rsi
import eval.real_benchmark_runtime as real_benchmark_runtime
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

# Keep the old direct branch generator as a compatibility path. The live watcher
# wraps it; then the memory layer restores old pre/post-branch cleanup while adding
# current MLX-LM KV quantization and OOM recovery. Search/reward semantics are untouched.
_legacy_rsi_branch_generate = phase4_pro_rsi._generate_branches_same_model
install_phase4_stream(phase4_pro_rsi)
install_rsi_generation_memory_hardening(
    phase4_pro_rsi,
    live_generation_stream,
    _legacy_rsi_branch_generate,
)

# Capture the true Phase-1-miss-only RSI before historical recovery installs its
# optional extra autonomous RLVR tasks. Historical Learn/retention/parameter-delta
# behavior is still restored, then RSI is put back to Phase-1 misses only.
capture_before_historical_merge(phase4_pro_rsi)
install_historical_good_merge(phase4_pro_rsi)
enforce_after_historical_merge(phase4_pro_rsi)

# Persist completed RSI items across Ctrl+C/restarts. This wraps the final
# Phase-1-miss-only RSI function, so passed Phase-1 questions and DialogueRecall
# remain excluded exactly as before.
install_rsi_resume_hardening(phase4_pro_rsi)

# Add stage telemetry and fail-closed proof that every supplied LearningFact is
# queued, trained, consolidated, moves real trainable weights, and is persisted.
install_stage_integrity_telemetry(phase4_pro_rsi, Master4000EvaluationEngine)
# Recover the useful old Stage-2 checks for future runs: 5-session/10-fact semantic
# history must be queryable and the bounded 30-item DSL MCTS teaching pass must
# actually store its teaching edges. Resume can prove this from persisted state/DB.
install_stage2_future_hardening(phase4_pro_rsi, stage_integrity_telemetry)
# Restore the useful pre-rewrite training semantics around that final fail-closed
# Phase-3 stack: completion-only CE plus transaction rollback on any failure.
install_rsi_legacy_training_hardening(phase4_pro_rsi)
# After normal Phase 3, teach a tiny independent fact set through the production
# awake-conversation MLX update path and test those facts again after Phase 4.
install_conversation_teach_hardening(phase4_pro_rsi, Master4000EvaluationEngine)
# Keep the full structured JSONL telemetry, but render RSI heartbeat/progress in a
# short Stage-1-style single line on the console.
install_rsi_telemetry_compact(stage_integrity_telemetry)
# Add measured TPS to every stage telemetry path. Generation stages use decode TPS;
# Learn/Phase-3 use actual token-work throughput instead of a fabricated constant.
install_stage_tps_hardening(stage_integrity_telemetry, phase4_pro_rsi, Master4000EvaluationEngine)
install_scoring_hardening(Master4000EvaluationEngine, phase4_pro_rsi)
# Final narrow adapter: real published benchmark data, schema-correct prompts and
# verifiers, fresh REAL-* cache IDs, and a 32K benchmark ceiling. This intentionally
# runs last so it replaces only recovered synthetic benchmark behavior.
real_benchmark_runtime.install(BenchmarkDatasetProvider, master_runtime, phase4_pro_rsi, Master4000EvaluationEngine)

if __name__ == "__main__":
    try:
        Master4000EvaluationEngine(max_duration_hours=72.0).run_full_suite()
    except KeyboardInterrupt:
        # Inner evaluation loops save the active checkpoint before re-raising.
        # RSI completed-item progress is persisted separately by rsi_resume_hardening.
        # If Ctrl+C occurs during model loading, there is no new checkpoint state
        # to save. Either way, exit cleanly without an alarming traceback.
        print("\n[*] Ctrl+C: clean shutdown.", flush=True)
        raise SystemExit(130)
