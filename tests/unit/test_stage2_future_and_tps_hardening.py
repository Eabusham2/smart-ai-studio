"""Contracts for recovered Stage-2 integrity and honest TPS telemetry."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_future_stage2_keeps_useful_old_dialogue_proof_and_bounded_mcts_teaching():
    source = _src("eval/stage2_future_hardening.py")
    assert "_EXPECTED_SESSIONS = len(HISTORICAL_DIALOGUE_SESSIONS)" in source
    assert "_EXPECTED_HISTORICAL_FACTS" in source
    assert "_semantic_recall_proof" in source
    assert "TensorGraphDSL-300" in source
    assert "[:30]" in source
    assert "predicate='evaluates_to'" in source
    assert "mcts_edges" in source
    assert "Stage 2 integrity" in source
    assert "recovered_without_replay" in source


def test_old_unbounded_stage2_rlvr_curriculum_is_not_restored():
    source = _src("eval/stage2_future_hardening.py")
    assert "execute_extended_self_play" not in source
    assert "target_verified_traces=350" not in source
    assert "branch_count=16" not in source
    assert "old unbounded N=16/K>=350 RLVR curriculum is intentionally not restored" in source


def test_stage2_fresh_run_measures_work_tps_but_resume_does_not_fabricate_it():
    source = _src("eval/stage2_future_hardening.py")
    assert "_STAGE2_STARTED = time.perf_counter()" in source
    assert "tps = work_tokens / elapsed" in source
    assert "else:\n            tps = 0.0" in source
    assert '"recovered_without_replay": not fresh_run' in source


def test_stage_wide_tps_uses_measured_decode_or_real_token_work_not_fixed_constant():
    source = _src("eval/stage_tps_hardening.py")
    assert "last_tok_per_sec" in source
    assert "_learn_tokens" in source
    assert "_phase3_tokens" in source
    assert "time.perf_counter()" in source
    assert 'fields["tps"]' in source
    assert "12.0" not in source
    assert "5.0" not in source


def test_rsi_streamed_branches_publish_real_generation_tps_to_stage_owner():
    source = _src("eval/rsi_generation_memory_hardening.py")
    assert "_publish_measured_tps" in source
    assert 'getattr(response, "generation_tps", 0.0)' in source
    assert 'getattr(response, "generation_tokens", 0)' in source
    assert "len(self.engine.tokenizer.encode(str(text)))" in source
    assert "self.last_tok_per_sec = float(tps)" in source
    assert "time.perf_counter() - started" in source


def test_phase3_training_tps_counts_the_actual_256_token_training_window():
    source = _src("eval/stage_tps_hardening.py")
    assert "min(_encode_len(self.engine.tokenizer, text), 256)" in source
    assert '"Phase 3 Consolidation"' in source
    assert "base_phase3(self)" in source


def test_compact_rsi_console_includes_tps_and_launcher_installs_tps_last():
    compact = _src("eval/rsi_telemetry_compact.py")
    launcher = _src("master_4000_eval_suite.py")
    assert "TPS: {tps}" in compact
    assert 'return "calculating"' in compact
    assert 'return f"{tps:.1f}t/s"' in compact
    assert "install_stage2_future_hardening" in launcher
    compact_i = launcher.index("install_rsi_telemetry_compact(stage_integrity_telemetry)")
    tps_i = launcher.index("install_stage_tps_hardening(stage_integrity_telemetry, phase4_pro_rsi, Master4000EvaluationEngine)")
    scoring_i = launcher.index("install_scoring_hardening(Master4000EvaluationEngine, phase4_pro_rsi)")
    assert compact_i < tps_i < scoring_i


def test_phase3b_and_final_conversation_recall_report_tps():
    source = _src("eval/conversation_teach_hardening.py")
    assert "teach_tps = train_tokens / teach_seconds" in source
    assert "TPS={teach_tps:.1f}t/s" in source
    assert "total_tokens / elapsed" in source
    assert "TPS={tps:.1f}t/s" in source
