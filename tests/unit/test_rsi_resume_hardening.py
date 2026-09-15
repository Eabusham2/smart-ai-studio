"""Regression contracts for Ctrl+C-safe RSI resume and concise console telemetry."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_launcher_installs_resume_before_stage_telemetry_and_compact_after():
    source = _src("master_4000_eval_suite.py")
    resume = source.index("install_rsi_resume_hardening(phase4_pro_rsi)")
    stage = source.index("install_stage_integrity_telemetry(phase4_pro_rsi, Master4000EvaluationEngine)")
    compact = source.index("install_rsi_telemetry_compact(stage_integrity_telemetry)")
    assert resume < stage < compact


def test_rsi_resume_is_item_level_miss_only_and_recovers_verified_trace():
    source = _src("eval/rsi_resume_hardening.py")
    assert 'RSI_PROGRESS_PATH = Path("eval_results/rsi_progress.json")' in source
    assert 'if "DialogueRecall" in split_name' in source
    assert 'cache.get(f"Phase 1: Baseline_{item[\'id\']}") is False' in source
    assert "return out[:64]" in source
    assert "_interrupted_telemetry" in source
    assert "_raw_verified_candidates" in source
    assert "self.engine.kg.log_interaction" in source
    assert 'filtered_cache[f"Phase 1: Baseline_{item[\'id\']}"] = "RSI_RESUMED_DONE"' in source
    assert "bool(passed) or int(round_idx) >= 2" in source
    assert "_atomic_save(progress)" in source


def test_partial_item_is_not_checkpointed_and_training_trace_is_required_for_success_skip():
    source = _src("eval/rsi_resume_hardening.py")
    assert "if not (bool(passed) or int(round_idx) >= 2):" in source
    assert "Cannot prove/rebuild the training trace: rerun honestly." in source
    assert "completed.pop(key, None)" in source


def test_compact_console_keeps_full_jsonl_and_stage1_style_fields():
    source = _src("eval/rsi_telemetry_compact.py")
    assert 'f.write(json.dumps(payload' in source
    assert 'f"[RSI] {current:<28} | Item {done}/{total}' in source
    assert "Verified: {verified}" in source
    assert "ETA: {eta}" in source
    assert "RAM: {ram_gb:.1f}GB" in source
    assert "[Telemetry] RSI" not in source
