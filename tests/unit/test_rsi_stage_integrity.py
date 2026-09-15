"""Regression contracts for miss-only RSI and stage-wide training telemetry."""
from pathlib import Path

import eval.rsi_miss_only_hardening as miss_only


ROOT = Path(__file__).resolve().parents[2]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_historical_extra_rsi_is_not_used_by_master_pipeline():
    class P4:
        pass

    def miss_only_base(*args, **kwargs):
        return 3

    p4 = P4()
    p4._run_rsi_self_improvement = miss_only_base
    miss_only.capture_before_historical_merge(p4)

    def historical_extra(*args, **kwargs):
        return 10

    p4._run_rsi_self_improvement = historical_extra
    miss_only.enforce_after_historical_merge(p4)
    assert p4._run_rsi_self_improvement is miss_only_base
    assert p4._rsi_phase1_miss_only_enforced is True


def test_launcher_captures_before_historical_and_enforces_after():
    source = _src("master_4000_eval_suite.py")
    capture = source.index("capture_before_historical_merge(phase4_pro_rsi)")
    historical = source.index("install_historical_good_merge(phase4_pro_rsi)")
    enforce = source.index("enforce_after_historical_merge(phase4_pro_rsi)")
    telemetry = source.index("install_stage_integrity_telemetry(phase4_pro_rsi, Master4000EvaluationEngine)")
    assert capture < historical < enforce < telemetry


def test_stage_telemetry_covers_all_pipeline_stages_and_fail_closed_training():
    source = _src("eval/stage_integrity_telemetry.py")
    for label in (
        "Phase 2 Learn",
        "RSI",
        "Phase 3 Consolidation",
        "Learning Retention",
    ):
        assert label in source
    assert "evaluate_all_with_stage_telemetry" in source  # Phase 1 + Phase 4
    assert "Learn seed integrity failure" in source
    assert "DialogueRecall trace entered RSI" in source
    assert "non-miss RSI trace" in source
    assert "queued {len(learn_rows)}/{target_learn} supplied LearningFacts" in source
    assert "Phase 3 parameter delta is zero" in source
    assert "RSI adapter was not persisted" in source
    assert "left {len(not_marked)} trained memories unconsolidated" in source


def test_rsi_progress_total_is_dynamic_not_hardcoded_to_current_21_misses():
    source = _src("eval/stage_integrity_telemetry.py")
    assert "cap = min(64, len(eligible))" in source
    assert '"total": cap' in source
    assert "total=cap" in source
    assert "done=0" in source
    assert "percent=0.0" in source
    assert "21" not in source


def test_core_rsi_still_selects_only_phase1_false_and_cap_64():
    phase = _src("eval/phase4_pro_rsi.py")
    assert 'key = f"Phase 1: Baseline_{item[\'id\']}"' in phase
    assert "if cache.get(key) is False:" in phase
    assert "for split_name, item in misses[:64]:" in phase


def test_dialogue_recall_is_still_deferred_from_rsi():
    hard = _src("eval/code_prompt_hardening.py")
    assert 'rsi_cache[key] = "DEFERRED_PRE_LEARN_MEMORY"' in hard
    assert "return original_rsi(self, splits, rsi_cache)" in hard


def test_real_weight_update_path_remains_intact():
    phase = _src("eval/phase4_pro_rsi.py")
    hist = _src("eval/historical_good_merge.py")
    assert "nn.value_and_grad" in phase
    assert "opt.update(self.engine.model" in phase
    assert "rsi_post_phase3.safetensors" in phase
    assert "real_trainable_delta_l2" in hist
    assert "Phase 3 claimed parameter updates but real trainable-weight delta is zero" in hist
