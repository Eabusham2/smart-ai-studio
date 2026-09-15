"""Regression contracts for useful legacy RSI feedback merged into current RSI."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_round2_gets_only_answer_blind_deterministic_verifier_feedback():
    source = _src("eval/rsi_feedback_hardening.py")
    assert "Answer-blind deterministic verifier feedback from that attempt" in source
    assert "_answer_blind_failure_feedback" in source
    assert 'if not passed and round_idx == 1 and p4._has_answer_blind_verifier(split_name):' in source
    for split in ("HumanEval", "LiveCodeBench", "DeepSWE", "TensorGraphDSL", "BFCL"):
        assert split in source
    # Hidden benchmark expected fields must not enter feedback construction.
    helper = source.split("def _answer_blind_failure_feedback", 1)[1].split("def install", 1)[0]
    assert "expected_keyword" not in helper
    assert 'item.get("expected"' not in helper
    assert "item['expected']" not in helper


def test_current_rsi_search_width_rounds_and_hidden_reward_order_are_preserved():
    source = _src("eval/rsi_feedback_hardening.py")
    assert "for round_idx in (1, 2):" in source
    assert "temps = [0.20, 0.38, 0.58, 0.82]" in source
    selection = source.index("p4._choose_without_ground_truth")
    reward = source.index("p4._hidden_reward_only_after_selection")
    log = source.index("p4._append_rsi_log")
    assert selection < reward < log
    assert "max_tokens=min(p4._benchmark_ceiling(self), 16384)" in source


def test_same_model_is_rechecked_each_recursive_round_and_unused_branch_strings_are_released():
    source = _src("eval/rsi_feedback_hardening.py")
    assert 'p4._assert_same_model(self, model_identity, f"RSI round {round_idx}")' in source
    assert "branches.clear()" in source
    assert "del branches" in source
    assert "gc.collect()" in source


def test_bad_legacy_gold_answer_seeding_and_extra_rlvr_tasks_are_not_restored():
    source = _src("eval/rsi_feedback_hardening.py")
    assert "_gold_completion" not in source
    assert "AUTONOMOUS_RLVR_TASKS" not in source
    assert "HistoricalRLVR" not in source
