"""Contracts for real chat-style learning after Phase 3."""
from pathlib import Path

from core.online_consolidator import AwakeOnlineConsolidator


ROOT = Path(__file__).resolve().parents[2]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_awake_consolidator_converts_role_content_into_real_training_pairs():
    history = [
        {"role": "user", "content": "Remember: ALPHA maps to cedar."},
        {"role": "assistant", "content": "ALPHA maps to cedar."},
        {"role": "user", "content": "Remember: BETA maps to amber."},
        {"role": "assistant", "content": "BETA maps to amber."},
    ]
    assert AwakeOnlineConsolidator._conversation_training_pairs(history) == [
        {"prompt": "Remember: ALPHA maps to cedar.", "completion": "ALPHA maps to cedar."},
        {"prompt": "Remember: BETA maps to amber.", "completion": "BETA maps to amber."},
    ]


def test_phase3b_reuses_production_awake_trainer_on_same_model_and_requires_real_delta():
    source = _src("eval/conversation_teach_hardening.py")
    assert "AwakeOnlineConsolidator" in source
    assert "backend.model = self.engine.model" in source
    assert "consolidator._run_shadow_consolidation(_chat_history())" in source
    assert "p4._assert_same_model(self, model_identity" in source
    assert "delta <= 0.0" in source
    assert "p4.RSI_ADAPTER_PATH" in source
    assert "conversation_teach_state.json" in source


def test_final_stage_generates_recall_for_every_new_conversation_fact_without_scoring_4k():
    source = _src("eval/conversation_teach_hardening.py")
    assert "Final Conversation Recall" in source
    assert "self._fast_generate(formatted, max_tokens=256)" in source
    assert "p4._append_raw_generation_log" in source
    assert "CONVERSATION_TEACH_EXAMPLES" in source
    assert "same updated model" in source
    assert "4,014 benchmark score" in source


def test_launcher_installs_conversation_teach_after_transactional_phase3_wrapper():
    source = _src("master_4000_eval_suite.py")
    legacy = source.index("install_rsi_legacy_training_hardening(phase4_pro_rsi)")
    convo = source.index("install_conversation_teach_hardening(phase4_pro_rsi, Master4000EvaluationEngine)")
    scoring = source.index("install_scoring_hardening(Master4000EvaluationEngine, phase4_pro_rsi)")
    assert legacy < convo < scoring
