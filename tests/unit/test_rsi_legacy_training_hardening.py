"""Regression tests for restored pre-rewrite Phase-3 training strengths."""
from pathlib import Path

from eval.rsi_legacy_training_hardening import (
    ASSISTANT_MARKER,
    FISHER_ANCHOR_COUNT,
    TRAIN_WINDOW_TOKENS,
    _CompletionWindowTokenizer,
)


ROOT = Path(__file__).resolve().parents[2]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


class CharTokenizer:
    def encode(self, text, *args, **kwargs):
        return [ord(ch) for ch in str(text)]


def test_completion_window_keeps_answer_when_prompt_is_long_and_marks_prompt_loss_boundary():
    state = {}
    tok = _CompletionWindowTokenizer(CharTokenizer(), state, window=64)
    prompt = "P" * 200
    answer = "ANSWER-ONLY-TARGET"
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n{ASSISTANT_MARKER}{answer}<|im_end|>"

    ids = tok.encode(text)
    decoded = "".join(chr(x) for x in ids)

    assert len(ids) == 64
    assert answer in decoded
    assert state["completion_mask_active"] is True
    assert state["loss_start"] >= 0
    assert state["loss_start"] < len(ids) - 1


def test_short_example_keeps_full_sequence_but_masks_user_prompt_from_ce():
    state = {}
    tok = _CompletionWindowTokenizer(CharTokenizer(), state, window=TRAIN_WINDOW_TOKENS)
    text = f"<|im_start|>user\nQ<|im_end|>\n{ASSISTANT_MARKER}A<|im_end|>"
    ids = tok.encode(text)

    assert ids == CharTokenizer().encode(text)
    assert state["completion_mask_active"] is True
    assert state["loss_start"] > 0


def test_phase3_wrapper_is_installed_outside_fail_closed_integrity_stack():
    launcher = _src("master_4000_eval_suite.py")
    telemetry = launcher.index("install_stage_integrity_telemetry(phase4_pro_rsi, Master4000EvaluationEngine)")
    transactional = launcher.index("install_rsi_legacy_training_hardening(phase4_pro_rsi)")
    scoring = launcher.index("install_scoring_hardening(Master4000EvaluationEngine, phase4_pro_rsi)")
    assert telemetry < transactional < scoring


def test_training_wrapper_preserves_current_optimizer_math_and_restores_real_old_protections():
    hard = _src("eval/rsi_legacy_training_hardening.py")
    phase = _src("eval/phase4_pro_rsi.py")

    # Current real trainer remains the implementation being wrapped.
    assert "nn.value_and_grad" in phase
    assert "optim.AdamW" in phase
    assert "ogp_projector.project_gradient" in phase
    assert "opt.update(self.engine.model, grads)" in phase
    assert "rsi_post_phase3.safetensors" in phase

    # Real pre-rewrite strengths restored around it.
    assert "completion_only_cross_entropy" in hard
    assert "ASSISTANT_MARKER" in hard
    assert "_compute_real_fisher" in hard
    assert "compute_mlx_fisher" in hard
    assert "get_anchor_texts()[:FISHER_ANCHOR_COUNT]" in hard
    assert "ewc_value_and_grad" in hard
    assert "ewc_lambda" in hard
    assert FISHER_ANCHOR_COUNT == 4
    assert "_snapshot_trainables" in hard
    assert "_restore_trainables" in hard
    assert "_snapshot_moe_buffers" in hard
    assert "_restore_moe_buffers" in hard
    assert "UPDATE episodic_interactions SET consolidated=0" in hard
    assert ".pre_phase3.bak" in hard
    assert "except BaseException:" in hard


def test_old_fake_training_behaviors_are_not_reintroduced():
    hard = _src("eval/rsi_legacy_training_hardening.py")
    assert "random.normal" not in hard
    assert "param_drift = 0.0018" not in hard
    assert "default_code" not in hard
    assert "MockSlowLoRAModel" not in hard
