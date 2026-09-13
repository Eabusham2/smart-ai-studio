import inspect

import master_4000_eval_suite as master
from eval import code_prompt_hardening
from eval import historical_good_merge
from eval import master_4000_runtime as runtime
from eval import phase4_pro_rsi
from eval import scoring_hardening as scoring


class _DSLVerifier:
    @staticmethod
    def evaluate_dsl_expression(expr):
        if expr == "dsl":
            return [4, 8, 0]
        raise AssertionError(expr)


class _Engine:
    sandbox = _DSLVerifier()


class _Runner:
    engine = _Engine()


def test_math_never_passes_by_substring():
    runner = _Runner()
    item = {"expected": "15"}
    assert scoring.strict_score(runner, "MATH-500", item, r"\boxed{15}") is True
    assert scoring.strict_score(runner, "MATH-500", item, r"\boxed{115}") is False
    assert scoring.strict_score(runner, "MATH-500", item, "Final answer: 15") is True
    assert scoring.strict_score(runner, "MATH-500", item, "Final answer: 115") is False


def test_aime_accepts_numeric_equivalence_without_substring_leak():
    runner = _Runner()
    item = {"expected": "003"}
    assert scoring.strict_score(runner, "AIME-150", item, r"\boxed{003}") is True
    assert scoring.strict_score(runner, "AIME-150", item, r"\boxed{3}") is True
    assert scoring.strict_score(runner, "AIME-150", item, r"\boxed{103}") is False


def test_multiple_choice_requires_explicit_final_choice():
    runner = _Runner()
    item = {"expected": "A"}
    assert scoring.strict_score(runner, "GPQA-400", item, "Reasoning mentions A and B.\nAnswer: A") is True
    assert scoring.strict_score(runner, "GPQA-400", item, "A is tempting, but B is correct.\nAnswer: B") is False
    assert scoring.strict_score(runner, "MMLU-Pro-1000", item, "A appears somewhere in this prose") is False


def test_zebra_numeric_answer_is_exact():
    runner = _Runner()
    item = {"expected": "1"}
    assert scoring.strict_score(runner, "ZebraLogic-200", item, "Final answer: 1") is True
    assert scoring.strict_score(runner, "ZebraLogic-200", item, "Final answer: 11") is False


def test_dsl_scores_final_list_not_intermediate_list():
    runner = _Runner()
    item = {"dsl_expr": "dsl"}
    assert scoring.strict_score(runner, "TensorGraphDSL-300", item, "[1, 2, 3]\n[4, 8, 0]") is True
    assert scoring.strict_score(runner, "TensorGraphDSL-300", item, "[4, 8, 0]\n[1, 2, 3]") is False


def test_gemini_base_plus_global_anti_loop_rule_is_shared_across_every_model_stage():
    gemini = (
        "You are a fast symbolic computing engine. "
        "Keep your internal scratchpad (<think>) strictly minimal: write only concise intermediate formulas or numbers. "
        "No conversational monologue, no self-reflection, and no verification loops. "
        "Close </think> immediately once calculated and output the answer."
    )
    expected = gemini + code_prompt_hardening.GLOBAL_SYSTEM_SUFFIX
    assert runtime.SYSTEM_PROMPT == expected
    assert phase4_pro_rsi.SYSTEM_PROMPT == expected
    assert runtime.SYSTEM_PROMPT.startswith(gemini)
    assert runtime.SYSTEM_PROMPT.count(code_prompt_hardening.GLOBAL_SYSTEM_SUFFIX.strip()) == 1

    runtime_src = inspect.getsource(runtime.evaluate_one)
    assert "prompt = _chat(tok, user_message)" in runtime_src
    for marker in (
        "HumanEval",
        "LiveCodeBench",
        "DeepSWE",
        "RSI",
        "DialogueRecall",
        "LearningFacts",
        "GSM8K",
        "MATH",
        "AIME",
        "TensorGraphDSL",
        "ZebraLogic",
        "HLE",
        "BFCL",
    ):
        assert marker in runtime_src

    phase_src = inspect.getsource(phase4_pro_rsi)
    history_src = inspect.getsource(historical_good_merge)
    assert "system=SYSTEM_PROMPT" in phase_src
    assert "system=p4.SYSTEM_PROMPT" in history_src


def test_strict_scoring_is_installed_after_pro_layer():
    cls = master.Master4000EvaluationEngine
    assert getattr(cls, "_strict_scoring_installed", False) is True
    assert cls._evaluate_single_item.__module__ == scoring.__name__
    assert phase4_pro_rsi._hidden_reward_only_after_selection.__module__ == scoring.__name__
