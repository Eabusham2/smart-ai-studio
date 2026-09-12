"""Regression tests for task-specific low-yap prompt routing."""

import eval.master_4000_runtime as runtime
import eval.phase4_pro_rsi as phase4
import master_4000_eval_suite as entry


VERIFIED_GEMINI_PROMPT = (
    "You are a fast symbolic computing engine. "
    "Keep your internal scratchpad (<think>) strictly minimal: write only concise intermediate formulas or numbers. "
    "No conversational monologue, no self-reflection, and no verification loops. "
    "Close </think> immediately once calculated and output the answer."
)


def test_global_system_prompt_remains_exact_verified_gemini_prompt():
    assert runtime.SYSTEM_PROMPT == VERIFIED_GEMINI_PROMPT
    assert phase4.SYSTEM_PROMPT == VERIFIED_GEMINI_PROMPT
    assert "unknown" not in runtime.SYSTEM_PROMPT.lower()


def test_prompt_hardening_is_installed_before_phase4_capture():
    assert getattr(entry.Master4000EvaluationEngine, "_code_prompt_hardening_installed", False)


def test_code_families_get_brief_but_real_reasoning():
    human = phase4._task_user_prompt("HumanEval-164", {"prompt": "def f(x):\n    pass"})
    lcb = phase4._task_user_prompt("LiveCodeBench-Hard", {"prompt": "Write f(arr)."})
    swe = phase4._task_user_prompt(
        "DeepSWE-50",
        {"repo_files": {"a.py": "x=1\n"}, "test_cmd": "pytest -q"},
    )
    for prompt in (human, lcb):
        assert "Start with the solution, not a restatement" in prompt
        assert "no self-talk" in prompt
        assert "repeated verification" in prompt
        assert "Close </think> when implementation is clear" in prompt
    assert "short 2-6 line repair sketch" in swe
    assert "output ONLY the unified diff patch" in swe


def test_problematic_non_code_families_get_small_targeted_clarifications():
    aime = phase4._task_user_prompt("AIME-150", {"prompt": "AIME prompt"})
    gpqa = phase4._task_user_prompt("GPQA-400", {"prompt": "MC prompt"})
    mmlu = phase4._task_user_prompt("MMLU-Pro-1000", {"prompt": "MC prompt"})
    hle = phase4._task_user_prompt("HLE-100", {"prompt": "HLE prompt"})
    bfcl = phase4._task_user_prompt("BFCL-200", {"prompt": "tool prompt"})
    dsl = phase4._task_user_prompt("TensorGraphDSL-300", {"prompt": "dsl"})
    auto = phase4._task_user_prompt("AutonomousEvolution-200", {"prompt": "group"})
    dialogue = phase4._task_user_prompt("DialogueRecall-150", {"prompt": "recall"})

    assert "shortest valid calculation" in aime
    assert "output ONLY the option letter" in gpqa
    assert "output ONLY the option letter" in mmlu
    assert "synthetic notation literally" in hle
    assert "do not emit JSON" in bfcl
    assert "arr[k:] + arr[:k]" in dsl
    assert "canonical nonnegative exponent" in auto
    assert "Do not discuss yourself, memory access" in dialogue


def test_already_good_prompt_families_keep_original_policy():
    gsm = phase4._task_user_prompt("GSM8K-500", {"prompt": "Compute 2+2"})
    math = phase4._task_user_prompt("MATH-500", {"prompt": "Compute 2+2"})
    zebra = phase4._task_user_prompt("ZebraLogic-200", {"prompt": "logic"})

    assert "Solve this problem using a minimal scratchpad" in gsm
    assert "Solve this problem using a minimal scratchpad" in math
    assert "Deduce the solution directly. State the final answer on the last line." in zebra
    assert "Start with the solution, not a restatement" not in gsm
    assert "Start with the solution, not a restatement" not in math
