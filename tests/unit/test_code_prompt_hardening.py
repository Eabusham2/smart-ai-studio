"""Regression tests for the A/B-proven code-task low-thinking prompt layer."""

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


def test_code_prompt_hardening_is_installed_before_phase4_capture():
    assert getattr(entry.Master4000EvaluationEngine, "_code_prompt_hardening_installed", False)


def test_lcb_gets_tight_code_reasoning_instruction():
    item = {
        "prompt": "Write function f(arr).",
        "entry_point": "f",
    }
    prompt = phase4._task_user_prompt("LiveCodeBench-Hard", item)
    assert "Inside <think>, do not restate the task" in prompt
    assert "at most one terse implementation/repair note" in prompt
    assert "merge-sort inversions; O(n log n)" in prompt
    assert "After </think>, output ONLY valid executable Python" in prompt
    assert "Use scratchpad only for logic outline" not in prompt


def test_humaneval_gets_tight_code_reasoning_instruction():
    item = {"prompt": "def f(x):\n    pass"}
    prompt = phase4._task_user_prompt("HumanEval-164", item)
    assert "Inside <think>, do not restate the task" in prompt
    assert "After </think>, output ONLY the valid executable Python" in prompt
    assert "Use scratchpad only for logic outline" not in prompt


def test_deepswe_gets_tight_patch_reasoning_instruction():
    item = {
        "repo_files": {"a.py": "def f():\n    return 0\n"},
        "test_cmd": "pytest -q",
    }
    prompt = phase4._task_user_prompt("DeepSWE-50", item)
    assert "Inside <think>, do not restate the task" in prompt
    assert "After </think>, output ONLY the unified diff patch" in prompt
    assert "Test command: pytest -q" in prompt


def test_non_code_handlers_keep_existing_prompt_policy():
    item = {"prompt": "Compute 2+2", "expected": "4"}
    prompt = phase4._task_user_prompt("MATH-500", item)
    assert "Solve this problem using a minimal scratchpad" in prompt
    assert "Inside <think>, do not restate the task" not in prompt
