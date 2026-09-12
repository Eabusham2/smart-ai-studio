"""Minimal task-specific prompt hardening backed by real-model smoke tests.

The exact Gemini-tested global SYSTEM_PROMPT stays unchanged. Only benchmark
families that demonstrated a concrete failure get a short user-side clarification.
Families that already behaved well keep their original task prompt.

Installed before phase4_pro_rsi.install(), so the same policy is used by Phase 1,
RSI task generation, and the final Phase-4 miss-only retest.
"""
from __future__ import annotations

from typing import Any, Dict

from eval.scoring_hardening import strict_score


CODE_THINK = (
    "Inside <think>, start solving immediately. Do not restate the task or narrate. "
    "Keep only needed algorithm/invariant/edge-case notes; no repeated checking. Close </think> when ready. "
)

DEEPSWE_THINK = (
    "Inside <think>, diagnose immediately: failure -> file/change -> important edge/test. "
    "Keep it brief; no task/repository restatement or repeated checking. Close </think> when the patch is clear. "
)


def _human_eval_user(item: Dict[str, Any]) -> str:
    return (
        f"{item['prompt']}\n\nComplete the Python function above. "
        + CODE_THINK
        + "After </think>, output ONLY the valid executable Python code wrapped in ```python ... ```."
    )


def _lcb_user(item: Dict[str, Any]) -> str:
    return (
        f"{item['prompt']}\n\nWrite the complete Python solution requested above. "
        + CODE_THINK
        + "After </think>, output ONLY valid executable Python wrapped in ```python ... ```."
    )


def _deep_swe_user(item: Dict[str, Any]) -> str:
    repo_text = "\n\n".join(
        f"### {path}\n```\n{body}\n```" for path, body in item["repo_files"].items()
    )
    return (
        "Repair the repository so the test command passes. "
        + DEEPSWE_THINK
        + "After </think>, output ONLY the unified diff patch.\n\n"
        f"Repository files:\n{repo_text}\n\nTest command: {item['test_cmd']}"
    )


def _aime_user(item: Dict[str, Any]) -> str:
    return (
        f"{item['prompt']}\n\nUse the shortest valid calculation. No restatement, alternate method, or double-check. "
        "Close </think> once determined; output ONLY \\boxed{answer}."
    )


def _choice_user(item: Dict[str, Any]) -> str:
    return (
        f"{item['prompt']}\nUse only the stated premises. Make one terse deduction; no restatement, meta-commentary, "
        "outside context, or re-check. Close </think>; output ONLY A, B, C, or D."
    )


def _hle_user(item: Dict[str, Any]) -> str:
    return (
        f"{item['prompt']}\nTreat the synthetic notation literally as defined here. One terse substitution only; "
        "no outside hierarchy discussion or re-check. Close </think>; output ONLY the exact Con(...) expression."
    )


def _bfcl_user(item: Dict[str, Any]) -> str:
    return (
        f"{item['prompt']}\nDo not emit JSON inside <think>; check tool name/arguments once. "
        "Close </think>; emit exactly ONE JSON object with keys `name` and `arguments`."
    )


def _dsl_user(item: Dict[str, Any]) -> str:
    # Preserve the original Gemini-tested DSL prompt; add only the exact fold rule
    # exposed by the random k=2 smoke failure.
    return (
        f"{item['prompt']}\n"
        "DSL Rules:\n"
        "- `arr >>~fold(k)`: Rotates list left by k positions: `arr[k:] + arr[:k]`.\n"
        "- `arr <#>scale(s)`: Multiplies each element by scalar s.\n"
        "- `arr1 @fuse arr2`: Element-wise addition.\n"
        "Calculate on scratchpad and output the final numeric list [x, y, ...] directly."
    )


def _autoevol_user(item: Dict[str, Any]) -> str:
    return (
        f"{item['prompt']}\nApply the given relations once and reduce the exponent modulo the stated order. "
        "No restatement, equivalent-form list, or re-check. Close </think>; output ONLY the final symbolic power."
    )


def _dialogue_user(item: Dict[str, Any]) -> str:
    return (
        f"{item['prompt']}\nAttempt recall tersely; do not discuss yourself or memory access. "
        "Close </think>; output ONLY the recalled fact if known, otherwise `unknown`."
    )


def _task_user(split: str, item: Dict[str, Any], original) -> str:
    if "HumanEval" in split:
        return _human_eval_user(item)
    if "LiveCodeBench" in split:
        return _lcb_user(item)
    if "DeepSWE" in split:
        return _deep_swe_user(item)
    if "AIME" in split:
        return _aime_user(item)
    if "GPQA" in split or "MMLU-Pro" in split:
        return _choice_user(item)
    if "HLE" in split:
        return _hle_user(item)
    if "BFCL" in split:
        return _bfcl_user(item)
    if "TensorGraphDSL" in split:
        return _dsl_user(item)
    if "AutonomousEvolution" in split:
        return _autoevol_user(item)
    if "DialogueRecall" in split:
        return _dialogue_user(item)
    return original(split, item)


def install(runtime_module, phase4_module, cls) -> None:
    """Install concise task prompts without replacing the verified system prompt."""
    if getattr(cls, "_code_prompt_hardening_installed", False):
        return

    original_eval = cls._evaluate_single_item
    original_task_user_prompt = phase4_module._task_user_prompt
    targeted = (
        "HumanEval", "LiveCodeBench", "DeepSWE", "AIME", "GPQA", "MMLU-Pro", "HLE",
        "BFCL", "TensorGraphDSL", "AutonomousEvolution", "DialogueRecall",
    )

    def hardened_eval(self, split, item):
        if not any(name in split for name in targeted):
            return original_eval(self, split, item)

        tok = self.engine.tokenizer
        user_message = _task_user(split, item, original_task_user_prompt)
        prompt = runtime_module._chat(tok, user_message, system=runtime_module.SYSTEM_PROMPT)
        ceiling = getattr(self, "benchmark_max_tokens", None) or runtime_module._benchmark_ceiling(self)
        out = self._fast_generate(prompt, max_tokens=ceiling)
        self.last_raw_out = out
        runtime_module._append_raw_generation_log(self, prompt, user_message, out)

        if "HumanEval" in split:
            code = runtime_module.clean_output(out)
            return bool(self.engine.sandbox.execute_python_code(item["prompt"] + "\n" + code, item["test"]).passed)
        if "LiveCodeBench" in split:
            code = runtime_module.clean_output(out)
            return bool(self.engine.sandbox.execute_python_code(code, item["test"]).passed)
        if "DeepSWE" in split:
            patch = runtime_module.clean_output(out)
            return bool(self.engine.sandbox.verify_git_diff_patch(item["repo_files"], patch, item["test_cmd"]).passed)
        if "BFCL" in split:
            value = runtime_module._parse_json_object(out)
            if not isinstance(value, dict):
                return False
            if isinstance(value.get("function"), dict):
                value = value["function"]
            name = value.get("name") or value.get("tool")
            args = value.get("arguments") or value.get("args")
            if isinstance(args, str):
                try:
                    import json
                    args = json.loads(args)
                except Exception:
                    return False
            return name == item.get("expected_tool") and args == item.get("expected_args")

        strict = strict_score(self, split, item, out)
        return bool(strict) if strict is not None else False

    def hardened_task_user_prompt(split: str, item: Dict[str, Any]) -> str:
        return _task_user(split, item, original_task_user_prompt)

    cls._evaluate_single_item = hardened_eval
    phase4_module._task_user_prompt = hardened_task_user_prompt
    cls._code_prompt_hardening_installed = True
