"""Task-specific prompt hardening backed by real-model smoke tests.

The exact Gemini-tested global SYSTEM_PROMPT is intentionally unchanged.
Only benchmark families that demonstrated a concrete failure mode get a small
user-side clarification. Families whose existing prompt behaved well keep it.

This layer is installed before phase4_pro_rsi.install(), so the same task policy
is used by Phase 1, Phase-4 retests, and RSI via _task_user_prompt().
"""
from __future__ import annotations

from typing import Any, Dict

from eval.scoring_hardening import strict_score


CODE_THINK = (
    "Think briefly and concretely inside <think>. Start with the solution, not a restatement. "
    "Use only the algorithm, invariants, and edge cases actually needed; no self-talk, requirement list, "
    "approach-shopping after one works, or repeated verification. Close </think> when implementation is clear. "
)

DEEPSWE_THINK = (
    "Diagnose immediately inside <think>. Use only a short 2-6 line repair sketch: failing behavior, file/change, "
    "important edge if any, and test implication. Do not restate the task/repository, self-narrate, or re-check the "
    "same conclusion. Close </think> when the patch is clear. "
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
        + "A standard algorithm may be named tersely with its key invariant/edge case. "
        "After </think>, output ONLY valid executable Python wrapped in ```python ... ```."
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
        f"{item['prompt']}\n\nUse the shortest valid calculation. Inside <think>, do not restate the problem, "
        "derive unused quantities, use an alternate method, or double-check after the result is known. "
        "Close </think> once determined, then output ONLY the final \\boxed{answer}."
    )


def _choice_user(item: Dict[str, Any]) -> str:
    return (
        f"{item['prompt']}\nUse only the stated premises. Inside <think>, make one terse deduction; do not discuss "
        "the user, benchmark/question type, outside context, or re-check the same conclusion. "
        "Close </think>, then output ONLY the option letter A, B, C, or D."
    )


def _hle_user(item: Dict[str, Any]) -> str:
    return (
        f"{item['prompt']}\nTreat the synthetic notation literally as defined here; do not import outside meanings. "
        "Use one terse substitution, no restatement or re-check. Close </think>, then output ONLY the exact Con(...) expression."
    )


def _bfcl_user(item: Dict[str, Any]) -> str:
    return (
        f"{item['prompt']}\nInside <think>, do not emit JSON; verify the requested tool name and arguments once in a terse line. "
        "Close </think>, then emit exactly ONE JSON object with keys `name` and `arguments`, and nothing else."
    )


def _dsl_user(item: Dict[str, Any]) -> str:
    # Keep the original Gemini-tested DSL wording, adding only the exact left-rotation formula
    # exposed as necessary by a random k=2 smoke failure.
    return (
        f"{item['prompt']}\n"
        "DSL Rules:\n"
        "- `arr >>~fold(k)`: Rotates list left by k positions, exactly `arr[k:] + arr[:k]`.\n"
        "- `arr <#>scale(s)`: Multiplies each element by scalar s.\n"
        "- `arr1 @fuse arr2`: Element-wise addition.\n"
        "Calculate on scratchpad and output the final numeric list [x, y, ...] directly."
    )


def _autoevol_user(item: Dict[str, Any]) -> str:
    return (
        f"{item['prompt']}\nApply the given relations directly once. Reduce the exponent modulo the stated order "
        "to the canonical nonnegative exponent. Do not restate, list equivalent forms, or re-check. "
        "Close </think>, then output ONLY the final symbolic power."
    )


def _dialogue_user(item: Dict[str, Any]) -> str:
    return (
        f"{item['prompt']}\nDo not discuss yourself, memory access, or the request. Inside <think>, make only a terse "
        "retrieval attempt. Close </think>, then output ONLY the exact recalled fact if known; otherwise output `unknown`."
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
            # Preserve proven-good original handlers (GSM8K/MATH/Zebra/etc.) exactly.
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
