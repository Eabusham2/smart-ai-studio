"""Minimal task-specific prompt hardening backed by real-model smoke tests.

The verified Gemini SYSTEM_PROMPT remains the base everywhere. Families that
already behave well keep the base prompt and original task wording. A few stubborn
families get only a tiny task-specific system suffix plus a short user instruction.

Installed before phase4_pro_rsi.install(), so baseline and final Phase-4 retests
use the same policy. RSI is routed through the same per-split system/user policy
without leaking a task suffix into LearningFacts or historical RLVR.
"""
from __future__ import annotations

from typing import Any, Dict

from eval.scoring_hardening import strict_score


DIRECT_SUFFIX = (
    " Begin with the solution itself. Never describe the user, request, or question; "
    "do not narrate what you will do, repeat a settled answer, or double-check a settled result."
)

SYSTEM_SUFFIXES = {
    "LiveCodeBench": DIRECT_SUFFIX + " In <think>, use only terse implementation notes; put final code after </think>.",
    "AIME": DIRECT_SUFFIX + " Use only the shortest calculation needed.",
    "MMLU-Pro": DIRECT_SUFFIX + " Use only the stated premises; do not import outside context.",
    "HLE": DIRECT_SUFFIX + " Treat the synthetic notation literally; do not import outside hierarchy meanings.",
    "BFCL": DIRECT_SUFFIX + " In <think>, check schema only; emit the final JSON once after </think>.",
    "AutonomousEvolution": DIRECT_SUFFIX + " Apply the given relations once, reduce, and stop.",
}


def _system_for_split(split: str, base: str) -> str:
    for name, suffix in SYSTEM_SUFFIXES.items():
        if name in split:
            return base + suffix
    return base


def _lcb_user(item: Dict[str, Any]) -> str:
    return (
        f"{item['prompt']}\n\n"
        "Think only: algorithm + key invariant/edge case. Close </think>; output ONLY executable Python in ```python ... ```."
    )


def _deep_swe_user(item: Dict[str, Any]) -> str:
    repo_text = "\n\n".join(
        f"### {path}\n```\n{body}\n```" for path, body in item["repo_files"].items()
    )
    return (
        "Repair the repository. Think only: failure -> file/change -> important edge/test. "
        "Close </think>; output ONLY the unified diff patch.\n\n"
        f"Repository files:\n{repo_text}\n\nTest command: {item['test_cmd']}"
    )


def _aime_user(item: Dict[str, Any]) -> str:
    return f"{item['prompt']}\nUse the shortest calculation; close </think>; output ONLY \\boxed{{answer}}."


def _choice_user(item: Dict[str, Any]) -> str:
    return f"{item['prompt']}\nUse only the stated premises; one deduction, then </think>; output ONLY A, B, C, or D."


def _hle_user(item: Dict[str, Any]) -> str:
    return f"{item['prompt']}\nSubstitute the stated I-index literally; close </think>; output ONLY the exact Con(...) expression."


def _bfcl_user(item: Dict[str, Any]) -> str:
    return (
        f"{item['prompt']}\nCheck the tool name and arguments once. Keep final JSON out of <think>; "
        "after </think> emit exactly ONE JSON object with keys `name` and `arguments`."
    )


def _dsl_user(item: Dict[str, Any]) -> str:
    # Preserve the Gemini-style DSL prompt; add only the exact left-rotation rule
    # exposed by the random k=2 failure.
    return (
        f"{item['prompt']}\n"
        "DSL Rules:\n"
        "- `arr >>~fold(k)`: Rotates list left by k positions: `arr[k:] + arr[:k]`.\n"
        "- `arr <#>scale(s)`: Multiplies each element by scalar s.\n"
        "- `arr1 @fuse arr2`: Element-wise addition.\n"
        "Calculate on scratchpad and output the final numeric list [x, y, ...] directly."
    )


def _autoevol_user(item: Dict[str, Any]) -> str:
    return f"{item['prompt']}\nApply the relations once, reduce the exponent modulo the order, close </think>, output ONLY the final power."


def _dialogue_user(item: Dict[str, Any]) -> str:
    return f"{item['prompt']}\nRecall only; close </think>; output ONLY the fact if known, otherwise `unknown`."


def _task_user(split: str, item: Dict[str, Any], original) -> str:
    # HumanEval/GSM8K/MATH/Zebra already behaved well: preserve original handlers.
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
    """Install concise task prompts without replacing the verified Gemini base."""
    if getattr(cls, "_code_prompt_hardening_installed", False):
        return

    original_eval = cls._evaluate_single_item
    original_task_user_prompt = phase4_module._task_user_prompt
    original_rsi = phase4_module._run_rsi_self_improvement
    base_phase4_system = phase4_module.SYSTEM_PROMPT

    targeted = (
        "LiveCodeBench", "DeepSWE", "AIME", "GPQA", "MMLU-Pro", "HLE",
        "BFCL", "TensorGraphDSL", "AutonomousEvolution", "DialogueRecall",
    )

    def hardened_eval(self, split, item):
        if not any(name in split for name in targeted):
            return original_eval(self, split, item)

        tok = self.engine.tokenizer
        user_message = _task_user(split, item, original_task_user_prompt)
        system_message = _system_for_split(split, runtime_module.SYSTEM_PROMPT)
        prompt = runtime_module._chat(tok, user_message, system=system_message)
        ceiling = getattr(self, "benchmark_max_tokens", None) or runtime_module._benchmark_ceiling(self)
        out = self._fast_generate(prompt, max_tokens=ceiling)
        self.last_raw_out = out
        runtime_module._append_raw_generation_log(self, prompt, user_message, out)

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
        if getattr(phase4_module, "_task_system_routing_active", False):
            phase4_module.SYSTEM_PROMPT = _system_for_split(split, base_phase4_system)
        return _task_user(split, item, original_task_user_prompt)

    def routed_rsi(self, splits, cache):
        previous = phase4_module.SYSTEM_PROMPT
        phase4_module._task_system_routing_active = True
        try:
            return original_rsi(self, splits, cache)
        finally:
            phase4_module._task_system_routing_active = False
            phase4_module.SYSTEM_PROMPT = previous

    cls._evaluate_single_item = hardened_eval
    phase4_module._task_user_prompt = hardened_task_user_prompt
    phase4_module._run_rsi_self_improvement = routed_rsi
    cls._code_prompt_hardening_installed = True
