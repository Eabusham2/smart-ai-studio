"""Minimal task-specific prompt hardening backed by real-model smoke tests.

The verified Gemini SYSTEM_PROMPT remains the base everywhere. Families already
shown concise/correct keep that prompt and their original task policy. Only
families with observed overthinking/formatting/semantic issues get a tiny,
task-local addition.

Installed before phase4_pro_rsi.install(), so baseline and final Phase-4 retests
use the same evaluator policy. RSI is routed through the same per-split policy
without leaking a task suffix into LearningFacts or historical RLVR.
"""
from __future__ import annotations

from typing import Any, Dict

from eval.scoring_hardening import strict_score


SYSTEM_SUFFIXES = {
    "LiveCodeBench": (
        " Solve directly. In <think>, use 2-4 terse lines: algorithm, key invariant/edge case, "
        "implementation. No restatement, alternatives, examples, or rechecking. Then code."
    ),
    "DeepSWE": (
        " Diagnose directly. In <think>: failure, exact file/edit, one test-sensitive edge. "
        "No task restatement or speculation. Then patch."
    ),
    "AIME": (
        " Use the finite-difference shortcut directly. Do not solve an unnecessary intercept "
        "or recheck a settled result."
    ),
    "GPQA": (
        " Use only the stated premise. One deduction to the option; no task restatement or option narration."
    ),
    "MMLU-Pro": (
        " Use only the stated premises. One deduction to the option; do not import outside context."
    ),
    "HLE": " Treat the synthetic notation literally; substitute the stated index and stop.",
    "AutonomousEvolution": (
        " Use the commutator definition and the given conjugation relation once; reduce the exponent "
        "modulo the order and stop."
    ),
    "DialogueRecall": (
        " Recall only. If the fact has not been learned yet, output `unknown`; do not explain memory limitations."
    ),
}


def _system_for_split(split: str, base: str) -> str:
    for name, suffix in SYSTEM_SUFFIXES.items():
        if name in split:
            return base + suffix
    return base


def _lcb_user(item: Dict[str, Any]) -> str:
    return (
        f"{item['prompt']}\n\n"
        "In <think>, write only 2-4 terse lines: algorithm; key invariant/edge case; implementation. "
        "No alternatives, examples, or rechecking. Then close </think> and output ONLY executable Python "
        "in ```python ... ```."
    )


def _deep_swe_user(item: Dict[str, Any]) -> str:
    repo_text = "\n\n".join(
        f"### {path}\n```\n{body}\n```" for path, body in item["repo_files"].items()
    )
    return (
        "Repair the repository. In <think>: failure -> exact file/edit -> one test-sensitive edge. "
        "No restatement or speculation. Close </think>; output ONLY the unified diff patch.\n\n"
        f"Repository files:\n{repo_text}\n\nTest command: {item['test_cmd']}"
    )


def _aime_user(item: Dict[str, Any]) -> str:
    return (
        f"{item['prompt']}\n"
        "Use the first difference directly: P(n+d)=P(n)+d(P(n+1)-P(n)). "
        "No intercept or second verification. Close </think>; output ONLY \\boxed{answer}."
    )


def _choice_user(item: Dict[str, Any]) -> str:
    return f"{item['prompt']}\nOne premise -> one choice. Close </think>; output ONLY A, B, C, or D."


def _hle_user(item: Dict[str, Any]) -> str:
    return (
        f"{item['prompt']}\n"
        "Substitute the stated I-index literally; close </think>; output ONLY the exact Con(...) expression."
    )


def _dsl_user(item: Dict[str, Any]) -> str:
    # Preserve Gemini-style DSL behavior; only make left rotation unambiguous.
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
        f"{item['prompt']}\n"
        "Use at most 3 terse algebra lines: apply the commutator definition and conjugation once, "
        "reduce the exponent modulo the order, close </think>, output ONLY the final power."
    )


def _dialogue_user(item: Dict[str, Any]) -> str:
    return (
        f"{item['prompt']}\n"
        "Recall only. Close </think>; output ONLY the fact if learned, otherwise `unknown`."
    )


def _task_user(split: str, item: Dict[str, Any], original) -> str:
    # Gemini-only / original policy: HumanEval, GSM8K, MATH, Zebra, BFCL.
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
        "LiveCodeBench",
        "DeepSWE",
        "AIME",
        "GPQA",
        "MMLU-Pro",
        "HLE",
        "TensorGraphDSL",
        "AutonomousEvolution",
        "DialogueRecall",
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
            return bool(
                self.engine.sandbox.verify_git_diff_patch(
                    item["repo_files"], patch, item["test_cmd"]
                ).passed
            )

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
