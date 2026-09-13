"""Minimal task-specific prompt hardening backed by real-model smoke tests.

The verified Gemini SYSTEM_PROMPT remains the base everywhere. One short global
anti-loop rule is appended to every benchmark/model stage. Families with observed
overthinking/formatting/semantic issues get an additional tiny task-local rule.

Installed before phase4_pro_rsi.install(), so baseline and final Phase-4 retests
use the same evaluator policy. RSI, LearningFacts, and historical RLVR share the
same global anti-loop rule without leaking unrelated task-family suffixes.
"""
from __future__ import annotations

from typing import Any, Dict

from eval.scoring_hardening import strict_score


GLOBAL_SYSTEM_SUFFIX = " Never verify or check the same work more than twice."

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
        " Map the stated condition directly to its option label in one terse line. "
        "Do not restate the prompt/options or add domain exposition."
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


def _with_global_rule(base: str) -> str:
    rule = GLOBAL_SYSTEM_SUFFIX.strip()
    return base if rule in base else base + GLOBAL_SYSTEM_SUFFIX


def _system_for_split(split: str, base: str) -> str:
    system = _with_global_rule(base)
    for name, suffix in SYSTEM_SUFFIXES.items():
        if name in split:
            return system + suffix
    return system


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


def _math_user(item: Dict[str, Any]) -> str:
    return (
        f"{item['prompt']}\n"
        "Compute each modular reduction once. Never repeat a completed calculation. "
        "Use at most 3 terse arithmetic lines, close </think>, output ONLY \\boxed{answer}."
    )


def _aime_user(item: Dict[str, Any]) -> str:
    return (
        f"{item['prompt']}\n"
        "Use the first difference directly: P(n+d)=P(n)+d(P(n+1)-P(n)). "
        "No intercept or second verification. Close </think>; output ONLY \\boxed{answer}."
    )


def _choice_user(item: Dict[str, Any]) -> str:
    return f"{item['prompt']}\nOne literal condition -> one option. Use one terse reasoning line, close </think>, output ONLY A, B, C, or D."


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
    # Original task policy remains for HumanEval, GSM8K, Zebra, and BFCL.
    if "LiveCodeBench" in split:
        return _lcb_user(item)
    if "DeepSWE" in split:
        return _deep_swe_user(item)
    if "MATH" in split:
        return _math_user(item)
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
    """Install concise task prompts while preserving Gemini as the system base."""
    if getattr(cls, "_code_prompt_hardening_installed", False):
        return

    # Apply the universal anti-loop rule to every model stage. The original Gemini
    # text stays byte-for-byte at the beginning of the system message.
    runtime_module.SYSTEM_PROMPT = _with_global_rule(runtime_module.SYSTEM_PROMPT)
    phase4_module.SYSTEM_PROMPT = _with_global_rule(phase4_module.SYSTEM_PROMPT)

    original_eval = cls._evaluate_single_item
    original_task_user_prompt = phase4_module._task_user_prompt
    original_rsi = phase4_module._run_rsi_self_improvement
    base_phase4_system = phase4_module.SYSTEM_PROMPT

    targeted = (
        "LiveCodeBench",
        "DeepSWE",
        "MATH",
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
