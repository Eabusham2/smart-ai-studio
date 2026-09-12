"""Code-task prompt hardening proven by live A/B tests.

The global Gemini-tested SYSTEM_PROMPT is intentionally unchanged. This layer
only tightens user-side instructions for HumanEval, LiveCodeBench and DeepSWE so
the reasoning model starts solving immediately instead of restating/tutorializing.

Observed reference on LiveCodeBench-Hard_0 with the same 27B model:
- old code instruction: PASS, 1024 thinking tokens, no </think>, ~181 s wall
- tight code instruction: PASS, 11 thinking tokens, closed </think>, ~43 s wall

The production policy does NOT target 11 tokens. Trivial tasks may need one terse
line; normal coding should keep a compact 2-4 line reasoning sketch; complex code
or repository repairs may use a short 2-6 step implementation/repair sketch.
Reasoning should be brief but real, then close </think> once the implementation
path is sufficiently worked out.

Installation happens before phase4_pro_rsi.install(), so Phase 4 captures the
same hardened evaluator and RSI task-prompt builder.
"""
from __future__ import annotations

from typing import Any, Dict


COMMON_CODE_THINK_RULES = (
    "Start solving immediately inside <think>. "
    "Do not restate the task, describe what the user wants, list requirements, explain definitions, "
    "compare alternatives after an adequate approach is clear, narrate a plan, or repeat verification. "
    "Keep reasoning concise but sufficient: trivial code may use one terse line; ordinary code should use a compact "
    "2-4 line implementation sketch; genuinely complex code may use a few terse steps for the key algorithm, invariants, "
    "and edge cases. Do not stop before the implementation path is clear, but do not tutorialize. "
    "Once the solution is sufficiently worked out, close </think>. "
    "Do NOT emit the requested code until after </think>. "
)

DEEPSWE_THINK_RULES = (
    "Start diagnosing immediately inside <think>. "
    "Do not restate the task, describe what the user wants, list the repository back to yourself, explain generic concepts, "
    "or repeat verification. Use a short concrete repair sketch, usually 2-6 terse lines: identify the failing behavior, "
    "the file/change, any important compatibility edge case, and the test implication. "
    "Reason enough to make the patch reliable, then close </think>. "
    "Do NOT emit the patch until after </think>. "
)


def _human_eval_user(item: Dict[str, Any]) -> str:
    return (
        f"{item['prompt']}\n\n"
        "Complete the Python function above. "
        + COMMON_CODE_THINK_RULES
        + "After </think>, output ONLY the valid executable Python code wrapped in ```python ... ```."
    )


def _lcb_user(item: Dict[str, Any]) -> str:
    return (
        f"{item['prompt']}\n\n"
        "Write the complete Python solution requested above. "
        + COMMON_CODE_THINK_RULES
        + "For a standard algorithm, name it tersely and note only the key invariant/edge case needed before coding. "
        "After </think>, output ONLY valid executable Python wrapped in ```python ... ```."
    )


def _deep_swe_user(item: Dict[str, Any]) -> str:
    repo_text = "\n\n".join(
        f"### {path}\n```\n{body}\n```"
        for path, body in item["repo_files"].items()
    )
    return (
        "Repair the repository so the test command passes. "
        + DEEPSWE_THINK_RULES
        + "After </think>, output ONLY the unified diff patch.\n\n"
        f"Repository files:\n{repo_text}\n\nTest command: {item['test_cmd']}"
    )


def install(runtime_module, phase4_module, cls) -> None:
    """Install concise code prompts without changing generation/scoring/system prompt."""
    if getattr(cls, "_code_prompt_hardening_installed", False):
        return

    original_eval = cls._evaluate_single_item
    original_task_user_prompt = phase4_module._task_user_prompt

    def hardened_eval(self, split, item):
        if not any(name in split for name in ("HumanEval", "LiveCodeBench", "DeepSWE")):
            return original_eval(self, split, item)

        tok = self.engine.tokenizer

        def gen(user_message: str):
            prompt = runtime_module._chat(tok, user_message, system=runtime_module.SYSTEM_PROMPT)
            ceiling = getattr(self, "benchmark_max_tokens", None) or runtime_module._benchmark_ceiling(self)
            out = self._fast_generate(prompt, max_tokens=ceiling)
            self.last_raw_out = out
            runtime_module._append_raw_generation_log(self, prompt, user_message, out)
            return out

        if "HumanEval" in split:
            out = gen(_human_eval_user(item))
            code = runtime_module.clean_output(out)
            result = self.engine.sandbox.execute_python_code(
                item["prompt"] + "\n" + code,
                item["test"],
            )
            return bool(result.passed)

        if "LiveCodeBench" in split:
            out = gen(_lcb_user(item))
            code = runtime_module.clean_output(out)
            result = self.engine.sandbox.execute_python_code(code, item["test"])
            return bool(result.passed)

        out = gen(_deep_swe_user(item))
        patch = runtime_module.clean_output(out)
        return bool(
            self.engine.sandbox.verify_git_diff_patch(
                item["repo_files"],
                patch,
                item["test_cmd"],
            ).passed
        )

    def hardened_task_user_prompt(split: str, item: Dict[str, Any]) -> str:
        if "HumanEval" in split:
            return _human_eval_user(item)
        if "LiveCodeBench" in split:
            return _lcb_user(item)
        if "DeepSWE" in split:
            return _deep_swe_user(item)
        return original_task_user_prompt(split, item)

    cls._evaluate_single_item = hardened_eval
    phase4_module._task_user_prompt = hardened_task_user_prompt
    cls._code_prompt_hardening_installed = True
