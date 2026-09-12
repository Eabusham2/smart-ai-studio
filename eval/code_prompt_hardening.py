"""Code-task prompt hardening proven by the live LCB A/B test.

The global Gemini-tested SYSTEM_PROMPT is intentionally unchanged.  This layer
only tightens the user-side instructions for HumanEval, LiveCodeBench and
DeepSWE so the reasoning model does not turn the template-opened <think> block
into a tutorial/verification monologue.

Measured reference on LiveCodeBench-Hard_0 with the same 27B model:
- old code instruction: PASS, 1024 thinking tokens, no </think>, ~181 s wall
- tight code instruction: PASS, 11 thinking tokens, closed </think>, ~43 s wall

Installation happens before phase4_pro_rsi.install(), so Phase 4 captures this
baseline evaluator and its RSI task-prompt builder is patched consistently.
"""
from __future__ import annotations

from typing import Any, Dict


CODE_THINK_RULES = (
    "Inside <think>, do not restate the task, list requirements, explain definitions, compare algorithms, "
    "walk examples, or verify an already-known solution. "
    "Use at most one terse implementation/repair note, then close </think> immediately. "
    "Do NOT emit the requested code or patch until after </think>. "
)


def _human_eval_user(item: Dict[str, Any]) -> str:
    return (
        f"{item['prompt']}\n\n"
        "Complete the Python function above. "
        + CODE_THINK_RULES
        + "After </think>, output ONLY the valid executable Python code wrapped in ```python ... ```."
    )


def _lcb_user(item: Dict[str, Any]) -> str:
    return (
        f"{item['prompt']}\n\n"
        "Write the complete Python solution requested above. "
        + CODE_THINK_RULES
        + "For an obvious standard algorithm, one short tag such as `merge-sort inversions; O(n log n)` is sufficient scratch work. "
        "After </think>, output ONLY valid executable Python wrapped in ```python ... ```."
    )


def _deep_swe_user(item: Dict[str, Any]) -> str:
    repo_text = "\n\n".join(
        f"### {path}\n```\n{body}\n```"
        for path, body in item["repo_files"].items()
    )
    return (
        "Repair the repository so the test command passes. "
        + CODE_THINK_RULES
        + "After </think>, output ONLY the unified diff patch.\n\n"
        f"Repository files:\n{repo_text}\n\nTest command: {item['test_cmd']}"
    )


def install(runtime_module, phase4_module, cls) -> None:
    """Install tight code prompts without changing generation/scoring/system prompt."""
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
