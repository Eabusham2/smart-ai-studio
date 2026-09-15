"""Restore useful legacy verifier-feedback behavior to current miss-only RSI.

Only answer-blind deterministic verifiers may produce correction feedback. Hidden
benchmark expected answers remain reward-only after branch selection and are never
inserted into generation. The current two-round/four-branch RSI policy is retained.
"""
from __future__ import annotations

import gc
import json
from typing import Any, Dict


def _clip_feedback(text: str, limit: int = 1200) -> str:
    clean = " ".join(str(text or "").split())
    return clean[:limit]


def _sandbox_feedback(result, prefix: str) -> str:
    if bool(getattr(result, "passed", False)):
        return ""
    detail = getattr(result, "error", None) or getattr(result, "output", None) or "deterministic verifier rejected the candidate"
    return _clip_feedback(f"{prefix}: {detail}")


def _answer_blind_failure_feedback(p4, self, split: str, item: Dict[str, Any], candidate: str) -> str:
    """Return verifier feedback derived only from prompt/tests, never expected_* fields."""
    try:
        if "HumanEval" in split:
            code = p4.clean_output(candidate)
            result = self.engine.sandbox.execute_python_code(item["prompt"] + "\n" + code, item["test"])
            return _sandbox_feedback(result, "Python verifier failure")

        if "LiveCodeBench" in split:
            code = p4.clean_output(candidate)
            result = self.engine.sandbox.execute_python_code(code, item["test"])
            return _sandbox_feedback(result, "Python verifier failure")

        if "DeepSWE" in split:
            patch = p4.clean_output(candidate)
            result = self.engine.sandbox.verify_git_diff_patch(item["repo_files"], patch, item["test_cmd"])
            return _sandbox_feedback(result, "Patch verifier failure")

        if "TensorGraphDSL" in split:
            value = self.engine.sandbox.evaluate_dsl_expression(item["dsl_expr"])
            if value is None:
                return "DSL verifier could not evaluate the literal expression; re-apply the stated operators exactly."
            expected = str(value).replace(" ", "")
            cleaned = p4.clean_output(candidate).replace(" ", "")
            if expected not in cleaned:
                return "DSL verifier rejected the selected result; re-apply the literal fold/scale/fuse semantics exactly once."
            return ""

        if "BFCL" in split:
            requested_name, requested_args = p4._prompt_requested_bfcl(item)
            value = p4._parse_json_object(candidate)
            if not isinstance(value, dict):
                return "Tool-call verifier could not parse one JSON object; return exactly the requested tool call JSON."
            if isinstance(value.get("function"), dict):
                value = value["function"]
            name = value.get("name") or value.get("tool")
            args = value.get("arguments") or value.get("args")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    return "Tool-call arguments were not valid JSON; emit the requested arguments as one JSON object."
            if name != requested_name or args != requested_args:
                return "Tool-call verifier rejected the selected tool name or arguments; copy them literally from the user request."
    except Exception as exc:
        return _clip_feedback(f"Answer-blind verifier raised {type(exc).__name__}: {exc}")
    return ""


def install(p4) -> None:
    """Use answer-blind verifier feedback in RSI round 2 without changing search width."""
    if getattr(p4, "_rsi_feedback_hardening_installed", False):
        return

    def feedback_rsi(self, splits, cache) -> int:
        p4._delete_unconsumed_session_rows(self, p4.RSI_SESSION_ID)
        model_identity = id(self.engine.model)
        seeded = 0
        attempted = 0

        misses = []
        for split_name, items in splits.items():
            for item in items:
                key = f"Phase 1: Baseline_{item['id']}"
                if cache.get(key) is False:
                    misses.append((split_name, item))

        if not misses:
            self._phase1_rsi_seed_count = 0
            print("[*] RSI: no Phase-1 misses to self-improve.", flush=True)
            return 0

        for split_name, item in misses[:64]:
            p4._assert_same_model(self, model_identity, "RSI")
            attempted += 1
            original_user = p4._task_user_prompt(split_name, item)
            previous = ""
            verifier_feedback = ""
            success = False

            for round_idx in (1, 2):
                # Stronger same-model invariant from the legacy loop: verify at
                # each recursive round, not just once at item entry.
                p4._assert_same_model(self, model_identity, f"RSI round {round_idx}")

                if round_idx == 1:
                    rsi_user = (
                        original_user
                        + "\n\nRecursive Self-Improvement: solve this task yourself from first principles. "
                        "Do not assume or request a hidden answer. Before finalizing, internally check likely failure modes, "
                        "then output the best corrected final response in the requested format."
                    )
                else:
                    rsi_user = (
                        original_user
                        + "\n\nRecursive Self-Improvement round 2. Your previous self-generated attempt was:\n"
                        + previous
                    )
                    if verifier_feedback:
                        rsi_user += (
                            "\n\nAnswer-blind deterministic verifier feedback from that attempt:\n"
                            + verifier_feedback
                        )
                    rsi_user += (
                        "\n\nCritique your own attempt, identify what may be wrong without access to any hidden answer, "
                        "and produce a materially improved final response in the requested format."
                    )

                # rsi_prompt_hardening still removes the system role and appends
                # any family-specific guidance to this user message.
                formatted = p4._chat(self.engine.tokenizer, rsi_user, system=p4.SYSTEM_PROMPT)
                temps = [0.20, 0.38, 0.58, 0.82]
                branches = p4._generate_branches_same_model(
                    self,
                    formatted,
                    temps,
                    max_tokens=min(p4._benchmark_ceiling(self), 16384),
                    top_p=0.92,
                )
                candidate, _, _, selection = p4._choose_without_ground_truth(self, split_name, item, branches)
                previous = candidate

                # Release the other potentially huge branch strings before
                # verification/next round; only the selected candidate survives.
                branches.clear()
                del branches
                gc.collect()

                passed = p4._hidden_reward_only_after_selection(self, split_name, item, candidate)
                if not passed and round_idx == 1 and p4._has_answer_blind_verifier(split_name):
                    verifier_feedback = _answer_blind_failure_feedback(p4, self, split_name, item, candidate)

                p4._append_rsi_log(
                    self,
                    split_name,
                    str(item.get("id", "unknown")),
                    round_idx,
                    candidate,
                    passed,
                    selection,
                )

                if passed:
                    try:
                        self.engine.kg.log_interaction(
                            p4.RSI_SESSION_ID,
                            str(item.get("prompt", "")),
                            candidate,
                            1.0,
                            1.0,
                            domain=f"RSI::{split_name}",
                        )
                        seeded += 1
                        success = True
                    except Exception:
                        pass
                    break

            if not success:
                continue

        self._phase1_rsi_seed_count = seeded
        print(
            f"[✓] RSI: {seeded}/{attempted} Phase-1 misses produced self-generated verified corrections.",
            flush=True,
        )
        return seeded

    p4._run_rsi_self_improvement = feedback_rsi
    p4._rsi_feedback_hardening_installed = True
