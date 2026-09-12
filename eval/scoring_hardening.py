"""Strict answer scoring for the 4,014-item evaluator.

This layer fixes false-positive grading without changing generation, prompts,
checkpoint semantics, telemetry, or the Pro/RSI search policy.

Important invariants:
- Code/tool tasks keep their deterministic sandbox/schema verifiers.
- Math is scored by an exact final numeric answer, never substring matching.
- Multiple-choice answers are parsed from an explicit final choice, never from
  a stray A/B/C/D appearing somewhere in prose.
- TensorGraphDSL is scored from the final list value, not from an intermediate
  scratchpad line.
- RSI still selects candidates without ground truth; hidden ground truth is
  consulted only after selection to assign reward.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, Optional

from eval.master_4000_runtime import clean_output


_LABEL_RE = re.compile(
    r"(?i)(?:final\s+answer|answer|result|choice|option)\s*(?:is\s*)?[:=\-]?\s*"
)


def _last_nonempty_line(text: str) -> str:
    cleaned = clean_output(text or "")
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return lines[-1] if lines else cleaned.strip()


def _normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("**", "").replace("$", "")
    text = _LABEL_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip().rstrip(".。")


def _boxed_values(text: str):
    return re.findall(r"\\boxed\{([^{}]+)\}", text or "")


def _integer_value(text: str) -> Optional[int]:
    s = _normalize_text(text)
    s = s.replace(",", "").strip()
    if re.fullmatch(r"[+-]?\d+", s):
        try:
            return int(s)
        except Exception:
            return None
    return None


def _final_integer(text: str) -> Optional[int]:
    boxed = _boxed_values(text)
    if boxed:
        return _integer_value(boxed[-1])

    line = _last_nonempty_line(text)
    direct = _integer_value(line)
    if direct is not None:
        return direct

    # Accept explicit final-answer prose while refusing arbitrary substring hits.
    match = re.search(
        r"(?i)(?:final\s+answer|answer|result)\s*(?:is\s*)?[:=\-]?\s*([+-]?\d[\d,]*)\b",
        line,
    )
    if match:
        return _integer_value(match.group(1))
    return None


def _final_choice(text: str) -> Optional[str]:
    boxed = _boxed_values(text)
    for value in reversed(boxed):
        m = re.fullmatch(r"\s*\(?([A-Da-d])\)?\s*", value)
        if m:
            return m.group(1).upper()

    line = _last_nonempty_line(text)
    m = re.fullmatch(r"\s*(?:\*\*)?\(?([A-Da-d])\)?[\).:]?(?:\*\*)?\s*", line)
    if m:
        return m.group(1).upper()

    matches = re.findall(
        r"(?i)(?:final\s+answer|answer|choice|option)\s*(?:is\s*)?[:=\-]?\s*\(?([A-D])\)?\b",
        line,
    )
    return matches[-1].upper() if matches else None


def _strict_math(expected: str, output: str) -> bool:
    expected_int = _integer_value(expected)
    if expected_int is not None:
        actual = _final_integer(output)
        return actual is not None and actual == expected_int

    boxed = _boxed_values(output)
    if boxed:
        return _normalize_text(boxed[-1]).casefold() == _normalize_text(expected).casefold()
    return _normalize_text(_last_nonempty_line(output)).casefold() == _normalize_text(expected).casefold()


def _strict_text(expected: str, output: str) -> bool:
    exp = _normalize_text(expected).casefold()
    if not exp:
        return False
    line = _normalize_text(_last_nonempty_line(output)).casefold()
    if line == exp:
        return True
    # Multi-token facts may naturally appear in a short final sentence.
    if len(exp) >= 4 and exp in line:
        return True
    return False


def _strict_dsl(self, item: Dict[str, Any], output: str) -> bool:
    try:
        expected = self.engine.sandbox.evaluate_dsl_expression(item["dsl_expr"])
    except Exception:
        return False
    if expected is None:
        return False

    candidates = re.findall(r"\[[^\[\]]*\]", clean_output(output or ""))
    if not candidates:
        candidates = re.findall(r"\[[^\[\]]*\]", output or "")
    if not candidates:
        return False

    try:
        actual = ast.literal_eval(candidates[-1])
    except Exception:
        return False
    return actual == expected


def strict_score(self, split: str, item: Dict[str, Any], output: str) -> Optional[bool]:
    """Return a strict score, or None when the existing deterministic verifier owns scoring."""
    if any(name in split for name in ("HumanEval", "LiveCodeBench", "DeepSWE", "BFCL")):
        return None

    if "TensorGraphDSL" in split:
        return _strict_dsl(self, item, output)

    if any(name in split for name in ("GSM8K", "MATH", "AIME")):
        return _strict_math(str(item.get("expected", "")), output)

    expected = str(
        item.get("expected", item.get("expected_token", item.get("expected_keyword", "")))
    ).strip()
    if not expected:
        return None

    # GPQA/MMLU-style multiple choice must return an explicit final option.
    if expected.upper() in {"A", "B", "C", "D"} and len(expected) == 1:
        return _final_choice(output) == expected.upper()

    # Zebra's small numeric index is an exact final integer, not a substring.
    if "ZebraLogic" in split and _integer_value(expected) is not None:
        return _final_integer(output) == _integer_value(expected)

    return _strict_text(expected, output)


def install(cls, phase4_module) -> None:
    """Install strict grading after the existing runtime/Pro/historical layers."""
    if getattr(cls, "_strict_scoring_installed", False):
        return

    original_eval = cls._evaluate_single_item
    original_answer_blind = phase4_module._candidate_passes_answer_blind

    def hardened_evaluate(self, split, item):
        original = bool(original_eval(self, split, item))
        strict = strict_score(self, split, item, getattr(self, "last_raw_out", ""))
        return original if strict is None else bool(strict)

    def hardened_answer_blind(self, split, item, candidate):
        if "TensorGraphDSL" in split:
            return _strict_dsl(self, item, candidate)
        return original_answer_blind(self, split, item, candidate)

    def hardened_hidden_reward(self, split, item, candidate):
        # Deterministic task verifiers are answer-blind and remain authoritative.
        verified = hardened_answer_blind(self, split, item, candidate)
        if verified is not None:
            return bool(verified)

        # Ground truth is consulted only now, after candidate selection.
        strict = strict_score(self, split, item, candidate)
        return bool(strict) if strict is not None else False

    cls._evaluate_single_item = hardened_evaluate
    cls._strict_scoring_installed = True
    phase4_module._candidate_passes_answer_blind = hardened_answer_blind
    phase4_module._hidden_reward_only_after_selection = hardened_hidden_reward
