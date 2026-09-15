"""Regression coverage for RSI's clean verified-Gemini system prompt."""
from eval.code_prompt_hardening import GLOBAL_SYSTEM_SUFFIX, SYSTEM_SUFFIXES
from eval.rsi_prompt_hardening import _without_global_rule


VERIFIED_GEMINI_PROMPT = (
    "You are a fast symbolic computing engine. "
    "Keep your internal scratchpad (<think>) strictly minimal: write only concise intermediate formulas or numbers. "
    "No conversational monologue, no self-reflection, and no verification loops. "
    "Close </think> immediately once calculated and output the answer."
)


def test_rsi_removes_only_global_anti_loop_suffix():
    family = SYSTEM_SUFFIXES["HLE"]
    routed = VERIFIED_GEMINI_PROMPT + GLOBAL_SYSTEM_SUFFIX + family
    cleaned = _without_global_rule(routed)
    assert cleaned == VERIFIED_GEMINI_PROMPT + family
    assert GLOBAL_SYSTEM_SUFFIX.strip() not in cleaned


def test_clean_gemini_base_is_byte_preserved():
    assert _without_global_rule(VERIFIED_GEMINI_PROMPT + GLOBAL_SYSTEM_SUFFIX) == VERIFIED_GEMINI_PROMPT


def test_unrelated_text_is_not_modified():
    text = VERIFIED_GEMINI_PROMPT + " family-specific guidance"
    assert _without_global_rule(text) == text
