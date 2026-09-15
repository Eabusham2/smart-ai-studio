"""Regression coverage for RSI systemless prompting with family guidance retained."""
from eval.code_prompt_hardening import GLOBAL_SYSTEM_SUFFIX, SYSTEM_SUFFIXES
from eval.rsi_prompt_hardening import _family_guidance_from_system


VERIFIED_GEMINI_PROMPT = (
    "You are a fast symbolic computing engine. "
    "Keep your internal scratchpad (<think>) strictly minimal: write only concise intermediate formulas or numbers. "
    "No conversational monologue, no self-reflection, and no verification loops. "
    "Close </think> immediately once calculated and output the answer."
)


def test_rsi_extracts_family_guidance_without_gemini_or_global_text():
    family = SYSTEM_SUFFIXES["HLE"]
    routed = VERIFIED_GEMINI_PROMPT + GLOBAL_SYSTEM_SUFFIX + family
    extracted = _family_guidance_from_system(routed)
    assert extracted == family.strip()
    assert VERIFIED_GEMINI_PROMPT not in extracted
    assert GLOBAL_SYSTEM_SUFFIX.strip() not in extracted


def test_rsi_family_guidance_is_empty_when_split_has_no_suffix():
    routed = VERIFIED_GEMINI_PROMPT + GLOBAL_SYSTEM_SUFFIX
    assert _family_guidance_from_system(routed) == ""


def test_family_guidance_registry_remains_available_for_rsi():
    for name in ("LiveCodeBench", "DeepSWE", "AIME", "GPQA", "MMLU-Pro", "HLE", "AutonomousEvolution", "DialogueRecall"):
        assert SYSTEM_SUFFIXES[name].strip()
