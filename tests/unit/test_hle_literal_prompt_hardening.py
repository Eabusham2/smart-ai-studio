"""Regression coverage for HLE literal-token prompt hardening."""

import eval.code_prompt_hardening as hard


def _original(split, item):
    return "ORIGINAL"


def test_hle_literal_token_is_extracted_from_prompt_not_expected_answer():
    for token in ("I0", "I1", "I2", "I17", "I_ALPHA"):
        item = {
            "prompt": (
                f"In this synthetic notation, T = ZFC + {token}, where {token} is the stated axiom. "
                "Give the relative consistency bound using the form Con(ZFC + Ik)."
            ),
            "expected": "SHOULD_NOT_BE_READ",
        }
        assert hard._hle_literal_token(item["prompt"]) == token
        routed = hard._task_user("HLE-100", item, _original)
        assert token in routed
        assert "SHOULD_NOT_BE_READ" not in routed
        assert "using the form Con(ZFC + Ik)" not in routed
        assert "opaque literal axiom token" in routed
        assert "exactly one substitution line" in routed


def test_hle_system_rule_is_family_wide_and_generic():
    routed = hard._system_for_split("HLE-100", "GEMINI_BASE")
    assert routed.startswith("GEMINI_BASE")
    assert "axiom token after `T = ZFC +` is opaque literal text" in routed
    assert "never reinterpret" in routed
    assert "I0" not in hard.SYSTEM_SUFFIXES["HLE"]
    assert "I1" not in hard.SYSTEM_SUFFIXES["HLE"]
    assert "I2" not in hard.SYSTEM_SUFFIXES["HLE"]


def test_hle_generation_has_small_family_backstop():
    assert 32 <= hard.HLE_GENERATION_CEILING <= 512
