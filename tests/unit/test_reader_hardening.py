"""Regression tests for narrow final-answer reader fixes."""

import eval.scoring_hardening as scoring
import master_4000_eval_suite  # installs reader hardening


def test_parenthesized_labeled_choice_with_text_is_explicit_final_answer():
    assert scoring._final_choice("<think>short deduction</think>\n(D) Asymptotic") == "D"


def test_letter_punctuation_with_text_is_explicit_final_answer():
    assert scoring._final_choice("A. Measure preserved") == "A"
    assert scoring._final_choice("C) Invariant Dual") == "C"


def test_reader_does_not_restore_loose_stray_letter_matching():
    assert scoring._final_choice("A transformation can preserve measure") is None
    assert scoring._final_choice("The discussion mentioned option D earlier, but no final choice was given") is None
