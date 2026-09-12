import inspect

import master_4000_eval_suite as master
from eval import master_4000_runtime as runtime
from eval import phase4_pro_rsi
from eval import scoring_hardening


def test_master_runtime_is_installed():
    cls = master.Master4000EvaluationEngine
    for name in ("_fast_generate", "_evaluate_single_item", "_evaluate_all_splits", "run_full_suite"):
        assert callable(getattr(cls, name))

    assert cls._fast_generate.__module__ == phase4_pro_rsi.__name__
    assert cls._evaluate_single_item.__module__ == scoring_hardening.__name__
    assert cls._evaluate_all_splits.__module__ == phase4_pro_rsi.__name__
    assert cls.run_full_suite.__module__ == phase4_pro_rsi.__name__


def test_master_contract_keeps_final_prompt_and_fastpath():
    assert "strictly minimal" in runtime.SYSTEM_PROMPT
    assert "concise intermediate formulas or numbers" in runtime.SYSTEM_PROMPT
    assert "No conversational monologue" in runtime.SYSTEM_PROMPT
    sig = inspect.signature(runtime.fast_generate)
    assert sig.parameters["max_tokens"].default == 16384
    src = inspect.getsource(runtime.fast_generate)
    assert "mx.argmax" in src
    assert "mx.eval(next_arr)" in src
    assert "clear_cache" in src


def test_all_master_stages_are_wired():
    src = inspect.getsource(phase4_pro_rsi)
    for marker in (
        "Phase 1: Baseline",
        "DialogueTimelineGraphIngester",
        "search_best_invariant",
        "RSI: RECURSIVE SELF-IMPROVEMENT",
        "PHASE 3: LEARN + RSI PARAMETRIC CONSOLIDATION",
        "Learning Test",
        "Phase 4: Post-Consolidation",
        "_generate_master_report",
    ):
        assert marker in src
