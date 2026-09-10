import inspect

import master_4000_eval_suite as master
from eval import master_4000_runtime as runtime


def test_master_runtime_is_installed():
    cls = master.Master4000EvaluationEngine
    assert cls._fast_generate is runtime.fast_generate
    assert cls._evaluate_single_item is runtime.evaluate_one
    assert cls._evaluate_all_splits is runtime.evaluate_all
    assert cls.run_full_suite is runtime.run_full


def test_master_contract_keeps_final_prompt_and_fastpath():
    assert "strictly minimal" in runtime.SYSTEM_PROMPT
    assert "No conversational monologue" in runtime.SYSTEM_PROMPT
    sig = inspect.signature(runtime.fast_generate)
    assert sig.parameters["max_tokens"].default == 4096
    src = inspect.getsource(runtime.fast_generate)
    assert "mx.argmax" in src
    assert "mx.eval(a)" in src
    assert "clear_cache" in src


def test_all_master_stages_are_wired():
    src = inspect.getsource(runtime.run_full)
    for marker in (
        "Phase 1: Baseline",
        "DialogueTimelineGraphIngester",
        "search_best_invariant",
        "OGP",
        "Phase 4: Post-Consolidation",
        "_generate_master_report",
    ):
        assert marker in src
