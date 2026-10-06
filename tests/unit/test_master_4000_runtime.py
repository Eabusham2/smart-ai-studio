from pathlib import Path
import inspect

import master_4000_eval_suite as master
from eval import master_4000_runtime as runtime
from eval import phase4_pro_rsi
from eval import scoring_hardening
from eval import stage_integrity_telemetry


def test_master_runtime_is_installed():
    cls = master.Master4000EvaluationEngine
    for name in ("_fast_generate", "_evaluate_single_item", "_evaluate_all_splits", "run_full_suite"):
        assert callable(getattr(cls, name))

    assert cls._fast_generate.__module__ == "eval.unified_context_budget"
    assert cls._evaluate_single_item.__module__ == "eval.real_phase4_context"
    assert callable(cls._evaluate_all_splits)
    assert callable(cls.run_full_suite)


def test_master_contract_keeps_final_prompt_and_fastpath():
    assert "strictly minimal" in runtime.SYSTEM_PROMPT
    assert "concise intermediate formulas or numbers" in runtime.SYSTEM_PROMPT
    assert "No conversational monologue" in runtime.SYSTEM_PROMPT
    sig = inspect.signature(runtime.fast_generate)
    assert sig.parameters["max_tokens"].default == 16384
    src = inspect.getsource(runtime.fast_generate)
    assert "_remaining(" in src
    assert "original_fast(" in src
    speedup = (Path(__file__).resolve().parents[2] / "eval/lossless_baseline_speedup.py").read_text(encoding="utf-8")
    assert "stream_generate" in speedup
    assert "make_sampler" in speedup


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
