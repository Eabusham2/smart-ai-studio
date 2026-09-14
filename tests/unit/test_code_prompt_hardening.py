"""Regression tests for evidence-driven task-specific prompt routing."""

from pathlib import Path

import eval.code_prompt_hardening as hardening
import eval.master_4000_runtime as runtime
import eval.phase4_pro_rsi as phase4
import master_4000_eval_suite as entry


VERIFIED_GEMINI_PROMPT = (
    "You are a fast symbolic computing engine. "
    "Keep your internal scratchpad (<think>) strictly minimal: write only concise intermediate formulas or numbers. "
    "No conversational monologue, no self-reflection, and no verification loops. "
    "Close </think> immediately once calculated and output the answer."
)
EXPECTED_GLOBAL_PROMPT = VERIFIED_GEMINI_PROMPT + hardening.GLOBAL_SYSTEM_SUFFIX


def test_global_system_prompt_keeps_verified_gemini_base_and_anti_loop_rule():
    assert runtime.SYSTEM_PROMPT == EXPECTED_GLOBAL_PROMPT
    assert phase4.SYSTEM_PROMPT == EXPECTED_GLOBAL_PROMPT
    assert runtime.SYSTEM_PROMPT.startswith(VERIFIED_GEMINI_PROMPT)
    assert runtime.SYSTEM_PROMPT.count(hardening.GLOBAL_SYSTEM_SUFFIX.strip()) == 1
    assert "more than twice" in runtime.SYSTEM_PROMPT
    assert "unknown" not in runtime.SYSTEM_PROMPT.lower()


def test_prompt_hardening_is_installed_before_phase4_capture():
    assert getattr(entry.Master4000EvaluationEngine, "_code_prompt_hardening_installed", False)
    source = Path("master_4000_eval_suite.py").read_text(encoding="utf-8")
    assert source.index("install_code_prompt_hardening(") < source.index("phase4_pro_rsi.install(")
    phase_source = Path("eval/phase4_pro_rsi.py").read_text(encoding="utf-8")
    assert "base_eval = cls._evaluate_single_item" in phase_source
    assert "result = base_eval(self, split, item)" in phase_source
    assert "original_user = _task_user_prompt(split_name, item)" in phase_source


def test_global_rule_reaches_every_family_and_only_proven_overthinkers_get_extra_suffixes():
    base = VERIFIED_GEMINI_PROMPT
    assert "2-4 terse lines" in hardening._system_for_split("LiveCodeBench-Hard", base)
    assert "finite-difference shortcut" in hardening._system_for_split("AIME-150", base)
    assert "one terse line" in hardening._system_for_split("GPQA-400", base)
    assert "stated premises" in hardening._system_for_split("MMLU-Pro-1000", base)
    assert "opaque literal text" in hardening._system_for_split("HLE-100", base)
    assert "commutator definition" in hardening._system_for_split("AutonomousEvolution-200", base)
    dialogue_system = hardening._system_for_split("DialogueRecall-150", base)
    assert "output `unknown` exactly once" in dialogue_system
    assert "Never expand abbreviations" in dialogue_system
    assert "re-check the same missing fact" in dialogue_system

    for split in (
        "HumanEval-164", "GSM8K-500", "MATH-500", "ZebraLogic-200",
        "BFCL-200", "TensorGraphDSL-300",
    ):
        routed = hardening._system_for_split(split, base)
        assert routed.startswith(base)
        assert routed.count(hardening.GLOBAL_SYSTEM_SUFFIX.strip()) == 1

    assert hardening._system_for_split("MATH-500", base) == EXPECTED_GLOBAL_PROMPT


def test_lcb_and_deepswe_prompts_are_brief_but_not_reasoning_free():
    lcb = phase4._task_user_prompt("LiveCodeBench-Hard", {"prompt": "Write f(arr)."})
    swe = phase4._task_user_prompt(
        "DeepSWE-50",
        {"repo_files": {"a.py": "x=1\n"}, "test_cmd": "pytest -q"},
    )
    assert "2-4 terse lines" in lcb
    assert "algorithm" in lcb and "invariant/edge case" in lcb and "implementation" in lcb
    assert "alternatives, examples, or rechecking" in lcb
    assert "failure -> exact file/edit -> one test-sensitive edge" in swe
    assert "output ONLY the unified diff patch" in swe


def test_known_good_families_keep_original_task_policy():
    human = phase4._task_user_prompt("HumanEval-164", {"prompt": "def f(x):\n    pass"})
    gsm = phase4._task_user_prompt("GSM8K-500", {"prompt": "Compute 2+2"})
    zebra = phase4._task_user_prompt("ZebraLogic-200", {"prompt": "logic"})
    bfcl = phase4._task_user_prompt("BFCL-200", {"prompt": "tool prompt"})
    assert "Use scratchpad only for logic outline" in human
    assert "Solve this problem using a minimal scratchpad" in gsm
    assert "Deduce the solution directly. State the final answer on the last line." in zebra
    assert "Return ONLY one JSON object" in bfcl
    assert "Check the tool name and arguments once" not in bfcl


def test_targeted_non_code_prompts_are_short_and_specific():
    math = phase4._task_user_prompt("MATH-500", {"prompt": "modular residue"})
    aime = phase4._task_user_prompt("AIME-150", {"prompt": "AIME prompt"})
    gpqa = phase4._task_user_prompt("GPQA-400", {"prompt": "MC prompt"})
    mmlu = phase4._task_user_prompt("MMLU-Pro-1000", {"prompt": "MC prompt"})
    hle = phase4._task_user_prompt("HLE-100", {"prompt": "HLE prompt"})
    dsl = phase4._task_user_prompt("TensorGraphDSL-300", {"prompt": "dsl"})
    auto = phase4._task_user_prompt("AutonomousEvolution-200", {"prompt": "group"})
    dialogue = phase4._task_user_prompt("DialogueRecall-150", {"prompt": "recall"})
    assert "Never repeat a completed calculation" in math
    assert "at most 3 terse arithmetic lines" in math
    assert "One literal condition -> one option" in gpqa
    assert "One literal condition -> one option" in mmlu
    assert "first difference directly" in aime
    assert "Treat the axiom token after `T = ZFC +` as opaque literal text" in hle
    assert "arr[k:] + arr[:k]" in dsl
    assert "at most 3 terse algebra lines" in auto
    assert "otherwise `unknown` once" in dialogue
    assert "Do not expand abbreviations" in dialogue


def test_all_150_dialogue_recall_baseline_items_fail_immediately_without_model_generation():
    class Dummy:
        _current_phase = "Phase 1: Baseline"
        last_raw_out = "should be cleared"
        last_output_tokens = 99
        last_generation_seconds = 99.0

        @property
        def engine(self):
            raise AssertionError("baseline DialogueRecall must not touch the model/tokenizer")

    for i in range(150):
        dummy = Dummy()
        result = entry.Master4000EvaluationEngine._evaluate_single_item(
            dummy,
            "DialogueRecall-150",
            {
                "id": f"Dialogue_{i}",
                "prompt": f"Recall developer decision #{i % 10}",
                "expected_keyword": "BD PROCHOT",
            },
        )
        assert result is False
        assert dummy.last_raw_out == ""
        assert dummy.last_output_tokens == 0
        assert dummy.last_generation_seconds == 0.0


def test_dialogue_recall_is_masked_from_rsi_but_remains_false_in_real_cache_for_phase4():
    source = Path("eval/code_prompt_hardening.py").read_text(encoding="utf-8")
    assert 'rsi_cache = dict(cache)' in source
    assert 'rsi_cache[key] = "DEFERRED_PRE_LEARN_MEMORY"' in source
    assert 'return original_rsi(self, splits, rsi_cache)' in source
    assert 'return original_rsi(self, splits, cache)' not in source

    phase_source = Path("eval/phase4_pro_rsi.py").read_text(encoding="utf-8")
    assert 'cache.get(f"Phase 1: Baseline_{item[' in phase_source
    assert ') is False' in phase_source


def test_focused_smoke_is_exactly_five_one_go_families():
    source = Path("tools/random_changed_split_smoke.py").read_text(encoding="utf-8")
    for split in (
        "LiveCodeBench-Hard", "AIME-150", "GPQA-400",
        "DeepSWE-50", "AutonomousEvolution-200",
    ):
        assert split in source
    assert "DialogueRecall-150" not in source
    assert "rt.evaluate_one(" not in source
    assert "All-Split Smoke OLD Reference" not in source
    assert "eval_checkpoint_4000" not in source
    assert "load_checkpoint(" not in source
    assert "save_checkpoint(" not in source
    assert "five generations total" in source


def test_rsi_system_routing_restores_the_global_gemini_base_after_use():
    source = Path("eval/code_prompt_hardening.py").read_text(encoding="utf-8")
    assert "phase4_module._task_system_routing_active = True" in source
    assert "phase4_module.SYSTEM_PROMPT = _system_for_split" in source
    assert "phase4_module.SYSTEM_PROMPT = previous" in source
