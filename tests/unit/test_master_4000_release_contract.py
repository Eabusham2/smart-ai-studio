"""Release-contract tests for the recovered 4,014-item evaluator.

These tests are intentionally source-level. They verify architecture/invariants
without loading the 27B MLX model in CI.
"""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _literal_assignment(path: str, name: str):
    tree = ast.parse(_src(path), filename=path)
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = None
            value = None
            if isinstance(node, ast.Assign):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    target = node.targets[0].id
                    value = node.value
            elif isinstance(node.target, ast.Name):
                target = node.target.id
                value = node.value
            if target == name:
                return ast.literal_eval(value)
    raise AssertionError(f"{name} not found in {path}")


def _has_call(path: str, attr_name: str) -> bool:
    tree = ast.parse(_src(path), filename=path)
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr_name
        for node in ast.walk(tree)
    )


def test_exact_gemini_low_overthink_prompt_and_raw_logger():
    runtime = _src("eval/master_4000_runtime.py")
    assert "write only concise intermediate formulas or numbers." in runtime
    assert "No conversational monologue, no self-reflection, and no verification loops." in runtime
    assert "Close </think> immediately once calculated and output the answer." in runtime
    assert 'RAW_OUTPUT_LOG = os.path.join("eval_results", "raw_model_outputs.log")' in runtime
    assert "METRICS AFTER RAW OUTPUT:" in runtime


def test_phase1_status_format_is_preserved():
    runtime = _src("eval/master_4000_runtime.py")
    assert 'f"[{phase}] {split_name:<22} | Item {overall}/{total} ({pct:5.2f}%) | "' in runtime
    assert 'f"Speed: {speed:4.1f}t/s | ETA: {eta} | RAM: {_real_ram_gb():.1f}GB"' in runtime


def test_historical_learning_facts_are_restored_additively():
    sessions = _literal_assignment("memory/dialogue_history_ingest.py", "HISTORICAL_DIALOGUE_SESSIONS")
    assert len(sessions) == 5
    assert sum(len(s.get("key_facts", [])) for s in sessions) == 10

    merge = _src("eval/historical_good_merge.py")
    assert "EPISODIC_DIALOGUE_RECALL_PROBE" in merge
    assert "HISTORICAL_DIALOGUE_SESSIONS" in merge
    assert "LEARN::LearningFacts" in merge
    assert "newer facts + recovered Sessions A-E" in merge
    assert "No 4K benchmark answer is inserted into the Learn payload." in merge


def test_learn_and_rsi_are_separate_and_rsi_is_self_generated():
    phase = _src("eval/phase4_pro_rsi.py")
    merge = _src("eval/historical_good_merge.py")

    assert "LEARN_SESSION_ID" in phase
    assert "RSI_SESSION_ID" in phase
    assert "SELF-GENERATED CANDIDATE" in phase
    assert "Critique your own attempt" in phase
    assert "Do not assume or request a hidden answer" in phase

    assert "AUTONOMOUS_RLVR_TASKS" in merge
    assert "No reference answer is available" in merge
    assert "Verifier feedback from that attempt" in merge
    assert "default_code" not in merge


def test_phase3_really_updates_and_measures_trainable_weights():
    phase = _src("eval/phase4_pro_rsi.py")
    merge = _src("eval/historical_good_merge.py")

    assert "nn.value_and_grad" in phase
    assert "optim.AdamW" in phase
    assert "opt.update(self.engine.model" in phase
    assert "rsi_post_phase3.safetensors" in phase
    assert "Refresh B from the trained" in phase

    assert "real_trainable_delta_l2" in merge
    assert "real_layer_deltas" in merge
    assert "Phase 3 claimed parameter updates but real trainable-weight delta is zero" in merge


def test_phase4_is_same_model_miss_only_and_answer_blind_pro():
    phase = _src("eval/phase4_pro_rsi.py")

    assert "model object was replaced/reloaded" in phase
    assert "backend.model = self.engine.model" in phase
    assert "backend.tokenizer = self.engine.tokenizer" in phase
    assert not _has_call("eval/phase4_pro_rsi.py", "load_model")
    assert "Phase 4 retests Phase-1 misses only" in phase
    assert 'cache.get(f"Phase 1: Baseline_{item[\'id\']}") is False' in phase

    assert "_choose_without_ground_truth" in phase
    assert "answer-blind consensus" in phase
    assert "low_threshold=0.25" in phase
    assert "high_threshold=0.70" in phase
    assert "pro_branches_mid=8" in phase
    assert "pro_branches_high=16" in phase
    assert "get_ladder_temperatures" in phase


def test_learning_retention_uses_model_not_kg_shortcut():
    merge = _src("eval/historical_good_merge.py")
    assert "no KG shortcut" in merge
    assert "State the exact learned fact directly." in merge
    assert "self._fast_generate(" in merge


def test_math_lcb_and_model_driven_scoring_repairs_remain():
    runtime = _src("eval/master_4000_runtime.py")
    assert "_repair_suite" in runtime
    assert 'splits.get("MATH-500", [])' in runtime
    assert "exponent = (i % 5) + 1" in runtime
    assert 'if "LiveCodeBench" in split:' in runtime
    assert "LCB prompt is natural language, not a Python stub" in runtime


def test_awake_learning_is_wired_for_normal_app_generation():
    hook = _src("core/awake_auto_hook.py")
    init = _src("core/__init__.py")
    assert "consolidator.check_and_prune(history)" in hook
    assert "stream_solve_with_awake_learning" in hook
    assert "solve_with_awake_learning" in hook
    assert "install_awake_auto_learning(ProReasoningEngine)" in init


def test_launcher_install_order_keeps_new_and_historical_layers():
    launcher = _src("master_4000_eval_suite.py")
    base = launcher.index("install(Master4000EvaluationEngine)")
    pro = launcher.index("phase4_pro_rsi.install(Master4000EvaluationEngine)")
    hist = launcher.index("install_historical_good_merge(phase4_pro_rsi)")
    assert base < pro < hist
