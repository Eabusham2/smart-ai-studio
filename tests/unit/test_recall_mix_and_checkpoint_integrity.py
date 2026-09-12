"""Regression coverage for mixed DialogueRecall and marker-only checkpoint recovery."""

import eval.master_4000_runtime as runtime
import eval.phase4_pro_rsi as phase4
import master_4000_eval_suite  # installs dataset/historical/checkpoint hardening
from eval._master_4000_base import BenchmarkDatasetProvider
from eval.dataset_hardening import PROJECT_RECALL_PROBES
from eval.flagship_benchmarks import EPISODIC_DIALOGUE_RECALL_PROBE


def test_dialogue_recall_150_mixes_project_and_original_historical_styles():
    splits = runtime._repair_suite(BenchmarkDatasetProvider().load_all_4000_items())
    recall = splits["DialogueRecall-150"]

    assert len(recall) == 150
    assert {item.get("recall_style") for item in recall} == {"project", "historical"}

    pairs = {(item["prompt"], item["expected_keyword"]) for item in recall}
    assert len(pairs) == len(PROJECT_RECALL_PROBES) + len(EPISODIC_DIALOGUE_RECALL_PROBE) == 17

    learned = {(str(q), str(a)) for q, a in phase4.LEARN_EXAMPLES}
    assert pairs.issubset(learned)

    assert any(item["prompt"] == "What DNS service runs on the ASUS ROG GT-BE19000?" for item in recall)
    assert any(
        item["prompt"] == "What IPC architecture was chosen during Session A for zero-copy message exchange?"
        for item in recall
    )
    assert any("token TTL" in item["prompt"] for item in recall)


def test_marker_without_boolean_result_is_rerun_not_silently_skipped():
    key = "Phase 1: Baseline_Dialogue_0"

    assert runtime._completed({key: True}, key) is True
    assert runtime._completed({key: False}, key) is True

    for prefix in ("__v2done__:", "__eyad_v3_done__:", "__eyad_v5_done__:"):
        assert runtime._completed({prefix + key: True}, key) is False
