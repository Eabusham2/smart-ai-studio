from eval.dataset_hardening import (
    _harden_aime,
    _harden_gpqa,
    _harden_hle,
    _harden_lcb_fallback,
    _harden_mmlu,
    _harden_zebra,
)


def test_synthetic_lcb_fallback_gets_real_exact_tests():
    splits = {
        "LiveCodeBench-Hard": [
            {
                "id": "LCB_Hard_7",
                "entry_point": "min_operations_7",
                "prompt": "Write a Python function `min_operations_7(arr)` that returns the minimum operations to sort array with shift step 8.",
                "test": "assert min_operations_7([3,1,2]) >= 0\n",
            }
        ]
    }
    item = _harden_lcb_fallback(splits)["LiveCodeBench-Hard"][0]
    assert "minimum number of adjacent swaps" in item["prompt"]
    assert "inversion count" in item["prompt"]
    assert ">= 0" not in item["test"]
    assert "assert min_operations_7([3, 1, 2]) == 2" in item["test"]
    assert "assert min_operations_7([4, 3, 2, 1]) == 6" in item["test"]
    assert "assert min_operations_7([2, 1, 1]) == 2" in item["test"]


def test_real_or_unknown_lcb_item_is_not_rewritten():
    item = {
        "id": "real-123",
        "entry_point": "solve",
        "prompt": "Solve the official contest problem exactly.",
        "test": "assert solve() == 42\n",
    }
    splits = {"LiveCodeBench-Hard": [item.copy()]}
    out = _harden_lcb_fallback(splits)
    assert out["LiveCodeBench-Hard"][0] == item


def test_aime_polynomial_is_determined_and_expected_matches_linear_interpolation():
    item = {"id": "AIME_0", "prompt": "bad", "expected": "bad"}
    out = _harden_aime({"AIME-150": [item]})["AIME-150"][0]
    assert "linear polynomial" in out["prompt"]
    # i=0: P(1)=3, P(2)=28, slope=25, so P(5)=103.
    assert out["expected"] == "103"


def test_gpqa_prompt_contains_the_fact_needed_for_its_option():
    items = [{"id": f"GPQA_{i}", "prompt": "bad", "expected": "?"} for i in range(4)]
    out = _harden_gpqa({"GPQA-400": items})["GPQA-400"]
    assert [x["expected"] for x in out] == ["A", "B", "C", "D"]
    assert "commutes with the parity operator" in out[0]["prompt"]
    assert "opposite-parity states" in out[1]["prompt"]
    assert "energy-degenerate" in out[2]["prompt"]
    assert "decays to zero" in out[3]["prompt"]


def test_mmlu_each_expected_option_has_a_corresponding_explicit_premise():
    items = [{"id": f"MMLU_Pro_{i}", "prompt": "bad", "expected": "?"} for i in range(4)]
    out = _harden_mmlu({"MMLU-Pro-1000": items})["MMLU-Pro-1000"]
    assert [x["expected"] for x in out] == ["A", "D", "C", "B"]
    for item in out:
        assert "Options:" in item["prompt"]
        assert "Return the valid option" in item["prompt"]


def test_zebra_target_is_logically_forced():
    items = [{"id": f"Zebra_{i}", "prompt": "bad", "expected": "?"} for i in range(4)]
    out = _harden_zebra({"ZebraLogic-200": items})["ZebraLogic-200"]
    assert [x["expected"] for x in out] == ["1", "2", "3", "4"]
    assert "Green is house 2" in out[0]["prompt"]
    assert "Red is directly left of Green" in out[0]["prompt"]


def test_hle_consistency_notation_contains_required_axiom_level():
    items = [{"id": f"HLE_{i}", "prompt": "bad", "expected": "?"} for i in range(3)]
    out = _harden_hle({"HLE-100": items})["HLE-100"]
    assert [x["expected"] for x in out] == [
        "Con(ZFC + I0)",
        "Con(ZFC + I1)",
        "Con(ZFC + I2)",
    ]
    assert "T = ZFC + I0" in out[0]["prompt"]
