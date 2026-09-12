from eval.dataset_hardening import _harden_lcb_fallback


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
    out = _harden_lcb_fallback(splits)
    item = out["LiveCodeBench-Hard"][0]
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
