"""Model-free checks for native branch accounting at the temperature boundary."""
import threading
from types import SimpleNamespace

import pytest

from core.temperature_policy import install_chat, pro_ladder


def _engine(*, cancel_at=None, missing_at=None, fail_at=None):
    backend = SimpleNamespace(runtime="prism_speculative", last_branch_usage=None, last_generation={})

    class Engine:
        controller_backend = None
        gguf_backend = backend

        def __init__(self):
            self.calls = []

        def generate_parallel_branches(self, prompt, branch_count=16, history=None,
                                       temperatures=None, cancel_event=None):
            index = len(self.calls)
            self.calls.append((branch_count, list(temperatures)))
            if index == fail_at:
                raise RuntimeError("failed extra pass")
            if index == cancel_at:
                backend.last_generation = {"status": "cancelled"}
                backend.last_branch_usage = None
                return []
            backend.last_generation = {"status": "completed"}
            backend.last_branch_usage = (None if index == missing_at else {
                "prompt_tokens": 10 * branch_count,
                "completion_tokens": 2 * branch_count,
                "total_tokens": 12 * branch_count,
            })
            return ["answer"] * branch_count

        def solve(self):
            return "answer", {"raw_branches": ["a", "b", "c", "d"],
                              "temp_ladder": pro_ladder(3), "verified": True,
                              "winning_branch": 3, "entropy": None, "surprise_score": None}

    module = SimpleNamespace(ProReasoningEngine=Engine)
    install_chat(module)
    return Engine(), backend


def test_native_usage_includes_routed_and_fixed_anchor_passes():
    engine, backend = _engine()
    assert len(engine.generate_parallel_branches("question", branch_count=3)) == 4
    assert engine.calls == [(3, pro_ladder(3)[:3]), (1, [0.65])]
    assert backend.last_branch_usage == {"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48}


@pytest.mark.parametrize("missing_at", [0, 1])
def test_missing_native_usage_never_publishes_partial_search_totals(missing_at):
    engine, backend = _engine(missing_at=missing_at)
    engine.generate_parallel_branches("question", branch_count=3)
    assert not backend.last_branch_usage


@pytest.mark.parametrize("cancel_at,expected_calls", [(0, 1), (1, 2)])
def test_cancelled_native_search_does_not_restart_or_publish_stale_usage(cancel_at, expected_calls):
    engine, backend = _engine(cancel_at=cancel_at)
    backend.last_branch_usage = {"completion_tokens": 99999}
    engine.generate_parallel_branches("question", branch_count=3, cancel_event=threading.Event())
    assert len(engine.calls) == expected_calls
    assert not backend.last_branch_usage


def test_failed_native_anchor_does_not_publish_routed_usage():
    engine, backend = _engine(fail_at=1)
    with pytest.raises(RuntimeError, match="failed extra"):
        engine.generate_parallel_branches("question", branch_count=3)
    assert not backend.last_branch_usage


def test_unavailable_native_entropy_does_not_fabricate_surprise_score():
    engine, _ = _engine()
    _, metadata = engine.solve()
    assert metadata["branch_count"] == 4
    assert metadata["entropy"] is None
    assert metadata["surprise_score"] is None
