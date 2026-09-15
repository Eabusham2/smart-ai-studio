"""Regression coverage for memory-safe RSI generation with legacy search semantics intact."""
from __future__ import annotations

import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import eval.rsi_generation_memory_hardening as hard


ROOT = Path(__file__).resolve().parents[2]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_memory_policy_uses_existing_hardware_budget_without_reducing_output_limit(monkeypatch):
    monkeypatch.setattr(hard, "compute_auto_kv_budget", lambda: 2048)
    runner = SimpleNamespace(_phase4_pro_backend=None)

    normal = hard._memory_policy(runner, aggressive=False)
    retry = hard._memory_policy(runner, aggressive=True)

    assert normal == {
        "max_kv_size": 2048,
        "prefill_step_size": 256,
        "kv_bits": 4,
        "kv_group_size": 64,
        "quantized_kv_start": 512,
    }
    assert retry["max_kv_size"] == 1024
    assert retry["prefill_step_size"] == 128
    assert retry["kv_bits"] == 4

    source = _src("eval/rsi_generation_memory_hardening.py")
    assert '"max_tokens": max(1, int(max_tokens))' in source
    assert "16,384-token allowance unchanged" in source
    assert "max_tokens=min" not in source


def test_metal_oom_detection_is_specific():
    assert hard._is_metal_oom(RuntimeError("[METAL] Command buffer execution failed: Insufficient Memory"))
    assert hard._is_metal_oom(RuntimeError("kIOGPUCommandBufferCallbackErrorOutOfMemory"))
    assert not hard._is_metal_oom(RuntimeError("ordinary verifier failure"))


def _fake_modules(monkeypatch, *, oom_first=False):
    calls = []
    clear_calls = []

    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda temp, top_p: (float(temp), float(top_p))
    pkg = types.ModuleType("mlx_lm")
    pkg.__path__ = []
    pkg.sample_utils = sample_utils
    monkeypatch.setitem(sys.modules, "mlx_lm", pkg)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)

    class FakeMX:
        @staticmethod
        def clear_cache():
            clear_calls.append(True)

    class FakeLM:
        @staticmethod
        def stream_generate(model, tokenizer, **kwargs):
            calls.append(dict(kwargs))
            call_index = len(calls)

            def iterator():
                if oom_first and call_index == 1:
                    raise RuntimeError("[METAL] Command buffer execution failed: Insufficient Memory")
                yield SimpleNamespace(text=f"branch-{call_index}")

            return iterator()

    p4 = SimpleNamespace(
        MLX_AVAILABLE=True,
        mx=FakeMX(),
        mlx_lm=FakeLM(),
        METAL_STREAM_LOCK=threading.RLock(),
        _generate_branches_same_model=lambda *args, **kwargs: ["legacy"],
    )
    live = SimpleNamespace(
        _write_live_header=lambda *args, **kwargs: None,
        _append_live_text=lambda *args, **kwargs: None,
    )
    runner = SimpleNamespace(
        engine=SimpleNamespace(model=object(), tokenizer=object()),
        _phase4_pro_backend=None,
    )
    return p4, live, runner, calls, clear_calls


def test_streamed_rsi_keeps_full_16384_allowance_and_tears_down_each_branch(monkeypatch):
    monkeypatch.setattr(hard, "compute_auto_kv_budget", lambda: 2048)
    p4, live, runner, calls, clear_calls = _fake_modules(monkeypatch)
    legacy = p4._generate_branches_same_model

    hard.install(p4, live, legacy)
    out = p4._generate_branches_same_model(
        runner,
        "prompt",
        [0.20, 0.38],
        16384,
        0.92,
    )

    assert out == ["branch-1", "branch-2"]
    assert len(calls) == 2
    assert all(call["max_tokens"] == 16384 for call in calls)
    assert all(call["max_kv_size"] == 2048 for call in calls)
    assert all(call["prefill_step_size"] == 256 for call in calls)
    assert all(call["kv_bits"] == 4 for call in calls)
    # At least one pre + one post clear for each sequential branch.
    assert len(clear_calls) >= 4


def test_only_failed_branch_retries_on_metal_oom_with_smaller_kv_not_fewer_tokens(monkeypatch):
    monkeypatch.setattr(hard, "compute_auto_kv_budget", lambda: 2048)
    p4, live, runner, calls, _ = _fake_modules(monkeypatch, oom_first=True)
    legacy = p4._generate_branches_same_model

    hard.install(p4, live, legacy)
    out = p4._generate_branches_same_model(
        runner,
        "prompt",
        [0.58],
        16384,
        0.92,
    )

    assert out == ["branch-2"]
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == calls[1]["max_tokens"] == 16384
    assert calls[0]["max_kv_size"] == 2048
    assert calls[1]["max_kv_size"] == 1024
    assert calls[0]["prefill_step_size"] == 256
    assert calls[1]["prefill_step_size"] == 128


def test_corrected_legacy_rsi_search_algorithm_is_still_the_current_core():
    phase = _src("eval/phase4_pro_rsi.py")
    assert "for round_idx in (1, 2):" in phase
    assert "temps = [0.20, 0.38, 0.58, 0.82]" in phase
    assert "_choose_without_ground_truth" in phase
    assert "_hidden_reward_only_after_selection" in phase
    assert "Critique your own attempt" in phase
    assert "Do not assume or request a hidden answer" in phase
    assert "if cache.get(key) is False:" in phase
    assert "for split_name, item in misses[:64]:" in phase


def test_memory_layer_wraps_live_stream_after_capturing_legacy_generator():
    launcher = _src("master_4000_eval_suite.py")
    capture = launcher.index("_legacy_rsi_branch_generate = phase4_pro_rsi._generate_branches_same_model")
    live = launcher.index("install_phase4_stream(phase4_pro_rsi)")
    memory = launcher.index("install_rsi_generation_memory_hardening(")
    miss_only = launcher.index("capture_before_historical_merge(phase4_pro_rsi)")
    assert capture < live < memory < miss_only

    source = _src("eval/rsi_generation_memory_hardening.py")
    assert "legacy_generate(" in source
    assert "_close_iterator(iterator)" in source
    assert "METAL_STREAM_LOCK" in source
    assert "retrying this branch" in source
