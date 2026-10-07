"""Regression coverage for memory-safe RSI generation with old branch isolation retained."""
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


def test_memory_policy_keeps_full_precision_kv_and_changes_only_prefill():
    normal = hard._memory_policy(aggressive=False)
    retry = hard._memory_policy(aggressive=True)

    assert set(normal) == {"prefill_step_size"}
    assert set(retry) == {"prefill_step_size"}
    assert retry["prefill_step_size"] <= normal["prefill_step_size"]

    source = _src("eval/rsi_generation_memory_hardening.py")
    assert "KV=full" in source
    assert "full-precision KV" in source
    assert '"max_kv_size":' not in source
    assert "kv_bits" not in source
    assert "quantized_kv_start" not in source
    assert '"max_tokens": max(1, int(max_tokens))' in source
    assert "token allowance are unchanged" in source

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
    tokenizer = SimpleNamespace(encode=lambda text: str(text).split())
    runner = SimpleNamespace(engine=SimpleNamespace(model=object(), tokenizer=tokenizer))
    monkeypatch.setattr(hard, "make_prompt_cache", lambda model: object())
    return p4, live, runner, calls, clear_calls


def test_streamed_rsi_keeps_full_allowance_and_full_precision_kv(monkeypatch):
    p4, live, runner, calls, _ = _fake_modules(monkeypatch)
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
    assert all("max_kv_size" not in call for call in calls)
    assert all("kv_bits" not in call for call in calls)
    assert all("quantized_kv_start" not in call for call in calls)
    assert all(
        call["prefill_step_size"] == hard._memory_policy(aggressive=False)["prefill_step_size"]
        for call in calls
    )

def test_only_failed_branch_retries_on_metal_oom_with_smaller_prefill(monkeypatch):
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
    assert "kv_bits" not in calls[0] and "kv_bits" not in calls[1]
    assert "max_kv_size" not in calls[0] and "max_kv_size" not in calls[1]
    assert calls[0]["prefill_step_size"] == hard._memory_policy(aggressive=False)["prefill_step_size"]
    assert calls[1]["prefill_step_size"] == hard._memory_policy(aggressive=True)["prefill_step_size"]
    assert calls[1]["prefill_step_size"] <= calls[0]["prefill_step_size"]

def test_current_rsi_search_reward_semantics_are_not_rewritten():
    phase = _src("eval/phase4_pro_rsi.py")
    assert "for round_idx in (1, 2):" in phase
    assert "temps = [0.20, 0.38, 0.58, 0.82]" in phase
    assert "_choose_without_ground_truth" in phase
    assert "_hidden_reward_only_after_selection" in phase
    assert "Critique your own attempt" in phase
    assert "Do not assume or request a hidden answer" in phase
    assert "if cache.get(key) is False:" in phase
    assert "for split_name, item in misses[:64]:" in phase


def test_memory_layer_wraps_live_stream_after_capturing_old_direct_generator():
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
    assert "retrying only this branch" in source
