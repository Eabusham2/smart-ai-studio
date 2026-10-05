"""Contracts for behavior reconciled from old diverged branches.

These tests intentionally lock only the useful pieces we preserved. They also lock
the newer policies that must NOT regress (question-free RSI memory, full-precision
KV, and disabled speculative decoding until equivalence is proven).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _section(source: str, start: str, end: str) -> str:
    a = source.index(start)
    b = source.index(end, a)
    return source[a:b]


def test_rsi_round2_keeps_answer_blind_feedback_without_old_persistence():
    phase = _src("eval/phase4_pro_rsi.py")
    rsi = _section(
        phase,
        "def _run_rsi_self_improvement",
        "def _fetch_benchmark_training_memories",
    )
    assert "Answer-blind deterministic verifier feedback from that attempt" in rsi
    assert "_answer_blind_failure_feedback(" in rsi
    assert '_assert_same_model(self, model_identity, f"RSI round {round_idx}")' in rsi
    assert "branches.clear()" in rsi
    assert "log_rsi_self_memory(candidate)" in rsi
    assert "log_interaction(" not in rsi


def test_rsi_memory_keeps_newer_full_precision_policy_plus_safe_sync():
    source = _src("eval/rsi_generation_memory_hardening.py")
    cleanup = _section(source, "def _clear_runtime_memory", "def _memory_policy")
    assert 'sync = getattr(mx, "synchronize", None)' in cleanup
    assert "if callable(sync):" in cleanup
    assert "mx.clear_cache()" in cleanup
    assert "KV=full" in source
    assert "full-precision KV" in source
    # Old branch's lossy/quantized fallback must stay retired.
    assert "kv_bits=" not in source
    assert "quantized_kv_start=" not in source


def test_recovered_drafter_is_diagnostic_only_and_does_not_enable_speculation():
    drafter = _src("core/drafter.py")
    speculative = _src("core/speculative_engine.py")
    assert "class DraftTelemetry" in drafter
    assert "class GrammarGuidedASTTrieDrafter" in drafter
    assert "ASTPrefixTrieDrafter = GrammarGuidedASTTrieDrafter" in drafter
    assert "def get_telemetry" in drafter
    assert "Speculative decoding is disabled" in speculative
    assert "def propose_draft_tokens" in speculative
    assert "return []" in speculative


def test_recovered_lif_helpers_do_not_replace_current_runtime_policy():
    lif = _src("core/lif_gating.py")
    pro = _src("core/pro_engine.py")
    assert "def convex_temperature_ladder" in lif
    assert "def normalized_shannon_from_probabilities" in lif
    assert "def topk_shannon_from_logits" in lif
    assert "def spike_count" in lif
    # Current calibrated Pro policy remains owned by pro_engine/temperature policy.
    assert "def get_ladder_temperatures" in pro
    assert "gamma: float = 1.35" in pro


def test_projected_daemon_gains_lifecycle_telemetry_without_old_architecture_rewire():
    daemon = _src("consolidation/projected_daemon.py")
    pro = _src("core/pro_engine.py")
    assert "def stop(self)" in daemon
    assert "def status(self)" in daemon
    assert "self.last_error" in daemon
    assert "self.last_update_time" in daemon
    assert "gc.collect(2)" in daemon
    assert "AwakeOnlineConsolidator" in pro
    # Do not revive the old MLX-only wrapper architecture.
    assert "_LIFRouterAdapter" not in pro
    assert "_MemoryReplayBridge" not in pro
