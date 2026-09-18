from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_temperature_constants_are_pinned():
    src = _src("core/temperature_policy.py")
    assert "CHAT_N1_TEMPERATURE = 0.65" in src
    assert "EVAL_N1_TEMPERATURE = 0.55" in src
    assert "FIXED_PRO_EXTRA_TEMPERATURE = 0.65" in src
    assert "PRO_T_MIN = 0.20" in src
    assert "PRO_T_MAX = 0.95" in src
    assert "PRO_GAMMA = 1.35" in src


def test_core_chat_defaults_cannot_drift_back_to_old_values():
    src = _src("core/pro_engine.py")
    assert "t_max: float = 0.95" in src
    assert "return [0.65]" in src
    assert "temperature: float = 0.65" in src
    assert "t_max: float = 0.88" not in src


def test_chat_policy_installs_before_awake_stream_wrapper():
    src = _src("core/__init__.py")
    temp = src.index("install_chat_temperature_policy(pro_engine_module)")
    awake = src.index("install_awake_auto_learning(ProReasoningEngine)")
    assert temp < awake


def test_phase1_eval_uses_tuned_single_pass_after_legacy_live_wrapper():
    launcher = _src("master_4000_eval_suite.py")
    live = launcher.index("install_baseline_stream(master_runtime, Master4000EvaluationEngine)")
    tuned = launcher.index(
        "lossless_baseline_speedup.install(master_runtime, live_generation_stream, Master4000EvaluationEngine)"
    )
    assert live < tuned

    speedup = _src("eval/lossless_baseline_speedup.py")
    assert "from core.temperature_policy import EVAL_N1_TEMPERATURE" in speedup
    assert "make_sampler(temp=EVAL_N1_TEMPERATURE, top_p=0.92)" in speedup
