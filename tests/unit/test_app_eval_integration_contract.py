"""Source contracts for the additive desktop Eval integration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_eval_button_is_text_only_and_cancel_is_confirmed():
    src = _src("core/gui_eval_panel.py")
    assert 'text="Eval"' in src
    assert 'model_type", "text"' in src
    assert "btn.pack_forget()" in src
    assert 'messagebox.askyesno(' in src
    assert '"Cancel Evaluation"' in src
    assert "pause.flag" in src
    assert "CTRL_BREAK_EVENT" in src
    assert "signal.SIGINT" in src


def test_eval_window_has_live_output_and_process_tree_memory_limit():
    src = _src("core/gui_eval_panel.py")
    runner = _src("eval/app_eval_runner.py")
    assert "os.read(fd, 4096)" in src
    assert "getincrementaldecoder" in src
    assert "children(recursive=True)" in src
    assert "memory_limit_gb" in src
    assert "SmartAI-EvalMemoryGuard" in runner
    assert "root.children(recursive=True)" in runner
    assert "memory_limit_enabled" in runner


def test_app_eval_reuses_production_backends_and_fails_closed_on_fake_training():
    bridge = _src("eval/app_cross_platform_bridge.py")
    assert "ProReasoningEngine" in bridge
    assert "self.pro.load_model(" in bridge
    assert "backend.stream_generate_tokens(" in bridge
    assert "backend.generate_branches(" in bridge
    assert "training_ready" in bridge
    assert "train_mini_batch" in bridge
    assert "zero parameter drift" in bridge
    assert "no persisted adapter/model artifact" in bridge
    assert "_backend_context_limit" in bridge
    assert "_clamp_output_to_backend" in bridge


def test_eval_state_is_isolated_but_native_runtime_cache_remains_shared():
    runner = _src("eval/app_eval_runner.py")
    bridge = _src("eval/app_cross_platform_bridge.py")
    gguf = _src("core/engines/gguf_engine.py")
    bitnet = _src("core/engines/bitnet_cpp_engine.py")
    controller = _src("core/controller_runtime.py")

    assert 'os.environ["SMARTAI_DATA_DIR"]' not in runner
    assert "SMARTAI_GGUF_ADAPTER_ROOT" in runner
    assert "SMARTAI_BITNET_TRAINING_ROOT" in runner
    assert "SMARTAI_CONTROLLER_ADAPTER_ROOT" in runner
    assert 'backend_state", "mlx", "adapters.safetensors' in bridge
    assert "SMARTAI_GGUF_ADAPTER_ROOT" in gguf
    assert "SMARTAI_BITNET_TRAINING_ROOT" in bitnet
    assert "SMARTAI_CONTROLLER_ADAPTER_ROOT" in controller


def test_canonical_suite_order_and_standalone_packaging_are_preserved():
    suite = _src("master_4000_eval_suite.py")
    build = _src("build_app.py")
    bridge = suite.index("app_cross_platform_bridge.install(master_runtime")
    integrity = suite.index("install_stage_integrity_telemetry(phase4_pro_rsi")
    legacy = suite.index("install_rsi_legacy_training_hardening(phase4_pro_rsi")
    real_data = suite.index("real_benchmark_runtime.install(BenchmarkDatasetProvider")
    context = suite.index("unified_context_budget.install(")
    assert bridge < integrity < legacy < real_data < context
    assert build.count('"eval"') >= 3
    assert '"master_4000_eval_suite.py"' in build
    assert '"run_studio_complete.py"' in build


def test_bonsai2_is_the_default_eval_model_and_current_model_is_reported():
    runtime = _src("run_studio_complete.py")
    report = _src("eval/_master_4000_base.py")
    assert 'mlx_model_path: str="prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"' in runtime
    assert "_eval_target_model_label" in report
    assert "Bonsai 2 27B Ternary Multimodal" in report


def test_top_app_controls_explicitly_include_context_limit_and_memory_watcher():
    app = _src("app_gui.py")
    context = _src("core/gui_generation_cap.py")
    eval_ui = _src("core/gui_eval_panel.py")

    # One existing app Context control + one existing top-app memory control.
    # Eval gets its own memory watcher but must not clone the normal app controls.
    assert app.count("install_gui_generation_cap()") == 1
    assert context.count('text="Context Limit"') == 1
    assert app.count('text="Mem Limit"') == 1
    assert 'text="Context Limit"' not in eval_ui
    assert 'text="Mem Limit"' not in eval_ui

    assert "self.memory_limit_var = tk.BooleanVar" in app
    assert "self.entry_memory_limit.pack(" in app
    assert 'text="GB"' in app
    assert "tk.Toplevel(self.root)" in eval_ui
