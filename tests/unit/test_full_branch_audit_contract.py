"""Full-branch audit contracts.

Locks concrete regressions found while auditing fix/real-benchmarks-final-32k
against master and the historical Gemini handoff. These are source contracts, not
claims that heavyweight native backends were executed on every target machine.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_eval_has_no_fabricated_speed_or_speculative_fallbacks():
    base = _src("eval/_master_4000_base.py")
    runtime = _src("eval/master_4000_runtime.py")
    assert "self.last_tok_per_sec = 12.0" not in base
    assert "getattr(self, 'last_tok_per_sec', 15.0)" not in base
    assert "spec_rate=42.5" not in base
    assert "[Offline:" not in base
    assert "refusing synthetic/offline benchmark output" in base
    assert "process_memory_bytes()" in runtime


def test_cross_platform_eval_imports_do_not_require_mlx_at_module_import():
    live = _src("eval/live_generation_stream.py")
    rsi = _src("eval/rsi_generation_memory_hardening.py")
    assert "try:\n    from mlx_lm.models.cache import make_prompt_cache" in live
    assert "make_prompt_cache = None" in live
    assert "try:\n    from mlx_lm.models.cache import make_prompt_cache" in rsi
    assert "make_prompt_cache = None" in rsi


def test_app_eval_adapter_satisfies_legacy_engine_surface_and_gguf_teach_identity():
    bridge = _src("eval/app_cross_platform_bridge.py")
    teach = _src("eval/conversation_teach_hardening.py")
    assert "NeuromorphicLIFController" in bridge
    assert "self.lif = NeuromorphicLIFController()" in bridge
    assert 'if getattr(backend, "model", None) is not self.engine.model:' in teach
    assert 'if getattr(backend, "tokenizer", None) is not self.engine.tokenizer:' in teach


def test_one_context_budget_controls_generation_for_every_text_backend():
    awake = _src("core/awake_auto_hook.py")
    assert "def _remaining_generation_tokens" in awake
    assert "Packed prompt requires" in awake
    assert "previous_max = getattr(self.settings, "max_new_tokens", None)" in awake
    assert "self.settings.max_new_tokens = int(remaining)" in awake
    assert "self.settings.max_new_tokens = previous_max" in awake


def test_eval_memory_limit_uses_mac_physical_footprint_when_available():
    memory = _src("core/training_memory.py")
    gui = _src("core/gui_eval_panel.py")
    runner = _src("eval/app_eval_runner.py")
    assert "def process_memory_bytes" in memory
    assert "ri_phys_footprint" in memory
    assert "process_memory_bytes(item.pid)" in gui
    assert "process_memory_bytes(proc.pid)" in runner


def test_mlx_lora_is_adapter_only_and_transactional():
    src = _src("core/_mlx_engine_base.py")
    assert src.count('unfreeze(keys=["lora_a", "lora_b"], recurse=False)') >= 5
    assert "MLX training completed but measured zero parameter change" in src
    assert "rollback_params" in src
    assert ".next.safetensors" in src
    assert ".previous" in src
    assert "self.model.update(" in src


def test_non_mlx_training_and_cancel_paths_roll_back():
    gguf = _src("core/engines/gguf_engine.py")
    bitnet = _src("core/engines/bitnet_cpp_engine.py")
    controller = _src("core/controller_runtime.py")
    gguf_trainer = _src("core/gguf_lora_trainer.py")
    bitnet_trainer = _src("core/bitnet_rebuild_trainer.py")
    bridge = _src("eval/app_cross_platform_bridge.py")

    assert "except BaseException:" in gguf
    assert "except BaseException:" in bitnet
    assert "except BaseException:" in gguf_trainer
    assert "except BaseException:" in bitnet_trainer
    assert "before = {" in controller
    assert "model.save_pretrained(tmp" in controller
    assert "with torch.no_grad():" in controller
    persisted = bridge.index("persisted = bool(adapter_path")
    mark = bridge.index("mark_consolidated(learn_ids)")
    assert persisted < mark


def test_media_learning_never_equates_file_existence_with_real_learning():
    media = _src("core/media_learning.py")
    assert "def _persisted_lora_delta_proven" in media
    assert "real nonzero LoRA update factor could not be proven" in media
    assert "delta_sq <= 0" in media
    assert "except BaseException:" in media


def test_swe_patch_verification_is_portable():
    src = _src("eval/swe_verifier_hardening.py")
    assert 'platform.system() == "Windows"' in src
    assert 'shutil.which("patch")' in src
    assert 'shutil.which("git")' in src
    assert '"apply", "--check"' in src


def test_report_target_model_source_is_valid_python_not_literal_escape():
    src = _src("eval/_master_4000_base.py")
    assert '_eval_target_model_label' in src
    assert '))\\n        md.append' not in src


def test_cross_platform_eval_uses_native_token_counting_before_fallback():
    bridge = _src("eval/app_cross_platform_bridge.py")
    assert 'tokenize = getattr(tokenizer, "tokenize", None)' in bridge
    assert 'counter = getattr(backend, "count_tokens", None)' in bridge
    assert 'backend=getattr(self.engine, "backend", None)' in bridge


def test_non_mlx_phase3_marks_learn_and_rsi_atomically():
    bridge = _src("eval/app_cross_platform_bridge.py")
    persisted = bridge.index("persisted = bool(adapter_path")
    transaction = bridge.index("with sqlite3.connect(db_path) as conn:")
    learn_update = bridge.index("UPDATE episodic_interactions SET consolidated=1")
    rsi_update = bridge.index("UPDATE rsi_self_memories SET consolidated=1")
    assert persisted < transaction < learn_update < rsi_update


def test_rollback_backups_survive_until_successful_return():
    controller = _src("core/controller_runtime.py")
    gguf = _src("core/engines/gguf_engine.py")
    bitnet = _src("core/engines/bitnet_cpp_engine.py")
    for source in (controller, gguf, bitnet):
        assert "success = False" in source
        assert "success = True" in source
        assert "except BaseException:" in source
        assert "finally:" in source


def test_eval_memory_guard_interrupts_main_thread_portably():
    runner = _src("eval/app_eval_runner.py")
    assert "import _thread" in runner
    assert "_thread.interrupt_main()" in runner
