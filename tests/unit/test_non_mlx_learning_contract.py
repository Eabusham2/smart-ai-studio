from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_bonsai_gguf_presets_pin_prism_ternary_and_training_lineage():
    from config.bonsai_models import bonsai_model_info
    app = _src("app_gui.py")
    for crack in (False, True):
        info = bonsai_model_info(crack=crack, is_apple_silicon=False)
        assert info["gguf_file"].endswith("PTQ1_0-mtp.gguf")
        assert info["prism_llama_fork"]
        assert info["gguf_training_base_model_id"] == "Qwen/Qwen3.8-27B"
        assert info["mmproj_file"] == "Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf"
    assert "allow_patterns=primary_patterns" in app
    assert "required_files=primary_required" in app


def test_gguf_learning_is_real_and_reloaded_into_inference():
    engine = _src("core/engines/gguf_engine.py")
    trainer = _src("core/gguf_lora_trainer.py")
    prism = _src("core/engines/prism_gguf_engine.py")
    worker = _src("core/engines/prism_gguf_worker.py")

    assert "def train_mini_batch(" in engine
    assert "GGUFLoRATrainer" in engine
    assert "lora_path=self.adapter_path or None" in engine
    assert "update_lora(" in trainer
    optimizer = _src("core/torch_context_learning.py")
    assert "loss.backward()" in optimizer
    assert "optimizer.step()" in optimizer
    assert "convert_lora_to_gguf.py" in trainer
    assert "zero parameter change" in optimizer

    assert "PrismGGUFReasoningBackend" in prism
    assert "PRISM_REPO" in prism
    assert "core.engines.prism_gguf_worker" in prism
    assert 'lora_path=cfg.get("adapter_path")' in worker


def test_ordinary_chat_consolidation_is_backend_capability_driven():
    online = _src("core/online_consolidator.py")
    awake = _src("core/awake_auto_hook.py")
    pro = _src("core/pro_engine.py")

    assert 'callable(getattr(self.engine, "train_mini_batch", None))' in online
    assert "success = self.consolidate_chunk_sync(evicted_chunk)" in online
    assert "(retained_history, True) if success else (active, False)" in online

    assert "def _active_trainable_backend(engine)" in awake
    assert 'getattr(engine, "controller_backend", None)' in awake
    assert 'getattr(engine, "gguf_backend", None)' in awake
    assert 'getattr(engine, "bitnet_backend", None)' in awake

    assert "self.awake_consolidator.engine = self.gguf_backend" in pro
    assert "self.awake_consolidator.engine = self.bitnet_backend" in pro
    assert "self.awake_consolidator.engine = self.controller_backend" in pro


def test_learn_and_rsi_use_active_real_trainable_backend():
    learner = _src("core/autonomous_learner.py")
    assert "def _require_live_trainable_backend" in learner
    assert 'getattr(self.engine, "gguf_backend", None)' in learner
    assert 'getattr(self.engine, "bitnet_backend", None)' in learner
    assert 'getattr(self.engine, "controller_backend", None)' in learner
    assert "service.train_pairs(" in learner
    assert "def recursive_self_improve(" in learner
    assert "def consolidate_rsi_parameters(" in learner
    assert "validation_callback=validate" in _src("core/online_consolidator.py")


def test_bitnet_training_rebuilds_and_hot_reloads_real_weights():
    backend = _src("core/engines/bitnet_cpp_engine.py")
    trainer = _src("core/bitnet_rebuild_trainer.py")
    pro = _src("core/pro_engine.py")

    assert "def training_ready(self) -> bool:" in backend
    assert "def train_mini_batch(" in backend
    assert "BitNetRebuildTrainer" in backend
    assert "learned-i2_s.gguf" in backend
    assert "update_deployment(" in backend
    transaction = _src("core/native_context_learning.py")
    assert "backend.unload_model()" in transaction
    assert "backend.load_model()" in transaction

    assert "PeftModel.from_pretrained" in trainer
    assert "get_peft_model" in trainer
    assert "merge_and_unload" in trainer
    assert "preprocess-huggingface-bitnet.py" in trainer
    assert "convert-ms-to-gguf-bitnet.py" in trainer
    assert '"I2_S"' in trainer
    assert "update_lora(" in trainer
    optimizer = _src("core/torch_context_learning.py")
    assert "loss.backward()" in optimizer
    assert "optimizer.step()" in optimizer
    assert "zero parameter change" in optimizer

    assert 'training_base_model_id=info.get("bitnet_training_base_model_id")' in pro
    assert '"trainable": bool(self.bitnet_backend.training_ready())' in pro


def test_custom_metadata_drives_gguf_bitnet_and_multimodal_routing():
    policy = _src("core/model_policy.py")
    runtime = _src("core/controller_runtime.py")
    app = _src("app_gui.py")
    from core.model_import_entry import make_imported_model_entry
    metadata = {
        "task": "text", "backend_family": "gguf", "controller_runtime": "gguf",
        "runtime_family": "custom", "prism_llama_fork": False,
        "gguf_training_base_model_id": "owner/gguf-master",
        "bitnet_training_base_model_id": "owner/bitnet-master",
    }
    entry = make_imported_model_entry(metadata, repo_id="owner/custom", runtime_preference="native")
    assert "make_imported_model_entry(" in app

    for key in (
        "backend_family",
        "controller_runtime",
        "runtime_family",
        "prism_llama_fork",
        "gguf_training_base_model_id",
        "bitnet_training_base_model_id",
    ):
        assert key in policy
        assert entry[key] == metadata[key]

    assert "supports_media_input" in runtime
    assert "review_media_input" in runtime
    from config.bonsai_models import bonsai_model_info
    assert "bonsai_model_info(" in app
    for apple in (False, True):
        for crack in (False, True):
            info = bonsai_model_info(is_apple_silicon=apple, crack=crack)
            assert info["input_modalities"] == ["text", "image", "video"]
