from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_bonsai_non_mlx_artifacts_are_exact_ternary_and_prism_only():
    from config.bonsai_models import bonsai_model_info
    app = _src("app_gui.py")
    assert '"model_1": bonsai_model_info(is_apple_silicon=is_apple_silicon)' in app
    assert '"model_2": bonsai_model_info(crack=True, is_apple_silicon=is_apple_silicon)' in app
    for crack in (False, True):
        mlx = bonsai_model_info(crack=crack)
        gguf = bonsai_model_info(crack=crack, is_apple_silicon=False)
        assert mlx["repo_id"] != gguf["repo_id"]
        assert mlx["gguf_file"] is None
        assert gguf["gguf_file"].endswith("PTQ1_0-mtp.gguf")
        assert gguf["prism_llama_fork"] and gguf["ternary"]
        assert gguf["mmproj_file"] == "Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf"
        assert gguf["gguf_training_base_model_id"] == "Qwen/Qwen3.8-27B"


def test_prism_runtime_isolated_from_universal_gguf():
    pro = _src("core/pro_engine.py")
    prism = _src("core/engines/prism_gguf_engine.py")
    worker = _src("core/engines/prism_gguf_worker.py")

    assert "from core.engines.prism_gguf_engine import PrismGGUFReasoningBackend" in pro
    assert 'if bool(info.get("prism_llama_fork"))' in pro
    assert "else GGUFReasoningBackend" in pro

    assert "PRISM_REPO = \"https://github.com/PrismML-Eng/llama.cpp.git\"" in prism
    assert 'env["LLAMA_CPP_LIB_PATH"]' in prism
    assert 'env["MTMD_CPP_LIB"]' in prism
    assert "core.engines.prism_gguf_worker" in prism
    assert "from llama_cpp import Llama" in worker
    assert "lora_path=" in worker
    assert "MTMDChatHandler" in worker


def test_gguf_always_on_consolidation_and_learning_are_real():
    awake = _src("core/awake_auto_hook.py")
    online = _src("core/online_consolidator.py")
    learner = _src("core/autonomous_learner.py")
    gguf = _src("core/engines/gguf_engine.py")

    assert "def _active_trainable_backend" in awake
    assert '"gguf": getattr(engine, "gguf_backend", None)' in awake
    assert "backend = named.get(active)" in awake
    assert "consolidator.consolidate_chunk_sync(chunk)" in awake
    assert "if not learned:" in awake

    assert "def _real_training_ready" in online
    assert "success = self.consolidate_chunk_sync(evicted_chunk)" in online
    assert "(retained_history, True) if success else (active, False)" in online

    assert "_require_live_trainable_backend" in learner
    assert "service.train_pairs(" in learner
    assert "backend.train_mini_batch(" in online

    assert "def train_mini_batch(" in gguf
    assert "GGUFLoRATrainer" in gguf
    assert "lora_path=self.adapter_path or None" in gguf
    assert "update_deployment(" in gguf
    staged = _src("core/native_context_learning.py")
    assert staged.index("validate_candidate(backend") < staged.index("txn.commit(")


def test_bitnet_uses_real_official_runtime_and_verified_rebuild_learning():
    pro = _src("core/pro_engine.py")
    bitnet = _src("core/engines/bitnet_cpp_engine.py")
    trainer = _src("core/bitnet_rebuild_trainer.py")

    assert "from core.engines.bitnet_cpp_engine import BitNetCppReasoningBackend" in pro
    assert "BitNetCppReasoningBackend(" in pro
    assert "BitNetReasoningBackend(" not in pro
    assert "training_base_model_id=info.get(\"bitnet_training_base_model_id\")" in pro
    assert '"trainable": bool(self.bitnet_backend.training_ready())' in pro

    assert "https://github.com/microsoft/BitNet.git" in bitnet
    assert "setup_env.py" in bitnet
    assert "llama-server" in bitnet
    assert "def training_ready(self) -> bool:" in bitnet
    assert "def train_mini_batch(" in bitnet
    assert "BitNetRebuildTrainer" in bitnet
    assert "learned-i2_s.gguf" in bitnet

    assert "merge_and_unload" in trainer
    assert "convert-hf-to-gguf-bitnet.py" in trainer
    assert '"I2_S"' in trainer
