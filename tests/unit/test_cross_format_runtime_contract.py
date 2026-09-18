from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_bonsai_non_mlx_artifacts_are_exact_ternary_and_prism_only():
    app = _src("app_gui.py")
    assert '"repo_id": "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit" if is_apple_silicon else "prism-ml/Ternary-Bonsai-2-27B-gguf"' in app
    assert '"gguf_file": None if is_apple_silicon else "Ternary-Bonsai-2-27B-PTQ1_0.gguf"' in app
    assert '"repo_id": "dealignai/Bonsai-2-27B-CRACK-Ternary-JANG" if is_apple_silicon else "dealignai/Bonsai-2-27B-1bit-CRACK-GGUF"' in app
    assert '"gguf_file": None if is_apple_silicon else "Bonsai-2-27B-PTQ1_0-CRACK.gguf"' in app
    assert app.count('"prism_llama_fork": not is_apple_silicon') >= 2
    assert app.count('"mmproj_file": None if is_apple_silicon else "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"') >= 2
    assert app.count('"gguf_training_base_model_id": None if is_apple_silicon else "Qwen/Qwen3.8-27B"') >= 2


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
    assert 'str(getattr(engine, "active_backend", "") or "").lower() == "gguf"' in awake
    assert "consolidator.consolidate_chunk_sync(chunk)" in awake
    assert "if not learned:" in awake

    assert "def _real_training_ready" in online
    assert "success = self.consolidate_chunk_sync(evicted_chunk)" in online
    assert "(retained_history, True) if success else (conversation_history, False)" in online

    assert 'str(getattr(self.engine, "active_backend", "") or "").lower() == "gguf"' in learner
    assert "backend.train_mini_batch(" in learner

    assert "def train_mini_batch(" in gguf
    assert "GGUFLoRATrainer" in gguf
    assert "lora_path=self.adapter_path if os.path.isfile(self.adapter_path) else None" in gguf
    assert "peft_backup" in gguf


def test_bitnet_uses_real_official_runtime_not_synthetic_backend():
    pro = _src("core/pro_engine.py")
    bitnet = _src("core/engines/bitnet_cpp_engine.py")

    assert "from core.engines.bitnet_cpp_engine import BitNetCppReasoningBackend" in pro
    assert "BitNetCppReasoningBackend(" in pro
    assert "BitNetReasoningBackend(" not in pro

    assert "https://github.com/microsoft/BitNet.git" in bitnet
    assert "setup_env.py" in bitnet
    assert "llama-server" in bitnet
    assert "def training_ready(self) -> bool:" in bitnet
    assert "return False" in bitnet  # fail closed: inference support, no fake parameter updates
