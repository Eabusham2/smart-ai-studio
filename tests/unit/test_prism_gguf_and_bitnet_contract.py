from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_bonsai_presets_are_pinned_to_prism_runtime_only():
    app = _src("app_gui.py")
    assert app.count('"prism_llama_fork": not is_apple_silicon') == 2
    assert '"Ternary-Bonsai-2-27B-PTQ1_0.gguf"' in app
    assert '"Bonsai-2-27B-PTQ1_0-CRACK.gguf"' in app
    assert app.count('"Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"') >= 2

    pro = _src("core/pro_engine.py")
    assert "PrismGGUFReasoningBackend" in pro
    assert 'if bool(info.get("prism_llama_fork"))' in pro
    assert "else GGUFReasoningBackend" in pro


def test_prism_native_library_is_process_isolated():
    src = _src("core/engines/prism_gguf_engine.py")
    worker = _src("core/engines/prism_gguf_worker.py")
    assert 'env["LLAMA_CPP_LIB_PATH"] = llama_dir' in src
    assert 'env["MTMD_CPP_LIB"] = mtmd_dir' in src
    assert '"-m", "core.engines.prism_gguf_worker"' in src
    assert "from core.engines.gguf_engine import GGUFReasoningBackend" in src
    assert "from llama_cpp import Llama" in worker


def test_normal_chat_consolidation_can_use_active_gguf_backend():
    src = _src("core/awake_auto_hook.py")
    assert "def _active_trainable_backend(engine)" in src
    assert 'getattr(engine, "gguf_backend", None)' in src
    assert "backend = _active_trainable_backend(self)" in src
    assert 'tokenize(formatted.encode("utf-8"))' in src
    assert "consolidate_chunk_sync(chunk)" in src


def test_gguf_learning_is_real_sidecar_and_persistent():
    trainer = _src("core/gguf_lora_trainer.py")
    engine = _src("core/engines/gguf_engine.py")
    assert "PeftModel.from_pretrained" in trainer
    assert "BitsAndBytesConfig" in trainer
    assert "convert_lora_to_gguf.py" in trainer
    assert '"in_proj_qkv"' in trainer
    assert '"in_proj_z"' in trainer
    assert "loss.backward()" in trainer
    assert "optimizer.step()" in trainer
    assert "lora_path=self.adapter_path if os.path.isfile(self.adapter_path) else None" in engine


def test_bitnet_backend_is_real_and_training_fail_closed():
    src = _src("core/engines/bitnet_engine.py")
    assert 'BITNET_REPO = "https://github.com/microsoft/BitNet.git"' in src
    assert '"setup_env.py"' in src
    assert '"llama-cli"' in src
    assert "subprocess.run(" in src
    assert "BitLinear" not in src
    assert "synthetic" not in src.lower()
    assert "def training_ready(self) -> bool:" in src
    assert "return False" in src
    assert "refusing to fabricate a parameter update" in src
