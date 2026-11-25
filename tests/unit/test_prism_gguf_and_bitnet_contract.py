import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
_PINNING_SPEC = importlib.util.spec_from_file_location(
    "_native_runtime_pinning_contract", ROOT / "core" / "native_runtime_pinning.py"
)
_PINNING = importlib.util.module_from_spec(_PINNING_SPEC)
assert _PINNING_SPEC.loader is not None
_PINNING_SPEC.loader.exec_module(_PINNING)
require_full_commit = _PINNING.require_full_commit
validate_runtime_receipt = _PINNING.validate_runtime_receipt
write_runtime_receipt = _PINNING.write_runtime_receipt


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


@pytest.mark.parametrize("mutable_ref", ["", "prism", "main", "v1.0", "0123456789abcdef"])
def test_native_runtime_commit_settings_reject_mutable_refs(mutable_ref):
    with pytest.raises(RuntimeError, match="full 40-character Git commit"):
        require_full_commit(mutable_ref, "RUNTIME_COMMIT")


def test_native_runtime_commit_settings_accept_full_hashes():
    commit = "0123456789ABCDEF0123456789ABCDEF01234567"
    assert require_full_commit(commit, "RUNTIME_COMMIT") == commit.lower()


def test_native_runtime_receipts_bind_source_and_artifact_bytes(tmp_path):
    commit = "0123456789abcdef0123456789abcdef01234567"
    build = tmp_path / "build"
    build.mkdir()
    library = build / "libllama.so"
    library.write_bytes(b"reviewed build output")
    receipt = build / "smartai-runtime.json"
    inputs = {"build_command": ["cmake", "--build", "build"]}

    write_runtime_receipt(
        receipt,
        build,
        "https://example.invalid/reviewed.git",
        commit,
        {"llama": library},
        inputs,
    )
    assert validate_runtime_receipt(
        receipt,
        build,
        "https://example.invalid/reviewed.git",
        commit,
        {"llama": library},
        inputs,
    )
    assert not validate_runtime_receipt(
        receipt,
        build,
        "https://example.invalid/reviewed.git",
        "1123456789abcdef0123456789abcdef01234567",
        {"llama": library},
        inputs,
    )

    library.write_bytes(b"changed output")
    assert not validate_runtime_receipt(
        receipt,
        build,
        "https://example.invalid/reviewed.git",
        commit,
        {"llama": library},
        inputs,
    )


def test_managed_native_sources_are_verified_before_execution():
    pinning = _src("core/native_runtime_pinning.py")
    bitnet = _src("core/engines/bitnet_cpp_engine.py")
    alternate_bitnet = _src("core/engines/bitnet_engine.py")
    prism = _src("core/engines/prism_gguf_engine.py")

    assert '"fetch", "--depth", "1", "--no-tags", "origin", approved' in pinning
    assert '"rev-parse", "--verify", "HEAD^{commit}"' in pinning
    assert '"status",\n        "--porcelain=v1"' in pinning

    bitnet_prepare = bitnet[bitnet.index("def _prepare_official_runtime"):]
    assert bitnet_prepare.index("commit = ensure_pinned_checkout(") < bitnet_prepare.index(
        "subprocess.run(command"
    )
    assert bitnet_prepare.index("verify_pinned_checkout(") < bitnet_prepare.index(
        "write_runtime_receipt("
    )
    assert "SMARTAI_BITNET_SERVER" in bitnet_prepare
    assert 'getattr(self.bitnet_backend, "last_error", "")' in _src("core/pro_engine.py")

    alternate_prepare = alternate_bitnet[alternate_bitnet.index("def _ensure_runtime"):]
    assert alternate_prepare.index("commit = ensure_pinned_checkout(") < alternate_prepare.index(
        "subprocess.run("
    )
    assert alternate_prepare.index("verify_pinned_checkout(") < alternate_prepare.index(
        "write_runtime_receipt("
    )

    prism_prepare = prism[prism.index("def ensure_prism_native_runtime"):]
    assert "SMARTAI_PRISM_LLAMA_REF" not in prism
    assert prism_prepare.index("commit = ensure_pinned_checkout(") < prism_prepare.index("configure = [")
    assert prism_prepare.index("verify_pinned_checkout(") < prism_prepare.index("write_runtime_receipt(")
    assert "SMARTAI_PRISM_LLAMA_LIB_PATH" in prism_prepare
    assert "SMARTAI_PRISM_MTMD_LIB_PATH" in prism_prepare


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
    assert "UniversalControllerBackend._transformers_lora_targets" in trainer
    targets = _src("core/controller_runtime.py")
    assert '"in_proj_qkv"' in targets and '"in_proj_z"' in targets
    assert "update_lora(" in trainer
    optimizer = _src("core/torch_context_learning.py")
    assert "loss.backward()" in optimizer
    assert "optimizer.step()" in optimizer
    assert "lora_path=self.adapter_path or None" in engine


def test_bitnet_backend_is_real_and_training_rebuilds_deployment_weights():
    src = _src("core/engines/bitnet_cpp_engine.py")
    trainer = _src("core/bitnet_rebuild_trainer.py")
    assert '"https://github.com/microsoft/BitNet.git"' in src
    assert '"setup_env.py"' in src
    assert '"build/bin/llama-server"' in src
    assert '"build/bin/Release/llama-server.exe"' in src
    assert "def training_ready(self) -> bool:" in src
    assert "BitNetRebuildTrainer" in src
    assert "def train_mini_batch(" in src
    assert "training_base_model_id" in src
    assert "learned-i2_s.gguf" in src

    assert "PeftModel.from_pretrained" in trainer
    assert "get_peft_model" in trainer
    assert "merge_and_unload" in trainer
    assert "convert-hf-to-gguf-bitnet.py" in trainer
    assert '"I2_S"' in trainer
    assert "update_lora(" in trainer
    optimizer = _src("core/torch_context_learning.py")
    assert "loss.backward()" in optimizer
    assert "optimizer.step()" in optimizer
    assert "zero parameter change" in optimizer
