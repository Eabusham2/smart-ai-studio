from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "smartai_model_policy_contract",
    ROOT / "core" / "model_policy.py",
)
_model_policy = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_model_policy)
derive_runtime_metadata = _model_policy.derive_runtime_metadata


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_prism_metadata_selects_exact_ternary_artifact_and_projector():
    meta = derive_runtime_metadata(
        "prism-ml bonsai prism_hadamard ternary",
        file_names=[
            "model-Q4_K_M.gguf",
            "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
            "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf",
        ],
        repo_id="owner/model",
        ternary=True,
    )
    assert meta["backend_family"] == "prism_gguf"
    assert meta["controller_runtime"] == "gguf"
    assert meta["prism_llama_fork"] is True
    assert meta["gguf_file"] == "Ternary-Bonsai-2-27B-PTQ1_0.gguf"
    assert meta["mmproj_file"] == "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"


def test_bitnet_architecture_beats_gguf_container_and_keeps_training_master_separate():
    meta = derive_runtime_metadata(
        "microsoft bitnet BitLinear 1.58-bit",
        file_names=["ggml-model-i2_s.gguf"],
        repo_id="microsoft/bitnet-b1.58-2B-4T-gguf",
        ternary=True,
    )
    assert meta["backend_family"] == "bitnet"
    assert meta["controller_runtime"] == "bitnet"
    assert meta["gguf_file"] == "ggml-model-i2_s.gguf"
    assert meta["bitnet_training_base_model_id"] == "microsoft/bitnet-b1.58-2B-4T-bf16"
    assert not meta.get("gguf_training_base_model_id")


def test_multiple_ambiguous_generic_ggufs_are_never_guessed_as_ternary():
    meta = derive_runtime_metadata(
        "ternary model",
        file_names=["model-Q4_K_M.gguf", "model-Q5_K_M.gguf"],
        repo_id="owner/model",
        ternary=True,
    )
    assert meta["backend_family"] == "gguf"
    assert meta["gguf_file"] is None


def test_custom_fetch_uses_detected_exact_files_and_persists_runtime_metadata():
    app = _src("app_gui.py")
    assert 'gguf_file = str(verified_policy_meta.get("gguf_file")' in app
    assert "allow_patterns=patterns" in app
    assert "required_files=required" in app
    assert '"backend_family": str(verified_policy_meta.get("backend_family")' in app
    assert '"runtime_family": str(verified_policy_meta.get("runtime_family")' in app
    assert '"prism_llama_fork": bool(verified_policy_meta.get("prism_llama_fork"' in app
    assert '"bitnet_training_base_model_id": verified_policy_meta.get("bitnet_training_base_model_id")' in app


def test_custom_gguf_training_never_assumes_qwen_when_lineage_is_missing():
    gguf = _src("core/engines/gguf_engine.py")
    pro = _src("core/pro_engine.py")
    assert 'training_base_model_id or "Qwen/Qwen3.8-27B"' not in gguf
    assert 'info.get("gguf_training_base_model_id") or "Qwen/Qwen3.8-27B"' not in pro
    assert "base_model_id=self.training_base_model_id" in gguf


def test_learn_uses_active_verified_trainable_backend_not_an_mlx_only_gate():
    learner = _src("core/autonomous_learner.py")
    assert "def _require_live_trainable_backend(self):" in learner
    assert 'active == "gguf"' in learner
    assert 'active == "bitnet"' in learner
    assert 'active == "controller"' in learner
    assert "training_ready" in learner
    assert "refusing fake/offline learning" in learner


def test_bitnet_parameter_updates_remain_fail_closed_until_real_adapter_support_exists():
    bitnet = _src("core/engines/bitnet_engine.py")
    assert "def training_ready(self) -> bool:" in bitnet
    assert "return False" in bitnet
    assert "refusing to fabricate a parameter update" in bitnet
