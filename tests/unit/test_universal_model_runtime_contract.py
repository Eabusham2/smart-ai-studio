from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_model_loading_is_capability_driven():
    pro = _src("core/pro_engine.py")
    runtime = _src("core/controller_runtime.py")
    policy = _src("core/model_policy.py")
    assert "resolve_controller_runtime(info" in pro
    assert "register_runtime_family" in runtime
    assert "_RUNTIME_FAMILY_RESOLVERS" in runtime
    assert "infer_controller_runtime" in policy
    assert "transformers_auto" in runtime
    assert "resolve_gguf_artifacts" in runtime


def test_specialized_mlx_vlms_remain_on_real_trainable_mlx_backend():
    pro = _src("core/pro_engine.py")
    mlx = _src("core/mlx_engine.py")
    persistence = _src("core/mlx_adapter_persistence.py")
    assert '{"mlx_vlm", "mlx_repo_vlm", "jang_vlm"}' in pro
    assert "model_info=info" in pro
    assert "self.awake_consolidator.engine = self.mlx_backend" in pro
    assert "def get_training_model(self):" in mlx
    assert 'getattr(self.model, "language_model", None)' in mlx
    assert 'return self._run_training_method("train_mini_batch"' in mlx
    assert 'return self._run_training_method("compute_mlx_fisher"' in mlx
    assert "training_model.update(" in persistence


def test_bundled_vlm_runtime_is_used_instead_of_plain_loader():
    runtime = _src("core/controller_runtime.py")
    mlx = _src("core/mlx_engine.py")
    assert '"runtime_dir") or "runtime"' in runtime
    assert '"runtime_module") or "vision_artifact"' in runtime
    assert '"runtime_loader") or "load_vl_model"' in runtime
    assert "self._vlm_loader = loader" in mlx
    assert "self._vlm_loader._mlx_prompt" in mlx


def test_controller_perception_supports_image_video_audio_families():
    runtime = _src("core/controller_runtime.py")
    policy = _src("core/model_policy.py")
    assert 'kind in {"image", "video"}' in runtime
    assert '"audio" in media' in runtime
    assert '"videos" in kwargs' in runtime
    assert '"automatic-speech-recognition"' in policy
    assert '"video-to-text"' in policy
    assert '"image-to-text"' in policy


def test_generators_remain_separate_from_controller_models():
    policy = _src("core/model_policy.py")
    assert '"text-to-image"' in policy
    assert '"text-to-video"' in policy
    assert '"text-to-audio"' in policy
    assert '"audio-generation"' in policy


def test_gui_passes_full_model_capabilities_into_pro_engine():
    app = _src("app_gui.py")
    assert 'model_info=target_info' in app
    assert '"controller_runtime": str(policy_meta.get("controller_runtime")' in app
    assert '"runtime_family": "bonsai2_hadamard"' in app
    assert '"input_modalities": ["text", "image", "video"]' in app
    assert '"model_16"' not in app
