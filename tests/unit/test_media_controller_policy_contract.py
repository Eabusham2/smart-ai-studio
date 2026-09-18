from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_all_text_controllers_keep_generation_and_media_learn_tools():
    src = _src("core/media_orchestrator.py")
    assert '"media_generate"' in src
    assert '"media_learn"' in src
    assert "Tools always available: media_list_models, media_generate, media_pro, media_gather, media_learn" in src


def test_media_rsi_requires_controller_perception():
    src = _src("core/media_orchestrator.py")
    assert 'if name == "media_rsi" and not can_perceive:' in src
    assert "Generation and explicit media Learn remain available." in src
    assert "active controller multimodal self-grade" in src


def test_user_modality_expansion_rule_is_preserved():
    src = _src("core/media_orchestrator.py")
    assert '"video" in raw or {"image", "audio"}.issubset(raw)' in src
    assert 'return {"image", "video", "audio"}' in src


def test_media_rsi_updates_media_model_not_text_trainer():
    src = _src("core/media_orchestrator.py")
    assert 'learned = self.call(' in src
    assert '"media_learn"' in src
    assert '"model_id":winner["model_id"]' in src
    assert 'weights_updated":learned.get("weights_updated",False)' in src


def test_normal_chat_can_authorize_media_learning_only_when_user_asks():
    src = _src("core/media_orchestrator.py")
    assert "allow_media_update = bool(re.search" in src
    assert 'requested_name in ("media_learn", "media_rsi")' in src


def test_media_training_resolution_is_architecture_driven_and_extensible():
    learning = _src("core/media_learning.py")
    backends = _src("core/media_training_backends.py")
    assert "matcher" in learning
    assert "Prefer the first *available* compatible trainer" in learning
    assert "def _pipeline_class(info)" in backends
    assert "def _training_repo(info)" in backends
    assert 'register_media_training_backend("flux2_lora"' in backends
    assert 'register_media_training_backend("zimage_lora"' in backends
    assert '"finetrainers_lora"' in backends
    assert '"wan21_lora"' in backends


def test_original_text_phase3_weight_update_path_is_untouched():
    src = _src("eval/phase4_pro_rsi.py")
    assert "optim.AdamW" in src
    assert "nn.value_and_grad" in src
    assert "project_gradient" in src
    assert "opt.update" in src
