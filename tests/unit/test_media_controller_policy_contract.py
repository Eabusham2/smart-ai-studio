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


def test_multimodal_capabilities_survive_memory_pause_resume():
    src = _src("core/media_orchestrator.py")
    assert "original_input_modalities = set(" in src
    assert "engine.active_input_modalities = set(original_input_modalities)" in src


def test_mflux_training_resolution_is_architecture_not_repo_id_driven():
    src = _src("core/media_training_backends.py")
    start = src.index("def _mflux_image_matches(info):")
    end = src.index("def _mflux_image_available", start)
    matcher = src[start:end]
    assert "_arch_blob(info)" in matcher
    assert 'info.get("repo_id"' not in matcher
    assert "def _mflux_training_key(info)" in src
    assert "_training_repo(info)" in src


def test_exact_controller_modality_matrix_is_locked():
    src = _src("core/media_orchestrator.py")
    # image-only -> image RSI; audio-only -> audio RSI.
    assert 'return raw.intersection({"image", "audio"})' in src
    # video-capable OR image+audio -> image+video+audio RSI.
    assert 'if "video" in raw or {"image", "audio"}.issubset(raw):' in src
    assert 'return {"image", "video", "audio"}' in src


def test_text_only_controller_can_generate_and_learn_but_not_grade_or_rsi():
    src = _src("core/media_orchestrator.py")
    assert "This controller is text-only for media: it may generate media and use explicit media Learn" in src
    assert '"media_generate"' in src
    assert '"media_learn"' in src
    assert "cannot directly ingest" in src
    assert "cannot grade its own generated output or perform genuine media RSI" in src


def test_media_learn_is_independent_of_controller_perception():
    src = _src("core/media_orchestrator.py")
    learn_at = src.index('if name == "media_learn":')
    rsi_gate_at = src.index('if name == "media_rsi" and not can_perceive:')
    assert learn_at > rsi_gate_at
    learn_block = src[learn_at:learn_at + 1800]
    assert "_can_perceive" not in learn_block
    assert "self.learning.learn(" in learn_block


def test_media_learn_accepts_local_and_online_sources():
    orchestrator = _src("core/media_orchestrator.py")
    learning = _src("core/media_learning.py")
    assert "file/folder/JSONL/URL/search query" in orchestrator
    assert "gather_samples(" in orchestrator
    assert "urllib.request" in learning
    assert "os.path.isdir" in learning or "Path(source).is_dir" in learning
    assert ".jsonl" in learning


def test_imported_text_models_carry_detected_input_modalities():
    policy = _src("core/model_policy.py")
    app = _src("app_gui.py")
    assert "def infer_input_modalities" in policy
    assert '"video" in raw or {"image", "audio"}.issubset(raw)' not in policy
    assert '"input_modalities": input_modalities' in policy
    assert 'target_info.get("input_modalities") or ["text"]' in app
