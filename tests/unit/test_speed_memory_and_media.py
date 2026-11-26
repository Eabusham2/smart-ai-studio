"""Exercise application memory accounting and media/tuning lock boundaries."""
import sys
import threading
from types import SimpleNamespace

from core.media_orchestrator import MediaController
from core.native_speed import tuning_memory_ok


def test_worker_accounts_for_parent_and_siblings_in_user_limit(monkeypatch):
    gb = 1024 ** 3
    processes = {}

    class Process:
        def __init__(self, pid=None):
            self.pid = 300 if pid is None else pid

        def memory_info(self):
            return SimpleNamespace(rss=processes[self.pid][0])

        def children(self, recursive=False):
            assert recursive
            return [Process(pid) for pid in processes[self.pid][1]]

    processes.update({100: (3 * gb, [200, 300]), 200: (2 * gb, []), 300: (4 * gb, [])})
    fake = SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(total=32 * gb, available=12 * gb),
        Process=Process, NoSuchProcess=ProcessLookupError, AccessDenied=PermissionError,
    )
    monkeypatch.setitem(sys.modules, "psutil", fake)
    assert tuning_memory_ok(8)  # The worker alone fits.
    assert not tuning_memory_ok(8, owner_pid=100)  # The complete app does not.
    assert tuning_memory_ok(10, owner_pid=100)


def test_media_pause_takes_lifecycle_before_runtime_and_reenters_for_reload():
    held = []

    class CheckedLock:
        def __init__(self, name):
            self.name = name
            self.lock = threading.RLock()

        def __enter__(self):
            if self.name == "runtime":
                assert "lifecycle" in held
            self.lock.acquire()
            held.append(self.name)
            return self

        def __exit__(self, *args):
            assert held.pop() == self.name
            self.lock.release()

    lifecycle, runtime = CheckedLock("lifecycle"), CheckedLock("runtime")
    engine = SimpleNamespace(_model_lock=lifecycle, awake_consolidator=None)
    backend = SimpleNamespace(_runtime_model_lock=runtime)
    controller = MediaController.__new__(MediaController)
    with controller._pause_guard(engine, backend):
        # unload/load reenters the lifecycle lock while the lease holds runtime.
        with engine._model_lock:
            assert held == ["lifecycle", "runtime", "lifecycle"]
    assert held == []


def test_prism_memory_provider_returns_only_its_live_owned_worker():
    from core.engines.prism_gguf_engine import PrismGGUFReasoningBackend

    backend = PrismGGUFReasoningBackend.__new__(PrismGGUFReasoningBackend)
    backend._worker = None
    assert backend.owned_process_ids() == []
    backend._worker = SimpleNamespace(pid=1234, poll=lambda: None)
    assert backend.owned_process_ids() == [1234]
    backend._worker = SimpleNamespace(pid=1234, poll=lambda: 0)
    assert backend.owned_process_ids() == []
    backend._worker = SimpleNamespace(pid=True, poll=lambda: None)
    assert backend.owned_process_ids() == []


def test_every_builtin_media_choice_has_an_exact_download_or_runtime_reason():
    """Exercise the real GUI catalog without importing Tk or neural runtimes."""
    import ast
    from pathlib import Path, PurePosixPath
    import re

    from config.bonsai_models import bonsai_model_info
    from config.media_models import apply_builtin_media_catalog

    module = ast.parse((Path(__file__).parents[2] / "app_gui.py").read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef)
               and node.name == "SmartAIChatbotApp")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    catalog = next(node.value for node in init.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Attribute) and target.attr == "models_config"
                           for target in node.targets))
    for apple in (True, False):
        models = eval(compile(ast.Expression(catalog), "builtin-media", "eval"), {
            "bonsai_model_info": bonsai_model_info, "is_apple_silicon": apple,
        })
        original_repos = {key: value["repo_id"] for key, value in models.items()}
        apply_builtin_media_catalog(models)
        assert original_repos == {key: value["repo_id"] for key, value in models.items()}
        assert set(models) == {f"model_{index}" for index in range(1, 16)}
        for field in ("precision", "raw_params", "base_params", "vram", "est_speed"):
            assert models["model_3"][field] == models["model_6"][field]
        assert "Ideogram Instant" in models["model_6"]["name"]
        assert "RealVisXL" in models["model_6"]["name"]
        assert models["model_6"]["raw_params"] is None
        assert "AIO" not in models["model_5"]["name"]
        assert models["model_5"]["raw_params"] is None
        assert models["model_9"]["raw_params"] == 33_000_000_000
        assert "AfterMidnight" in models["model_9"]["name"]
        assert "rank64" in models["model_9"]["precision"]
        assert models["model_9"]["download_bytes"] == 104_246_158_600
        for info in models.values():
            if info.get("model_type") not in {"image", "video", "audio"}:
                continue
            assert info["builtin_media_preset"]
            required = info.get("download_required_files", [])
            reason = info.get("download_unavailable_reason", "")
            assert required and not reason, info["repo_id"]
            assert re.fullmatch(r"[0-9a-f]{40}", info["repo_revision"])
            assert required == info["download_allow_patterns"]
            assert len(required) == len(set(required))
            assert info["download_bytes"] > 0
            assert any(name.endswith((".safetensors", ".gguf")) for name in required)
            components = info.get("download_components", [])
            assert info["download_bytes"] == info["download_primary_bytes"] + sum(c["size_bytes"] for c in components)
            assert len({c["role"] for c in components}) == len(components)
            for component in components:
                assert re.fullmatch(r"[0-9a-f]{40}", component["revision"])
                assert component["repo_id"] and component["size_bytes"] > 0
                assert component["required_files"]
                required = [*required, *component["required_files"]]
            for name in required:
                assert not PurePosixPath(name).is_absolute()
                assert ".." not in PurePosixPath(name).parts
                assert not any(char in name for char in "*?[]\\")


def test_realvis_legacy_slot_selects_one_complete_fp16_pipeline():
    from config.media_models import REALVIS_REPO, media_download_metadata
    from config.settings import MODEL_PRESETS

    info = MODEL_PRESETS["model_6"]
    assert info["default_repo_id"] == REALVIS_REPO
    assert set(info["artifacts"].values()) == {REALVIS_REPO}
    assert "Ideogram Instant" in info["name"] and "RealVisXL" in info["name"]
    assert "16GB" not in info["name"] and "Turbo" not in info["name"]
    assert info["diffusers_variant"] == "fp16"
    assert info["inference_steps"] >= 30
    selected = set(info["download_required_files"])
    assert {name for name in selected if name.endswith(".safetensors")} == {
        "text_encoder/model.fp16.safetensors", "text_encoder_2/model.fp16.safetensors",
        "unet/diffusion_pytorch_model.fp16.safetensors", "vae/diffusion_pytorch_model.fp16.safetensors",
    }
    for component in ("text_encoder", "text_encoder_2", "unet", "vae"):
        assert f"{component}/config.json" in selected
    for tokenizer in ("tokenizer", "tokenizer_2"):
        assert {f"{tokenizer}/{name}" for name in ("vocab.json", "merges.txt", "tokenizer_config.json")} <= selected
    assert {"model_index.json", "scheduler/scheduler_config.json"} <= selected
    assert info["download_bytes"] < 7_000_000_000
    first = media_download_metadata(REALVIS_REPO)
    first["download_required_files"].clear()
    assert media_download_metadata(REALVIS_REPO)["download_required_files"]


def test_supported_audio_keeps_its_text_conditioner_and_gated_access_route():
    from config.media_models import media_download_metadata

    for owner in ("mlx-community", "stabilityai"):
        for variant in ("music", "sfx"):
            info = media_download_metadata(f"{owner}/stable-audio-3-small-{variant}")
            assert not info.get("download_unavailable_reason")
            assert {"model.safetensors", "model_config.json", "t5gemma-b-b-ul2/model.safetensors",
                    "t5gemma-b-b-ul2/config.json", "t5gemma-b-b-ul2/tokenizer.model"} <= set(info["download_required_files"])
            assert bool(info.get("requires_hf_auth")) == (owner == "stabilityai")
    assert media_download_metadata("user/a-different-selected-model") == {}
