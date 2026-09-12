import os

from core.hf_downloader import (
    is_model_cached_locally,
    register_loaded_model,
    unregister_loaded_model,
)


def setup_function():
    unregister_loaded_model()


def teardown_function():
    unregister_loaded_model()


def test_local_weight_directory_is_installed(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    assert is_model_cached_locally(str(model_dir)) is True


def test_metadata_only_directory_is_not_installed(tmp_path):
    model_dir = tmp_path / "partial"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    assert is_model_cached_locally(str(model_dir)) is False


def test_hf_hub_snapshot_in_configured_cache_is_installed(tmp_path, monkeypatch):
    cache_root = tmp_path / "hub"
    snapshot = cache_root / "models--owner--model" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"weights")

    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache_root))

    assert is_model_cached_locally("owner/model") is True


def test_successfully_loaded_model_is_ready_even_when_cache_scan_cannot_find_it(tmp_path, monkeypatch):
    empty_cache = tmp_path / "empty-hub"
    empty_cache.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(empty_cache))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(empty_cache))

    assert is_model_cached_locally("owner/runtime-loaded") is False
    register_loaded_model("owner/runtime-loaded")
    assert is_model_cached_locally("owner/runtime-loaded") is True
    unregister_loaded_model("owner/runtime-loaded")
    assert is_model_cached_locally("owner/runtime-loaded") is False
