import os
import errno
import fnmatch
import sys
import threading
import types

import pytest

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


def _fake_hub(monkeypatch, *, file_download, snapshot_download=None, filenames=()):
    """Expose real public-API boundaries without installing/downloading a model."""
    # These substituted functions simulate online faults entirely in memory.
    # The test suite's socket/network prohibition remains active.
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    hub = types.ModuleType("huggingface_hub")
    hub.__path__ = []
    hub.hf_hub_download = file_download
    hub.snapshot_download = snapshot_download or (lambda *a, **k: "/cached/snapshot")
    hub.constants = types.SimpleNamespace(HF_HUB_DISABLE_XET=False, HF_HUB_ENABLE_HF_TRANSFER=True,
                                          HF_HUB_OFFLINE=False)
    info_calls = []
    class Api:
        def __init__(self, **kwargs):
            pass
        def repo_info(self, **kwargs):
            info_calls.append(kwargs)
            return types.SimpleNamespace(sha="a" * 40,
                                         siblings=[types.SimpleNamespace(rfilename=n) for n in filenames])
    hub.HfApi = Api
    utils = types.ModuleType("huggingface_hub.utils")
    def selected(items, allow_patterns=None, ignore_patterns=None):
        allow = [allow_patterns] if isinstance(allow_patterns, str) else allow_patterns
        ignore = [ignore_patterns] if isinstance(ignore_patterns, str) else ignore_patterns
        return [item for item in items
                if (allow is None or any(fnmatch.fnmatchcase(item, p) for p in allow))
                and not any(fnmatch.fnmatchcase(item, p) for p in ignore or [])]
    utils.filter_repo_objects = selected
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", utils)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", hub.constants)
    for name in ("huggingface_hub.file_download", "huggingface_hub._snapshot_download"):
        module = types.ModuleType(name)
        module.hf_hub_download = file_download
        module.snapshot_download = hub.snapshot_download
        monkeypatch.setitem(sys.modules, name, module)
    return hub, info_calls


def test_xet_file_failure_restarts_only_that_file_and_restores_user_settings(tmp_path, monkeypatch):
    from core.hf_transfer import hf_hub_download
    partial = tmp_path / "failed.incomplete"
    complete = tmp_path / "already-complete"
    complete.write_bytes(b"keep")
    calls = []
    def download(*args, **kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            partial.write_bytes(b"invalid-noncontiguous-old-xet-data")
            raise RuntimeError("Task error: File reconstruction error: Internal Writer Error: Background writer channel closed")
        assert hub.constants.HF_HUB_DISABLE_XET is True
        assert hub.constants.HF_HUB_ENABLE_HF_TRANSFER is False
        assert kwargs["force_download"] is True
        partial.write_bytes(b"correct")  # Hub force_download safely restarts its own partial.
        return str(partial)
    hub, _ = _fake_hub(monkeypatch, file_download=download)
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
    result = hf_hub_download("owner/model", filename="weights.bin", revision="a" * 40, token="existing-token")
    assert result == str(partial) and partial.read_bytes() == b"correct"
    assert complete.read_bytes() == b"keep"
    assert len(calls) == 2 and calls[1]["revision"] == "a" * 40
    assert calls[1]["token"] == "existing-token"
    assert os.environ["HF_HUB_DISABLE_XET"] == "0"
    assert os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "1"
    assert hub.constants.HF_HUB_DISABLE_XET is False


def test_xet_snapshot_retry_reuses_completed_files_and_keeps_exact_selection(tmp_path, monkeypatch):
    from core.hf_transfer import snapshot_download
    complete = tmp_path / "config.json"
    complete.write_text("original")
    file_calls, snapshot_calls = [], []
    def file_download(**kwargs):
        file_calls.append(dict(kwargs))
        assert kwargs["revision"] == "a" * 40
        path = tmp_path / kwargs["filename"]
        if kwargs.get("local_files_only"):
            if path.exists():
                return str(path)
            raise FileNotFoundError("not cached")
        assert kwargs["force_download"] is True
        assert hub.constants.HF_HUB_DISABLE_XET is True
        path.write_bytes(b"actual selected weights")
        return str(path)
    def snapshot(repo_id, **kwargs):
        snapshot_calls.append(dict(kwargs))
        if len(snapshot_calls) == 1:
            raise RuntimeError("CAS service error: file reconstruction error")
        assert kwargs["local_files_only"] is True
        assert kwargs["revision"] == "a" * 40
        return str(tmp_path)
    hub, info_calls = _fake_hub(monkeypatch, file_download=file_download, snapshot_download=snapshot,
                                filenames=["config.json", "model.safetensors", "unselected-large-model.gguf"])
    selected = ["config.json", "model.safetensors"]
    assert snapshot_download("owner/model", revision="a" * 40, allow_patterns=selected,
                             token="existing-token", max_workers=4) == str(tmp_path)
    assert complete.read_text() == "original"
    assert {call["filename"] for call in file_calls} == set(selected)
    assert [call["filename"] for call in file_calls if not call.get("local_files_only")] == ["model.safetensors"]
    assert info_calls[0]["revision"] == "a" * 40
    assert snapshot_calls[-1]["allow_patterns"] == selected
    assert hub.constants.HF_HUB_DISABLE_XET is False


@pytest.mark.parametrize("error", [OSError(errno.ENOSPC, "No space left on device"),
                                  PermissionError(errno.EACCES, "Permission denied"),
                                  ValueError("Invalid revision")])
def test_nontransfer_failures_are_not_retried(monkeypatch, error):
    from core.hf_transfer import hf_hub_download
    calls = []
    def fail(**kwargs):
        calls.append(kwargs)
        raise error
    _fake_hub(monkeypatch, file_download=fail)
    with pytest.raises(type(error)) as caught:
        hf_hub_download(repo_id="owner/model", filename="model.bin")
    assert caught.value is error and len(calls) == 1


def test_cancel_and_offline_do_not_start_http_fallback(monkeypatch):
    from core.hf_transfer import DownloadCancelled, hf_hub_download
    event, calls = threading.Event(), []
    def fail(**kwargs):
        calls.append(kwargs)
        event.set()
        raise RuntimeError("File reconstruction error")
    _fake_hub(monkeypatch, file_download=fail)
    with pytest.raises(DownloadCancelled):
        hf_hub_download(repo_id="owner/model", filename="model.bin", cancel_event=event)
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="reconstruction"):
        hf_hub_download(repo_id="owner/model", filename="model.bin", local_files_only=True)
    assert len(calls) == 2


def test_implicit_library_alias_uses_same_transfer_recovery(monkeypatch):
    from core.hf_transfer import run_hf_operation
    calls = []
    def download(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("Xet reconstruction failure")
        assert kwargs["force_download"] is True
        return "/cached/weights"
    _fake_hub(monkeypatch, file_download=download)
    library = types.ModuleType("diffusers.test_transfer_boundary")
    library.cached_download_alias = download
    monkeypatch.setitem(sys.modules, library.__name__, library)
    assert run_hf_operation(lambda: library.cached_download_alias("owner/model", filename="model.bin")) == "/cached/weights"
    assert len(calls) == 2


def test_python_child_keeps_interpreter_arguments_and_installs_transfer_policy(tmp_path):
    from core.hf_transfer import hf_subprocess_command
    command = hf_subprocess_command([sys.executable, "-u", "-m", "trainer.cli", "--model", "owner/model"])
    assert command[:3] == [sys.executable, "-u", "-c"]
    assert "install_hf_transfer_policy" in command[3]
    assert command[-4:] == ["module", "trainer.cli", "--model", "owner/model"]
    executable = tmp_path / "mflux-generate"
    executable.write_text("#!" + sys.executable + "\nprint('entry')\n")
    wrapped = hf_subprocess_command([str(executable), "--path", "/cached/model"])
    assert wrapped[0] == sys.executable
    assert wrapped[-4:] == ["script", str(executable), "--path", "/cached/model"]


def test_local_cache_probe_never_waits_for_network_transfer_lock(monkeypatch):
    from core import hf_transfer
    _fake_hub(monkeypatch, file_download=lambda **kwargs: "/cached/model.bin")
    result, done = [], threading.Event()
    def probe():
        try:
            result.append(hf_transfer.snapshot_download("owner/model", local_files_only=True))
        finally:
            done.set()
    with hf_transfer._TRANSFER_LOCK:
        worker = threading.Thread(target=probe, daemon=True)
        worker.start()
        completed_without_network_lock = done.wait(2)
    worker.join(2)
    assert completed_without_network_lock, "Cache checks must not block the UI during another download"
    assert result == ["/cached/snapshot"]


def test_generic_pinned_cache_requires_exact_revision_and_all_companions(tmp_path, monkeypatch):
    from core import downloader, hf_transfer
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "selected.gguf").write_bytes(b"weights")
    (snapshot / "projector.gguf").write_bytes(b"projector")
    calls = []
    def cached(**kwargs):
        calls.append(kwargs)
        assert kwargs["local_files_only"] is True
        assert kwargs["revision"] == "a" * 40
        return str(snapshot)
    monkeypatch.setattr(hf_transfer, "snapshot_download", cached)
    monkeypatch.setattr(downloader, "is_model_available_locally",
                        lambda *args: pytest.fail("An arbitrary cached revision is not the selected pin"))
    options = dict(revision="a" * 40, filename="selected.gguf", auto_download=False,
                   required_files=["selected.gguf", "projector.gguf"])
    ready = downloader.ensure_model_available("owner/model", **options)
    assert ready["status"] == "ready" and ready["path"] == str(snapshot / "selected.gguf")
    (snapshot / "projector.gguf").unlink()
    assert downloader.ensure_model_available("owner/model", **options)["status"] == "not_downloaded"
    assert calls[0]["allow_patterns"] == ["selected.gguf", "projector.gguf"]


def test_generic_selected_file_cannot_skip_missing_companion(tmp_path, monkeypatch):
    from core import downloader
    (tmp_path / "selected.gguf").write_bytes(b"weights")
    monkeypatch.setattr(downloader, "is_model_available_locally", lambda _: (True, str(tmp_path)))
    result = downloader.ensure_model_available("owner/model", filename="selected.gguf",
                                              required_files=["selected.gguf", "projector.gguf"],
                                              auto_download=False)
    assert result["status"] == "not_downloaded"


def test_pinned_download_uses_commit_cache_and_fetches_chosen_companions(tmp_path, monkeypatch):
    from core import downloader, hf_transfer
    calls = []
    def snapshot(**kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise FileNotFoundError("not cached")
        assert "local_dir" not in kwargs, "Pinned revisions must not merge in an unversioned folder"
        assert kwargs["revision"] == "b" * 40
        assert kwargs["allow_patterns"] == ["selected.gguf", "projector.gguf"]
        (tmp_path / "selected.gguf").write_bytes(b"weights")
        (tmp_path / "projector.gguf").write_bytes(b"projector")
        return str(tmp_path)
    monkeypatch.setattr(hf_transfer, "snapshot_download", snapshot)
    result = downloader.ensure_model_available("owner/model", revision="b" * 40,
                                              filename="selected.gguf",
                                              required_files=["selected.gguf", "projector.gguf"])
    assert result == {"status": "ready", "path": str(tmp_path / "selected.gguf"), "cached": False}
    assert len(calls) == 2


def test_empty_allowlist_does_not_expand_to_the_entire_repository(tmp_path, monkeypatch):
    from core.hf_downloader import download_model_from_hf
    from core import hf_transfer
    calls = []
    monkeypatch.setattr(hf_transfer, "snapshot_download", lambda **kwargs: calls.append(kwargs) or str(tmp_path))
    result = download_model_from_hf("owner/model", allow_patterns=[])
    assert calls[0]["allow_patterns"] == []
    assert result["status"] == "error"


def test_failed_selected_download_never_uses_unrelated_cached_weights(tmp_path, monkeypatch):
    from core.hf_downloader import download_model_from_hf
    from core import hf_transfer
    (tmp_path / "unselected.gguf").write_bytes(b"unrelated")
    register_loaded_model("owner/model")
    def fail(**kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(hf_transfer, "snapshot_download", fail)
    result = download_model_from_hf("owner/model", revision="c" * 40, required_files=["selected.gguf"])
    assert result["status"] == "error"
    assert "Not enough free disk space" in result["error"]


def test_selected_cache_ignores_an_unselected_incomplete_format(tmp_path):
    from core.hf_downloader import resolve_cached_model_path
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"selected weights")
    alternate = tmp_path / "alternate"
    alternate.mkdir()
    (alternate / "model.safetensors.index.json").write_text('{"weight_map":{"x":"missing.safetensors"}}')
    info = {"path": str(tmp_path), "repo_revision": "a" * 40,
            "download_required_files": ["config.json", "model.safetensors"]}
    assert resolve_cached_model_path(info) == str(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    assert resolve_cached_model_path(info) is None


def test_cache_resolution_keeps_empty_allowlist_and_uses_only_selected_revision(tmp_path, monkeypatch):
    from core import hf_downloader, hf_transfer
    wrong = tmp_path / "models--owner--model" / "snapshots" / ("b" * 40)
    wrong.mkdir(parents=True)
    (wrong / "model.safetensors").write_bytes(b"other revision")
    calls = []
    def absent(**kwargs):
        calls.append(kwargs)
        raise FileNotFoundError("exact selection absent")
    monkeypatch.setattr(hf_transfer, "snapshot_download", absent)
    monkeypatch.setattr(hf_downloader, "_candidate_hf_cache_roots", lambda: iter([str(tmp_path)]))
    info = {"repo_id": "owner/model", "repo_revision": "a" * 40, "path": str(wrong),
            "download_required_files": ["model.safetensors"], "download_allow_patterns": []}
    assert hf_downloader.resolve_cached_model_path(info) is None
    assert calls == [{"repo_id": "owner/model", "revision": "a" * 40,
                      "local_files_only": True, "allow_patterns": []}]
