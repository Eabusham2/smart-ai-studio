"""A remote model selection cannot drift to another revision or quantization."""
import threading
from types import SimpleNamespace
import sys

import pytest

from core import model_import_download as imports
from core.controller_runtime import resolve_gguf_artifacts
from core.hf_downloader import download_model_from_hf
from tests.unit.test_model_admission import write_gguf


def selection(*files):
    return {"repo_id": "owner/chosen", "repo_revision": "a" * 40,
            "gguf_file": files[0], "download_required_files": list(files),
            "download_allow_patterns": list(files), "runtime_preference": "native"}


def test_download_pins_revision_keeps_all_chosen_shards_and_inspects_bytes(tmp_path, monkeypatch):
    paths = [write_gguf(tmp_path / f"quant/chosen-{i+1:05d}-of-00002.gguf", shard=i, count=2)
             for i in range(2)]
    write_gguf(tmp_path / "other-Q8.gguf", quant=8)
    meta = selection(*(str(p.relative_to(tmp_path)).replace("\\", "/") for p in paths))
    calls = []
    def download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return {"status": "success", "local_dir": str(tmp_path)}
    monkeypatch.setattr(imports, "download_model_from_hf", download)
    result = imports.download_imported_model("owner/chosen", meta)
    assert result["status"] == "success", result
    assert calls[0][1]["revision"] == "a" * 40
    assert calls[0][1]["required_files"] == meta["download_required_files"]
    assert calls[0][1]["allow_patterns"] == meta["download_allow_patterns"]
    assert result["metadata"]["gguf_files"] == meta["download_required_files"]
    assert result["metadata"]["source_revision"] == "a" * 40
    assert result["metadata"]["runtime_preference"] == "native"
    assert result["metadata"]["requires_local_validation"] is False
    assert result["metadata"]["precision"] == "GGUF Q4_0"


def test_corrupt_download_never_becomes_a_registered_model(tmp_path, monkeypatch):
    (tmp_path / "chosen.gguf").write_bytes(b"not weights")
    monkeypatch.setattr(imports, "download_model_from_hf", lambda *a, **k: {
        "status": "success", "local_dir": str(tmp_path)})
    result = imports.download_imported_model("owner/chosen", selection("chosen.gguf"))
    assert result["status"] == "error"
    assert "inspection failed" in result["error"]
    assert "metadata" not in result


@pytest.mark.parametrize("change", [
    {"repo_revision": "main"}, {"repo_id": "someone/else"},
    {"download_required_files": ["../chosen.gguf"]},
    {"download_allow_patterns": ["*.gguf"]}, {"download_required_files": []},
])
def test_invalid_selection_cannot_start_a_download(change, monkeypatch):
    monkeypatch.setattr(imports, "download_model_from_hf", lambda *a, **k: pytest.fail("unverified download"))
    meta = {**selection("chosen.gguf"), **change}
    assert imports.download_imported_model("owner/chosen", meta)["status"] == "error"


def test_cancelled_transfer_does_not_inspect_or_register(tmp_path, monkeypatch):
    event = threading.Event()
    def download(*args, **kwargs):
        event.set()
        return {"status": "success", "local_dir": str(tmp_path)}
    monkeypatch.setattr(imports, "download_model_from_hf", download)
    monkeypatch.setattr(imports, "admit_local_model", lambda *a, **k: pytest.fail("cancelled inspection"))
    assert imports.download_imported_model("owner/chosen", selection("chosen.gguf"), cancel_event=event)["status"] == "cancelled"


def test_downloader_passes_revision_and_does_not_report_success_after_cancel(tmp_path, monkeypatch):
    event, calls = threading.Event(), []
    def snapshot_download(**kwargs):
        calls.append(kwargs)
        event.set()
        return str(tmp_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    result = download_model_from_hf("owner/chosen", revision="b" * 40, cancel_event=event)
    assert calls[0]["revision"] == "b" * 40
    assert result["status"] == "cancelled"
    result = download_model_from_hf("owner/chosen", revision="b" * 40, cancel_event=event)
    assert result["status"] == "cancelled" and len(calls) == 1


def test_gguf_resolver_uses_exact_relative_path_even_with_identical_basenames(tmp_path):
    chosen = write_gguf(tmp_path / "chosen/model.gguf")
    write_gguf(tmp_path / "other/model.gguf", quant=8)
    model, _ = resolve_gguf_artifacts(str(tmp_path), {"gguf_file": "chosen/model.gguf"})
    assert model == str(chosen)
    with pytest.raises(RuntimeError, match="Ambiguous"):
        resolve_gguf_artifacts(str(tmp_path), {"gguf_file": "model.gguf"})
    with pytest.raises(RuntimeError, match="Selected GGUF is missing"):
        resolve_gguf_artifacts(str(tmp_path), {"gguf_file": "missing/model.gguf"})


def test_gguf_resolver_requires_complete_chosen_shards(tmp_path):
    write_gguf(tmp_path / "selected-00001-of-00002.gguf", count=2)
    with pytest.raises(RuntimeError, match="shard is missing"):
        resolve_gguf_artifacts(str(tmp_path), {
            "gguf_file": "selected-00001-of-00002.gguf",
            "gguf_files": ["selected-00001-of-00002.gguf", "selected-00002-of-00002.gguf"],
        })


def test_gguf_resolver_rejects_parent_traversal(tmp_path):
    write_gguf(tmp_path / "chosen.gguf")
    with pytest.raises(RuntimeError, match="Invalid selected"):
        resolve_gguf_artifacts(str(tmp_path), {"gguf_file": "../chosen.gguf"})


def test_exact_model_file_keeps_its_selected_projector_relative_to_snapshot(tmp_path):
    model = write_gguf(tmp_path / "quant/model.gguf")
    projector = write_gguf(tmp_path / "vision/mmproj.gguf")
    resolved, mmproj = resolve_gguf_artifacts(str(model), {
        "path": str(tmp_path), "gguf_file": "quant/model.gguf",
        "gguf_files": ["quant/model.gguf"], "mmproj_file": "vision/mmproj.gguf",
    })
    assert resolved == str(model) and mmproj == str(projector)


def test_verified_remote_media_task_survives_local_tensor_inspection(tmp_path, monkeypatch):
    from tests.unit.test_model_admission import write_checkpoint
    write_checkpoint(tmp_path, config={"model_type": "bark", "torch_dtype": "float16"})
    meta = selection("model.safetensors")
    meta.pop("gguf_file")
    meta.update(task="media", model_type="audio", pipeline_tag="text-to-audio",
                controller_runtime="", backend_family="", input_modalities=[])
    monkeypatch.setattr(imports, "download_model_from_hf", lambda *a, **k: {
        "status": "success", "local_dir": str(tmp_path)})
    result = imports.download_imported_model("owner/chosen", meta)
    assert result["status"] == "success", result
    assert result["metadata"]["model_type"] == "audio"
    assert result["metadata"]["controller_runtime"] == ""
