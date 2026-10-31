"""Runtime selection preserves the user's inspected model and actual precision."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from core.model_import_entry import make_imported_model_entry
from core.model_policy import admit_local_model
from tests.unit.gguf_fixtures import write_gguf


def inspected(folder, name="my-finetune.gguf", *, architecture="qwen4exp", quant=2):
    path = folder / name
    write_gguf(path, tensor_type=quant, extra={
        "general.architecture": architecture, architecture + ".block_count": 4,
        architecture + ".context_length": 32768,
    })
    allowed, reason, info = admit_local_model(str(path))
    assert allowed, reason
    return path, info


def test_auto_keeps_exact_noncanonical_file_and_precision(tmp_path):
    selected, info = inspected(tmp_path)
    neighbour, _ = inspected(tmp_path, "canonical-Q8.gguf", quant=8)
    original = deepcopy(info)
    entry = make_imported_model_entry(info, local_path=str(tmp_path), display_name="Chosen fine-tune")
    assert entry["controller_runtime"] == "strata"
    assert entry["runtime_preference"] == "auto"
    assert entry["model_path"] == entry["strata_model_path"] == str(selected)
    assert [item["path"] for item in entry["strata_source_files"]] == [str(selected)]
    assert entry["precision"] == "GGUF Q4_0" and entry["ternary"] is False
    assert entry["execution_compatibility"] == "unverified"
    assert entry["repo_id"] is None
    assert neighbour.is_file() and info == original


@pytest.mark.parametrize("quant,family", [(34, "gguf"), (35, "gguf"), (143, "prism_gguf")])
def test_true_ternary_is_not_silently_sent_to_strata(tmp_path, quant, family):
    selected, info = inspected(tmp_path, quant=quant)
    entry = make_imported_model_entry(info, local_path=str(selected))
    assert entry["controller_runtime"] == "gguf" and entry["backend_family"] == family
    assert entry["model_path"] == str(selected)
    assert entry["ternary"] is True and entry["precision"] == info["precision"]
    assert "strata_source_id" not in entry
    with pytest.raises(ValueError, match="cannot execute.*tensor encoding"):
        make_imported_model_entry(info, local_path=str(selected), runtime_preference="strata")


def test_mixed_ternary_false_label_still_cannot_hide_unsupported_tensor_type(tmp_path):
    files = []
    for index, quant in enumerate((34, 2)):
        path = tmp_path / f"mixed-{index+1:05d}-of-00002.gguf"
        write_gguf(path, names=[f"blk.{index}.attn_q.weight"], tensor_type=quant, extra={
            "general.architecture": "qwen4exp", "qwen4exp.block_count": 2,
            "split.no": index, "split.count": 2, "split.tensors.count": 2,
        })
        files.append(path)
    allowed, reason, info = admit_local_model(str(files[1]))
    assert allowed, reason
    assert info["ternary"] is False
    entry = make_imported_model_entry(info, local_path=str(files[1]))
    assert entry["controller_runtime"] == "gguf"
    assert entry["model_path"] == str(files[0])
    assert entry["gguf_files"] == [path.name for path in files]
    with pytest.raises(ValueError, match="TQ1_0"):
        make_imported_model_entry(info, local_path=str(files[1]), runtime_preference="strata")


def test_trellis_suffix_does_not_make_the_real_encoding_ternary(tmp_path):
    selected, info = inspected(tmp_path, "Gyro-TQ1_0.gguf", quant=145)
    entry = make_imported_model_entry(info, local_path=str(selected))
    assert entry["controller_runtime"] == "strata"
    assert entry["precision"] == "GGUF TQK6" and entry["ternary"] is False


def test_gguf_v2_remains_native_because_stratas_full_preflight_requires_v3(tmp_path):
    selected, _ = inspected(tmp_path)
    raw = bytearray(selected.read_bytes())
    raw[4:8] = (2).to_bytes(4, "little")
    selected.write_bytes(raw)
    allowed, reason, info = admit_local_model(str(selected))
    assert allowed, reason
    assert info["gguf_metadata"]["gguf_version"] == 2
    entry = make_imported_model_entry(info, local_path=str(selected))
    assert entry["controller_runtime"] == "gguf" and entry["model_path"] == str(selected)
    with pytest.raises(ValueError, match="GGUF v3"):
        make_imported_model_entry(info, local_path=str(selected), runtime_preference="strata")


@pytest.mark.parametrize("architecture,quant,expected", [
    ("qwen4exp", 999, "gguf"), ("qwen35moe", 2, "gguf"),
    ("llama", 2, "gguf"), ("deepseek41", 39, "strata"),
])
def test_auto_uses_actual_architecture_and_engine_family_formats(tmp_path, architecture, quant, expected):
    selected, info = inspected(tmp_path, architecture=architecture, quant=quant)
    entry = make_imported_model_entry(info, local_path=str(selected))
    assert entry["controller_runtime"] == expected
    assert entry["precision"] == info["precision"]


def test_recipe_name_and_global_flags_cannot_override_observed_architecture(tmp_path):
    selected, info = inspected(tmp_path, architecture="llama")
    info.update(strata_candidate=True, architecture="qwen4exp", name="Qwen3.8")
    entry = make_imported_model_entry(info, local_path=str(selected))
    assert entry["controller_runtime"] == "gguf"


def test_automatic_strata_does_not_discard_native_image_capability(tmp_path):
    selected, info = inspected(tmp_path)
    info["input_modalities"] = ["text", "image"]
    entry = make_imported_model_entry(info, local_path=str(selected))
    assert entry["controller_runtime"] == "gguf"
    assert entry["input_modalities"] == ["text", "image"]


def test_explicit_strata_preserves_unknown_local_model_for_authoritative_preflight(tmp_path):
    selected, info = inspected(tmp_path, architecture="future_architecture")
    entry = make_imported_model_entry(info, local_path=str(selected), runtime_preference="strata")
    assert entry["controller_runtime"] == "strata"
    assert entry["model_path"] == str(selected)
    assert entry["execution_compatibility"] == "unverified"
    assert not entry.get("strata_family") and not entry.get("native_strata_model")


def test_native_choice_and_source_survive_serialized_selector_roundtrip(tmp_path):
    selected, info = inspected(tmp_path)
    entry = make_imported_model_entry(info, local_path=str(selected), display_name="Fine-tune",
                                      runtime_preference="native")
    restored = make_imported_model_entry(json.loads(json.dumps(entry)))
    assert restored["runtime_preference"] == "native"
    assert restored["controller_runtime"] == "gguf"
    assert restored["name"] == entry["name"] == "Fine-tune (Custom)"
    assert restored["model_path"] == str(selected)
    assert restored["precision"] == info["precision"]


def test_saved_strata_choice_can_return_to_its_actual_native_backend(tmp_path):
    selected, info = inspected(tmp_path)
    strata = make_imported_model_entry(info, local_path=str(selected), runtime_preference="strata")
    native = make_imported_model_entry(json.loads(json.dumps(strata)), runtime_preference="native")
    assert native["controller_runtime"] == "gguf" and native["backend_family"] == "gguf"
    assert native["model_path"] == str(selected)
    assert native["precision"] == info["precision"]
    assert "strata_source_id" not in native and "model_key" not in native


@pytest.mark.parametrize("preference", ["auto", "native", "strata"])
def test_pending_hub_choice_waits_for_real_downloaded_tensor_inspection(tmp_path, monkeypatch, preference):
    from core import model_import_download
    selected, _ = inspected(tmp_path, "chosen.gguf")
    inspected(tmp_path, "other.gguf", quant=8)
    revision = "a" * 40
    pending = {
        "repo_id": "maker/custom", "repo_revision": revision, "source_revision": revision,
        "source_kind": "hf_fetch", "source_origin": "hf_fetch", "requires_local_validation": True,
        "gguf_file": selected.name, "gguf_files": [selected.name], "controller_runtime": "gguf",
        "backend_family": "gguf", "model_type": "text", "task": "text",
        "download_required_files": [selected.name], "download_allow_patterns": [selected.name],
        "precision": "GGUF; tensor precision verified after download",
    }
    saved = make_imported_model_entry(pending, repo_id="maker/custom", runtime_preference=preference)
    assert saved["runtime_preference"] == preference
    assert saved["model_path"] is None and saved["controller_runtime"] == "gguf"
    assert saved["requires_local_validation"] is True
    calls = []
    def download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return {"status": "success", "local_dir": str(tmp_path)}
    monkeypatch.setattr(model_import_download, "download_model_from_hf", download)
    downloaded = model_import_download.download_imported_model("maker/custom", saved)
    assert downloaded["status"] == "success"
    entry = make_imported_model_entry(downloaded["metadata"], local_path=str(tmp_path),
                                      repo_id="maker/custom", fetched=True)
    assert entry["runtime_preference"] == preference
    assert entry["controller_runtime"] == ("gguf" if preference == "native" else "strata")
    assert entry["model_path"] == str(selected) and entry["precision"] == "GGUF Q4_0"
    assert entry["source_origin"] == "hf_fetch" and entry["source_repo_id"] == "maker/custom"
    assert entry["source_revision"] == revision
    assert calls[0][1]["revision"] == revision
    assert calls[0][1]["required_files"] == [selected.name]


@pytest.mark.parametrize("invalid", ["", "fastest", True, ["native"]])
def test_invalid_runtime_preference_is_an_actionable_validation_error(tmp_path, invalid):
    selected, info = inspected(tmp_path)
    with pytest.raises(ValueError, match="Automatic, Native, or Strata"):
        make_imported_model_entry(info, local_path=str(selected), runtime_preference=invalid)


def test_mismatched_selected_local_file_is_not_silently_replaced(tmp_path):
    _, info = inspected(tmp_path)
    other, _ = inspected(tmp_path, "other.gguf", quant=8)
    with pytest.raises(ValueError, match="differs from the inspected model"):
        make_imported_model_entry(info, local_path=str(other))


def test_uninspected_source_cannot_be_promoted_or_forced_to_strata(tmp_path):
    selected, info = inspected(tmp_path)
    info.pop("gguf_metadata")
    info.update(precision="Ternary by filename", strata_candidate=True)
    assert make_imported_model_entry(info, local_path=str(selected))["controller_runtime"] == "gguf"
    with pytest.raises(ValueError, match="inspected local GGUF"):
        make_imported_model_entry(info, local_path=str(selected), runtime_preference="strata")


def test_checkpoint_keeps_native_runtime_and_cannot_be_relabelled_as_gguf(tmp_path):
    info = {"source_kind": "local", "model_path": str(tmp_path), "task": "text", "model_type": "text",
            "controller_runtime": "mlx_lm", "backend_family": "mlx_lm", "precision": "4-bit MLX",
            "max_context": 8192, "ternary": False}
    entry = make_imported_model_entry(info, local_path=str(tmp_path))
    assert entry["controller_runtime"] == "mlx_lm" and entry["precision"] == "4-bit MLX"
    assert entry["model_path"] == str(tmp_path)
    with pytest.raises(ValueError, match="inspected local GGUF"):
        make_imported_model_entry(info, local_path=str(tmp_path), runtime_preference="strata")
