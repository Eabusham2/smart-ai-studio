"""User-selected artifacts are admitted by structure, not a model/precision list."""
import json
from pathlib import Path
import struct
import sys
from types import ModuleType, SimpleNamespace

import pytest

from core.model_policy import admit_hf_model, admit_local_model


def _string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def write_gguf(path, *, architecture="llama", quant=2, shard=0, count=1, context=8192):
    meta = {"general.architecture": architecture, architecture + ".block_count": 2,
            architecture + ".context_length": context, architecture + ".embedding_length": 32,
            architecture + ".attention.head_count": 2}
    if count > 1:
        meta.update({"split.no": shard, "split.count": count, "split.tensors.count": count})
    data = b"GGUF" + struct.pack("<IQQ", 3, 1, len(meta))
    for key, value in meta.items():
        data += _string(key)
        data += struct.pack("<I", 8 if isinstance(value, str) else 4)
        data += _string(value) if isinstance(value, str) else struct.pack("<I", value)
    data += _string(f"blk.{shard}.attn_q.weight") + struct.pack("<IQQIQ", 2, 32, 1, quant, 0)
    data += b"\0" * (-len(data) % 32) + b"\0" * 32
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def write_checkpoint(folder, *, config=None, media=None, metadata=None, dtype="F16"):
    folder.mkdir(parents=True, exist_ok=True)
    if config is not None:
        (folder / "config.json").write_text(json.dumps(config))
    if media is not None:
        (folder / "model_index.json").write_text(json.dumps(media))
    header = {"weight": {"dtype": dtype, "shape": [2], "data_offsets": [0, 4]}}
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    (folder / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 4)
    return folder


def test_ordinary_nonternary_gguf_is_accepted_from_its_real_header(tmp_path):
    path = write_gguf(tmp_path / "A-model-I-want-Q4.gguf")
    allowed, reason, meta = admit_local_model(str(path))
    assert allowed, reason
    assert meta["model_path"] == str(path)
    assert meta["controller_runtime"] == "gguf" and meta["backend_family"] == "gguf"
    assert meta["ternary"] is False and meta["ternary_verified"] is False
    assert meta["precision"] == "GGUF Q4_0"
    assert meta["max_context"] == 8192
    assert meta["execution_compatibility"] == "unverified"


def test_new_architecture_can_register_without_claiming_engine_compatibility(tmp_path):
    path = write_gguf(tmp_path / "custom.gguf", architecture="future_architecture")
    allowed, reason, meta = admit_local_model(str(path))
    assert allowed, reason
    assert meta["architecture"] == "future_architecture"
    assert meta["strata_candidate"] is False
    assert meta["execution_compatibility"] == "unverified"


@pytest.mark.parametrize("quant,ternary,precision", [(145, False, "TQK6"), (143, True, "PTQ1_0"), (34, True, "TQ1_0")])
def test_filename_cannot_change_real_ternary_identity(tmp_path, quant, ternary, precision):
    path = write_gguf(tmp_path / "Definitely-Ternary-TQ1_0.gguf", architecture="qwen4exp", quant=quant)
    allowed, reason, meta = admit_local_model(str(path))
    assert allowed, reason
    assert meta["ternary"] is ternary
    assert meta["ternary_verified"] is ternary
    assert precision in meta["precision"]
    assert meta["strata_candidate"] is True


def test_ambiguous_quants_require_the_users_exact_file(tmp_path):
    write_gguf(tmp_path / "model-Q4.gguf")
    write_gguf(tmp_path / "model-Q8.gguf", quant=8)
    allowed, reason, meta = admit_local_model(str(tmp_path))
    assert not allowed and meta["needs_artifact_selection"]
    assert meta["gguf_choices"] == ["model-Q4.gguf", "model-Q8.gguf"]
    allowed, reason, meta = admit_local_model(str(tmp_path), gguf_file="model-Q8.gguf")
    assert allowed, reason
    assert meta["gguf_files"] == ["model-Q8.gguf"]
    assert meta["precision"] == "GGUF Q8_0"


def test_explicit_later_shard_selects_only_its_complete_group(tmp_path):
    first = write_gguf(tmp_path / "chosen-00001-of-00002.gguf", shard=0, count=2)
    second = write_gguf(tmp_path / "chosen-00002-of-00002.gguf", shard=1, count=2)
    write_gguf(tmp_path / "other.gguf", quant=8)
    allowed, reason, meta = admit_local_model(str(second))
    assert allowed, reason
    assert meta["model_path"] == str(first)
    assert meta["gguf_files"] == [first.name, second.name]


def test_incomplete_shards_are_not_replaced_with_another_model(tmp_path):
    first = write_gguf(tmp_path / "chosen-00001-of-00002.gguf", count=2)
    write_gguf(tmp_path / "other.gguf")
    allowed, reason, _ = admit_local_model(str(first))
    assert not allowed and "missing shards" in reason


def test_hub_style_symlink_group_keeps_original_artifact_names(tmp_path):
    source, snapshot = tmp_path / "blobs", tmp_path / "snapshots"
    source.mkdir()
    snapshot.mkdir()
    for i in range(2):
        blob = write_gguf(source / f"opaque{i}", shard=i, count=2)
        (snapshot / f"chosen-{i+1:05d}-of-00002.gguf").symlink_to(blob)
    allowed, reason, meta = admit_local_model(str(snapshot / "chosen-00002-of-00002.gguf"))
    assert allowed, reason
    assert meta["model_path"] == str(snapshot / "chosen-00001-of-00002.gguf")


def test_invalid_gguf_header_cannot_be_admitted_by_a_ternary_name(tmp_path):
    path = tmp_path / "ternary-bitnet.gguf"
    path.write_bytes(b"not a GGUF")
    allowed, _, meta = admit_local_model(str(path))
    assert not allowed and meta["ok"] is False


def test_selected_unique_projector_keeps_native_image_capability(tmp_path):
    model = write_gguf(tmp_path / "chosen.gguf", architecture="qwen4exp")
    projector = write_gguf(tmp_path / "mmproj-chosen.gguf", architecture="clip")
    allowed, reason, meta = admit_local_model(str(model))
    assert allowed, reason
    assert meta["input_modalities"] == ["image", "text"]
    assert meta["mmproj_file"] == projector.name
    assert meta["projector_metadata"]["model_path"] == str(projector)


def test_invalid_selected_projector_is_not_claimed_as_image_capable(tmp_path):
    model = write_gguf(tmp_path / "chosen.gguf")
    (tmp_path / "mmproj-broken.gguf").write_bytes(b"broken")
    allowed, _, _ = admit_local_model(str(model))
    assert not allowed


def test_nonternary_mlx_checkpoint_is_typed_from_config(tmp_path):
    folder = write_checkpoint(tmp_path / "not-a-model-whitelist", config={
        "model_type": "qwen3", "quantization": {"bits": 4, "group_size": 64}, "max_position_embeddings": 32768})
    allowed, reason, meta = admit_local_model(str(folder))
    assert allowed, reason
    assert meta["controller_runtime"] == "mlx_lm"
    assert meta["backend_family"] == "mlx_lm"
    assert meta["ternary"] is False and "4-bit" in meta["precision"]


def test_qwen48_custom_t5_is_not_mislabeled_as_ordinary_affine_mlx(tmp_path):
    folder = write_checkpoint(tmp_path, config={
        "model_type": "qwen4_exp", "quantization": {"bits": 4, "group_size": 64},
        "omlx_t5": {"format": "base3_5trits_per_byte", "group_size": 128},
    })
    before = (folder / "model.safetensors").read_bytes()
    allowed, reason, meta = admit_local_model(str(folder))
    assert not allowed and meta["ok"] is False
    assert "Qwen48" in reason and "packed ternary loader" in reason
    assert (folder / "model.safetensors").read_bytes() == before


def test_unquantized_safetensors_is_not_called_mlx_from_folder_name(tmp_path):
    folder = write_checkpoint(tmp_path / "My-MLX-ternary-model", config={"model_type": "llama", "torch_dtype": "float16"})
    allowed, reason, meta = admit_local_model(str(folder))
    assert allowed, reason
    assert meta["controller_runtime"] == "transformers_auto"
    assert meta["ternary"] is False and meta["precision"] == "float16"


def test_checkpoint_quantization_claim_does_not_prove_packed_ternary_weights(tmp_path):
    folder = write_checkpoint(tmp_path, config={"model_type": "bitnet", "quantization_config": {"quant_method": "bitnet"}})
    allowed, reason, meta = admit_local_model(str(folder))
    assert allowed, reason
    assert meta["declared_ternary"] is True
    assert meta["ternary_verified"] is False and meta["ternary"] is False
    assert "declared" in meta["precision"]


def test_vision_checkpoint_retains_actual_input_modalities(tmp_path):
    folder = write_checkpoint(tmp_path, config={"model_type": "qwen_vl", "vision_config": {},
                              "text_config": {"model_type": "qwen3", "max_position_embeddings": 16384}})
    allowed, reason, meta = admit_local_model(str(folder))
    assert allowed, reason
    assert meta["input_modalities"] == ["image", "text"]
    assert meta["max_context"] == 16384


def test_media_pipeline_stays_a_media_pipeline(tmp_path):
    folder = write_checkpoint(tmp_path, media={"_class_name": "FluxPipeline"})
    allowed, reason, meta = admit_local_model(str(folder))
    assert allowed, reason
    assert meta["task"] == "media" and meta["model_type"] == "image"
    assert meta["input_modalities"] == [] and meta["controller_runtime"] == ""


def test_local_media_pipeline_keeps_mixed_component_formats_and_nested_index(tmp_path):
    write_checkpoint(tmp_path / "transformer", config={"hidden_size": 2})
    (tmp_path / "transformer" / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model.safetensors"}}))
    (tmp_path / "vae").mkdir()
    (tmp_path / "vae" / "config.json").write_text("{}")
    (tmp_path / "vae" / "diffusion_pytorch_model.bin").write_bytes(b"not deserialized during inspection")
    (tmp_path / "scheduler").mkdir()
    (tmp_path / "scheduler" / "scheduler_config.json").write_text("{}")
    (tmp_path / "model_index.json").write_text(json.dumps({
        "_class_name": "FluxPipeline", "transformer": ["diffusers", "FluxTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"], "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    }))
    write_checkpoint(tmp_path / "unused-quant", config={"model_type": "llama"})
    allowed, reason, meta = admit_local_model(str(tmp_path))
    assert allowed, reason
    assert meta["model_type"] == "image"
    assert meta["weight_files"] == ["transformer/model.safetensors", "vae/diffusion_pytorch_model.bin"]
    assert "scheduler/scheduler_config.json" in meta["companion_files"]
    assert "unused-quant/config.json" not in meta["companion_files"]
    (tmp_path / "vae" / "diffusion_pytorch_model.bin").unlink()
    assert admit_local_model(str(tmp_path))[0] is False


def test_local_repository_mlx_requires_its_actual_loader(tmp_path):
    write_checkpoint(tmp_path, config={"model_type": "qwen_vl", "quantization": {
        "bits": 2, "group_size": 128, "quant_method": "prism_hadamard"}})
    allowed, reason, _ = admit_local_model(str(tmp_path))
    assert not allowed and "runtime/vision_artifact.py" in reason
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "vision_artifact.py").write_text("raise AssertionError('must never execute on inspection')")
    allowed, reason, meta = admit_local_model(str(tmp_path))
    assert allowed, reason
    assert meta["controller_runtime"] == "mlx_repo_vlm"
    assert "runtime/vision_artifact.py" in meta["companion_files"]


def test_config_alone_is_not_a_complete_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    allowed, reason, _ = admit_local_model(str(tmp_path))
    assert not allowed and "No complete model weights" in reason


@pytest.mark.parametrize("weight", ["missing.safetensors", "../outside.safetensors"])
def test_checkpoint_index_must_name_present_files_inside_source(tmp_path, weight):
    write_checkpoint(tmp_path, config={"model_type": "llama"})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": weight}}))
    allowed, _, _ = admit_local_model(str(tmp_path))
    assert not allowed


def test_truncated_safetensors_payload_is_rejected(tmp_path):
    write_checkpoint(tmp_path, config={"model_type": "llama"})
    path = tmp_path / "model.safetensors"
    path.write_bytes(path.read_bytes()[:-2])
    allowed, reason, _ = admit_local_model(str(tmp_path))
    assert not allowed and "truncated safetensors tensor" in reason


def fake_hub(monkeypatch, tmp_path, names, *, config=None, pipeline="text-generation", library="", revision="a" * 40, objects=None, sizes=None):
    downloaded = []
    metadata = {}
    if config is not None:
        path = tmp_path / "remote-config.json"
        path.write_text(json.dumps(config))
        metadata["config.json"] = path
        names = [*names, "config.json"]
    for name, value in (objects or {}).items():
        path = tmp_path / ("remote-" + name.replace("/", "--"))
        path.write_text(json.dumps(value))
        metadata[name] = path
        names = [*names, name]
    info = SimpleNamespace(sha=revision, pipeline_tag=pipeline, library_name=library,
                           siblings=[SimpleNamespace(rfilename=name, size=(sizes or {}).get(name, metadata[name].stat().st_size if name in metadata else 512))
                                     for name in names])
    class Api:
        def model_info(self, repo, *, files_metadata):
            assert repo == "owner/chosen" and files_metadata is True
            return info
    def download(*, repo_id, filename, revision):
        downloaded.append((repo_id, filename, revision))
        return str(metadata[filename])
    module = ModuleType("huggingface_hub")
    module.HfApi, module.hf_hub_download = Api, download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return downloaded


def test_hf_ambiguous_artifacts_require_choice_before_weight_download(tmp_path, monkeypatch):
    calls = fake_hub(monkeypatch, tmp_path, ["model-Q4.gguf", "model-Q8.gguf"])
    allowed, _, meta = admit_hf_model("owner/chosen")
    assert not allowed and meta["needs_artifact_selection"] is True
    assert len(meta["gguf_choices"]) == 2 and calls == []


def test_hf_selection_pins_revision_and_only_selected_shard_downloads(tmp_path, monkeypatch):
    selected = ["Q4/chosen-00001-of-00002.gguf", "Q4/chosen-00002-of-00002.gguf"]
    calls = fake_hub(monkeypatch, tmp_path, [*selected, "Q8/other.gguf", "tokenizer.json", "README.md"])
    allowed, reason, meta = admit_hf_model("owner/chosen", gguf_file=selected[1])
    assert allowed, reason
    assert meta["gguf_file"] == selected[0] and meta["gguf_files"] == selected
    assert meta["download_required_files"] == selected
    assert meta["download_allow_patterns"] == [*selected, "tokenizer.json"]
    assert meta["repo_revision"] == "a" * 40
    assert meta["requires_local_validation"] is True
    assert meta["ternary"] is False and calls == []


def test_hf_mlx_metadata_download_is_revision_pinned(tmp_path, monkeypatch):
    calls = fake_hub(monkeypatch, tmp_path, ["model.safetensors", "tokenizer.json"],
                     config={"model_type": "qwen3", "quantization": {"bits": 4, "group_size": 64}}, library="mlx")
    allowed, reason, meta = admit_hf_model("owner/chosen")
    assert allowed, reason
    assert meta["controller_runtime"] == "mlx_lm" and meta["ternary"] is False
    assert calls == [("owner/chosen", "config.json", "a" * 40)]
    assert meta["download_required_files"] == ["config.json", "model.safetensors", "tokenizer.json"]


def test_hf_unique_projector_preserves_image_intent_and_exact_files(tmp_path, monkeypatch):
    fake_hub(monkeypatch, tmp_path, ["chosen.gguf", "mmproj-chosen.gguf"])
    allowed, reason, meta = admit_hf_model("owner/chosen")
    assert allowed, reason
    assert meta["input_modalities"] == ["image", "text"]
    assert meta["download_required_files"] == ["chosen.gguf", "mmproj-chosen.gguf"]


def test_hf_repository_mlx_fetches_its_pinned_runtime_sources_without_other_weights(tmp_path, monkeypatch):
    calls = fake_hub(monkeypatch, tmp_path, [
        "model.safetensors", "unused-quant/model.safetensors", "runtime/vision_artifact.py",
        "runtime/helpers.py", "runtime/kernel.metal", "runtime/layout.json", "tokenizer.json",
    ], library="mlx", config={"model_type": "qwen_vl", "quantization": {
        "bits": 2, "group_size": 128, "quant_method": "prism_hadamard"}})
    allowed, reason, meta = admit_hf_model("owner/chosen")
    assert allowed, reason
    assert meta["controller_runtime"] == "mlx_repo_vlm"
    assert {"runtime/vision_artifact.py", "runtime/helpers.py", "runtime/kernel.metal", "runtime/layout.json"} <= set(meta["download_required_files"])
    assert "unused-quant/model.safetensors" not in meta["download_allow_patterns"]
    assert calls == [("owner/chosen", "config.json", "a" * 40)]


def test_published_prism_architecture_retains_runtime_and_hadamard_assets(tmp_path, monkeypatch):
    # The publisher puts the runtime marker in model_type, while its ordinary
    # quantization object remains affine. Inspect that structure, not its name.
    config = {"model_type": "prism_hadamard_qwen35", "text_config": {"model_type": "qwen3_5"},
              "vision_config": {}, "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
              "requires_runtime": "runtime/artifact.py", "hadamard_config": "hadamard.json"}
    calls = fake_hub(monkeypatch, tmp_path, [
        "model.safetensors", "runtime/artifact.py", "runtime/vision_artifact.py", "runtime/codec.py",
        "hadamard.json", "files.json", "video_preprocessor_config.json",
    ], config=config, library="mlx")
    allowed, reason, meta = admit_hf_model("owner/chosen")
    assert allowed, reason
    assert meta["controller_runtime"] == "mlx_repo_vlm"
    assert meta["input_modalities"] == ["image", "text"]
    assert "prism_hadamard" in meta["precision"]
    assert {"runtime/artifact.py", "runtime/vision_artifact.py", "runtime/codec.py", "hadamard.json", "files.json",
            "video_preprocessor_config.json"} <= set(meta["download_required_files"])
    assert calls == [("owner/chosen", "config.json", "a" * 40)]
    local = write_checkpoint(tmp_path / "local", config=config)
    (local / "runtime").mkdir()
    for name in ("artifact.py", "vision_artifact.py"):
        (local / "runtime" / name).write_text("raise AssertionError('inspection must not execute this')")
    allowed, reason, _ = admit_local_model(str(local))
    assert not allowed and "hadamard.json" in reason
    (local / "hadamard.json").write_text("{}")
    allowed, reason, meta = admit_local_model(str(local))
    assert allowed, reason
    assert meta["controller_runtime"] == "mlx_repo_vlm"


def test_hf_transformers_auto_map_keeps_same_revision_sibling_source(tmp_path, monkeypatch):
    calls = fake_hub(monkeypatch, tmp_path, [
        "model.safetensors", "modeling_custom.py", "configuration_custom.py", "helpers/attention.py",
        "unused-quant/model.safetensors",
    ], config={"model_type": "custom", "auto_map": {"AutoModelForCausalLM": "modeling_custom.Model"}})
    allowed, reason, meta = admit_hf_model("owner/chosen")
    assert allowed, reason
    assert {"modeling_custom.py", "configuration_custom.py", "helpers/attention.py"} <= set(meta["download_required_files"])
    assert "unused-quant/model.safetensors" not in meta["download_allow_patterns"]
    assert calls == [("owner/chosen", "config.json", "a" * 40)]


def test_hf_media_components_keep_bin_stage_and_nested_indexes_at_one_revision(tmp_path, monkeypatch):
    calls = fake_hub(monkeypatch, tmp_path, [
        "transformer/a.safetensors", "transformer/b.safetensors", "transformer/config.json",
        "vae/diffusion_pytorch_model.bin", "vae/config.json", "scheduler/scheduler_config.json",
        "unused-quant/model.safetensors",
    ], pipeline="text-to-image", objects={
        "model_index.json": {"_class_name": "FluxPipeline", "transformer": ["diffusers", "FluxTransformer2DModel"],
                             "vae": ["diffusers", "AutoencoderKL"], "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"]},
        "transformer/diffusion_pytorch_model.safetensors.index.json": {"weight_map": {"a": "a.safetensors", "b": "b.safetensors"}},
    })
    allowed, reason, meta = admit_hf_model("owner/chosen")
    assert allowed, reason
    assert meta["task"] == "media" and meta["model_type"] == "image"
    assert meta["weight_files"] == ["transformer/a.safetensors", "transformer/b.safetensors", "vae/diffusion_pytorch_model.bin"]
    assert "scheduler/scheduler_config.json" in meta["download_required_files"]
    assert "unused-quant/model.safetensors" not in meta["download_allow_patterns"]
    assert all(revision == "a" * 40 for _, _, revision in calls)
    assert {name for _, name, _ in calls} == {"model_index.json", "transformer/diffusion_pytorch_model.safetensors.index.json"}


def test_hf_runtime_companions_are_bounded_before_weight_download(tmp_path, monkeypatch):
    calls = fake_hub(monkeypatch, tmp_path, ["model.safetensors", "modeling_custom.py"],
                     config={"model_type": "custom"}, sizes={"modeling_custom.py": 9 * 1024 * 1024})
    allowed, reason, _ = admit_hf_model("owner/chosen")
    assert not allowed and "bounded download limit" in reason
    assert calls == [("owner/chosen", "config.json", "a" * 40)]


def test_hf_custom_t5_requires_publisher_runtime_before_weight_download(tmp_path, monkeypatch):
    calls = fake_hub(monkeypatch, tmp_path, ["model.safetensors"], library="mlx", config={
        "model_type": "qwen4_exp", "quantization": {"bits": 4, "group_size": 64},
        "omlx_t5": {"format": "base3_5trits_per_byte"},
    })
    allowed, reason, meta = admit_hf_model("owner/chosen")
    assert not allowed and meta["ok"] is False
    assert "Qwen48" in reason and "packed ternary loader" in reason
    assert calls == [("owner/chosen", "config.json", "a" * 40)]


def test_hf_no_immutable_revision_fails_closed(tmp_path, monkeypatch):
    fake_hub(monkeypatch, tmp_path, ["model.gguf"], revision="main")
    allowed, reason, _ = admit_hf_model("owner/chosen")
    assert not allowed and "immutable revision" in reason


def test_hf_filename_cannot_escape_requested_repository(tmp_path, monkeypatch):
    fake_hub(monkeypatch, tmp_path, ["model.gguf"])
    allowed, reason, _ = admit_hf_model("owner/chosen", gguf_file="../model.gguf")
    assert not allowed and "inside" in reason
