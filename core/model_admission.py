"""Artifact-based admission for user-selected models, without a precision whitelist.

Registration checks format and source identity. Native libraries still decide
whether an architecture can execute; registration never promises a memory fit.
No checkpoint code, pickle or tensor payload is executed during inspection.
"""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
from typing import Any

from core.model_policy import infer_input_modalities, derive_runtime_metadata

_SHARD = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
_MAX_FILES = 8192
_MAX_JSON = 2 * 1024 * 1024
_MAX_SAFE_HEADER = 32 * 1024 * 1024
_MAX_INDEX_BYTES = 16 * 1024 * 1024
_MAX_COMPANION_BYTES = 128 * 1024 * 1024
_SOURCE_SUFFIXES = {".py", ".pyi", ".metal", ".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh", ".m", ".mm"}
_METADATA_NAMES = {
    "config.json", "model_config.json", "model_index.json", "generation_config.json",
    "preprocessor_config.json", "video_preprocessor_config.json", "processor_config.json", "scheduler_config.json",
    "feature_extractor_config.json", "tokenizer_config.json", "special_tokens_map.json",
    "tokenizer.json", "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt",
    "tokenizer.model", "spiece.model", "chat_template.jinja",
}
_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4",
    42: "Q2_0", 143: "PTQ1_0", 144: "TQ2_T", 145: "TQK6", 146: "TQK7",
    147: "TQP5", 148: "TQP6", 149: "TQP7",
}


def _relative_name(value: Any) -> str:
    if not isinstance(value, str) or not value or "\0" in value or "\\" in value:
        raise ValueError("Model artifact names must be relative file paths.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or ":" in path.parts[0]:
        raise ValueError("Model artifact names must stay inside the selected source.")
    return str(path)


def _is_projector(name: str) -> bool:
    return PurePosixPath(name).name.lower().startswith(("mmproj", "projector"))


def _gguf_group(names: list[str], requested: str | None) -> tuple[list[str] | None, dict]:
    language = sorted({n for n in names if n.lower().endswith(".gguf") and not _is_projector(n)})
    groups: dict[tuple, list[str]] = {}
    for name in language:
        match = _SHARD.fullmatch(name)
        key = (match.group(1), match.group(3)) if match else (name, "1")
        groups.setdefault(key, []).append(name)
    choices = [min(group) for group in groups.values()]
    info = {"gguf_choices": sorted(choices), "needs_artifact_selection": False}
    if requested:
        requested = _relative_name(requested)
        if requested not in language:
            raise ValueError("The selected GGUF filename is not a model artifact in this source.")
        selected = requested
    elif len(groups) > 1:
        info["needs_artifact_selection"] = True
        return None, info
    elif not groups:
        return [], info
    else:
        selected = choices[0]
    split = _SHARD.fullmatch(selected)
    if split:
        count, index = int(split.group(3)), int(split.group(2))
        if not 1 <= index <= count <= 256:
            raise ValueError("The selected GGUF has an invalid shard count.")
        files = [f"{split.group(1)}-{i:05d}-of-{count:05d}{PurePosixPath(selected).suffix}" for i in range(1, count + 1)]
        missing = [n for n in files if n not in language]
        if missing:
            raise ValueError("The selected model is missing shards: " + ", ".join(missing[:8]))
    else:
        files = [selected]
    info.update(gguf_file=files[0], gguf_files=files)
    projectors = sorted(n for n in names if n.lower().endswith(".gguf") and _is_projector(n))
    info["mmproj_choices"] = projectors
    info["mmproj_file"] = projectors[0] if len(projectors) == 1 else None
    return files, info


def _read_json(path: Path) -> dict:
    if path.stat().st_size > _MAX_JSON:
        raise ValueError(f"Model metadata exceeds the inspection limit: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Model metadata is not an object: {path.name}")
    return value


def _local_inventory(folder: Path) -> list[str]:
    found = []
    directories = 0
    for parent, dirs, files in os.walk(folder, followlinks=False):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv"}]
        directories += 1
        if directories > 1024 or len(found) + len(files) > _MAX_FILES:
            raise ValueError("This folder contains too many files; choose the exact model folder or GGUF file.")
        found.extend((Path(parent) / name).relative_to(folder).as_posix() for name in files)
    return sorted(found)


def _context(config: dict) -> int | None:
    for row in (config.get("text_config"), config):
        if not isinstance(row, dict):
            continue
        for key in ("max_position_embeddings", "max_sequence_length", "seq_length", "context_length", "n_positions"):
            value = row.get(key)
            if type(value) is int and 1 <= value <= 100_000_000:
                return value
    return None


def _structured_metadata(config: dict, model_index: dict, *, library="", pipeline="") -> dict:
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    arch = str(text.get("model_type") or config.get("model_type") or "")
    architectures = config.get("architectures") or text.get("architectures") or []
    class_name = str(model_index.get("_class_name") or "")
    media = bool(class_name) or any(word in pipeline for word in (
        "text-to-image", "image-to-image", "text-to-video", "image-to-video", "text-to-audio", "audio-generation", "text-to-speech"))
    if media:
        label = (pipeline + " " + class_name).lower()
        model_type = "video" if any(s in label for s in ("video", "wan", "hunyuanvideo")) else (
            "audio" if any(s in label for s in ("audio", "speech", "music", "bark")) else "image")
        return {"task": "media", "model_type": model_type, "pipeline_tag": pipeline,
                "input_modalities": [], "controller_runtime": "", "backend_family": "",
                "ternary": False, "ternary_verified": False, "precision": "From selected media checkpoint"}
    if not arch and not architectures:
        raise ValueError("A checkpoint needs structured architecture metadata in config.json before it can be registered.")
    if isinstance(config.get("omlx_t5"), dict) and config["omlx_t5"].get("format"):
        # The publisher also records affine defaults in `quantization`. Those
        # defaults do not describe the separate packed ternary expert banks.
        # This marker requires the Qwen48 fork's native Bonsai/Metal loader;
        # treating it as ordinary MLX would download a model we cannot execute.
        raise ValueError(
            "This omlx_t5 checkpoint requires the publisher's custom Qwen48 oMLX/Bonsai runtime. "
            "Studio does not implement that packed ternary loader; stock MLX and managed Strata "
            "cannot load this artifact. Its weights and quantization have not been replaced."
        )
    modalities = infer_input_modalities(pipeline, architectures, arch,
                                       "image_input" if "vision_config" in config else "",
                                       "input_audio" if "audio_config" in config else "")
    quant = config.get("quantization") or config.get("quantization_config") or {}
    if not isinstance(quant, dict):
        quant = {}
    method = str(quant.get("quant_method") or quant.get("method") or "")
    prism_format = str(config.get("model_type") or "").startswith("prism_hadamard_") or method == "prism_hadamard"
    if prism_format:
        method = "prism_hadamard"
    mlx = library == "mlx" or (isinstance(config.get("quantization"), dict)
                               and "bits" in config["quantization"] and "group_size" in config["quantization"]) or prism_format
    runtime = "mlx_vlm" if mlx and set(modalities) != {"text"} else "mlx_lm" if mlx else "transformers_auto"
    family = runtime if mlx else "transformers"
    if mlx and method == "prism_hadamard":
        runtime = family = "mlx_repo_vlm"
    declared_ternary = method.lower() in {"bitnet", "ternary", "trit", "bitlinear"}
    bits = quant.get("bits")
    precision = (f"{bits}-bit {method or ('MLX' if mlx else 'quantized')}" if isinstance(bits, (int, float)) and not isinstance(bits, bool)
                 else method or str(config.get("torch_dtype") or config.get("dtype") or "Checkpoint precision not declared"))
    result = {"task": "text", "model_type": "text", "architecture": arch, "architectures": architectures,
              "pipeline_tag": pipeline, "input_modalities": modalities, "controller_runtime": runtime,
              "backend_family": family, "ternary": False, "ternary_verified": False,
              "declared_ternary": declared_ternary,
              "precision": precision + (" (declared; packed ternary format unverified)" if declared_ternary else ""),
              "quantization_config": quant, "execution_compatibility": "unverified"}
    context = _context(config)
    if context:
        result["max_context"] = context
    if prism_format:
        result["runtime_required_files"] = [_relative_name(config[key]) for key in ("requires_runtime", "hadamard_config")
                                            if isinstance(config.get(key), str) and config[key]]
    return result


def _components(model_index: dict) -> list[str]:
    return sorted(_relative_name(name) for name, value in model_index.items()
                  if not name.startswith("_") and isinstance(value, (list, tuple))
                  and len(value) >= 2 and value[0] and value[1])


def _checkpoint_index_names(names: list[str], model_index: dict) -> list[str]:
    directories = set(_components(model_index)) or {"."}
    result = [name for name in names if PurePosixPath(name).parent.as_posix() in directories
              and name.endswith((".safetensors.index.json", ".bin.index.json"))]
    if len(result) > 128:
        raise ValueError("The model has too many checkpoint indexes to inspect safely.")
    return sorted(result)


def _checkpoint_files(names: list[str], config: dict, indices: dict[str, dict], model_index=None) -> list[str]:
    """Choose one storage representation per declared pipeline component."""
    components = _components(model_index or {})
    weights = []
    for directory in components or ["."]:
        present = [name for name in names if PurePosixPath(name).parent.as_posix() == directory]
        if components and not any(name.startswith(directory + "/") for name in names):
            raise ValueError(f"The pipeline is missing its declared component: {directory}")
        candidates = [name for name in indices if PurePosixPath(name).parent.as_posix() == directory]
        if candidates:
            # A pipeline can mix safetensors and .bin components. Choosing one
            # format for the entire repository would silently drop some stages.
            chosen = min(candidates, key=lambda name: (not name.endswith(".safetensors.index.json"), name))
            mapping = indices[chosen].get("weight_map")
            if not isinstance(mapping, dict) or not mapping or not all(isinstance(v, str) for v in mapping.values()):
                raise ValueError("The checkpoint index does not contain a usable weight_map.")
            selected = sorted({str(PurePosixPath(chosen).parent / _relative_name(v)) for v in mapping.values()})
            if any(name not in names for name in selected):
                raise ValueError("The checkpoint is missing files named by its weight index.")
        else:
            selected = [name for name in present if name.endswith(".safetensors")
                        and "adapter_model" not in PurePosixPath(name).name]
            if not selected:
                selected = [name for name in present if name.endswith(".npz")]
            if not selected:
                selected = [name for name in present if name.endswith(".bin")
                            and PurePosixPath(name).name.startswith(("pytorch_model", "diffusion_pytorch_model", "model.bin"))]
        if components and not selected and directory in {
            "unet", "transformer", "vae", "text_encoder", "text_encoder_2", "text_encoder_3",
            "controlnet", "prior", "decoder", "vocoder", "audio_encoder",
        }:
            raise ValueError(f"The pipeline component has no complete model weights: {directory}")
        weights.extend(selected)
    if not weights:
        raise ValueError("No complete model weights were found; a config or adapter alone is not a model.")
    return sorted(set(weights))


def _companion_files(names: list[str], metadata: dict, model_index: dict, *, checkpoint: bool) -> list[str]:
    """Keep exact, same-revision loader assets without allowing another weight set."""
    directories = {".", *_components(model_index)}
    companions = []
    for name in names:
        path = PurePosixPath(name)
        if path.name in _METADATA_NAMES and (path.parent.as_posix() in directories or name.startswith("tokenizer/")):
            companions.append(name)
        elif checkpoint and path.suffix.lower() in _SOURCE_SUFFIXES:
            # Transformers auto_map and tokenizer code can import sibling source
            # modules. Existing repository MLX runtimes also carry Metal sources.
            companions.append(name)
        elif checkpoint and name.startswith("runtime/") and path.suffix.lower() in {".json", ".txt", ".toml", ".yaml", ".yml"}:
            companions.append(name)
        elif checkpoint and metadata.get("controller_runtime") == "mlx_repo_vlm" and path.parent.as_posix() == "." and path.suffix == ".json":
            companions.append(name)
    if checkpoint:
        companions.extend(_checkpoint_index_names(names, model_index))
    if metadata.get("controller_runtime") == "mlx_repo_vlm" and "runtime/vision_artifact.py" not in names:
        raise ValueError("This repository MLX model requires its bundled runtime/vision_artifact.py loader.")
    for name in metadata.get("runtime_required_files") or ():
        name = _relative_name(name)
        if name not in names or PurePosixPath(name).suffix.lower() not in _SOURCE_SUFFIXES | {".json", ".txt", ".toml", ".yaml", ".yml"}:
            raise ValueError(f"A declared runtime companion is missing or unsupported: {name}")
        companions.append(name)
    return sorted(set(companions))


def _check_safetensors(path: Path) -> dict:
    """Validate bounded header/payload extents; never load arrays or pickle data."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors file: {path.name}")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_size <= min(_MAX_SAFE_HEADER, size - 8):
            raise ValueError(f"Invalid or oversized safetensors header: {path.name}")
        header = json.loads(stream.read(header_size).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header: {path.name}")
    extents, dtypes = [], set()
    for name, row in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(row, dict):
            raise ValueError(f"Invalid tensor descriptor: {path.name}")
        offsets, shape = row.get("data_offsets"), row.get("shape")
        if (not isinstance(offsets, list) or len(offsets) != 2 or any(type(v) is not int for v in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= size - 8 - header_size
                or not isinstance(shape, list) or any(type(v) is not int or v < 0 for v in shape)
                or not isinstance(row.get("dtype"), str)):
            raise ValueError(f"Invalid or truncated safetensors tensor: {path.name}")
        extents.append(tuple(offsets))
        dtypes.add(row["dtype"])
    extents.sort()
    if not extents or any(a[1] > b[0] for a, b in zip(extents, extents[1:])):
        raise ValueError(f"Missing or overlapping safetensors tensors: {path.name}")
    return {"dtypes": sorted(dtypes), "metadata": header.get("__metadata__") or {}}


def _gguf_metadata(path: Path, group: dict, root: Path) -> dict:
    from core.gguf_metadata import inspect_gguf
    checked = inspect_gguf(str(path))
    projector = inspect_gguf(str(root / group["mmproj_file"])) if group.get("mmproj_file") else None
    observed = set(checked.get("quantization_types") or [])
    precision = "GGUF " + ", ".join(_TYPE_NAMES.get(t, f"type {t}") for t in sorted(observed))
    # Trellis formats (144..149) are not ternary, regardless of filenames.
    true_ternary = bool(observed & {34, 35, 143}) and not (observed - {0, 1, 24, 25, 26, 27, 28, 30, 34, 35, 143})
    architecture = str(checked.get("architecture") or "")
    family = "prism_gguf" if 143 in observed else "bitnet" if architecture.lower().startswith("bitnet") else "gguf"
    meta = {**group, "ok": True, "task": "text", "model_type": "text", "input_modalities": ["image", "text"] if projector else ["text"],
            "path": str(root), "model_path": checked["model_path"], "source_kind": "local",
            "architecture": architecture, "controller_runtime": "bitnet" if family == "bitnet" else "gguf",
            "backend_family": family, "prism_llama_fork": family == "prism_gguf",
            "ternary": true_ternary, "ternary_verified": true_ternary, "precision": precision,
            "gguf_metadata": checked, "execution_compatibility": "unverified", "requires_local_validation": False}
    if projector:
        meta["projector_metadata"] = projector
    meta["gguf_files"] = [os.path.relpath(p, root).replace(os.sep, "/") for p in checked["files"]]
    meta["gguf_file"] = meta["gguf_files"][0]
    if type(checked.get("context_length")) is int and checked["context_length"] > 0:
        meta["max_context"] = checked["context_length"]
    # Eligibility is an inspection route, never a compatibility assertion.
    meta["strata_candidate"] = architecture in {"qwen4exp", "qwen35moe", "deepseek41"}
    return meta


def admit_local_model(path: str, gguf_file: str | None = None) -> tuple[bool, str, dict]:
    try:
        if not isinstance(path, (str, os.PathLike)) or not str(path).strip() or "\0" in str(path):
            raise ValueError("Choose a local model path.")
        selected = Path(os.path.abspath(os.path.expanduser(str(path))))
        if not selected.exists():
            raise ValueError("Local model path does not exist.")
        root = selected if selected.is_dir() else selected.parent
        names = _local_inventory(root) if selected.is_dir() else [p.name for p in root.iterdir() if p.is_file()]
        requested = gguf_file
        if selected.is_file() and selected.suffix.lower() == ".gguf":
            if requested and os.path.abspath(str(root / requested)) != str(selected):
                raise ValueError("The selected GGUF conflicts with the local file path.")
            requested = selected.name
        if requested and Path(requested).is_absolute():
            requested = Path(requested).relative_to(root).as_posix()
        files, group = _gguf_group(names, requested)
        if files is None:
            return False, "Choose the exact GGUF filename; this source contains multiple model artifacts.", {"ok": False, "path": str(root), **group}
        if files:
            meta = _gguf_metadata(root / files[0], group, root)
            return True, "Selected GGUF structure verified; runtime support and memory placement are checked when loading.", meta
        if selected.is_file() and selected.name not in {"config.json", "model_index.json"} and selected.suffix.lower() not in {".safetensors", ".npz", ".bin"}:
            raise ValueError("Choose a model directory, GGUF, or checkpoint weight file.")
        config = _read_json(root / "config.json") if (root / "config.json").is_file() else {}
        model_index = _read_json(root / "model_index.json") if (root / "model_index.json").is_file() else {}
        index_names = _checkpoint_index_names(names, model_index)
        if sum((root / name).stat().st_size for name in index_names) > _MAX_INDEX_BYTES:
            raise ValueError("Checkpoint indexes exceed the aggregate inspection limit.")
        indices = {name: _read_json(root / name) for name in index_names}
        weights = _checkpoint_files(names, config, indices, model_index)
        dtypes, formats = set(), set()
        for name in weights:
            weight = root / name
            if not weight.is_file() or weight.stat().st_size <= 0:
                raise ValueError(f"Missing or empty model weight: {name}")
            if name.endswith(".safetensors"):
                checked = _check_safetensors(weight)
                dtypes.update(checked["dtypes"])
                if isinstance(checked["metadata"], dict):
                    formats.add(str(checked["metadata"].get("format") or ""))
        meta = _structured_metadata(config, model_index, library="mlx" if "mlx" in formats else "")
        companions = _companion_files(names, meta, model_index, checkpoint=True)
        if sum((root / name).stat().st_size for name in companions) > _MAX_COMPANION_BYTES:
            raise ValueError("Runtime and metadata companions exceed the bounded inspection limit.")
        if dtypes and meta["precision"] == "Checkpoint precision not declared":
            meta["precision"] = "SafeTensors " + ", ".join(sorted(dtypes))
        meta.update(ok=True, path=str(root), model_path=str(root), source_kind="local", weight_files=weights,
                    companion_files=companions,
                    requires_local_validation=False, needs_artifact_selection=False)
        return True, "Selected checkpoint metadata and files verified; backend compatibility is checked when loading.", meta
    except (OSError, ValueError, TypeError, KeyError) as error:
        return False, str(error), {"ok": False, "path": str(path), "reason": str(error)}


def admit_hf_model(repo_id: str, gguf_file: str | None = None) -> tuple[bool, str, dict]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
        repo_id = str(repo_id or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id):
            raise ValueError("Use a Hugging Face model repository ID in owner/name form.")
        info = HfApi().model_info(repo_id, files_metadata=True)
        revision = str(getattr(info, "sha", "") or "")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError("The model repository did not provide an immutable revision.")
        siblings = list(getattr(info, "siblings", None) or [])
        if len(siblings) > _MAX_FILES:
            raise ValueError("The repository has too many artifacts; choose a dedicated model repository.")
        rows = {str(getattr(row, "rfilename", "")): row for row in siblings}
        names = sorted(_relative_name(n) for n in rows if n)
        pipeline = str(getattr(info, "pipeline_tag", "") or "")
        library = str(getattr(info, "library_name", "") or "")
        files, group = _gguf_group(names, gguf_file)
        common = {"repo_id": repo_id, "repo_revision": revision, "source_revision": revision,
                  "source_repo_id": repo_id, "source_kind": "hf_fetch", "source_origin": "hf_fetch",
                  "pipeline_tag": pipeline, "requires_local_validation": True, **group}
        if files is None:
            return False, "Choose the exact GGUF filename; this repository contains multiple model artifacts.", {"ok": False, **common}
        config, model_index, indices = {}, {}, {}
        inspected_bytes = 0
        def read_remote_object(name):
            nonlocal inspected_bytes
            size = getattr(rows[name], "size", None)
            if type(size) is not int or not 0 < size <= _MAX_JSON:
                raise ValueError(f"Repository metadata exceeds the inspection limit: {name}")
            inspected_bytes += size
            if inspected_bytes > _MAX_INDEX_BYTES:
                raise ValueError("Checkpoint metadata exceeds the aggregate inspection limit.")
            local = hf_hub_download(repo_id=repo_id, filename=name, revision=revision)
            return _read_json(Path(local))
        for name in (() if files else ("config.json", "model_index.json")):
            if name not in rows:
                continue
            value = read_remote_object(name)
            if name == "config.json":
                config = value
            elif name == "model_index.json":
                model_index = value
        if not files:
            indices = {name: read_remote_object(name) for name in _checkpoint_index_names(names, model_index)}
        if files:
            # No card/repository name can prove a selected GGUF's tensor types.
            # The exact downloaded group is structurally inspected before use.
            runtime = derive_runtime_metadata(json.dumps(config), file_names=files, repo_id=repo_id, ternary=False)
            meta = {**runtime, **common, "task": "text", "model_type": "text", "input_modalities": ["image", "text"] if group.get("mmproj_file") else ["text"],
                    "controller_runtime": "gguf", "backend_family": "gguf", "prism_llama_fork": False,
                    "ternary": False, "ternary_verified": False, "precision": "GGUF; tensor precision verified after download"}
            required = list(files)
            if group.get("mmproj_file"):
                required.append(group["mmproj_file"])
        else:
            meta = {**_structured_metadata(config, model_index, library=library, pipeline=pipeline), **common}
            required = _checkpoint_files(names, config, indices, model_index)
            meta["weight_files"] = list(required)
        metadata_files = _companion_files(names, meta, model_index, checkpoint=not files)
        companion_bytes = 0
        for name in metadata_files:
            size = getattr(rows[name], "size", None)
            limit = 8 * 1024 * 1024 if PurePosixPath(name).suffix.lower() in _SOURCE_SUFFIXES else 64 * 1024 * 1024
            if type(size) is not int or not 0 <= size <= limit:
                raise ValueError(f"A model runtime/metadata companion exceeds the bounded download limit: {name}")
            companion_bytes += size
        if companion_bytes > _MAX_COMPANION_BYTES:
            raise ValueError("Model runtime and metadata companions exceed the bounded download limit.")
        if not files:
            required = sorted(set(required + metadata_files))
        meta.update(ok=True, download_required_files=required,
                    download_allow_patterns=sorted(set(required + metadata_files)),
                    companion_files=metadata_files,
                    execution_compatibility="unverified")
        return True, "Selected source and artifacts verified; the downloaded files must pass local inspection before loading.", meta
    except Exception as error:
        return False, f"Could not verify the selected Hugging Face model: {error}", {"ok": False, "repo_id": str(repo_id), "reason": str(error)}
