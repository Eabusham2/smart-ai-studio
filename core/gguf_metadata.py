"""Bounded GGUF directory inspection; tensor payloads are never read.

Storage spans include alignment padding. They are conservative inputs to a
placement estimate, not a promise about a backend's allocations or inference speed.
"""
from __future__ import annotations

from pathlib import Path
import os
import re
import struct


MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_TENSORS = 250_000
MAX_METADATA = 100_000
MAX_ARRAY_ITEMS = 2_000_000
MAX_SHARDS = 128
_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_SUFFIXES = (
    ".block_count", ".context_length", ".embedding_length", ".attention.head_count",
    ".attention.head_count_kv", ".attention.key_length", ".attention.value_length", ".expert_count",
)
_SHARD = re.compile(r"^(.*)-(\d{5})-of-(\d{5})(\.gguf)$", re.IGNORECASE)


class _Reader:
    def __init__(self, handle, size, limit):
        self.handle, self.size, self.limit = handle, size, min(size, limit)

    def read(self, count):
        if count < 0 or self.handle.tell() + count > self.limit:
            raise ValueError("GGUF metadata exceeds its bounded header limit or is truncated.")
        data = self.handle.read(count)
        if len(data) != count:
            raise ValueError("GGUF metadata is truncated.")
        return data

    def skip(self, count):
        if count < 0 or self.handle.tell() + count > self.limit:
            raise ValueError("GGUF metadata exceeds its bounded header limit or is truncated.")
        self.handle.seek(count, 1)

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def string(self, keep=False, maximum=4096):
        length = self.number("Q")
        if keep:
            if length > maximum:
                raise ValueError("GGUF metadata string is too long.")
            try:
                return self.read(length).decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError("GGUF metadata contains invalid UTF-8.") from None
        self.skip(length)
        return None

    def value(self, kind, keep=False):
        if kind in _SCALARS:
            return self.number(_SCALARS[kind])
        if kind == 8:
            return self.string(keep)
        if kind != 9:
            raise ValueError("Unsupported GGUF metadata value type.")
        subtype, count = self.number("I"), self.number("Q")
        if count > MAX_ARRAY_ITEMS or subtype not in (*_SCALARS, 8):
            raise ValueError("Invalid or oversized GGUF metadata array.")
        if keep and count <= 4096 and subtype in _SCALARS:
            return [self.number(_SCALARS[subtype]) for _ in range(count)]
        if subtype in _SCALARS:
            self.skip(count * struct.calcsize("<" + _SCALARS[subtype]))
        else:
            for _ in range(count):
                self.string()
        return {"array_count": count}


def _positive(value, default=0):
    if isinstance(value, list):
        return max((_positive(item) for item in value), default=default)
    return value if type(value) is int and value > 0 else default


def _minimum_positive(value):
    values = value if isinstance(value, list) else [value]
    return min((item for item in values if type(item) is int and item > 0), default=0)


def _single(path: Path, header_budget: int):
    size = path.stat().st_size
    metadata, tensors = {}, []
    with path.open("rb") as handle:
        reader = _Reader(handle, size, header_budget)
        if reader.read(4) != b"GGUF":
            raise ValueError("The selected file is not a GGUF model.")
        version = reader.number("I")
        if version not in (2, 3):
            raise ValueError("This inspector supports little-endian GGUF versions 2 and 3.")
        tensor_count, metadata_count = reader.number("Q"), reader.number("Q")
        if tensor_count > MAX_TENSORS or metadata_count > MAX_METADATA:
            raise ValueError("Invalid or oversized GGUF tensor/metadata directory.")
        seen = set()
        for _ in range(metadata_count):
            key, kind = reader.string(True), reader.number("I")
            if key in seen:
                raise ValueError("GGUF metadata contains duplicate keys.")
            seen.add(key)
            keep = key.startswith("split.") or key in {"general.architecture", "general.alignment"} or key.endswith(_SUFFIXES)
            value = reader.value(kind, keep)
            if keep:
                metadata[key] = value
            elif key == "tokenizer.ggml.tokens" and isinstance(value, dict):
                metadata["vocab_size"] = value["array_count"]
        names = set()
        for _ in range(tensor_count):
            name, dimensions = reader.string(True), reader.number("I")
            if name in names or not 1 <= dimensions <= 4:
                raise ValueError("GGUF tensor names or dimensions are invalid.")
            names.add(name)
            shape = [reader.number("Q") for _ in range(dimensions)]
            if any(value <= 0 for value in shape):
                raise ValueError("GGUF tensor dimensions must be positive.")
            kind, offset = reader.number("I"), reader.number("Q")
            tensors.append({"name": name, "shape": shape, "type": kind, "offset": offset})
        alignment = metadata.get("general.alignment", 32)
        if type(alignment) is not int or not 1 <= alignment <= 1024 * 1024 or alignment & (alignment - 1):
            raise ValueError("GGUF tensor alignment is invalid.")
        data_start = (handle.tell() + alignment - 1) // alignment * alignment if tensors else handle.tell()
        header_bytes = data_start
    tensors.sort(key=lambda item: item["offset"])
    for index, tensor in enumerate(tensors):
        end = tensors[index + 1]["offset"] if index + 1 < len(tensors) else size - data_start
        if tensor["offset"] % alignment or end <= tensor["offset"] or data_start + end > size:
            raise ValueError("GGUF tensor offsets are overlapping, misaligned, or outside the file.")
        tensor["bytes"] = end - tensor["offset"]
    return {"version": version, "metadata": metadata, "tensors": tensors,
            "file_bytes": size, "header_bytes": header_bytes}


def inspect_gguf(path: str) -> dict:
    """Inspect an exact local GGUF and all required shards without reading weights.

    An explicit later shard resolves to the first member of the same complete
    group. Missing or conflicting members raise instead of choosing another model.
    """
    # HF snapshot members can be symlinks into content-addressed blobs. Preserve
    # the chosen snapshot filenames when finding the other shards.
    selected = Path(os.path.abspath(os.path.expanduser(path)))
    if not selected.is_file():
        raise ValueError("Select a GGUF file, not an ambiguous model directory.")
    first = _single(selected, MAX_HEADER_BYTES)
    count = first["metadata"].get("split.count", 1)
    if type(count) is not int or not 1 <= count <= MAX_SHARDS:
        raise ValueError("GGUF split count is invalid or unsupported.")
    match = _SHARD.match(selected.name)
    if count > 1:
        if match is None or int(match[3]) != count:
            raise ValueError("Split GGUF filenames must identify their complete shard group.")
        files = [selected.with_name(f"{match[1]}-{number:05d}-of-{count:05d}{match[4]}")
                 for number in range(1, count + 1)]
    else:
        if match is not None and int(match[3]) > 1:
            raise ValueError("GGUF split metadata does not match its filename.")
        files = [selected]
    if any(not member.is_file() for member in files):
        raise ValueError("The selected split GGUF is incomplete; every shard is required.")
    budget = MAX_HEADER_BYTES - first["header_bytes"]
    records = []
    for number, member in enumerate(files):
        record = first if member == selected else _single(member, budget)
        if member != selected:
            budget -= record["header_bytes"]
        if budget < 0:
            raise ValueError("GGUF shard metadata exceeds the total bounded header limit.")
        meta = record["metadata"]
        if count > 1 and (meta.get("split.count") != count or meta.get("split.no") != number):
            raise ValueError("GGUF shard identities do not match the selected group.")
        records.append(record)
    metadata = records[0]["metadata"]
    architecture = metadata.get("general.architecture")
    if not isinstance(architecture, str) or not architecture:
        raise ValueError("GGUF general.architecture is missing.")
    tensors = []
    names = set()
    for record in records:
        if record["version"] != records[0]["version"]:
            raise ValueError("GGUF shards disagree on their format version.")
        for key, value in record["metadata"].items():
            if key in metadata and not key.startswith("split.") and metadata[key] != value:
                raise ValueError(f"GGUF shards disagree on {key}.")
        for tensor in record["tensors"]:
            if tensor["name"] in names:
                raise ValueError("GGUF shards contain duplicate tensor names.")
            names.add(tensor["name"])
            tensors.append(tensor)
    if not 0 < len(tensors) <= MAX_TENSORS:
        raise ValueError("GGUF shard group must contain a bounded, nonempty tensor directory.")
    for record in records:
        declared_tensors = record["metadata"].get("split.tensors.count")
        if declared_tensors is not None and declared_tensors != len(tensors):
            raise ValueError("GGUF shard group does not contain its declared tensor count.")
    layer_count = _positive(metadata.get(architecture + ".block_count"))
    if layer_count > 100_000:
        raise ValueError("GGUF layer count is too large.")
    layers = [0] * layer_count
    non_layer, output = 0, 0
    by_type = {}
    embedding, vocab = 0, _positive(metadata.get("vocab_size"))
    for tensor in tensors:
        size, name = tensor["bytes"], tensor["name"]
        key = str(tensor["type"])
        by_type[key] = by_type.get(key, 0) + size
        layer = re.match(r"^blk\.(\d+)\.", name)
        if layer and int(layer[1]) < layer_count:
            layers[int(layer[1])] += size
        else:
            non_layer += size
        if name.startswith("output.") or name.startswith("output_norm."):
            output += size
        if name == "token_embd.weight" and len(tensor["shape"]) == 2:
            embedding, vocab = tensor["shape"][0], vocab or tensor["shape"][1]
    expert_count = _positive(metadata.get(architecture + ".expert_count"))
    return {
        "model_path": str(files[0]), "files": [str(member) for member in files],
        "gguf_version": records[0]["version"],
        "architecture": architecture, "context_length": _positive(metadata.get(architecture + ".context_length")),
        "layer_count": layer_count, "embedding_length": _positive(metadata.get(architecture + ".embedding_length"), embedding),
        "head_count": _positive(metadata.get(architecture + ".attention.head_count")),
        "head_count_min": _minimum_positive(metadata.get(architecture + ".attention.head_count")),
        "head_count_kv": _positive(metadata.get(architecture + ".attention.head_count_kv")),
        "key_length": _positive(metadata.get(architecture + ".attention.key_length")),
        "value_length": _positive(metadata.get(architecture + ".attention.value_length")),
        "vocab_size": vocab, "is_moe": bool(expert_count), "expert_count": expert_count,
        "quantization_types": sorted(int(key) for key in by_type), "tensor_type_bytes": by_type,
        "layer_bytes": layers, "non_layer_bytes": non_layer, "output_bytes": output,
        "tensor_bytes": sum(by_type.values()), "file_bytes": sum(record["file_bytes"] for record in records),
        "split_count": count,
    }
