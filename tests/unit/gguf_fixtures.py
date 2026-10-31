"""Small/sparse GGUF directory fixtures; no actual inference weights are needed."""
from pathlib import Path
import struct


def _string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def write_gguf(path, *, layers=4, names=None, span=64, extra=None, tensor_type=0, payload=True):
    path = Path(path)
    names = (["token_embd.weight", *(f"blk.{i}.attn_q.weight" for i in range(layers)), "output.weight"]
             if names is None else list(names))
    metadata = {"general.architecture": "llama", "llama.block_count": layers,
                "llama.context_length": 4096, "llama.embedding_length": 16,
                "llama.attention.head_count": 4, "llama.attention.head_count_kv": 2,
                "tokenizer.ggml.tokens": [f"t{i}" for i in range(16)], **(extra or {})}
    header = b"GGUF" + struct.pack("<IQQ", 3, len(names), len(metadata))
    for key, value in metadata.items():
        header += _string(key)
        if isinstance(value, str):
            header += struct.pack("<I", 8) + _string(value)
        elif isinstance(value, list):
            subtype = 4 if value and isinstance(value[0], int) else 8
            header += struct.pack("<IIQ", 9, subtype, len(value))
            header += b"".join(struct.pack("<I", item) if subtype == 4 else _string(item) for item in value)
        else:
            header += struct.pack("<IQ", 10, value)
    for index, name in enumerate(names):
        header += _string(name) + struct.pack("<IQIQ", 1, max(1, span // 4), tensor_type, index * span)
    header += bytes(-len(header) % 32)
    with path.open("wb") as handle:
        handle.write(header)
        if payload:
            handle.truncate(len(header) + span * len(names))
    return len(header)
