"""Read exact GGUF metadata and shard identity without materializing weights."""
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from core.gguf_metadata import inspect_gguf
from tests.unit.gguf_fixtures import write_gguf


def test_inspects_tensor_spans_types_and_context_without_reading_payload(tmp_path, monkeypatch):
    path = tmp_path / "large.gguf"
    boundary = write_gguf(path, span=1024 ** 3, tensor_type=143, payload=False)
    original = Path.open
    original_stat = Path.stat

    def logical_stat(member, **kwargs):
        actual = original_stat(member, **kwargs)
        if member == path:
            return SimpleNamespace(st_size=boundary + 6 * 1024 ** 3, st_mode=actual.st_mode)
        return actual

    class HeaderOnly:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def read(self, count=-1):
            assert count >= 0 and self.handle.tell() + count <= boundary, "Read GGUF tensor payload"
            return self.handle.read(count)

    monkeypatch.setattr(Path, "open", lambda value, *args, **kw: HeaderOnly(original(value, *args, **kw)))
    # Do not allocate a multi-GB file on platforms where truncate is not sparse.
    monkeypatch.setattr(Path, "stat", logical_stat)
    meta = inspect_gguf(str(path))
    assert meta["architecture"] == "llama" and meta["context_length"] == 4096
    assert meta["gguf_version"] == 3
    assert meta["layer_count"] == 4 and meta["embedding_length"] == 16
    assert meta["layer_bytes"] == [1024 ** 3] * 4
    assert meta["tensor_type_bytes"] == {"143": 6 * 1024 ** 3}
    assert meta["vocab_size"] == 16 and meta["quantization_types"] == [143]


def test_heterogeneous_head_dimensions_keep_a_conservative_kv_bound(tmp_path):
    path = tmp_path / "hybrid.gguf"
    write_gguf(path, extra={"llama.attention.head_count": [4, 2, 4, 2]})
    meta = inspect_gguf(str(path))
    assert meta["head_count"] == 4 and meta["head_count_min"] == 2
    from core.gguf_placement import plan_gguf_placement
    plan = plan_gguf_placement(meta, context_tokens=1024, hardware={
        "ram_available_bytes": 4 * 1024 ** 3, "ram_total_bytes": 8 * 1024 ** 3,
    })
    assert plan["kv_bytes_estimate"] == 1024 * 2 * (8 + 8) * 2 * 4


def test_complete_shards_include_metadata_only_first_shard_and_later_selection(tmp_path):
    first = tmp_path / "model-00001-of-00002.gguf"
    second = tmp_path / "model-00002-of-00002.gguf"
    common = {"split.count": 2, "split.tensors.count": 6}
    write_gguf(first, names=[], extra={**common, "split.no": 0})
    write_gguf(second, extra={**common, "split.no": 1})
    meta = inspect_gguf(str(second))
    assert meta["model_path"] == str(first)
    assert meta["files"] == [str(first), str(second)]
    assert meta["tensor_bytes"] == 384
    assert meta["split_count"] == 2
    second.unlink()
    with pytest.raises(ValueError, match="incomplete"):
        inspect_gguf(str(first))


def test_split_identity_and_tensor_count_are_checked(tmp_path):
    first = tmp_path / "model-00001-of-00002.gguf"
    second = tmp_path / "model-00002-of-00002.gguf"
    common = {"split.count": 2, "split.tensors.count": 6}
    write_gguf(first, names=[], extra={**common, "split.no": 0})
    write_gguf(second, extra={**common, "split.no": 0})
    with pytest.raises(ValueError, match="identities"):
        inspect_gguf(str(first))
    write_gguf(second, extra={**common, "split.no": 1}, names=["blk.0.attn_q.weight"])
    with pytest.raises(ValueError, match="declared tensor count"):
        inspect_gguf(str(first))
    write_gguf(second, extra={**common, "split.no": 1, "split.tensors.count": 7})
    with pytest.raises(ValueError, match="declared tensor count"):
        inspect_gguf(str(first))


def test_version_two_is_identified_and_split_versions_must_agree(tmp_path):
    path = tmp_path / "model.gguf"
    write_gguf(path)
    with path.open("r+b") as handle:
        handle.seek(4)
        handle.write(struct.pack("<I", 2))
    assert inspect_gguf(str(path))["gguf_version"] == 2
    first = tmp_path / "model-00001-of-00002.gguf"
    second = tmp_path / "model-00002-of-00002.gguf"
    write_gguf(first, names=[], extra={"split.count": 2, "split.no": 0})
    write_gguf(second, extra={"split.count": 2, "split.no": 1})
    with second.open("r+b") as handle:
        handle.seek(4)
        handle.write(struct.pack("<I", 2))
    with pytest.raises(ValueError, match="format version"):
        inspect_gguf(str(first))


@pytest.mark.parametrize("data", [b"not GGUF", b"GGUF" + struct.pack("<IQQ", 3, 1, 1),
                                    b"GGUF" + struct.pack("<IQQ", 3, 300_000, 0),
                                    b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + struct.pack("<Q", 2 ** 63)])
def test_malformed_and_unbounded_headers_fail_before_tensor_read(tmp_path, data):
    path = tmp_path / "bad.gguf"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        inspect_gguf(str(path))


def test_empty_model_and_shard_metadata_conflicts_are_rejected(tmp_path):
    path = tmp_path / "empty.gguf"
    write_gguf(path, names=[])
    with pytest.raises(ValueError, match="nonempty"):
        inspect_gguf(str(path))
    first = tmp_path / "model-00001-of-00002.gguf"
    second = tmp_path / "model-00002-of-00002.gguf"
    write_gguf(first, names=[], extra={"split.count": 2, "split.no": 0})
    write_gguf(second, extra={"split.count": 2, "split.no": 1, "general.architecture": "different"})
    with pytest.raises(ValueError, match="disagree"):
        inspect_gguf(str(first))
