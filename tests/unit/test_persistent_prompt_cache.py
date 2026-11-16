"""File-backed cache contracts using fake tensors only; no model/runtime import."""
import base64
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("persistent_prompt_cache_test", ROOT / "core/persistent_prompt_cache.py")
cache_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_module)


class Tensor:
    def __init__(self, data, shape=(1, 2, 4, 2), dtype="float16"):
        self.data = bytes(data)
        self.shape = shape
        self.dtype = dtype
        self.nbytes = len(self.data)


class KVCache:
    step = 256
    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0


class FakeMX:
    array = Tensor
    """Stand-in for the nonexecuting safetensors I/O and tensor evaluation APIs."""
    def eval(self, *arrays):
        pass

    def save_safetensors(self, path, arrays):
        Path(path).write_text(json.dumps({key: {"data": base64.b64encode(value.data).decode(),
                                               "shape": value.shape, "dtype": value.dtype}
                                         for key, value in arrays.items()}))

    def load(self, path):
        return {key: Tensor(base64.b64decode(value["data"]), tuple(value["shape"]), value["dtype"])
                for key, value in json.loads(Path(path).read_text()).items()}


def identity(**changes):
    values = dict(model={"sha256": "weights-a"}, tokenizer={"sha256": "tokenizer-a", "template": "v1"},
                  adapter=None, cache_policy={"type": "native-kv", "precision": "unchanged"},
                  versions={"mlx": "1.0", "mlx-lm": "2.0", "codec": "1"})
    values.update(changes)
    return cache_module.prompt_cache_identity(**values)


def layers(tokens=(1, 2, 3), *, number=2, seed=0):
    result = []
    for index in range(number):
        layer = KVCache()
        layer.keys = Tensor(bytes([seed + index]) * 32)
        layer.values = Tensor(bytes([seed + index + 1]) * 32)
        layer.offset = len(tokens)
        if index:
            layer.step = 64
        result.append(layer)
    return result


def store(tmp_path, **kwargs):
    return cache_module.PersistentPromptStore(tmp_path, enabled=True,
                                              codec=cache_module.SafeKVCodec(FakeMX(), KVCache), **kwargs)


def test_disabled_store_never_creates_directory_or_imports_runtime(tmp_path):
    target = tmp_path / "disabled"
    disk = cache_module.PersistentPromptStore(target)
    with patch.object(cache_module, "SafeKVCodec", side_effect=AssertionError("runtime import")):
        assert disk.save((1, 2, 3), layers(), identity()) is False
        assert disk.load((1, 2, 3, 4), identity()) is None
    assert not target.exists()


def test_exact_native_roundtrip_is_bit_identical_independent_and_preserves_allocation(tmp_path):
    original = layers()
    disk = store(tmp_path)
    assert disk.save((1, 2, 3), original, identity())
    restored_tokens, restored = disk.load((1, 2, 3, 9), identity())
    assert restored_tokens == (1, 2, 3)
    for old, new in zip(original, restored):
        assert old is not new
        assert old.keys is not new.keys
        assert old.values is not new.values
        assert vars(old) == {**vars(new), "keys": old.keys, "values": old.values}
        assert old.keys.data == new.keys.data
        assert old.values.data == new.values.data
        assert old.keys.shape == new.keys.shape == (1, 2, 4, 2)
        assert old.keys.dtype == new.keys.dtype
    restored[0].offset = 999
    assert disk.load((1, 2, 3, 9), identity())[1][0].offset == 3


def test_identity_invalidates_every_state_input_and_unknown_versions(tmp_path):
    disk = store(tmp_path)
    assert disk.save((1, 2, 3), layers(), identity())
    changes = [dict(model={"sha256": "weights-b"}), dict(tokenizer={"sha256": "tokenizer-b"}),
               dict(adapter={"sha256": "adapter-b"}), dict(cache_policy={"precision": "different"}),
               dict(versions={"mlx": "changed"})]
    for change in changes:
        assert disk.load((1, 2, 3, 4), identity(**change)) is None
    invalid = {**identity(), "versions": {"mlx": "unknown"}}
    assert not disk.save((1, 2, 3), layers(), invalid)


def test_longest_strict_prefix_and_corrupt_longest_falls_back(tmp_path):
    disk = store(tmp_path)
    assert disk.save((1, 2), layers((1, 2), seed=8), identity())
    assert disk.save((1, 2, 3), layers(), identity())
    assert disk.load((1, 2, 3, 4), identity())[0] == (1, 2, 3)
    assert disk.load((1, 2, 3), identity())[0] == (1, 2)
    assert disk.load((1, 8, 3), identity()) is None
    for path in tmp_path.glob("*.json"):
        manifest = json.loads(path.read_text())
        if len(manifest["tokens"]) == 3:
            (tmp_path / manifest["tensor_file"]).write_bytes(b"corrupt")
    assert disk.load((1, 2, 3, 4), identity())[0] == (1, 2)


def test_unknown_cache_types_extra_state_and_wrong_offsets_are_rejected(tmp_path):
    disk = store(tmp_path)
    class CustomKV(KVCache):
        pass
    wrong_type = layers()
    wrong_type[0].__class__ = CustomKV
    assert not disk.save((1, 2, 3), wrong_type, identity())
    unknown_state = layers()
    unknown_state[0].rolling_position = 1
    assert not disk.save((1, 2, 3), unknown_state, identity())
    wrong_offset = layers()
    wrong_offset[0].offset = 2
    assert not disk.save((1, 2, 3), wrong_offset, identity())


def test_content_checksum_detects_valid_looking_tensor_bit_flip(tmp_path):
    disk = store(tmp_path)
    assert disk.save((1, 2, 3), layers(), identity())
    payload = next(tmp_path.glob("*.safetensors"))
    data = payload.read_bytes()
    payload.write_bytes(data.replace(b"AAAA", b"BBBB", 1))
    assert payload.stat().st_size == len(data)
    assert disk.load((1, 2, 3, 4), identity()) is None


def test_malicious_or_malformed_metadata_never_selects_class_or_external_path(tmp_path):
    for update in ({"tensor_file": "../../outside.safetensors"}, {"schema": 999},
                   {"descriptor": {"codec": "pickle", "class": "os.system"}}):
        disk = store(tmp_path)
        assert disk.save((1, 2, 3), layers(), identity())
        path = next(tmp_path.glob("*.json"))
        manifest = json.loads(path.read_text())
        manifest.update(update)
        path.write_text(json.dumps(manifest))
        assert disk.load((1, 2, 3, 4), identity()) is None
        disk.clear()


def test_bounds_evict_oldest_and_never_write_oversized_entry(tmp_path):
    disk = store(tmp_path, max_entries=2, max_bytes=6000)
    for end in (3, 4, 5):
        assert disk.save((1, 2, end), layers(seed=end), identity())
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert sum(path.stat().st_size for path in tmp_path.iterdir()) <= 6000
    assert disk.load((1, 2, 3, 9), identity()) is None
    tiny = store(tmp_path / "tiny", max_bytes=32)
    assert not tiny.save((1, 2, 3), layers(), identity())
    assert not (tmp_path / "tiny").exists()


def test_failed_atomic_manifest_replace_preserves_old_snapshot_without_orphans(tmp_path):
    disk = store(tmp_path)
    assert disk.save((1, 2, 3), layers(seed=0), identity())
    replace = os.replace
    def fail_manifest(source, target):
        if str(target).endswith(".json"):
            raise OSError("simulated failed manifest publish")
        return replace(source, target)
    with patch.object(cache_module.os, "replace", fail_manifest):
        assert not disk.save((1, 2, 3), layers(seed=7), identity())
    assert disk.load((1, 2, 3, 4), identity())[1][0].keys.data == bytes([0]) * 32
    assert len(list(tmp_path.glob("*.safetensors"))) == 1
    assert not list(tmp_path.glob(".prompt-*"))


def test_clear_preserves_unrelated_files_and_close_releases_codec(tmp_path):
    disk = store(tmp_path)
    assert disk.save((1, 2, 3), layers(), identity())
    unrelated = tmp_path / "notes.json"
    unrelated.write_text("keep me")
    disk.clear()
    assert list(tmp_path.iterdir()) == [unrelated]
    disk.close()
    assert disk._codec is None
    assert disk.load((1, 2, 3, 4), identity()) is None


def test_local_artifacts_use_full_content_not_size_or_mtime(tmp_path):
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"first")
    before = weights.stat()
    old = cache_module.local_artifact_identity(tmp_path)
    weights.write_bytes(b"other")
    os.utime(weights, ns=(before.st_atime_ns, before.st_mtime_ns))
    new = cache_module.local_artifact_identity(tmp_path)
    assert old["files_sha256"] != new["files_sha256"]
    assert cache_module.local_artifact_identity(tmp_path / "hub-does-not-exist") is None


class QuantizedKVCache(KVCache):
    def __init__(self, group_size=64, bits=4):
        super().__init__()
        self.group_size = group_size
        self.bits = bits


class ArraysCache:
    def __init__(self, size):
        self.cache = [None] * size
        self.left_padding = None
        self.lengths = None


def test_quantized_and_recurrent_state_roundtrip_keeps_exact_types_and_bytes(tmp_path):
    quantized = QuantizedKVCache()
    quantized.offset = 3
    quantized.keys = tuple(Tensor(bytes([index]) * 16, dtype="uint32" if index == 0 else "float16")
                           for index in range(3))
    quantized.values = tuple(Tensor(bytes([index + 3]) * 16, dtype="uint32" if index == 0 else "float16")
                             for index in range(3))
    recurrent = ArraysCache(3)
    recurrent.cache = [Tensor(bytes([7]) * 32, shape=(1, 4, 8)), None,
                       Tensor(bytes([8]) * 32, shape=(1, 2, 3, 4))]
    codec = cache_module.SafeKVCodec(FakeMX(), cache_types={"KVCache": KVCache,
                                     "QuantizedKVCache": QuantizedKVCache, "ArraysCache": ArraysCache})
    disk = cache_module.PersistentPromptStore(tmp_path, enabled=True, codec=codec)
    assert disk.save((1, 2, 3), [quantized, recurrent], identity())
    _, restored = disk.load((1, 2, 3, 4), identity())
    assert type(restored[0]) is QuantizedKVCache
    assert type(restored[0].keys) is tuple
    assert restored[0].bits == 4 and restored[0].group_size == 64
    for old, new in zip(quantized.keys + quantized.values, restored[0].keys + restored[0].values):
        assert old is not new
        assert (old.data, old.dtype, old.shape) == (new.data, new.dtype, new.shape)
    assert type(restored[1]) is ArraysCache
    assert restored[1].cache is not recurrent.cache
    assert restored[1].cache[0].data == recurrent.cache[0].data
    assert restored[1].cache[0].shape == recurrent.cache[0].shape
    assert restored[1].cache[1] is None
    assert restored[1].left_padding is None and restored[1].lengths is None
    recurrent.lengths = Tensor(b"batch state", shape=(1,))
    assert not disk.save((1, 2, 3), [quantized, recurrent], identity())


def test_manifest_token_corruption_is_not_silently_reused(tmp_path):
    disk = store(tmp_path)
    assert disk.save((1, 2, 3), layers(), identity())
    manifest_path = next(tmp_path.glob("*.json"))
    manifest = json.loads(manifest_path.read_text())
    manifest["tokens"] = [1, 8, 3]
    manifest_path.write_text(json.dumps(manifest))
    assert disk.load((1, 8, 3, 4), identity()) is None
