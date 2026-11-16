"""Opt-in, bounded local snapshots of exact native MLX KV state.

Files contain JSON and safetensors only. Disk metadata never imports a Python
class or executes deserialization code. Unknown identities/cache layouts and all
I/O failures are ordinary misses. This module does not import MLX until enabled
persistence is actually used, and never calls a model or tokenizer.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any


SCHEMA_VERSION = 1
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_PREFIX_TOKENS = 262144
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _json_bytes(value: Any) -> bytes:
    def check(item):
        if item is None or type(item) in (str, int, float, bool):
            return
        if type(item) in (list, tuple):
            for member in item:
                check(member)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for member in item.values():
                check(member)
            return
        raise ValueError("Identity and cache metadata must contain JSON values only")
    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def prompt_cache_identity(*, model, tokenizer, adapter, cache_policy, versions) -> dict:
    """Require explicit identities for every input that can change native KV state.

    The caller must invalidate/disable disk reuse after in-memory weight edits.
    Unknown package versions, hub names alone and timestamp-only weight identities
    are insufficient. ``local_artifact_identity`` supplies content-based identities.
    """
    if not model or not tokenizer or not cache_policy:
        raise ValueError("A complete model/tokenizer/cache-policy identity is required")
    if not isinstance(versions, dict) or not versions or any(
        not isinstance(value, str) or value.lower() in {"", "unknown", "none"}
        for value in versions.values()
    ):
        raise ValueError("Exact runtime versions are required")
    value = dict(model=model, tokenizer=tokenizer, adapter=adapter,
                 cache_policy=cache_policy, versions=versions, schema=SCHEMA_VERSION)
    _json_bytes(value)
    return value


def _fingerprint(identity: dict) -> str:
    if type(identity) is not dict:
        raise ValueError("An explicit identity is required")
    required = ("model", "tokenizer", "adapter", "cache_policy", "versions")
    checked = prompt_cache_identity(**{key: identity[key] for key in required})
    # Preserve any additional caller revision/implementation discriminator.
    return hashlib.sha256(_json_bytes({**identity, **checked})).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_artifact_identity(path) -> dict | None:
    """Hash complete local artifacts, including weights; never resolve/download hubs.

    This intentionally pays one full read on opt-in initialization. File sizes or
    mtimes alone do not prove identical weights. Unreadable/changing artifacts and
    symlinked subdirectories fail closed instead of creating a weak identity.
    """
    try:
        root = Path(path).expanduser().resolve(strict=True)
        def files():
            if root.is_file():
                return [root]
            found = []
            def fail(error):
                raise error
            for directory, dirs, names in os.walk(root, followlinks=False, onerror=fail):
                if any((Path(directory) / name).is_symlink() for name in dirs):
                    raise ValueError("Cannot establish identity through a directory symlink")
                found.extend(Path(directory) / name for name in names)
            return sorted(found)
        selected = files()
        if not selected:
            return None
        entries = []
        signatures = {}
        signature = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        for item in selected:
            if not item.is_file():
                return None
            before = item.stat()
            digest = _sha_file(item)
            after = item.stat()
            if signature(before) != signature(after):
                return None
            signatures[item] = signature(after)
            entries.append({"name": item.name if root.is_file() else str(item.relative_to(root)),
                            "bytes": after.st_size, "sha256": digest})
        if selected != files() or any(signature(item.stat()) != signatures[item] for item in selected):
            return None
        return {"path": str(root), "files_sha256": hashlib.sha256(_json_bytes(entries)).hexdigest(),
                "file_count": len(entries)}
    except (OSError, TypeError, ValueError):
        return None


def _tokens(tokens) -> tuple[int, ...]:
    values = tuple(tokens)
    if not values or len(values) > _MAX_PREFIX_TOKENS or any(type(t) is not int or t < 0 for t in values):
        raise ValueError("An exact bounded nonempty token sequence is required")
    return values


class SafeKVCodec:
    """Lossless native KV, QuantizedKV and unbatched ArraysCache state.

    Cache classes are resolved from an explicit local allowlist, never imported
    from metadata. TurboQuant, rotating/windowed and custom caches fail closed.
    Every instance field must have a known snapshot contract. Quantized tensors
    retain their existing precision; recurrent arrays are saved whole, untrimmed.
    """
    _FIELDS = {
        "KVCache": {"keys", "values", "offset", "step"},
        "QuantizedKVCache": {"keys", "values", "offset", "step", "group_size", "bits"},
        "ArraysCache": {"cache", "left_padding", "lengths"},
    }

    def __init__(self, mx=None, cache_type=None, *, cache_types=None):
        if mx is None:
            import mlx.core as mx
        if cache_types is None:
            if cache_type is not None:
                cache_types = {"KVCache": cache_type}
            else:
                from mlx_lm.models import cache as native
                cache_types = {name: getattr(native, name) for name in self._FIELDS
                               if isinstance(getattr(native, name, None), type)}
        if any(name not in self._FIELDS for name in cache_types):
            raise ValueError("Unknown native cache allowlist entry")
        self.mx = mx
        self.cache_types = dict(cache_types)
        self.array_type = mx.array

    def _arrays(self, value):
        if type(value) is self.array_type:
            return [value]
        if value is None or type(value) in (str, int, float, bool):
            return []
        if type(value) in (list, tuple):
            return [array for item in value for array in self._arrays(item)]
        raise ValueError("Unknown native cache state object")

    def _validate(self, cache, token_count):
        if not isinstance(cache, (tuple, list)) or not cache:
            raise ValueError("Missing native cache layers")
        kinds = {cls: name for name, cls in self.cache_types.items()}
        for layer in cache:
            kind = kinds.get(type(layer))
            state = vars(layer)
            if kind is None or set(state) - self._FIELDS[kind]:
                raise ValueError("Unsupported cache class or unknown native state")
            if kind == "ArraysCache":
                if type(state.get("cache")) is not list or not state["cache"]:
                    raise ValueError("Missing recurrent state")
                if state.get("left_padding") is not None or state.get("lengths") is not None:
                    raise ValueError("Batched recurrent state is unsupported")
                if not self._arrays(state["cache"]):
                    raise ValueError("Empty recurrent checkpoint")
            else:
                if type(state.get("offset")) is not int or layer.offset != token_count:
                    raise ValueError("Cache offset does not match its prefix")
                keys = self._arrays(state.get("keys"))
                values = self._arrays(state.get("values"))
                expected = 1 if kind == "KVCache" else 3
                if len(keys) != expected or len(values) != expected:
                    raise ValueError("Invalid native KV tensors")
                for array in keys + values:
                    if len(array.shape) != 4 or array.shape[2] < token_count:
                        raise ValueError("Invalid native KV shape")
                if keys[0].shape[:3] != values[0].shape[:3]:
                    raise ValueError("Mismatched native KV shapes")
                if "step" in state and (type(layer.step) is not int or layer.step <= 0):
                    raise ValueError("Invalid cache step")
                if kind == "QuantizedKVCache":
                    if any(type(state.get(name)) is not int or state[name] <= 0 for name in ("group_size", "bits")):
                        raise ValueError("Invalid quantized cache policy")
            for value in state.values():
                for array in self._arrays(value):
                    if (not array.shape or array.shape[0] != 1
                            or any(int(n) <= 0 for n in array.shape) or int(array.nbytes) < 0):
                        raise ValueError("Invalid or batched native state tensor")

    def nbytes(self, cache, token_count):
        self._validate(cache, token_count)
        return sum(int(array.nbytes) for layer in cache for value in vars(layer).values()
                   for array in self._arrays(value))

    def write(self, path, cache, token_count):
        self._validate(cache, token_count)
        arrays = {}
        kinds = {cls: name for name, cls in self.cache_types.items()}
        def encode(value):
            if type(value) is self.array_type:
                key = "tensor_" + str(len(arrays))
                arrays[key] = value
                return {"type": "array", "key": key, "shape": list(value.shape), "dtype": str(value.dtype)}
            if type(value) in (list, tuple):
                return {"type": "list" if type(value) is list else "tuple",
                        "items": [encode(item) for item in value]}
            if value is None or type(value) in (str, int, float, bool):
                return {"type": "value", "value": value}
            raise ValueError("Unknown native cache state object")
        layers = [{"kind": kinds[type(layer)],
                   "attrs": {name: encode(value) for name, value in vars(layer).items()}}
                  for layer in cache]
        self.mx.eval(*arrays.values())
        self.mx.save_safetensors(str(path), arrays)
        return {"codec": "native-state-v2", "layers": layers}

    def read(self, path, descriptor, token_count):
        if (type(descriptor) is not dict or set(descriptor) != {"codec", "layers"}
                or descriptor["codec"] != "native-state-v2"):
            raise ValueError("Unknown cache codec")
        layers = descriptor["layers"]
        if not isinstance(layers, list) or not 1 <= len(layers) <= 4096:
            raise ValueError("Invalid layer count")
        arrays = self.mx.load(str(path))
        if not isinstance(arrays, dict):
            raise ValueError("Invalid tensor payload")
        used = set()
        def decode(value, depth=0):
            if depth > 16 or type(value) is not dict:
                raise ValueError("Invalid cache state tree")
            kind = value.get("type")
            if kind == "array" and set(value) == {"type", "key", "shape", "dtype"}:
                key = value["key"]
                if key in used or key not in arrays:
                    raise ValueError("Missing or aliased tensor")
                array = arrays[key]
                if type(array) is not self.array_type or list(array.shape) != value["shape"] or str(array.dtype) != value["dtype"]:
                    raise ValueError("Tensor metadata mismatch")
                used.add(key)
                return array
            if kind in {"list", "tuple"} and set(value) == {"type", "items"} and type(value["items"]) is list:
                items = [decode(item, depth + 1) for item in value["items"]]
                return items if kind == "list" else tuple(items)
            if kind == "value" and set(value) == {"type", "value"}:
                item = value["value"]
                if item is None or type(item) in (str, int, float, bool):
                    return item
            raise ValueError("Unknown cache state node")
        result = []
        for entry in layers:
            if type(entry) is not dict or set(entry) != {"kind", "attrs"}:
                raise ValueError("Invalid native layer metadata")
            kind = entry["kind"]
            cls = self.cache_types.get(kind)
            if cls is None or type(entry["attrs"]) is not dict or set(entry["attrs"]) - self._FIELDS[kind]:
                raise ValueError("Unknown native cache class or state")
            # Native Python cache state consists exclusively of the allowlisted
            # fields. No stored constructor, reducer or import path is executed.
            layer = object.__new__(cls)
            for name, value in entry["attrs"].items():
                setattr(layer, name, decode(value))
            result.append(layer)
        if set(arrays) != used:
            raise ValueError("Unexpected tensor payload")
        self._validate(result, token_count)
        self.mx.eval(*arrays.values())
        return result


class PersistentPromptStore:
    """Atomic manifests plus content-addressed tensors; every failure is a miss.

    Callers retain ownership of model locks and in-memory snapshots. The store
    never retains a model, tokenizer or cache object. Disk reads return new arrays.
    The private directory contains prompt tokens and KV state: enabling it is an
    explicit privacy/storage choice. ``clear`` removes only this store's files.
    """
    def __init__(self, directory, *, enabled=False, max_bytes=256 * 1024 * 1024,
                 max_entries=8, codec=None):
        self.directory = Path(directory).expanduser()
        self.enabled = bool(enabled)
        self.max_bytes = max(0, int(max_bytes))
        self.max_entries = max(0, int(max_entries))
        self._codec = codec
        self._lock = threading.RLock()

    def _get_codec(self):
        if self._codec is None:
            self._codec = SafeKVCodec()
        return self._codec

    def _read_manifest(self, path):
        if path.stat().st_size > min(_MAX_MANIFEST_BYTES, self.max_bytes):
            raise ValueError("Oversized cache manifest")
        value = json.loads(path.read_bytes())
        required = {"schema", "identity", "tokens", "tensor_file", "tensor_sha256", "tensor_bytes", "descriptor", "manifest_sha256"}
        if type(value) is not dict or set(value) != required or value["schema"] != SCHEMA_VERSION:
            raise ValueError("Unknown cache schema")
        checksum = value["manifest_sha256"]
        if checksum != hashlib.sha256(_json_bytes({key: item for key, item in value.items() if key != "manifest_sha256"})).hexdigest():
            raise ValueError("Corrupt cache manifest")
        if not isinstance(value["tensor_sha256"], str) or not _DIGEST.fullmatch(value["tensor_sha256"]):
            raise ValueError("Invalid content hash")
        if value["tensor_file"] != value["tensor_sha256"] + ".safetensors":
            raise ValueError("Invalid payload path")
        if type(value["tensor_bytes"]) is not int or not 0 < value["tensor_bytes"] <= self.max_bytes:
            raise ValueError("Invalid payload size")
        _tokens(value["tokens"])
        return value

    def save(self, tokens, cache, identity):
        if not self.enabled or not self.max_bytes or not self.max_entries:
            return False
        temporary = None
        try:
            prefix = _tokens(tokens)
            fingerprint = _fingerprint(identity)
            codec = self._get_codec()
            if codec.nbytes(cache, len(prefix)) > self.max_bytes:
                return False
            with self._lock:
                self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                self._prune()
                fd, name = tempfile.mkstemp(prefix=".prompt-", suffix=".safetensors", dir=self.directory)
                os.close(fd)
                temporary = Path(name)
                descriptor = codec.write(temporary, cache, len(prefix))
                # Windows _commit requires a writable descriptor. Opening the
                # finished payload read/write preserves its bytes while making
                # the same durability barrier portable across all platforms.
                with temporary.open("r+b") as handle:
                    os.fsync(handle.fileno())
                size = temporary.stat().st_size
                digest = _sha_file(temporary)
                manifest = {"schema": SCHEMA_VERSION, "identity": fingerprint, "tokens": prefix,
                            "tensor_file": digest + ".safetensors", "tensor_sha256": digest,
                            "tensor_bytes": size, "descriptor": descriptor}
                manifest["manifest_sha256"] = hashlib.sha256(_json_bytes(manifest)).hexdigest()
                raw = _json_bytes(manifest)
                if len(raw) > _MAX_MANIFEST_BYTES or size + len(raw) > self.max_bytes:
                    return False
                payload = self.directory / manifest["tensor_file"]
                os.replace(temporary, payload)
                temporary = None
                key = hashlib.sha256(fingerprint.encode() + _json_bytes(prefix)).hexdigest()
                fd, name = tempfile.mkstemp(prefix=".prompt-", suffix=".json", dir=self.directory)
                temporary = Path(name)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                destination = self.directory / (key + ".json")
                os.replace(temporary, destination)
                temporary = None
                self._touch_manifest(destination)
                self._prune()
                return destination.exists()
        except Exception:
            try:
                with self._lock:
                    self._prune()
            except Exception:
                pass
            return False
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def load(self, tokens, identity):
        if not self.enabled or not self.max_bytes or not self.max_entries:
            return None
        try:
            requested = _tokens(tokens)
            fingerprint = _fingerprint(identity)
            with self._lock:
                candidates = []
                for path in self.directory.glob("*.json"):
                    if not _DIGEST.fullmatch(path.stem):
                        continue
                    try:
                        value = self._read_manifest(path)
                        prefix = tuple(value["tokens"])
                        if value["identity"] == fingerprint and len(prefix) < len(requested) and requested[:len(prefix)] == prefix:
                            candidates.append((len(prefix), path, value))
                    except Exception:
                        continue
                for _, path, value in sorted(candidates, reverse=True):
                    try:
                        payload = self.directory / value["tensor_file"]
                        if payload.stat().st_size != value["tensor_bytes"] or _sha_file(payload) != value["tensor_sha256"]:
                            continue
                        cache = self._get_codec().read(payload, value["descriptor"], len(value["tokens"]))
                        self._touch_manifest(path)
                        return tuple(value["tokens"]), cache
                    except Exception:
                        continue
        except Exception:
            pass
        return None

    def _touch_manifest(self, path):
        # Fast saves can share one filesystem timestamp tick. Explicit ordered
        # timestamps preserve oldest-first eviction instead of hash-name order.
        latest = max((p.stat().st_mtime_ns for p in self.directory.glob("*.json")
                      if _DIGEST.fullmatch(p.stem)), default=0)
        stamp = max(time.time_ns(), latest + 1)
        os.utime(path, ns=(stamp, stamp))

    def _prune(self):
        manifests = []
        for path in self.directory.glob("*.json"):
            if not _DIGEST.fullmatch(path.stem):
                continue
            try:
                value = self._read_manifest(path)
                manifests.append((path.stat().st_mtime_ns, path, value))
            except Exception:
                path.unlink(missing_ok=True)
        keep, used, referenced = 0, 0, set()
        for _, path, value in sorted(manifests, reverse=True):
            size = path.stat().st_size + (0 if value["tensor_file"] in referenced else value["tensor_bytes"])
            if keep >= self.max_entries or used + size > self.max_bytes:
                path.unlink(missing_ok=True)
            else:
                keep += 1
                used += size
                referenced.add(value["tensor_file"])
        for payload in self.directory.glob("*.safetensors"):
            if _DIGEST.fullmatch(payload.stem) and payload.name not in referenced:
                payload.unlink(missing_ok=True)

    def clear(self):
        with self._lock:
            for suffix in ("*.json", "*.safetensors"):
                for path in self.directory.glob(suffix):
                    if _DIGEST.fullmatch(path.stem):
                        try:
                            path.unlink()
                        except OSError:
                            pass

    def close(self):
        with self._lock:
            self.enabled = False
            self._codec = None
