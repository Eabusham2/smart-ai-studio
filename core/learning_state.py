"""Model-bound, crash-consistent generations for persistent learned parameters.

A checkpoint and its retirement/rehearsal metadata become visible through one
atomic pointer replacement. Readers verify identity and hashes before restoration.
No framework or model is imported here. Stored replay is bounded by the caller;
only the current and previous committed parameter generations are retained.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
import uuid
from typing import Any


SCHEMA_VERSION = 1
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_LOCK = threading.Lock()


@contextmanager
def _process_commit_lock(directory: Path):
    """Serialize compare-and-swap publication across desktop/CLI processes."""
    with open(directory / "commit.lock", "a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def identity_digest(identity: dict) -> str:
    return hashlib.sha256(_canonical(identity)).hexdigest()


def _check_cancelled(event) -> None:
    if event is not None and (event() if callable(event) else event.is_set()):
        raise InterruptedError("Learning cancelled before checkpoint publication")


def file_sha256(path, *, cancel_event=None) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            _check_cancelled(cancel_event)
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with open(temporary, "wb") as stream:
            stream.write(_canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _state_root(state_root=None) -> Path:
    if state_root is not None:
        return Path(state_root).expanduser().resolve()
    from config.paths import get_portable_data_dir
    return (Path(get_portable_data_dir()) / "learned_memory").resolve()


def _local_model_fingerprint(source: Path, root: Path) -> dict:
    """Hash immutable source files once; reuse hashes only for unchanged stat keys."""
    cache_key = hashlib.sha256(str(source.resolve()).encode()).hexdigest()
    cache_path = root / "identity_cache" / (cache_key + ".json")
    try:
        old = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = {}
    if source.is_file():
        paths = [source]
        base = source.parent
        shard = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", source.name, re.I)
        if shard:
            count = int(shard.group(3))
            paths = [source.with_name(f"{shard.group(1)}-{index:05d}-of-{count:05d}.gguf") for index in range(1, count + 1)]
            if not all(path.is_file() for path in paths):
                raise RuntimeError("Learning identity requires the complete selected GGUF shard set")
    else:
        base = source
        paths = sorted(path for path in source.rglob("*") if path.is_file() and (
            path.suffix.lower() in {".safetensors", ".gguf", ".npz", ".bin", ".model", ".tiktoken", ".json", ".jinja", ".jinja2", ".py", ".yaml", ".yml"}
            or path.name in {"merges.txt", "vocab.txt", "tokenizer.tiktoken"}
        ) and not any(part in {".git", "learned_memory", "controller_lora", "adapters"} for part in path.relative_to(base).parts))
    fingerprints = {}
    cache = {}
    for path in paths:
        relative = path.relative_to(base).as_posix()
        stat = path.stat()
        stamp = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        previous = old.get(relative, {})
        digest = previous.get("sha256") if previous.get("stat") == stamp else None
        if not isinstance(digest, str) or len(digest) != 64:
            digest = file_sha256(path)
        cache[relative] = {"stat": stamp, "sha256": digest}
        fingerprints[relative] = {"size": stat.st_size, "sha256": digest}
    if cache:
        _atomic_json(cache_path, cache)
    return fingerprints


def requested_revision(info: dict) -> str | None:
    return next((str(info[key]) for key in ("resolved_revision", "repo_revision", "source_revision", "revision", "commit_sha") if info.get(key)), None)


def cached_model_source(source: str, info: dict | None = None) -> Path:
    """Resolve only the exact requested revision already in the local Hub cache."""
    info = dict(info or {})
    local = Path(source).expanduser()
    if local.exists():
        return local
    if source.count("/") == 1 and not source.startswith((".", "/")):
        try:
            from huggingface_hub import snapshot_download
            return Path(snapshot_download(source, revision=requested_revision(info), local_files_only=True))
        except (ImportError, OSError, ValueError):
            pass
    return local


def model_identity(model_path, model_info=None, *, adapter_format: str = "mlx", state_root=None) -> dict:
    info = dict(model_info or {})
    source = str(model_path or info.get("model_path") or info.get("repo_id") or "").strip()
    if not source:
        raise ValueError("Persistent learning requires a specific base model identity")
    local = cached_model_source(source, info)
    files = _local_model_fingerprint(local.absolute(), _state_root(state_root)) if local.exists() else {}
    repo = str(info.get("repo_id") or info.get("hf_repo_id") or (source if not Path(source).expanduser().exists() else ""))
    revision = str(requested_revision(info) or "")
    if "snapshots" in local.parts:
        position = local.parts.index("snapshots")
        if len(local.parts) > position + 1:
            revision = local.parts[position + 1]
    identity = {
        "schema": SCHEMA_VERSION,
        "source": repo or str(local.resolve()),
        "revision": revision,
        "adapter_format": str(adapter_format),
        "runtime_family": str(info.get("runtime_family") or ""),
        "selected_artifact": str(info.get("gguf_file") or info.get("filename") or (local.name if local.is_file() else "")),
        "training_parent": str(info.get("training_base_model_id") or info.get("gguf_training_base_model_id") or info.get("bf16_model_id") or ""),
        "artifact_sha256": str(info.get("sha256") or info.get("source_sha256") or ""),
        "files": files,
    }
    projector = info.get("mmproj_path")
    if projector and Path(projector).is_file():
        identity["projector"] = _local_model_fingerprint(Path(projector), _state_root(state_root))
    identity["source_verified"] = bool(
        any(Path(name).suffix.lower() in {".safetensors", ".gguf", ".npz", ".bin"} for name in files)
        or re.fullmatch(r"[0-9a-fA-F]{40,64}", revision)
        or re.fullmatch(r"[0-9a-fA-F]{64}", identity["artifact_sha256"])
    )
    # Pinned model ID/revision plus source fingerprints separate Bonsai, CRACK,
    # tokenizer revisions and custom local artifacts without a global adapter.
    return identity


class LearningStateStore:
    def __init__(self, identity: dict, *, state_root=None):
        self.identity = json.loads(_canonical(identity))
        self.identity_digest = identity_digest(self.identity)
        self.directory = _state_root(state_root) / self.identity_digest
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "generations").mkdir(exist_ok=True)
        _fsync_directory(self.directory)
        _fsync_directory(self.directory.parent)
        with _LOCKS_LOCK:
            self.lock = _LOCKS.setdefault(str(self.directory), threading.RLock())

    def _read_pointer(self) -> dict | None:
        path = self.directory / "current.json"
        if not path.exists():
            return None
        try:
            pointer = json.loads(path.read_text(encoding="utf-8"))
            generation = pointer["generation"]
            if not isinstance(generation, str) or not generation or Path(generation).name != generation:
                raise ValueError("Invalid learning generation")
            return pointer
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("Learned-memory pointer is invalid; refusing to discard saved memory") from exc

    def current_manifest(self, *, verify: bool = True) -> dict | None:
        with self.lock:
            pointer = self._read_pointer()
            if pointer is None:
                return None
            directory = self.directory / "generations" / pointer["generation"]
            path = directory / "manifest.json"
            if not path.is_file() or file_sha256(path) != pointer.get("manifest_sha256"):
                raise RuntimeError("Learned-memory manifest is missing or changed")
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("identity_digest") != self.identity_digest or manifest.get("identity") != self.identity:
                raise RuntimeError("Learned checkpoint belongs to a different model/tokenizer")
            if verify:
                for relative, expected in manifest.get("artifacts", {}).items():
                    artifact = _contained(directory, relative)
                    if not artifact.is_file() or artifact.stat().st_size != expected["size"] or file_sha256(artifact) != expected["sha256"]:
                        raise RuntimeError("Learned checkpoint failed integrity verification: " + relative)
            return manifest

    def current_checkpoint(self) -> str | None:
        manifest = self.current_manifest()
        if manifest is None:
            return None
        return str(_contained(self.directory / "generations" / manifest["generation"], manifest["checkpoint"]))

    def begin(self, update_id=None, metadata=None):
        return LearningTransaction(self, update_id=update_id, metadata=metadata)


def _contained(directory: Path, relative) -> Path:
    item = Path(str(relative))
    if item.is_absolute() or ".." in item.parts:
        raise ValueError("Checkpoint paths must remain inside their generation")
    candidate = directory / item
    if not candidate.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Checkpoint path escapes its generation")
    return candidate


class LearningTransaction:
    def __init__(self, store: LearningStateStore, *, update_id=None, metadata=None):
        self.store = store
        if store.identity.get("source_verified") is False:
            raise RuntimeError("Learning requires locally verified model weights or a pinned source revision; reload the selected model before training")
        self.generation = uuid.uuid4().hex
        self.update_id = str(update_id or (metadata or {}).get("update_id") or uuid.uuid4().hex)
        self.directory = store.directory / "generations" / self.generation
        with store.lock:
            pointer = store._read_pointer()
            self.parent = pointer["generation"] if pointer else None
            supplied = dict(metadata or {})
            if "expected_parent" in supplied and supplied["expected_parent"] != self.parent:
                raise RuntimeError("The memory plan belongs to an older parameter generation; retry consolidation")
            parent_manifest = store.current_manifest(verify=False)
            inherited = dict((parent_manifest or {}).get("metadata") or {})
            inherited.pop("expected_parent", None)
            inherited.pop("update_id", None)
            self.metadata = {**inherited, **supplied}
        self.directory.mkdir(parents=True, exist_ok=False)
        self.committed = False

    def commit(self, checkpoint: str, metadata=None, *, cancel_event=None) -> dict:
        with self.store.lock, _process_commit_lock(self.store.directory):
            _check_cancelled(cancel_event)
            pointer = self.store._read_pointer()
            if (pointer["generation"] if pointer else None) != self.parent:
                raise RuntimeError("Learned memory changed during this update; retry against the current generation")
            target = _contained(self.directory, checkpoint)
            if not target.exists():
                raise RuntimeError("Training did not produce its declared checkpoint")
            artifacts = {}
            for path in sorted(self.directory.rglob("*")):
                _check_cancelled(cancel_event)
                if path.is_symlink():
                    raise RuntimeError("Learning artifacts must be regular files, not symbolic links")
                if path.is_file() and path.name != "manifest.json":
                    if path.stat().st_size <= 0:
                        raise RuntimeError("Training produced an empty artifact")
                    with open(path, "rb") as stream:
                        os.fsync(stream.fileno())
                    artifacts[path.relative_to(self.directory).as_posix()] = {"size": path.stat().st_size, "sha256": file_sha256(path, cancel_event=cancel_event)}
            if not artifacts:
                raise RuntimeError("Training produced no persisted parameters")
            relative_target = target.relative_to(self.directory).as_posix()
            if target.is_file():
                authenticated = relative_target in artifacts
            else:
                prefix = "" if relative_target == "." else relative_target + "/"
                authenticated = any(name.startswith(prefix) for name in artifacts)
            if not authenticated:
                raise RuntimeError("Declared checkpoint contains no authenticated artifact")
            manifest = {
                "schema": SCHEMA_VERSION, "identity": self.store.identity,
                "identity_digest": self.store.identity_digest,
                "generation": self.generation, "parent": self.parent,
                "update_id": self.update_id, "committed_at": time.time(),
                "checkpoint": Path(checkpoint).as_posix(), "artifacts": artifacts,
                "metadata": {**self.metadata, **dict(metadata or {})},
            }
            _atomic_json(self.directory / "manifest.json", manifest)
            directories = {self.directory, self.directory.parent}
            for relative in artifacts:
                parent = (self.directory / relative).parent
                while parent != self.directory:
                    directories.add(parent)
                    parent = parent.parent
            for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
                _fsync_directory(directory)
            _check_cancelled(cancel_event)
            new_pointer = {
                "generation": self.generation,
                "manifest_sha256": file_sha256(self.directory / "manifest.json"),
            }
            try:
                _atomic_json(self.store.directory / "current.json", new_pointer)
                self.committed = True
            except BaseException as exc:
                # An fsync error or process interrupt after publication must
                # never erase the selected generation or roll live weights
                # back while leaving the new pointer active.
                # On restart the atomic pointer selects an entire old or new
                # generation, including its matching retirement metadata.
                if self.store._read_pointer() != new_pointer:
                    raise
                self.committed = True
                if not isinstance(exc, OSError):
                    raise
                manifest["durability_warning"] = str(exc)
            # Garbage collection is not part of commit correctness. Keep the old
            # committed generation for recovery; never erase a concurrent stage.
            try:
                for old in (self.store.directory / "generations").iterdir():
                    if old.name not in {self.generation, self.parent} and (old / "manifest.json").is_file():
                        shutil.rmtree(old)
            except OSError:
                pass
            return manifest

    def abort(self) -> None:
        if not self.committed:
            with self.store.lock, _process_commit_lock(self.store.directory):
                pointer = self.store._read_pointer()
                if pointer and pointer["generation"] == self.generation:
                    self.committed = True
                    return
                shutil.rmtree(self.directory, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if not self.committed:
            self.abort()


def bind_learning_state(backend, *, model_path=None, model_info=None, state_root=None, adapter_format="mlx") -> LearningStateStore:
    identity = model_identity(
        model_path or getattr(backend, "model_path", None) or getattr(backend, "hf_repo_id", None),
        model_info if model_info is not None else getattr(backend, "model_info", None),
        adapter_format=adapter_format, state_root=state_root,
    )
    existing = getattr(backend, "learning_state", None)
    intended_directory = _state_root(state_root) / identity_digest(identity)
    if isinstance(existing, LearningStateStore) and existing.identity == identity and existing.directory.resolve() == intended_directory.resolve():
        return existing
    store = LearningStateStore(identity, state_root=state_root)
    backend.learning_state = store
    backend.learning_identity = store.identity
    return store


def validate_checkpoint(backend, path, *, allow_legacy=False) -> dict | None:
    store = getattr(backend, "learning_state", None)
    if store is None:
        raise RuntimeError("Bind a model identity before restoring learned weights")
    selected = Path(path).resolve()
    current = store.current_checkpoint()
    if current and selected == Path(current).resolve():
        return store.current_manifest()
    for parent in ([selected] if selected.is_dir() else []) + list(selected.parents):
        adjacent = parent / "manifest.json"
        if adjacent.is_file() and parent.parent.name == "generations":
            manifest = json.loads(adjacent.read_text(encoding="utf-8"))
            if manifest.get("identity_digest") != store.identity_digest or manifest.get("identity") != store.identity:
                raise RuntimeError("Selected managed checkpoint belongs to a different model or tokenizer")
            raise RuntimeError("Selected managed checkpoint is not the current generation; use an explicit memory rollback before loading it")
    sidecar = Path(str(selected) + ".memory.json")
    if not sidecar.is_file():
        if allow_legacy:
            return None
        raise RuntimeError("Saved adapter has no proven model identity; select the legacy adapter explicitly to migrate it")
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    if manifest.get("identity_digest") != store.identity_digest or manifest.get("identity") != store.identity:
        raise RuntimeError("Selected adapter belongs to a different model or tokenizer")
    if selected.is_file() and manifest.get("sha256") != file_sha256(selected):
        raise RuntimeError("Selected adapter failed integrity verification")
    return manifest


def write_checkpoint_manifest(backend, path, *, metadata=None) -> dict:
    store = getattr(backend, "learning_state", None)
    if store is None:
        raise RuntimeError("Bind a model identity before exporting learned weights")
    selected = Path(path)
    if not selected.is_file():
        raise ValueError("Standalone checkpoint export requires a regular file")
    manifest = {"schema": SCHEMA_VERSION, "identity": store.identity, "identity_digest": store.identity_digest,
                "sha256": file_sha256(selected), "metadata": dict(metadata or {})}
    _atomic_json(Path(str(selected) + ".memory.json"), manifest)
    return manifest
