"""Model-bound, crash-consistent generations for persistent learned parameters.

A checkpoint and its retirement/rehearsal metadata become visible through one
atomic pointer replacement. Readers verify identity and hashes before restoration.
No framework or model is imported here. Stored replay is bounded by the caller;
current, previous and explicitly pinned generations are retained.
"""
from __future__ import annotations

import hashlib
from contextlib import ExitStack, closing, contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import threading
import time
import uuid
from weakref import WeakValueDictionary

from core.memory_scopes import as_scope


SCHEMA_VERSION = 1
_LOCKS = WeakValueDictionary()
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


def _fsync_staged_file(path: Path) -> None:
    """Flush a finished staging file without truncation. Windows requires write access."""
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


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
    def __init__(self, identity: dict, *, state_root=None, scope=None, require_new=False):
        self.identity = json.loads(_canonical(identity))
        self.identity_digest = identity_digest(self.identity)
        self.root = _state_root(state_root)
        self.scope = as_scope(scope)
        if require_new and self.scope.kind != "request":
            raise ValueError("Only transient requests use exclusive scope creation")
        self.directory = self.root / self.identity_digest
        self.bank_path = self.directory / "adapter_bank.sqlite3"
        if self.scope.kind != "model":
            self.directory = self.root / "scopes" / self.identity_digest / self.scope.kind / self.scope.scope_id
        self.directory.parent.mkdir(parents=True, exist_ok=True)
        self._lock_directory = self.directory.parent if self.scope.kind == "request" else self.directory
        with _LOCKS_LOCK:
            self.lock = _LOCKS.setdefault(str(self._lock_directory), threading.RLock())
        with self.lock:
            self._lock_directory.mkdir(parents=True, exist_ok=True)
            with _process_commit_lock(self._lock_directory):
                self.directory.mkdir(parents=True, exist_ok=not require_new)
                (self.directory / "generations").mkdir(exist_ok=True)
                if self.scope.kind == "request":
                    marker = self.directory / "request.json"
                    if not marker.exists():
                        _atomic_json(marker, uuid.uuid4().hex)
                    self._request_id = json.loads(marker.read_text(encoding="utf-8"))
                _fsync_directory(self.directory)
                _fsync_directory(self.directory.parent)

    def for_scope(self, scope, *, require_new=False):
        """Atomically reserve a fresh transient namespace when require_new is set."""
        scoped = type(self)(self.identity, state_root=self.root, scope=scope, require_new=require_new)
        scoped.implementation_base_root = getattr(self, "implementation_base_root", self.root)
        return scoped

    def _check_request(self):
        if self.scope.kind == "request":
            marker = self.directory / "request.json"
            if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != self._request_id:
                raise RuntimeError("This request has ended")

    def discard_request(self):
        """After unloading, remove exactly this request, fencing stale transactions."""
        if self.scope.kind != "request":
            raise ValueError("Only request scope supports transient cleanup")
        with self.lock, _process_commit_lock(self._lock_directory):
            if not self.directory.exists():
                return
            self._check_request()
            shutil.rmtree(self.directory)
            _fsync_directory(self.directory.parent)

    def seed_from(self, parent_store, *, cancel_event=None, expected_generation=...):
        """Fork verified serving parameters into an empty transient request before loading it."""
        if (self.scope.kind != "request" or parent_store.scope.kind == "request"
                or self.identity != parent_store.identity or self.root != parent_store.root):
            raise ValueError("A request may only inherit its own durable model/tokenizer identity")
        with parent_store.locked_generation(cancel_event=cancel_event) as (source, pointer):
            _check_cancelled(cancel_event)
            source_generation = source["generation"] if source else None
            if expected_generation is not ... and source_generation != expected_generation:
                raise RuntimeError("The source generation changed before the request could inherit it")
            if self.current_manifest() is not None:
                raise RuntimeError("Request memory already has parameters")
            if source is None:
                return None
            metadata = dict(source.get("metadata") or {})
            metadata.pop("update_id", None)
            metadata["expected_parent"] = None
            metadata["request_seed"] = {"scope": parent_store.scope.record,
                                        "generation": source["generation"],
                                        "manifest_sha256": pointer["manifest_sha256"]}
            base = parent_store.directory / "generations" / source["generation"]
            with self.begin(metadata=metadata) as txn:
                for relative, expected in source["artifacts"].items():
                    original = _contained(base, relative)
                    destination = _contained(txn.directory, relative)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    with original.open("rb") as reader, destination.open("xb") as writer:
                        for block in iter(lambda: reader.read(1024 * 1024), b""):
                            _check_cancelled(cancel_event)
                            digest.update(block)
                            writer.write(block)
                    if digest.hexdigest() != expected["sha256"]:
                        raise RuntimeError("Source checkpoint changed while seeding request memory")
                memory = metadata.get("memory", {})
                if memory.get("retirement_head"):
                    self._inherit_receipts(parent_store, memory["retirement_head"], cancel_event)
                if memory.get("replay_head"):
                    from core.learning_recall import ReplayBank
                    ReplayBank(self.directory).inherit(parent_store.directory, memory["replay_head"], cancel_event)
                if memory.get("stream_head"):
                    from core.learning_recall import StreamCursorBank
                    StreamCursorBank(self.directory).inherit(parent_store.directory, memory["stream_head"], cancel_event)
                return txn.commit(source["checkpoint"], cancel_event=cancel_event)

    def _inherit_receipts(self, parent_store, head, cancel_event):
        from core.context_memory import RetirementLedger
        origin = parent_store.directory / "retirement.sqlite3"
        if not origin.is_file():
            raise RuntimeError("The source checkpoint has no retirement ledger")
        ledger = RetirementLedger(self.directory)
        with closing(sqlite3.connect(ledger.path)) as db, db:
            db.execute("ATTACH DATABASE ? AS origin", (str(origin),))
            db.execute("BEGIN IMMEDIATE")
            _check_cancelled(cancel_event)
            db.execute("""WITH RECURSIVE lineage(head,parent,turns) AS (
                SELECT head,parent,turns FROM origin.receipts WHERE head=?
                UNION ALL SELECT r.head,r.parent,r.turns FROM origin.receipts AS r
                JOIN lineage AS l ON l.parent=r.head
            ) INSERT OR IGNORE INTO main.receipts SELECT * FROM lineage""", (head,))
            ledger._index(db, head)
            _check_cancelled(cancel_event)
        recall = parent_store.directory / "recall.sqlite3"
        if recall.is_file():
            from core.learning_recall import RecallBank
            target = RecallBank(self.directory)
            with closing(sqlite3.connect(target.path)) as db, db:
                db.execute("ATTACH DATABASE ? AS origin", (str(recall),))
                db.execute("""WITH RECURSIVE sets(head,parent,depth,checks) AS (
                    SELECT head,parent,depth,checks FROM origin.probe_sets WHERE head=?
                    UNION ALL SELECT r.head,r.parent,r.depth,r.checks FROM origin.probe_sets AS r
                    JOIN sets AS s ON s.parent=r.head
                ) INSERT OR IGNORE INTO main.probe_sets SELECT * FROM sets""", (head,))
                if db.execute("SELECT 1 FROM origin.sqlite_master WHERE type='table' AND name='probes'").fetchone():
                    db.execute("""INSERT OR IGNORE INTO main.probes
                                  SELECT p.* FROM origin.probes p JOIN main.probe_sets s ON s.head=p.head""")
                _check_cancelled(cancel_event)

    def _read_pointer(self) -> dict | None:
        self._check_request()
        if self.scope.pin:
            if self.bank_path.is_file():
                with closing(sqlite3.connect(self.bank_path)) as db:
                    row = db.execute("""SELECT manifest_sha256 FROM entries
                                        WHERE pin=? AND document_scope=? AND generation=?""",
                                     (self.scope.pin, self.scope.scope_id, self.scope.generation)).fetchone()
                    if row:
                        return {"generation": self.scope.generation, "manifest_sha256": row[0]}
            raise RuntimeError("The selected document generation is no longer pinned")
        path = self.directory / "current.json"
        if not path.exists():
            return None
        try:
            pointer = json.loads(path.read_text(encoding="utf-8"))
            generation = pointer["generation"]
            if not isinstance(generation, str) or not re.fullmatch(r"[0-9a-f]{32}", generation):
                raise ValueError("Invalid learning generation")
            return pointer
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("Learned-memory pointer is invalid; refusing to discard saved memory") from exc

    def current_manifest(self, *, verify: bool = True, cancel_event=None) -> dict | None:
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
            if manifest.get("generation") != pointer["generation"]:
                raise RuntimeError("Learned checkpoint generation does not match its pointer")
            self.verify_scope(manifest)
            if verify:
                for relative, expected in manifest.get("artifacts", {}).items():
                    artifact = _contained(directory, relative)
                    if not artifact.is_file() or artifact.stat().st_size != expected["size"] or file_sha256(artifact, cancel_event=cancel_event) != expected["sha256"]:
                        raise RuntimeError("Learned checkpoint failed integrity verification: " + relative)
            return manifest

    def verify_scope(self, manifest):
        if manifest.get("scope") != self.scope.record:
            if self.scope.kind != "model" or "scope" in manifest:
                raise RuntimeError("Selected learned checkpoint belongs to a different memory scope")

    @contextmanager
    def locked_generation(self, *, cancel_event=None):
        """Hold the commit/GC lock while verifying and registering a durable pin."""
        with self.lock, _process_commit_lock(self._lock_directory):
            yield self.current_manifest(cancel_event=cancel_event), self._read_pointer()

    def _collect_generations(self, current, parent):
        with ExitStack() as stack:
            db = stack.enter_context(closing(sqlite3.connect(self.bank_path))) if self.bank_path.is_file() else None
            for old in (self.directory / "generations").iterdir():
                if old.name in {current, parent} or not (old / "manifest.json").is_file():
                    continue
                if db and db.execute("""SELECT 1 FROM entries WHERE document_scope=?
                                         AND generation=? LIMIT 1""", (self.scope.scope_id, old.name)).fetchone():
                    continue
                shutil.rmtree(old)

    def current_checkpoint(self) -> str | None:
        manifest = self.current_manifest()
        if manifest is None:
            return None
        return str(_contained(self.directory / "generations" / manifest["generation"], manifest["checkpoint"]))

    def begin(self, update_id=None, metadata=None, *, inherit_metadata=True):
        return LearningTransaction(self, update_id=update_id, metadata=metadata, inherit_metadata=inherit_metadata)


def _contained(directory: Path, relative) -> Path:
    item = Path(str(relative))
    if item.is_absolute() or ".." in item.parts:
        raise ValueError("Checkpoint paths must remain inside their generation")
    candidate = directory / item
    if not candidate.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Checkpoint path escapes its generation")
    return candidate


class LearningTransaction:
    def __init__(self, store: LearningStateStore, *, update_id=None, metadata=None, inherit_metadata=True):
        self.store = store
        if store.scope.pin:
            raise RuntimeError("A selected document generation is read-only")
        if store.identity.get("source_verified") is False:
            raise RuntimeError("Learning requires locally verified model weights or a pinned source revision; reload the selected model before training")
        self.generation = uuid.uuid4().hex
        self.update_id = str(update_id or (metadata or {}).get("update_id") or uuid.uuid4().hex)
        self.directory = store.directory / "generations" / self.generation
        with store.lock, _process_commit_lock(store._lock_directory):
            pointer = store._read_pointer()
            self.parent = pointer["generation"] if pointer else None
            supplied = dict(metadata or {})
            self.rollout_group = supplied.pop("_sampled_rollout_group", None)
            if "expected_parent" in supplied and supplied["expected_parent"] != self.parent:
                raise RuntimeError("The memory plan belongs to an older parameter generation; retry consolidation")
            parent_manifest = store.current_manifest(verify=False) if inherit_metadata else None
            inherited = dict((parent_manifest or {}).get("metadata") or {})
            inherited.pop("expected_parent", None)
            inherited.pop("update_id", None)
            inherited.pop("_sampled_rollout_group", None)
            self.metadata = {**inherited, **supplied}
            self.directory.mkdir(parents=True, exist_ok=False)
        self.committed = False

    def commit(self, checkpoint: str, metadata=None, *, cancel_event=None) -> dict:
        with self.store.lock, _process_commit_lock(self.store._lock_directory):
            _check_cancelled(cancel_event)
            pointer = self.store._read_pointer()
            if (pointer["generation"] if pointer else None) != self.parent:
                raise RuntimeError("Learned memory changed during this update; retry against the current generation")
            target = _contained(self.directory, checkpoint)
            if not target.exists():
                raise RuntimeError("Training did not produce its declared checkpoint")
            if self.rollout_group is not None:
                # This file is hashed and fsynced alongside the weights. Fresh
                # policy groups never enter supervised replay or a descendant manifest.
                with (self.directory / "sampled-rollouts.json").open("x", encoding="utf-8") as file:
                    json.dump(self.rollout_group, file, separators=(",", ":"), allow_nan=False)
            artifacts = {}
            for path in sorted(self.directory.rglob("*")):
                _check_cancelled(cancel_event)
                if path.is_symlink():
                    raise RuntimeError("Learning artifacts must be regular files, not symbolic links")
                if path.is_file() and path.name != "manifest.json":
                    if path.stat().st_size <= 0:
                        raise RuntimeError("Training produced an empty artifact")
                    _fsync_staged_file(path)
                    artifacts[path.relative_to(self.directory).as_posix()] = {"size": path.stat().st_size, "sha256": file_sha256(path, cancel_event=cancel_event)}
            if not artifacts:
                raise RuntimeError("Training produced no persisted parameters")
            relative_target = target.relative_to(self.directory).as_posix()
            if target.is_file():
                authenticated = relative_target in artifacts and relative_target != "sampled-rollouts.json"
            else:
                prefix = "" if relative_target == "." else relative_target + "/"
                authenticated = any(name.startswith(prefix) and name != "sampled-rollouts.json" for name in artifacts)
            if not authenticated:
                raise RuntimeError("Declared checkpoint contains no authenticated artifact")
            published_metadata = {**self.metadata, **dict(metadata or {})}
            published_metadata.pop("_sampled_rollout_group", None)
            manifest = {
                "schema": SCHEMA_VERSION, "identity": self.store.identity,
                "identity_digest": self.store.identity_digest,
                "generation": self.generation, "parent": self.parent,
                "update_id": self.update_id, "committed_at": time.time(),
                "checkpoint": Path(checkpoint).as_posix(), "artifacts": artifacts,
                "metadata": published_metadata,
            }
            if self.store.scope.kind != "model":
                manifest["scope"] = self.store.scope.record
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
                self.store._collect_generations(self.generation, self.parent)
            except (OSError, sqlite3.Error):
                pass
            return manifest

    def abort(self) -> None:
        if not self.committed:
            with self.store.lock, _process_commit_lock(self.store._lock_directory):
                if not self.directory.exists():
                    return
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


def bind_learning_state(backend, *, model_path=None, model_info=None, state_root=None, adapter_format="mlx", scope=None) -> LearningStateStore:
    root = _state_root(state_root)
    existing = getattr(backend, "learning_state", None)
    # Scope switches can supply the effective root of an already bound store.
    # Retain its base so repeated binds, including a mode switch, are idempotent.
    if isinstance(existing, LearningStateStore) and root == existing.root:
        root = getattr(existing, "implementation_base_root", root)
    base_root = root
    if getattr(backend, "learning_implementation", None) == "legacy":
        root = root / "legacy-implementation"
    identity = model_identity(
        model_path or getattr(backend, "model_path", None) or getattr(backend, "hf_repo_id", None),
        model_info if model_info is not None else getattr(backend, "model_info", None),
        adapter_format=adapter_format, state_root=base_root,
    )
    scope = as_scope(scope if scope is not None else getattr(backend, "learning_scope", None))
    if (isinstance(existing, LearningStateStore) and existing.identity == identity
            and existing.scope == scope and existing.root == root):
        existing._check_request()
        return existing
    store = LearningStateStore(identity, state_root=root, scope=scope)
    store.implementation_base_root = base_root
    backend.learning_scope = scope
    backend.learning_state = store
    backend.learning_identity = store.identity
    return store


def select_legacy_checkpoint(backend, store, current, legacy_path, selected, adapter_format):
    """Require explicit confirmation of the old, unfingerprinted model-wide artifact."""
    backend.learning_migration_status = None
    if current or getattr(getattr(store, "scope", None), "kind", "model") != "model":
        return None
    legacy = Path(legacy_path).absolute()
    marker = legacy / "adapter_config.json" if adapter_format == "peft" else legacy
    if not marker.is_file():
        if selected:
            raise RuntimeError(f"The selected legacy checkpoint is missing from its original location: {legacy}")
        return None
    if not selected or Path(selected).resolve() != legacy.resolve():
        backend.learning_migration_status = {"status": "legacy_adapter_required", "format": adapter_format,
                                            "path": str(legacy)}
        raise RuntimeError("The existing learned artifact has no verified model identity. "
                           f"Confirm this checkpoint on its original base model: {legacy}")
    return str(legacy)


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
            store.verify_scope(manifest)
            raise RuntimeError("Selected managed checkpoint is not the current generation; use an explicit memory rollback before loading it")
    sidecar = Path(str(selected) + ".memory.json")
    if not sidecar.is_file():
        if allow_legacy:
            return None
        raise RuntimeError("Saved adapter has no proven model identity; select the legacy adapter explicitly to migrate it")
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    if manifest.get("identity_digest") != store.identity_digest or manifest.get("identity") != store.identity:
        raise RuntimeError("Selected adapter belongs to a different model or tokenizer")
    store.verify_scope(manifest)
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
    if store.scope.kind != "model":
        manifest["scope"] = store.scope.record
    _atomic_json(Path(str(selected) + ".memory.json"), manifest)
    return manifest
