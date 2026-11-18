"""Search local document/case checkpoint metadata and route to verified parameters."""
from __future__ import annotations

from collections import Counter
import codecs
from contextlib import closing
import hashlib
import itertools
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import time
from uuid import uuid4
import zipfile

from core.learning_state import _canonical, _check_cancelled, _contained, _fsync_directory
from core.memory_scopes import LearningScope


_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_COLUMNS = "entry_key AS key, title, source, content_sha256, generation, registered_at"
_COMMON = frozenset("a an and are as at be by can did do does for from how i in is it of on or "
                    "that the their this to was were what which with would you your".split())


def _terms(text):
    return (word[:64] for word in _WORD.findall(text.casefold()) if len(word) >= 2 and word not in _COMMON)


def _content_blocks(content):
    """Read the complete source without loading its vocabulary or text into RAM."""
    if isinstance(content, Path):
        with content.open("rb") as stream:
            yield from iter(lambda: stream.read(64 * 1024), b"")
    elif isinstance(content, str):
        for offset in range(0, len(content), 16 * 1024):
            yield content[offset:offset + 16 * 1024].encode("utf-8")
    elif isinstance(content, (bytes, bytearray)):
        view = memoryview(content)
        for offset in range(0, len(view), 64 * 1024):
            yield view[offset:offset + 64 * 1024]
    else:
        raise TypeError("Document content must be text, bytes, or a local Path")


def _index_content(db, owner, key, content, cancel_event):
    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    word = ""

    def index(text, final=False):
        nonlocal word
        counts = Counter()
        for char in text.casefold() + (" " if final else ""):
            if char.isalnum():
                if len(word) < 64:
                    word += char
            elif word:
                if len(word) >= 2 and word not in _COMMON:
                    counts[word] += 1
                word = ""
        db.executemany("""INSERT INTO terms(owner,entry_key,term,weight) VALUES (?,?,?,?)
            ON CONFLICT(owner,term,entry_key) DO UPDATE SET weight=min(6,terms.weight+excluded.weight)""",
            ((owner, key, term, min(count, 6)) for term, count in counts.items()))

    extracted = isinstance(content, Path) and content.suffix.lower() in {".pdf", ".docx", ".parquet"}
    for block in _content_blocks(content):
        _check_cancelled(cancel_event)
        digest.update(block)
        if not extracted:
            index(decoder.decode(block))
    if extracted:
        # Registration binds the original file digest while ranking the same
        # extracted passages that supplied the learning objective.
        from core.learning_sources import iter_file_chunks
        from core.learning_state import file_sha256
        for fragment in iter_file_chunks(content, 16384):
            _check_cancelled(cancel_event)
            index(fragment)
        if file_sha256(content, cancel_event=cancel_event) != digest.hexdigest():
            raise RuntimeError("Document content changed during indexing")
    index(decoder.decode(b"", final=True), final=True)
    _check_cancelled(cancel_event)
    return digest.hexdigest()


def _bounded_text(value, name, limit):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be nonempty text up to {limit} characters")
    return value


class AdapterMemoryBank:
    """One owner namespace on an immutable model/tokenizer identity."""

    def __init__(self, owner_store):
        self.owner = owner_store
        if self.owner.scope.kind == "request" or self.owner.scope.pin:
            raise ValueError("Transient or read-only scopes cannot own a durable adapter bank")
        self.owner_id = self.owner.scope.scope_id
        self.owner.bank_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS entries (
                    owner TEXT NOT NULL, entry_key TEXT NOT NULL, document_scope TEXT NOT NULL,
                    title TEXT NOT NULL, source TEXT NOT NULL, content_sha256 TEXT NOT NULL,
                    generation TEXT NOT NULL, manifest_sha256 TEXT NOT NULL,
                    pin TEXT NOT NULL UNIQUE, registered_at REAL NOT NULL,
                    PRIMARY KEY (owner, entry_key)
                );
                CREATE INDEX IF NOT EXISTS generations_pinned ON entries(document_scope, generation);
                CREATE TABLE IF NOT EXISTS terms (
                    owner TEXT NOT NULL, entry_key TEXT NOT NULL, term TEXT NOT NULL, weight INTEGER NOT NULL,
                    PRIMARY KEY(owner, term, entry_key),
                    FOREIGN KEY(owner, entry_key) REFERENCES entries(owner, entry_key) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS entry_terms ON terms(owner, entry_key);
            """)

    def _connect(self):
        db = sqlite3.connect(self.owner.bank_path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA temp_store=FILE")
        return db

    def document_scope(self, key):
        key = _bounded_text(key, "Document key", 1024)
        name = hashlib.sha256(_canonical([self.owner_id, key])).hexdigest()
        return LearningScope("document", name)

    def register(self, key, document_store, *, title, source, content, keywords=(), expected_generation=None,
                 expected_content_sha256=None, cancel_event=None):
        """Index a committed document generation. Entry and GC pin change in one transaction."""
        scope = self.document_scope(key)
        if (document_store.identity != self.owner.identity or document_store.root != self.owner.root
                or document_store.scope != scope):
            raise ValueError("Document checkpoint belongs to another model, tokenizer, scope or owner")
        title = _bounded_text(title, "Title", 512)
        source = _bounded_text(source, "Source", 2048)
        if isinstance(keywords, str):
            keywords = [keywords]
        hints = " ".join(_bounded_text(word, "Keyword", 128) for word in itertools.islice(keywords, 64))
        weights = {}
        for text, bonus in [(title, 10), (source, 2), (hints, 8)]:
            for term in set(_terms(text)):
                weights[term] = weights.get(term, 0) + bonus

        with document_store.locked_generation(cancel_event=cancel_event) as (manifest, pointer):
            if not manifest or (expected_generation is not None and expected_generation != manifest["generation"]):
                raise RuntimeError("The requested committed document generation is unavailable")
            generation = manifest["generation"]
            with closing(self._connect()) as db, db:
                db.execute("BEGIN IMMEDIATE")
                db.execute("DELETE FROM entries WHERE owner=? AND entry_key=?", (self.owner_id, key))
                db.execute("""INSERT INTO entries(owner,entry_key,document_scope,title,source,content_sha256,
                              generation,manifest_sha256,pin,registered_at) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                           (self.owner_id, key, scope.scope_id, title, source, "", generation,
                            pointer["manifest_sha256"], uuid4().hex, time.time()))
                digest = _index_content(db, self.owner_id, key, content, cancel_event)
                if expected_content_sha256 is not None and digest != expected_content_sha256:
                    raise RuntimeError("Document content changed during learning")
                db.execute("UPDATE entries SET content_sha256=? WHERE owner=? AND entry_key=?",
                           (digest, self.owner_id, key))
                db.executemany("""INSERT INTO terms(owner,entry_key,term,weight) VALUES (?,?,?,?)
                    ON CONFLICT(owner,term,entry_key) DO UPDATE SET weight=terms.weight+excluded.weight""",
                               ((self.owner_id, key, term, weight) for term, weight in weights.items()))
                _check_cancelled(cancel_event)
                return self._entry(db, key)

    def _entry(self, db, key):
        row = db.execute(f"SELECT {_COLUMNS},pin FROM entries WHERE owner=? AND entry_key=?",
                         (self.owner_id, key)).fetchone()
        if not row:
            raise KeyError(f"No document entry for {key!r} in this scope")
        return dict(row)

    def list_entries(self, *, limit=100, offset=0):
        if not isinstance(limit, int) or not 1 <= limit <= 256 or not isinstance(offset, int) or offset < 0:
            raise ValueError("Use a positive page size up to 256 and a nonnegative offset")
        with closing(self._connect()) as db:
            rows = db.execute(f"""SELECT {_COLUMNS},pin FROM entries WHERE owner=?
                                  ORDER BY entry_key LIMIT ? OFFSET ?""", (self.owner_id, limit, offset))
            return [dict(row) for row in rows]

    def select(self, query, top_k=1):
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be positive")
        if not isinstance(query, str):
            raise ValueError("Memory query must be text")
        terms = set(_terms(query))
        if not terms:
            return []
        # Only postings matching this owner's query terms enter ranking. SQLite sorts on disk as needed.
        with closing(self._connect()) as db:
            db.execute("CREATE TEMP TABLE query_terms(term TEXT PRIMARY KEY)")
            db.executemany("INSERT INTO query_terms(term) VALUES (?)", ((term,) for term in terms))
            rows = db.execute(f"""WITH ranked AS (
                SELECT entry_key,count(*) AS overlap,sum(weight) AS score FROM terms
                WHERE owner=? AND term IN (SELECT term FROM query_terms) GROUP BY entry_key
                ORDER BY overlap DESC,score DESC,entry_key LIMIT ?
            ) SELECT e.entry_key AS key,title,source,content_sha256,generation,registered_at,pin,score,overlap
              FROM ranked JOIN entries AS e ON e.owner=? AND e.entry_key=ranked.entry_key
              ORDER BY overlap DESC,score DESC,e.entry_key""", (self.owner_id, top_k, self.owner_id))
            return [dict(row) for row in rows]

    def _pinned_scope(self, entry):
        with closing(self._connect()) as db:
            current = self._entry(db, entry["key"])
        if current["pin"] != entry["pin"] or current["generation"] != entry["generation"]:
            raise RuntimeError("Document selection changed or belongs to a different owner")
        base = self.document_scope(current["key"])
        return LearningScope(base.kind, base.key, current["generation"], current["pin"])

    def activation_scope(self, entry):
        """Select a read-only generation through the backend's normal restore path."""
        scope = self._pinned_scope(entry)
        self.owner.for_scope(scope).current_manifest()
        return scope

    def resolve(self, entry, loaded_store):
        scope = self._pinned_scope(entry)
        if (loaded_store.identity != self.owner.identity or loaded_store.scope != scope
                or loaded_store.root != self.owner.root):
            raise ValueError("Loaded model, tokenizer or memory scope does not match the selected adapter")
        return loaded_store.current_checkpoint()

    def delete(self, key):
        """Remove the owner's catalog entry and pin. The current training checkpoint remains intact."""
        with closing(self._connect()) as db, db:
            return db.execute("DELETE FROM entries WHERE owner=? AND entry_key=?", (self.owner_id, key)).rowcount == 1

    def export(self, key, destination):
        """Write a new ZIP with verified artifact files, checkpoint manifest and source metadata."""
        destination = Path(destination)
        with closing(self._connect()) as db:
            entry = self._entry(db, key)
        store = self.owner.for_scope(self._pinned_scope(entry))
        with store.locked_generation() as (manifest, pointer):
            directory = store.directory / "generations" / manifest["generation"]
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as staged:
                temporary = Path(staged.name)
            try:
                with zipfile.ZipFile(temporary, "w", allowZip64=True) as bundle:
                    bundle.writestr("entry.json", _canonical({k: v for k, v in entry.items() if k != "pin"}))
                    bundle.write(directory / "manifest.json", "manifest.json")
                    for relative in manifest["artifacts"]:
                        bundle.write(_contained(directory, relative), "artifacts/" + relative)
                with temporary.open("rb") as stream:
                    os.fsync(stream.fileno())
                os.link(temporary, destination)
                _fsync_directory(destination.parent)
            finally:
                temporary.unlink(missing_ok=True)
        return destination
