"""Durable chat transcripts and checkpoint-scoped context retirement.

Retirement receipts contain turn IDs only. A receipt becomes visible when the
adapter checkpoint publishing its head becomes current. Abandoned candidates
cannot hide turns, and loading an earlier checkpoint restores its own view.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Sequence
from contextlib import closing, contextmanager
from itertools import chain, islice, zip_longest
from operator import index as integer_index
from pathlib import Path
from uuid import uuid4


class ContextOverflow(RuntimeError):
    """Protected or unverified text still fills the available working window."""


class TranscriptTurn(dict):
    """The tokenizer sees ordinary chat fields. The archive owns the receipt ID."""

    def __init__(self, values, turn_id):
        super().__init__(values)
        self.turn_id = turn_id

    def copy(self):
        copied = type(self)(self, self.turn_id)
        if hasattr(self, "_capacity_recovery"):
            copied._capacity_recovery = self._capacity_recovery
        return copied


class ChatArchive(Sequence):
    """SQLite-backed visible transcript; only the active model projection is cached."""

    def __init__(self, db_path, tab):
        self.path = db_path
        self.tab = tab
        self._projection = None
        self._lock = threading.RLock()
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS chat_archive (
                seq INTEGER PRIMARY KEY, tab TEXT NOT NULL, turn_id TEXT UNIQUE NOT NULL,
                message TEXT NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS chat_archive_tab ON chat_archive(tab, seq)")
            db.execute("""CREATE TABLE IF NOT EXISTS chat_epochs (
                tab TEXT PRIMARY KEY, epoch INTEGER NOT NULL)""")
            db.execute("INSERT OR IGNORE INTO chat_epochs VALUES (?,0)", (tab,))

    @staticmethod
    def _decode(row):
        return TranscriptTurn(json.loads(row[1]), row[0])

    def __len__(self):
        with closing(sqlite3.connect(self.path)) as db:
            return db.execute("SELECT count(*) FROM chat_archive WHERE tab=?", (self.tab,)).fetchone()[0]

    def __bool__(self):
        with closing(sqlite3.connect(self.path)) as db:
            return db.execute("SELECT 1 FROM chat_archive WHERE tab=? LIMIT 1", (self.tab,)).fetchone() is not None

    def iter_pages(self, page_size=128):
        """Stream one consistent SQLite read snapshot in transcript order."""
        if not 1 <= integer_index(page_size) <= 1024:
            raise ValueError("page_size must be between 1 and 1024")
        with closing(sqlite3.connect(self.path)) as db:
            rows = db.execute("SELECT turn_id,message FROM chat_archive WHERE tab=? ORDER BY seq", (self.tab,))
            while page := rows.fetchmany(page_size):
                yield [self._decode(row) for row in page]

    def __iter__(self):
        return chain.from_iterable(self.iter_pages())

    def __getitem__(self, key):
        with closing(sqlite3.connect(self.path)) as db:
            if not isinstance(key, slice):
                index = integer_index(key)
                order, offset = ("ASC", index) if index >= 0 else ("DESC", -index - 1)
                if offset < 2**63:
                    row = db.execute(f"""SELECT turn_id,message FROM chat_archive WHERE tab=?
                                         ORDER BY seq {order} LIMIT 1 OFFSET ?""", (self.tab, offset)).fetchone()
                    if row is not None:
                        return self._decode(row)
                raise IndexError("Transcript index out of range")
            db.execute("BEGIN")
            count = db.execute("SELECT count(*) FROM chat_archive WHERE tab=?", (self.tab,)).fetchone()[0]
            start, stop, step = key.indices(count)
            size = len(range(start, stop, step))
            if not size:
                return []
            last = start + (size - 1) * step
            low, high = min(start, last), max(start, last)
            # Start at whichever end of the indexed tab is closer to this page.
            backwards = count - 1 - high < low
            order, offset = ("DESC", count - 1 - high) if backwards else ("ASC", low)
            rows = db.execute(f"""SELECT turn_id,message FROM chat_archive WHERE tab=?
                                  ORDER BY seq {order} LIMIT ? OFFSET ?""", (self.tab, high - low + 1, offset))
            page = [self._decode(row) for i, row in enumerate(rows) if i % abs(step) == 0]
            if backwards != (step < 0):
                page.reverse()
            return page

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, (ChatArchive, list)):
            return NotImplemented
        return all(a == b for a, b in zip_longest(self, other, fillvalue=object()))

    def append(self, turn):
        with self._lock, closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT INTO chat_archive(tab,turn_id,message) VALUES (?,?,?)",
                       (self.tab, uuid4().hex, json.dumps(turn, ensure_ascii=False)))

    def clear(self):
        with self._lock, closing(sqlite3.connect(self.path)) as db, db:
            db.execute("DELETE FROM chat_archive WHERE tab=?", (self.tab,))
            db.execute("UPDATE chat_epochs SET epoch=epoch+1 WHERE tab=?", (self.tab,))
            self._projection = None

    def active_view(self, ledger, head):
        """Decode unretired rows only, even on restart or checkpoint rollback."""
        with self._lock:
            cached = self._projection
            if cached and cached[0] == ledger.path:
                _, old_head, old_epoch, last_seq, active = cached
                with closing(sqlite3.connect(self.path)) as db:
                    db.execute("BEGIN")
                    epoch, newest = self._watermark(db)
                    if old_epoch == epoch and (old_head == head or ledger.descends_from(head, old_head)):
                        rows = db.execute("""SELECT turn_id,message FROM chat_archive
                                             WHERE tab=? AND seq>? ORDER BY seq""", (self.tab, last_seq))
                        appended = map(self._decode, rows)
                        if old_head == head:
                            result = active + list(appended)
                        else:
                            result = ledger.active_turns(chain(active, appended), head)
                        self._projection = (ledger.path, head, epoch, newest, result)
                        return [turn.copy() for turn in result]

            with self._active_rows(ledger, head) as (db, rows):
                epoch, newest = self._watermark(db, schema="transcript.")
                result = [self._decode(row) for row in rows]
            self._projection = (ledger.path, head, epoch, newest, result)
            return [turn.copy() for turn in result]

    _active_filter = """FROM transcript.chat_archive AS a WHERE a.tab=? AND NOT EXISTS
                     (SELECT 1 FROM retired_turns AS r WHERE r.turn_id=a.turn_id)"""

    @contextmanager
    def _active_query(self, ledger, head):
        with closing(sqlite3.connect(ledger.path)) as db, db:
            db.execute("ATTACH DATABASE ? AS transcript", (str(self.path),))
            db.execute("BEGIN IMMEDIATE")
            ledger._index(db, head)
            yield db

    @contextmanager
    def _active_rows(self, ledger, head):
        with self._active_query(ledger, head) as db:
            rows = db.execute("SELECT a.turn_id,a.message " + self._active_filter + " ORDER BY a.seq", (self.tab,))
            yield db, rows

    def projected_history(self, ledger, head):
        # Keep the small-chat append cache for interactive use. Determine its
        # size in SQLite, before bringing message bodies into Python.
        with self._active_query(ledger, head) as db:
            rows, size = db.execute("SELECT count(*),coalesce(sum(n),0) FROM (SELECT length(a.message) AS n "
                                    + self._active_filter + " LIMIT 129)", (self.tab,)).fetchone()
        if rows <= 128 and size <= 65536:
            return self.active_view(ledger, head)
        return ActiveChatHistory(self, ledger, head)

    def iter_active_pages(self, ledger, head, page_size=128):
        """Stream unretired rows without allocating a complete active projection."""
        if not head:
            yield from self.iter_pages(page_size)
            return
        if not 1 <= integer_index(page_size) <= 1024:
            raise ValueError("page_size must be between 1 and 1024")
        with self._active_rows(ledger, head) as (_, rows):
            while page := rows.fetchmany(page_size):
                yield [self._decode(row) for row in page]

    def active_turns_after(self, turn_id, current_receipt, page_size=16):
        """Walk later active rows once, releasing the ledger before each training attempt."""
        if not 1 <= integer_index(page_size) <= 1024:
            raise ValueError("page_size must be between 1 and 1024")
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute("SELECT seq FROM chat_archive WHERE tab=? AND turn_id=?", (self.tab, turn_id)).fetchone()
            if row is None:
                return
            after = row[0]
            _, end = self._watermark(db)
        while after < end:
            ledger, head = current_receipt()
            if head:
                with self._active_query(ledger, head) as db:
                    rows = db.execute("SELECT a.seq,a.turn_id,a.message " + self._active_filter
                                      + " AND a.seq>? AND a.seq<=? ORDER BY a.seq LIMIT ?",
                                      (self.tab, after, end, page_size)).fetchall()
            else:
                with closing(sqlite3.connect(self.path)) as db:
                    rows = db.execute("SELECT seq,turn_id,message FROM chat_archive WHERE tab=? "
                                      "AND seq>? AND seq<=? ORDER BY seq LIMIT ?",
                                      (self.tab, after, end, page_size)).fetchall()
            if not rows:
                return
            after = rows[-1][0]
            for _, key, message in rows:
                yield self._decode((key, message))

    def _watermark(self, db, schema=""):
        return db.execute(f"""SELECT epoch,(SELECT coalesce(max(seq),0) FROM {schema}chat_archive
                              WHERE tab=?) FROM {schema}chat_epochs WHERE tab=?""",
                          (self.tab, self.tab)).fetchone()


class LegacyChatArchive(ChatArchive):
    """Retain every visible turn while the original learner compacts a working view.

    Legacy learning has no source-free recall validator. Its existing successful
    parameter update may retire a completed exchange from the working prompt
    only when a durable, matching loaded checkpoint exists. Missing, replaced
    or rolled-back weights restore the unretired source text automatically.
    """

    def __init__(self, db_path, tab):
        super().__init__(db_path, tab)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS legacy_chat_receipts (
                tab TEXT NOT NULL, model_key TEXT NOT NULL, generation TEXT NOT NULL,
                retired TEXT NOT NULL, PRIMARY KEY(tab, model_key, generation))""")

    @staticmethod
    def _checkpoint(backend):
        store = getattr(backend, "learning_state", None)
        if store is None:
            return None
        try:
            manifest = store.current_manifest(verify=False)
            if not manifest:
                return None
            generation = manifest["generation"]
            expected = (Path(store.directory) / "generations" / generation /
                        manifest["checkpoint"]).resolve()
            # A current pointer on disk is not sufficient: the live runtime
            # must have activated that exact artifact before using its receipt.
            actual = next((getattr(backend, name, None) for name in (
                "_loaded_learning_checkpoint", "_last_managed_checkpoint",
                "_loaded_legacy_deployment", "adapter_path")
                if isinstance(getattr(backend, name, None), (str, Path))
                and getattr(backend, name, None)), None)
            if actual is None or Path(actual).resolve() != expected or not expected.exists():
                return None
            if getattr(backend, "_mlx_parent_generation", generation) != generation:
                return None
            identity = json.dumps([str(Path(store.directory).resolve()), store.identity_digest],
                                  ensure_ascii=False, separators=(",", ":"))
            return identity, generation, manifest.get("parent")
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
            return None

    def legacy_projection(self, backend):
        state = self._checkpoint(backend)
        retired = set()
        if state:
            model_key, generation, parent = state
            with closing(sqlite3.connect(self.path)) as db:
                row = db.execute("SELECT retired FROM legacy_chat_receipts WHERE tab=? AND model_key=? AND generation=?",
                                 (self.tab, model_key, generation)).fetchone()
                if row is None and parent:
                    # A direct descendant includes its authenticated parent's
                    # learned weights. No receipt from another model may apply.
                    row = db.execute("SELECT retired FROM legacy_chat_receipts WHERE tab=? AND model_key=? AND generation=?",
                                     (self.tab, model_key, parent)).fetchone()
            if row:
                retired = set(json.loads(row[0]))
        return [turn for turn in self if turn.turn_id not in retired], retired, state

    def record_legacy_update(self, backend, previous, retired, learned):
        """Bind retirement IDs to the actual newly published learned generation."""
        state = self._checkpoint(backend)
        if not state or state == previous:
            return None
        model_key, generation, parent = state
        if previous is None:
            if parent is not None:
                return None
        elif previous[0] != model_key or previous[1] != parent:
            return None
        updated = set(retired) | {turn.turn_id for turn in learned if isinstance(turn, TranscriptTurn)}
        with self._lock, closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT OR REPLACE INTO legacy_chat_receipts VALUES (?,?,?,?)",
                       (self.tab, model_key, generation, json.dumps(sorted(updated))))
            # Match the normal checkpoint store's current/parent retention.
            db.execute("DELETE FROM legacy_chat_receipts WHERE tab=? AND model_key=? AND generation NOT IN (?,?)",
                       (self.tab, model_key, generation, parent or generation))
        return updated, state

    def clear(self):
        super().clear()
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("DELETE FROM legacy_chat_receipts WHERE tab=?", (self.tab,))


class ActiveChatHistory(Sequence):
    """Indexable, paged projection of a large transcript against a particular receipt."""

    def __init__(self, archive, ledger, head):
        self.archive = archive
        self.ledger = ledger
        self.head = head

    def __iter__(self):
        return chain.from_iterable(self.archive.iter_active_pages(self.ledger, self.head, page_size=16))

    def __len__(self):
        with self.archive._active_query(self.ledger, self.head) as db:
            return db.execute("SELECT count(*) " + self.archive._active_filter, (self.archive.tab,)).fetchone()[0]

    def __getitem__(self, key):
        with self.archive._active_query(self.ledger, self.head) as db:
            query = "SELECT a.turn_id,a.message " + self.archive._active_filter
            if isinstance(key, slice):
                size = db.execute("SELECT count(*) " + self.archive._active_filter, (self.archive.tab,)).fetchone()[0]
                start, stop, step = key.indices(size)
                count = len(range(start, stop, step))
                if not count:
                    return []
                direction = "ASC" if step > 0 else "DESC"
                offset = start if step > 0 else size - 1 - start
                width = (count - 1) * abs(step) + 1
                rows = db.execute(query + f" ORDER BY a.seq {direction} LIMIT ? OFFSET ?",
                                  (self.archive.tab, width, offset))
                return [self.archive._decode(row) for i, row in enumerate(rows) if i % abs(step) == 0]
            index = integer_index(key)
            direction, offset = ("ASC", index) if index >= 0 else ("DESC", -index - 1)
            if offset < 2**63:
                row = db.execute(query + f" ORDER BY a.seq {direction} LIMIT 1 OFFSET ?",
                                 (self.archive.tab, offset)).fetchone()
                if row is not None:
                    return self.archive._decode(row)
            raise IndexError("Transcript index out of range")

    def __eq__(self, other):
        if not isinstance(other, Sequence):
            return NotImplemented
        return all(a == b for a, b in zip_longest(self, other, fillvalue=object()))


def working_history(history, session_id=""):
    """Identify plain API transcripts across full-list resubmissions, including repeats.

    GUI turns already carry per-session IDs. Tool-round additions in that case
    are ephemeral and are deliberately never eligible for retirement.
    """
    history = list(history or ())
    if any(isinstance(turn, TranscriptTurn) for turn in history):
        return history
    digest = hashlib.sha256(str(session_id).encode("utf-8") + b"\x00")
    result = []
    for turn in history:
        if turn.get("role") not in {"user", "assistant"}:
            result.append(turn)
            continue
        record = json.dumps(turn, sort_keys=True, ensure_ascii=False).encode()
        digest.update(len(record).to_bytes(8, "big"))
        digest.update(record)
        result.append(TranscriptTurn(turn, digest.hexdigest()))
    return result


def ordinary_turn(turn):
    text = turn.get("content")
    return (isinstance(turn, TranscriptTurn) and isinstance(text, str) and bool(text.strip())
            and not turn.get("tool_calls") and not turn.get("function_call")
            and "<tool_call>" not in text and "<media_call>" not in text
            and not text.startswith(("Actual tool result (", "Local media tool result (")))


def _ordinary_exchanges(history):
    """Recognize a completed assistant reply only after its following role is known."""
    pending = []
    reply = None
    for index, turn in enumerate(history):
        role = turn.get("role")
        if reply is not None:
            if role not in {"assistant", "tool", "function"}:
                yield [*pending, reply]
            pending = []
            reply = None
        if role == "user" and ordinary_turn(turn):
            pending.append((index, turn))
        elif role == "assistant" and pending and ordinary_turn(turn):
            reply = (index, turn)
        else:
            pending = []
    if reply is not None:
        yield [*pending, reply]


def iter_completed_exchanges(history):
    for group in _ordinary_exchanges(history):
        yield [turn for _, turn in group]


def completed_groups(history):
    """Identify contiguous ordinary dialogue. Keep control turns and tool exchanges in text."""
    return [tuple(i for i, _ in group) for group in _ordinary_exchanges(history)]


def oldest_completed_chunk(history, ratio=0.4, *, keep_recent=True):
    groups = completed_groups(history)
    eligible = groups[:-1] if keep_recent else groups
    target = max(2, int(len(history) * min(0.8, max(0.05, ratio))))
    indices = set()
    for group in eligible:
        indices.update(group)
        if len(indices) >= target:
            break
    return ([turn for index, turn in enumerate(history) if index in indices],
            [turn for index, turn in enumerate(history) if index not in indices])


class RetirementLedger:
    """Persistent immutable receipt chain referenced by the checkpoint manifest."""

    def __init__(self, directory):
        self.path = Path(directory) / "retirement.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS receipts (
                head TEXT PRIMARY KEY, parent TEXT, turns TEXT NOT NULL)""")
            db.execute("CREATE TABLE IF NOT EXISTS retired_turns (turn_id TEXT PRIMARY KEY)")
            db.execute("CREATE TABLE IF NOT EXISTS indexed_head (singleton INTEGER PRIMARY KEY, head TEXT)")

    def stage(self, head, parent, turn_ids):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT INTO receipts(head,parent,turns) VALUES (?,?,?)",
                       (head, parent, json.dumps(turn_ids)))

    def discard(self, head):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("DELETE FROM receipts WHERE head=?", (head,))

    # Traverse only the suffix after the already indexed head when possible.
    _lineage = """WITH RECURSIVE lineage(head, parent, turns) AS (
        SELECT head, parent, turns FROM receipts WHERE head=? AND head IS NOT ?
        UNION ALL
        SELECT r.head, r.parent, r.turns FROM receipts r JOIN lineage l ON r.head=l.parent
        WHERE r.head IS NOT ?
    ) """

    def descends_from(self, head, ancestor):
        with closing(sqlite3.connect(self.path)) as db:
            return bool(db.execute(self._lineage + "SELECT 1 FROM lineage WHERE parent IS ? LIMIT 1",
                                   (head, ancestor, ancestor, ancestor)).fetchone())

    def _index(self, db, head):
        row = db.execute("SELECT head FROM indexed_head WHERE singleton=1").fetchone()
        indexed = row[0] if row else None
        if indexed == head:
            return

        def reaches(ancestor):
            return db.execute(self._lineage + "SELECT 1 FROM lineage WHERE parent IS ? LIMIT 1",
                              (head, ancestor, ancestor, ancestor)).fetchone()

        if not reaches(indexed):
            if not indexed or not reaches(None):
                raise RuntimeError("The checkpoint's retirement receipts are incomplete. Retaining the transcript.")
            # Model rollback or restoring another branch: rebuild the on-disk index.
            db.execute("DELETE FROM retired_turns")
            indexed = None
        db.execute(self._lineage + """INSERT OR IGNORE INTO retired_turns(turn_id)
                   SELECT j.value FROM lineage l, json_each(l.turns) j""", (head, indexed, indexed))
        db.execute("INSERT OR REPLACE INTO indexed_head(singleton,head) VALUES (1,?)", (head,))

    def active_turns(self, turns, head):
        """Query membership in batches. Lifetime retirement IDs stay on disk."""
        result = []
        source = iter(turns)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            self._index(db, head)
            while batch := list(islice(source, 256)):
                ids = [getattr(turn, "turn_id", None) for turn in batch]
                parameters = ",".join("?" for _ in ids)
                rows = db.execute(f"SELECT turn_id FROM retired_turns WHERE turn_id IN ({parameters})", ids)
                retired = {row[0] for row in rows}
                result.extend(turn for turn, key in zip(batch, ids) if key not in retired)
        return result
