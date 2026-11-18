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
from contextlib import closing
from itertools import chain, islice
from pathlib import Path
from uuid import uuid4


class TranscriptTurn(dict):
    """The tokenizer sees ordinary chat fields. The archive owns the receipt ID."""

    def __init__(self, values, turn_id):
        super().__init__(values)
        self.turn_id = turn_id

    def copy(self):
        return type(self)(self, self.turn_id)


class ChatArchive(list):
    """Append-only visible transcript for a GUI model tab, resumable on restart."""

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
            rows = db.execute("SELECT turn_id, message FROM chat_archive WHERE tab=? ORDER BY seq", (tab,))
            super().__init__(TranscriptTurn(json.loads(message), key) for key, message in rows)

    def append(self, turn):
        with self._lock:
            turn_id = uuid4().hex
            with closing(sqlite3.connect(self.path)) as db, db:
                db.execute("INSERT INTO chat_archive(tab,turn_id,message) VALUES (?,?,?)",
                           (self.tab, turn_id, json.dumps(turn, ensure_ascii=False)))
            super().append(TranscriptTurn(turn, turn_id))

    def clear(self):
        with self._lock:
            with closing(sqlite3.connect(self.path)) as db, db:
                db.execute("DELETE FROM chat_archive WHERE tab=?", (self.tab,))
            super().clear()
            self._projection = None

    def active_view(self, ledger, head):
        """Revisit only active and newly appended turns while the model lineage advances."""
        with self._lock:
            cached = self._projection
            source = iter(self)
            if cached and cached[0] == ledger.path:
                _, old_head, length, active = cached
                if old_head == head:
                    result = active + self[length:]
                    self._projection = (ledger.path, head, len(self), result)
                    return list(result)
                if ledger.descends_from(head, old_head):
                    source = chain(active, self[length:])
            result = ledger.active_turns(source, head)
            self._projection = (ledger.path, head, len(self), result)
            return list(result)


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


def completed_groups(history):
    """Identify contiguous ordinary dialogue. Keep control turns and tool exchanges in text."""
    groups = []
    pending = []
    for index, turn in enumerate(history):
        role = turn.get("role")
        text = turn.get("content")
        plain = (isinstance(turn, TranscriptTurn) and isinstance(text, str) and bool(text.strip())
                 and not turn.get("tool_calls") and not turn.get("function_call")
                 and "<tool_call>" not in text and "<media_call>" not in text
                 and not text.startswith(("Actual tool result (", "Local media tool result (")))
        if plain and role == "user":
            pending.append(index)
        elif plain and role == "assistant" and pending:
            next_role = history[index + 1].get("role") if index + 1 < len(history) else None
            if next_role not in {"assistant", "tool", "function"}:
                groups.append((*pending, index))
            pending = []
        else:
            pending = []
    return groups


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
