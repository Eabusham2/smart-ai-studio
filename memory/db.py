"""
Episodic SQLite memory manager.
Captures interactive traces, surprise scores, sandbox assertions,
and tracks consolidated parameter updates across sleep cycles.
"""

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class _ClosingSQLiteConnection(sqlite3.Connection):
    """sqlite connection whose context manager also releases the OS file handle.

    Python's standard sqlite3.Connection context manager commits/rolls back but
    deliberately leaves the connection open.  Most of this project uses
    ``with db._get_connection() as conn`` and historically assumed that meant
    closed-on-exit. POSIX masks the leak because an open file can be unlinked;
    Windows correctly rejects deletion of the still-open database file.

    Preserve sqlite's transaction semantics first, then always close the handle.
    """

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


class EpisodicMemoryDB:
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = os.fspath(db_path)
        # Fresh release archives do not retain empty data/ directories.
        # Create the parent before SQLite opens, without touching existing data.
        parent = os.path.dirname(os.path.abspath(self.db_path))
        os.makedirs(parent, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, factory=_ClosingSQLiteConnection)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Initializes database schema and indices."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt TEXT NOT NULL,
                    completion TEXT NOT NULL,
                    raw_branches TEXT,
                    verified_reward REAL DEFAULT 0.0,
                    surprise_score REAL DEFAULT 0.0,
                    mode TEXT,
                    entropy REAL,
                    winning_branch INTEGER DEFAULT 0,
                    winning_temp REAL DEFAULT 0.20,
                    test_cases TEXT,
                    consolidated INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""CREATE TABLE IF NOT EXISTS verified_sleep_batch (
                slot INTEGER PRIMARY KEY CHECK (slot = 1), token TEXT NOT NULL,
                target TEXT NOT NULL, episode_ids TEXT NOT NULL)""")
            try:
                cursor.execute("ALTER TABLE interactions ADD COLUMN winning_temp REAL DEFAULT 0.20")
            except sqlite3.OperationalError:
                pass
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    memories_count INTEGER NOT NULL,
                    anchors_count INTEGER NOT NULL,
                    ewc_lambda REAL NOT NULL,
                    avg_task_loss REAL,
                    avg_ewc_loss REAL,
                    adapter_path TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_interactions_consolidation
                ON interactions (verified_reward, consolidated, surprise_score DESC)
            """)
            conn.commit()

    def log_interaction(
        self,
        prompt: str,
        completion: str,
        raw_branches: Optional[List[str]] = None,
        verified_reward: float = 0.0,
        surprise_score: float = 0.0,
        mode: str = "Instant",
        entropy: float = 0.0,
        winning_branch: int = 0,
        test_cases: Optional[str] = None,
        winning_temp: float = 0.20,
        **kwargs
    ) -> int:
        """Logs an interactive trace into episodic memory with winning branch temperature."""
        if "winning_temp" in kwargs:
            winning_temp = kwargs["winning_temp"]
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO interactions (
                    prompt, completion, raw_branches, verified_reward,
                    surprise_score, mode, entropy, winning_branch,
                    winning_temp, test_cases, consolidated, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """, (
                prompt,
                completion,
                json.dumps(raw_branches) if raw_branches else None,
                float(verified_reward),
                float(surprise_score),
                mode,
                float(entropy),
                int(winning_branch),
                float(winning_temp),
                test_cases,
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()
            return cursor.lastrowid

    def get_unconsolidated_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves unconsolidated interaction traces for UI table and inspector."""
        query = """
            SELECT id, prompt, completion, verified_reward, surprise_score, mode, entropy, winning_branch, winning_temp, consolidated, created_at as timestamp
            FROM interactions
            ORDER BY id DESC LIMIT ?
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def fetch_surprise_replay_data(
        self,
        limit: int = 50,
        unconsolidated_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieves verified interactions prioritized by surprise score
        for generative replay during sleep consolidation.
        """
        query = """
            SELECT id, prompt, completion, surprise_score, entropy, mode
            FROM interactions
            WHERE verified_reward = 1.0
        """
        if unconsolidated_only:
            query += """ AND consolidated = 0 AND id NOT IN
                (SELECT value FROM verified_sleep_batch, json_each(episode_ids))"""
        query += " ORDER BY surprise_score DESC LIMIT ?"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def sleep_queue_lock(self):
        """Serialize queue consumers across processes, without blocking chat/database writers."""
        from core.learning_state import _process_commit_lock

        path = Path(self.db_path).resolve()
        lock_dir = path.with_name(path.name + ".sleep")
        lock_dir.mkdir(exist_ok=True)
        return _process_commit_lock(lock_dir)

    @staticmethod
    def _reserved_sleep_rows(conn, ids):
        return [dict(row) for row in conn.execute("""SELECT interactions.id,prompt,completion
                FROM interactions JOIN json_each(?) AS episode ON interactions.id=episode.value
                WHERE verified_reward=1 AND consolidated=0 ORDER BY CAST(episode.key AS INTEGER)""", (ids,))]

    @staticmethod
    def episode_digest(rows):
        digest = hashlib.sha256()
        for row in rows:
            record = [row["id"], row["prompt"], row["completion"]]
            digest.update(json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode() + b"\n")
        return digest.hexdigest()

    def prepare_verified_sleep(self, target, limit):
        """Reserve a stable batch before training. Caller holds the queue's consumer lock."""
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            pending = conn.execute("SELECT * FROM verified_sleep_batch WHERE slot=1").fetchone()
            if pending:
                if pending["target"] != target:
                    raise RuntimeError("Resume the unfinished verified Sleep batch on its original "
                                       + "model and memory scope: " + pending["target"])
                rows = self._reserved_sleep_rows(conn, pending["episode_ids"])
                if len(rows) != len(json.loads(pending["episode_ids"])):
                    raise RuntimeError("Reserved verified Sleep episodes changed before acknowledgement")
                return pending["token"], rows
            rows = conn.execute("""SELECT id,prompt,completion FROM interactions
                 WHERE verified_reward=1 AND consolidated=0 ORDER BY surprise_score DESC,id ASC LIMIT ?""",
                                (limit,)).fetchall()
            if not rows:
                return None, []
            token = uuid4().hex
            conn.execute("INSERT INTO verified_sleep_batch VALUES (1,?,?,?)",
                         (token, target, json.dumps([row["id"] for row in rows])))
            return token, [dict(row) for row in rows]

    def release_verified_sleep(self, token):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM verified_sleep_batch WHERE token=?", (token,))

    def finish_verified_sleep(self, token, checkpoint, receipt):
        """Atomically acknowledge the checkpoint's exact batch and write the legacy Sleep log."""
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT episode_ids FROM verified_sleep_batch WHERE token=?", (token,)).fetchone()
            if not row:
                raise RuntimeError("Verified Sleep reservation is missing")
            rows = self._reserved_sleep_rows(conn, row[0])
            if self.episode_digest(rows) != receipt["episode_digest"]:
                raise RuntimeError("Verified episode content changed before checkpoint acknowledgement")
            conn.execute("UPDATE interactions SET consolidated=1 WHERE id IN (SELECT value FROM json_each(?))",
                         (row[0],))
            conn.execute("""INSERT INTO consolidation_logs (memories_count,anchors_count,ewc_lambda,adapter_path)
                         VALUES (?,?,?,?)""", (len(rows), receipt["anchors"], receipt["ewc_lambda"], checkpoint))
            conn.execute("DELETE FROM verified_sleep_batch WHERE token=?", (token,))

    def mark_consolidated(self, interaction_ids: List[int]) -> None:
        """Flags episodic memories as consolidated into model parameters."""
        if not interaction_ids:
            return
        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(interaction_ids))
            cursor.execute(
                f"UPDATE interactions SET consolidated = 1 WHERE id IN ({placeholders})",
                interaction_ids
            )
            conn.commit()

    def log_consolidation(
        self,
        memories_count: int,
        anchors_count: int,
        ewc_lambda: float,
        avg_task_loss: float,
        avg_ewc_loss: float,
        adapter_path: str
    ) -> int:
        """Records metadata for a completed sleep consolidation cycle."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO consolidation_logs (
                    memories_count, anchors_count, ewc_lambda,
                    avg_task_loss, avg_ewc_loss, adapter_path, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                memories_count,
                anchors_count,
                float(ewc_lambda),
                float(avg_task_loss),
                float(avg_ewc_loss),
                adapter_path,
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()
            return cursor.lastrowid

    def get_stats(self) -> Dict[str, Any]:
        """Provides summary metrics of memory and consolidation activity."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM interactions")
            total_interactions = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM interactions WHERE verified_reward = 1.0")
            verified_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM interactions WHERE verified_reward = 1.0 AND consolidated = 0")
            unconsolidated_verified = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM consolidation_logs")
            consolidation_cycles = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(surprise_score) FROM interactions WHERE verified_reward = 1.0")
            avg_surprise = cursor.fetchone()[0] or 0.0

            return {
                "total_interactions": total_interactions,
                "verified_count": verified_count,
                "unconsolidated_verified": unconsolidated_verified,
                "consolidation_cycles": consolidation_cycles,
                "average_verified_surprise": round(float(avg_surprise), 4),
            }

    def purge_all_for_reset(self) -> None:
        """Purges all interactions and consolidation logs for clean-slate resets."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM verified_sleep_batch")
            cursor.execute("DELETE FROM interactions")
            cursor.execute("DELETE FROM consolidation_logs")
            try:
                cursor.execute("DELETE FROM semantic_memory_index")
                cursor.execute("DELETE FROM dialogue_sessions")
            except Exception:
                pass
            conn.commit()
