"""
Episodic SQLite memory manager.
Captures interactive traces, surprise scores, sandbox assertions,
and tracks consolidated parameter updates across sleep cycles.
"""

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class EpisodicMemoryDB:
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
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
                    test_cases TEXT,
                    consolidated INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
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
        test_cases: Optional[str] = None
    ) -> int:
        """Logs an interactive trace into episodic memory."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO interactions (
                    prompt, completion, raw_branches, verified_reward, 
                    surprise_score, mode, entropy, winning_branch, 
                    test_cases, consolidated, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """, (
                prompt,
                completion,
                json.dumps(raw_branches) if raw_branches else None,
                float(verified_reward),
                float(surprise_score),
                mode,
                float(entropy),
                int(winning_branch),
                test_cases,
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()
            return cursor.lastrowid

    def get_unconsolidated_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves unconsolidated interaction traces for UI table and inspector."""
        query = """
            SELECT id, prompt, completion, verified_reward, surprise_score, mode, entropy, consolidated, created_at as timestamp
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
            query += " AND consolidated = 0"
        query += " ORDER BY surprise_score DESC LIMIT ?"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

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
