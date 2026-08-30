import sqlite3
import time
from typing import Dict, Any, List

class EpisodicMemoryDB:
    def __init__(self, db_path: str = "data/memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                prompt TEXT,
                completion TEXT,
                verified_reward REAL,
                surprise_score REAL,
                mode TEXT,
                consolidated INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()

    def log_interaction(self, prompt: str, completion: str, raw_branches: List[str] = None,
                        verified_reward: float = 1.0, surprise_score: float = 0.5, mode: str = "live"):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO interactions (timestamp, prompt, completion, verified_reward, surprise_score, mode, consolidated)
            VALUES (?, ?, ?, ?, ?, ?, 0)
        """, (time.strftime("%Y-%m-%dT%H:%M:%SZ"), prompt, completion, verified_reward, surprise_score, mode))
        conn.commit()
        conn.close()

    def fetch_unconsolidated(self, limit: int = 15) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, prompt, completion, verified_reward, surprise_score FROM interactions WHERE consolidated = 0 ORDER BY surprise_score DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [{"id": r[0], "prompt": r[1], "completion": r[2], "reward": r[3], "surprise": r[4]} for r in rows]
