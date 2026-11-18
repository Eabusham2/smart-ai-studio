"""Persist the GUI's selected durable memory namespace alongside its transcript."""
from contextlib import closing
import json
import sqlite3

from core.memory_scopes import LearningScope, as_scope


def load_memory_preference(database_path):
    with closing(sqlite3.connect(database_path)) as db:
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_preferences'").fetchone()
        if not exists:
            return LearningScope()
        row = db.execute("SELECT value FROM memory_preferences WHERE key='active_scope'").fetchone()
    scope = as_scope(json.loads(row[0])) if row else LearningScope()
    if scope.kind not in {"model", "session", "project"}:
        raise ValueError("The saved GUI memory must be a model, session or project")
    return scope


def save_memory_preference(database_path, scope):
    scope = as_scope(scope)
    if scope.kind not in {"model", "session", "project"}:
        raise ValueError("Only a durable user-selected GUI memory can be saved")
    with closing(sqlite3.connect(database_path)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS memory_preferences (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute("INSERT OR REPLACE INTO memory_preferences VALUES ('active_scope', ?)",
                   (json.dumps({"kind": scope.kind, "key": scope.key}),))
