"""Persist the GUI's selected durable memory namespace alongside its transcript."""
from contextlib import closing
import json
import sqlite3

from core.memory_scopes import LearningScope, as_scope


def _scope_key(implementation):
    from core.learning_implementation import get_learning_implementation
    mode = get_learning_implementation(implementation)
    if mode == "legacy":
        return None
    return "hybrid_active_scope" if mode == "hybrid" else "active_scope"


def load_memory_preference(database_path, *, implementation="managed"):
    key = _scope_key(implementation)
    if key is None:
        return LearningScope()
    with closing(sqlite3.connect(database_path)) as db:
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_preferences'").fetchone()
        if not exists:
            return LearningScope()
        row = db.execute("SELECT value FROM memory_preferences WHERE key=?", (key,)).fetchone()
    scope = as_scope(json.loads(row[0])) if row else LearningScope()
    if scope.kind not in {"model", "session", "project"}:
        raise ValueError("The saved GUI memory must be a model, session or project")
    return scope


def _scope_preference(scope, implementation):
    key = _scope_key(implementation)
    scope = as_scope(scope)
    if scope.kind not in {"model", "session", "project"}:
        raise ValueError("Only a durable user-selected GUI memory can be saved")
    if key is None:
        if scope.kind != "model":
            raise ValueError("Original learning uses model-wide memory")
        return []
    return [(key, json.dumps({"kind": scope.kind, "key": scope.key}))]


def _save_preferences(database_path, entries):
    with closing(sqlite3.connect(database_path)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS memory_preferences (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.executemany("INSERT OR REPLACE INTO memory_preferences VALUES (?, ?)", entries)


def save_memory_preference(database_path, scope, *, implementation="managed"):
    entries = _scope_preference(scope, implementation)
    if entries:
        _save_preferences(database_path, entries)


def load_learning_implementation(database_path, default="legacy"):
    from core.learning_implementation import get_learning_implementation
    with closing(sqlite3.connect(database_path)) as db:
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_preferences'").fetchone()
        row = db.execute("SELECT value FROM memory_preferences WHERE key='learning_implementation'").fetchone() if exists else None
    return get_learning_implementation(json.loads(row[0]) if row else default)


def save_learning_implementation(database_path, implementation, *, scope=None):
    from core.learning_implementation import get_learning_implementation
    implementation = get_learning_implementation(implementation)
    entries = [("learning_implementation", json.dumps(implementation))]
    if scope is not None:
        entries.extend(_scope_preference(scope, implementation))
    _save_preferences(database_path, entries)
