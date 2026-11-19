"""Stable per-model episodic databases; never assign old shared rows by guesswork."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from core.learning_implementation import get_learning_implementation


def model_memory_identity(model_info, *, tab_id=None):
    """Identify a catalog source independently of its name or installation path.

    Publisher/revision metadata takes precedence over downloaded bundle paths.
    Local-only imports use their canonical selected path; the catalog ID is a
    fallback for entries with no source yet. Neither label renames nor moving
    the packaged application changes a published model's memory namespace.
    This is an episodic ownership key, not a weights authenticity certificate.
    """
    info = dict(model_info or {})
    source = next((str(info[key]).strip() for key in ("source_repo_id", "repo_id", "hf_repo_id")
                   if info.get(key)), "")
    if not source:
        local = info.get("strata_model_path") or info.get("model_path")
        if local:
            source = "local:" + str(Path(local).expanduser().resolve())
        elif tab_id:
            source = "catalog:" + str(tab_id)
        else:
            raise ValueError("Model memory requires a repository, local source, or catalog ID")
    revision = next((str(info[key]) for key in (
        "resolved_revision", "repo_revision", "source_revision", "revision", "commit_sha") if info.get(key)), "")
    artifact = next((str(info[key]) for key in (
        "gguf_file", "gguf_filename", "model_filename", "filename") if info.get(key)), "")
    local = info.get("strata_model_path") or info.get("model_path")
    if not artifact and local and str(local).lower().endswith(".gguf"):
        artifact = Path(local).name
    return {"schema": 1, "source": source, "revision": revision, "artifact": artifact}


def model_episodic_db_path(base_database_path, model_info, *, implementation="hybrid", tab_id=None):
    """Return a path only; do not copy, open, change or infer ownership of legacy data."""
    mode = get_learning_implementation(implementation)
    identity = model_memory_identity(model_info, tab_id=tab_id)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    base = Path(base_database_path).expanduser().resolve()
    return str(base.parent / (base.stem + "_models") / digest / (mode + ".db"))
