"""Model, request, session, project and document learning namespaces."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from uuid import uuid4


@dataclass(frozen=True)
class LearningScope:
    """Keys are identifiers, never paths. A pin selects a read-only document version."""

    kind: str = "model"
    key: str = ""
    generation: str = ""
    pin: str = ""

    def __post_init__(self):
        if self.kind not in {"model", "request", "session", "project", "document"}:
            raise ValueError("Unknown learning scope")
        if not isinstance(self.key, str) or len(self.key) > 2048 or bool(self.key) != (self.kind != "model"):
            raise ValueError("A named learning scope requires a key of at most 2048 characters")
        if self.pin or self.generation:
            if self.kind != "document" or not all(isinstance(key, str) and len(key) == 32
                    and all(c in "0123456789abcdef" for c in key) for key in (self.pin, self.generation)):
                raise ValueError("Invalid pinned document generation")

    @classmethod
    def request(cls):
        return cls("request", uuid4().hex)

    @property
    def scope_id(self):
        payload = json.dumps([self.kind, self.key], separators=(",", ":"), ensure_ascii=False).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def record(self):
        return {"kind": self.kind, "id": self.scope_id}


def as_scope(value):
    if value is None or value == "model":
        return LearningScope()
    if isinstance(value, LearningScope):
        return value
    if isinstance(value, dict):
        return LearningScope(**value)
    raise ValueError("Use LearningScope or a scope descriptor with kind and key")
