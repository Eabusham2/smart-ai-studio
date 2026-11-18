"""Answer-blind behavioral probes for the selected serving model."""
from __future__ import annotations

import hashlib
import inspect
import json
import sqlite3
from contextlib import closing
from itertools import zip_longest

from core.context_memory import completed_groups
from core.learning_data import encode_text, format_training_row


def check_cancel(event):
    if event is not None and event.is_set():
        raise InterruptedError("Learning cancelled before checkpoint commit")


class EmptyGeneration(RuntimeError):
    """EOS or a completion with no decodable text."""


class PromptTooLong(ValueError):
    """The complete prompt cannot be sent within the serving context."""


def _vlm_formatter(backend):
    if getattr(backend, "_vlm_runtime", None):
        return backend._prepare_vlm_prompt
    runtime = getattr(backend, "runtime", "")
    if runtime.startswith("mlx") or runtime in ("jang_vlm", "jang-vlm", "repo_mlx_vlm"):
        return getattr(backend, "_mlx_prompt", None)
    return None


def generation_prefix(backend, prompt):
    """Text conditioning used by generation and reward scoring for this route."""
    formatter = _vlm_formatter(backend)
    if formatter:
        return formatter(prompt, 0)
    _, prefix = format_training_row(backend.tokenizer, {"prompt": prompt, "completion": "x"})
    return prefix


def generation_room(backend, prefix, requested):
    tokenizer = backend.tokenizer
    model = getattr(backend, "model", None)
    config = getattr(model, "config", None)
    limits = []
    for owner in (getattr(config, "text_config", None), config, getattr(model, "args", None),
                  tokenizer, backend, model):
        for name in ("max_position_embeddings", "n_positions", "max_seq_len", "context_length",
                     "model_max_length", "n_ctx", "context_budget_tokens", "_learning_context_budget_tokens"):
            value = getattr(owner, name, None)
            value = value() if callable(value) else value
            if type(value) is int and 1 < value <= 10000000:
                limits.append(value)
    if not limits:
        return requested
    if callable(getattr(tokenizer, "encode", None)):
        used = len(tokenizer.encode(prefix))
    elif callable(getattr(tokenizer, "tokenize", None)):
        used = len(tokenizer.tokenize(prefix.encode("utf-8")))
    else:
        used = backend.count_tokens([{"role": "user", "content": prefix}])
    room = min(limits) - used
    if room < 1:
        raise PromptTooLong("The complete learning prompt exceeds the active model context")
    return min(requested, room)


def generate_answer(backend, prompt, *, max_tokens, temperature, cancel_event=None):
    check_cancel(cancel_event)
    # Select completion-only output. Some nonstreaming controller routes return
    # the prompt together with the answer, which is invalid as a recall witness.
    generate = getattr(backend, "stream_generate_tokens", None)
    if not callable(generate):
        raise RuntimeError("The serving backend has no completion-only generation route")
    prefix = generation_prefix(backend, prompt)
    kwargs = {"max_tokens": generation_room(backend, prefix, max_tokens), "temperature": temperature}
    if "cancelled" in inspect.signature(generate).parameters and cancel_event is not None:
        kwargs["cancelled"] = cancel_event.is_set
    pieces = generate(prompt if _vlm_formatter(backend) else prefix, **kwargs)
    try:
        tokens = []
        for piece in pieces:
            check_cancel(cancel_event)
            tokens.append(piece)
        answer = "".join(tokens)
    finally:
        close = getattr(pieces, "close", None)
        if close:
            close()
    check_cancel(cancel_event)
    if not answer.strip():
        raise EmptyGeneration("The serving model generated no decodable text")
    if answer.lstrip().startswith(("⚠", "❌")):
        raise RuntimeError("The serving model returned no usable completion")
    return answer


def object_response(answer):
    text = answer.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.partition("\n")[2].rsplit("```", 1)[0]
    try:
        obj = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return obj if isinstance(obj, dict) else {}


def grounded_question(obj, source, *, max_question_chars=256, max_answer_chars=192, max_evidence_chars=1024):
    """Accept only short answer-blind questions with contiguous quoted evidence."""
    question, answer, evidence = (obj.get(k) for k in ("question", "answer", "evidence"))
    if not all(isinstance(x, str) and x.strip() for x in (question, answer, evidence)):
        return None
    question, answer = question.strip(), answer.strip()
    if (any(limit is not None and len(text) > limit for text, limit in (
                (question, max_question_chars), (answer, max_answer_chars), (evidence, max_evidence_chars)))
            or answer not in evidence or evidence not in source
            or answer.casefold() in question.casefold()):
        return None
    return question, answer


def _digest(text):
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


_QUESTION_PREFIX = (
    "Form one short recall question about this exchange. Reply with a JSON object "
    "with question, answer, evidence. Copy answer and evidence exactly from a single turn. "
    "Keep the answer out of the question.\nExchange:\n"
)


def _required_tokens(backend, answer):
    # Allow boundary tokenization differences and a terminal token.
    encode = getattr(backend.tokenizer, "encode", None)
    return (len(encode(answer)) if callable(encode) else len(encode_text(backend.tokenizer, answer))) + 4


def _fits(backend, cue, requested):
    try:
        return generation_room(backend, generation_prefix(backend, cue), requested) >= requested
    except PromptTooLong:
        return False


def _question(backend, source, texts, event):
    response = generate_answer(backend, _QUESTION_PREFIX + source, max_tokens=144,
                               temperature=0, cancel_event=event)
    obj = object_response(response)
    return next((qa for t in texts if (qa := grounded_question(obj, t))), None)


def needs_segmented_recall(backend, turns):
    for group in completed_groups(turns):
        exchange = [turns[i] for i in group]
        # This is a working batch size. Larger inputs go through resumable
        # bounded updates, even when the model has a huge physical window.
        if sum(len(t["content"]) for t in exchange) > 1024:
            return True
        source = json.dumps(exchange, ensure_ascii=False)
        if not _fits(backend, _QUESTION_PREFIX + source, 32):
            return True
        for t in exchange:
            if t["content"] != t["content"].strip():
                return True
            if not _fits(backend, _QUESTION_PREFIX, _required_tokens(backend, t["content"])):
                return True
    return False


def retirement_probes(backend, turns, cancel_event):
    """Compact exchanges keep their existing natural-language recall questions."""
    checks = []
    groups = completed_groups(turns)
    if not groups or sum(map(len, groups)) != len(turns):
        return []
    for group in groups:
        exchange = [turns[i] for i in group]
        verified = _question(backend, json.dumps(exchange, ensure_ascii=False),
                             [t["content"] for t in exchange], cancel_event)
        if not verified:
            return []
        cue, answer = verified
        checks.append(("Reply with only the earlier answer. " + cue, _digest(answer),
                       _required_tokens(backend, answer)))
        for sequence, turn in enumerate(exchange):
            if turn["content"].strip().casefold() in cue.casefold():
                return []
            prompt = (f"Quote the entire original {turn['role']} message (turn {sequence + 1}) "
                      f"of the earlier exchange, exactly, with no commentary. Recall question: {cue}")
            checks.append((prompt, hashlib.sha256(turn["content"].encode()).hexdigest(),
                           _required_tokens(backend, turn["content"]), "exact_text"))
    return checks


class SegmentedRecall:
    """Reiterable, exactly covering passage addresses; each answer fits the live route."""

    def __init__(self, backend, turns, event):
        self.backend = backend
        self.turns = turns
        self.event = event
        self.groups = ([(0,)] if len(turns) == 1 and turns[0].get("role") == "document"
                       else completed_groups(turns))
        if not self.groups or sum(map(len, self.groups)) != len(turns):
            raise ValueError("Only complete ordinary exchanges or explicit document sources can be segmented")

    def passages(self, start_turn=0, start_char=0):
        for index in range(start_turn, len(self.turns)):
            turn = self.turns[index]
            text = turn["content"]
            offset = start_char if index == start_turn else 0
            while offset < len(text):
                check_cancel(self.event)
                cue = f"Recall {turn['role']} {turn.turn_id} at character {offset}. Return the exact passage as a JSON string."
                room = generation_room(self.backend, generation_prefix(self.backend, cue), 256)
                width = min(768, len(text) - offset)
                while True:
                    part = text[offset:offset + width]
                    answer = json.dumps(part, ensure_ascii=False)
                    needed = _required_tokens(self.backend, answer)
                    if needed <= room:
                        break
                    if width == 1:
                        raise PromptTooLong("The model context cannot fit one addressed recall passage")
                    width = max(1, width // 2)
                yield {"prompt": cue, "completion": answer}, (cue, hashlib.sha256(part.encode()).hexdigest(),
                                                              needed, "json_string"), (index, offset + width)
                offset += width

    def semantic_checks(self):
        """A separate grounded, answer-blind question must also work for each exchange."""
        for group in self.groups:
            verified = None
            for i in group:
                text = self.turns[i]["content"]
                offset = 0
                while offset < len(text):
                    check_cancel(self.event)
                    width = min(768, len(text) - offset)
                    while width > 1 and not _fits(self.backend, _QUESTION_PREFIX + text[offset:offset + width], 24):
                        width //= 2
                    excerpt = text[offset:offset + width]
                    if not _fits(self.backend, _QUESTION_PREFIX + excerpt, 24):
                        raise PromptTooLong("The serving context cannot fit the grounded recall question")
                    verified = _question(self.backend, excerpt, [excerpt], self.event)
                    if verified:
                        break
                    offset += width
                if verified:
                    break
            if not verified:
                raise ValueError("No grounded recall question for this exchange")
            question, answer = verified
            yield ("Reply with only the earlier answer. " + question, _digest(answer),
                   _required_tokens(self.backend, answer))

    def checks(self):
        yield from self.semantic_checks()
        for _, probe, _ in self.passages():
            yield probe


def passes_recall(backend, checks, cancel_event):
    for cue, digest, limit, *kind in checks:
        answer = generate_answer(backend, cue, max_tokens=limit, temperature=0, cancel_event=cancel_event)
        if kind == ["exact_text"]:
            if hashlib.sha256(answer.encode()).hexdigest() != digest:
                return False
        elif kind:
            if kind != ["json_string"]:
                raise ValueError("Unknown recall witness format")
            try:
                part = json.loads(answer)
            except ValueError:
                return False
            if not isinstance(part, str) or hashlib.sha256(part.encode()).hexdigest() != digest:
                return False
        elif _digest(answer) != digest:
            return False
    return True


class _StoredChecks:
    def __init__(self, path, heads):
        self.path = path
        self.heads = heads

    def __iter__(self):
        with closing(sqlite3.connect(self.path)) as db:
            for head in self.heads:
                legacy, = db.execute("SELECT checks FROM probe_sets WHERE head=?", (head,)).fetchone()
                yield from json.loads(legacy)
                for row, in db.execute("SELECT probe FROM probes WHERE head=? ORDER BY ordinal", (head,)):
                    yield json.loads(row)

    def __eq__(self, other):
        if not isinstance(other, (list, tuple, _StoredChecks)):
            return NotImplemented
        return all(a == b for a, b in zip_longest(self, other, fillvalue=object()))

    def __len__(self):
        with closing(sqlite3.connect(self.path)) as db:
            return sum(db.execute("""SELECT json_array_length(checks) +
                       (SELECT count(*) FROM probes p WHERE p.head=s.head)
                       FROM probe_sets s WHERE head=?""", (head,)).fetchone()[0] for head in self.heads)


class RecallBank:
    """Immutable probe sets attached to the exact retirement-receipt lineage."""

    def __init__(self, directory):
        self.path = directory / "recall.sqlite3"
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS probe_sets
                       (head TEXT PRIMARY KEY, parent TEXT, depth INTEGER, checks TEXT)""")
            db.execute("""CREATE TABLE IF NOT EXISTS probes
                       (head TEXT, ordinal INTEGER, probe TEXT, PRIMARY KEY(head, ordinal))""")

    def stage(self, head, parent, checks):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""INSERT INTO probe_sets VALUES (?, ?,
                       COALESCE((SELECT depth FROM probe_sets WHERE head=?), 0) + 1, '[]')""",
                       (head, parent, parent))
            db.executemany("INSERT INTO probes VALUES (?,?,?)",
                           ((head, i, json.dumps(probe)) for i, probe in enumerate(checks)))
        return _StoredChecks(self.path, [head])

    def discard(self, head):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("DELETE FROM probes WHERE head=?", (head,))
            db.execute("DELETE FROM probe_sets WHERE head=?", (head,))

    def sample(self, head, generation):
        if not head:
            return []
        with closing(sqlite3.connect(self.path)) as db:
            depth = db.execute("SELECT depth FROM probe_sets WHERE head=?", (head,)).fetchone()
            if not depth:
                raise RuntimeError("The current retirement receipt has no behavioral recall probes")
            rotating = 1 + int(_digest(str(generation))[:16], 16) % depth[0]
            rows = db.execute("""WITH RECURSIVE chain(head, parent, depth) AS (
                              SELECT head,parent,depth FROM probe_sets WHERE head=? UNION ALL
                              SELECT p.head,p.parent,p.depth FROM probe_sets p JOIN chain c ON p.head=c.parent)
                              SELECT head FROM chain WHERE depth IN (1, ?, ?)""", (head, rotating, depth[0]))
            return _StoredChecks(self.path, [row[0] for row in rows])


class ReplayBank:
    """Sample historical source probes without growing per-checkpoint RAM or metadata.

    Each node stores one bounded batch and links to ancestors 1, 2, 4, ... steps
    back. The committed checkpoint selects a lineage, including after rollback.
    """
    filename = "replay.sqlite3"

    def __init__(self, directory):
        self.path = directory / self.filename
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS source_batches
                       (head TEXT PRIMARY KEY,parent TEXT,depth INTEGER,links TEXT,rows TEXT)""")

    @staticmethod
    def _node(db, head):
        node = db.execute("SELECT depth,links,rows FROM source_batches WHERE head=?", (head,)).fetchone()
        if not node:
            raise RuntimeError("Source replay history is missing for the current checkpoint")
        return node[0], json.loads(node[1]), json.loads(node[2])

    def stage(self, head, parent, rows):
        with closing(sqlite3.connect(self.path)) as db, db:
            depth = 1
            links = []
            if parent:
                depth = self._node(db, parent)[0] + 1
                links.append(parent)
                while True:
                    ancestor_links = self._node(db, links[-1])[1]
                    level = len(links) - 1
                    if level >= len(ancestor_links):
                        break
                    links.append(ancestor_links[level])
            db.execute("INSERT INTO source_batches VALUES (?,?,?,?,?)",
                       (head, parent, depth, json.dumps(links), json.dumps(rows)))

    def discard(self, head):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("DELETE FROM source_batches WHERE head=?", (head,))

    def sample(self, head, generation):
        if not head:
            return []
        with closing(sqlite3.connect(self.path)) as db:
            depth, _, _ = self._node(db, head)
            rotating = int(_digest(str(generation))[:16], 16) % depth
            rows = []
            for distance in dict.fromkeys([depth - 1, rotating, 0]):
                node = self._node(db, head)
                while distance:
                    level = distance.bit_length() - 1
                    node = self._node(db, node[1][level])
                    distance -= 1 << level
                rows.extend(row for row in node[2] if row not in rows)
            return rows

    def inherit(self, source_directory, head, cancel_event=None):
        origin = source_directory / self.filename
        if not origin.is_file():
            raise RuntimeError("The source checkpoint has no replay bank")
        check_cancel(cancel_event)
        with closing(sqlite3.connect(self.path)) as db, db:
            if cancel_event is not None:
                db.set_progress_handler(cancel_event.is_set, 1000)
            db.execute("ATTACH DATABASE ? AS origin", (str(origin),))
            try:
                db.execute("""WITH RECURSIVE chain(head,parent) AS (
                           SELECT head,parent FROM origin.source_batches WHERE head=?
                           UNION ALL SELECT b.head,b.parent FROM origin.source_batches b
                           JOIN chain c ON b.head=c.parent)
                           INSERT OR IGNORE INTO main.source_batches
                           SELECT b.* FROM origin.source_batches b JOIN chain c ON b.head=c.head""", (head,))
            except sqlite3.OperationalError:
                check_cancel(cancel_event)
                raise
            self._node(db, head)
            check_cancel(cancel_event)


class StreamCursorBank(ReplayBank):
    """One manifest head authenticates branched cursor history for any number of sources."""
    filename = "source-cursors.sqlite3"

    def __init__(self, directory):
        super().__init__(directory)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""CREATE INDEX IF NOT EXISTS source_job
                       ON source_batches(json_extract(rows, '$[0].job'), depth DESC)""")

    @staticmethod
    def _key(parent, job, cursor):
        record = json.dumps([parent, job, cursor], sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(record.encode()).hexdigest()

    def record(self, parent, job, cursor):
        head = self._key(parent, job, cursor)
        try:
            self.stage(head, parent, [{"job": job, "cursor": cursor}])
        except sqlite3.IntegrityError:
            # Concurrent identical candidates and crash retries share an
            # immutable record. Verify it before incorporating the pointer.
            if self.lookup(head, job) != cursor:
                raise
        return head

    def lookup(self, head, job):
        if not head:
            return {}
        with closing(sqlite3.connect(self.path)) as db:
            depth, _, _ = self._node(db, head)
            rows = db.execute("""SELECT head,parent,depth,rows FROM source_batches
                              WHERE json_extract(rows, '$[0].job')=? AND depth<=?
                              ORDER BY depth DESC""", (job, depth))
            for candidate, parent, candidate_depth, raw in rows:
                ancestor = head
                distance = depth - candidate_depth
                while distance:
                    level = distance.bit_length() - 1
                    ancestor = self._node(db, ancestor)[1][level]
                    distance -= 1 << level
                if ancestor == candidate:
                    cursor = json.loads(raw)[0]["cursor"]
                    if self._key(parent, job, cursor) != candidate:
                        raise RuntimeError("Stored source cursor failed its checkpoint digest")
                    return cursor
            return {}
