"""Explicit online adaptation workflows on the already-loaded model.

Direct callers supply an AwakeOnlineConsolidator bound to the desired adapter
scope. Every update uses its normal validation, rollback, and checkpoint path.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import time
from collections import Counter
from contextlib import closing
from itertools import islice, permutations
from pathlib import Path

from core.learning_recall import (EmptyGeneration, PromptTooLong, StreamCursorBank, check_cancel, generate_answer,
                                  generation_prefix, grounded_question, object_response)
from core.learning_sources import iter_file_positions, iter_learning_files, source_snapshot
from core.online_consolidator import source_probe
from core.training_memory import process_rss_mb


def _integer(value, minimum, maximum, name):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _text(value, name, limit):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must contain 1–{limit} characters")
    return value


def _task_prompt(demos, query):
    examples = json.dumps(demos, ensure_ascii=False)
    return f"Follow the mapping in the examples. Return only the output.\nExamples: {examples}\nInput: {query}"


def _verifier(spec):
    if callable(spec):
        return spec
    if isinstance(spec, dict) and spec.get("kind") == "exact":
        expected = _text(spec.get("expected"), "training verifier target", 2048)
        return lambda answer: float(answer.strip() == expected.strip())
    if isinstance(spec, dict) and spec.get("kind") == "python_tests":
        from core.learning_code_verifier import PythonTaskVerifier
        return PythonTaskVerifier(spec)
    raise ValueError("Verifier mode requires a callable, exact target or Python function tests")


def _final_answer(text):
    final = re.findall(r"(?im)^final answer\s*:\s*(.+)$", text)
    return final[-1].strip() if final else text.strip()


def _rewarded(answers, mode, verifier, extract_answer):
    if mode == "consensus":
        if verifier is not None:
            raise ValueError("Consensus mode accepts no verifier or answer key")
        keys = [extract_answer(text) for text in answers]
        if any(not isinstance(key, str) or not key.strip() for key in keys):
            raise ValueError("A rollout has no extractable answer")
        winner, count = Counter(keys).most_common(1)[0]
        if count <= len(keys) // 2:
            return [], "no strict majority"
        return [float(k == winner) for k in keys], ""
    if mode != "verifier":
        raise ValueError("Reward mode must be verifier or consensus")
    score = _verifier(verifier)
    rewards = [float(score(text)) for text in answers]
    if not all(math.isfinite(r) and 0 <= r <= 1 for r in rewards):
        raise ValueError("Verifier rewards must be finite numbers between 0 and 1")
    return rewards, ""


class OnlineLearning:
    def __init__(self, service, generate=None):
        self.service = service
        self.backend = service.engine
        self.generate = generate

    def _answer(self, prompt, tokens, temperature, event):
        check_cancel(event)
        if self.generate is None:
            return generate_answer(self.backend, prompt, max_tokens=tokens, temperature=temperature,
                                   cancel_event=event)
        result = self.generate(prompt, max_tokens=tokens, temperature=temperature)
        check_cancel(event)
        return _text(result, "generated completion", 32768)

    def _require(self, method):
        from core.adaptation_capabilities import adaptation_capabilities
        capabilities = adaptation_capabilities(self.backend)
        if not capabilities.get(method):
            raise RuntimeError(capabilities.get("reasons", {}).get(method)
                               or f"Active backend does not support {method}")

    def run_request(self, request, *, cancel_event=None, progress=None):
        """Dispatch a JSON-shaped request, sharing the consolidator's update lock.

        A progress callback receives a dictionary after each consumed source
        chunk. Cooperative cancellation leaves finished chunks durable.
        """
        if not isinstance(request, dict):
            raise ValueError("An online learning request must be a JSON object")
        handlers = {"ingest": self.ingest, "few_shot": self.few_shot,
                    "query_adapt": self.query_adapt, "reinforce": self.reinforce,
                    "consolidate_candidates": self.consolidate_candidates}
        options = dict(request)
        method = options.pop("method", "")
        if method not in handlers:
            raise ValueError("Unknown online learning method: " + str(method))
        if progress is not None and method == "ingest":
            options["progress"] = progress
        with self.service._update_lock:
            check_cancel(cancel_event)
            started = time.monotonic()
            peak = [process_rss_mb()]
            done = threading.Event()

            def sample():
                while not done.wait(0.10):
                    peak[0] = max(peak[0], process_rss_mb())

            monitor = threading.Thread(target=sample, daemon=True)
            monitor.start()
            try:
                result = handlers[method](**options, cancel_event=cancel_event)
            finally:
                done.set()
                monitor.join()
            return {**result, "method": method, "wall_seconds": time.monotonic() - started,
                    "sampled_peak_rss_mb": max(peak[0], process_rss_mb())}

    def _source_rows(self, chunk, cue, strategy, question, budget, event, prefix=""):
        if not chunk.strip():
            return []
        if strategy == "stream":
            return [{"prompt": cue, "completion": chunk}]
        passage = prefix + chunk
        if strategy == "evidence":
            prompt = ("Identify evidence answering the question. Return JSON with spans, an array of "
                      "short exact quotations from this passage, or an empty array if irrelevant.\n"
                      f"Question: {question}\nPassage:\n{passage}")
            obj = object_response(self._answer(prompt, 240, 0, event))
            spans = obj.get("spans")
            if not isinstance(spans, list):
                return []
            rows = []
            used = 0
            for span in spans[:8]:
                if (isinstance(span, str) and span.strip() and span in passage
                        and passage.rfind(span) + len(span) > len(prefix)
                        and used + len(span) <= budget
                        and not any(row["completion"] == span for row in rows)):
                    rows.append({"prompt": cue, "completion": span})
                    used += len(span)
            return rows
        prompt = ("Elicit one factual question about this passage. Return a JSON object with question, "
                  "answer and evidence. Copy the answer and its supporting evidence verbatim from the "
                  f"passage. Keep the answer out of the question. Focus: {question}\nPassage:\n{passage}")
        obj = object_response(self._answer(prompt, 200, 0, event))
        qa = grounded_question(obj, passage)
        if (not qa or len(qa[1]) > budget or obj["evidence"].strip() == qa[1]
                or passage.rfind(obj["evidence"]) + len(obj["evidence"]) <= len(prefix)):
            return []
        cue, answer = qa
        # Select knowledge absent from the source-free student. The grounded
        # teacher target is an extractive answer, never an unverified synthesis.
        if self._answer(cue, 64, 0, event).strip() == answer:
            return []
        return [{"prompt": cue, "completion": answer}]

    def _bounded_selection(self, chunk, cue, strategy, question, budget, event, tail):
        try:
            return self._source_rows(chunk, cue, strategy, question, budget, event, prefix=tail)
        except PromptTooLong:
            if len(chunk) < 32:
                raise
            split = len(chunk) // 2
            left = self._bounded_selection(chunk[:split], cue, strategy, question, budget, event, "")
            used = sum(len(row["completion"]) for row in left)
            right = self._bounded_selection(chunk[split:], cue, strategy, question, budget - used, event, "")
            return left + [row for row in right if row not in left]

    def ingest(self, source, *, strategy="stream", question="", chunk_chars=2048,
               span_chars=768, max_chunks=None, steps=1, learning_rate=1e-4,
               cancel_event=None, progress=None):
        if strategy not in {"stream", "evidence", "elicitation"}:
            raise ValueError("Unknown streaming strategy")
        if strategy == "evidence":
            _text(question, "evidence question", 512)
        if not isinstance(question, str) or len(question) > 512:
            raise ValueError("Source focus exceeds 512 characters")
        _integer(chunk_chars, 32, 8192, "chunk_chars")
        _integer(span_chars, 1, 4096, "span_chars")
        _integer(steps, 1, 8, "steps")
        if max_chunks is not None:
            _integer(max_chunks, 1, 1000000, "max_chunks")
        root = Path(source).expanduser().resolve(strict=True)
        store = self.backend.learning_state
        excluded = (store.directory,)
        version, files = source_snapshot(root, cancel_event, exclude=excluded)
        descriptor = [1, str(root), version, strategy, question, chunk_chars, span_chars, steps, learning_rate]
        job = hashlib.sha256(json.dumps(descriptor).encode()).hexdigest()
        manifest = store.current_manifest(verify=False) or {}
        state = {"next": 0, "trained_chunks": 0, "selected_chars": 0, "source_chars": 0}
        memory = manifest.get("metadata", {}).get("memory", {})
        cursor_bank = StreamCursorBank(store.directory)
        state.update(cursor_bank.lookup(memory.get("stream_head"), job) or memory.get("streams", {}).get(job, {}))
        db_path = store.directory / "source-progress.sqlite3"
        with closing(sqlite3.connect(db_path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS progress (job TEXT, generation TEXT, state TEXT, PRIMARY KEY(job,generation))")
            generation = manifest.get("generation", "")
            record = db.execute("SELECT state FROM progress WHERE job=? AND generation=?", (job, generation)).fetchone()
            if record:
                cached = json.loads(record[0])
                if cached["next"] >= state["next"]:
                    state = cached
        start = state["next"]

        def save():
            # Also record consumed chunks that selected no trainable text. The
            # authenticated checkpoint cursor is authoritative for all updates.
            with closing(sqlite3.connect(db_path)) as db, db:
                db.execute("DELETE FROM progress WHERE job=? AND generation!=?", (job, generation))
                db.execute("INSERT OR REPLACE INTO progress VALUES (?, ?, ?)",
                           (job, generation, json.dumps(state)))

        def report(status):
            return {"status": status, "job_id": job, "files": files, "strategy": strategy,
                    "chunks_this_run": state["next"] - start,
                    **{k: v for k, v in state.items() if k not in {"tail", "seek"}}}

        if state.get("complete"):
            return report("completed")
        ordinal = start
        resume_file = state.get("file")
        for path in iter_learning_files(root, exclude=excluded):
            label = path.name if root.is_file() else path.relative_to(root).as_posix()
            if resume_file and label != resume_file:
                continue
            file_index = int(state["file_chunk"]) + 1 if resume_file else 0
            seek = state.get("seek") if resume_file else None
            tail = state.get("tail", "") if resume_file else ""
            resume_file = None
            for chunk, next_seek in iter_file_positions(path, chunk_chars, first=file_index, seek=seek):
                check_cancel(cancel_event)
                if max_chunks and state["next"] - start >= max_chunks:
                    return report("paused")
                ordinal += 1
                cue = f"Recall passage {file_index + 1} from {label}."
                if tail and strategy == "stream":
                    cue += "\nPrevious passage ends with:\n" + tail
                rows = self._bounded_selection(chunk, cue, strategy, question, span_chars, cancel_event, tail)
                count = sum(len(row["completion"]) for row in rows)
                pending = {"next": ordinal, "file": label, "file_chunk": file_index,
                           "seek": next_seek, "tail": chunk[-64:],
                           "source_chars": state["source_chars"] + len(chunk),
                           "trained_chunks": state["trained_chunks"] + bool(rows),
                           "selected_chars": state["selected_chars"] + count}
                if rows:
                    update = self.service.train_pairs(rows, evidence=[r["completion"] for r in rows],
                                             mode="source_" + strategy, steps=steps, learning_rate=learning_rate,
                                             training_options={"supervise_prompt": False}, stream_cursor=(job, pending),
                                             cancel_event=cancel_event)
                    generation = update["generation"]
                file_index += 1
                tail = pending["tail"]
                state = pending
                save()
                if progress:
                    progress(report("running"))
        if source_snapshot(root, cancel_event, exclude=excluded)[0] != version:
            raise RuntimeError("Source files changed during learning; resume the new source revision")
        state["complete"] = True
        save()
        return report("completed")

    def few_shot(self, demonstrations, query, *, votes=3, steps=1, learning_rate=1e-4,
                 max_tokens=64, cancel_event=None):
        _integer(votes, 1, 8, "votes")
        _integer(max_tokens, 1, 256, "max_tokens")
        _integer(steps, 1, 8, "steps")
        _text(query, "query", 2048)
        if not isinstance(demonstrations, list) or not 2 <= len(demonstrations) <= 12:
            raise ValueError("Provide 2–12 independent demonstrations")
        demos = [{"prompt": _text(d["prompt"], "demonstration input", 512),
                  "completion": _text(d["completion"], "demonstration output", 512)} for d in demonstrations]
        if len({d["prompt"].strip() for d in demos} | {query.strip()}) != len(demos) + 1:
            raise ValueError("Demonstration inputs and the unanswered query must be distinct")
        rows = []
        for i, held in enumerate(demos):
            rest = demos[:i] + demos[i + 1:]
            for ordering in (rest, list(reversed(rest))) if len(rest) > 1 else (rest,):
                rows.append({"prompt": _task_prompt(ordering, held["prompt"]), "completion": held["completion"]})
        result = self.service.train_pairs(rows, evidence=[d["completion"] for d in demos], mode="few_shot",
                                          validation_rows=demos, training_options={"supervise_prompt": False},
                                          steps=steps, learning_rate=learning_rate, cancel_event=cancel_event)
        answers = []
        for ordering in islice(permutations(demos), votes):
            answers.append(self._answer(_task_prompt(ordering, query), max_tokens, 0, cancel_event).strip())
        counts = Counter(answers).most_common()
        return {**result, "answer": counts[0][0], "votes": dict(counts), "training_rows": len(rows),
                "tied": len(counts) > 1 and counts[0][1] == counts[1][1]}

    def query_adapt(self, context, query, *, max_tokens=64, max_context_tokens=2048, max_window_tokens=64,
                    steps=2, learning_rate=1e-4, cancel_event=None):
        self._require("query_only")
        _text(context, "query adaptation context", 32768)
        _text(query, "query", 2048)
        _integer(max_tokens, 1, 256, "max_tokens")
        _integer(max_context_tokens, 3, 4096, "max_context_tokens")
        _integer(max_window_tokens, 2, 512, "max_window_tokens")
        _integer(steps, 1, 8, "steps")
        probe = source_probe(context)
        if probe is None:
            raise ValueError("Context provides no independent continuation probe")
        result = self.service.train_pairs([probe], evidence=[context], mode="query_adapt", steps=steps,
                                          training_options={"objective": "query_only", "context_text": context,
                                                            "max_context_tokens": max_context_tokens,
                                                            "max_window_tokens": max_window_tokens},
                                          learning_rate=learning_rate, cancel_event=cancel_event)
        question = f"Use the passage to answer the question.\nPassage:\n{context}\nQuestion: {query}"
        return {**result, "answer": self._answer(question, max_tokens, 0, cancel_event),
                "generation_includes_context": True}

    def _rollout_update(self, prompt, *, policy, mode, verifier, rollouts, rounds, max_tokens, steps,
                        learning_rate, extract_answer, cancel_event):
        _text(prompt, "rollout prompt", 2048)
        _integer(rollouts, 2, 12, "rollouts")
        _integer(rounds, 1, 8, "rounds")
        _integer(max_tokens, 1, 192, "max_tokens")
        _integer(steps, 1, 8, "steps")
        if mode == "verifier":
            verifier = _verifier(verifier)
        if policy:
            self._require("reward_policy")
        history = []
        best_sample, best_reward = None, None
        for _ in range(rounds):
            answers = []
            for _attempt in range(rollouts * 2):
                try:
                    answers.append(self._answer(prompt, max_tokens, 0.8, cancel_event))
                except EmptyGeneration:
                    continue
                if len(answers) == rollouts:
                    break
            if len(answers) != rollouts:
                history.append({"updated": False, "reason": "too few nonempty rollouts",
                                "usable_rollouts": len(answers)})
                break
            rewards, reason = _rewarded(answers, mode, verifier, extract_answer or _final_answer)
            if rewards and max(rewards) > 0 and (best_reward is None or max(rewards) > best_reward):
                best_reward = max(rewards)
                best_sample = answers[rewards.index(best_reward)]
            if not rewards or (policy and min(rewards) == max(rewards)) or max(rewards) == 0:
                history.append({"updated": False, "reason": reason or "no informative reward signal",
                                "rewards": rewards})
                break
            if policy:
                prefix = generation_prefix(self.backend, prompt)
                rows = [{"prompt": prompt, "completion": a, "rollout_prefix": prefix, "reward": r}
                        for a, r in zip(answers, rewards)]
                options = {"objective": "reward_policy", "max_window_tokens": 512}
            else:
                # Distillation takes a verified/consensus-selected trajectory.
                # Only the sampled text is a supervised target.
                selected = dict.fromkeys(a for a, r in zip(answers, rewards) if r == max(rewards))
                rows = [{"prompt": prompt, "completion": answer} for answer in selected]
                options = {"supervise_prompt": False}
            update = self.service.train_pairs(rows, mode="rollout_" + mode, training_options=options,
                                              steps=steps, learning_rate=learning_rate, cancel_event=cancel_event)
            history.append({"updated": True, "rewards": rewards, "update": update})
        measurements = getattr(verifier, "results", None) if mode == "verifier" else None
        return {"status": "completed" if any(h["updated"] for h in history) else "skipped",
                **({"verifier_measurements": measurements} if measurements is not None else {}),
                "reward_source": "model_consensus" if mode == "consensus" else "task_verifier",
                "rounds": history, "updates": sum(h["updated"] for h in history),
                "best_sample": best_sample, "best_reward": best_reward}

    def reinforce(self, prompt, *, mode="consensus", verifier=None, rollouts=4, rounds=1,
                  max_tokens=64, steps=1, learning_rate=1e-4, extract_answer=None, cancel_event=None):
        return self._rollout_update(prompt, policy=True, mode=mode, verifier=verifier, rollouts=rollouts,
                                    rounds=rounds, max_tokens=max_tokens, steps=steps, learning_rate=learning_rate,
                                    extract_answer=extract_answer, cancel_event=cancel_event)

    def consolidate_candidates(self, prompt, *, mode="consensus", verifier=None, rollouts=4, rounds=1,
                               max_tokens=64, steps=1, learning_rate=1e-4, extract_answer=None, cancel_event=None):
        return self._rollout_update(prompt, policy=False, mode=mode, verifier=verifier, rollouts=rollouts,
                                    rounds=rounds, max_tokens=max_tokens, steps=steps, learning_rate=learning_rate,
                                    extract_answer=extract_answer, cancel_event=cancel_event)
