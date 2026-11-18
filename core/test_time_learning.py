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
from itertools import permutations
from pathlib import Path

from core.learning_recall import (EmptyGeneration, PromptTooLong, StreamCursorBank, check_cancel, generate_answer,
                                  generation_prefix, generation_room, grounded_question, object_response)
from core.discovery_search import DiscoveryArchive
from core import arc_augmentation as arc
from core.candidate_trajectories import candidate_trajectories, joint_rows
from core.policy_reference import policy_reference
from core.learning_sources import iter_attachment_positions, iter_file_positions, iter_learning_files, source_snapshot
from core.online_consolidator import LearningValidationError, source_probe
from core.training_memory import process_rss_mb


def _integer(value, minimum, name):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _number(value, name, *, allow_zero=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (value < 0 if allow_zero else value <= 0)):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must contain nonempty text")
    return value


def _causal_options(max_window_tokens):
    options = {"supervise_prompt": False}
    if max_window_tokens is not None:
        options["max_window_tokens"] = _integer(max_window_tokens, 2, "max_window_tokens")
    return options


def _task_prompt(demos, query, *, grids=False):
    if grids:
        examples = [{"input": json.loads(d["prompt"]), "output": json.loads(d["completion"])} for d in demos]
        return ("Follow the example grid mapping. Return only the output grid as a JSON array of rows."
                f"\nExamples: {json.dumps(examples, separators=(',', ':'))}\nInput: {query}")
    examples = json.dumps(demos, ensure_ascii=False)
    return f"Follow the mapping in the examples. Return only the output.\nExamples: {examples}\nInput: {query}"


def _verifier(spec):
    if callable(spec):
        return spec
    if isinstance(spec, dict) and spec.get("kind") == "exact":
        expected = _text(spec.get("expected"), "training verifier target")
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
    def __init__(self, service, generate=None, *, attachment_errors=None):
        self.service = service
        self.backend = service.engine
        self.generate = generate
        self.attachment_errors = attachment_errors

    def _answer(self, prompt, tokens, temperature, event):
        check_cancel(event)
        if self.generate is None:
            return generate_answer(self.backend, prompt, max_tokens=tokens, temperature=temperature,
                                   cancel_event=event)
        result = self.generate(prompt, max_tokens=tokens, temperature=temperature)
        check_cancel(event)
        if isinstance(result, str) and not result.strip():
            raise EmptyGeneration("The generator returned an empty completion")
        return _text(result, "generated completion")

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
                    "query_adapt": self.query_adapt, "reinforce": self.reinforce, "discover": self.discover,
                    "consolidate_candidates": self.consolidate_candidates}
        options = dict(request)
        method = options.pop("method", "")
        if method not in handlers:
            raise ValueError("Unknown online learning method: " + str(method))
        if progress is not None and method in {"ingest", "discover"}:
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

    def _source_rows(self, chunk, cue, strategy, question, budget, event, prefix, selection_tokens, answer_tokens):
        if not chunk.strip():
            return []
        if strategy == "stream":
            return [{"prompt": cue, "completion": chunk}]
        passage = prefix + chunk
        if strategy == "evidence":
            prompt = ("Identify evidence answering the question. Return JSON with spans, an array of "
                      "short exact quotations from this passage, or an empty array if irrelevant.\n"
                      f"Question: {question}\nPassage:\n{passage}")
            obj = object_response(self._answer(prompt, selection_tokens, 0, event))
            spans = obj.get("spans")
            if not isinstance(spans, list):
                return []
            rows = []
            used = 0
            for span in spans:
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
        obj = object_response(self._answer(prompt, selection_tokens, 0, event))
        qa = grounded_question(obj, passage, max_question_chars=None, max_answer_chars=budget, max_evidence_chars=None)
        if (not qa or len(qa[1]) > budget or obj["evidence"].strip() == qa[1]
                or passage.rfind(obj["evidence"]) + len(obj["evidence"]) <= len(prefix)):
            return []
        cue, answer = qa
        # Select knowledge absent from the source-free student. The grounded
        # teacher target is an extractive answer, never an unverified synthesis.
        if self._answer(cue, answer_tokens, 0, event).strip() == answer:
            return []
        return [{"prompt": cue, "completion": answer}]

    def _bounded_selection(self, chunk, cue, strategy, question, budget, event, tail,
                           selection_tokens, answer_tokens, overlap_chars):
        try:
            return self._source_rows(chunk, cue, strategy, question, budget, event, tail, selection_tokens, answer_tokens)
        except PromptTooLong:
            if tail:
                remaining = len(tail) // 2
                return self._bounded_selection(chunk, cue, strategy, question, budget, event,
                    tail[-remaining:] if remaining else "", selection_tokens, answer_tokens, overlap_chars)
            if len(chunk) < 2:
                raise
            split = len(chunk) // 2
            left = self._bounded_selection(chunk[:split], cue, strategy, question, budget, event, "",
                                           selection_tokens, answer_tokens, overlap_chars)
            used = sum(len(row["completion"]) for row in left)
            if used >= budget:
                return left
            previous = chunk[max(0, split - overlap_chars):split]
            right = self._bounded_selection(chunk[split:], cue, strategy, question, budget - used, event,
                                            previous, selection_tokens, answer_tokens, overlap_chars)
            return left + [row for row in right if row not in left]

    def ingest(self, source, *, strategy="stream", question="", chunk_chars=2048,
               span_chars=768, max_chunks=None, steps=1, learning_rate=1e-4, selection_tokens=None, answer_tokens=64,
               overlap_chars=64, max_window_tokens=None, cancel_event=None, progress=None):
        training_options = _causal_options(max_window_tokens)
        if strategy not in {"stream", "evidence", "elicitation"}:
            raise ValueError("Unknown streaming strategy")
        selection_tokens = (240 if strategy == "evidence" else 200) if selection_tokens is None else selection_tokens
        _integer(selection_tokens, 1, "selection_tokens")
        _integer(answer_tokens, 1, "answer_tokens")
        if strategy == "evidence":
            _text(question, "evidence question")
        if not isinstance(question, str):
            raise ValueError("Source focus must be a string")
        _integer(chunk_chars, 1, "chunk_chars")
        _integer(overlap_chars, 0, "overlap_chars")
        _integer(span_chars, 1, "span_chars")
        _integer(steps, 1, "steps")
        if max_chunks is not None:
            _integer(max_chunks, 1, "max_chunks")
        root = Path(source).expanduser().resolve(strict=True)
        store = self.backend.learning_state
        excluded = (store.directory,)
        version, files = source_snapshot(root, cancel_event, exclude=excluded)
        descriptor = [1, str(root), version, strategy, question, chunk_chars, span_chars, steps, learning_rate]
        if strategy != "stream" and (selection_tokens != (240 if strategy == "evidence" else 200) or answer_tokens != 64):
            descriptor.extend((selection_tokens, answer_tokens))
        if max_window_tokens is not None:
            descriptor.append({"max_window_tokens": max_window_tokens})
        if overlap_chars != 64:
            descriptor.append({"overlap_chars": overlap_chars})
        strict_job = hashlib.sha256(json.dumps(descriptor).encode()).hexdigest()
        job = strict_job
        if self.attachment_errors is not None:
            descriptor.append("legacy-attachment-reading-v1")
            job = hashlib.sha256(json.dumps(descriptor).encode()).hexdigest()
        manifest = store.current_manifest(verify=False) or {}
        state = {"next": 0, "trained_chunks": 0, "selected_chars": 0, "source_chars": 0}
        memory = manifest.get("metadata", {}).get("memory", {})
        cursor_bank = StreamCursorBank(store.directory)
        # A previously strict-read prefix is also valid for a tolerant Learn attachment.
        # Strict ingestion must never inherit progress past a skipped or repaired document.
        eligible_jobs = [strict_job, job] if job != strict_job else [job]
        for prior_job in eligible_jobs:
            prior = cursor_bank.lookup(memory.get("stream_head"), prior_job) or memory.get("streams", {}).get(prior_job, {})
            if prior.get("next", -1) >= state["next"]:
                state.update(prior)
        db_path = store.directory / "source-progress.sqlite3"
        with closing(sqlite3.connect(db_path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS progress (job TEXT, generation TEXT, state TEXT, PRIMARY KEY(job,generation))")
            generation = manifest.get("generation", "")
            for prior_job in eligible_jobs:
                record = db.execute("SELECT state FROM progress WHERE job=? AND generation=?", (prior_job, generation)).fetchone()
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
            chunks = (iter_file_positions(path, chunk_chars, first=file_index, seek=seek)
                      if self.attachment_errors is None else iter_attachment_positions(
                          path, chunk_chars, self.attachment_errors, first=file_index, seek=seek))
            for chunk, next_seek in chunks:
                check_cancel(cancel_event)
                if max_chunks and state["next"] - start >= max_chunks:
                    return report("paused")
                ordinal += 1
                cue = f"Recall passage {file_index + 1} from {label}."
                if tail and strategy == "stream":
                    cue += "\nPrevious passage ends with:\n" + tail
                rows = self._bounded_selection(chunk, cue, strategy, question, span_chars, cancel_event, tail,
                                               selection_tokens, answer_tokens, overlap_chars)
                count = sum(len(row["completion"]) for row in rows)
                pending = {"next": ordinal, "file": label, "file_chunk": file_index,
                           "seek": next_seek, "tail": chunk[-overlap_chars:] if overlap_chars else "",
                           "source_chars": state["source_chars"] + len(chunk),
                           "trained_chunks": state["trained_chunks"] + bool(rows),
                           "selected_chars": state["selected_chars"] + count}
                if rows:
                    update = self.service.train_pairs(rows, evidence=[r["completion"] for r in rows],
                                             mode="source_" + strategy, steps=steps, learning_rate=learning_rate,
                                             training_options=training_options, stream_cursor=(job, pending),
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
                 max_tokens=64, arc_augment="auto", max_window_tokens=None, cancel_event=None):
        training_options = _causal_options(max_window_tokens)
        _integer(votes, 1, "votes")
        _integer(max_tokens, 1, "max_tokens")
        _integer(steps, 1, "steps")
        if not isinstance(demonstrations, list) or len(demonstrations) < 2:
            raise ValueError("Provide at least two independent demonstrations")
        if any(not isinstance(d, dict) for d in demonstrations):
            raise ValueError("Each demonstration must have an input and output")
        if arc_augment != "auto" and type(arc_augment) is not bool:
            raise ValueError("arc_augment must be auto, true or false")
        raw = [(d.get("prompt", d.get("input")), d.get("completion", d.get("output"))) for d in demonstrations]
        grids = [(arc.grid(x), arc.grid(y)) for x, y in raw]
        grid_query = arc.grid(query)
        is_grid = grid_query is not None and all(x is not None and y is not None for x, y in grids)
        if arc_augment is True and not is_grid:
            raise ValueError("ARC transformations require rectangular grids of colors 0–9")
        if is_grid and arc_augment is False and isinstance(query, str) and all(
                isinstance(x, str) and isinstance(y, str) for x, y in raw):
            is_grid = False
        if is_grid:
            variants = arc.VARIANTS if arc_augment is not False else arc.VARIANTS[:1]
            views = []
            for variant in variants:
                def render(g):
                    return arc.grid_text(arc.transform(g, variant))
                views.append(([{"prompt": render(x), "completion": render(y)} for x, y in grids],
                              render(grid_query), variant))
        else:
            demos = [{"prompt": _text(x, "demonstration input"), "completion": _text(y, "demonstration output")}
                     for x, y in raw]
            views = [(demos, _text(query, "query"), None)]
        demos, original, _ = views[0]
        if len({d["prompt"].strip() for d in demos} | {original.strip()}) != len(demos) + 1:
            raise ValueError("Demonstration inputs and the unanswered query must be distinct")
        rows = []
        seen = set()
        for examples, _, variant in views:
            for i, held in enumerate(examples):
                check_cancel(cancel_event)
                rest = examples[:i] + examples[i + 1:]
                for order in (rest, rest[::-1]) if len(rest) > 1 else (rest,):
                    pair = (_task_prompt(order, held["prompt"], grids=variant is not None), held["completion"])
                    if pair not in seen:
                        rows.append({"prompt": pair[0], "completion": pair[1]})
                        seen.add(pair)
        result = self.service.train_pairs(rows, evidence=[d["completion"] for d in demos], mode="few_shot",
                                          validation_rows=demos, training_options=training_options,
                                          steps=steps, learning_rate=learning_rate, cancel_event=cancel_event)
        answers = []
        attempts = invalid = 0
        for order in permutations(range(len(demos))):
            for examples, unanswered, variant in views:
                question = _task_prompt([examples[i] for i in order], unanswered, grids=variant is not None)
                answer = self._answer(question, max_tokens, 0, cancel_event).strip()
                attempts += 1
                if variant is None:
                    answers.append(answer)
                else:
                    predicted = arc.grid(answer)
                    if predicted is None:
                        invalid += 1
                    else:
                        answers.append(arc.grid_text(arc.transform(predicted, variant, inverse=True)))
                if attempts == votes:
                    break
            if attempts == votes:
                break
        counts = Counter(answers).most_common()
        return {**result, "answer": counts[0][0] if counts else None, "votes": dict(counts),
                "training_rows": len(rows), "invalid_votes": invalid, "arc_views": len(views) if is_grid else 0,
                "tied": len(counts) > 1 and counts[0][1] == counts[1][1]}

    def query_adapt(self, context, query, *, max_tokens=64, max_context_tokens=None, max_window_tokens=64,
                    steps=2, learning_rate=1e-4, cancel_event=None):
        self._require("query_only")
        _text(context, "query adaptation context")
        _text(query, "query")
        _integer(max_tokens, 1, "max_tokens")
        if max_context_tokens is not None:
            _integer(max_context_tokens, 3, "max_context_tokens")
        _integer(max_window_tokens, 2, "max_window_tokens")
        _integer(steps, 1, "steps")
        probe = source_probe(context)
        if probe is None:
            raise ValueError("Context provides no independent continuation probe")
        question = f"Use the passage to answer the question.\nPassage:\n{context}\nQuestion: {query}"
        if self.generate is None:
            generation_room(self.backend, generation_prefix(self.backend, question), max_tokens)
        result = self.service.train_pairs([probe], evidence=[context], mode="query_adapt", steps=steps,
                                          training_options={"objective": "query_only", "context_text": context,
                                                            "max_context_tokens": max_context_tokens,
                                                            "max_window_tokens": max_window_tokens},
                                          learning_rate=learning_rate, cancel_event=cancel_event)
        return {**result, "answer": self._answer(question, max_tokens, 0, cancel_event),
                "generation_includes_context": True}

    def _sample_group(self, prompt, rollouts, max_tokens, cancel_event, *, temperature=.8, reference=None):
        answers = []
        for _attempt in range(rollouts * 2):
            try:
                item = (reference.sample(prompt, max_tokens, temperature, cancel_event) if reference is not None
                        else self._answer(prompt, max_tokens, temperature, cancel_event))
                if reference is not None:
                    item.update(sampling_rule=reference.sampling_rule, sampling_temperature=temperature)
                answers.append(item)
            except EmptyGeneration:
                continue
            if len(answers) == rollouts:
                break
        return answers

    def _update_policy(self, prompt, answers, rewards, *, mode, steps, learning_rate,
                       reward_weighting, entropic_kl_target, cancel_event, policy_parent=...,
                       samples=None, reference_kl_coef=0., max_window_tokens=None):
        prefix = generation_prefix(self.backend, prompt)
        rows = [{"prompt": prompt, "completion": answer, "rollout_prefix": prefix, "reward": reward}
                for answer, reward in zip(answers, rewards)]
        if samples is not None:
            for row, sample in zip(rows, samples):
                row.update(sample)
        if max_window_tokens is None:
            if samples is not None:
                max_window_tokens = max(len(r["rollout_prompt_ids"]) + len(r["rollout_completion_ids"]) for r in rows)
            else:
                tokenizer = self.backend.tokenizer
                encode = getattr(tokenizer, "encode", None)
                if callable(encode):
                    max_window_tokens = max(len(encode(prefix + answer)) for answer in answers)
                else:
                    from core.learning_data import encode_text
                    max_window_tokens = max(len(encode_text(tokenizer, prefix + answer)) for answer in answers)
            max_window_tokens = max(2, max_window_tokens)
        return self.service.train_pairs(rows, mode=mode, training_options={
            "objective": "reward_policy", "max_window_tokens": max_window_tokens, "reference_kl_coef": reference_kl_coef, "reward_weighting": reward_weighting,
            "entropic_kl_target": entropic_kl_target}, steps=steps, learning_rate=learning_rate,
            cancel_event=cancel_event, policy_parent=policy_parent)

    def _rollout_update(self, prompt, *, policy, mode, verifier, rollouts, rounds, max_tokens, steps,
                        learning_rate, extract_answer, cancel_event, reward_weighting="group_relative",
                        entropic_kl_target=math.log(2), trajectories=None, synthesize=False,
                        max_window_tokens=None, temperature=.8):
        _text(prompt, "rollout prompt")
        _integer(rollouts, 2, "rollouts")
        _integer(rounds, 1, "rounds")
        _integer(max_tokens, 1, "max_tokens")
        _integer(steps, 1, "steps")
        _number(learning_rate, "learning_rate")
        _number(temperature, "temperature", allow_zero=True)
        if max_window_tokens is not None:
            _integer(max_window_tokens, 2, "max_window_tokens")
        if type(synthesize) is not bool:
            raise ValueError("synthesize must be a boolean")
        supplied = candidate_trajectories(trajectories) if trajectories is not None else None
        if mode == "verifier":
            verifier = _verifier(verifier)
        if policy:
            self._require("reward_policy")
        sampler = policy_reference(self.backend, frozen=False) if policy and self.generate is None else None
        history = []
        best_sample, best_reward = None, None
        for iteration in range(rounds):
            check_cancel(cancel_event)
            policy_parent = (self.backend.learning_state.current_manifest(verify=False) or {}).get("generation") if policy else None
            traces = supplied if iteration == 0 else None
            if traces:
                answers = [trace["messages"][-1]["content"] for trace in traces]
            else:
                candidates = self._sample_group(prompt, rollouts, max_tokens, cancel_event,
                                                temperature=temperature, reference=sampler)
                answers = [c["completion"] for c in candidates] if sampler is not None else candidates
            if not traces and len(answers) != rollouts:
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
                update = self._update_policy(prompt, answers, rewards, mode="rollout_" + mode, steps=steps,
                                             learning_rate=learning_rate, reward_weighting=reward_weighting,
                                             entropic_kl_target=entropic_kl_target, cancel_event=cancel_event,
                                             policy_parent=policy_parent, max_window_tokens=max_window_tokens,
                                             samples=candidates if sampler is not None else None)
            else:
                traces = traces or candidate_trajectories(answers)
                rows = joint_rows(prompt, traces, rewards)
                update = self.service.train_pairs(rows, mode="rollout_" + mode,
                                                  training_options=_causal_options(max_window_tokens),
                                                  steps=steps, learning_rate=learning_rate, cancel_event=cancel_event)
            history.append({"updated": True, "rewards": rewards, "update": update})
        synthesized = None
        synthesis_details = {}
        if synthesize and any(h["updated"] for h in history):
            try:
                synthesized = self._answer(prompt, max_tokens, 0, cancel_event)
            except EmptyGeneration as exc:
                synthesis_details["synthesis_error"] = str(exc)
            if synthesized is not None and mode == "verifier":
                scores, _ = _rewarded([synthesized], mode, verifier, None)
                synthesis_details["answer_reward"] = scores[0]
        measurements = getattr(verifier, "results", None) if mode == "verifier" else None
        return {"status": "completed" if any(h["updated"] for h in history) else "skipped",
                **({"answer": synthesized, "synthesized": synthesized is not None, **synthesis_details} if not policy else {}),
                **({"verifier_measurements": measurements} if measurements is not None else {}),
                "reward_source": "model_consensus" if mode == "consensus" else "task_verifier",
                "rounds": history, "updates": sum(h["updated"] for h in history),
                "best_sample": best_sample, "best_reward": best_reward}

    def reinforce(self, prompt, *, mode="consensus", verifier=None, rollouts=4, rounds=1,
                  max_tokens=64, steps=1, learning_rate=1e-4, extract_answer=None, cancel_event=None,
                  reward_weighting="group_relative", entropic_kl_target=math.log(2), max_window_tokens=None, temperature=.8):
        return self._rollout_update(prompt, policy=True, mode=mode, verifier=verifier, rollouts=rollouts,
                                    rounds=rounds, max_tokens=max_tokens, steps=steps, learning_rate=learning_rate,
                                    extract_answer=extract_answer, cancel_event=cancel_event,
                                    reward_weighting=reward_weighting, entropic_kl_target=entropic_kl_target,
                                    max_window_tokens=max_window_tokens, temperature=temperature)

    def consolidate_candidates(self, prompt, *, mode="consensus", verifier=None, rollouts=4, rounds=1,
                               max_tokens=64, steps=1, learning_rate=1e-4, extract_answer=None, cancel_event=None,
                               trajectories=None, synthesize=True, temperature=.8, max_window_tokens=None):
        return self._rollout_update(prompt, policy=False, mode=mode, verifier=verifier, rollouts=rollouts,
                                    rounds=rounds, max_tokens=max_tokens, steps=steps, learning_rate=learning_rate,
                                    extract_answer=extract_answer, cancel_event=cancel_event, trajectories=trajectories,
                                    synthesize=synthesize, temperature=temperature, max_window_tokens=max_window_tokens)

    def discover(self, prompt, *, verifier, seed=None, rounds=None, rollouts=4, max_tokens=96, archive_size=16,
                 exploration=1.0, entropic_kl_target=math.log(2), learning_rate=1e-4, steps=1,
                 reference_kl_coef=None, temperature=1., max_window_tokens=None, schedule=None,
                 children_per_expansion=2, cancel_event=None, progress=None):
        """Expand PUCT parents using fresh sampled actions from each successive policy."""
        _text(prompt, "discovery task")
        if schedule is None:
            schedule = []
        if not isinstance(schedule, list) or not all(isinstance(r, dict) for r in schedule):
            raise ValueError("schedule must contain per-round option dictionaries")
        rounds = _integer(rounds if rounds is not None else len(schedule) or 3, 1, "rounds")
        if len(schedule) > rounds:
            raise ValueError("schedule has more entries than rounds")
        defaults = dict(rollouts=rollouts, max_tokens=max_tokens, steps=steps, learning_rate=learning_rate,
                        temperature=temperature, entropic_kl_target=entropic_kl_target, exploration=exploration)
        for entry in [{}, *schedule]:
            if entry.keys() - defaults.keys():
                raise ValueError("Unknown discovery schedule options: " + ", ".join(sorted(entry.keys() - defaults.keys())))
            config = {**defaults, **entry}
            for key, minimum in (("rollouts", 2), ("max_tokens", 1), ("steps", 1)):
                _integer(config[key], minimum, key)
            for key in ("temperature", "learning_rate", "exploration", "entropic_kl_target"):
                _number(config[key], key, allow_zero=key in {"temperature", "exploration"})
        _integer(archive_size, 2, "archive_size")
        _integer(children_per_expansion, 1, "children_per_expansion")
        if max_window_tokens is not None:
            _integer(max_window_tokens, 2, "max_window_tokens")
        if reference_kl_coef is not None:
            _number(reference_kl_coef, "reference_kl_coef", allow_zero=True)
        seeds = [] if seed is None else ([seed] if isinstance(seed, str) else seed)
        if not isinstance(seeds, list):
            raise ValueError("seed must be one solution or a list of distinct solutions")
        for item in seeds:
            _text(item, "seed solution")
        if len(set(seeds)) != len(seeds) or len(seeds) >= archive_size:
            raise ValueError("Use distinct seeds and reserve at least one archive slot for a child")
        reference_ms = None
        if isinstance(verifier, dict) and verifier.get("kind") == "python_tests":
            reference_ms = verifier.get("baseline_ms", 1.0)
            if reference_ms is None:
                raise ValueError("Discovery requires a positive runtime reference for Python tests")
            verifier = {**verifier, "baseline_ms": reference_ms}
        elif not callable(verifier):
            raise ValueError("Discovery requires a continuous task verifier or Python function tests")
        verify = _verifier(verifier)
        self._require("reward_policy")

        def score(answer):
            check_cancel(cancel_event)
            value = float(verify(answer))
            check_cancel(cancel_event)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Discovery verifier rewards must be finite and in [0, 1]")
            return value

        with self.service._update_lock:
            reference = None
            if self.generate is None:
                reference = policy_reference(self.backend, frozen=reference_kl_coef != 0)
            if reference_kl_coef and reference is None:
                raise ValueError("This generation route cannot capture exact sampled IDs and frozen policy scores")
            coefficient = (.1 if reference is not None else 0.) if reference_kl_coef is None else reference_kl_coef
            reference_generation = (self.backend.learning_state.current_manifest(verify=False) or {}).get("generation")
            initial = [(text, score(text)) for text in seeds] or [(None, 0.0)]
            archive = DiscoveryArchive(initial, archive_size, exploration, children_per_expansion)
            history = []
            store = self.backend.learning_state
            for iteration in range(rounds):
                check_cancel(cancel_event)
                config = {**defaults, **(schedule[iteration] if iteration < len(schedule) else {})}
                archive.exploration = config["exploration"]
                parent, selection = archive.select()
                conditioning = prompt if parent.text is None else (
                    f"{prompt}\n\nPrevious candidate:\n{parent.text}\n\nReturn a complete improved solution.")
                before = (store.current_manifest(verify=False) or {}).get("generation")
                group = self._sample_group(conditioning, config["rollouts"], config["max_tokens"], cancel_event,
                                           temperature=config["temperature"], reference=reference)
                if coefficient:
                    for sample in group:
                        sample["reference_generation"] = reference_generation
                answers = [sample["completion"] for sample in group] if reference is not None else group
                rewards = [score(answer) for answer in answers]
                record = {"parent": parent.id, "policy_generation": before, "schedule": config, "selection": selection, "rewards": rewards, "updated": False}
                if answers:
                    record["inserted"] = archive.expand(parent.id, list(zip(answers, rewards)))
                if len(answers) < config["rollouts"]:
                    record["reason"] = "too few nonempty rollouts"
                elif max(rewards) == min(rewards):
                    record["reason"] = "uniform rollout rewards"
                else:
                    try:
                        result = self._update_policy(conditioning, answers, rewards, mode="discover",
                            steps=config["steps"], learning_rate=config["learning_rate"], reward_weighting="entropic",
                            entropic_kl_target=config["entropic_kl_target"], cancel_event=cancel_event, policy_parent=before,
                            samples=group if reference is not None else None, reference_kl_coef=coefficient,
                            max_window_tokens=max_window_tokens)
                        record.update(updated=True, update=result)
                    except LearningValidationError as exc:
                        record["reason"] = str(exc)
                history.append(record)
                if progress:
                    progress({"phase": "discover", "round": len(history), "parent": parent.id,
                              "updated": record["updated"], "archive_size": len(archive.active)})
            ranked = archive.ranked()
            best = next((archive.states[i] for i in ranked if archive.states[i].text is not None), None)
            updates = sum(r["updated"] for r in history)
            return {"status": "completed" if best is not None and (best.reward > 0 or updates) else "skipped",
                    "updates": updates, "rounds": history, "expansions": archive.expansions,
                    "archive": [{**archive.states[i].report(), "text": archive.states[i].text} for i in ranked],
                    "lineage": [state.report() for state in archive.states.values()],
                    "best_state_id": best.id if best else None, "best_sample": best.text if best else None,
                    "best_reward": best.reward if best else None, "reward_source": "task_verifier",
                    "reward_weighting": "entropic", "entropic_kl_target": entropic_kl_target,
                    "reference_kl_coefficient": coefficient, "reference_generation": reference_generation,
                    "reference_policy": "frozen_at_start" if coefficient else "disabled",
                    "reference_logprob_distribution": "raw_causal_logits" if coefficient else None,
                    "sampling_rule": reference.sampling_rule if reference is not None else "backend_text_generator",
                    "exploration": exploration, "archive_capacity": archive_size,
                    **({"runtime_reference_ms": reference_ms, "verifier_measurements": verify.results}
                       if reference_ms is not None else {})}
