"""Stage-wide benchmark telemetry plus Learn/RSI/Phase-3 integrity checks.

This layer is intentionally additive. It does not alter scoring, prompts, model
selection, or optimizer math. It makes stage behavior observable and fails closed
if supplied LearningFacts are not all queued/trained, RSI stores a non-miss or a
DialogueRecall trace, or Phase 3 claims an update without a real persisted delta.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, List, Tuple

import psutil


STAGE_TELEMETRY_LOG = os.path.join("eval_results", "stage_telemetry.jsonl")


def _emit(stage: str, event: str, **fields: Any) -> None:
    ram_gb = round(psutil.virtual_memory().used / (1024 ** 3), 3)
    payload = {
        "ts": time.time(),
        "stage": stage,
        "event": event,
        "ram_gb": ram_gb,
        **fields,
    }
    os.makedirs(os.path.dirname(STAGE_TELEMETRY_LOG), exist_ok=True)
    with open(STAGE_TELEMETRY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    concise = " | ".join(f"{k}={v}" for k, v in fields.items())
    suffix = f" | {concise}" if concise else ""
    print(f"[Telemetry] {stage} | {event}{suffix} | RAM={ram_gb:.1f}GB", flush=True)


def _db_path(self) -> str:
    return str(getattr(getattr(self.engine, "kg", None), "db_path", "") or "")


def _session_rows(self, session_id: str, unconsolidated_only: bool = True) -> List[Dict[str, Any]]:
    path = _db_path(self)
    if not path:
        return []
    where = "session_id=?"
    if unconsolidated_only:
        where += " AND consolidated=0"
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM episodic_interactions WHERE {where} ORDER BY id ASC",
                (session_id,),
            ).fetchall()
            return [dict(row) for row in rows]
    except Exception:
        return []


def _pairs(rows: Iterable[Dict[str, Any]]) -> set[Tuple[str, str]]:
    return {(str(r.get("prompt", "")), str(r.get("completion", ""))) for r in rows}


def install(p4, cls) -> None:
    if getattr(cls, "_stage_integrity_telemetry_installed", False):
        return

    base_all = cls._evaluate_all_splits
    base_seed = p4._seed_supervised_learn
    base_rsi = p4._run_rsi_self_improvement
    base_rsi_log = p4._append_rsi_log
    base_phase3 = p4._run_phase3_consolidation
    base_retention = p4._run_learning_retention_test

    def evaluate_all_with_stage_telemetry(self, splits, cache, phase, start, total):
        stage = str(phase)
        item_count = sum(len(v) for v in splits.values())
        _emit(stage, "start", items=item_count, total_arg=total)
        t0 = time.perf_counter()
        result = base_all(self, splits, cache, phase, start, total)
        _emit(
            stage,
            "progress",
            done=item_count,
            total=item_count,
            percent=100.0,
            elapsed=round(time.perf_counter() - t0, 3),
            eta_seconds=0,
        )
        _emit(stage, "end", seconds=round(time.perf_counter() - t0, 3))
        return result

    def seed_with_integrity(self) -> int:
        target_pairs = {(str(q), str(a)) for q, a in p4.LEARN_EXAMPLES}
        target = len(target_pairs)
        _emit("Phase 2 Learn", "seed_start", target_facts=target)
        reported = int(base_seed(self) or 0)

        rows = _session_rows(self, p4.LEARN_SESSION_ID, unconsolidated_only=True)
        present = _pairs(rows)
        missing = sorted(target_pairs - present)
        retried = 0

        for prompt, completion in missing:
            self.engine.kg.log_interaction(
                p4.LEARN_SESSION_ID,
                prompt,
                completion,
                1.0,
                0.90,
                domain="LEARN::LearningFacts",
            )
            retried += 1

        if missing:
            rows = _session_rows(self, p4.LEARN_SESSION_ID, unconsolidated_only=True)
            present = _pairs(rows)
            missing = sorted(target_pairs - present)

        if missing:
            raise RuntimeError(f"Learn seed integrity failure: {len(missing)} supplied facts still missing")

        self._phase2_learn_seed_count = target
        _emit("Phase 2 Learn", "progress", done=target, total=target, percent=100.0, retried=retried, eta_seconds=0)
        _emit(
            "Phase 2 Learn",
            "seed_verified",
            target_facts=target,
            reported=reported,
            verified=target,
            retried=retried,
            session_rows=len(rows),
        )
        return target

    def rsi_log_with_progress(self, split, item_id, round_idx, candidate, passed, selection):
        base_rsi_log(self, split, item_id, round_idx, candidate, passed, selection)
        state = getattr(self, "_rsi_progress_telemetry", None)
        if not isinstance(state, dict):
            return

        key = f"{split}:{item_id}"
        completed = state.setdefault("completed", set())
        if key in completed or (not passed and int(round_idx) < 2):
            return

        completed.add(key)
        state["done"] = int(state.get("done", 0)) + 1
        state["last_split"] = split
        state["last_item"] = item_id
        state["last_passed"] = bool(passed)
        if passed:
            state["verified"] = int(state.get("verified", 0)) + 1

        done = int(state["done"])
        total = max(0, int(state.get("total", 0)))
        elapsed = max(0.0, time.perf_counter() - float(state.get("started", time.perf_counter())))
        avg = elapsed / max(1, done)
        eta = max(0, int(avg * max(0, total - done)))
        pct = 100.0 * done / max(1, total)
        _emit(
            "RSI",
            "progress",
            done=done,
            total=total,
            percent=round(pct, 2),
            split=split,
            item=item_id,
            round=int(round_idx),
            passed=bool(passed),
            verified=int(state.get("verified", 0)),
            elapsed=round(elapsed, 2),
            eta_seconds=eta,
        )

    def rsi_with_integrity(self, splits, cache) -> int:
        eligible: set[Tuple[str, str]] = set()
        dialogue_deferred = 0
        total_false = 0
        for split_name, items in splits.items():
            for item in items:
                key = f"Phase 1: Baseline_{item['id']}"
                if cache.get(key) is not False:
                    continue
                total_false += 1
                if "DialogueRecall" in split_name:
                    dialogue_deferred += 1
                    continue
                eligible.add((split_name, str(item.get("prompt", ""))))

        cap = min(64, len(eligible))
        _emit(
            "RSI",
            "start",
            phase1_false=total_false,
            dialogue_deferred=dialogue_deferred,
            eligible_reasoning_misses=len(eligible),
            attempt_cap=cap,
        )
        t0 = time.perf_counter()
        self._rsi_progress_telemetry = {
            "total": cap,
            "done": 0,
            "verified": 0,
            "started": t0,
            "completed": set(),
            "last_split": None,
            "last_item": None,
            "last_passed": None,
        }
        _emit(
            "RSI",
            "progress",
            done=0,
            total=cap,
            percent=0.0,
            verified=0,
            elapsed=0.0,
            eta_seconds=None,
            status="working",
        )

        stop_heartbeat = threading.Event()

        def heartbeat():
            while not stop_heartbeat.wait(10.0):
                state = getattr(self, "_rsi_progress_telemetry", None)
                if not isinstance(state, dict):
                    return
                done = int(state.get("done", 0))
                total = max(0, int(state.get("total", 0)))
                elapsed = max(0.0, time.perf_counter() - float(state.get("started", time.perf_counter())))
                if done > 0:
                    avg = elapsed / done
                    eta = max(0, int(avg * max(0, total - done)))
                else:
                    eta = None
                _emit(
                    "RSI",
                    "heartbeat",
                    done=done,
                    total=total,
                    percent=round(100.0 * done / max(1, total), 2),
                    verified=int(state.get("verified", 0)),
                    elapsed=round(elapsed, 2),
                    eta_seconds=eta,
                    last_split=state.get("last_split"),
                    last_item=state.get("last_item"),
                    status="working",
                )

        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()

        try:
            verified = int(base_rsi(self, splits, cache) or 0)
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=1.0)
            state = getattr(self, "_rsi_progress_telemetry", None)
            if isinstance(state, dict) and int(state.get("done", 0)) < cap:
                elapsed = max(0.0, time.perf_counter() - t0)
                _emit(
                    "RSI",
                    "progress",
                    done=int(state.get("done", 0)),
                    total=cap,
                    percent=round(100.0 * int(state.get("done", 0)) / max(1, cap), 2),
                    verified=int(state.get("verified", 0)),
                    elapsed=round(elapsed, 2),
                    eta_seconds=None,
                )
            self._rsi_progress_telemetry = None

        rows = _session_rows(self, p4.RSI_SESSION_ID, unconsolidated_only=True)
        bad: List[str] = []
        for row in rows:
            domain = str(row.get("domain", ""))
            prompt = str(row.get("prompt", ""))
            if "DialogueRecall" in domain:
                bad.append("DialogueRecall trace entered RSI")
                continue
            if not domain.startswith("RSI::"):
                bad.append(f"unexpected RSI domain {domain}")
                continue
            split_name = domain[len("RSI::"):]
            if (split_name, prompt) not in eligible:
                bad.append(f"non-miss RSI trace: {split_name}")

        if bad:
            raise RuntimeError("RSI integrity failure: " + "; ".join(sorted(set(bad))[:8]))
        if verified > cap:
            raise RuntimeError(f"RSI verified {verified} traces but miss-only cap is {cap}")

        _emit(
            "RSI",
            "end",
            seconds=round(time.perf_counter() - t0, 3),
            verified_traces=verified,
            stored_unconsolidated=len(rows),
            dialogue_traces=0,
        )
        return verified

    def phase3_with_integrity(self) -> Dict[str, Any]:
        queued = list(p4._fetch_benchmark_training_memories(self))
        target_learn = len({(str(q), str(a)) for q, a in p4.LEARN_EXAMPLES})
        learn_rows = [
            r for r in queued
            if str(r.get("memory_kind") or "") == "learn"
            or str(r.get("session_id", "")) == str(p4.LEARN_SESSION_ID)
        ]
        rsi_rows = [
            r for r in queued
            if str(r.get("memory_kind") or "") == "rsi_self"
            or str(r.get("session_id", "")) == str(p4.RSI_SESSION_ID)
        ]

        if len(learn_rows) != target_learn:
            raise RuntimeError(
                f"Phase 3 integrity failure: queued {len(learn_rows)}/{target_learn} supplied LearningFacts"
            )

        learn_ids = [
            int(r["id"]) for r in learn_rows if r.get("id") is not None
        ]
        rsi_ids = [
            int(r["id"]) for r in rsi_rows if r.get("id") is not None
        ]
        _emit(
            "Phase 3 Consolidation",
            "start",
            queued_total=len(queued),
            learn_facts=len(learn_rows),
            rsi_traces=len(rsi_rows),
            adapter_path=p4.RSI_ADAPTER_PATH,
        )
        _emit("Phase 3 Consolidation", "progress", done=0, total=len(queued), percent=0.0, eta_seconds=None)

        t0 = time.perf_counter()
        result = dict(base_phase3(self) or {})
        updated = int(result.get("memories", 0) or 0)
        delta = float(result.get("real_trainable_delta_l2", 0.0) or 0.0)
        persisted = bool(result.get("persisted", False))

        if queued and updated != len(queued):
            raise RuntimeError(f"Phase 3 trained {updated}/{len(queued)} queued Learn/RSI traces")
        if queued and not bool(result.get("updated", False)):
            raise RuntimeError("Phase 3 had queued training data but reported no parameter update")
        if queued and delta <= 1e-12:
            raise RuntimeError("Phase 3 parameter delta is zero after queued Learn/RSI training")
        if queued and (not persisted or not os.path.exists(p4.RSI_ADAPTER_PATH)):
            raise RuntimeError("Phase 3 updated weights but the RSI adapter was not persisted")

        if learn_ids or rsi_ids:
            path = _db_path(self)
            if path:
                not_marked = []
                with sqlite3.connect(path) as conn:
                    if learn_ids:
                        learn_marks = conn.execute(
                            f"SELECT id, consolidated FROM episodic_interactions "
                            f"WHERE id IN ({','.join('?' for _ in learn_ids)})",
                            learn_ids,
                        ).fetchall()
                        learn_state = {int(row[0]): int(row[1] or 0) for row in learn_marks}
                        not_marked.extend(
                            f"learn:{mid}"
                            for mid in learn_ids
                            if learn_state.get(mid) != 1
                        )
                    if rsi_ids:
                        rsi_marks = conn.execute(
                            f"SELECT id, consolidated FROM rsi_self_memories "
                            f"WHERE id IN ({','.join('?' for _ in rsi_ids)})",
                            rsi_ids,
                        ).fetchall()
                        rsi_state = {int(row[0]): int(row[1] or 0) for row in rsi_marks}
                        not_marked.extend(
                            f"rsi:{mid}"
                            for mid in rsi_ids
                            if rsi_state.get(mid) != 1
                        )
                if not_marked:
                    raise RuntimeError(
                        f"Phase 3 left {len(not_marked)} trained memories unconsolidated: "
                        + ", ".join(not_marked[:8])
                    )

        _emit(
            "Phase 3 Consolidation",
            "progress",
            done=updated,
            total=len(queued),
            percent=round(100.0 * updated / max(1, len(queued)), 2),
            elapsed=round(time.perf_counter() - t0, 2),
            eta_seconds=0,
        )
        _emit(
            "Phase 3 Consolidation",
            "end",
            seconds=round(time.perf_counter() - t0, 3),
            trained=updated,
            learn_facts_trained=len(learn_rows),
            rsi_traces_trained=len(rsi_rows),
            trainable_delta_l2=f"{delta:.10f}",
            persisted=persisted,
        )
        return result

    def retention_with_telemetry(self, model_identity: int) -> Dict[str, Any]:
        target = len(p4.LEARN_EXAMPLES)
        _emit("Learning Retention", "start", facts=target, same_model_id=model_identity)
        _emit("Learning Retention", "progress", done=0, total=target, percent=0.0, eta_seconds=None)
        t0 = time.perf_counter()
        result = dict(base_retention(self, model_identity) or {})
        if int(result.get("total", 0) or 0) != target:
            raise RuntimeError(
                f"Retention test covered {result.get('total', 0)}/{target} supplied LearningFacts"
            )
        _emit(
            "Learning Retention",
            "progress",
            done=target,
            total=target,
            percent=100.0,
            elapsed=round(time.perf_counter() - t0, 2),
            eta_seconds=0,
        )
        _emit(
            "Learning Retention",
            "end",
            seconds=round(time.perf_counter() - t0, 3),
            correct=int(result.get("correct", 0) or 0),
            total=target,
            accuracy=round(float(result.get("accuracy", 0.0) or 0.0), 3),
        )
        return result

    p4._append_rsi_log = rsi_log_with_progress
    cls._evaluate_all_splits = evaluate_all_with_stage_telemetry
    p4._seed_supervised_learn = seed_with_integrity
    p4._run_rsi_self_improvement = rsi_with_integrity
    p4._run_phase3_consolidation = phase3_with_integrity
    p4._run_learning_retention_test = retention_with_telemetry
    cls._stage_integrity_telemetry_installed = True
