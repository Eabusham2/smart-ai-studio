"""Crash/Ctrl+C-safe item-level RSI resume support.

The Phase-1 checkpoint already survives Ctrl+C, but RSI historically restarted
from item zero. This layer checkpoints completed RSI items and reconstructs any
verified training traces lost by an interrupted restart.

A partially generated RSI item is never marked complete; it reruns from round 1.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


RSI_PROGRESS_PATH = Path("eval_results/rsi_progress.json")
STAGE_TELEMETRY_PATH = Path("eval_results/stage_telemetry.jsonl")
RAW_OUTPUT_LOG = Path("eval_results/raw_model_outputs.log")
_VERSION = 1


def _item_key(split: str, item_id: str) -> str:
    return f"{split}::{item_id}"


def _eligible(splits, cache) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    for split_name, items in splits.items():
        if "DialogueRecall" in split_name:
            continue
        for item in items:
            if cache.get(f"Phase 1: Baseline_{item['id']}") is False:
                out.append((split_name, item))
    return out[:64]


def _fingerprint(items: List[Tuple[str, Dict[str, Any]]]) -> str:
    material = [
        [str(split), str(item.get("id", "")), str(item.get("prompt", ""))]
        for split, item in items
    ]
    raw = json.dumps(material, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _atomic_save(data: Dict[str, Any]) -> None:
    RSI_PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RSI_PROGRESS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, RSI_PROGRESS_PATH)


def _load_progress(fingerprint: str) -> Dict[str, Any] | None:
    try:
        data = json.loads(RSI_PROGRESS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if int(data.get("version", 0)) != _VERSION:
        return None
    if str(data.get("fingerprint", "")) != fingerprint:
        return None
    if not isinstance(data.get("completed"), dict):
        return None
    return data


def _db_path(self) -> str:
    return str(getattr(getattr(self.engine, "kg", None), "db_path", "") or "")


def _existing_rsi_rows(self, session_id: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    path = _db_path(self)
    if not path:
        return {}
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM episodic_interactions WHERE session_id=? AND consolidated=0 ORDER BY id ASC",
                (session_id,),
            ).fetchall()
    except Exception:
        return {}

    found: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        record = dict(row)
        domain = str(record.get("domain", ""))
        if domain.startswith("RSI::"):
            found[(domain[len("RSI::"):], str(record.get("prompt", "")))] = record
    return found


def _interrupted_telemetry() -> Tuple[Dict[str, Dict[str, Any]], float]:
    """Recover terminal item events from prior interrupted RSI starts.

    The telemetry wrapper writes the *current* RSI start immediately before this
    function runs. Recovery is therefore enabled only when there are at least two
    starts since the last RSI end: an interrupted older start plus this one.
    """
    if not STAGE_TELEMETRY_PATH.exists():
        return {}, 0.0
    events: List[Dict[str, Any]] = []
    try:
        for line in STAGE_TELEMETRY_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if str(row.get("stage", "")) == "RSI":
                events.append(row)
    except Exception:
        return {}, 0.0

    last_end = -1
    for idx, row in enumerate(events):
        if row.get("event") == "end":
            last_end = idx
    window = events[last_end + 1:]
    starts = [i for i, row in enumerate(window) if row.get("event") == "start"]
    if len(starts) < 2:
        return {}, 0.0

    historical = window[:starts[-1]]
    completed: Dict[str, Dict[str, Any]] = {}
    elapsed = 0.0
    for row in historical:
        try:
            elapsed = max(elapsed, float(row.get("elapsed", 0.0) or 0.0))
        except Exception:
            pass
        if row.get("event") != "progress":
            continue
        split = str(row.get("split", "") or "")
        item_id = str(row.get("item", "") or "")
        if not split or not item_id:
            continue
        completed[_item_key(split, item_id)] = {
            "passed": bool(row.get("passed", False)),
            "round": int(row.get("round", 2) or 2),
            "split": split,
            "item_id": item_id,
        }
    return completed, elapsed


def _raw_verified_candidates() -> Dict[Tuple[str, str], str]:
    """Return the latest PASS candidate by (split, item_id) from the append-only raw log."""
    if not RAW_OUTPUT_LOG.exists():
        return {}
    try:
        text = RAW_OUTPUT_LOG.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}
    pattern = re.compile(
        r"RSI SELF-IMPROVEMENT \| (?P<split>[^|\n]+?) \| (?P<item>[^|\n]+?) \| round (?P<round>\d+)\n"
        r"Selection:.*?\nSELF-GENERATED CANDIDATE:\n(?P<candidate>.*?)\n"
        r"Hidden reward after selection: (?P<status>PASS|FAIL)\n",
        re.DOTALL,
    )
    result: Dict[Tuple[str, str], str] = {}
    for match in pattern.finditer(text):
        if match.group("status") == "PASS":
            result[(match.group("split").strip(), match.group("item").strip())] = match.group("candidate").rstrip()
    return result


def install(p4) -> None:
    if getattr(p4, "_rsi_resume_hardening_installed", False):
        return

    base_rsi = p4._run_rsi_self_improvement

    def resumable_rsi(self, splits, cache) -> int:
        eligible = _eligible(splits, cache)
        fingerprint = _fingerprint(eligible)
        by_key = {
            _item_key(split, str(item.get("id", ""))): (split, item)
            for split, item in eligible
        }
        progress = _load_progress(fingerprint)

        if progress is None:
            recovered, prior_elapsed = _interrupted_telemetry()
            progress = {
                "version": _VERSION,
                "fingerprint": fingerprint,
                "completed": {},
                "elapsed_seconds": prior_elapsed,
                "complete": False,
            }
            for key, status in recovered.items():
                if key in by_key:
                    progress["completed"][key] = status
        else:
            prior_elapsed = float(progress.get("elapsed_seconds", 0.0) or 0.0)

        completed: Dict[str, Dict[str, Any]] = dict(progress.get("completed", {}))
        existing_rows = _existing_rsi_rows(self, p4.RSI_SESSION_ID)
        raw_candidates = _raw_verified_candidates()

        # A successful recovered item may be skipped only if its exact training
        # trace still exists or can be reconstructed from a logged PASS candidate.
        for key in list(completed):
            if key not in by_key:
                completed.pop(key, None)
                continue
            status = completed[key]
            if not bool(status.get("passed", False)):
                continue
            split, item = by_key[key]
            prompt = str(item.get("prompt", ""))
            if (split, prompt) in existing_rows:
                continue
            candidate = raw_candidates.get((split, str(item.get("id", ""))))
            if not candidate:
                completed.pop(key, None)
                continue
            try:
                self.engine.kg.log_interaction(
                    p4.RSI_SESSION_ID,
                    prompt,
                    candidate,
                    1.0,
                    1.0,
                    domain=f"RSI::{split}",
                )
                existing_rows[(split, prompt)] = {"completion": candidate}
            except Exception:
                completed.pop(key, None)

        progress["completed"] = completed
        progress["elapsed_seconds"] = prior_elapsed
        _atomic_save(progress)

        verified_before = sum(1 for s in completed.values() if bool(s.get("passed", False)))
        done_before = len(completed)
        state = getattr(self, "_rsi_progress_telemetry", None)
        if isinstance(state, dict):
            state["done"] = done_before
            state["verified"] = verified_before
            state["completed"] = set(completed)
            state["started"] = time.perf_counter() - max(0.0, prior_elapsed)
            if completed:
                last = list(completed.values())[-1]
                state["last_split"] = last.get("split")
                state["last_item"] = last.get("item_id")
                state["last_passed"] = bool(last.get("passed", False))

        if done_before:
            print(
                f"[RSI] Resumed | Item {done_before}/{len(eligible)} "
                f"({100.0 * done_before / max(1, len(eligible)):.2f}%) | Verified: {verified_before}",
                flush=True,
            )

        filtered_cache = dict(cache)
        for key in completed:
            _, item = by_key[key]
            filtered_cache[f"Phase 1: Baseline_{item['id']}"] = "RSI_RESUMED_DONE"

        original_delete = p4._delete_unconsumed_session_rows
        original_log = p4._append_rsi_log
        original_task_prompt = p4._task_user_prompt

        def guarded_delete(owner, session_id):
            if session_id == p4.RSI_SESSION_ID and completed:
                return None
            return original_delete(owner, session_id)

        def tracking_task_prompt(split_name, item):
            state_now = getattr(self, "_rsi_progress_telemetry", None)
            if isinstance(state_now, dict):
                state_now["last_split"] = split_name
                state_now["last_item"] = str(item.get("id", ""))
            return original_task_prompt(split_name, item)

        run_started = time.perf_counter()

        def checkpointing_log(owner, split, item_id, round_idx, candidate, passed, selection):
            original_log(owner, split, item_id, round_idx, candidate, passed, selection)
            if not (bool(passed) or int(round_idx) >= 2):
                return
            key = _item_key(str(split), str(item_id))
            if key not in by_key or key in progress["completed"]:
                return
            progress["completed"][key] = {
                "passed": bool(passed),
                "round": int(round_idx),
                "split": str(split),
                "item_id": str(item_id),
            }
            state_now = getattr(self, "_rsi_progress_telemetry", None)
            if isinstance(state_now, dict):
                elapsed_total = max(
                    0.0,
                    time.perf_counter() - float(state_now.get("started", time.perf_counter())),
                )
            else:
                elapsed_total = prior_elapsed + max(0.0, time.perf_counter() - run_started)
            progress["elapsed_seconds"] = elapsed_total
            progress["complete"] = len(progress["completed"]) >= len(eligible)
            _atomic_save(progress)

        p4._delete_unconsumed_session_rows = guarded_delete
        p4._append_rsi_log = checkpointing_log
        p4._task_user_prompt = tracking_task_prompt
        try:
            base_rsi(self, splits, filtered_cache)
        finally:
            p4._delete_unconsumed_session_rows = original_delete
            p4._append_rsi_log = original_log
            p4._task_user_prompt = original_task_prompt

        final_progress = _load_progress(fingerprint) or progress
        final_completed = dict(final_progress.get("completed", {}))
        total_verified = sum(1 for s in final_completed.values() if bool(s.get("passed", False)))
        final_progress["complete"] = len(final_completed) >= len(eligible)
        _atomic_save(final_progress)
        self._phase1_rsi_seed_count = total_verified
        return total_verified

    p4._run_rsi_self_improvement = resumable_rsi
    p4._rsi_resume_hardening_installed = True
