"""Future-run Stage-2 integrity recovered from the older flagship curriculum.

Keeps the useful old behavior: prove the 5-session/10-fact historical dialogue
curriculum is semantically indexed and queryable. It also proves the current
bounded 30-item TensorGraphDSL MCTS teaching pass actually stored its teaching
edges. The old unbounded N=16/K>=350 RLVR curriculum is intentionally not restored.

This layer is resume-safe: a run already past Stage 2 may prove the same facts from
the persisted DB/telemetry and continue without rerunning Stage 2.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict

from memory.dialogue_history_ingest import (
    HISTORICAL_DIALOGUE_SESSIONS,
    recall_historical_fact,
)


STATE_PATH = Path("eval_results/stage2_learning_state.json")
_VERSION = 1
_EXPECTED_SESSIONS = len(HISTORICAL_DIALOGUE_SESSIONS)
_EXPECTED_HISTORICAL_FACTS = sum(len(s.get("key_facts", [])) for s in HISTORICAL_DIALOGUE_SESSIONS)


def _atomic_save(payload: Dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def _count_tokens(tokenizer, texts) -> int:
    total = 0
    for text in texts:
        try:
            total += len(tokenizer.encode(str(text)))
        except Exception:
            total += max(1, len(str(text)) // 4)
    return total


def _db_counts(db_path: str, dsl_exprs) -> Dict[str, int]:
    result = {"dialogue_sessions": 0, "semantic_facts": 0, "mcts_edges": 0}
    if not db_path or not os.path.exists(db_path):
        return result
    try:
        with sqlite3.connect(db_path) as conn:
            try:
                result["dialogue_sessions"] = int(conn.execute("SELECT COUNT(*) FROM dialogue_sessions").fetchone()[0])
                result["semantic_facts"] = int(conn.execute("SELECT COUNT(*) FROM semantic_memory_index").fetchone()[0])
            except sqlite3.OperationalError:
                pass
            exprs = [str(x) for x in dsl_exprs if str(x)]
            if exprs:
                q = ",".join("?" for _ in exprs)
                try:
                    result["mcts_edges"] = int(
                        conn.execute(
                            f"SELECT COUNT(*) FROM graph_edges WHERE predicate='evaluates_to' AND source_entity IN ({q})",
                            exprs,
                        ).fetchone()[0]
                    )
                except sqlite3.OperationalError:
                    pass
    except Exception:
        pass
    return result


def _semantic_recall_proof(db_path: str) -> bool:
    probes = [
        ("What IPC ring buffer architecture was selected in Session A?", "Zero-Copy"),
        ("What token TTL was selected in the security session?", "30-second"),
        ("What sandbox memory cap was selected?", "512MB"),
    ]
    for query, needle in probes:
        ok, answer, _ = recall_historical_fact(query, db_path=db_path)
        if not ok or needle.lower() not in str(answer).lower():
            return False
    return True


def install(p4, stage_module) -> None:
    if getattr(p4, "_stage2_future_hardening_installed", False):
        return

    base_rsi = p4._run_rsi_self_improvement

    def verify_stage2_before_rsi(self, splits, cache):
        db_path = str(getattr(getattr(self.engine, "kg", None), "db_path", "") or "")
        dsl_items = list(splits.get("TensorGraphDSL-300", []))[:30]
        dsl_exprs = [item.get("dsl_expr", "") for item in dsl_items]
        counts = _db_counts(db_path, dsl_exprs)

        if counts["dialogue_sessions"] < _EXPECTED_SESSIONS:
            raise RuntimeError(
                f"Stage 2 integrity: {counts['dialogue_sessions']}/{_EXPECTED_SESSIONS} historical sessions indexed"
            )
        if counts["semantic_facts"] < _EXPECTED_HISTORICAL_FACTS:
            raise RuntimeError(
                f"Stage 2 integrity: {counts['semantic_facts']}/{_EXPECTED_HISTORICAL_FACTS} semantic facts indexed"
            )
        if not _semantic_recall_proof(db_path):
            raise RuntimeError("Stage 2 integrity: historical semantic recall proof failed")

        expected_mcts = len(dsl_exprs)
        if expected_mcts and counts["mcts_edges"] < expected_mcts:
            raise RuntimeError(
                f"Stage 2 integrity: {counts['mcts_edges']}/{expected_mcts} TensorGraphDSL MCTS teaching edges stored"
            )

        texts = []
        for q, a in p4.LEARN_EXAMPLES:
            texts.extend((q, a))
        for session in HISTORICAL_DIALOGUE_SESSIONS:
            texts.extend(session.get("key_facts", []))
        texts.extend(dsl_exprs)
        work_tokens = _count_tokens(self.engine.tokenizer, texts)

        # This is a future-run integrity proof, not a request to replay Stage 2.
        payload = {
            "version": _VERSION,
            "verified": True,
            "learn_facts": len(p4.LEARN_EXAMPLES),
            "dialogue_sessions": counts["dialogue_sessions"],
            "semantic_facts": counts["semantic_facts"],
            "mcts_edges": counts["mcts_edges"],
            "mcts_target": expected_mcts,
            "work_tokens": work_tokens,
            "verified_at": time.time(),
        }
        _atomic_save(payload)
        stage_module._emit(
            "Phase 2 Learn",
            "teaching_verified",
            dialogue_sessions=counts["dialogue_sessions"],
            semantic_facts=counts["semantic_facts"],
            mcts_edges=counts["mcts_edges"],
            mcts_target=expected_mcts,
            work_tokens=work_tokens,
        )
        return base_rsi(self, splits, cache)

    p4._run_rsi_self_improvement = verify_stage2_before_rsi
    p4._stage2_future_hardening_installed = True
