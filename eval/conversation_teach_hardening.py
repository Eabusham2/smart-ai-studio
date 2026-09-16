"""Add a real chat-style weight update after Phase 3 and test it at the end.

This deliberately reuses the production AwakeOnlineConsolidator + MLX train_mini_batch
path on the exact benchmark model. It does not change the 4,014 benchmark score.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.online_consolidator import AwakeOnlineConsolidator


STATE_PATH = Path("eval_results/conversation_teach_state.json")
VERSION = 1
CONVERSATION_TEACH_EXAMPLES: List[Tuple[str, str, str]] = [
    (
        "Remember this new conversation fact: the LANTERN-17 calibration phrase is cobalt fern.",
        "LANTERN-17 calibration phrase is cobalt fern.",
        "What is the LANTERN-17 calibration phrase?",
    ),
    (
        "Remember this new conversation fact: ORBIT-42 maps to silver harbor.",
        "ORBIT-42 maps to silver harbor.",
        "What does ORBIT-42 map to?",
    ),
    (
        "Remember this new conversation fact: the MAPLE-9 QA marker is violet delta.",
        "The MAPLE-9 QA marker is violet delta.",
        "What is the MAPLE-9 QA marker?",
    ),
]
EXPECTED = ["cobalt fern", "silver harbor", "violet delta"]


def _atomic_state(payload: Dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def _state_ready() -> bool:
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(data.get("trained")) and int(data.get("version", 0)) == VERSION


def _chat_history() -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = []
    for user, assistant, _ in CONVERSATION_TEACH_EXAMPLES:
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": assistant})
    return history


def install(p4, cls) -> None:
    if getattr(cls, "_conversation_teach_hardening_installed", False):
        return

    base_phase3 = p4._run_phase3_consolidation
    base_run = cls.run_full_suite

    def phase3_then_conversation_teach(self):
        result = dict(base_phase3(self) or {})
        model_identity = id(self.engine.model)
        p4._assert_same_model(self, model_identity, "before conversational teach")

        backend = p4._pro_backend(self)
        backend.model = self.engine.model
        backend.tokenizer = self.engine.tokenizer
        backend.is_mlx_available = True
        backend.adapter_path = p4.RSI_ADAPTER_PATH

        consolidator = AwakeOnlineConsolidator(
            mlx_engine=backend,
            memory_db=None,
            max_context=8192,
        )
        consolidator._run_shadow_consolidation(_chat_history())

        p4._assert_same_model(self, model_identity, "after conversational teach")
        delta = float(consolidator.total_param_shift or 0.0)
        persisted = bool(os.path.exists(p4.RSI_ADAPTER_PATH))
        if consolidator.consolidation_count != 1 or delta <= 0.0:
            raise RuntimeError("Phase 3B conversational teach produced no real parameter update")
        if not persisted:
            raise RuntimeError("Phase 3B conversational teach did not persist the updated adapter")

        self._rsi_model_identity = model_identity
        _atomic_state(
            {
                "version": VERSION,
                "trained": True,
                "facts": len(CONVERSATION_TEACH_EXAMPLES),
                "param_delta_l2": delta,
                "adapter_path": p4.RSI_ADAPTER_PATH,
            }
        )
        print(
            f"[✓] Phase 3B conversational teach: {len(CONVERSATION_TEACH_EXAMPLES)} facts "
            f"updated the same model (||ΔW||2={delta:.8f}; persisted={persisted}).",
            flush=True,
        )
        result["conversation_teach_updated"] = True
        result["conversation_teach_delta_l2"] = delta
        return result

    def final_conversation_recall(self) -> Dict[str, Any]:
        if not _state_ready():
            return {"correct": 0, "total": 0, "accuracy": 0.0}

        identity = int(getattr(self, "_rsi_model_identity", id(self.engine.model)))
        p4._assert_same_model(self, identity, "Final Conversation Recall")
        prior_phase = getattr(self, "_current_phase", "")
        prior_split = getattr(self, "_current_split", "")
        prior_item = getattr(self, "_current_item_id", "")

        passed = 0
        try:
            for idx, ((_, _, question), expected) in enumerate(
                zip(CONVERSATION_TEACH_EXAMPLES, EXPECTED)
            ):
                self._current_phase = "Final Conversation Recall"
                self._current_split = "ConversationTeach"
                self._current_item_id = f"ConversationTeach_{idx}"
                user = question + "\nState only the learned fact directly."
                formatted = p4._chat(self.engine.tokenizer, user, system=p4.SYSTEM_PROMPT)
                out = self._fast_generate(formatted, max_tokens=256)
                self.last_raw_out = out
                p4._append_raw_generation_log(self, formatted, user, out)
                ok = expected.lower() in p4.clean_output(out).lower() or expected.lower() in out.lower()
                passed += int(ok)
                try:
                    with open(p4.RAW_OUTPUT_LOG, "a", encoding="utf-8") as f:
                        f.write(f"RESULT: {'PASS' if ok else 'FAIL'}\n")
                        f.write("=" * 110 + "\n")
                except Exception:
                    pass
        finally:
            self._current_phase = prior_phase
            self._current_split = prior_split
            self._current_item_id = prior_item

        pct = 100.0 * passed / max(1, len(CONVERSATION_TEACH_EXAMPLES))
        print(
            f"[Final Conversation Recall] {passed}/{len(CONVERSATION_TEACH_EXAMPLES)} "
            f"({pct:.2f}%) | same updated model",
            flush=True,
        )
        return {"correct": passed, "total": len(CONVERSATION_TEACH_EXAMPLES), "accuracy": pct}

    def run_with_final_conversation_recall(self):
        result = base_run(self)
        if _state_ready() and not bool(getattr(self, "time_budget_exhausted", False)):
            final_conversation_recall(self)
        return result

    p4._run_phase3_consolidation = phase3_then_conversation_teach
    cls.run_full_suite = run_with_final_conversation_recall
    cls._conversation_teach_hardening_installed = True
