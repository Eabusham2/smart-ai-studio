"""Autonomous learning & research orchestrator.

/live /learn semantics:
1. Gather genuinely retrieved external/source facts.
2. Ask the already-loaded model to synthesize only from those sources.
3. Require a model-produced claim plus a verbatim source-evidence span.
4. Train the same active MLX model's LoRA parameters on the verified synthesis.
5. Measure real parameter drift and persist the adapter.

No hard-coded `return True` self-test, fake search dossier, fake synapse counter,
or second model load is accepted as learning evidence.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from config.settings import Settings, get_settings
from core.pro_engine import ProReasoningEngine
from core.tools import AgentToolRegistry
from core.learning_sources import extract_learning_text
from memory.anchor_dataset import get_anchor_texts
from memory.db import EpisodicMemoryDB


class AutonomousLearner:
    def __init__(
        self,
        engine: Optional[ProReasoningEngine] = None,
        tools: Optional[AgentToolRegistry] = None,
        db: Optional[EpisodicMemoryDB] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.engine = engine or ProReasoningEngine(settings=self.settings)
        self.db = db or EpisodicMemoryDB(db_path=self.settings.database_path)
        self.tools = tools or AgentToolRegistry(db_path=self.settings.database_path)
        self._last_research: Optional[Dict[str, Any]] = None
        self._last_synthesis: str = ""

    @staticmethod
    def _source_blob(research: Optional[Dict[str, Any]]) -> str:
        if not research:
            return ""
        parts = []
        for key in ("source_text", "crawl_report", "search_report"):
            value = research.get(key)
            if value:
                parts.append(str(value))
        return "\n\n".join(parts).strip()

    @staticmethod
    def _usable_retrieved_source(ok: bool, text: Any) -> bool:
        if not ok or not str(text or "").strip():
            return False
        lowered = str(text).casefold()
        rejected_markers = (
            "no results found",
            "search service may be unavailable",
            "failed to fetch",
            "could not retrieve page content",
            # Legacy topic-mode crawler fabricated this sentence without fetching a page.
            "foundational concepts, api architectures, and execution rules for",
        )
        return not any(marker in lowered for marker in rejected_markers)

    def _require_live_mlx(self):
        backend = getattr(self.engine, "mlx_backend", None)
        if (
            backend is None
            or getattr(backend, "model", None) is None
            or getattr(backend, "tokenizer", None) is None
            or not getattr(backend, "is_mlx_available", False)
        ):
            raise RuntimeError(
                "/learn requires the real active MLX model to be loaded; refusing fake/offline learning."
            )
        return backend

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        candidates = [text]
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            candidates.append(m.group(0))
        for candidate in candidates:
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        return None

    def research_from_source(self, source_path: str, topic: str) -> Dict[str, Any]:
        """Extract deterministic text from an attached file/folder for the existing Learn pipeline."""
        extracted = extract_learning_text(source_path)
        source_text = str(extracted.get("text") or "").strip()
        if not source_text:
            raise RuntimeError(
                "/learn attachment contained no extractable text. "
                "Media files can still be routed to a media generator Learn target."
            )
        result = {
            "topic": str(topic or "").strip(),
            "source_path": str(extracted.get("path") or source_path),
            "source_text": source_text,
            "sources_found": len(extracted.get("text_files") or []),
            "media_kinds": list(extracted.get("media_kinds") or []),
            "extraction_errors": list(extracted.get("extraction_errors") or []),
        }
        self._last_research = result
        return result

    def crawl_and_research(self, topic: str) -> Dict[str, Any]:
        """Gather only actually retrieved source material through the existing tools."""
        target = str(topic or "").strip()
        is_url = target.startswith(("http://", "https://"))
        crawl_parts: List[str] = []
        search_report = ""
        sources_found = 0

        if is_url:
            ok_fetch, fetch_res = self.tools.execute_tool("web_fetch", {"url": target})
            if self._usable_retrieved_source(ok_fetch, fetch_res):
                crawl_parts.append(str(fetch_res))
                sources_found += 1

            ok_crawl, crawl_res = self.tools.execute_tool(
                "web_crawler",
                {"query_or_url": target, "max_pages": 3, "max_depth": 1},
            )
            if self._usable_retrieved_source(ok_crawl, crawl_res):
                crawl_parts.append(str(crawl_res))
                sources_found += 1
        else:
            # The legacy crawler fabricates a placeholder dossier for topic strings,
            # so topic learning uses only the real network-backed search path.
            ok_search, search_res = self.tools.execute_tool("web_search", {"query": target})
            if self._usable_retrieved_source(ok_search, search_res):
                search_report = str(search_res)
                sources_found += 1

        result = {
            "topic": target,
            "crawl_report": "\n\n".join(crawl_parts),
            "search_report": search_report,
            "sources_found": sources_found,
        }
        self._last_research = result
        return result

    def synthesize_knowledge(self, topic: str, research: Dict[str, Any]) -> str:
        """Use the real active model to synthesize source-grounded knowledge."""
        self._require_live_mlx()
        source = self._source_blob(research)
        if not source:
            raise RuntimeError("/learn research returned no usable source material; refusing to fabricate a lesson.")

        source = source[:16000]
        prompt = (
            f"Study the following source material about {topic}. Produce a concise technical learning note "
            "containing only claims supported by the supplied source. Do not invent facts, citations, or code. "
            "Preserve important names, values, constraints, and causal relationships.\n\n"
            f"SOURCE MATERIAL:\n{source}"
        )
        response, _meta = self.engine.solve(
            prompt,
            force_branch_count=1,
            temperature=0.20,
        )
        synthesis = str(response or "").strip()
        if not synthesis or "weights are not" in synthesis.lower():
            raise RuntimeError("/learn model produced no usable synthesis.")

        self._last_research = research
        self._last_synthesis = synthesis
        return f"### 💡 Synthesized Knowledge Base: **{topic.strip()}**\n\n{synthesis}"

    def self_test_and_verify(
        self,
        topic: str,
        research: Optional[Dict[str, Any]] = None,
        synthesis: Optional[str] = None,
    ) -> Tuple[bool, str, float]:
        """Verify a self-generated claim against a verbatim evidence span from the sources."""
        self._require_live_mlx()
        research = research or self._last_research
        synthesis = synthesis or self._last_synthesis
        source = self._source_blob(research)
        if not source or not synthesis:
            return False, "✗ Source-grounded verification unavailable.", 0.0

        source = source[:16000]
        verify_prompt = (
            f"You are verifying what you just learned about {topic}. From the SOURCE below, return ONLY one JSON object "
            "with keys `claim` and `evidence`. `claim` must be a concise factual statement from the learned note. "
            "`evidence` must be an exact verbatim substring copied from SOURCE that directly supports the claim.\n\n"
            f"LEARNED NOTE:\n{synthesis[:8000]}\n\nSOURCE:\n{source}"
        )
        answer, _meta = self.engine.solve(
            verify_prompt,
            force_branch_count=1,
            temperature=0.20,
        )
        obj = self._extract_json_object(str(answer or ""))
        if not obj:
            return False, "✗ Source verification failed: model did not return valid JSON evidence.", 0.0

        claim = str(obj.get("claim", "")).strip()
        evidence = str(obj.get("evidence", "")).strip()
        passed = bool(claim and len(evidence) >= 12 and evidence.casefold() in source.casefold())
        if not passed:
            return False, "✗ Source verification failed: evidence was not a verbatim source span.", 0.0

        return (
            True,
            f"✓ Source-grounded verification passed ({len(evidence)} evidence characters matched verbatim).",
            1.0,
        )

    def consolidate_parameters(
        self,
        topic: str,
        completion_text: str,
        reward: float = 1.0,
    ) -> Dict[str, Any]:
        """Update the same loaded MLX LoRA parameters and persist the measured change."""
        if reward < 1.0:
            return {"status": "skipped", "reason": "Learning trace was not source-verified.", "parameter_drift_l2": 0.0}

        backend = self._require_live_mlx()
        adapter_path = getattr(backend, "adapter_path", None) or getattr(self.engine, "lora_adapter_path", None)
        if not adapter_path:
            adapter_path = os.path.abspath("./consolidated_slow_lora/adapters.safetensors")
            backend.adapter_path = adapter_path
            self.engine.lora_adapter_path = adapter_path

        memory_id = self.db.log_interaction(
            prompt=f"Source-grounded autonomous learning task: {topic}",
            completion=completion_text,
            raw_branches=[completion_text],
            verified_reward=1.0,
            surprise_score=0.35,
            mode="Autonomous Learn: Source-Grounded",
            entropy=0.0,
            winning_branch=0,
            test_cases="verbatim source evidence required",
        )

        fisher = None
        try:
            fisher = backend.compute_mlx_fisher(get_anchor_texts()[:4])
        except Exception:
            fisher = None

        updated_adapters, param_drift = backend.train_mini_batch(
            adapters=getattr(backend, "adapters", {}) or {},
            data=[
                {
                    "prompt": f"What should be remembered about {topic}?",
                    "completion": completion_text,
                }
            ],
            fisher_matrix=fisher,
            lambda_ewc=float(getattr(self.settings, "ewc_lambda", 400.0)) if fisher else 0.0,
            learning_rate=float(getattr(self.settings, "consolidation_lr", 1e-4)),
            steps=3,
            save_path=adapter_path,
        )
        drift = float(param_drift)
        if drift <= 0.0:
            raise RuntimeError("/learn completed a training call but measured zero parameter change.")

        self.db.mark_consolidated([memory_id])
        touched = 0
        for value in (updated_adapters or {}).values():
            try:
                touched += int(value.size)
            except Exception:
                pass

        return {
            "status": "success",
            "memories_consolidated": 1,
            "parameter_drift_l2": drift,
            "trainable_parameters_touched": touched,
            "trainable_parameters_m": touched / 1_000_000.0,
            "ewc_active": bool(fisher),
            "adapter_saved_to": adapter_path,
        }

    def run_learning_session(
        self,
        topic: str,
        cancel_event: Optional[Any] = None,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
        max_cycles: int = 2,
        source_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run research → real synthesis → source verification → real parameter update."""
        total_params_m = 0.0
        total_drift = 0.0
        cycles_completed = 0
        clean_topic = topic.replace("/learn", "").strip() or "General Autonomous Reasoning"

        if progress_callback:
            progress_callback(
                "init",
                (
                    f"🎓 **Autonomous Learning Initiated**: studying attached source for **\"{clean_topic}\"**..."
                    if source_path
                    else f"🎓 **Autonomous Learning Initiated**: researching **\"{clean_topic}\"** with the active model..."
                ),
                0.0,
            )

        for cycle in range(1, max_cycles + 1):
            if cancel_event and cancel_event.is_set():
                if progress_callback:
                    progress_callback("stopped", "⏹ Learning session stopped by user.", 0.0)
                break

            if progress_callback:
                progress_callback(
                    "crawling",
                    (
                        f"📚 **[Cycle {cycle}/{max_cycles}] Reading attached learning source**..."
                        if source_path
                        else f"🕷️ **[Cycle {cycle}/{max_cycles}] Gathering independent sources**..."
                    ),
                    0.0,
                )
            research = (
                self.research_from_source(source_path, clean_topic)
                if source_path
                else self.crawl_and_research(clean_topic)
            )
            if not self._source_blob(research):
                raise RuntimeError("/learn found no source material; no parameter update was attempted.")

            if cancel_event and cancel_event.is_set():
                break
            if progress_callback:
                progress_callback("synthesizing", f"🧠 **[Cycle {cycle}/{max_cycles}] Synthesizing with the active model**...", 0.0)
            synthesis = self.synthesize_knowledge(clean_topic, research)

            if cancel_event and cancel_event.is_set():
                break
            if progress_callback:
                progress_callback("verifying", f"🧪 **[Cycle {cycle}/{max_cycles}] Checking a verbatim source-grounded claim**...", 0.0)
            passed, test_details, reward = self.self_test_and_verify(clean_topic, research, synthesis)
            if not passed:
                raise RuntimeError(test_details)

            if cancel_event and cancel_event.is_set():
                break
            result = self.consolidate_parameters(clean_topic, synthesis, reward)
            if result.get("status") != "success":
                raise RuntimeError(str(result.get("reason") or "parameter consolidation failed"))

            params_m = float(result.get("trainable_parameters_m", 0.0) or 0.0)
            drift = float(result.get("parameter_drift_l2", 0.0) or 0.0)
            total_params_m += params_m
            total_drift += drift
            cycles_completed += 1

            if progress_callback:
                progress_callback(
                    "consolidating",
                    f"📈 **[Cycle {cycle}/{max_cycles}] Real MLX LoRA update complete** "
                    f"(||ΔW||₂={drift:.6f}, touched={params_m:.3f}M trainable params, "
                    f"EWC={'on' if result.get('ewc_active') else 'off'})\n\n{synthesis}\n\n{test_details}",
                    params_m,
                )

        cancelled = bool(cancel_event and cancel_event.is_set())
        if progress_callback and not cancelled:
            progress_callback(
                "done",
                f"✅ **Learning Complete**: {cycles_completed} verified cycle(s), "
                f"cumulative ||ΔW||₂={total_drift:.6f}, touched {total_params_m:.3f}M trainable parameters for `{clean_topic}`.",
                0.0,
            )

        return {
            "status": "cancelled" if cancelled else "completed",
            "topic": clean_topic,
            "cycles_completed": cycles_completed,
            "synapses_learned_m": total_params_m,
            "parameter_drift_l2": total_drift,
        }
