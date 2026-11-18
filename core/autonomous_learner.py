"""Autonomous learning & research orchestrator.

/live /learn semantics:
1. Gather genuinely retrieved external/source facts.
2. Ask the already-loaded model to synthesize only from those sources.
3. Require a model-produced claim plus a verbatim source-evidence span.
4. Train the same active backend's real adapter parameters on the verified synthesis.
5. Measure real parameter drift and persist the adapter.

No hard-coded `return True` self-test, fake search dossier, fake synapse counter,
or second model load is accepted as learning evidence.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from config.settings import Settings, get_settings
from core.pro_engine import ProReasoningEngine
from core.tools import AgentToolRegistry
from core.learning_sources import iter_attachment_positions, iter_learning_files, learning_media_kinds, source_snapshot
from core.test_time_learning import OnlineLearning
from core.online_consolidator import AwakeOnlineConsolidator
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
        self._verified_lesson = None

    @staticmethod
    def _source_blob(research: Optional[Dict[str, Any]]) -> str:
        if not research:
            return ""
        if research.get("selected_source_text"):
            return str(research["selected_source_text"])
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

    def _require_live_trainable_backend(self):
        """Return the real active backend only when it can perform a genuine update."""
        active = str(getattr(self.engine, "active_backend", "") or "").lower()
        candidates = []
        if active == "gguf":
            candidates.append(getattr(self.engine, "gguf_backend", None))
        elif active == "bitnet":
            candidates.append(getattr(self.engine, "bitnet_backend", None))
        elif active == "controller":
            candidates.append(getattr(self.engine, "controller_backend", None))
        elif active == "mlx":
            candidates.extend([
                getattr(self.engine, "mlx_engine", None),
                getattr(self.engine, "mlx_backend", None),
            ])
        else:
            candidates.extend([
                getattr(self.engine, "controller_backend", None),
                getattr(self.engine, "gguf_backend", None),
                getattr(self.engine, "bitnet_backend", None),
                getattr(self.engine, "mlx_engine", None),
                getattr(self.engine, "mlx_backend", None),
            ])

        seen = set()
        for backend in candidates:
            if backend is None or id(backend) in seen:
                continue
            seen.add(id(backend))
            if getattr(backend, "model", None) is None:
                continue
            if not callable(getattr(backend, "train_mini_batch", None)):
                continue
            capability = getattr(backend, "training_ready", None)
            if callable(capability):
                try:
                    if not bool(capability()):
                        continue
                except Exception:
                    continue
            return backend

        raise RuntimeError(
            "/learn requires the currently loaded backend to expose a verified real "
            "parameter-update path; refusing fake/offline learning."
        )

    # Compatibility alias for older callers/tests; no MLX-only semantics remain here.
    def _require_live_mlx(self):
        return self._require_live_trainable_backend()

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

    def research_from_source(self, source_path, topic, *, cancel_event=None, progress_callback=None):
        """Continue the complete attachment stream before bounded Learn/RSI synthesis."""
        _, file_count = source_snapshot(source_path, cancel_event)
        preview = []
        extraction_errors = []
        size = 0
        scanned = 0
        has_more = False
        for path in iter_learning_files(source_path):
            first_chunk = True
            for chunk, _ in iter_attachment_positions(path, 2048, extraction_errors):
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Learning source reading cancelled")
                scanned += len(chunk)
                if not preview:
                    chunk = chunk.lstrip()
                    if not chunk:
                        continue
                if first_chunk and file_count > 1:
                    chunk = f"\n\n===== SOURCE FILE: {path.name} =====\n" + chunk
                first_chunk = False
                room = 16000 - size
                piece = chunk[:room]
                preview.append(piece)
                size += len(piece)
                if len(chunk) > room:
                    has_more = True
                    break
            if has_more:
                break
        source_text = "".join(preview)
        if not source_text.strip():
            detail = " " + "\n".join(extraction_errors) if extraction_errors else ""
            raise RuntimeError("/learn attachment contained no usable text in its synthesis excerpt." + detail)
        result = {"topic": topic, "source_path": str(source_path), "source_text": source_text,
                  "sources_found": file_count, "media_kinds": learning_media_kinds(source_path), "extraction_errors": extraction_errors}
        if has_more or scanned > 4096:
            def progress(update):
                if progress_callback:
                    progress_callback("ingesting", f"Read {update['next']} source chunks and "
                                      f"trained {update['trained_chunks']}.", 0.0)

            job = OnlineLearning(self._consolidation_service(), attachment_errors=extraction_errors).run_request(
                {"method": "ingest", "source": str(source_path), "strategy": "stream",
                 "learning_rate": self.settings.consolidation_lr},
                cancel_event=cancel_event, progress=progress)
            result["stream"] = job
            if progress_callback:
                progress_callback("ingested", f"The adapter processed all {job['source_chars']} source characters. "
                                  f"Learn/RSI will synthesize from a {size}-character excerpt.", 0.0)
        if extraction_errors and progress_callback:
            progress_callback("source_warnings", "Source extraction warnings: " + "\n".join(extraction_errors), 0.0)
        self._last_research = result
        return result

    def _lesson_source(self, research, topic):
        source = self._source_blob(research)
        if research.get("source_path") or len(source) <= 4096:
            return source
        # Retrieved web reports are already strings from the research tools.
        # Visit all bounded passages and keep a bounded sample of validated
        # relevant quotes from across the report, including its suffix.
        import hashlib
        selected = []
        for index in range(0, len(source), 2048):
            passage = source[index:index + 2048]
            request = ("Return ONLY one JSON object with an evidence string. Copy the most relevant "
                       f"short verbatim quote about {topic} from this passage, or an empty string.\n{passage}")
            output, _ = self.engine.solve(request, force_branch_count=1, temperature=0.20)
            quote = (self._extract_json_object(str(output)) or {}).get("evidence")
            if (isinstance(quote, str) and 12 <= len(quote) <= 512 and quote in passage
                    and not any(text == quote for _, text in selected)):
                selected.append((hashlib.sha256(quote.encode()).digest(), quote))
                selected.sort()
                del selected[8:]
        if not selected:
            raise RuntimeError("The model selected no valid verbatim evidence across the retrieved report")
        research["selected_source_text"] = "\n".join(quote for _, quote in selected)
        research["source_selection"] = {"scanned_characters": len(source), "quotes": len(selected)}
        return research["selected_source_text"]

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
        self._require_live_trainable_backend()
        source = self._source_blob(research)
        if not source:
            raise RuntimeError("/learn research returned no usable source material; refusing to fabricate a lesson.")

        source = self._lesson_source(research, topic)
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
        self._require_live_trainable_backend()
        self._verified_lesson = None
        research = research or self._last_research
        synthesis = synthesis or self._last_synthesis
        source = self._source_blob(research)
        if not source or not synthesis:
            return False, "✗ Source-grounded verification unavailable.", 0.0

        source = self._lesson_source(research, topic)
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
        passed = bool(claim and len(evidence) >= 12 and evidence in source)
        if not passed:
            return False, "✗ Source verification failed: evidence was not a verbatim source span.", 0.0

        self._verified_lesson = (topic, synthesis, evidence)
        return (
            True,
            f"✓ Source-grounded verification passed ({len(evidence)} evidence characters matched verbatim).",
            1.0,
        )

    def _consolidation_service(self):
        backend = self._require_live_trainable_backend()
        service = getattr(self.engine, "awake_consolidator", None)
        if service is None:
            service = AwakeOnlineConsolidator(backend, lambda_ewc=float(self.settings.ewc_lambda))
            self.engine.awake_consolidator = service
        if service.engine is not backend:
            raise RuntimeError("The loaded model changed before learning could begin")
        return service

    def consolidate_parameters(self, topic, completion_text, reward=1.0, *, cancel_event=None):
        """Commit a verified lesson through the same transaction as chat and sleep."""
        if reward != 1.0:
            return {"status": "skipped", "reason": "Learning trace was not source-verified.",
                    "parameter_drift_l2": 0.0}
        prompt = f"What should be remembered about {topic}?"
        service = self._consolidation_service()
        witness = self._verified_lesson
        evidence = witness[2] if witness and witness[:2] == (topic, completion_text) else completion_text
        result = service.train_pairs([{"prompt": prompt, "completion": completion_text}],
                                     evidence=[evidence], mode="learn", cancel_event=cancel_event,
                                     learning_rate=self.settings.consolidation_lr)
        self._verified_lesson = None
        memory_id = self.db.log_interaction(
            prompt=f"Source-grounded autonomous learning task: {topic}",
            completion=completion_text, raw_branches=[completion_text],
            verified_reward=1.0, surprise_score=0.35, mode="Autonomous Learn: Source-Grounded",
            entropy=0.0, winning_branch=0, test_cases="verbatim source evidence required",
        )
        self.db.mark_consolidated([memory_id])
        return result

    def recursive_self_improve(
        self,
        topic: str,
        research: Dict[str, Any],
        learned_note: str,
    ) -> str:
        """Generate one answer-blind RSI revision using the already-updated active model."""
        self._require_live_trainable_backend()
        source = self._lesson_source(research, topic)
        if not source or not str(learned_note or "").strip():
            raise RuntimeError("RSI requires the completed Learn note and its source material.")

        prompt = (
            f"Recursively improve your own learned note about {topic}. "
            "Use only the supplied source and your previous note. Find weaknesses, omissions, "
            "or unclear reasoning yourself; do not ask for or assume a hidden answer. "
            "Return only the improved technical note. Do not include a score, pass/fail verdict, "
            "verification result, or commentary about this instruction.\n\n"
            f"PREVIOUS NOTE:\n{str(learned_note)[:8000]}\n\n"
            f"SOURCE MATERIAL:\n{source}"
        )
        improved, _meta = self.engine.solve(prompt)
        improved = str(improved or "").strip()
        if not improved:
            raise RuntimeError("RSI produced no improved self-generated trace.")
        return improved

    def consolidate_rsi_parameters(self, trace, *, cancel_event=None):
        """Train only the self-generated trace; no source question, answer key or reward."""
        service = self._consolidation_service()
        return service.train_pairs([{
            "prompt": "Internalize this self-generated reasoning pattern and improve future problem solving.",
            "completion": str(trace).strip(),
        }], mode="rsi", cancel_event=cancel_event, learning_rate=self.settings.consolidation_lr)

    def run_learning_session(
        self,
        topic: str,
        cancel_event: Optional[Any] = None,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
        max_cycles: int = 2,
        source_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run source Learn → real update → recursive self-improvement → second real update."""
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
                self.research_from_source(source_path, clean_topic, cancel_event=cancel_event, progress_callback=progress_callback)
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

            # Learn and RSI stay distinct. Learn first writes the independently
            # researched fact/note into the active adapter.
            try:
                learn_result = self.consolidate_parameters(clean_topic, synthesis, reward, cancel_event=cancel_event)
            except InterruptedError:
                if cancel_event is None or not cancel_event.is_set():
                    raise
                break
            if learn_result.get("status") != "success":
                raise RuntimeError(str(learn_result.get("reason") or "parameter consolidation failed"))

            learn_params_m = float(learn_result.get("trainable_parameters_m", 0.0) or 0.0)
            learn_drift = float(learn_result.get("parameter_drift_l2", 0.0) or 0.0)

            if cancel_event and cancel_event.is_set():
                break
            if progress_callback:
                progress_callback(
                    "rsi",
                    f"🔁 **[Cycle {cycle}/{max_cycles}] Recursive self-improvement on the updated model**...",
                    learn_params_m,
                )

            # RSI now runs on the same already-updated model. The verifier is invoked
            # only after the revision is finished; its verdict is never fed into the
            # model or stored beside the RSI trace.
            rsi_trace = self.recursive_self_improve(clean_topic, research, synthesis)
            rsi_passed, rsi_details, _rsi_reward = self.self_test_and_verify(
                clean_topic,
                research,
                rsi_trace,
            )
            if not rsi_passed:
                raise RuntimeError(rsi_details)

            if cancel_event and cancel_event.is_set():
                break
            try:
                rsi_result = self.consolidate_rsi_parameters(rsi_trace, cancel_event=cancel_event)
            except InterruptedError:
                if cancel_event is None or not cancel_event.is_set():
                    raise
                break
            if rsi_result.get("status") != "success":
                raise RuntimeError(str(rsi_result.get("reason") or "RSI parameter consolidation failed"))

            rsi_params_m = float(rsi_result.get("trainable_parameters_m", 0.0) or 0.0)
            rsi_drift = float(rsi_result.get("parameter_drift_l2", 0.0) or 0.0)
            params_m = learn_params_m + rsi_params_m
            drift = learn_drift + rsi_drift
            total_params_m += params_m
            total_drift += drift
            cycles_completed += 1

            if progress_callback:
                progress_callback(
                    "consolidating",
                    f"📈 **[Cycle {cycle}/{max_cycles}] Learn + RSI parameter updates complete** "
                    f"(Learn ||ΔW||₂={learn_drift:.6f}; RSI ||ΔW||₂={rsi_drift:.6f}; "
                    f"touched={params_m:.3f}M trainable params; "
                    f"RAM after cleanup={float(rsi_result.get('ram_after_cleanup_mb', 0.0) or 0.0):.0f} MB; "
                    f"backend={rsi_result.get('backend') or learn_result.get('backend') or 'unknown'})\n\n"
                    f"**Learn note**\n{synthesis}\n\n"
                    f"**RSI revision**\n{rsi_trace}\n\n{rsi_details}",
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
